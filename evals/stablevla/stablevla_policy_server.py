# Vendored from vla-bottleneck/eval/libero_plus/stablevla_policy_server.py so that
# ONF + StableVLA + LIBERO-Plus is self-contained (no external vla-bottleneck tree needed).
# Imports resolve off $STABLEVLA_ROOT on PYTHONPATH (set by scripts/run_sr.py).
"""
stablevla_policy_server.py

Policy server that exposes a *StableVLA* (VLA-Adapter / OpenVLA-for-action) checkpoint
behind the VLA-JEPA websocket policy contract so that the vendored LIBERO-Plus sim client
(``evals/libero_plus/harness/eval_libero.py``) can drive it remotely.

Architecture: ONE server process holds the (single) GPU-resident StableVLA model and
serves it over a websocket to a pool of CPU/sim client processes (one per shard).
The websocket layer (`WebsocketPolicyServer`) calls
    policy.predict_action(**payload)
and wraps whatever we return as {"status": "ok", "data": <return>}.

--------------------------------------------------------------------------------
NORMALIZATION SCHEME  (scheme "b": server returns ALREADY-UNNORMALIZED actions)
--------------------------------------------------------------------------------
StableVLA unnormalizes actions internally using NormalizationType.BOUNDS_Q99
(i.e. q01/q99 from dataset_statistics.json) inside
`OpenVLAForActionPrediction.predict_action` -> `_unnormalize_actions`
(see stablevla/prismatic/extern/hf/modeling_prismatic.py ~line 1074).

The VLA-JEPA reference client (`model2libero_interface.py`) instead re-implements
unnormalization with min/max (NormalizationType.BOUNDS) on whatever it receives under
"normalized_actions". Those two schemes are NOT equivalent (q01/q99 vs min/max + different
gripper handling), so letting the client unnormalize would silently produce wrong actions.

Therefore we choose scheme (b): the SERVER runs StableVLA's authoritative in-process
action pipeline end-to-end (`openvla_utils.get_vla_action` -> `model.predict_action`),
which yields fully-unnormalized actions, and returns them as-is. The client
(`model2libero_interface.py`) reads these under the key "actions" and passes them
straight through WITHOUT re-normalizing. This guarantees the remote pipeline is
bit-for-bit identical to native StableVLA LIBERO eval
(`experiments/robot/libero/run_libero_eval.py`).

Returned payload:
    {"actions": np.ndarray of shape [B, NUM_ACTIONS_CHUNK(=8), ACTION_DIM(=7)]}
(B == number of samples in `batch_images`; the StableVLA model itself is batch-1, so we
loop over the batch.) Values are unnormalized; the gripper dimension is left as produced
by StableVLA and is binarized sim-side by eval_libero.py.

CLI mirrors deployment/model_server/server_policy.py:
    --ckpt_path --port --cuda
"""

import argparse
import inspect
import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np


# --------------------------------------------------------------------------------------
# Path setup: make both source trees importable.
#   - StableVLA lives under <repo>/stablevla (provides `experiments`, `prismatic`)
#   - `deployment` is vendored under evals/libero_plus/harness/, supplied via the policy
#     descriptor's pythonpath (evals/policies/stablevla.yaml)
# This file is <repo>/evals/stablevla/stablevla_policy_server.py, so <repo> is two up.
# --------------------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STABLEVLA_ROOT = str(_REPO_ROOT / "stablevla")
if _STABLEVLA_ROOT not in sys.path:
    sys.path.insert(0, _STABLEVLA_ROOT)


def _select_cuda_device(cuda: Union[int, str]) -> None:
    """Pin the process to a single CUDA device BEFORE torch / model code is imported.

    StableVLA's `experiments.robot.openvla_utils.DEVICE` is hardcoded to ``cuda:0``, so we
    expose exactly one device as index 0. We respect an externally-set
    ``CUDA_VISIBLE_DEVICES`` (the launcher sets ``CUDA_VISIBLE_DEVICES=$GPU`` and
    passes ``--cuda 0``); only set it ourselves when it is not already constrained.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    # Reduce tokenizer fork warnings, mirroring eval_libero.py.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class StableVLAConfig:
    """Minimal stand-in for run_libero_eval.py's `GenerateConfig`.

    Only the fields read by `get_vla` / `get_action_head` / `get_proprio_projector` /
    `get_processor` / `get_vla_action` are included; defaults mirror the LIBERO defaults
    in stablevla/experiments/robot/libero/run_libero_eval.py.
    """

    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    # VLANeXt-style: proprio token fused into the VLM sequence.
    fuse_proprio_in_vlm: bool = False

    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8  # full chunk; should match NUM_ACTIONS_CHUNK
    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # VLANeXt flow-matching generative head (set use_l1_regression False to use it). Its sampling
    # is encapsulated in predict_action, so the rest of this server is identical to the L1 path.
    use_flow_matching: bool = False
    flow_hidden_size: int = 384
    flow_depth: int = 12
    flow_heads: int = 6
    flow_num_train_timesteps: int = 1000
    flow_num_inference_steps: int = 10

    use_minivlm: bool = True
    use_film: bool = False

    # Faithful VLA-JEPA full vision fine-tune: load the fine-tuned vision_backbone--*.pt over the base.
    load_vision_backbone: bool = False

    # Utils mirrored from GenerateConfig (consumed by get_action_head / model.set_version)
    save_version: str = "vla-adapter"
    use_pro_version: bool = True
    phase: str = "Inference"


_AUTO_MAP = {
    "AutoConfig": "configuration_prismatic.OpenVLAConfig",
    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
}


def _serialize_auto_map_update() -> None:
    """Make the trunk's config.json rewrite idempotent and one-at-a-time.

    openvla_utils.update_auto_map backs up and rewrites <checkpoint>/config.json on every load,
    so sixteen servers starting together read half-written files (JSONDecodeError) and leave a
    backup each. Here a server that finds the auto_map already set does nothing, and one that
    must write holds a file lock while the trunk's own function does it.
    """
    import fcntl
    import json
    from experiments.robot import openvla_utils

    original = openvla_utils.update_auto_map

    def already_set(config_path: Path) -> bool:
        try:
            return json.loads(config_path.read_text()).get("auto_map") == _AUTO_MAP
        except (OSError, ValueError):
            return False

    def update_auto_map(pretrained_checkpoint: str) -> None:
        checkpoint = Path(pretrained_checkpoint)
        config_path = checkpoint / "config.json"
        if not checkpoint.is_dir() or not config_path.is_file() or already_set(config_path):
            return
        with open(checkpoint / ".config.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not already_set(config_path):
                original(pretrained_checkpoint)

    openvla_utils.update_auto_map = update_auto_map


class StableVLAPolicy:
    """Holds the GPU-resident StableVLA model and exposes ``predict_action`` for the server."""

    def __init__(self, cfg: StableVLAConfig) -> None:
        # Imports are deferred until after CUDA_VISIBLE_DEVICES is set.
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_proprio_projector,
            get_processor,
            get_vla,
            get_vla_action,
        )
        _serialize_auto_map_update()

        self.cfg = cfg
        self._get_vla_action = get_vla_action

        # --- Build model + components, mirroring run_libero_eval.initialize_model ---
        logging.info("Loading StableVLA from checkpoint: %s", cfg.pretrained_checkpoint)
        self.vla = get_vla(cfg)
        self.vla.set_version(cfg.save_version)

        # Faithful VLA-JEPA (full vision fine-tune): the merged model.safetensors is rebuilt from the
        # ORIGINAL base + captured RAW_STATE_DICT, so it carries the ORIGINAL (frozen) encoder. Load the
        # fine-tuned vision backbone saved during training OVER it, so eval uses the reshaped encoder.
        if getattr(cfg, "load_vision_backbone", False):
            import json as _json
            import torch as _torch
            from pathlib import Path as _Path
            from experiments.robot.openvla_utils import find_checkpoint_file
            vb_ckpt = find_checkpoint_file(cfg.pretrained_checkpoint, "vision_backbone")
            sd = _torch.load(vb_ckpt, weights_only=True)
            # The saved state_dict is LoRA-wrapped (LoRA targets "all-linear", incl. vision linears):
            # `P.base_layer.weight` + `P.lora_A.default.weight` + `P.lora_B.default.weight`. The eval
            # model (merged model.safetensors) expects PLAIN keys `P.weight`, so merge LoRA -> plain.
            if any(k.endswith(".base_layer.weight") for k in sd):
                scaling = 2.0
                try:
                    acfg = _json.loads((_Path(cfg.pretrained_checkpoint) / "lora_adapter" / "adapter_config.json").read_text())
                    scaling = float(acfg["lora_alpha"]) / float(acfg["r"])
                except Exception:
                    pass
                merged, prefixes = {}, set()
                for k in sd:
                    if k.endswith(".base_layer.weight"):
                        prefixes.add(k[: -len(".base_layer.weight")])
                for k, v in sd.items():
                    if ".lora_A." in k or ".lora_B." in k or ".base_layer." in k:
                        continue
                    merged[k] = v  # plain (non-linear) params pass through
                for p in prefixes:
                    W = sd[p + ".base_layer.weight"].float()      # Linear [out,in] or Conv2d [out,in,kh,kw]
                    A = sd[p + ".lora_A.default.weight"].float()  # [r,in] or [r,in,kh,kw]
                    B = sd[p + ".lora_B.default.weight"].float()  # [out,r] or [out,r,1,1]
                    # General reshape handles Linear AND Conv2d (LoRA-B is 1x1 for conv patch-embed).
                    delta = (B.reshape(W.shape[0], -1) @ A.reshape(A.shape[0], -1)).reshape(W.shape)
                    merged[p + ".weight"] = (W + scaling * delta).to(sd[p + ".base_layer.weight"].dtype)
                    if (p + ".base_layer.bias") in sd:
                        merged[p + ".bias"] = sd[p + ".base_layer.bias"]
                sd = merged
                logging.info("Merged LoRA into fine-tuned vision backbone (scaling=%.3f, %d linears)", scaling, len(prefixes))
            missing, unexpected = self.vla.vision_backbone.load_state_dict(sd, strict=False)
            self.vla.vision_backbone = self.vla.vision_backbone.to(_torch.bfloat16)
            logging.info(
                "Loaded FINE-TUNED vision backbone from %s (missing=%d unexpected=%d)",
                vb_ckpt, len(missing), len(unexpected),
            )
            if len(missing) > 10:
                logging.warning("Vision backbone load: %d missing keys -> fine-tuned encoder may be INCOMPLETE! sample=%s",
                                len(missing), missing[:5])

        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                cfg,
                self.vla.llm_dim,
                proprio_dim=8,  # 8-dimensional proprio for LIBERO
            )

        self.action_head = None
        if cfg.use_l1_regression or cfg.use_flow_matching:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        self.noisy_action_projector = None  # diffusion path unused for L1 regression

        self.processor = None
        if cfg.model_family == "openvla":
            self.processor = get_processor(cfg)

        # Resolve a default unnorm_key now so failures surface at load time, not first infer.
        self._default_unnorm_key = self._resolve_unnorm_key(cfg.unnorm_key or None)
        logging.info(
            "StableVLA ready. Available norm_stats keys: %s; default unnorm_key=%s",
            list(getattr(self.vla, "norm_stats", {}).keys()),
            self._default_unnorm_key,
        )

    def _resolve_unnorm_key(self, unnorm_key: Optional[str]) -> str:
        """Validate / auto-resolve an unnorm_key against the model's norm_stats."""
        norm_stats = getattr(self.vla, "norm_stats", {})
        if unnorm_key is None:
            if len(norm_stats) == 1:
                return next(iter(norm_stats.keys()))
            raise ValueError(
                "Model was trained on more than one dataset; an `unnorm_key` must be "
                f"provided. Options: {list(norm_stats.keys())}"
            )
        # Mirror run_libero_eval.check_unnorm_key fallback for "_no_noops" datasets.
        if unnorm_key not in norm_stats and f"{unnorm_key}_no_noops" in norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in norm_stats, (
            f"unnorm_key '{unnorm_key}' not found in norm_stats; options: {list(norm_stats.keys())}"
        )
        return unnorm_key

    def predict_action(
        self,
        batch_images: List[List[Any]],
        instructions: List[str],
        unnorm_key: Optional[str] = None,
        state: Optional[List[np.ndarray]] = None,
        episode_step: Optional[List[int]] = None,
        worker_id: Optional[List[int]] = None,
        **kw: Any,
    ) -> dict:
        """Run StableVLA's action pipeline for a batch of observations.

        Args mirror the VLA-JEPA client payload (model2libero_interface.py):
            batch_images: list (len B) of per-sample image lists [primary_PIL, wrist_PIL].
                          NOTE: images arrive ALREADY rotated 180deg by the sim client
                          (eval_libero.py rotates agentview/wrist), matching StableVLA's
                          training preprocessing, so we do NOT rotate again here.
            instructions: list (len B) of task strings.
            unnorm_key:   dataset-statistics key for (un)normalization; falls back to the
                          server default resolved at load time.
            state:        optional list (len B) of np arrays shape (8,) proprio.

        Returns:
            {"actions": np.ndarray [B, NUM_ACTIONS_CHUNK, ACTION_DIM]} of UNNORMALIZED
            actions (scheme b). See module docstring.
        """
        resolved_key = self._resolve_unnorm_key(unnorm_key or self.cfg.unnorm_key or None)
        self.cfg.unnorm_key = resolved_key

        batch_actions = []
        for i, images in enumerate(batch_images):
            # `images` is [primary, wrist] (PIL.Image, already 224x224 from the client).
            primary = np.asarray(images[0])

            obs = {"full_image": primary}

            # Wrist image(s): get_vla_action collects keys containing "wrist" when
            # cfg.num_images_in_input > 1 (here == 2 for LIBERO two-view).
            if self.cfg.num_images_in_input > 1 and len(images) > 1:
                obs["wrist_image"] = np.asarray(images[1])

            # Proprio state (8-dim) if provided / required.
            if self.cfg.use_proprio:
                assert state is not None, (
                    "cfg.use_proprio=True but no `state` was sent by the client."
                )
                # The VLA-JEPA client wraps proprio as (1, 8); flatten to (8,) so the obs
                # matches native run_libero_eval.prepare_observation (which builds a 1-D
                # state via np.concatenate). normalize_proprio / the proprio projector then
                # see exactly the native shape.
                obs["state"] = np.asarray(state[i]).reshape(-1)

            task_label = instructions[i]

            # get_vla_action returns a list of up to cfg.num_open_loop_steps unnormalized
            # np arrays, each shape (ACTION_DIM,). With num_open_loop_steps == chunk it is
            # the full chunk. (Unnormalization done internally via BOUNDS_Q99.)
            # These six are hooks on the vla-bottleneck-side StableVLA trunk this server was
            # vendored from; the PINNED public trunk (HumanNet 68eb86d, docs/SETUP.md step 1)
            # has none of them and raises TypeError on the first one. They are hardcoded None
            # either way, so pass only what the resolved callee actually accepts.
            optional_hooks = {
                "jepa_conditioner": None, "goal_servo": None, "manifold_denoiser": None,
                "vib": None, "jepa_loop": None, "aff_head": None,
            }
            accepted = inspect.signature(self._get_vla_action).parameters
            optional_hooks = {k: v for k, v in optional_hooks.items() if k in accepted}

            actions_list = self._get_vla_action(
                cfg=self.cfg,
                vla=self.vla,
                processor=self.processor,
                obs=obs,
                task_label=task_label,
                action_head=self.action_head,
                **optional_hooks,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
                use_film=self.cfg.use_film,
                use_minivlm=self.cfg.use_minivlm,
            )
            sample_actions = np.asarray(actions_list, dtype=np.float32)  # [chunk, ACTION_DIM]
            batch_actions.append(sample_actions)

        # [B, NUM_ACTIONS_CHUNK, ACTION_DIM]; np.ndarray so msgpack_numpy can serialize it.
        actions = np.stack(batch_actions, axis=0).astype(np.float32)

        return {"actions": actions}


def main(args: argparse.Namespace) -> None:
    # Build config from CLI; everything else uses the LIBERO defaults baked into
    # StableVLAConfig (which mirror run_libero_eval.GenerateConfig).
    cfg = StableVLAConfig(pretrained_checkpoint=args.ckpt_path)
    cfg.load_vision_backbone = args.load_vision_backbone
    if args.fuse_proprio_in_vlm:
        cfg.fuse_proprio_in_vlm = True
    if args.use_flow_matching:
        # Serve a flow-matching checkpoint: swap the deterministic L1 head for the generative one.
        cfg.use_l1_regression = False
        cfg.use_flow_matching = True
        cfg.flow_num_inference_steps = args.flow_num_inference_steps

    policy = StableVLAPolicy(cfg)

    # Import the websocket server only after model load so import errors there don't mask
    # model-load diagnostics.
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "127.0.0.1"
    logging.info("Creating server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={"env": "libero_plus", "model": "stablevla"},
        workers=args.workers,
    )
    logging.info("StableVLA policy server running ...")
    server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    # Mirrors deployment/model_server/server_policy.py.
    parser = argparse.ArgumentParser(description="StableVLA websocket policy server.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="StableVLA checkpoint directory (e.g. outputs/object), containing "
        "config.json, *action_head*checkpoint.pt, *proprio_projector*checkpoint.pt, and "
        "dataset_statistics.json.",
    )
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--workers", type=int, default=2,
                        help="request threads; 2 overlaps one request's CPU preprocessing with "
                             "another's GPU forward (1 = one request at a time)")
    parser.add_argument(
        "--cuda",
        default=0,
        help="CUDA device index. Sets CUDA_VISIBLE_DEVICES if not already set; the model "
        "then runs on cuda:0 (StableVLA's hardcoded DEVICE).",
    )
    parser.add_argument(
        "--use_flow_matching",
        action="store_true",
        help="Serve a VLANeXt flow-matching generative head checkpoint (instead of the L1 head).",
    )
    parser.add_argument(
        "--flow_num_inference_steps",
        type=int,
        default=10,
        help="Euler ODE steps for the flow-matching head at sampling time.",
    )
    parser.add_argument(
        "--fuse_proprio_in_vlm",
        action="store_true",
        help="Serve a checkpoint trained with proprio fused into the VLM sequence.",
    )
    parser.add_argument(
        "--load_vision_backbone",
        action="store_true",
        help="Load a fine-tuned vision_backbone--*.pt over the base (faithful VLA-JEPA full vision FT).",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    # IMPORTANT: pin the CUDA device before any torch/model import happens.
    _select_cuda_device(args.cuda)
    main(args)
