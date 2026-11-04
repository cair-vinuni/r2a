#!/usr/bin/env python
"""pi0.5 websocket policy server: an openpi policy behind the harness's websocket contract.

Runs under external/openpi/.venv. Loads the policy in-process the way openpi's own
scripts/serve_policy.py does and answers the sim client's request (batch_images, instructions,
state) with unnormalized action chunks. --config pi05_libero prepares LIBERO frames exactly as
openpi's examples/libero/main.py does (resize_with_pad to 224, uint8) and answers 7-dim chunks;
--config robotwin_aloha serves RoboTwin's Aloha-AgileX cotrain checkpoint (evals/pi05/robotwin_config.py)
and answers 14-dim joint chunks. A request for one item goes through openpi's Policy.infer
unchanged; a batch gathered from several clients is padded to a compiled size and sampled in one call.
A request carrying the episode's seed draws its flow-matching noise from (seed, episode_step), so an
episode samples the same chunks whichever clients share its batches; without it the noise comes
from the policy's own generator, as openpi serves it.

Usage:
    external/openpi/.venv/bin/python evals/pi05/ws_server.py --checkpoint <dir> --port 14090
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

REPO_ROOT = os.environ.get("ROBOT_OOD_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
HARNESS_DIR = os.path.join(REPO_ROOT, "evals", "libero_plus", "harness")
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)
os.environ.setdefault("OPENPI_DATA_HOME", os.path.join(REPO_ROOT, "pi05"))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
# Compiled batch sizes persist here, so a server's warm-up is seconds after the first launch.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(os.environ["OPENPI_DATA_HOME"], "jax_cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.environ["OPENPI_DATA_HOME"], "torch_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.environ["OPENPI_DATA_HOME"], "triton_cache"))
# cuBLAS picks its reduction split from the workspace it finds free, which varies with what else
# is in flight; pinning the workspace is what makes a batched matmul repeat itself run to run.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

POLICY_IMAGE_SIZE = 224
DEFAULT_CONFIG = "pi05_libero"
ROBOTWIN_CONFIG = "robotwin_aloha"     # evals/pi05/robotwin_config.py, not an openpi config name
DEFAULT_MAX_BATCH = 8


def _observation(inputs):
    """openpi's Observation for a batch of transformed inputs, JAX arrays or torch tensors."""
    from openpi.models import model as _model

    return _model.Observation.from_dict(inputs)


def batch_buckets(max_batch: int) -> tuple[int, ...]:
    """The batch sizes to compile: every size up to max_batch, so no batch is padded.

    Args:
        max_batch: The largest batch the server will be handed.

    Returns:
        Ascending sizes. Compiled executables persist in $JAX_COMPILATION_CACHE_DIR, so the
        compile cost is paid once per machine.
    """
    return tuple(range(1, max(1, int(max_batch)) + 1))


def chunk_noise(seed: int, step: int, shape: tuple[int, int]) -> np.ndarray:
    """The flow-matching noise of one chunk, a function of the episode's seed and step alone.

    Args:
        seed: The episode's scene seed.
        step: The env step the chunk is sampled at.
        shape: (action_horizon, action_dim).

    Returns:
        float32 standard normal noise, the same on every call and every machine.
    """
    return np.random.default_rng([int(seed), int(step)]).standard_normal(shape, dtype=np.float32)



def _pin_kernels() -> None:
    """Make the forward pass repeat itself bit for bit across runs.

    The sampling noise is already a function of (seed, episode_step), and the simulator replays a
    fixed action tape bit for bit, so an episode that is replayed with the same seed should follow
    the same trajectory. It does not: autotuned convolution algorithms and reduction orders chosen
    from whatever workspace happens to be free differ between processes, and a closed loop turns a
    last-bit difference into a different outcome. That is a noise floor on every A/B this benchmark
    is used for, so the kernels are pinned instead of left to the autotuner.
    """
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class Pi05WSPolicy:
    """openpi policy exposing predict_action(**payload) for WebsocketPolicyServer."""

    def __init__(self, checkpoint: str, config_name: str = DEFAULT_CONFIG,
                 max_batch: int = DEFAULT_MAX_BATCH) -> None:
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as train_config
        _pin_kernels()
        logging.info("Loading %s from %s", config_name, checkpoint)
        self.robotwin = config_name == ROBOTWIN_CONFIG
        if self.robotwin:
            import robotwin_config
            self.policy = robotwin_config.load_policy(checkpoint)
        else:
            ckpt_dir = download.maybe_download(checkpoint)
            self.policy = policy_config.create_trained_policy(train_config.get_config(config_name), ckpt_dir)
        self.pytorch = bool(getattr(self.policy, "_is_pytorch_model", False))
        self.buckets = batch_buckets(max_batch)
        model = self.policy._model
        spec = getattr(model, "config", model)      # the PyTorch model keeps its config; the JAX one the fields
        self.noise_shape = (int(spec.action_horizon), int(spec.action_dim))
        self._warm_up()
        logging.info("pi0.5 policy ready, batch sizes %s", list(self.buckets))

    def _warm_up(self) -> None:
        """Compile (or load from the cache) every batch size now, so no client pays it inside an episode."""
        if self.robotwin:
            blank = np.zeros((240, 320, 3), np.uint8)
            element = self._aloha_element([blank, blank, blank], np.zeros(14, np.float32), "warm up")
        else:
            blank = np.zeros((POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE, 3), np.uint8)
            element = self._element(blank, blank, np.zeros(8, np.float32), "warm up")
        # A JAX policy compiles one executable per batch size; a PyTorch one has nothing to compile,
        # so it only runs the widest batch once to page the weights in.
        # A JAX policy compiles one executable per batch size; a PyTorch one pads every batch to the
        # widest, so it compiles that shape once.
        for b in (self.buckets[-1],) if self.pytorch else self.buckets:
            t0 = time.monotonic()
            self._infer([element] * b) if b > 1 else self.policy.infer(element)
            logging.info("warmed batch %d in %.1f s", b, time.monotonic() - t0)

    @staticmethod
    def _element(primary, wrist, proprio, instruction) -> dict:
        from openpi_client import image_tools
        return {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(primary, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)),
            "observation/state": np.asarray(proprio, dtype=np.float32).reshape(-1),
            "prompt": str(instruction),
        }

    @staticmethod
    def _aloha_element(views, state, instruction) -> dict:
        """RoboTwin's observation as openpi's Aloha input: three cameras channel-first, 14 joints."""
        import robotwin_config
        images = {name: np.ascontiguousarray(np.transpose(np.asarray(view, dtype=np.uint8), (2, 0, 1)))
                  for name, view in zip(robotwin_config.CAMERAS, views)}
        return {"images": images, "state": np.asarray(state, dtype=np.float32).reshape(-1), "prompt": str(instruction)}

    def _infer_torch(self, elements: list[dict], noise: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """openpi's Policy.infer over several elements at once, for a PyTorch checkpoint.

        Every batch is padded to the widest bucket: torch.compile compiles per shape and each new
        one costs about a minute, so the server pays for one shape and reuses it.

        Args:
            elements: Transformed-input dicts, one per item.
            noise: Per item, the chunk's flow-matching noise; None draws it from the model.

        Returns:
            Per item, the unnormalized action chunk.
        """
        import jax
        import torch
        policy = self.policy
        device = policy._pytorch_device
        n = len(elements)
        padded = elements + [elements[-1]] * (self.buckets[-1] - n)
        items = [policy._input_transform(jax.tree.map(lambda x: x, e)) for e in padded]
        inputs = jax.tree.map(lambda *xs: torch.from_numpy(np.stack([np.asarray(x) for x in xs])).to(device), *items)
        observation = _observation(inputs)
        kwargs = dict(policy._sample_kwargs)
        if noise is not None:
            kwargs["noise"] = torch.from_numpy(_padded_noise(noise, len(padded))).to(device)
        with torch.no_grad():
            actions = policy._sample_actions(device, observation, **kwargs)
        actions = np.asarray(actions.detach().cpu(), dtype=np.float32)
        states = np.asarray(inputs["state"].detach().cpu())
        return [np.asarray(policy._output_transform({"state": states[i], "actions": actions[i]})["actions"],
                           dtype=np.float32) for i in range(n)]

    def _infer(self, elements: list[dict], noise: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """openpi's Policy.infer over several elements at once, padded to a compiled batch size.

        Args:
            elements: Transformed-input dicts, one per item.
            noise: Per item, the chunk's flow-matching noise; None draws it from the policy's generator.

        Returns:
            Per item, the unnormalized action chunk, as Policy.infer would return it.
        """
        if self.pytorch:
            return self._infer_torch(elements, noise)
        import jax
        import jax.numpy as jnp
        policy = self.policy
        n = len(elements)
        padded = elements + [elements[-1]] * (self.buckets[-1] - n)
        items = [policy._input_transform(jax.tree.map(lambda x: x, e)) for e in padded]
        inputs = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *items)
        policy._rng, rng = jax.random.split(policy._rng)
        observation = _observation(inputs)
        kwargs = dict(policy._sample_kwargs)
        if noise is not None:
            kwargs["noise"] = jnp.asarray(_padded_noise(noise, len(padded)))
        actions = np.asarray(policy._sample_actions(rng, observation, **kwargs))
        states = np.asarray(inputs["state"])
        return [np.asarray(policy._output_transform({"state": states[i], "actions": actions[i]})["actions"],
                           dtype=np.float32) for i in range(n)]

    def predict_action(self, batch_images, instructions, state=None, episode_step=None,
                       worker_id=None, seed=None, **kw):
        """Action chunks for a batch of observations.

        Args:
            batch_images: Per item, [primary, wrist] uint8 frames already rotated 180 degrees.
            instructions: Per item, the task string.
            state: Per item, the 8-dim proprio [eef_pos(3), axis_angle(3), gripper_qpos(2)].
            episode_step: Per item, the env step the chunk is sampled at.
            seed: Per item, the episode's scene seed; with it the chunk's noise is chunk_noise(seed,
                episode_step), without it the policy draws its own.

        Returns:
            actions: float32 array [B, chunk, 7], the gripper in the environment's own convention.
        """
        noise = None
        if seed is not None:
            noise = [chunk_noise(s, t, self.noise_shape) for s, t in zip(seed, episode_step)]
        elements = []
        for i, images in enumerate(batch_images):
            if self.robotwin:
                elements.append(self._aloha_element(images, state[i], instructions[i]))
                continue
            primary = np.asarray(images[0])
            wrist = np.asarray(images[1]) if len(images) > 1 else primary
            proprio = np.asarray(state[i], dtype=np.float32) if state is not None else np.zeros(8, np.float32)
            elements.append(self._element(primary, wrist, proprio, instructions[i]))
        if len(elements) == 1 and not self.pytorch:
            single = self.policy.infer(elements[0], noise=None if noise is None else noise[0])
            out = [np.asarray(single["actions"], dtype=np.float32)]
        else:
            widest = self.buckets[-1]
            out = [a for i in range(0, len(elements), widest)
                   for a in self._infer(elements[i:i + widest],
                                        None if noise is None else noise[i:i + widest])]
        return {"actions": np.stack(out, axis=0)}


def _padded_noise(noise: list[np.ndarray], width: int) -> np.ndarray:
    """[width, H, D] the items' noise, the last item's repeated over the padding rows."""
    return np.stack(list(noise) + [noise[-1]] * (width - len(noise)), axis=0).astype(np.float32)


def main() -> None:
    """Serve the policy until killed."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="checkpoint directory or gs:// path")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"openpi training config name, or {ROBOTWIN_CONFIG} for RoboTwin's checkpoint")
    p.add_argument("--port", type=int, default=14090)
    p.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH,
                   help="requests folded into one sample_actions call; 8 doubled items/s on an A100")
    p.add_argument("--batch-wait-ms", type=float, default=3.0,
                   help="how long a batch waits for stragglers after the first request")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    policy = Pi05WSPolicy(args.checkpoint, args.config, max_batch=args.max_batch)
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    server = WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port,
                                   metadata={"env": "robotwin" if args.config == ROBOTWIN_CONFIG else "libero",
                                             "model": args.config},
                                   max_batch=args.max_batch, batch_wait_ms=args.batch_wait_ms)
    logging.info("pi0.5 websocket policy server running on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
