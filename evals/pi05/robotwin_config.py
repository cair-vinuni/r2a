"""The openpi training config of pi0.5 on RoboTwin's Aloha-AgileX, for inference and finetuning.

XPolicyLab's fork of openpi trains RoboTwin's cotrain checkpoint under a config openpi does not
ship: pi0.5, the LeRobot Aloha data pipeline with adapt_to_pi off (a simulated robot, no Trossen
joint remap), delta joint actions, and the prompt taken from the task. This module states that
config for the openpi checkout in external/ and loads a checkpoint with its own normalisation
statistics, so a policy is served exactly as it was trained.

The same config, given a dataset (`robotwin_data.py`) and pi0.5's base weights, is the finetune
RoboTwin's leaderboard runs (policy/pi0's `*_aloha_robotwin_full`: full finetune, batch 32,
30k steps, openpi's default cosine schedule): `finetune_config`, run by `train_robotwin.py`.
"""

from __future__ import annotations

import os
import pathlib

COTRAIN_ASSET_ID = "RoboTwin_lerobot_v30"   # the norm-stats directory shipped inside the cotrain checkpoint
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
STATE_DIM = 14
BASE_WEIGHTS = "gs://openpi-assets/checkpoints/pi05_base/params"
ASSETS_BASE_DIR = "pi05/assets"
CHECKPOINT_BASE_DIR = "pi05/checkpoints"


METADATA = "metadata.pt"     # a PyTorch checkpoint's own TrainConfig, as openpi's torch trainer writes it


def trained_under(checkpoint: pathlib.Path) -> dict:
    """The training contract a checkpoint states about itself, empty when it states none.

    A checkpoint openpi's PyTorch trainer wrote carries its whole TrainConfig pickled beside the
    weights. The pickle names classes of whatever fork trained it -- a learning-rate schedule it
    defined itself, a transform written as a lambda -- which are not on this side; each unknown
    name is stubbed as it is asked for, since nothing here reads them. What is read is the shape
    of the model and of the data pipeline, so a checkpoint is served the way it was trained.

    Args:
        checkpoint: The checkpoint directory.

    Returns:
        model and data, as the mappings the trainer recorded, or {}.
    """
    import importlib
    import re

    path = pathlib.Path(checkpoint) / METADATA
    if not path.exists():
        return {}
    import torch

    missing = re.compile(r"Can't get attribute '([^']+)' on <module '([^']+)'")
    for _ in range(64):
        try:
            doc = torch.load(path, map_location="cpu", weights_only=False)
            break
        except AttributeError as exc:
            found = missing.search(str(exc))
            if not found:
                raise
            name, module = found.groups()
            setattr(importlib.import_module(module), name,
                    type(name, (), {"__init__": lambda self, *a, **k: None}))
    else:
        raise RuntimeError(f"{path}: too many names this side does not have")
    return dict(doc.get("config") or {})


def _data_config(repo_id: str, contract: dict):
    import openpi.transforms as _transforms
    from openpi.training import config as _config

    data = contract.get("data") or {}
    return _config.LeRobotAlohaDataConfig(
        repo_id=repo_id,
        # RoboTwin is simulated, so the cotrain checkpoint was trained with no Trossen joint remap;
        # a checkpoint trained with one says so, and is served with it.
        adapt_to_pi=bool(data.get("adapt_to_pi", False)),
        default_prompt=data.get("default_prompt"),
        repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform({
            "images": {c: f"observation.images.{c}" for c in CAMERAS},
            "state": "observation.state", "actions": "action", "prompt": "prompt",
        })]),
        base_config=_config.DataConfig(prompt_from_task=True),
    )


def model_config(lora: bool = False, compile_mode: str | None = "max-autotune", contract: dict | None = None):
    """pi0.5, all of it trainable, or with the LoRA adapters RoboTwin's low-rank finetune trains.

    `compile_mode` is what a PyTorch checkpoint's sampler is compiled with; it does not touch the
    JAX path. "max-autotune" costs minutes of tuning per batch shape at every server start,
    "default" compiles in seconds, and None runs eager.
    """
    from openpi.models import pi0_config

    variants = {} if not lora else {"paligemma_variant": "gemma_2b_lora",
                                    "action_expert_variant": "gemma_300m_lora"}
    model = (contract or {}).get("model") or {}
    horizon = {"action_horizon": int(model["action_horizon"])} if "action_horizon" in model else {}
    return pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=compile_mode, **horizon, **variants)


def train_config(repo_id: str = COTRAIN_ASSET_ID, name: str = "robotwin_aloha", *, lora: bool = False,
                 compile_mode: str | None = "max-autotune", contract: dict | None = None):
    """The TrainConfig a RoboTwin checkpoint was trained under, minus training-only fields.

    `contract` is what the checkpoint says about itself (`trained_under`); the fields it does not
    state are the cotrain checkpoint's, which states none.
    """
    from openpi.training import config as _config

    return _config.TrainConfig(name=name, model=model_config(lora, compile_mode, contract),
                               data=_data_config(repo_id, contract or {}))


def finetune_config(repo_id: str, exp_name: str, *, name: str | None = None, batch_size: int = 32,
                    num_train_steps: int = 30_000, fsdp_devices: int = 1, num_workers: int = 8,
                    save_interval: int = 5_000, lora: bool = False, overwrite: bool = False,
                    resume: bool = False):
    """pi0.5 finetuned from its base weights on a RoboTwin LeRobot dataset.

    Args:
        repo_id: The dataset under $HF_LEROBOT_HOME; also the norm-stats asset id of the checkpoint.
        exp_name: The run; checkpoints go to pi05/checkpoints/<name>/<exp_name>/<step>.
        name: The config name (default the repo id).
        batch_size: Global batch, split over the visible devices.
        num_train_steps: Optimizer steps.
        fsdp_devices: Devices a parameter shard spans; 1 is plain data parallel.
        num_workers: Data-loader processes.
        save_interval: Steps between checkpoints; the last step is always saved. Only the newest is
            kept -- one is 42 GB with its optimizer state, and only `params/` is served.
        lora: Train the LoRA adapters and the action expert instead of every weight, as RoboTwin's
            low-rank finetune does; EMA is off there, as openpi's own LoRA configs have it.
        overwrite: Start over in an existing checkpoint directory.
        resume: Continue from the directory's last checkpoint.

    Returns:
        The TrainConfig.
    """
    import dataclasses

    from openpi.training import weight_loaders

    base = train_config(repo_id, name=name or repo_id, lora=lora)
    return dataclasses.replace(
        base, exp_name=exp_name, weight_loader=weight_loaders.CheckpointWeightLoader(BASE_WEIGHTS),
        batch_size=batch_size, num_train_steps=num_train_steps, fsdp_devices=fsdp_devices,
        num_workers=num_workers, save_interval=save_interval, keep_period=None,
        freeze_filter=model_config(lora).get_freeze_filter(), ema_decay=None if lora else base.ema_decay,
        assets_base_dir=ASSETS_BASE_DIR, checkpoint_base_dir=CHECKPOINT_BASE_DIR,
        wandb_enabled=False, overwrite=overwrite, resume=resume,
    )


def register(config) -> None:
    """Make a config visible to openpi's `get_config`, which its scripts look configs up by."""
    from openpi.training import config as _config

    _config._CONFIGS_DICT[config.name] = config


MANIFEST = "finetune.json"     # written beside the step directories by train_robotwin.py


def write_manifest(checkpoint_dir: pathlib.Path, **fields) -> None:
    """Record how a run was trained, beside its step directories, for the server to read back."""
    import json

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / MANIFEST).write_text(json.dumps(fields, indent=2) + "\n")


def read_manifest(checkpoint: pathlib.Path) -> dict:
    """The manifest of the run a checkpoint belongs to; empty for a checkpoint trained elsewhere."""
    import json

    path = checkpoint.parent / MANIFEST
    return json.loads(path.read_text()) if path.exists() else {}


def asset_id(checkpoint: pathlib.Path) -> str:
    """The one norm-stats directory under <checkpoint>/assets, as a path relative to it."""
    assets = checkpoint / "assets"
    found = sorted(str(p.parent.relative_to(assets)) for p in assets.rglob("norm_stats.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"{assets} holds {found or 'no'} norm stats, expected exactly one")
    return found[0]


def load_policy(checkpoint: str | pathlib.Path):
    """The trained policy, with the checkpoint's own norm stats.

    A directory holding `model.safetensors` is a PyTorch checkpoint and openpi loads it as one; a
    directory holding `params/` is the JAX form. Either way the config is this module's, so the
    transforms and the action space are the ones the checkpoint was trained under.
    """
    from openpi.policies import policy_config
    from openpi.shared import normalize

    ckpt = pathlib.Path(checkpoint)
    repo_id = asset_id(ckpt)
    norm_stats = normalize.load(ckpt / "assets" / repo_id)
    lora = bool(read_manifest(ckpt).get("lora"))
    contract = trained_under(ckpt)
    # Serving compiles at every server start, so the cheap mode is the default here; the
    # checkpoint's own "max-autotune" is still available through the environment.
    compile_mode = os.environ.get("ROBOT_OOD_PI05_COMPILE", "default") or None
    return policy_config.create_trained_policy(
        train_config(repo_id, lora=lora, compile_mode=compile_mode, contract=contract),
        str(ckpt), norm_stats=norm_stats)
