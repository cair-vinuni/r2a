"""Finetune pi0.5 on a RoboTwin LeRobot dataset with openpi's own trainer.

openpi's scripts look their config up by name; this wrapper registers the RoboTwin finetune config
(`robotwin_config.finetune_config`) and runs the shipped `compute_norm_stats.py` and `train.py`
mains on it, so the training loop, sharding and checkpoint layout are openpi's, unchanged.
Training uses every visible GPU: the global batch is split across them (`--fsdp-devices 1`), or
parameters are additionally sharded over that many devices when a replica does not fit.

    P=external/openpi/.venv/bin/python
    $P evals/pi05/train_robotwin.py norm-stats --repo-id robotwin_g1_clean
    $P evals/pi05/train_robotwin.py train --repo-id robotwin_g1_clean --exp-name run0

Datasets are read from pi05/lerobot/<repo-id>, pi0.5's base weights from pi05/openpi-assets.
Checkpoints: pi05/checkpoints/<repo-id>/<exp-name>/<step>, each with assets/<repo-id>/norm_stats.json,
which is what `--policy pi05_robotwin_clean` serves (`ws_server.py --config robotwin_aloha`).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("OPENPI_DATA_HOME", str(REPO_ROOT / "pi05"))
os.environ.setdefault("HF_LEROBOT_HOME", str(REPO_ROOT / "pi05" / "lerobot"))
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(REPO_ROOT / "pi05" / "jax_cache"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import robotwin_config  # noqa: E402

OPENPI_SCRIPTS = pathlib.Path("external/openpi/scripts")


def _script(name: str):
    """openpi's scripts/<name>.py as a module."""
    if str(OPENPI_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(OPENPI_SCRIPTS))
    return __import__(name)


def _config(args):
    cfg = robotwin_config.finetune_config(
        args.repo_id, args.exp_name, batch_size=args.batch_size, num_train_steps=args.steps,
        fsdp_devices=args.fsdp_devices, num_workers=args.num_workers, save_interval=args.save_interval,
        lora=args.lora, overwrite=args.overwrite, resume=args.resume,
    )
    robotwin_config.register(cfg)
    return cfg


def norm_stats(args) -> None:
    cfg = _config(args)
    _script("compute_norm_stats").main(cfg.name, max_frames=args.max_frames)


def train(args) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    cfg = _config(args)
    robotwin_config.write_manifest(cfg.checkpoint_dir, repo_id=args.repo_id, lora=args.lora,
                                   batch_size=args.batch_size, num_train_steps=args.steps,
                                   remat_policy=os.environ.get("OPENPI_REMAT_POLICY", "nothing_saveable"))
    _script("train").main(cfg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-id", required=True, help="dataset under $HF_LEROBOT_HOME")
    common.add_argument("--exp-name", default="run0", help="checkpoint directory name")
    common.add_argument("--batch-size", type=int, default=32, help="global batch")
    common.add_argument("--steps", type=int, default=30_000)
    common.add_argument("--fsdp-devices", type=int, default=1, help="devices per parameter shard")
    common.add_argument("--num-workers", type=int, default=8)
    common.add_argument("--save-interval", type=int, default=5_000)
    common.add_argument("--lora", action="store_true", help="train the LoRA adapters, not every weight")
    common.add_argument("--overwrite", action="store_true")
    common.add_argument("--resume", action="store_true")

    ns = sub.add_parser("norm-stats", parents=[common], help="state/action statistics -> pi05/assets")
    ns.add_argument("--max-frames", type=int, default=None, help="sample this many frames instead of all")
    ns.set_defaults(fn=norm_stats)
    tr = sub.add_parser("train", parents=[common], help="finetune from pi05_base")
    tr.set_defaults(fn=train)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
