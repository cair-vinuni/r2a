"""Paths and settings for the ALOHA right-arm deployment. Edit here, or set the named env vars."""

from __future__ import annotations

import os
from pathlib import Path


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env) or default).expanduser()


ROOT = _path("ROBOT_OOD_ROOT", Path(__file__).resolve().parents[2])
SRC = _path("ROBOT_OOD_SRC", ROOT / "src")
ARTIFACTS = _path("ROBOT_OOD_GRAPH_DIR", ROOT / "outputs" / "aloha_right" / "artifacts")
CHECKPOINT = _path("ALOHA_CKPT", ROOT / "gr00t" / "hf_weights" / "aloha_right_multitask_local")

LANG_MODEL = os.environ.get("ROBOT_OOD_LANG_MODEL") or ""
if LANG_MODEL:
    os.environ["ROBOT_OOD_LANG_MODEL"] = LANG_MODEL

SERVER_IP = os.environ.get("SERVER_INFERENCE_IP") or "192.168.60.11"
PORT = int(os.environ.get("SERVER_INFERENCE_PORT") or 5558)
GPU = os.environ.get("ALOHA_GPU") or "0"

HOMING = (os.environ.get("ALOHA_HOMING") or "1") == "1"
LANG_PRELOAD = (os.environ.get("ALOHA_LANG_PRELOAD") or "1") == "1"
os.environ["ROBOT_OOD_LANG_QUERIES"] = "1" if LANG_PRELOAD else "0"
LOG_CORRECTION = (os.environ.get("ALOHA_LOG_CORRECTION") or "1") == "1"
