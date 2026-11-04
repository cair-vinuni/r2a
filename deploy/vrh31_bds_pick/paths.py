from __future__ import annotations

import os
from pathlib import Path

ARMS = ("left", "right")


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env) or default).expanduser()


ROOT = _path("ROBOT_OOD_ROOT", Path(__file__).resolve().parents[2])
SRC = _path("ROBOT_OOD_SRC", ROOT / "src")
HF_HOME = _path("HF_HOME", ROOT / ".local" / "cache" / "huggingface")
os.environ["HF_HOME"] = str(HF_HOME)
ARTIFACTS = {
    "left": _path("VRH31_GRAPH_DIR_LEFT", ROOT / "outputs" / "vrh31_bds_pick_left" / "artifacts"),
    "right": _path("VRH31_GRAPH_DIR_RIGHT", ROOT / "outputs" / "vrh31_bds_pick_right" / "artifacts"),
}
CHECKPOINT = _path("VRH31_CKPT", ROOT / "gr00t" / "hf_weights" / "vrh31_bds_pick_local")

LANG_MODEL = os.environ.get("ROBOT_OOD_LANG_MODEL") or ""
if LANG_MODEL:
    os.environ["ROBOT_OOD_LANG_MODEL"] = LANG_MODEL

SERVER_IP = os.environ.get("SERVER_INFERENCE_IP") or "192.168.60.11"
PORT = int(os.environ.get("SERVER_INFERENCE_PORT") or 5558)
GPU = os.environ.get("VRH31_GPU") or "0"

HOMING = (os.environ.get("VRH31_HOMING") or "1") == "1"
LANG_PRELOAD = (os.environ.get("VRH31_LANG_PRELOAD") or "1") == "1"
os.environ["ROBOT_OOD_LANG_QUERIES"] = "1" if LANG_PRELOAD else "0"
LOG_CORRECTION = (os.environ.get("VRH31_LOG_CORRECTION") or "1") == "1"
