import os
import sys
from pathlib import Path


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env) or default).expanduser()


ROOT = _path("ROBOT_OOD_ROOT", Path(__file__).resolve().parents[2])
SRC = ROOT / "src"
HF_HOME = _path("HF_HOME", ROOT / ".cache" / "hf")
os.environ["HF_HOME"] = str(HF_HOME)
# Lets the tokenizer load from the local cache when the Hub cannot be reached.
os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")

BUCKET = "hf://buckets/twanghcmut/robot-ood-iclr"
ARTIFACTS = _path("VRH31_ARTIFACTS", ROOT / "outputs" / "vrh31_waist" / "artifacts")
CHECKPOINT = _path("VRH31_CKPT", ARTIFACTS / "gr00t-n17-vrh31-pick-100eps")
PRIOR = ARTIFACTS / "prior"
DATASET = _path("VRH31_DATASET", ROOT.parent / "vrh31_bds_pick_100eps")
INSTRUCTION = "Grasp the large curved metal frame with both hands and lift it upward from the rack."

SERVER_IP = os.environ.get("SERVER_INFERENCE_IP") or "192.168.60.11"
PORT = int(os.environ.get("SERVER_INFERENCE_PORT") or 5558)
GPU = os.environ.get("VRH31_GPU") or "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU)

VIDEO_KEY = "cam_head"
IMAGE_SHAPE = (600, 960)  # cam_head in the demos: observation.images.color.head


def checkpoint_id(checkpoint: Path) -> str:
    """Hash of the checkpoint's normalisation statistics, which differ for every finetune."""
    import hashlib

    stats = next(p for p in (checkpoint / "processor" / "statistics.json", checkpoint / "statistics.json")
                 if p.exists())
    return hashlib.sha256(stats.read_bytes()).hexdigest()[:12]


if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
