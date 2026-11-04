#!/usr/bin/env bash
# evals/pi05/setup.sh — provision openpi and the pi05_libero checkpoint so `--policy pi05` runs.
#
# Idempotent. Clones Physical-Intelligence/openpi at a pinned commit into external/openpi, builds its
# own uv environment (Python 3.11, JAX CUDA wheels), downloads the published LIBERO checkpoint the
# way openpi itself does, and RoboTwin's pi0.5 cotrain checkpoint (XPolicyLab, Aloha-AgileX, joint
# actions) from the RoboTwin2.0 dataset, both into pi05/. Nothing is written under $HOME.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ./env.sh

ROOT=external/openpi
COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
CKPT=gs://openpi-assets/checkpoints/pi05_libero
export OPENPI_DATA_HOME="$PWD/pi05"
step() { echo; echo "=== $* ==="; }

step "1. checkout $ROOT @ ${COMMIT:0:7}"
if [ ! -d "$ROOT/.git" ]; then
    git clone https://github.com/Physical-Intelligence/openpi.git "$ROOT"
fi
git -C "$ROOT" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$ROOT" checkout --quiet "$COMMIT"
git -C "$ROOT" rev-parse --short HEAD

step "2. environment $ROOT/.venv"
(cd "$ROOT" && GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 && uv pip install websockets hf_transfer)

step "3. checkpoint $CKPT -> $OPENPI_DATA_HOME"
LOCAL="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
if [ ! -f "$LOCAL/params/_METADATA" ] && [ ! -d "$LOCAL/params" ]; then
    "$ROOT/.venv/bin/python" -c "from openpi.shared import download; print(download.maybe_download('$CKPT'))"
fi
ls "$LOCAL"

step "4. RoboTwin 2.0 cotrain checkpoint (XPolicyLab) -> pi05/robotwin"
RT_TAR=cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz
RT_DIR="$OPENPI_DATA_HOME/robotwin/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999"
if [ ! -d "$RT_DIR/params" ]; then
    "$ROOT/.venv/bin/python" - "$RT_TAR" "$OPENPI_DATA_HOME/robotwin" <<'PYEOF'
import os, sys
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download
print(hf_hub_download("TianxingChen/RoboTwin2.0", sys.argv[1], repo_type="dataset", local_dir=sys.argv[2]))
PYEOF
    tar xzf "$OPENPI_DATA_HOME/robotwin/$RT_TAR" -C "$OPENPI_DATA_HOME/robotwin"
    rm -f "$OPENPI_DATA_HOME/robotwin/$RT_TAR"
fi
ls "$RT_DIR"

step "5. import check"
"$ROOT/.venv/bin/python" -c "import openpi, openpi_client, websockets, msgpack, jax; print('jax', jax.__version__, jax.devices())"

echo
echo "ready: python -m robot_ood.cli eval --bench libero --suite long --mode base --policy pi05 --gpus 0"
