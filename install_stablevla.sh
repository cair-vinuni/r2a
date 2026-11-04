#!/usr/bin/env bash
# Install the toolchain: micromamba, the `stablevla` env (policy + simulator) and the `onf-core`
# env (the robot_ood package + the eval driver). Idempotent. Everything lands inside this
# repository; nothing is written under $HOME.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
MM=$MAMBA_ROOT_PREFIX/bin/micromamba
ENV=$MAMBA_ROOT_PREFIX/envs/stablevla
PY=$ENV/bin/python
CORE=$MAMBA_ROOT_PREFIX/envs/onf-core
CORE_PY=$CORE/bin/python
FLASH_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
step() { echo; echo "=== $* ==="; }

step "0a. micromamba (downloaded into $MAMBA_ROOT_PREFIX; no conda needed on the machine)"
if [ ! -x "$MM" ]; then
    mkdir -p "$MAMBA_ROOT_PREFIX/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C "$MAMBA_ROOT_PREFIX" bin/micromamba
fi
"$MM" --version

# mesalib + libgl/libopengl/libegl: headless rendering through mesa's EGL and llvmpipe when the
# machine has no NVIDIA GL driver. imagemagick: `Wand`, imported at module scope by LIBERO.
step "0b. env python3.10 + mesa/EGL + imagemagick (conda-forge)"
[ -x "$PY" ] || "$MM" create -y -p "$ENV" -c conda-forge \
    python=3.10 mesalib libgl libopengl libegl imagemagick

step "1. torch 2.2.0 (cu121)"
"$PY" -m pip install -q torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

step "2. StableVLA (vla_adapter, editable); pulls the OpenVLA-OFT transformers fork and tensorflow"
"$PY" -m pip install -q -e ./stablevla

step "3. flash-attn 2.5.5 (prebuilt wheel, no build)"
"$PY" -m pip install -q --no-deps "$FLASH_WHL"

# Every cloned LIBERO fork is installed editable. They all ship the top-level package `libero`, so
# the PYTHONPATH at run time decides which one is imported (robot_ood.eval.bench).
step "4. cloned LIBERO fork(s) + sim stack"
for tree in external/LIBERO-plus external/LIBERO; do
    [ -d "$tree" ] && "$PY" -m pip install -q --no-deps -e "./$tree"
done
"$PY" -m pip install -q \
    robosuite==1.4.1 mujoco==3.3.0 bddl==3.5.0 PyOpenGL==3.1.10 \
    opencv-python-headless==4.9.0.80 scipy h5py pyyaml tqdm \
    cloudpickle "gym==0.25.2" future tyro requests easydict scikit-image \
    matplotlib imageio imageio-ffmpeg Wand

step "5. websocket policy-server transport"
"$PY" -m pip install -q fastapi uvicorn websockets msgpack msgpack-numpy json-numpy

# pytorch_kinematics: forward kinematics of the arm, used at training time by the chunk, segment
# and fusion objectives. Deploy reads the pose from the robot, so onf-core does not need it, but
# the training steps run in this env.
step "6. pytorch_kinematics (training only, see src/robot_ood/core/kinematics.py)"
"$PY" -m pip install -q pytorch_kinematics

step "7. final pins: numpy<2 (tensorflow 2.15 + robosuite), tensorflow-metadata 1.14 (newer needs protobuf>=5, tf 2.15 pins 4.25)"
"$PY" -m pip install -q numpy==1.26.4 tensorflow-metadata==1.14.0 wandb==0.17.9

step "8. env onf-core (the robot_ood package, the eval driver and pytest)"
[ -x "$CORE_PY" ] || "$MM" create -y -p "$CORE" -c conda-forge python=3.10 pip
# pytorch_kinematics and matplotlib are training only, but tests/ exercises the training path.
"$CORE_PY" -m pip install -q -e . "numpy==1.26.4" pytest pytorch_kinematics matplotlib

step "SELF-TEST"
"$PY" - <<'PYEOF'
import torch, mujoco, robosuite, transformers, pytorch_kinematics
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "| robosuite", robosuite.__version__)
import flash_attn; print("flash_attn", flash_attn.__version__)
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=".1"/></body></worldbody></mujoco>')
d = mujoco.MjData(m); r = mujoco.Renderer(m, 64, 64); mujoco.mj_forward(m, d); r.update_scene(d)
print("EGL/llvmpipe render", r.render().shape)
PYEOF
if [ -d external/LIBERO-plus ]; then
    LIBERO_CONFIG_PATH="$PWD/external/LIBERO-plus/libero" PYTHONPATH="$PWD/external/LIBERO-plus" \
    "$PY" -c "
from libero.libero.benchmark import get_ids_by_category as g
print('LIBERO-plus Robot Initial States:', {k: len(v[0]) for k, v in g('Robot Initial States').items()})"
fi
"$CORE_PY" -c "import robot_ood, torch, numpy; print('onf-core ok | torch', torch.__version__, '| numpy', numpy.__version__)"
