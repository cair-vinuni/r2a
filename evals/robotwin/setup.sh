#!/usr/bin/env bash
# evals/robotwin/setup.sh — provision external/RoboTwin so `--bench robotwin` runs RoboTwin 2.0.
#
# Idempotent. Clones RoboTwin-Platform/RoboTwin at a pinned commit, builds the `robotwin` conda env
# inside the repo (Python 3.10, torch 2.4.1, SAPIEN 3.0.0b1, mplib, curobo, pytorch3d), downloads
# the benchmark assets from the RoboTwin2.0 dataset, and writes the Vulkan ICD manifest that lets
# SAPIEN's ray tracer find the NVIDIA driver extracted for EGL ($ROBOT_OOD_NVGL). Nothing under
# $HOME, no sudo. One patch is applied to the checkout: the render shader comes from
# $ROBOTWIN_SHADER (default rt, the benchmark's own) so a rasterised run can never be mistaken for
# a ray-traced one.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ./env.sh

ROOT=external/RoboTwin
COMMIT=6e75b8b80ee32d2f495eb756cf5687e4e16f0eea
CUROBO_COMMIT=ebb71702f3f70e767f40fd8e050674af0288abe8
ENV=robotwin
MM="$MAMBA_ROOT_PREFIX/bin/micromamba"
PY="$MAMBA_ROOT_PREFIX/envs/$ENV/bin/python"
DATASET=TianxingChen/RoboTwin2.0
PATCH=evals/robotwin/patches/shader-from-env.patch
step() { echo; echo "=== $* ==="; }

step "1. checkout $ROOT @ ${COMMIT:0:7}"
if [ ! -d "$ROOT/.git" ]; then
    git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ROOT"
fi
git -C "$ROOT" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$ROOT" checkout --quiet "$COMMIT"
git -C "$ROOT" rev-parse --short HEAD
if git -C "$ROOT" apply --reverse --check "$PWD/$PATCH" 2>/dev/null; then
    echo "shader patch already applied"
else
    git -C "$ROOT" apply "$PWD/$PATCH" && echo "shader patch applied"
fi

step "2. environment $ENV"
if [ ! -x "$PY" ]; then
    "$MM" create -y -q -n "$ENV" -c conda-forge python=3.10 libvulkan-loader vulkan-tools xorg-libx11 xorg-libxext
fi
# Description generation (azure, openai, wandb) is not part of running the benchmark.
grep -v -E '^(azure|openai|wandb)' "$ROOT/script/requirements.txt" > "$ROOT/script/requirements.eval.txt"
"$PY" -m pip install -q "setuptools<81" -r "$ROOT/script/requirements.eval.txt" websockets msgpack
SITE=$("$PY" -c 'import sapien, mplib, os; print(os.path.dirname(sapien.__file__), os.path.dirname(mplib.__file__))')
# The two edits RoboTwin's own script/_install.sh makes to its dependencies.
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${SITE% *}/wrapper/urdf_loader.py"
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${SITE#* }/planner.py"

step "3. pytorch3d + curobo (compiled against torch's CUDA)"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0"} MAX_JOBS=${MAX_JOBS:-16}
"$PY" -c 'import pytorch3d' 2>/dev/null || "$PY" -m pip install -q --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
if [ ! -d "$ROOT/envs/curobo/.git" ]; then
    git clone https://github.com/NVlabs/curobo.git "$ROOT/envs/curobo"
fi
git -C "$ROOT/envs/curobo" checkout --quiet "$CUROBO_COMMIT"
"$PY" -c 'import curobo' 2>/dev/null || "$PY" -m pip install -q --no-build-isolation -e "$ROOT/envs/curobo"
# curobo of that date reads warp's torch bridge as wp.torch; warp 1.9+ makes it an explicit import.
"$PY" -m pip install -q "warp-lang==1.8.1"

step "4. assets from $DATASET"
"$PY" - "$DATASET" "$ROOT/assets" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
for name in ("embodiments.zip", "objects.zip", "background_texture.zip"):
    print(hf_hub_download(sys.argv[1], name, repo_type="dataset", local_dir=sys.argv[2]))
PYEOF
for name in embodiments objects background_texture; do
    [ -d "$ROOT/assets/$name" ] || unzip -q "$ROOT/assets/$name.zip" -d "$ROOT/assets"
done
(cd "$ROOT" && "$PY" script/update_embodiment_config_path.py </dev/null)

step "5. Vulkan ICD manifest for the extracted NVIDIA driver"
if [ -f "${ROBOT_OOD_NVGL:-/none}/lib/libGLX_nvidia.so.0" ]; then
    printf '{\n  "file_format_version": "1.0.1",\n  "ICD": {"library_path": "%s/lib/libGLX_nvidia.so.0", "api_version": "1.4.341"}\n}\n' \
        "$ROBOT_OOD_NVGL" > "$ROBOT_OOD_NVGL/nvidia_icd.json"
    cat "$ROBOT_OOD_NVGL/nvidia_icd.json"
    # Beyond what EGL needs, a Vulkan device with ray tracing loads the RT core library, and the
    # OptiX denoiser its own; link them beside the EGL libraries when the extraction has them.
    for lib in libnvidia-rtcore.so.* libnvidia-tileiras.so.* libnvoptix.so.* nvoptix.bin; do
        for f in "$ROBOT_OOD_NVGL"/extracted/$lib; do
            [ -e "$f" ] && ln -sf "$f" "$ROBOT_OOD_NVGL/lib/$(basename "$f")"
        done
    done
    ls "$ROBOT_OOD_NVGL/lib" | grep -c nvidia
else
    echo "no $ROBOT_OOD_NVGL/lib: SAPIEN will render through whatever Vulkan driver the system has"
fi

step "6. import check"
source ./env.sh
"$PY" -c "import sapien, mplib, curobo, pytorch3d, torch; print('sapien', sapien.__version__, 'torch', torch.__version__, torch.cuda.is_available())"

echo
echo "ready: python -m robot_ood.cli eval --bench robotwin --suite g1 --axis Randomized --mode base --policy pi05_robotwin --gpus 0,1,2,3"
