REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO
export XDG_CACHE_HOME="$REPO/.local/cache"
export PIP_CACHE_DIR="$REPO/.local/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export HF_HOME="$REPO/.local/cache/huggingface"
export TORCH_HOME="$REPO/.local/cache/torch"
export MAMBA_ROOT_PREFIX="$REPO/.local/conda"
export CONDA_EXE="$MAMBA_ROOT_PREFIX/bin/micromamba"
export PATH="$MAMBA_ROOT_PREFIX/envs/stablevla/bin:$PATH"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/src:$REPO/external/LIBERO-plus"
export LIBERO_HOME="$REPO/external/LIBERO-plus"
export LIBERO_CONFIG_PATH="$REPO/external/LIBERO-plus/libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
_NV_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
export ROBOT_OOD_OUTPUTS="${ROBOT_OOD_OUTPUTS:-$REPO/outputs}"
export ROBOT_OOD_NVGL="${ROBOT_OOD_NVGL:-$(dirname "$REPO")/nvidia-gl-${_NV_DRIVER:-none}}"
if [ -f "$ROBOT_OOD_NVGL/lib/libEGL_nvidia.so.0" ]; then
  export ROBOT_OOD_NVGL_LIB="$ROBOT_OOD_NVGL/lib"
  export __EGL_VENDOR_LIBRARY_DIRS="$ROBOT_OOD_NVGL/egl_vendor.d"
  export LD_LIBRARY_PATH="$ROBOT_OOD_NVGL/lib:$MAMBA_ROOT_PREFIX/envs/stablevla/lib:${LD_LIBRARY_PATH:-}"
  if [ -d "$MAMBA_ROOT_PREFIX/envs/robotwin/lib" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/envs/robotwin/lib"
  fi
  unset MUJOCO_EGL_DEVICE_ID
  if [ -f "$ROBOT_OOD_NVGL/nvidia_icd.json" ]; then
    export VK_DRIVER_FILES="$ROBOT_OOD_NVGL/nvidia_icd.json" VK_ICD_FILENAMES="$ROBOT_OOD_NVGL/nvidia_icd.json"
  fi
else
  unset ROBOT_OOD_NVGL_LIB
  export MUJOCO_EGL_DEVICE_ID=${GPU:-0}
  export __EGL_VENDOR_LIBRARY_DIRS="$MAMBA_ROOT_PREFIX/envs/stablevla/share/glvnd/egl_vendor.d"
  export LD_LIBRARY_PATH="$MAMBA_ROOT_PREFIX/envs/stablevla/lib:${LD_LIBRARY_PATH:-}"
fi
