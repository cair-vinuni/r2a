#!/usr/bin/env bash
# Fetch the userspace GL/EGL libraries of the NVIDIA driver this box is running, without
# installing anything, so MuJoCo (LIBERO) renders on the GPU instead of mesa's llvmpipe.
#
# Why: the container is launched with NVIDIA_DRIVER_CAPABILITIES=compute,utility, so the
# toolkit injects libcuda but not libEGL_nvidia. The kernel module and the userspace GL libs
# must match to the patch version, so this script reads `nvidia-smi` and downloads exactly
# that .run file from NVIDIA, extracts it (--extract-only, no root, no install) and copies the
# handful of libraries EGL needs into $ROBOT_OOD_NVGL/lib plus the vendor ICD into
# $ROBOT_OOD_NVGL/egl_vendor.d. env.sh picks the directory up by driver version on the next
# `source env.sh`; robot_ood.eval.egl_devices maps each cell's CUDA device to the EGL index.
#
# Usage: scripts/setup_nvidia_gl.sh [target_dir]   (default: <repo>/../nvidia-gl-<driver>)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
[ -n "$VER" ] || { echo "nvidia-smi reports no driver version" >&2; exit 1; }
DEST="${1:-$(dirname "$REPO")/nvidia-gl-$VER}"
RUN="NVIDIA-Linux-x86_64-$VER.run"
URL="https://us.download.nvidia.com/XFree86/Linux-x86_64/$VER/$RUN"

mkdir -p "$DEST/lib" "$DEST/egl_vendor.d"
cd "$DEST"
[ -f "$RUN" ] || curl -fL --retry 3 -o "$RUN" "$URL"
[ -d extracted ] || sh "$RUN" --extract-only --target extracted

# The EGL vendor and what it dlopens, plus the driver's own glvnd front ends so the versions
# on LD_LIBRARY_PATH agree with each other rather than with conda's mesa build.
for lib in libEGL_nvidia libGLESv2_nvidia libGLESv1_CM_nvidia libGLX_nvidia \
           libnvidia-eglcore libnvidia-glcore libnvidia-glsi libnvidia-tls \
           libnvidia-glvkspirv libnvidia-gpucomp; do
  cp -f "extracted/$lib.so.$VER" lib/
done
ln -sf "libEGL_nvidia.so.$VER"    lib/libEGL_nvidia.so.0
ln -sf "libGLESv2_nvidia.so.$VER" lib/libGLESv2_nvidia.so.2
ln -sf "libGLX_nvidia.so.$VER"    lib/libGLX_nvidia.so.0
cp -f extracted/libEGL.so.1.1.0 lib/ && ln -sf libEGL.so.1.1.0 lib/libEGL.so.1
cp -f extracted/libGL.so.1.7.0  lib/ && ln -sf libGL.so.1.7.0  lib/libGL.so.1
cp -f extracted/libGLdispatch.so.0 extracted/libOpenGL.so.0 lib/
cp -f extracted/10_nvidia.json egl_vendor.d/

echo "NVIDIA GL $VER ready in $DEST"
echo "  source $REPO/env.sh   # picks it up when ROBOT_OOD_NVGL is unset"
