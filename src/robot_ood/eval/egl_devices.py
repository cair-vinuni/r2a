"""Which EGL device draws on which CUDA device, under the NVIDIA EGL driver.

NVIDIA's EGL enumerates GPUs in its own order, not CUDA's (on the four-A100 box EGL device 0 is
CUDA device 2). MuJoCo and robosuite pick the renderer by EGL index (MUJOCO_EGL_DEVICE_ID) while
the ladder pins a cell by CUDA index, so the cell needs the EGL index whose EGL_CUDA_DEVICE_NV
attribute is its CUDA index. Under mesa/llvmpipe there is one software device and no mapping;
the lookup reports None and the caller leaves MUJOCO_EGL_DEVICE_ID as env.sh set it.
"""

from __future__ import annotations

import ctypes
import functools
import os

__all__ = ["NVGL_LIB_ENV", "egl_cuda_map", "egl_device_for_cuda", "nvidia_egl_active"]

NVGL_LIB_ENV = "ROBOT_OOD_NVGL_LIB"
_EGL_CUDA_DEVICE_NV = 0x323A


def nvidia_egl_active() -> bool:
    """Whether env.sh put the NVIDIA EGL driver's libraries on the path."""
    return bool(os.environ.get(NVGL_LIB_ENV))


@functools.lru_cache(maxsize=1)
def egl_cuda_map() -> tuple[int, ...]:
    """EGL device index -> CUDA device index as the driver reports it, -1 for a non-CUDA device.

    Raises:
        RuntimeError: libEGL has no device-enumeration entry points, i.e. the NVIDIA vendor is not
            the one __EGL_VENDOR_LIBRARY_DIRS points at, or enumeration itself failed.
    """
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p

    def proc(name: str, restype: object, *argtypes: object):
        addr = egl.eglGetProcAddress(name.encode())
        if not addr:
            raise RuntimeError(
                f"EGL exposes no {name}; is the NVIDIA vendor on __EGL_VENDOR_LIBRARY_DIRS?"
            )
        return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

    query_devices = proc(
        "eglQueryDevicesEXT", ctypes.c_uint,
        ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
    )
    query_attrib = proc(
        "eglQueryDeviceAttribEXT", ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long),
    )
    devices = (ctypes.c_void_p * 64)()
    n = ctypes.c_int()
    if not query_devices(64, devices, ctypes.byref(n)):
        raise RuntimeError("eglQueryDevicesEXT failed")
    out = []
    for i in range(n.value):
        value = ctypes.c_long(-1)
        ok = query_attrib(devices[i], _EGL_CUDA_DEVICE_NV, ctypes.byref(value))
        out.append(int(value.value) if ok else -1)
    return tuple(out)


def egl_device_for_cuda(cuda_index: int) -> int | None:
    """The EGL device index that renders on the given CUDA device.

    Args:
        cuda_index: A physical CUDA device index, as the ladder's --gpus names them.

    Returns:
        The EGL index, or None when the NVIDIA EGL driver is not active and the question has no
        answer (mesa's single llvmpipe device).

    Raises:
        RuntimeError: The driver is active but no EGL device reports that CUDA index.
    """
    if not nvidia_egl_active():
        return None
    mapping = egl_cuda_map()
    for egl_index, cuda in enumerate(mapping):
        if cuda == cuda_index:
            return egl_index
    raise RuntimeError(
        f"no EGL device reports CUDA device {cuda_index}; the EGL->CUDA map is {mapping}"
    )
