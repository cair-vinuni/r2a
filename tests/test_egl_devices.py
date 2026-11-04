"""robot_ood.eval.egl_devices: the EGL-index-for-CUDA-index lookup and its mesa fallback."""

from types import SimpleNamespace

import pytest

from robot_ood.eval import egl_devices
from robot_ood.eval.runner import EGL_RENDERER, CellRunner

# A runner whose benchmark renders through EGL -- the only case with an overlay to test.
_EGL_RUNNER = SimpleNamespace(_config=SimpleNamespace(bench=SimpleNamespace(renderer=EGL_RENDERER)))


def test_without_the_nvidia_driver_there_is_no_mapping(monkeypatch):
    monkeypatch.delenv(egl_devices.NVGL_LIB_ENV, raising=False)
    assert egl_devices.egl_device_for_cuda(0) is None
    assert CellRunner._renderer_env(_EGL_RUNNER, 3) == {}


def test_the_egl_index_is_the_one_reporting_the_cuda_device(monkeypatch):
    monkeypatch.setenv(egl_devices.NVGL_LIB_ENV, "/nvgl/lib")
    monkeypatch.setattr(egl_devices, "egl_cuda_map", lambda: (2, 3, 0, 1))
    assert egl_devices.egl_device_for_cuda(2) == 0
    assert egl_devices.egl_device_for_cuda(0) == 2
    with pytest.raises(RuntimeError, match="no EGL device reports CUDA device 7"):
        egl_devices.egl_device_for_cuda(7)


def test_the_client_sees_its_gpu_first_and_the_egl_id_inside_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv(egl_devices.NVGL_LIB_ENV, "/nvgl/lib")
    monkeypatch.setattr(egl_devices, "egl_cuda_map", lambda: (2, 3, 0, 1))
    env = CellRunner._renderer_env(_EGL_RUNNER, 2)
    assert env == {"MUJOCO_EGL_DEVICE_ID": "0", "CUDA_VISIBLE_DEVICES": "2,0"}
    assert env["MUJOCO_EGL_DEVICE_ID"] in env["CUDA_VISIBLE_DEVICES"]   # robosuite's assert
    monkeypatch.setattr(egl_devices, "egl_cuda_map", lambda: (0, 1, 2, 3))
    assert CellRunner._renderer_env(_EGL_RUNNER, 1) == {"MUJOCO_EGL_DEVICE_ID": "1", "CUDA_VISIBLE_DEVICES": "1"}


def test_clients_get_one_blas_thread_and_servers_two():
    assert CellRunner._thread_env("client") == {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    assert CellRunner._thread_env("server")["OMP_NUM_THREADS"] == "2"
    with pytest.raises(KeyError):
        CellRunner._thread_env("cache")
