"""The child environment, and the policy server as a context manager."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


from robot_ood.eval.egl_devices import NVGL_LIB_ENV
from robot_ood.eval.ladder import Cell, LadderConfig, MODES, REPO_ROOT
from robot_ood.eval.policy import (Policy as PolicyDescriptor,
                                   resolve_policy_python, resolve_trunk_path)
from robot_ood.eval.scheduler import ProcessSupervisor
from robot_ood.pipeline.interpreter import nvml_preload_path, resolve_conda_env_prefix

BASE_CHILD_ENV = {
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
}

ROBOT_OOD_SRC = REPO_ROOT / "src"


class EnvBuilder:
    """Builds the environment every child process of a cell inherits."""

    def __init__(self, config: LadderConfig) -> None:
        self._config = config
        self._bench_env = config.bench.client_env()
        self._conda_prefix = resolve_conda_env_prefix(config.bench.conda_env)
        self._nvml_preload = nvml_preload_path()
        self._client_pythonpath = self._bench_env["PYTHONPATH"]

    def server_pythonpath(self, policy: PolicyDescriptor) -> str:
        return self._pythonpath([*(resolve_trunk_path(p) for p in policy.pythonpath), ROBOT_OOD_SRC])

    @property
    def client_pythonpath(self) -> str:
        return self._client_pythonpath

    def for_cell(self, cell: Cell) -> dict[str, str]:
        """The full env recipe for one cell, recorded verbatim in the ledger and run.json."""
        env: dict[str, str] = {
            **BASE_CHILD_ENV,
            "LOAD_VISION": "0",
            "QNDF_DIR": str(self._config.suite(cell.suite).fwm_dir),
            **{k: v for k, v in self._bench_env.items() if k != "PYTHONPATH"},
            # osmesa default; a caller-exported MUJOCO_GL=egl must win -- the wrong
            # renderer fails at the first env.reset(), it does not degrade.
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "osmesa"),
        }
        for name in ("EGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID", "__EGL_VENDOR_LIBRARY_DIRS"):
            if name in os.environ:                    # only meaningful under the egl renderer
                env[name] = os.environ[name]
        if self._conda_prefix is not None:
            env["MAGICK_HOME"] = str(self._conda_prefix)
            env["LD_LIBRARY_PATH"] = self._prepend(
                str(self._conda_prefix / "lib"), os.environ.get("LD_LIBRARY_PATH", "")
            )
        if os.environ.get(NVGL_LIB_ENV):
            # env.sh found the NVIDIA EGL driver: its glvnd and vendor libraries must beat the
            # conda env's mesa copies of libEGL.so.1 that were just put first.
            env["LD_LIBRARY_PATH"] = self._prepend(
                os.environ[NVGL_LIB_ENV], env.get("LD_LIBRARY_PATH", "")
            )
        if self._nvml_preload is not None:
            # torch's allocator hard-asserts on a failing nvmlInit_v2 when growing its
            # pool; every launched process inherits the preload fix from here.
            env["LD_PRELOAD"] = self._prepend(self._nvml_preload, os.environ.get("LD_PRELOAD", ""))
        if cell.filter_file is not None:
            env["LIBERO_INSTANCE_ID_FILTER"] = str(cell.filter_file)

        if cell.mode not in MODES:
            raise ValueError(f"unknown MODE={cell.mode!r}; expected one of {list(MODES)}")
        env["MODE"] = cell.mode
        for name in ("GR_GRAPH_DIR", "ROBOT_OOD_CONSTANTS", "ROBOT_OOD_ABLATE", "ROBOT_OOD_TRUST_MIN",
                     "ROBOT_OOD_OUTPUTS", "ROBOT_OOD_LANG_POSTERIOR", "ROBOT_OOD_TRUST_PHASE_MAX",
                     "ROBOT_OOD_REF", "ROBOT_OOD_KEY_FUTURE", "GR_SEG_TIER1_M", "ROBOT_OOD_POS",
                     "ROBOT_OOD_HARDNEG_STATE", "ROBOT_OOD_NEG_POSE_BAND", "ROBOT_OOD_POLICY_DELTAS",
                     "ROBOT_OOD_PARTNER", "ROBOT_OOD_EXEC_HORIZON", "ROBOT_OOD_PAIR", "ROBOT_OOD_REQUERY",
                     "ROBOT_OOD_STALL_PUSH", "ROBOT_OOD_STALL_EPS", "ROBOT_OOD_STALL_GAIN",
                     "ROBOT_OOD_HORIZON_BY_MOTION", "ROBOT_OOD_PUSH_LEAN", "ROBOT_OOD_STALL_DWELL",
                     "ROBOT_OOD_STALL_PHASE_MAX", "ROBOT_OOD_STALL_RETRY", "ROBOT_OOD_STALL_RETRY_BACK",
                     "ROBOT_OOD_CONTACT_REQUERY", "ROBOT_OOD_GRASP_SERVO", "ROBOT_OOD_SERVO_HOLD"):
            if name in os.environ:
                env[name] = os.environ[name]
        env.update(cell.extra_env)   # explicit per-cell overrides (a yaml cell's env: block) win last
        return self._absolutize(env)

    @staticmethod
    def _absolutize(env: dict[str, str]) -> dict[str, str]:
        """Resolve the two artifact paths against the repo root when they are relative."""
        for name in ("GR_GRAPH_DIR", "ROBOT_OOD_CONSTANTS"):
            value = env.get(name)
            if value and not os.path.isabs(value):
                env[name] = str(REPO_ROOT / value)
        return env

    @staticmethod
    def _prepend(value: str, existing: str) -> str:
        return f"{value}:{existing}" if existing else value

    @staticmethod
    def _pythonpath(dirs: Sequence[Path]) -> str:
        parts = [str(d) for d in dirs]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            parts.append(existing)
        return os.pathsep.join(parts)


class PolicyServer:
    """The GPU-resident StableVLA policy server for one cell, as a context manager."""

    def __init__(self, config: LadderConfig, supervisor: ProcessSupervisor,
                 checkpoint: Path, port: int, log_dir: Path, env: Mapping[str, str],
                 policy: "PolicyDescriptor", log_name: str = "server.log") -> None:
        self._config = config
        self._supervisor = supervisor
        self._checkpoint = checkpoint
        self._policy = policy
        self._python = resolve_policy_python(self._policy)
        if self._python is None:
            raise RuntimeError(f"cannot locate an interpreter for policy {self._policy.id!r} "
                               f"(python: {self._policy.python}, conda_env: {self._policy.conda_env})")
        self._port = port
        self._log_path = log_dir / log_name
        self._env = dict(env)
        self._proc: subprocess.Popen | None = None
        self._log: TextIO | None = None

    def __enter__(self) -> "PolicyServer":
        return self.start().wait_ready()

    def start(self) -> "PolicyServer":
        """Spawn the server without waiting; several servers then load their model at once."""
        self._log = open(self._log_path, "w")
        argv = self._policy.server_argv(checkpoint=str(self._checkpoint), port=self._port)
        argv[0] = str(REPO_ROOT / argv[0])
        self._proc = self._supervisor.spawn(
            [str(self._python), *argv], env=self._env, stdout=self._log,
        )
        return self

    def wait_ready(self) -> "PolicyServer":
        """Block until the port answers, tearing the server down if it dies or times out first."""
        try:
            self._wait_ready(self._config.defaults.server_ready_timeout_s)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._proc is not None:
            self._supervisor.forget(self._proc)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=15)
            self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def _wait_ready(self, timeout_s: float, poll_every: float = 2.0) -> None:
        assert self._proc is not None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"policy server exited before ready (rc={self._proc.returncode}); "
                    f"log tail:\n{self._log_tail()}"
                )
            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=1):
                    return
            except OSError:
                time.sleep(poll_every)
        raise RuntimeError(f"policy server not ready after {timeout_s:.0f}s on port {self._port}")

    def _log_tail(self, lines: int = 40) -> str:
        if not self._log_path.exists():
            return ""
        return "\n".join(self._log_path.read_text(errors="replace").splitlines()[-lines:])


@dataclass(frozen=True)
class LaunchResult:
    wall_s: float
    client_returncodes: tuple[int, ...]

    @property
    def crashed(self) -> bool:
        return any(rc != 0 for rc in self.client_returncodes)

    @property
    def crash_note(self) -> str:
        """How the RESULT line reports a crash, or "" when every client exited 0."""
        bad = [rc for rc in self.client_returncodes if rc != 0]
        if not bad:
            return ""
        return (f"  [CRASHED: {len(bad)}/{len(self.client_returncodes)} clients exited "
                f"{sorted(set(bad))} -- this cell did not finish; the episode count below is "
                "whatever it had written by then]")
