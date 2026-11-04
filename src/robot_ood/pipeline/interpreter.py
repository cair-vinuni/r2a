"""Which python a spawned step runs under, and what its dynamic linker needs before it starts.

Both the eval driver and the policy-action cache need this, and neither can set it from inside the
child: LD_PRELOAD and LD_LIBRARY_PATH are read before the interpreter's first line.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["NVML_PRELOAD_CANDIDATE", "STABLEVLA_CONDA_ENV", "nvml_preload_path",
           "resolve_conda_env_prefix", "resolve_env_python"]

STABLEVLA_CONDA_ENV = "stablevla"          # fixed operational fact: the conda env name this box uses

# The userspace NVML that matches this box's kernel module, since torch's allocator
# hard-asserts on the mismatch. Why, and what a box without it gets: docs/SETUP.md.
NVML_PRELOAD_CANDIDATE = Path(
    "/srv/data/HAMLET-Isaac-GR00T/.vkenv/nvidia570/lib/libnvidia-ml.so.570.172.08"
)


def resolve_conda_env_prefix(name: str) -> Path | None:
    """Resolve a conda env's prefix without sourcing a shell.

    Args:
        name: The env name, or a path to an env prefix (any value holding a slash).

    Returns:
        The prefix holding bin/python, or None when no candidate has one.
    """
    if "/" in name:
        prefix = Path(name)
        if not prefix.is_absolute():
            prefix = Path(__file__).resolve().parents[3] / prefix
        return prefix if (prefix / "bin" / "python").exists() else None
    candidates: list[Path] = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(Path(conda_exe).resolve().parent.parent / "envs" / name)
    candidates += [
        base / "envs" / name
        for base in (Path.home() / "miniconda3", Path.home() / "anaconda3",
                     Path.home() / "miniforge3", Path("/opt/conda"), Path("/opt/miniconda3"))
    ]
    return next((c for c in candidates if (c / "bin" / "python").exists()), None)


def resolve_env_python(env_name: str) -> Path | None:
    """A named conda env's interpreter.

    Args:
        env_name: The environment name.

    Returns:
        Its path, or None when the env is not installed anywhere known.
    """
    prefix = resolve_conda_env_prefix(env_name)
    return (prefix / "bin" / "python") if prefix else None


def nvml_preload_path() -> str | None:
    """The NVML shim to LD_PRELOAD into every child, or None on a box that doesn't need it.

    Returns:
        The path as a string, or None. See NVML_PRELOAD_CANDIDATE for why.
    """
    override = os.environ.get("ROBOT_OOD_NVML_PRELOAD")
    if override:
        return override if Path(override).exists() else None
    return str(NVML_PRELOAD_CANDIDATE) if NVML_PRELOAD_CANDIDATE.exists() else None
