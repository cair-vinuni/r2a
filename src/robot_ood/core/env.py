"""The one place the process environment is read, typed.

Every ROBOT_OOD_*, GR_*, SN_* and EN_* variable this package understands is parsed through these four
functions, so a variable's name appears exactly once outside the docs.
"""

from __future__ import annotations

import os

__all__ = ["env_bool", "env_float", "env_int", "env_str"]

def env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def env_bool(key: str, default: bool) -> bool:
    """Parse a bool from environ: "1" is true, anything else false."""
    return os.environ.get(key, "1" if default else "0") == "1"
