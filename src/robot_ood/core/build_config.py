"""Build defaults resolved from configs/robot_ood.yaml, the sibling of robot_ood.core.train_config.

One resolution rule, applied everywhere: CLI flag > environment variable > this file's
defaults.build (overridden per suite by suites.<name>.build) > the code default the caller passes.
A corpus recorded at 20 Hz and one recorded at 3 Hz do not want the same node duration or the same
segment horizon, so the knobs that set them belong beside the suite that names the corpus rather
than in one global constant.
"""

from __future__ import annotations

import os
from typing import Any

from robot_ood.core.paths import project_config, suite_config

__all__ = ["build_param"]


def build_param(step: str, key: str, default: Any, *, suite: str = "", env: str = "") -> Any:
    """One build knob, resolved through the file with the environment on top.

    Args:
        step: The build stage the knob belongs to ("corpus" | "graph" | "segments").
        key: The knob's name inside that stage's block ("coarsen", "segment_len", ...).
        default: The code default when neither the environment nor the file carries the knob.
        suite: Suite whose suites.<name>.build block outranks defaults.build; "" skips that layer.
        env: Environment variable that outranks the file; "" means the knob has none.

    Returns:
        The resolved value, cast to default's type when default is a bool, int or float.
    """
    raw = os.environ.get(env) if env else None
    if raw is None:
        for block in (suite_config(suite).get("build") if suite else None,
                      (project_config().get("defaults") or {}).get("build")):
            entry = (block or {}).get(step)
            if isinstance(entry, dict) and key in entry:
                raw = entry[key]
                break
    if raw is None:
        return default
    if isinstance(default, bool):
        return bool(raw)
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw
