"""Training defaults resolved from configs/robot_ood.yaml.

One resolution rule, applied everywhere: CLI flag > environment variable > this file's
defaults.train (overridden per suite by suites.<name>.train) > the code default the caller
passes. The consumers are robot_ood.cli.train (epochs) and robot_ood.train.tasks.locator
(the staged-schedule learning rates).
"""

from __future__ import annotations

import os
from typing import Any

from robot_ood.core.paths import project_config, suite_config

__all__ = ["train_param"]


def train_param(task: str, key: str, default: Any, *, suite: str = "",
                env: str = "") -> Any:
    """One training knob, resolved through the file with the environment on top.

    Args:
        task: The train verb the knob belongs to ("locator" | "entry" | "fcsr").
        key: The knob's name inside that task's block ("epochs", "gnn_lr", ...).
        default: The code default when neither the environment nor the file carries the knob.
        suite: Suite whose suites.<name>.train block outranks defaults.train; "" skips that layer.
        env: Environment variable that outranks the file; "" means the knob has none.

    Returns:
        The resolved value, cast to default's type when default is an int or float.
    """
    raw = os.environ.get(env) if env else None
    if raw is None:
        for block in (suite_config(suite).get("train") if suite else None,
                      (project_config().get("defaults") or {}).get("train")):
            entry = (block or {}).get(task)
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
