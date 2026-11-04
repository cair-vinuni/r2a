"""Finding a prior run under outputs/, and inheriting its resolved config.

A run directory is matched on its recorded config, not its name, so --resume and the skip-if-done
check both answer "has this exact configuration already run" rather than "is there a directory".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from robot_ood.cli.errors import CLIError
from robot_ood.cli.render import read_json

__all__ = ["find_matching_run", "load_resume_config", "pick"]


def find_matching_run(root: Path, suite: str, stage: str, config: dict) -> Path | None:
    """The newest COMPLETE run matching the config, or None.

    Args:
        root: Run-bookkeeping root (outputs/).
        suite: Suite short-name.
        stage: StageLogger stage label (graph / train).
        config: The fully-resolved config to match against.

    Returns:
        The newest run under <root>/<suite>/<stage>_* with matching config.json.
    """
    base = root / suite
    if not base.is_dir():
        return None
    norm_target = {k: v for k, v in config.items() if k != "name"}
    for run_dir in sorted(base.glob(f"{stage}_*"), reverse=True):
        if not run_dir.is_dir() or run_dir.is_symlink():
            continue
        if read_json(run_dir / "manifest.json").get("status") != "complete":
            continue
        norm_got = {k: v for k, v in read_json(run_dir / "config.json").items() if k != "name"}
        if norm_got == norm_target:
            return run_dir
    return None


# ==== --resume: inherit a prior run's resolved config ====
def load_resume_config(run_dir: str | os.PathLike) -> dict:
    """Load a prior run's config.json for --resume layering.

    Raises:
        CLIError: run_dir has no config.json, or it is unreadable/empty.
    """
    p = Path(run_dir) / "config.json"
    if not p.exists():
        raise CLIError(f"--resume {run_dir}: no config.json found there (not a StageLogger run dir?)")
    config = read_json(p)
    if not config:
        raise CLIError(f"--resume {run_dir}: unreadable or empty config.json")
    return config


def pick(explicit: Any, resumed: dict, key: str, default: Any) -> Any:
    """Return explicit if set, else resumed[key], else default."""
    if explicit is not None:
        return explicit
    if key in resumed:
        return resumed[key]
    return default
