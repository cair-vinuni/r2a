"""How this CLI puts a run on a terminal: tables, numbers, metrics and echoed config."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["echo_config", "fmt_metric", "fmt_num", "json_default", "print_table",
           "read_json"]


def json_default(obj: Any) -> Any:
    """json.dumps fallback: numpy scalars/arrays and Path -> plain JSON, anything else -> str."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return os.fspath(obj)
    return str(obj)


def echo_config(label: str, config: dict) -> None:
    """Print one fully-resolved config block to stdout."""
    print(f"===== {label}: fully-resolved config =====")
    print(json.dumps(config, indent=2, default=json_default, sort_keys=True))


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a minimal aligned text table."""
    if not rows:
        print("  (no rows)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def fmt_num(x: Any, nd: int = 4) -> str:
    """One scalar -> a fixed-precision display string; non-finite floats print verbatim."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        if not np.isfinite(x):
            return str(x)
        return f"{x:.{nd}f}"
    return str(x)


def fmt_metric(v: Any) -> str:
    """Format a metrics.json value for display."""
    if isinstance(v, dict) and {"lo", "point", "hi"} <= set(v.keys()):
        return f"{fmt_num(v['point'])} [{fmt_num(v['lo'])}, {fmt_num(v['hi'])}]"
    if isinstance(v, dict):
        return "{...}"     # nested, non-CI dict -- not flattened at this level
    if isinstance(v, list):
        return f"[{len(v)} items]" if len(v) > 6 else str(v)
    return fmt_num(v)


def read_json(path: Path) -> dict:
    """Read a JSON object, or {} on any error."""
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
