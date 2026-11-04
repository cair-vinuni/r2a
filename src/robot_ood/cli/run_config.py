"""One run_config.json per train run: what ran, from where, under which constants.

A snapshot for humans and audits, never a config source -- code reads its constants from
robot_ood.core.constants / robot_ood.data.schema / robot_ood.core.schemas, and this file only
records what those said when the run started.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

__all__ = ["RUN_CONFIG_JSON", "dump_run_config", "update_run_config"]

RUN_CONFIG_JSON = "run_config.json"


def dump_run_config(graph_dir: str | os.PathLike, resolved_cfg: Mapping[str, Any],
                    *, suite: str = "", log: Any = print) -> Path:
    """Write graph_dir/run_config.json before epoch one, rotating the previous run's copy.

    Args:
        graph_dir: The artifact directory the run trains into.
        resolved_cfg: The final hyperparameter values the run uses, env and CLI already merged.
        suite: Suite short-name, for the data-root lookup.
        log: One confirmation line goes here.

    Returns:
        The written path.
    """
    root = Path(graph_dir)
    warm = root / "g_head.npz"
    found = warm.is_file()
    doc = {
        "meta": {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "command": " ".join([Path(sys.executable).name, *sys.argv]),
            "policy": _suite_policy(suite),
        },
        "hparams": dict(resolved_cfg),
        "paths": {
            "graph_dir": str(root.resolve()),
            "warm_start": warm.name,
            "warm_start_found": found,
            "warm_start_sha256": _sha256(warm) if found else None,
            "data_root": _data_root(suite),
        },
        "constants": _constants_snapshot(suite),
        "selection": None,
    }
    out = root / RUN_CONFIG_JSON
    if out.exists():
        out.replace(root / "run_config.prev.json")
    _atomic_write(out, doc)
    log(f"[run_config] wrote {out} (commit {doc['meta']['git_commit']}, warm_start={found})")
    return out


def update_run_config(graph_dir: str | os.PathLike, selection_info: Mapping[str, Any]) -> None:
    """Fill the selection block once the run knows its winning epoch.

    Args:
        graph_dir: The artifact directory dump_run_config wrote into.
        selection_info: best_epoch / metric / tie_break, whatever the run selected on.
    """
    out = Path(graph_dir) / RUN_CONFIG_JSON
    doc = json.loads(out.read_text())
    doc["selection"] = dict(selection_info)
    _atomic_write(out, doc)


def _constants_snapshot(suite: str = "") -> dict:
    """The frozen constants that decide artifact/model compatibility, graph knobs off the suite's contract.

    Args:
        suite: Suite short-name; "" resolves to the module defaults.
    """
    from robot_ood.data.contract import resolved_contract
    from robot_ood.core.constants import SCHEMA_VERSION
    from robot_ood.data import schema

    contract = resolved_contract(suite)
    return {
        "schema_version": SCHEMA_VERSION,
        "n_relations": len(contract.vocab.relations),
        "relations": list(contract.vocab.relations),
        "grip_open_thr": schema.GRIP_OPEN_THR,
        "kernel_bw": schema.KERNEL_BW,
        "k_sibling": schema.K_SIBLING,
        "k_align": schema.K_ALIGN,
        "nbins_align": schema.NBINS_ALIGN,
        "psi_freqs": schema.PSI_FREQS,
        "coarsen": contract.build.coarsen,
        "dilations": list(contract.vocab.dilations),
    }


def _suite_policy(suite: str) -> str:
    """The suite's configured policy id, else the default."""
    from robot_ood.core.paths import suite_config

    return str(suite_config(suite).get("policy") or "stablevla") if suite else "stablevla"


def _git_commit() -> str:
    """The short HEAD hash, or "unknown" outside a git checkout."""
    try:
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10)
    except OSError:
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _data_root(suite: str) -> str | None:
    """The absolute demo-hdf5 directory this suite resolves to, or None when nothing does."""
    if not suite:
        return None
    from robot_ood.core.paths import default_paths

    try:
        return str(default_paths().hdf5(suite).resolve())
    except FileNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(out: Path, doc: dict) -> None:
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, out)
