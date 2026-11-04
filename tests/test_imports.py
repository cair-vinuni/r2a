"""Import hygiene: which modules are allowed to cost you a torch import.

robot_ood.core.paths exists to answer "where are the artifacts"; robot_ood.data.corpus says in its own
docstring that it is numpy-only so it can measure a graph without the training stack. Both were
transitively importing torch, because robot_ood/graph/__init__.py eagerly imported the GNN and every
from robot_ood.data import schema ran it.

Each check runs in a SUBPROCESS: sys.modules is process-global and the rest of this suite has
already imported torch, so an in-process assertion would be meaningless.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Importing any of these must NOT pull torch in. Keep this list short and justified -- it is a
# promise about cost, and every entry is a module something imports for a small, cheap reason.
TORCH_FREE_MODULES = [
    "robot_ood",              # the package root: import robot_ood must stay cheap
    "robot_ood.core.paths",         # path resolution; imported by ~14 modules for one constant module
    "robot_ood.core.config",        # the typed knobs those modules read alongside the paths
    "robot_ood.core.kinematics",                 # simulator accessors, numpy only
    "robot_ood.core.npz",                # .npz mechanics; every artifact reader is on its import chain
    "robot_ood.data.schema",       # constants
    "robot_ood.data.binning",      # phase bucketing, numpy only
    "robot_ood.data.corpus",       # graph measurement; its docstring promises numpy-only
    "robot_ood.data.nodes",        # node table (torch only inside NodeTable.torch())
]


def _import_in_subprocess(module: str) -> set[str]:
    """Import module in a fresh interpreter; return the set of top-level packages it loaded."""
    code = (
        f"import sys; import {module}; "
        "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "4"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"import {module} failed:\n{proc.stderr}"
    return set(proc.stdout.split())


@pytest.mark.parametrize("module", TORCH_FREE_MODULES)
def test_module_does_not_import_torch(module):
    loaded = _import_in_subprocess(module)
    assert "torch" not in loaded, (
        f"`import {module}` now pulls in torch. Something on its import chain gained a top-level "
        "torch dependency -- most likely a package __init__.py started re-exporting a name instead "
        "of staying docstring-only. Import the submodule you need directly, or move the torch use "
        "behind a function-local import."
    )


def test_the_check_can_actually_fail():
    """Positive control: a module that legitimately needs torch must show up as needing it.

    Without this, a bug that made _import_in_subprocess return an empty set would make every
    assertion above pass vacuously.
    """
    assert "torch" in _import_in_subprocess("robot_ood.models.retrieval.gnn")


def test_every_python_file_in_the_repo_compiles():
    """Every .py under src/, tests/, scripts/, evals/ must at least parse.

    scripts/ and evals/ are entry points, not libraries -- nothing in this suite imports
    evals/gr00t/ws_server.py, so a syntax error there survives a fully green test run and only
    surfaces at launch. This is the cheap guard against that.
    """
    import ast

    broken = []
    for d in ("src", "tests", "scripts", "evals"):
        for path in sorted((REPO_ROOT / d).rglob("*.py")):
            if "__pycache__" in path.parts or "external" in path.parts:
                continue
            try:
                ast.parse(path.read_text())
            except SyntaxError as e:
                broken.append(f"{path.relative_to(REPO_ROOT)}:{e.lineno}: {e.msg}")
    assert not broken, "files that do not parse:\n  " + "\n  ".join(broken)
