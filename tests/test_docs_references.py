"""Documentation must point at things that exist.

Two failure modes this repo has actually hit, both of which survive a fully green test run:

  * a module path in a docstring or a design doc that no longer resolves (robot_ood.sentinel.track
    was cited in two drivers for months; it has never existed), and
  * a file.py:123 line-number pin, which goes stale the moment anything above that line moves.

Both are cheap to check and impossible to notice by reading.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

CODE_DIRS = ("src", "tests", "scripts", "evals")
DOC_FILES = sorted((REPO_ROOT / "docs").rglob("*.md")) + [REPO_ROOT / "README.md"]
# UPSTREAM.md is excluded from the path check on purpose: it cites paths inside the UPSTREAM
# repositories (e.g. src/model/StableVLA), which are not expected to exist here.


def _files():
    for d in CODE_DIRS:
        for p in sorted((REPO_ROOT / d).rglob("*.py")):
            if "__pycache__" not in p.parts and p.name != Path(__file__).name:
                yield p
    for p in DOC_FILES:
        if p.exists():
            yield p


def _is_module(dotted: str) -> bool:
    base = SRC / Path(*dotted.split("."))
    return base.with_suffix(".py").exists() or (base / "__init__.py").exists()


def _resolves(dotted: str) -> bool:
    """True when dotted names a live module, or a live module plus an attribute path.

    Walking back to *any* resolving prefix is not enough, and that hole let 21 stale paths through:
    after the layer split, robot_ood.data.schema.HEAD_NPZ still "resolved" because robot_ood.data exists.

    The rule that catches it: find the longest prefix that IS a module. If that prefix is a PACKAGE
    (a directory) and the next segment is lowercase, the next segment was meant to be a module in
    that package -- and it is not one, so the path is stale. If the prefix is a plain module (a .py),
    whatever follows is an attribute and is out of scope here.
    """
    parts = dotted.split(".")
    for k in range(len(parts), 1, -1):
        prefix = ".".join(parts[:k])
        if not _is_module(prefix):
            continue
        if k == len(parts):
            return True                                    # the whole path is a module
        is_package = (SRC / Path(*prefix.split("."))).is_dir()
        next_seg = parts[k]
        return not (is_package and next_seg.islower())     # lowercase inside a package => stale module
    return False


def test_every_robot_ood_module_path_mentioned_anywhere_resolves():
    bad = []
    for path in _files():
        for i, line in enumerate(path.read_text().split("\n"), 1):
            for m in re.finditer(r"\brobot_ood(?:\.[a-z_][a-z0-9_]*)+", line):
                if m.group(0) == "robot_ood.yaml":     # the config file, not a module path
                    continue
                if not _resolves(m.group(0)):
                    bad.append(f"{path.relative_to(REPO_ROOT)}:{i}: {m.group(0)}")
    assert not bad, (
        "these robot_ood module paths do not resolve under src/ -- a rename left them behind, or they "
        "never existed:\n  " + "\n  ".join(bad)
    )


def test_docs_do_not_pin_line_numbers():
    bad = []
    for path in DOC_FILES:
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text().split("\n"), 1):
            for m in re.finditer(r"\b[\w/]+\.py:\d+", line):
                bad.append(f"{path.relative_to(REPO_ROOT)}:{i}: {m.group(0)}")
    assert not bad, (
        "docs pin source line numbers, which go stale on the next edit above them -- name the "
        "enclosing function or class instead:\n  " + "\n  ".join(bad)
    )


# Paths inside OTHER repositories, named in setup commands that run against a foreign checkout. The
# check's premise -- "a src/... path is a path in this repo" -- does not hold for these.
FOREIGN_PATHS = frozenset({
    # the StableVLA subtree of DAGroup-PKU/HumanNet, cloned per docs/operations.md
    "src/model/StableVLA",
})


def test_documented_paths_exist():
    """Every src/.../scripts/.../configs/... path named in the docs must be on disk."""
    bad = []
    # (?<![\w/.]) so a path inside a longer absolute path (…/envs/stablevla/bin/python) is not matched
    pattern = re.compile(r"(?<![\w/.])((?:src|scripts|configs|evals|tests|docs)/[\w./-]+?)(?=[\s`,)\]]|$)")
    for path in DOC_FILES:
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text().split("\n"), 1):
            for m in pattern.finditer(line):
                target = m.group(1).rstrip(".")
                if target not in FOREIGN_PATHS and not (REPO_ROOT / target).exists():
                    bad.append(f"{path.relative_to(REPO_ROOT)}:{i}: {target}")
    assert not bad, "docs name paths that do not exist:\n  " + "\n  ".join(bad)
