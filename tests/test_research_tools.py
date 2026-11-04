"""scripts/research/ -- the paired diff, exercised on fabricated result trees."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO_ROOT / "scripts" / "research" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


paired_diff = _load("paired_diff")


def _write_cell(root: Path, tag: str, outcomes: list[bool]) -> Path:
    d = root / "plus_libero_object" / "Objects_Layout" / tag
    d.mkdir(parents=True)
    for i, ok in enumerate(outcomes):
        suffix = "success" if ok else "failure"
        (d / f"rollout_t_episode{i}_{suffix}.mp4").write_bytes(b"x")
    return d


def test_paired_diff_pairs_the_same_tag_across_roots(tmp_path, capsys):
    _write_cell(tmp_path / "new", "t_base_object_Objects_Layout", [True, True, True, False])
    _write_cell(tmp_path / "old", "t_base_object_Objects_Layout", [True, False, False, False])
    rc = paired_diff.main([str(tmp_path / "new"), str(tmp_path / "old")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "b/c=0/2" in out and "(1 -> 3)" in out


def test_paired_diff_exits_nonzero_with_nothing_to_pair(tmp_path, capsys):
    _write_cell(tmp_path / "new", "t_base_object_Objects_Layout", [True])
    (tmp_path / "old").mkdir()
    assert paired_diff.main([str(tmp_path / "new"), str(tmp_path / "old")]) == 1

