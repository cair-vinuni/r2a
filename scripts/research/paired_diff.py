#!/usr/bin/env python
"""Paired McNemar between two results roots, matched per axis.

Usage: python scripts/research/paired_diff.py RESULTS_ROOT AGAINST_ROOT [--axis AXIS]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcnemar  # noqa: E402


def pair(baseline_dir: Path, cell_dir: Path, label: str) -> str:
    baseline = mcnemar.load(str(baseline_dir))
    measured = mcnemar.load(str(cell_dir))
    keys = sorted(set(baseline) & set(measured))
    if not keys:
        return f"McNemar vs {label}: no shared episodes"
    base_vec = np.array([baseline[k] for k in keys])
    cell_vec = np.array([measured[k] for k in keys])
    b_only = int((base_vec & (~cell_vec)).sum())
    c_only = int(((~base_vec) & cell_vec).sum())
    p = mcnemar.mcnemar_exact(b_only, c_only)
    return (f"McNemar vs {label}: b/c={b_only}/{c_only} p={p:.2e} n={len(keys)} "
            f"({sum(base_vec)} -> {sum(cell_vec)})")


def baseline_for(against_root: Path, family: str, axis: str, tag: str) -> Path | None:
    axis_dir = against_root / family / axis
    exact = axis_dir / tag
    if exact.is_dir():
        return exact
    tags = sorted(d for d in axis_dir.glob("*") if d.is_dir())
    return tags[0] if len(tags) == 1 else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("against_root", type=Path)
    parser.add_argument("--axis", default=None, help="only this perturbation axis")
    args = parser.parse_args(argv)

    printed = False
    for cell_dir in sorted(args.results_root.glob("*/*/*")):
        if not cell_dir.is_dir():
            continue
        family, axis, tag = cell_dir.parts[-3], cell_dir.parts[-2], cell_dir.parts[-1]
        if args.axis and axis != args.axis:
            continue
        baseline = baseline_for(args.against_root, family, axis, tag)
        if baseline is None:
            continue
        line = pair(baseline, cell_dir, f"{args.against_root.name}/{baseline.name}")
        print(f"{axis}/{tag}: {line}")
        printed = True
    if not printed:
        print("nothing to pair", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
