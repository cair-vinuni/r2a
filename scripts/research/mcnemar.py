"""scripts/research/mcnemar.py — paired significance for either deterministic LIBERO bench.

The benchmark is deterministic (same init -> same rollout), so two methods are PAIRED per episode.
For each comparison we build the 2x2 discordant table (b = A-only successes, c = B-only successes),
report the exact two-sided McNemar p-value (binomial on the discordant pairs), the paired accuracy
delta, and a 95% bootstrap CI on that delta over episodes.

Usage: python scripts/research/mcnemar.py --suite object --tags tag_a tag_b [--vs tag_a]
       add --prefix <p> to read another benchmark's result tree instead of LIBERO-Plus'.
"""

import glob
import math
import os
import re

import numpy as np


def results_root() -> str:
    if os.environ.get("ROBOT_OOD_RESULTS"):
        return os.environ["ROBOT_OOD_RESULTS"]
    root = os.environ.get("ROBOT_OOD_ROOT")
    if not root:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "results")


ROOT = results_root()
rng = np.random.RandomState(0)


def key(fn):
    b = os.path.basename(fn)
    b = re.sub(r"^rollout_", "", b)
    b = re.sub(r"_(success|failure)\.mp4$", "", b)
    return b


def load(dirpath):
    """episode_key -> success(bool) from *_success/_failure.mp4."""
    out = {}
    for f in glob.glob(f"{dirpath}/**/*.mp4", recursive=True):
        out[key(f)] = f.endswith("_success.mp4")
    return out


def mcnemar_exact(b, c):
    """two-sided exact McNemar (binomial, p=0.5) on n=b+c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def compare(nameA, dirA, nameB, dirB):
    A, B = load(dirA), load(dirB)
    keys = sorted(set(A) & set(B))
    if not keys:
        print(f"  [!] no shared episodes for {nameA} vs {nameB}")
        return
    a = np.array([A[k] for k in keys]); bb = np.array([B[k] for k in keys])
    b01 = int(((~a) & bb).sum())          # B-only success (B better)
    b10 = int((a & (~bb)).sum())          # A-only success (A better)
    p = mcnemar_exact(b10, b01)
    accA, accB = a.mean() * 100, bb.mean() * 100
    delta = accB - accA
    d = bb.astype(float) - a.astype(float)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(4000)]) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
    print(f"  {nameA:>10}={accA:5.1f}  {nameB:<10}={accB:5.1f}  Δ={delta:+5.1f}  "
          f"[{lo:+.1f},{hi:+.1f}]  disc(B+/A+)={b01}/{b10}  McNemar p={p:.2e} {sig}  n={len(keys)}")


def arm_battery(suite: str, axis: str, tags: list[str], vs: str | None,
                prefix: str = "plus") -> None:
    """Pairwise (or all-vs-vs) comparison of arbitrary result tags on one suite x axis cell.

    One load/compare machinery pointed at tags named on the command line, so a new arm
    never needs its own scoring script. Also prints the ORACLE union over the tags -- the share of
    episodes SOME arm solves, i.e. the headroom a per-episode selector could in principle reach.
    """
    R = f"{ROOT}/{prefix}_libero_{suite}/{axis}"
    print(f"=== {suite} x {axis} ===")
    loaded = {t: load(f"{R}/{t}") for t in tags}
    for t in tags:
        d = loaded[t]
        s = sum(d.values())
        print(f"  {t:<58} {s:>4}/{len(d):<4} = {100*s/len(d) if d else float('nan'):5.1f}%")
    print()
    pairs = [(vs, t) for t in tags if t != vs] if vs else [
        (a, b) for i, a in enumerate(tags) for b in tags[i + 1:]]
    for a, b in pairs:
        compare(a.split("_libero_")[0][:10], f"{R}/{a}", b.split("_libero_")[0][:10], f"{R}/{b}")
    common = set.intersection(*[set(d) for d in loaded.values()]) if loaded else set()
    if common and len(tags) > 1:
        u = sum(1 for k in common if any(loaded[t][k] for t in tags))
        print(f"\n  ORACLE union over {len(tags)} arms: {u}/{len(common)} = {100*u/len(common):5.1f}%"
              f"   (headroom a per-episode selector could reach)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="paired McNemar over LIBERO result tags")
    ap.add_argument("--suite", required=True, help="suite token as it appears in <prefix>_libero_<suite> (e.g. 10, object)")
    ap.add_argument("--axis", default="Robot_Initial_States")
    ap.add_argument("--tags", nargs="+", required=True, help="result dir names under that suite x axis")
    ap.add_argument("--vs", default=None, help="compare every tag against this one instead of pairwise")
    ap.add_argument("--prefix", default="plus", help="the benchmark's results namespace: plus | pro")
    args = ap.parse_args()
    return arm_battery(args.suite, args.axis, args.tags, args.vs, args.prefix)


if __name__ == "__main__":
    main()
