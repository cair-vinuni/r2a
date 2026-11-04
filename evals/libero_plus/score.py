"""evals/libero_plus/score.py — the result-dir scorer, for either benchmark.

Each client appends one record per episode to episodes.jsonl in its shard directory; the success
rate for a result dir is #success / #records (recursive over shard subdirs). A result dir from
before those records were written is counted by its *_success.mp4 / *.mp4 replays instead. Which
axes exist and where their result dirs live come from robot_ood.eval.bench, so --bench is the only
difference between scoring one benchmark and another.

Report: per-suite axis x method grid + TOTAL + GRAND TOTAL, for an arbitrary set of method tags.

--methods takes name:tag_prefix pairs, so it can score both the tags this repo's drivers write
(stablevla_<mode>_<suite>_<axis>, gr00t_...) and any pre-existing baseline family on disk.

Usage:
    python evals/libero_plus/score.py --methods base:stablevla_base entry:stablevla_entry
    python evals/libero_plus/score.py --suites spatial goal long --methods base:lp
    python evals/libero_plus/score.py --bench robotwin_plus --suites g5 --methods base:pi05_robotwin_base
    ROBOT_OOD_RESULTS=/abs/results python evals/libero_plus/score.py ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from robot_ood.eval.bench import DEFAULT_BENCH, load_bench


def results_root() -> str:
    """Resolve the results tree: $ROBOT_OOD_RESULTS, else $ROBOT_OOD_ROOT/results, else ./results."""
    if os.environ.get("ROBOT_OOD_RESULTS"):
        return os.environ["ROBOT_OOD_RESULTS"]
    root = os.environ.get("ROBOT_OOD_ROOT")
    if not root:
        # score.py lives at <root>/evals/libero_plus/score.py
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "results")


def count_dir(d: str):
    """(#success, #total) under result dir d, recursively.

    The clients' episodes.jsonl records are the count; a directory from before those were
    written (or run with videos on and records lost) is counted by its *_success.mp4 / *.mp4.
    """
    succ = tot = 0
    for path in glob.glob(f"{d}/**/episodes.jsonl", recursive=True):
        with open(path, errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                tot += 1
                succ += int(bool(rec.get("success")))
    if tot:
        return succ, tot
    succ = len(glob.glob(f"{d}/**/*_success.mp4", recursive=True))
    tot = len(glob.glob(f"{d}/**/*.mp4", recursive=True))
    return succ, tot


def dir_digest(d: str) -> str:
    """The artifact_digest run.json recorded for one result dir, "" when unrecorded."""
    try:
        with open(os.path.join(d, "run.json")) as f:
            return str(json.load(f).get("artifact_digest") or "")
    except (OSError, ValueError):
        return ""


def cell(root: str, suite: str, axis: str, tag: str, bench, digests=None):
    """Score one (suite, axis, method-tag) cell. tag is the result-dir name (a prefix match if
    the exact dir is absent, mirroring score_full_table.sh's ${tag}_* glob). digests, when given,
    collects each counted dir's recorded artifact_digest."""
    base = str(bench.axis_dir(Path(root), suite, axis))
    exact = os.path.join(base, tag)
    dirs = [exact] if os.path.isdir(exact) else [d for d in glob.glob(f"{base}/{tag}*")
                                                if os.path.isdir(d)]
    succ = tot = 0
    for d in dirs:
        s, t = count_dir(d)
        succ += s
        tot += t
        if digests is not None and t:
            digests.setdefault(dir_digest(d) or "unrecorded", []).append(os.path.basename(d))
    return succ, tot


def fmt(succ: int, tot: int) -> str:
    return f"{succ}/{tot}={100 * succ / tot:.1f}" if tot else "--"


def report_table(root, suites, methods, bench):
    """methods: list of (name, tag_prefix). Per-suite axis grid + TOTAL, then GRAND TOTAL."""
    grand = {name: [0, 0] for name, _ in methods}
    digests = {name: {} for name, _ in methods}
    for suite in suites:
        print(f"================= {suite} =================")
        hdr = f"{'axis':<22}" + "".join(f"{n:>16}" for n, _ in methods)
        print(hdr)
        subtot = {name: [0, 0] for name, _ in methods}
        for axis in bench.axes.values():
            line = f"{axis.label:<22}"
            for name, tag in methods:
                s, t = cell(root, suite, axis.name, tag, bench, digests[name])
                line += f"{fmt(s, t):>16}"
                subtot[name][0] += s
                subtot[name][1] += t
            print(line)
        line = f"{'TOTAL':<22}"
        for name, _ in methods:
            s, t = subtot[name]
            line += f"{fmt(s, t):>16}"
            grand[name][0] += s
            grand[name][1] += t
        print(line)
        print()
    print("================= GRAND TOTAL =================")
    parts = []
    for name, _ in methods:
        s, t = grand[name]
        parts.append(f"{name} {s}/{t}={100 * s / t:.1f}%" if t else f"{name} --")
    print("   ".join(parts))
    # pairwise deltas vs the first method
    if len(methods) > 1 and grand[methods[0][0]][1]:
        b_s, b_t = grand[methods[0][0]]
        base_rate = 100 * b_s / b_t
        for name, _ in methods[1:]:
            s, t = grand[name]
            if t:
                print(f"  Δ({name} - {methods[0][0]}) = {100 * s / t - base_rate:+.1f}pp")
    for name, _ in methods:
        seen = {k: v for k, v in digests[name].items() if k != "unrecorded"}
        if not seen:
            continue
        note = " !! MIXED TREES" if len(seen) > 1 else ""
        print(f"  {name}: artifact_digest " + ", ".join(f"{k[:12]} x{len(v)}"
                                                        for k in sorted(seen) for v in [seen[k]])
              + note)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+", default=["spatial", "goal", "long"])
    ap.add_argument("--methods", nargs="+", required=True,
                    help="name:tag_prefix pairs, e.g. base:stablevla_base entry:stablevla_entry")
    ap.add_argument("--bench", default=DEFAULT_BENCH, help="which benchmark's results to score")
    args = ap.parse_args()
    root = results_root()

    methods = []
    for m in args.methods:
        name, _, tag = m.partition(":")
        methods.append((name, tag or name))
    report_table(root, args.suites, methods, load_bench(args.bench))


if __name__ == "__main__":
    main()
