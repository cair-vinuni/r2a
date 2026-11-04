"""Measure phase_hat - phase_gt over held-out demos, into a replay report.

Training reads QueryContext.phase off the ground-truth phase of the query's own true segment;
deploy reads it off GraphLocator's belief estimate. This replays the locator over held-out demos
at the deployed check cadence and compares the two at every check, on the same by-demo split
robot_ood.train.loop holds out. robot_ood.train.tasks.locator runs this replay over its just-saved
checkpoint and embeds the report in g_head.npz; robot_ood.train.augment reads it from there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from robot_ood.core.constants import HELDOUT_STRIDE
from robot_ood.data.nodes import NodeTable
from robot_ood.runtime.locator import GraphLocator
from robot_ood.train.sequences import RawWindowDataset

if TYPE_CHECKING:
    from robot_ood.core.config import GraphConfig

__all__ = ["PHASE_BINS", "DEPLOY_HIST", "DEPLOY_CHECK_EVERY",
           "ERR_TAIL_THRESHOLD", "Sample", "embed_report", "held_out_owners", "replay_demo",
           "replay", "summarize"]


PHASE_BINS = (0.05, 0.25, 0.5, 0.75, 1.0)  # entry / early / mid / late / end, closed on the right
PCTS = (5, 25, 50, 75, 95)

# The deployed cadence robot_ood.core.config.SentinelConfig runs the filter at; the replay
# measures the filter as deployed, so both the training-time embed and build skew use these.
DEPLOY_HIST = 8
DEPLOY_CHECK_EVERY = 8


def held_out_owners(nodes: NodeTable) -> np.ndarray:
    """Same by-demo split robot_ood.train holds out -- robot_ood.train.loop.heldout_demos.

    Args:
        nodes: The loaded node table.

    Returns:
        [n_heldout] int64 demo ids.
    """
    from robot_ood.train.loop import heldout_demos

    return heldout_demos(nodes.n_demos)


@dataclass(frozen=True)
class Sample:
    """One belief-filter check's phase reading paired with ground truth.

    Attributes:
        owner: Demo id (true strand).
        phase_gt: True phase of the raw frame the check window ends on.
        phase_hat: The filter's belief-weighted phase estimate at that check.
        top1_owner: Demo id the belief's top-1 node belongs to -- the strand it is committed to,
            so top1_owner != owner is a strand-lock-off event.
    """

    owner: int
    phase_gt: float
    phase_hat: float
    top1_owner: int


def replay_demo(locator, nodes: NodeTable, owner: int, *, hist: int, check_every: int) -> list[Sample]:
    """Step the filter over one held-out demo and pair each check with its ground-truth phase.

    Args:
        locator: The locator under test; reset() is called before replay.
        nodes: The node table this locator's retriever was built from.
        owner: Demo id to replay.
        hist: Query history window length.
        check_every: Raw frames between checks.

    Returns:
        One Sample per check.
    """
    raw_ptr = nodes.raw_ptr
    lo, hi = int(raw_ptr[owner]), int(raw_ptr[owner + 1])
    traj = nodes.q_raw[lo:hi]
    windows = RawWindowDataset(traj, hist=hist, check_every=check_every)
    locator.reset()
    out: list[Sample] = []
    for i in range(len(windows)):
        q_window = windows[i]
        step = locator.step(q_window)
        raw_idx = lo + int(windows.hi[i]) - 1
        node = nodes.node_at_raw(owner, raw_idx)
        phase_gt = float(nodes.phase[node])
        top1_node = int(step.topk.idx[0])
        top1_owner = int(nodes.owner[top1_node])
        out.append(Sample(
            owner=owner, phase_gt=phase_gt, phase_hat=float(step.phase_hat),
            top1_owner=top1_owner,
        ))
    return out


def _phase_bin(phase_gt: np.ndarray) -> np.ndarray:
    """Bucket ground-truth phases into PHASE_BINS.

    Args:
        phase_gt: [N] ground-truth phases in [0, 1].

    Returns:
        [N] int bin index, 0 = entry (phase <= PHASE_BINS[0]).
    """
    return np.searchsorted(PHASE_BINS, phase_gt, side="left")


def _describe(err: np.ndarray) -> dict:
    """Summary statistics of one error array.

    Args:
        err: Signed or absolute error values.

    Returns:
        {"n", "mean", "median", "std", "p5"..."p95"}.
    """
    if err.size == 0:
        return {"n": 0}
    out = {
        "n": int(err.size), "mean": float(err.mean()), "median": float(np.median(err)),
        "std": float(err.std()),
    }
    out.update({f"p{p}": float(np.percentile(err, p)) for p in PCTS})
    return out


ERR_TAIL_THRESHOLD = 0.10  # |phase_hat - phase_gt| beyond this counts as "lost track" for the
                            # strand-lock correlation (well above every clean bin's measured p95)


def _strand_lock_stats(off_strand: np.ndarray, err: np.ndarray) -> dict:
    """How often the filter's top-1 belief sits on the wrong demo, split by error magnitude.

    Args:
        off_strand: [N] bool, True where top1_owner != owner (filter committed to a different demo).
        err: [N] signed phase error, phase_hat - phase_gt.

    Returns:
        {"overall", "small_err" (|err| <= ERR_TAIL_THRESHOLD), "large_err" (|err| > threshold)}
        each an {"n", "off_strand_frac"} dict.
    """
    # If losing track means locking onto a wrong strand, off_strand is far more common among
    # large-|error| checks than small ones.
    abs_err = np.abs(err)
    large = abs_err > ERR_TAIL_THRESHOLD
    out = {}
    for label, mask in (("overall", np.ones_like(large)), ("small_err", ~large), ("large_err", large)):
        n = int(mask.sum())
        out[label] = {
            "n": n,
            "off_strand_frac": float(off_strand[mask].mean()) if n > 0 else None,
        }
    return out


def summarize(samples: Sequence[Sample]) -> dict:
    """Full report: overall signed/abs error, per-phase-bin breakdown, and strand-lock correlation.

    Args:
        samples: Every Sample collected.

    Returns:
        JSON-able report dict.
    """
    phase_gt = np.array([s.phase_gt for s in samples], dtype=np.float64)
    phase_hat = np.array([s.phase_hat for s in samples], dtype=np.float64)
    owner = np.array([s.owner for s in samples], dtype=np.int64)
    top1_owner = np.array([s.top1_owner for s in samples], dtype=np.int64)
    off_strand = owner != top1_owner
    err = phase_hat - phase_gt
    bins = _phase_bin(phase_gt)
    bin_labels = ["entry(<=0.05)", "0.05-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]

    per_bin = {}
    for b, label in enumerate(bin_labels):
        mask = bins == b
        per_bin[label] = {
            "signed": _describe(err[mask]),
            "abs": _describe(np.abs(err[mask])),
            "off_strand_frac": float(off_strand[mask].mean()) if mask.any() else None,
            "strand_lock": _strand_lock_stats(off_strand[mask], err[mask]),
        }

    return {
        "n_heldout_demos": len({s.owner for s in samples}),
        "n_checks": len(samples),
        "signed": _describe(err),
        "abs": _describe(np.abs(err)),
        "by_phase_bin": per_bin,
        "strand_lock": {
            "err_tail_threshold": ERR_TAIL_THRESHOLD,
            "stats": _strand_lock_stats(off_strand, err),
        },
    }


def replay(graph_dir: str | os.PathLike, *, hist: int, check_every: int, device: str,
           limit: int = 0, on_demo: "Callable[[int, int], None] | None" = None,
           cfg: "GraphConfig | None" = None) -> dict:
    """Replay the belief filter over a graph's held-out demos and summarise the phase skew.

    Args:
        graph_dir: Artifact dir holding g_nodes/g_edges/g_head.npz.
        hist: Query history window length; the deployed value is 8.
        check_every: Raw frames between checks; the deployed value is 8.
        device: Torch device to run the filter on.
        limit: Replay only the first N held-out demos; 0 replays all of them.
        on_demo: Called with (demos done, demos total) after each demo, for progress.
        cfg: Graph config the filter runs under; None reads the environment. A training run
            replaying its own just-written checkpoint passes its live config, so the replayed
            filter matches the one that run trained.

    Returns:
        The report summarize() builds, plus the replay settings it was measured under.
    """
    from dataclasses import replace

    from robot_ood.core.config import GraphConfig

    root = Path(graph_dir)
    nodes = NodeTable.load(root / "g_nodes.npz")
    cfg = replace(cfg if cfg is not None else GraphConfig.from_env(), hist=hist, device=device)
    locator = GraphLocator.load(root, cfg=cfg, device=device)

    owners = held_out_owners(nodes)
    if limit:
        owners = owners[:limit]
    samples: list[Sample] = []
    for i, owner in enumerate(owners, 1):
        samples.extend(replay_demo(locator, nodes, int(owner), hist=hist, check_every=check_every))
        if on_demo is not None:
            on_demo(i, len(owners))
    report = summarize(samples)
    report["replay"] = {"hist": hist, "check_every": check_every,
                        "n_heldout": int(len(owners)), "n_demos": int(nodes.n_demos),
                        "heldout_stride": HELDOUT_STRIDE}
    return report


def embed_report(graph_dir: str | os.PathLike, *, device: str, cfg: "GraphConfig | None" = None,
                 log: "Callable[[str], None] | None" = None) -> dict:
    """Replay a just-saved head over its held-out demos and embed the report in g_head.npz.

    Args:
        graph_dir: Artifact dir holding g_nodes/g_edges/g_head.npz.
        device: Torch device to run the filter on.
        cfg: Graph config the filter runs under; None reads the environment.
        log: Progress sink; None is silent.

    Returns:
        The embedded report.
    """
    import json

    from robot_ood.core import npz
    from robot_ood.data import schema

    emit = log if log is not None else (lambda _msg: None)

    def progress(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            emit(f"[locator] phase-skew replay {done}/{total} held-out demos")

    report = replay(graph_dir, hist=DEPLOY_HIST, check_every=DEPLOY_CHECK_EVERY,
                    device=device, cfg=cfg, on_demo=progress)
    head_path = Path(graph_dir) / schema.HEAD_NPZ
    arrays = dict(np.load(head_path, allow_pickle=False))
    report["graph_hash"] = str(arrays.get("graph_hash", ""))
    arrays[schema.PHASE_SKEW_KEY] = np.array(json.dumps(report))
    npz.write_arrays(head_path, schema.HEAD_NPZ, arrays)
    entry = report["by_phase_bin"]["entry(<=0.05)"]["signed"]
    emit(f"[locator] phase skew embedded: {report['n_checks']} checks, "
         f"entry bias {entry['median']:+.4f}")
    return report
