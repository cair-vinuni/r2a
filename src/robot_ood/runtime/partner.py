from __future__ import annotations

import math
import os
from collections import deque
from pathlib import Path
from typing import Mapping

import numpy as np

from robot_ood.core.action import POSE_DIM
from robot_ood.core.constants import PHASE_TOL, STILL_TOL
from robot_ood.data import schema
from robot_ood.data.nodes import NodeTable
from robot_ood.data.segments import load_segments

__all__ = ["MOVING_Q", "PAUSE_Q", "PRIOR_FLOOR", "PartnerPrior", "frame_phase", "lagged", "moved_frames",
           "moving_frames", "moving_threshold", "node_spans", "pause_span"]

MOVING_Q = 5.0
PAUSE_Q = 95.0
PRIOR_FLOOR = 1e-6


def moved_frames(q_raw: np.ndarray, raw_ptr: np.ndarray) -> np.ndarray:
    q = np.asarray(q_raw, dtype=np.float64)
    moved = np.zeros(len(q), dtype=bool)
    moved[1:] = np.linalg.norm(np.diff(q, axis=0), axis=1) >= STILL_TOL
    moved[np.asarray(raw_ptr, dtype=np.int64)[:-1]] = False
    return moved


def _window_starts(raw_ptr: np.ndarray, window: int) -> np.ndarray:
    ptr = np.asarray(raw_ptr, dtype=np.int64)
    return np.concatenate([np.arange(a, b - window, window) for a, b in zip(ptr[:-1], ptr[1:])])


def moving_threshold(q_raw: np.ndarray, moved: np.ndarray, raw_ptr: np.ndarray, window: int) -> float:
    q = np.asarray(q_raw, dtype=np.float64)[:, :POSE_DIM]
    starts = _window_starts(raw_ptr, window)
    csum = np.concatenate([[0], np.cumsum(moved, dtype=np.int64)])
    frac = (csum[starts + window + 1] - csum[starts + 1]) / window
    disp = np.linalg.norm(q[starts + window] - q[starts], axis=1)
    return float(np.percentile(disp[frac >= 0.5], MOVING_Q))


def pause_span(moved: np.ndarray, raw_ptr: np.ndarray, window: int) -> int:
    ptr = np.asarray(raw_ptr, dtype=np.int64)
    runs = []
    for a, b in zip(ptr[:-1], ptr[1:]):
        idx = np.flatnonzero(moved[a:b])
        if len(idx) < 2:
            continue
        edges = np.diff(moved[a + idx[0]:a + idx[-1] + 1].astype(np.int8))
        runs.append(np.flatnonzero(edges == 1) - np.flatnonzero(edges == -1))
    pauses = np.concatenate(runs) if runs else np.zeros(0, dtype=np.int64)
    longest = np.percentile(pauses, PAUSE_Q) if len(pauses) else 0.0
    return max(int(math.ceil(longest / window)), 1)


def _demo_start(raw_ptr: np.ndarray) -> np.ndarray:
    ptr = np.asarray(raw_ptr, dtype=np.int64)
    return np.repeat(ptr[:-1], np.diff(ptr))


def moving_frames(q_raw: np.ndarray, raw_ptr: np.ndarray, window: int, span: int, threshold: float) -> np.ndarray:
    q = np.asarray(q_raw, dtype=np.float64)[:, :POSE_DIM]
    start, frame = _demo_start(raw_ptr), np.arange(len(q))
    moving = np.zeros(len(q), dtype=bool)
    for j in range(span):
        hi, lo = np.maximum(frame - j * window, start), np.maximum(frame - (j + 1) * window, start)
        moving |= np.linalg.norm(q[hi] - q[lo], axis=1) >= threshold
    return moving


def node_spans(t_raw: np.ndarray, owner: np.ndarray, raw_ptr: np.ndarray) -> np.ndarray:
    t_raw, owner, ptr = (np.asarray(x, dtype=np.int64) for x in (t_raw, owner, raw_ptr))
    same = np.append(owner[1:] == owner[:-1], False)
    end = np.where(same, np.append(t_raw[1:], 0), ptr[owner + 1])
    spans = np.stack([t_raw, end], axis=1)
    if spans[0, 0] != 0 or spans[-1, 1] != ptr[-1] or not np.array_equal(spans[1:, 0], spans[:-1, 1]):
        raise ValueError("the nodes do not tile their episodes frame for frame")
    return spans


def frame_phase(phase: np.ndarray, spans: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(phase, dtype=np.float64), spans[:, 1] - spans[:, 0])


def lagged(values: np.ndarray, raw_ptr: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values)
    return values[np.maximum(np.arange(len(values)) - window, _demo_start(raw_ptr))]


class PartnerPrior:
    def __init__(self, spans: Mapping[str, np.ndarray], moving: Mapping[str, np.ndarray],
                 phase: Mapping[str, np.ndarray], threshold: Mapping[str, float], span: Mapping[str, int],
                 window: int, sigma: float = PHASE_TOL) -> None:
        arms = set(spans)
        if len(arms) != 2 or any(set(m) != arms for m in (moving, phase, threshold, span)):
            raise ValueError(f"a partner prior names exactly two arms, got {sorted(spans)}")
        self.spans = {arm: np.asarray(s, dtype=np.int64) for arm, s in spans.items()}
        self.frames_moving = {arm: np.asarray(m, dtype=bool) for arm, m in moving.items()}
        self.frames_phase = {arm: np.asarray(p, dtype=np.float64) for arm, p in phase.items()}
        self.threshold = {arm: float(t) for arm, t in threshold.items()}
        self.span = {arm: int(s) for arm, s in span.items()}
        self.window, self.sigma = int(window), float(sigma)
        self.partner = {a: b for a in arms for b in arms if b != a}
        self.reset()

    @classmethod
    def from_lanes(cls, graph_dirs: Mapping[str, str | os.PathLike[str]], window: int) -> "PartnerPrior":
        if len(graph_dirs) != 2:
            raise ValueError(f"a partner prior needs exactly two lanes, got {sorted(graph_dirs)}")
        tables = {arm: NodeTable.load(Path(d) / schema.NODES_NPZ, strict=False) for arm, d in graph_dirs.items()}
        (a, ta), (b, tb) = tables.items()
        if not np.array_equal(ta.raw_ptr, tb.raw_ptr):
            raise ValueError(f"lanes {a} and {b} do not share their episodes frame for frame")
        spans, moving, phase, threshold, span = {}, {}, {}, {}, {}
        for arm, table in tables.items():
            nodes = node_spans(table.t_raw, table.owner, table.raw_ptr)
            spans[arm] = nodes[np.asarray(load_segments(Path(graph_dirs[arm])).start_node, dtype=np.int64)]
            moved = moved_frames(table.q_raw, table.raw_ptr)
            threshold[arm] = moving_threshold(table.q_raw, moved, table.raw_ptr, window)
            span[arm] = pause_span(moved, table.raw_ptr, window)
            moving[arm] = moving_frames(table.q_raw, table.raw_ptr, window, span[arm], threshold[arm])
            phase[arm] = lagged(frame_phase(table.phase, nodes), table.raw_ptr, window)
        return cls(spans, moving, phase, threshold, span, window)

    def reset(self) -> None:
        self._q = {arm: deque(maxlen=self.span[arm] * self.window + 1) for arm in self.partner}
        self._phase = {}

    def observe(self, q: Mapping[str, np.ndarray]) -> None:
        for arm, row in q.items():
            self._q[arm].append(np.asarray(row, dtype=np.float64)[:POSE_DIM])

    def observe_phase(self, phase: Mapping[str, float]) -> None:
        for arm, value in phase.items():
            self._phase[arm] = float(value)

    def moving(self, arm: str) -> bool | None:
        rows, w = np.asarray(self._q[arm]), self.window
        if len(rows) < w + 1:
            return None
        disp = [np.linalg.norm(rows[-1 - j * w] - rows[-1 - (j + 1) * w]) for j in range((len(rows) - 1) // w)]
        return max(disp) >= self.threshold[arm]

    def planned_moving(self, arm: str, targets: np.ndarray) -> bool:
        """Whether the policy's own plan moves this arm over the next window.

        Args:
            arm: The arm the plan is for.
            targets: [T, >=POSE_DIM] the policy's absolute joint targets for it, row 0 next.

        Returns:
            True when the plan's displacement over one window reaches the arm's moving threshold.
        """
        rows = np.asarray(targets, dtype=np.float64)[:, :POSE_DIM]
        if len(rows) < 2:
            return False
        now = rows[0] if not self._q[arm] else np.asarray(self._q[arm][-1])
        return float(np.linalg.norm(rows[min(self.window, len(rows) - 1)] - now)) >= self.threshold[arm]

    def prior(self, arm: str, planned: Mapping[str, np.ndarray] | None = None) -> np.ndarray | None:
        """Log prior over this arm's segments from what the other arm is doing.

        Args:
            arm: The arm whose retrieval the prior is for.
            planned: arm -> the policy's [T, >=POSE_DIM] absolute joint targets, or None. Before a
                window of joints has been observed (the first checks of an episode), the other
                arm's planned motion stands in for its observed motion, and its phase for the
                episode's start (0) when none has been read yet.

        Returns:
            [S] log prior, or None when neither observed nor planned motion is available.
        """
        other = self.partner[arm]
        moving = self.moving(other)
        phase = self._phase.get(other)
        if moving is None and planned is not None and planned.get(other) is not None:
            moving = self.planned_moving(other, planned[other])
            if phase is None:
                phase = 0.0
        if moving is None or phase is None:
            return None
        weight = ((self.frames_moving[other] == moving)
                  * np.exp(-0.5 * ((self.frames_phase[other] - phase) / self.sigma) ** 2))
        csum = np.concatenate([[0.0], np.cumsum(weight)])
        start, end = self.spans[arm].T
        return np.log(np.maximum((csum[end] - csum[start]) / (end - start), PRIOR_FLOOR))
