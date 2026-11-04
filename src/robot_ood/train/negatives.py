"""Mines hard-negative segments out of C1's segment memory; builds the on-device tensor view S1'/S2' read.

Deterministic given (table, seg_id): no RNG draw is involved, so two calls with the same seg_id
return the same negatives every time, and a query tiled across K CopyPlan copies carries identical
negatives on every copy without needing to store them per copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                                       # pragma: no cover -- typing only
    import torch

    from robot_ood.core.schemas import SegmentTable

__all__ = ["HardNegativeMiner", "SegmentTensors", "segment_tensors"]

# Prefilter width examined before DTW confirms or rejects, not an output size; must stay wider
# than n or a sparse corpus starves same_state_different_future.
_STATE_POOL = 64
# ROBOT_OOD_HARDNEG_STATE: "1" (default) mines same_state_different_future -- pose-near segments
# whose action chunk diverges -- as hard negatives; "0" drops that bucket; "far" replaces it with
# same_phase_far_state, same-task segments in the phase band whose start pose is farther than
# ROBOT_OOD_NEG_POSE_BAND. On a corpus where the future depends on unobserved object placement,
# the default bucket teaches the retriever to avoid the pose-nearest demos (results_dbg/audit_offset.py).
_STATE_BUCKET = __import__("os").environ.get("ROBOT_OOD_HARDNEG_STATE", "1")
_FAR_POSE = float(__import__("os").environ.get("ROBOT_OOD_NEG_POSE_BAND", "0.3") or 0.3)
# ROBOT_OOD_POS=xdemo: the InfoNCE / shortlist positive is the pose-nearest same-task segment of ANOTHER
# demo in the phase band (what deploy has to find), not the query's own segment.
POS_MODE = __import__("os").environ.get("ROBOT_OOD_POS", "own")


def _dtw_cost(anchor: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Dynamic-time-warp distance from one chunk to many, vectorised over the candidate axis.

    Args:
        anchor: [T1, D] float64, D >= 3.
        candidates: [M, T2, D] float64, D >= 3.

    Returns:
        [M] float64 mean warped per-step distance, metres (translation channel only).
    """
    import torch

    from robot_ood.train.align import dtw_cost

    m = candidates.shape[0]
    anchor_t = torch.as_tensor(
        np.broadcast_to(anchor[None, :, :3], (m, anchor.shape[0], 3)).copy(), dtype=torch.float64
    )
    cand_t = torch.as_tensor(np.ascontiguousarray(candidates[:, :, :3]), dtype=torch.float64)
    # Must equal robot_ood.build.measure.SegmentHardNegatives's normalisation bit-for-bit: dtw_thresh is
    # calibrated against this exact function's output.
    return dtw_cost(anchor_t, cand_t).numpy()


class HardNegativeMiner:
    """Mines hard-negative segments for one fixed SegmentTable.

    Attributes:
        dtw_thresh: Minimum DTW cost for a same-state candidate to count as "different future".
        phase_band: Half-width of the phase band all three buckets prefilter with.
        entry_band: Phase ceiling defining the entry region entry_same_state_different_task mines.
    """

    def __init__(
        self, table: "SegmentTable", *, dtw_thresh: float, phase_band: float, entry_band: float = 0.05,
    ) -> None:
        """Precomputes phase-sorted geometry once from a static SegmentTable.

        Args:
            table: The segment memory to mine from.
            dtw_thresh: Minimum action-chunk DTW cost, from
                robot_ood.core.config.calibrated_constants().value("seg_hard_neg_thresh").
            phase_band: Half-width of the phase band used to prefilter all three buckets.
            entry_band: Phase ceiling for entry_same_state_different_task, from robot_ood.core.constants'
                "seg_entry_band" -- the same value robot_ood.models.retrieval.scorer.SegmentScorer and
                robot_ood.train.data key entry-region behaviour on.
        """
        self.table = table
        self._memo: dict[tuple[int, int], np.ndarray] = {}
        self.dtw_thresh = float(dtw_thresh)
        self.phase_band = float(phase_band)
        self.entry_band = float(entry_band)
        self._phase = np.asarray(table.phase_start, dtype=np.float64)
        self._task = np.asarray(table.task_id, dtype=np.int64)
        self._owner = np.asarray(table.owner, dtype=np.int64)
        self._ee_start = np.asarray(table.ee_start, dtype=np.float64)
        self._chunk = np.asarray(table.action_chunk, dtype=np.float64)
        self._order = np.argsort(self._phase, kind="stable")
        self._sorted_phase = self._phase[self._order]
        self._entry_ids = np.flatnonzero(self._phase <= self.entry_band)

    def _phase_band_ids(self, seg_id: int, band: float) -> np.ndarray:
        """Segment ids within +/- band of seg_id's own phase, seg_id excluded."""
        phase = float(self._phase[seg_id])
        lo = np.searchsorted(self._sorted_phase, phase - band, side="left")
        hi = np.searchsorted(self._sorted_phase, phase + band, side="right")
        cand = self._order[lo:hi]
        return cand[cand != seg_id]

    def same_state_different_future(self, seg_id: int, n: int) -> np.ndarray:
        """Segments that look like seg_id's state but lead somewhere else.

        Args:
            seg_id: The anchor segment.
            n: Cap on the number of negatives.

        Returns:
            [<=n] int64 segment ids, most-divergent DTW first. Empty when the phase band around
            seg_id holds no segment whose action chunk clears dtw_thresh -- the corpus may simply
            not contain the failure mode at this point of the demo.
        """
        seg_id = int(seg_id)
        if n <= 0 or self.table.n_segments <= 1:
            return np.zeros(0, dtype=np.int64)
        # Approximation: only phase-band candidates are ever compared, so a segment outside the
        # band is never mined even if its ee_start coincides with seg_id's.
        band = self.phase_band
        cand = self._phase_band_ids(seg_id, band)
        # Widens geometrically until the state pool fills or the whole table is visible, so a
        # sparse corpus does not starve the bucket.
        while cand.size < _STATE_POOL and cand.size < self.table.n_segments - 1:
            band *= 2.0
            cand = self._phase_band_ids(seg_id, band)
        if cand.size == 0:
            return np.zeros(0, dtype=np.int64)

        d_state = np.linalg.norm(self._ee_start[cand] - self._ee_start[seg_id][None, :], axis=1)
        pool = cand[np.argsort(d_state)[:_STATE_POOL]]

        dtw = _dtw_cost(self._chunk[seg_id], self._chunk[pool])
        keep = pool[dtw > self.dtw_thresh]
        keep_dtw = dtw[dtw > self.dtw_thresh]
        order = np.argsort(-keep_dtw)                      # most divergent future first
        return keep[order][:n].astype(np.int64)

    def same_phase_far_state(self, seg_id: int, n: int) -> np.ndarray:
        """Same-task segments in the phase band whose start pose is beyond the pose band, farthest first."""
        seg_id = int(seg_id)
        if n <= 0 or self.table.n_segments <= 1:
            return np.zeros(0, dtype=np.int64)
        cand = self._phase_band_ids(seg_id, self.phase_band)
        cand = cand[(self._task[cand] == self._task[seg_id]) & (self._owner[cand] != self._owner[seg_id])]
        if cand.size == 0:
            return np.zeros(0, dtype=np.int64)
        d_state = np.linalg.norm(self._ee_start[cand] - self._ee_start[seg_id][None, :], axis=1)
        far = cand[d_state > _FAR_POSE]
        d_far = d_state[d_state > _FAR_POSE]
        order = np.argsort(d_far) if _STATE_BUCKET == "nearfar" else np.argsort(-d_far)
        return far[order][:n].astype(np.int64)

    def cross_demo_positive(self, seg_id: int) -> int:
        """The pose-nearest same-task segment of another demo within the phase band (widened until one
        exists); seg_id itself when the task has a single demo."""
        seg_id = int(seg_id)
        band = self.phase_band
        while True:
            cand = self._phase_band_ids(seg_id, band)
            cand = cand[(self._task[cand] == self._task[seg_id]) & (self._owner[cand] != self._owner[seg_id])]
            if cand.size or band > 1.0:
                break
            band *= 2.0
        if cand.size == 0:
            return seg_id
        d_state = np.linalg.norm(self._ee_start[cand] - self._ee_start[seg_id][None, :], axis=1)
        return int(cand[np.argmin(d_state)])

    def positive_batch(self, seg_ids: np.ndarray) -> np.ndarray:
        """cross_demo_positive() per row; -1 rows stay -1."""
        seg_ids = np.asarray(seg_ids, dtype=np.int64)
        out = np.full(seg_ids.shape, -1, dtype=np.int64)
        for row, sid in enumerate(seg_ids):
            if sid >= 0:
                out[row] = self.cross_demo_positive(int(sid))
        return out

    def same_phase_wrong_task(self, seg_id: int, n: int) -> np.ndarray:
        """Segments at a comparable phase but a different task.

        Args:
            seg_id: The anchor segment.
            n: Cap on the number of negatives.

        Returns:
            [<=n] int64 segment ids, nearest phase first.
        """
        seg_id = int(seg_id)
        if n <= 0 or self.table.n_segments <= 1:
            return np.zeros(0, dtype=np.int64)
        cand = self._phase_band_ids(seg_id, self.phase_band)
        cand = cand[self._task[cand] != self._task[seg_id]]
        if cand.size == 0:
            return np.zeros(0, dtype=np.int64)
        order = np.argsort(np.abs(self._phase[cand] - self._phase[seg_id]))
        return cand[order][:n].astype(np.int64)

    def entry_same_state_different_task(self, seg_id: int, n: int) -> np.ndarray:
        """The nearest-ee_start, different-task entry-region segments to seg_id.

        Args:
            seg_id: The anchor segment.
            n: Cap on the number of negatives.

        Returns:
            [<=n] int64 segment ids, nearest ee_start first. Empty when seg_id is outside the entry
            band or no other-task segment sits inside it.
        """
        seg_id = int(seg_id)
        # Mid-demo anchors already have task-discriminative state; only an entry-band anchor
        # needs this bucket.
        if n <= 0 or self.table.n_segments <= 1 or self._phase[seg_id] > self.entry_band:
            return np.zeros(0, dtype=np.int64)
        cand = self._entry_ids[self._entry_ids != seg_id]
        cand = cand[self._task[cand] != self._task[seg_id]]
        if cand.size == 0:
            return np.zeros(0, dtype=np.int64)
        d_state = np.linalg.norm(self._ee_start[cand] - self._ee_start[seg_id][None, :], axis=1)
        return cand[np.argsort(d_state)][:n].astype(np.int64)

    def negatives(self, seg_id: int, n: int) -> np.ndarray:
        key = (int(seg_id), int(n))
        hit = self._memo.get(key)
        if hit is None:
            hit = self._memo[key] = self._mine(*key)
        return hit

    def _mine(self, seg_id: int, n: int) -> np.ndarray:
        """Every applicable bucket combined, split evenly and topped up from whichever has more.

        Args:
            seg_id: The anchor segment.
            n: Cap on the number of negatives.

        Returns:
            [<=n] int64 unique segment ids. Never includes seg_id itself.
        """
        if n <= 0:
            return np.zeros(0, dtype=np.int64)
        if _STATE_BUCKET in ("far", "nearfar"):
            buckets = [self.same_phase_far_state, self.same_phase_wrong_task]
        elif _STATE_BUCKET == "0":
            buckets = [self.same_phase_wrong_task]
        else:
            buckets = [self.same_state_different_future, self.same_phase_wrong_task]
        is_entry_anchor = self.table.n_segments > 0 and float(self._phase[int(seg_id)]) <= self.entry_band
        if is_entry_anchor:
            buckets.append(self.entry_same_state_different_task)
        share = max(1, n // len(buckets))
        picks = [b(seg_id, share) for b in buckets]
        picked = np.unique(np.concatenate(picks)) if any(len(p) for p in picks) else np.zeros(0, dtype=np.int64)
        if len(picked) < n:
            more = [b(seg_id, n) for b in buckets]
            picked = np.unique(np.concatenate([picked, *more]))
        return picked[:n].astype(np.int64)

    def mine_batch(self, seg_ids: np.ndarray, n: int) -> np.ndarray:
        """negatives() for a batch of anchors, -1 padded to a fixed width.

        Args:
            seg_ids: [B] int64 anchor segment ids, -1 for a row that mines no negatives at all.
            n: Negatives per row.

        Returns:
            [B, n] int64, -1 padded.
        """
        seg_ids = np.asarray(seg_ids, dtype=np.int64)
        out = np.full((seg_ids.shape[0], max(n, 0)), -1, dtype=np.int64)
        for row, sid in enumerate(seg_ids):
            if sid < 0:
                continue
            negs = self.negatives(int(sid), n)
            out[row, : len(negs)] = negs
        return out


@dataclass(frozen=True)
class SegmentTensors:
    """On-device tensor view of one C1 SegmentTable -- what S1'/S2' read every training step.

    Attributes:
        action_chunk: [S, SEGMENT_LEN, ACTION_DIM] float32.
        phase_start: [S] float32.
        task_id: [S] int64.
        owner: [S] int64.
        arm: [S] int64.
        ee_start: [S, 6] float32.
        ee_end: [S, 6] float32.
        node_to_segment: [V] int64, the segment each real node starts, or -1 -- FUSION's own
            realisation of a WRONG_TASK/WRONG_PHASE corruption reads this to turn the node
            robot_ood.train.augment.RetrievalCorruptor picked into the segment id the candidate set
            is indexed by (see robot_ood.train.fcsr.realize_column0).
        n_segments: S.
    """

    action_chunk: "torch.Tensor"
    phase_start: "torch.Tensor"
    task_id: "torch.Tensor"
    owner: "torch.Tensor"
    arm: "torch.Tensor"
    ee_start: "torch.Tensor"
    ee_end: "torch.Tensor"
    node_to_segment: "torch.Tensor"
    n_segments: int


def segment_tensors(table: "SegmentTable", device: Any) -> SegmentTensors:
    """Upload the columns S1'/S2' read off a SegmentTable, once per run.

    Args:
        table: The segment memory.
        device: Target device.

    Returns:
        The device-resident bundle.
    """
    import torch

    return SegmentTensors(
        action_chunk=torch.as_tensor(np.asarray(table.action_chunk, dtype=np.float32), device=device),
        phase_start=torch.as_tensor(np.asarray(table.phase_start, dtype=np.float32), device=device),
        task_id=torch.as_tensor(np.asarray(table.task_id, dtype=np.int64), device=device),
        owner=torch.as_tensor(np.asarray(table.owner, dtype=np.int64), device=device),
        arm=torch.as_tensor(np.asarray(table.arm, dtype=np.int64), device=device),
        ee_start=torch.as_tensor(np.asarray(table.ee_start, dtype=np.float32), device=device),
        ee_end=torch.as_tensor(np.asarray(table.ee_end, dtype=np.float32), device=device),
        node_to_segment=torch.as_tensor(np.asarray(table.node_to_segment, dtype=np.int64), device=device),
        n_segments=table.n_segments,
    )
