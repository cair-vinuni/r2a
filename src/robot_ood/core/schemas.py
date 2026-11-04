"""Interface contract for Future-Conditioned Segment Retrieval (FCSR).

Shared by the segment memory, the scorer, the retriever, fusion and the data engine.
No parameter tensor may be dimensioned by V (nodes), S (segments), or n_tasks (inductive invariant).
A constant here may only be architectural (a width, a field order); a magnitude (a threshold, a
temperature, a band) belongs in robot_ood.build.calibration instead."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                       # torch stays out of the import path so
    from torch import Tensor                            # robot_ood.core.config and the CPU tools can read this

__all__ = [
    "SEGMENTS_NPZ",
    "MISSING_FCSR",
    "SEGMENT_LEN",
    "SEG_TOPK",
    "SEG_TIER1_M",
    "SEG_FEATURES",
    "SEG_FEATURE_DIM",
    "ACTION_DIM",
    "ACTION_EMBED_DIM",
    "GRIPPER_COL",
    "SegmentTable",
    "QueryContext",
    "SegmentCandidates",
]

# Artifact filenames: centralized so all components agree on the names.

SEGMENTS_NPZ = "g_segments.npz"        # the segment memory

MISSING_FCSR = ("no trained FCSR checkpoint. Train one with "
                "`python -m robot_ood.cli train fcsr --suite <suite>`.")

# --- architectural widths -------------------------------------------------------------------

SEGMENT_LEN = 16        # H: raw frames per segment. Default; robot_ood.core.constants may override as "segment_len".
SEG_TOPK = 8            # K structured candidates handed to fusion. Architectural: the corrector's K axis.
SEG_TIER1_M = 24



ACTION_DIM = 7          # 6-DoF delta + gripper, the policy's action space
ACTION_EMBED_DIM = 64
GRIPPER_COL = 6         # the column that must pass through fusion untouched

# The scorer's feature vector, in order. The tests assert on it by NAME. Any change here is a
# change to g_segments.npz's stamp and to every trained scorer checkpoint.
SEG_FEATURES: tuple[str, ...] = (
    "d_state",      # ||ee_query - segment.ee_start|| in the metric's normalized units
    "d_future",     # 1 - cos(ActionEncoder(query future), ActionEncoder(segment.action_chunk))
    "phase_diff",
    "task_affinity",  # belief task marginal at the candidate's task
)
SEG_FEATURE_DIM = len(SEG_FEATURES)


@dataclass(frozen=True, slots=True)
class SegmentTable:
    """Segment memory: S short trajectory pieces from the demonstration graph.

    Attributes:
        start_node: [S] int32, index into NodeTable, the segment's first node.
        end_node: [S] int32, its last node (inclusive).
        owner: [S] int32 demo id; both endpoints share it by construction.
        task_id: [S] int16.
        phase_start: [S] float32 in [0, 1].
        phase_end: [S] float32 in [0, 1].
        action_chunk: [S, SEGMENT_LEN, ACTION_DIM] float32, the demo's own recorded actions.
        ee_start: [S, 6] float32 base-frame concat(position, unnormalized axis-angle).
        ee_end: [S, 6] float32.
        node_to_segment: [V] int32, the segment each node starts, or -1 (never a nearest-real
            fallback: a caller must fail loudly on -1, not borrow a neighbour's continuation).
            Length V covers the virtual suffix too, so a caller can index it with any node id the
            readout produces.
        arm: [S] int16, which arm of the robot the segment was recorded on; None fills with 0, the
            single-arm case. Both arms share one frame -- the right one is mirrored into the left's
            (robot_ood.build.graph) -- so nothing else in the table tells them apart.
    """

    start_node: np.ndarray
    end_node: np.ndarray
    owner: np.ndarray
    task_id: np.ndarray
    phase_start: np.ndarray
    phase_end: np.ndarray
    action_chunk: np.ndarray
    ee_start: np.ndarray
    ee_end: np.ndarray
    node_to_segment: np.ndarray
    arm: "np.ndarray | None" = None

    def __post_init__(self) -> None:
        """Fill the bimanual column a single-arm graph leaves out, as NodeTable does."""
        if self.arm is None:
            object.__setattr__(self, "arm", np.zeros(self.n_segments, dtype=np.int16))

    @property
    def n_segments(self) -> int:
        """S."""
        return int(self.start_node.shape[0])


@dataclass(frozen=True, slots=True)
class QueryContext:
    """What QueryEncoder reads.

    Attributes:
        qh: [B, H, dim] joint positions, row 0 oldest.
        vh: [B, H, dim] joint velocities.
        gh: [B, H] gripper state.
        ee: [B, 6] current end-effector pose, base-frame concat(position, axis-angle), by FK.
        phase: [B] the belief's current phase estimate.
        a_hist: [B, H, ACTION_DIM] actions the policy commanded over the window, or None.
        e_lang: [B, D_lang] instruction embedding, already centered, or None.
        task_p: [B, n_tasks] the locator belief's task marginal, or None. This is what the locator
            already concluded from geometry, instruction and memory; retrieval inherits it instead
            of re-deriving a task from the instruction embedding alone.
        seg_prior: [B, S] log prior over the segment table, or None. It carries what the query
            window cannot see, such as what the other arm of a two-armed robot is doing, into
            the shortlist cut; the final score never sees it.
    """

    qh: "Tensor"
    vh: "Tensor"
    gh: "Tensor"
    ee: "Tensor | None" = None
    phase: "Tensor | None" = None
    a_hist: "Tensor | None" = None
    e_lang: "Tensor | None" = None
    task_p: "Tensor | None" = None
    seg_prior: "Tensor | None" = None


@dataclass(frozen=True, slots=True)
class SegmentCandidates:
    """Retrieval's output: K structured candidates, ordered by descending score.

    Attributes:
        idx: [K] int32 segment ids.
        score: [K] float32 raw segment scores.
        action_chunk: [K, SEGMENT_LEN, ACTION_DIM] float32.
        phase: [K] float32, each candidate's phase_start.
        ee_start: [K, 6] float32.
        ee_end: [K, 6] float32.
        start_node: [K] int32, each candidate segment's SegmentTable.start_node -- the node id
            robot_ood.models.fusion.target.node_pose_ref walks from to build the POSE reference
            fusion blends toward, as opposed to ee_start/ee_end's two fixed endpoints.
        end_node: [K] int32, each candidate segment's SegmentTable.end_node (inclusive).
    """

    idx: np.ndarray
    score: np.ndarray
    action_chunk: np.ndarray
    phase: np.ndarray
    ee_start: np.ndarray
    ee_end: np.ndarray
    start_node: np.ndarray
    end_node: np.ndarray

    @property
    def k(self) -> int:
        """K."""
        return int(self.idx.shape[0])
