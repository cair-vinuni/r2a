"""Segment views over the demonstration graph: torch reference segments for training, and the C1
SegmentTable memory bank (g_segments.npz) FCSR's retriever reads at deploy time.

The torch half (ee_segment_table, expected_ee_segment) feeds the action-space blend loss and is
unchanged by FCSR. The numpy half (SegmentBuilder, save_segments, load_segments) is new: it carves
the real-node prefix into SEGMENT_LEN-frame trajectory pieces, each carrying the demo's own future
action chunk, for robot_ood.core.schemas.SegmentTable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from robot_ood.core import npz
from robot_ood.core.schemas import ACTION_DIM, SEGMENT_LEN, SegmentTable
from robot_ood.data import schema

if TYPE_CHECKING:
    from robot_ood.data.nodes import NodeTable

__all__ = [
    "EXPECT_K", "ee_segment_table", "expected_ee_segment", "node_ee_table", "segment_frames",
    "segment_pause",
    "graph_plan_rows", "SegmentBuilder", "save_segments", "load_segments",
]

EXPECT_K = 32              # expected_ee_segment support width; matches ReadoutContext.topm
_EPS = 1e-12

_DTYPE: Final[dict[str, np.dtype]] = {
    "i4": np.dtype(np.int32), "i2": np.dtype(np.int16),
    "f4": np.dtype(np.float32), "f2": np.dtype(np.float16),
}


def ee_segment_table(nodes: "NodeTable", seg_k: int = schema.SEG_K) -> torch.Tensor:
    """Materialize every node's reference end-effector segment as one tensor.

    Args:
        nodes: The demonstration-graph node table.
        seg_k: Segment length in frames.

    Returns:
        [V, seg_k, 6] float32 base-frame concat(position, unnormalized axis-angle).

    Raises:
        ValueError: seg_k < 1, or some node's t_raw falls outside its own demo's raw range.
    """
    if seg_k < 1:
        raise ValueError(f"seg_k must be >= 1, got {seg_k}")
    owner = nodes.owner.astype(np.int64)
    raw_ptr = nodes.raw_ptr.astype(np.int64)
    start = nodes.t_raw.astype(np.int64)
    demo_start, demo_end = raw_ptr[owner], raw_ptr[owner + 1]

    outside = np.flatnonzero((start < demo_start) | (start >= demo_end))
    if outside.size:
        v = int(outside[0])
        raise ValueError(
            f"t_raw[{v}]={start[v]} outside demo {owner[v]}'s raw range "
            f"[{demo_start[v]}, {demo_end[v]})"
        )

    # Clamping the read index at the demo's last frame IS the edge padding: past that frame every
    # further row repeats it, which is what NodeTable._edge_pad produces.
    last = np.minimum(start + seg_k, demo_end) - 1
    offsets = np.arange(seg_k, dtype=np.int64)
    idx = np.minimum(start[:, None] + offsets[None, :], last[:, None])
    return torch.as_tensor(np.asarray(nodes.ee_base[idx], dtype=np.float32))


def expected_ee_segment(p: torch.Tensor, table: torch.Tensor, k: int = EXPECT_K) -> torch.Tensor:
    """Posterior expectation of the end-effector segment, differentiable w.r.t. p.

    Args:
        p: [V] or [B, V] distribution over nodes; need not already sum to 1.
        table: [V, seg_k, 6] segments from ee_segment_table.
        k: Support width, clamped to V.

    Returns:
        [seg_k, 6] or [B, seg_k, 6], in the promoted dtype of p and table.

    Raises:
        ValueError: k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Deliberately a fixed k, not the numpy readout's mass-based support (that would be ragged
    # once batched); the two agree exactly when the realised supports coincide.
    top_p, idx = torch.topk(p, min(k, p.shape[-1]), dim=-1)
    w = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    return (w[..., None, None] * table[idx]).sum(dim=-3)


def node_ee_table(nodes: "NodeTable") -> np.ndarray:
    """Per-node end-effector pose, in the robot base frame: nodes.ee_base at each node's own raw frame.

    Args:
        nodes: The demonstration-graph node table.

    Returns:
        [V, 6] float32 base-frame concat(position, unnormalized axis-angle).
    """
    return np.asarray(nodes.ee_base[nodes.t_raw], dtype=np.float32)


def segment_pause(nodes: "NodeTable", segments: SegmentTable) -> np.ndarray:
    start = np.asarray(segments.start_node, dtype=np.int64)
    hold = start == np.asarray(segments.end_node, dtype=np.int64)
    n_nodes = np.diff(np.asarray(nodes.demo_ptr, dtype=np.int64))
    moved = n_nodes[np.asarray(nodes.owner, dtype=np.int64)[start]] > 1
    return hold & moved


def segment_frames(nodes: "NodeTable", segments: SegmentTable) -> int:
    """Demo frames one segment spans, the median over the table.

    Args:
        nodes: The demonstration-graph node table the segments index into.
        segments: The segment memory bank.

    Returns:
        The frame count, at least one.
    """
    # Node coarsening sets this, not SEG_K: a segment runs from its start node to its end node.
    t_raw = np.asarray(nodes.t_raw, dtype=np.int64)
    start = t_raw[np.asarray(segments.start_node, dtype=np.int64)]
    end = t_raw[np.asarray(segments.end_node, dtype=np.int64)]
    return max(int(np.median(end - start)) + 1, 1)


def graph_plan_rows(graph_dir: str | os.PathLike[str], policy_chunk: int) -> int:
    """How many rows of one policy chunk a plan off this graph may cover.

    Args:
        graph_dir: The artifact directory holding g_nodes.npz and g_segments.npz.
        policy_chunk: Rows the policy executes per query.

    Returns:
        The row count, a divisor of policy_chunk.
    """
    from robot_ood.core.config import plan_rows
    from robot_ood.data.nodes import NodeTable

    nodes = NodeTable.load(graph_dir)
    return plan_rows(policy_chunk, segment_frames(nodes, load_segments(graph_dir)))


# ==== the segment memory bank -- robot_ood.core.schemas.SegmentTable / g_segments.npz ====
class SegmentBuilder:
    """Carves the real-node prefix of a built graph into SEGMENT_LEN-frame trajectory segments.

    Args:
        nodes: Built node table.
        segment_len: H, raw frames per segment. Defaults to the architectural SEGMENT_LEN; pass the
            calibrated robot_ood.core.constants value when one exists.
        stride: Real-node spacing between segment starts, counted from each demo's own first real
            node (local index mod stride == 0). None means every real node starts a segment.

    Raises:
        ValueError: segment_len < 1, or stride is not None and < 1.
    """

    def __init__(
        self,
        nodes: "NodeTable",
        *,
        segment_len: int = SEGMENT_LEN,
        stride: int | None = None,
    ) -> None:
        if segment_len < 1:
            raise ValueError(f"segment_len must be >= 1, got {segment_len}")
        if stride is not None and stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self._nodes = nodes
        self._segment_len = int(segment_len)
        self._stride = 1 if stride is None else int(stride)

    def build(self) -> SegmentTable:
        """Build the segment table.

        Returns:
            The segment memory bank.
        """
        nodes = self._nodes
        v, n_real = len(nodes), nodes.n_real
        if n_real == 0:
            empty_f4 = np.zeros((0, self._segment_len, ACTION_DIM), dtype=np.float32)
            return SegmentTable(
                start_node=np.zeros(0, dtype=np.int32), end_node=np.zeros(0, dtype=np.int32),
                owner=np.zeros(0, dtype=np.int32), task_id=np.zeros(0, dtype=np.int16),
                arm=np.zeros(0, dtype=np.int16),
                phase_start=np.zeros(0, dtype=np.float32), phase_end=np.zeros(0, dtype=np.float32),
                action_chunk=empty_f4, ee_start=np.zeros((0, 6), dtype=np.float32),
                ee_end=np.zeros((0, 6), dtype=np.float32),
                node_to_segment=np.full(v, -1, dtype=np.int32),
            )

        owner = nodes.owner[:n_real].astype(np.int64)
        demo_ptr = nodes.demo_ptr.astype(np.int64)
        raw_ptr = nodes.raw_ptr.astype(np.int64)
        t_raw = nodes.t_raw[:n_real].astype(np.int64)

        # Local position within the owning demo's real-node run; nodes are stored in raw-frame
        # order, so this is exactly the demo-relative segment-start stride the docstring promises.
        local = np.arange(n_real, dtype=np.int64) - demo_ptr[owner]
        start_node = np.nonzero(local % self._stride == 0)[0]

        s_owner = owner[start_node]
        start_raw = t_raw[start_node]
        demo_end_raw = raw_ptr[s_owner + 1]
        # Same edge padding as ee_segment_table: clamp the read index at the demo's last frame
        # rather than reading past it into the next demo.
        last_raw = np.minimum(start_raw + self._segment_len, demo_end_raw) - 1

        offsets = np.arange(self._segment_len, dtype=np.int64)
        idx = np.minimum(start_raw[:, None] + offsets[None, :], last_raw[:, None])
        # nodes.a_raw is the demo's own recorded action stream, never a_pi_raw.npz (a cached
        # policy's replayed actions): substituting one silently swaps what the demonstrator did.
        action_chunk = np.asarray(nodes.a_raw, dtype=np.float32)[idx]

        ee_base = np.asarray(nodes.ee_base, dtype=np.float32)
        ee_start = ee_base[start_raw]
        ee_end = ee_base[last_raw]

        end_node = nodes.node_at_raw_many(s_owner, last_raw)
        task_id = np.asarray(nodes.task_id, dtype=np.int16)[start_node]
        arm = np.asarray(nodes.arm, dtype=np.int16)[start_node]
        phase = np.asarray(nodes.phase, dtype=np.float32)
        phase_start = phase[start_node]
        phase_end = phase[end_node]

        node_to_segment = np.full(v, -1, dtype=np.int32)
        node_to_segment[start_node] = np.arange(len(start_node), dtype=np.int32)

        return SegmentTable(
            start_node=start_node.astype(np.int32),
            end_node=end_node.astype(np.int32),
            owner=s_owner.astype(np.int32),
            task_id=task_id,
            arm=arm,
            phase_start=phase_start,
            phase_end=phase_end,
            action_chunk=action_chunk,
            ee_start=ee_start,
            ee_end=ee_end,
            node_to_segment=node_to_segment,
        )

# What to tell a caller whose g_segments.npz is missing.
MISSING_SEGMENTS = "no segment table. Build one with `python -m robot_ood.cli build segments --suite <suite>`."


def save_segments(
    table: SegmentTable, path: str | os.PathLike[str], *, graph_hash: str | None = None,
) -> Path:
    """Write g_segments.npz.

    Args:
        table: Segment table to persist.
        path: Destination directory or full .npz path.
        graph_hash: The (NodeTable, EdgeSet) content hash this table was built against
            (robot_ood.data.geometry.graph_hash), or None to write an unstamped file. A caller
            that wants the load-time refusal g_head.npz gives must pass this.

    Returns:
        The written path.
    """
    payload = {name: getattr(table, name) for name in schema.ArtifactSchemas.SEGMENTS.keys}
    payload["graph_hash"] = np.array("" if graph_hash is None else graph_hash)
    return npz.write_arrays(path, schema.SEGMENTS_NPZ, payload, compress=True)


def load_segments(
    path: str | os.PathLike[str], *, graph_hash: str | None = None,
) -> SegmentTable:
    """Read g_segments.npz.

    Args:
        path: Source directory or full .npz path.
        graph_hash: When given, the stamp this file must carry -- see save_segments. Node ids are
            positional, so a segment table built off a foreign graph returns confidently wrong
            candidates with no shape error anywhere downstream.

    Returns:
        The loaded segment table.

    Raises:
        FileNotFoundError: Nothing is at the resolved path.
        ValueError: A required column is missing, or graph_hash is given and does not match the
            file's stamp.
    """
    src, archive = npz.read(path, schema.SEGMENTS_NPZ, MISSING_SEGMENTS)
    schema_keys = schema.ArtifactSchemas.SEGMENTS.keys
    missing = [k for k in schema_keys if k not in archive]
    if missing:
        raise ValueError(f"{src}: g_segments.npz missing column(s) {missing}. {MISSING_SEGMENTS}")
    fields = {
        name: np.asarray(archive[name]).astype(_DTYPE[code])
        for name, code in schema_keys.items()
    }
    stamped = str(archive["graph_hash"]) if "graph_hash" in archive else ""

    if graph_hash is not None and stamped != graph_hash:
        raise ValueError(
            f"{src}: graph_hash mismatch -- segments={stamped!r} graph={graph_hash!r}. This "
            "table was not built against THIS (NodeTable, EdgeSet) pair -- rebuild it via "
            "SegmentBuilder against this graph, or point at the artifacts it actually came from."
        )
    return SegmentTable(**fields)
