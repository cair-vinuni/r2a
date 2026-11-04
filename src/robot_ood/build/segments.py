"""C1's segment-memory builder: g_nodes.npz + g_edges.npz -> g_segments.npz.

Reads an already-built graph directory and writes g_segments.npz beside it, stamped with the same
graph_hash convention g_head.npz uses, so a stale segment table refuses to load against a
graph it was not built from.
"""

from __future__ import annotations

import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from robot_ood.data.store import record_artifacts
from robot_ood.core.constants import CONSTANTS_JSON, Constants
from robot_ood.core.schemas import SEGMENT_LEN
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.geometry import graph_hash
from robot_ood.data.nodes import NodeTable
from robot_ood.data.segments import SegmentBuilder, save_segments

__all__ = ["SegmentBuildResult", "build_segments", "resolve_segment_len"]


@dataclass(frozen=True)
class SegmentBuildResult:
    """What one segment build wrote.

    Attributes:
        path: The g_segments.npz written.
        n_segments: Rows in the table.
        segment_len: H, raw frames per segment, as actually used.
        graph_hash: The stamp a loader checks this table against.
        seconds: Wall-clock time of the build.
    """

    path: Path
    n_segments: int
    segment_len: int
    graph_hash: str
    seconds: float

    def summary(self) -> str:
        """One line naming what was written and what it was stamped against."""
        return (f"{self.path}: S={self.n_segments:,} segment_len={self.segment_len} "
                f"t={self.seconds:.1f}s "
                f"graph_hash={self.graph_hash[:12]}...")


def resolve_segment_len(graph_dir: Path) -> int:
    """H for this graph: its own calibrated constants when it has them, else the contract default.

    Args:
        graph_dir: Directory holding the graph artifacts.

    Returns:
        Raw frames per segment.
    """
    constants_path = graph_dir / CONSTANTS_JSON
    if not constants_path.is_file():
        return SEGMENT_LEN
    return Constants.from_json(constants_path).int_value("segment_len", SEGMENT_LEN)


def build_segments(graph_dir: str | Path, *, segment_len: int | None = None,
                   stride: int | None = None) -> SegmentBuildResult:
    """Build g_segments.npz beside an existing graph and record it.

    Args:
        graph_dir: Directory holding g_nodes.npz and g_edges.npz.
        segment_len: H, raw frames per segment; None resolves it from the graph's own constants.
        stride: Real-node spacing between segment starts; None means every real node.

    Returns:
        What was written.
    """
    graph_dir = Path(graph_dir)
    nodes = NodeTable.load(graph_dir / schema.NODES_NPZ)
    edges = EdgeSet.load(graph_dir / schema.EDGES_NPZ, strict=True)
    segment_len = resolve_segment_len(graph_dir) if segment_len is None else int(segment_len)

    t0 = time.perf_counter()
    kwargs: dict = {"segment_len": segment_len}
    if stride is not None:
        kwargs["stride"] = stride
    table = SegmentBuilder(nodes, **kwargs).build()

    stamp = graph_hash(nodes, edges, constants=graph_dir / CONSTANTS_JSON)
    out = save_segments(table, graph_dir, graph_hash=stamp)
    record_artifacts(out.parent, stage="segments", command=shlex.join(sys.argv),
                     filenames=[schema.SEGMENTS_NPZ])
    return SegmentBuildResult(path=out, n_segments=table.n_segments, segment_len=segment_len,
                              graph_hash=stamp,
                              seconds=time.perf_counter() - t0)
