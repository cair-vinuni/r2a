"""Numpy-only geometric measurements over a built NodeTable.

CorpusGeometry.spacing reproduces torch/cdist result BIT FOR BIT by replaying |a-b|^2 = |a|^2 + |b|^2 - 2a.b in float32; any other algorithm drifts ~1.3e-6 rad.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import numpy.typing as npt

from robot_ood.data import schema
from robot_ood.data.nodes import NodeTable

__all__ = [
    "CorpusGeometry",
    "measure_inter_demo_spacing",
]

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]


class CorpusGeometry:
    """Leave-one-demo-out geometric measurements over one NodeTable.

    Args:
        nodes: Built node table.
        block_size: Row-block size for the chunked distance matmul.

    Raises:
        ValueError: Fewer than 2 demos, so no leave-one-demo-out floor exists.
    """

    def __init__(self, nodes: NodeTable, *, block_size: int = schema.CDIST_BS) -> None:
        if nodes.n_demos < 2:
            raise ValueError(
                "measure_inter_demo_spacing: need >= 2 demos for a leave-one-demo-out floor"
            )

        self.block_size = int(block_size)
        # Normalised once here, never re-cast in a measurement path.
        self._q32: F32 = np.asarray(nodes.q, dtype=np.float32)
        self._owner: I64 = np.asarray(nodes.owner, dtype=np.int64)

    # ---- public measurements -------------------------------------------------------------------

    @cached_property
    def spacing(self) -> float:
        """Median leave-one-demo-out nearest-neighbour distance between ALL nodes, in rad."""
        return float(np.median(self._lodo_floors(self._q32, self._owner)))

    def _lodo_floors(self, points: F32, point_owner: I64) -> F64:
        """Nearest-neighbour distance from each point to the same set, excluding its own demo.

        Args:
            points: [n, d] float32 configs.
            point_owner: [n] demo id per row.

        Returns:
            [n] float64 leave-one-demo-out nearest-neighbour distances, in rad.
        """
        n = points.shape[0]
        floors = np.empty(n, dtype=np.float64)
        points_sq = np.sum(points * points, axis=1, dtype=np.float32)

        for i in range(0, n, self.block_size):
            block = points[i : i + self.block_size]
            block_sq = points_sq[i : i + self.block_size]
            d2 = block_sq[:, None] + points_sq[None, :] - np.float32(2.0) * (block @ points.T)
            np.maximum(d2, np.float32(0.0), out=d2)              # tiny negative round-off
            d = np.sqrt(d2, dtype=np.float32)
            d[point_owner[None, :] == point_owner[i : i + self.block_size, None]] = np.float32(np.inf)
            floors[i : i + self.block_size] = d.min(axis=1).astype(np.float64)

        return floors

# ==== module-level entry points ====


def measure_inter_demo_spacing(nodes: NodeTable) -> float:
    """Median leave-one-demo-out nearest-neighbour distance (rad) between nodes.

    Args:
        nodes: Built node table.

    Returns:
        INTER_DEMO_SPACING in radians.

    Raises:
        ValueError: Fewer than 2 demos.
    """
    return CorpusGeometry(nodes).spacing
