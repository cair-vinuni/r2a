"""Geometry and shared train/deploy plumbing for the demonstration graph.

One implementation each of finite_diff_vel, edge_pad_hist, off_manifold_distance.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from robot_ood.core.device import resolve_device
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable

__all__ = [
    "entry_perturbation_floor",
    "NeighborIndex",
    "PaddedHistory",
    "finite_diff_vel",
    "edge_pad_hist",
    "off_manifold_distance",
    "leave_one_demo_out_floors",
    "graph_hash",
    "CONSTANTS_JSON",
]

# C0's output, written beside g_nodes.npz and hashed into graph_hash; re-exported because the
# graph modules read it from here.
from robot_ood.core.constants import CONSTANTS_JSON

# Fields hashed into graph_hash. w is in the list: under C2 it is a LEARNED column, not a pure
# function of the geometry, so a geometry-only hash would call two different graphs the same one.
_HASH_NODE_FIELDS = ("q", "qdot", "owner", "task_id", "stage", "t_idx", "phase")
_HASH_EDGE_FIELDS = ("src", "dst", "rel", "indptr", "w")
_HASH_VERSION = b"graph_hash_v2"
_HASH_ABSENT = b"<absent>"

# Floor under every user-supplied divisor, so a degenerate 0 cannot turn a weight into inf/nan.
_EPS = 1e-8


class NeighborIndex:
    """Blocked nearest-neighbour queries against a fixed corpus of joint configs."""

    def __init__(
        self,
        corpus: np.ndarray,
        corpus_owner: Optional[np.ndarray] = None,
        block_size: Optional[int] = None,
    ) -> None:
        """Initializes with corpus and optional owner information.

        Args:
            corpus: [M, D] reference configs; cast to float32.
            corpus_owner: Optional [M] demo ids, required for owner-excluded queries.
            block_size: Query rows per cdist call; defaults to schema.CDIST_BS.
        """
        # Resolved once here, deliberately not plumbed through the module-level helpers:
        # CPU and CUDA disagree by up to ~1e-4 rad elementwise, enough to move a training label.
        self._device = resolve_device(None)
        self._block_size = int(block_size if block_size is not None else schema.CDIST_BS)
        self._corpus = torch.as_tensor(
            np.asarray(corpus, dtype=np.float32), device=self._device
        )
        self._owner: Optional[torch.Tensor] = None
        if corpus_owner is not None:
            self._owner = torch.as_tensor(
                np.asarray(corpus_owner, dtype=np.int64), device=self._device
            )

    @classmethod
    def from_nodes(cls, nodes: NodeTable, with_owner: bool = False) -> "NeighborIndex":
        """Build an index over a graph's real node configs (excluding synthesised nodes).

        Args:
            nodes: Source node table.
            with_owner: Attach nodes.owner to enable owner-excluded queries.

        Returns:
            A ready-to-query index.
        """
        n_real = int(np.asarray(nodes.demo_ptr)[-1])
        owner = np.asarray(nodes.owner[:n_real], dtype=np.int64) if with_owner else None
        return cls(np.asarray(nodes.q[:n_real], dtype=np.float32), owner)

    def __len__(self) -> int:
        return int(self._corpus.shape[0])

    def min_distance(
        self, points: np.ndarray, point_owner: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """L2 distance from each query row to its nearest corpus row.

        Args:
            points: [N, D] (or [D]) query configs; cast to float32.
            point_owner: Optional [N] demo ids; corpus rows with a matching owner are excluded
                from that row's minimum.

        Returns:
            [N] float64 distances, in radians.

        Raises:
            ValueError: point_owner given but the index carries no corpus owners.
        """
        query = np.atleast_2d(np.asarray(points, dtype=np.float32))
        n = query.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        if point_owner is not None and self._owner is None:
            raise ValueError("owner-excluded query requires a corpus_owner-bearing index")

        query_t = torch.as_tensor(query, device=self._device)
        owner_t: Optional[torch.Tensor] = None
        if point_owner is not None:
            owner_t = torch.as_tensor(
                np.asarray(point_owner, dtype=np.int64), device=self._device
            )

        out = np.empty(n, dtype=np.float64)
        for start in range(0, n, self._block_size):
            stop = start + self._block_size
            # Direct mode, not the matmul expansion: a floor small against the configs' own
            # magnitude cancels to exactly 0 in float32 under ||a||^2 + ||b||^2 - 2ab.
            dist = torch.cdist(query_t[start:stop], self._corpus,
                               compute_mode="donot_use_mm_for_euclid_dist")
            if owner_t is not None:
                excluded = self._owner[None, :] == owner_t[start:stop, None]
                dist = dist.masked_fill(excluded, float("inf"))
            out[start:stop] = dist.min(dim=1).values.double().cpu().numpy()
        return out


def finite_diff_vel(q_hist: np.ndarray) -> np.ndarray:
    """Finite differences edge-extrapolated at t=0, never zero-padded; axis -2 is time."""
    q_hist = np.asarray(q_hist, dtype=np.float32)
    v = np.zeros_like(q_hist)
    t = q_hist.shape[-2]
    if t > 1:
        v[..., 1:, :] = q_hist[..., 1:, :] - q_hist[..., :-1, :]
        v[..., 0, :] = v[..., 1, :]
    return v


def off_manifold_distance(nodes: NodeTable, q_last: np.ndarray) -> float:
    """Nearest-node L2 distance to any real node, float64 numpy (pinned in parity.json).

    Args:
        nodes: Graph node table.
        q_last: [D] joint config.

    Returns:
        The distance in radians.
    """
    q_last = np.asarray(q_last, dtype=np.float64).reshape(-1)
    q = nodes.q[: int(np.asarray(nodes.demo_ptr)[-1])].astype(np.float64)
    return float(np.min(np.linalg.norm(q - q_last[None, :], axis=1)))


@dataclass(frozen=True, slots=True)
class PaddedHistory:
    """edge_pad_hist's result.

    Attributes:
        window: [hist_len, ...] right-aligned array.
        was_padded: True when the input was shorter than hist_len and was edge-padded.
    """

    window: np.ndarray
    was_padded: bool

    def __iter__(self):
        """Yields window then was_padded."""
        yield self.window
        yield self.was_padded

    def __getitem__(self, i: int):
        """[0] is window, [1] is was_padded."""
        return (self.window, self.was_padded)[i]


def edge_pad_hist(arr: np.ndarray, hist_len: int) -> PaddedHistory:
    """Right-align arr to exactly hist_len rows: truncate if longer, edge-pad if shorter.

    Args:
        arr: [T, ...] input, oldest row first.
        hist_len: Target row count.

    Returns:
        The normalised window and whether padding occurred.
    """
    arr = np.asarray(arr)
    n = arr.shape[0]
    padded = n < hist_len
    if padded:
        pad_n = hist_len - n
        arr = np.concatenate([np.repeat(arr[:1], pad_n, axis=0), arr], axis=0)
    elif n > hist_len:
        arr = arr[-hist_len:]
    return PaddedHistory(arr, padded)


def graph_hash(
    nodes: NodeTable, edges: EdgeSet, *, constants: Any = None, c2_state: Any = None,
) -> str:
    """Deterministic content fingerprint of (graph, constants, C2) triple.

    Args:
        nodes: The node table.
        edges: Its edge set.
        constants: C0's constants.json, as a path or as raw bytes/str; None when the graph shipped
            none.
        c2_state: The C2 checkpoint, as a mapping of key to array (see
            robot_ood.models.retrieval.gnn.GraphRetrieverNet.c2_state); None when the head carries no C2.

    Returns:
        The hex digest.
    """
    h = hashlib.sha256()
    h.update(_HASH_VERSION)
    h.update(f"V={len(nodes)}|E={len(edges.src)}|dim={nodes.dim}".encode())
    for name in _HASH_NODE_FIELDS:
        h.update(np.ascontiguousarray(getattr(nodes, name)).tobytes())
    for name in _HASH_EDGE_FIELDS:
        h.update(np.ascontiguousarray(getattr(edges, name)).tobytes())
    h.update(b"|constants|")
    h.update(_constants_bytes(constants))
    h.update(b"|c2|")
    h.update(_c2_bytes(c2_state))
    return h.hexdigest()


def _constants_bytes(constants: Any) -> bytes:
    """C0's constants.json as bytes: a path is read, bytes/str pass through, None is the marker."""
    if constants is None:
        return _HASH_ABSENT
    if isinstance(constants, (bytes, bytearray)):
        return bytes(constants)
    if isinstance(constants, str) and not os.path.exists(constants):
        return constants.encode()
    path = os.fspath(constants)
    return _HASH_ABSENT if not os.path.exists(path) else open(path, "rb").read()


def _c2_bytes(c2_state: Any) -> bytes:
    """The C2 checkpoint as bytes: keys in sorted order, each followed by its array's raw buffer."""
    if c2_state is None:
        return _HASH_ABSENT
    parts: list[bytes] = []
    for key in sorted(c2_state):
        parts.append(key.encode())
        parts.append(np.ascontiguousarray(c2_state[key]).tobytes())
    return _HASH_ABSENT if not parts else b"|".join(parts)


def entry_perturbation_floor(nodes: NodeTable) -> dict[str, float]:
    """Leave-one-demo-out 1-NN distance (rad) from each raw frame to the other demos' frames.

    Args:
        nodes: Node table providing raw_ptr and q_raw.

    Returns:
        {"p50": ..., "p90": ..., "p99": ...} in radians, over the frames no other demo repeats.

    Raises:
        ValueError: Fewer than two demos, or a corpus whose every frame coincides with another
            demo's, which leaves nothing to perturb by.
    """
    raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
    q_raw = np.asarray(nodes.q_raw, dtype=np.float64)
    j_demos = len(raw_ptr) - 1
    if j_demos < 2:
        raise ValueError("entry_perturbation_floor: need >= 2 demos for a leave-one-demo-out floor")
    owner_raw = np.repeat(np.arange(j_demos, dtype=np.int64), np.diff(raw_ptr))
    floors = leave_one_demo_out_floors(q_raw, owner_raw, q_raw, owner_raw)
    # A frame another demo repeats verbatim (a parked arm, a shared home pose) says nothing about
    # how far apart the demonstrations run.
    floors = floors[floors > 0.0]
    if floors.size == 0:
        raise ValueError("entry_perturbation_floor: every raw frame coincides with another demo's frame")
    return {
        "p50": float(np.percentile(floors, 50)), "p90": float(np.percentile(floors, 90)),
        "p99": float(np.percentile(floors, 99)),
    }


def leave_one_demo_out_floors(
    points: np.ndarray, point_owner: np.ndarray, corpus: np.ndarray, corpus_owner: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour distance from each point to the corpus, excluding own demo.

    Args:
        points: [N, D] query configs.
        point_owner: [N] demo ids for points.
        corpus: [M, D] reference configs.
        corpus_owner: [M] demo ids for corpus.

    Returns:
        [N] float64 floors, in radians.
    """
    return NeighborIndex(corpus, corpus_owner).min_distance(points, point_owner)
