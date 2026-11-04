"""Readout arms -- collapse a retrieval distribution p(v|Q) into ONE concrete recovery target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from robot_ood.data import schema

if TYPE_CHECKING:
    from robot_ood.data.nodes import NodeTable

__all__ = [
    "Support",
    "weighted_topk",
    "ReadoutResult",
    "ReadoutContext",
    "read_out",
]

# ==== weighted_topk — mass-truncated support, shared by every aggregating arm ====
TOPK_K_MIN = 8
TOPK_K_MAX = 1024
TOPK_MASS = 0.99

_EPS = 1e-12


@dataclass(frozen=True, eq=False)
class Support:
    """A truncated, renormalised support over nodes.

    Attributes:
        idx: [K] int64 node indices, descending by probability.
        w: [K] float64 weights, summing to 1.
    """

    idx: torch.Tensor
    w: torch.Tensor

    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (idx, w) as host numpy arrays."""
        return self.idx.numpy(), self.w.numpy()


def weighted_topk(
    p: torch.Tensor, mass: float = TOPK_MASS, k_min: int = TOPK_K_MIN, k_max: int = TOPK_K_MAX
) -> Support:
    """Smallest-K support truncation over one query's distribution p(v|Q).

    Args:
        p: [V] float64 distribution over nodes.
        mass: Cumulative-mass target.
        k_min: Floor on the realised K.
        k_max: Ceiling on the realised K.

    Returns:
        The truncated support, sorted by descending probability.

    Raises:
        AssertionError: p is not 1-D.
    """
    assert p.dim() == 1, "weighted_topk expects one query's 1-D distribution over nodes"
    n = p.shape[0]
    # torch.sort, not np.argsort: tie-breaking changes which nodes enter the support, pinning q_star
    # bit-exactly in tests/golden/parity.json.
    sorted_p, sorted_idx = torch.sort(p, descending=True)
    frac = torch.cumsum(sorted_p, dim=0) / sorted_p.sum().clamp_min(_EPS)
    k_needed = int((frac < mass).sum().item()) + 1     # nodes required to first reach mass
    limit = min(k_max, n)
    k = min(max(k_needed, k_min), limit)
    if k < limit:
        # The contiguous run of entries equal to the last kept one; splitting it would pick
        # arbitrarily among indistinguishable nodes.
        tied = (sorted_p[k:limit] == sorted_p[k - 1]).to(torch.int64)
        k += int(tied.cumprod(dim=0).sum().item())
    w = sorted_p[:k]
    return Support(idx=sorted_idx[:k], w=w / w.sum().clamp_min(_EPS))


# ==== result / context ====
@dataclass(eq=False)
class ReadoutResult:
    """One arm's answer for one query.

    Attributes:
        q_star: [D] float64 target configuration.
        q_seg: [seg_k, D] float64 reference segment starting at the target.
        qdot_seg: [seg_k, D] float64.
        depth: Phase of the target — the recovery-depth readout.
        conf: Top-1 mass of the truncated support, filled by Readout.__call__.
        off_manifold: ||q_star - nearest demo config||, filled by Readout.__call__.
        info: Arm-specific telemetry.
    """

    q_star: np.ndarray
    q_seg: np.ndarray
    qdot_seg: np.ndarray
    depth: float
    conf: float = 0.0
    off_manifold: float = 0.0
    info: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.info is None:
            self.info = {}


@dataclass(eq=False)
class ReadoutContext:
    """Per-query resources shared by every arm.

    Attributes:
        nodes: The demonstration-graph node table.
        topm: k_max handed to weighted_topk.
        temp: p-tempering knob applied once in Readout.__call__.
        seg_k: Segment length in frames.
        q_now: Current arm config, the origin off_manifold is measured from.
    """

    nodes: NodeTable
    topm: int = 32
    temp: float = 1.0
    seg_k: int = schema.SEG_K
    q_now: Any = None

    def __post_init__(self) -> None:
        self._q = np.asarray(self.nodes.q, dtype=np.float64)
        self._phase = np.asarray(self.nodes.phase, dtype=np.float64)
        self._virtual = np.asarray(getattr(self.nodes, "is_virtual", 0), dtype=bool)
        self._virtual = np.broadcast_to(self._virtual, (self._q.shape[0],))

    @property
    def node_q(self) -> np.ndarray:
        """[V, D] float64 every node's configuration."""
        return self._q

    @property
    def is_virtual(self) -> np.ndarray:
        """[V] bool synthesised-node mask; never selectable, see Readout.__call__."""
        return self._virtual

    @property
    def node_phase(self) -> np.ndarray:
        """[V] float64 every node's phase."""
        return self._phase

    def segments(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gather several nodes' reference segments.

        Args:
            idx: [K] node indices.

        Returns:
            (q_seg, qdot_seg), each [K, seg_k, D] float64.
        """
        pairs = [self.nodes.segment(int(v), self.seg_k) for v in idx]
        q_seg = np.stack([np.asarray(a, dtype=np.float64) for a, _ in pairs], axis=0)
        qdot_seg = np.stack([np.asarray(b, dtype=np.float64) for _, b in pairs], axis=0)
        return q_seg, qdot_seg

    def nn_distance(self, q: np.ndarray) -> float:
        """||q - nearest demo config|| over every node.

        Args:
            q: [D] configuration.

        Returns:
            The distance, in radians.
        """
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        # Real nodes only: a synthesised node is a perturbation OFF the manifold, so measuring
        # against it would report the arm as on-manifold precisely when it is not.
        real = self._q[~self._virtual]
        return float(np.linalg.norm(real - q[None, :], axis=1).min())


def _temper(p: np.ndarray, temp: float) -> np.ndarray:
    """Raise p to the 1/temp power and renormalize.

    Args:
        p: [V] float64 normalised distribution.
        temp: Tempering knob.

    Returns:
        [V] float64 tempered distribution.

    Raises:
        ValueError: temp <= 0.
    """
    # temp == 1.0 is exactly the identity and the default.
    if temp == 1.0:
        return p
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    logp = np.log(np.clip(p, 1e-300, None)) / temp
    logp = logp - logp.max()
    q = np.exp(logp)
    return q / q.sum()


@dataclass(frozen=True, eq=False)
class _Aggregate:
    """The p-weighted constructions both arms build from the same truncated support.

    Attributes:
        idx: [K] int64 support node indices.
        q_anchor: [D] float64 mass-weighted mean configuration.
        depth: Mass-weighted phase.
        q_seg: [seg_k, D] float64 mass-weighted blended segment.
        qdot_seg: [seg_k, D] float64.
    """

    idx: np.ndarray
    q_anchor: np.ndarray
    depth: float
    q_seg: np.ndarray
    qdot_seg: np.ndarray


# ==== readout ====
def read_out(p: np.ndarray, ctx: ReadoutContext) -> ReadoutResult:
    """Collapse a retrieval distribution into one target: the mass-weighted mean config.

    Args:
        p: [V] raw retrieval distribution over nodes.
        ctx: Per-query resources.

    Returns:
        The target, with off_manifold and conf attached.

    Raises:
        ValueError: p has non-finite or non-positive total mass, or every node with mass is
            virtual -- which is a graph or seeding fault, not a target.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, None)
    # A virtual node may be seeded and may pass messages, but may never be the answer.
    p = np.where(ctx.is_virtual, 0.0, p)
    total = float(p.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError(
            "readout: p must be a finite, positive-mass distribution over REAL nodes "
            f"(sum={total} after the is_virtual mask)"
        )
    p = _temper(p / total, ctx.temp)
    agg = _aggregate(p, ctx)
    # The support's TOP-1 mass, not the mass it retained; recomputed off p, not the support's weights.
    head = np.sort(p)[::-1][:max(len(agg.idx), 1)]
    return ReadoutResult(
        q_star=agg.q_anchor,
        q_seg=agg.q_seg,
        qdot_seg=agg.qdot_seg,
        depth=agg.depth,
        info={"K": len(agg.idx)},
        off_manifold=ctx.nn_distance(agg.q_anchor),
        conf=float(head[0] / max(head.sum(), _EPS)),
    )


def _aggregate(p: np.ndarray, ctx: ReadoutContext) -> _Aggregate:
    """Truncate p and build the p-weighted anchor, depth, and blended segment.

    Args:
        p: [V] float64 normalised, tempered distribution.
        ctx: Per-query resources.

    Returns:
        The shared constructions.
    """
    idx, w = weighted_topk(torch.as_tensor(p, dtype=torch.float64), k_max=ctx.topm).numpy()
    q_seg_k, qdot_seg_k = ctx.segments(idx)
    return _Aggregate(
        idx=idx,
        q_anchor=(w[:, None] * ctx.node_q[idx]).sum(axis=0),
        depth=float(np.dot(w, ctx.node_phase[idx])),
        q_seg=np.tensordot(w, q_seg_k, axes=(0, 0)),
        qdot_seg=np.tensordot(w, qdot_seg_k, axes=(0, 0)),
    )
