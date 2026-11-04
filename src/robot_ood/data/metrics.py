"""C2: shared learned pair metric and two-tier learned seed.

d(u, v) = MLP([||dq||, ||d ee_pos||, ||d ee_rot||, cos(qdot), dphase, dgrip, same_task, cos_DINO]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from robot_ood.data import schema

__all__ = [
    "N_METRIC_FEATURES",
    "METRIC_HIDDEN",
    "SEED_TIER1_M",
    "SIGMA_GUARD_DECADES",
    "ENTROPY_GUARD_NATS",
    "NodeMetricColumns",
    "QueryMetricColumns",
    "node_pair_features",
    "query_node_features",
    "ee_pose_delta_norm",
    "C2Metric",
    "C2EdgeWeights",
    "LearnedSeed",
    "SeedCandidates",
    "SeedGuardReport",
    "tier1_recall",
]

N_METRIC_FEATURES = 8
METRIC_HIDDEN = 32
SEED_TIER1_M = schema.SEED_TIER1_M  # re-exported so robot_ood.core.config can default without importing torch

SIGMA_GUARD_DECADES = 1.0      # sigma must stay within one order of magnitude of the C0 value
ENTROPY_GUARD_NATS = 0.5       # H(a) may not exceed the backbone-only seed's entropy by more

_EPS = 1e-8
_NEG_INF = float("-inf")


# ==== the columns the metric reads ====
@dataclass(frozen=True, eq=False)
class NodeMetricColumns:
    """Per-node columns C2 reads, beyond the normalised configs the seed backbone already holds.

    Attributes:
        qdot: [V, D] raw node velocities; direction only, never magnitude.
        phase: [V] per-node phase in [0, 1].
        grip: [V] per-node gripper flag.
        task_id: [V] int64 task ids, or None to zero-fill the same_task slot.
        ee: [V, 6] base-frame concat(position, axis-angle), or None to zero-fill both ee slots.
    """

    qdot: Tensor
    phase: Tensor
    grip: Tensor
    task_id: Tensor | None = None
    ee: Tensor | None = None

    @classmethod
    def from_node_table(cls, nodes, device: torch.device, *, ee: Tensor | None = None) -> "NodeMetricColumns":
        """Read the columns off a NodeTable.

        Args:
            nodes: The graph's robot_ood.data.nodes.NodeTable.
            device: Device the columns go on.
            ee: Optional [V, 6] base-frame poses; None leaves both ee slots zero-filled rather than
                triggering the forward-kinematics pass NodeTable.ee_base would run.

        Returns:
            The columns, all resident on device.
        """
        def col(arr, dtype=torch.float32) -> Tensor:
            return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)

        return cls(
            qdot=col(nodes.qdot), phase=col(nodes.phase), grip=col(nodes.grip),
            task_id=col(nodes.task_id, torch.int64), ee=ee,
        )


@dataclass(frozen=True, eq=False)
class QueryMetricColumns:
    """The query half of the seed's pair features.

    Attributes:
        qdot: [..., H, D] per-row query velocities.
        grip: [..., H] per-row gripper flag.
        ee: [..., H, 6] base-frame query poses, or None to zero-fill both ee slots.
    """

    qdot: Tensor
    grip: Tensor
    ee: Tensor | None = None


def _unit(x: Tensor) -> Tensor:
    """L2-normalise along the last axis, with a floor so a zero row stays finite."""
    return x / (x.norm(dim=-1, keepdim=True) + _EPS)


def _ee_deltas(ee_u: Tensor | None, ee_v: Tensor | None) -> tuple[Tensor, Tensor] | None:
    """(||d position||, ||d axis-angle||) for a pair of [..., 6] pose stacks, or None if absent."""
    if ee_u is None or ee_v is None:
        return None
    return (ee_u[..., :3] - ee_v[..., :3]).norm(dim=-1), (ee_u[..., 3:] - ee_v[..., 3:]).norm(dim=-1)


def ee_pose_delta_norm(ee_a: Tensor, ee_b: Tensor) -> Tensor:
    """Norm of pose difference: ||ee_a - ee_b|| over position and axis-angle jointly.

    Args:
        ee_a: [..., 6] base-frame poses.
        ee_b: [..., 6] base-frame poses.

    Returns:
        [...] float, in unscaled metres and radians.
    """
    return (ee_a - ee_b).norm(dim=-1)


# Slot order is fixed across every call site; a site missing a feature zero-fills the slot
# here rather than reordering or shortening the vector.
def _stack_features(slots: list[Tensor | None], like: Tensor) -> Tensor:
    """Stack the fixed slot list into [..., N_METRIC_FEATURES], zero-filling every absent slot.

    Args:
        slots: Exactly N_METRIC_FEATURES entries, in the module docstring's order; None means
            "this site cannot supply this feature".
        like: Tensor whose shape/dtype/device the zero fill and the stack follow.

    Returns:
        [..., N_METRIC_FEATURES] float features.

    Raises:
        ValueError: The slot list is not N_METRIC_FEATURES long.
    """
    if len(slots) != N_METRIC_FEATURES:
        raise ValueError(f"expected {N_METRIC_FEATURES} metric feature slots, got {len(slots)}")
    zero = torch.zeros_like(like)
    return torch.stack([zero if s is None else s.to(like.dtype) for s in slots], dim=-1)


def node_pair_features(q_n: Tensor, cols: NodeMetricColumns, src: Tensor, dst: Tensor) -> Tensor:
    """Pair features for node-to-node pairs -- the edge-weight and fan-out sites.

    Args:
        q_n: [V, D] normalised node configs.
        cols: The graph's metric columns.
        src: [E] int64 source node ids.
        dst: [E] int64 destination node ids.

    Returns:
        [E, N_METRIC_FEATURES] features.
    """
    dq = (q_n[src] - q_n[dst]).norm(dim=-1)                                       # [E]
    ee = _ee_deltas(None if cols.ee is None else cols.ee[src],
                    None if cols.ee is None else cols.ee[dst])
    cos_v = (_unit(cols.qdot[src]) * _unit(cols.qdot[dst])).sum(-1)
    same_task = (
        None if cols.task_id is None
        else (cols.task_id[src] == cols.task_id[dst]).to(dq.dtype)
    )
    return _stack_features(
        [
            dq,
            None if ee is None else ee[0],
            None if ee is None else ee[1],
            cos_v,
            cols.phase[src] - cols.phase[dst],
            cols.grip[src] - cols.grip[dst],
            same_task,
            None,   # cos_DINO: a slot the trained metrics were built with, always zero
        ],
        like=dq,
    )


def query_node_features(
    qh_n: Tensor, query: QueryMetricColumns, q_n: Tensor, cols: NodeMetricColumns, idx: Tensor,
) -> Tensor:
    """Pair features between every window row and its tier-1 candidates.

    Args:
        qh_n: [..., H, D] normalised query window, row 0 oldest.
        query: The query's own velocity/grip (and optional pose) columns.
        q_n: [V, D] normalised node configs.
        cols: The graph's metric columns.
        idx: [..., H, M] candidate node ids, one set per window row.

    Returns:
        [..., H, M, N_METRIC_FEATURES] features.
    """
    q_cand = q_n[idx]                                                             # [..., H, M, D]
    dq = (qh_n[..., :, None, :] - q_cand).norm(dim=-1)                            # [..., H, M]
    ee = _ee_deltas(None if query.ee is None else query.ee[..., :, None, :],
                    None if cols.ee is None else cols.ee[idx])
    cos_v = (_unit(query.qdot)[..., :, None, :] * _unit(cols.qdot[idx])).sum(-1)
    return _stack_features(
        [
            dq,
            None if ee is None else ee[0],
            None if ee is None else ee[1],
            cos_v,
            None,                                                                 # query has no phase
            query.grip[..., :, None] - cols.grip[idx],
            None,                                                                 # query has no task
            None,                                                                 # query never sees vision
        ],
        like=dq,
    )


# ==== C2 -- the metric itself ====
class C2Metric(nn.Module):
    """The shared learned pair metric: d = MLP(features), w = softplus(-d).

    Args:
        hidden: Width of the single hidden layer.
        n_features: Feature-vector width.
    """

    def __init__(self, hidden: int = METRIC_HIDDEN, n_features: int = N_METRIC_FEATURES):
        super().__init__()
        self.n_features = int(n_features)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_features, int(hidden)), nn.ReLU(), nn.Linear(int(hidden), 1),
        )

    def distance(self, feats: Tensor) -> Tensor:
        """d(u, v) for a [..., n_features] feature stack -> [...]."""
        return self.mlp(feats).squeeze(-1)

    def weight(self, feats: Tensor) -> Tensor:
        """w(u, v) = softplus(-d) -- strictly positive, so an edge is never deleted, only silenced."""
        return F.softplus(-self.distance(feats))

    def forward(self, feats: Tensor) -> Tensor:
        """Alias of weight, so the module can be called directly."""
        return self.weight(feats)


# Weights EVERY relation, temporal edges included: pinning them at a flat weight would say
# next16 equals next1.
class C2EdgeWeights:
    """Replaces EdgeSet's default Gaussian weights with learned C2 metric weights.

    Args:
        metric: The trained metric.
        device: Device the feature build runs on; None uses the metric's own.
        chunk: Edges per feature-build block, bounding the [chunk, 8] temporary.
        ee: Optional [V, 6] base-frame node poses.
    """

    def __init__(
        self, metric: C2Metric, *, device: torch.device | None = None, chunk: int = 1 << 20,
        ee: Tensor | None = None,
    ) -> None:
        self.metric = metric
        self.device = device or next(metric.parameters()).device
        self.chunk = int(chunk)
        self.ee = ee

    def __call__(self, nodes, edges) -> np.ndarray:
        """Weight one edge list.

        Args:
            nodes: The NodeTable the edges index.
            edges: Anything carrying int src/dst arrays (robot_ood.data.edges.RelationEdges).

        Returns:
            [E] float32 weights.
        """
        cols = NodeMetricColumns.from_node_table(nodes, self.device, ee=self.ee)
        q_n = torch.as_tensor(np.asarray(nodes.q), dtype=torch.float32, device=self.device)
        src = torch.as_tensor(np.asarray(edges.src, dtype=np.int64), device=self.device)
        dst = torch.as_tensor(np.asarray(edges.dst, dtype=np.int64), device=self.device)
        out = torch.empty(src.shape[0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for lo in range(0, src.shape[0], self.chunk):
                hi = min(lo + self.chunk, src.shape[0])
                out[lo:hi] = self.metric.weight(
                    node_pair_features(q_n, cols, src[lo:hi], dst[lo:hi])
                )
        return out.cpu().numpy().astype(np.float32)


# ==== the learned seed ====
@dataclass(frozen=True, eq=False)
class SeedCandidates:
    """One tier-1 pass: which nodes survived, and the score that ranked them.

    Attributes:
        idx: [..., H, M] int64 candidate node ids, one set per window row.
        s1: [..., H, M] their tier-1 scores, carried into tier 2 so f gets a gradient.
    """

    idx: Tensor
    s1: Tensor


@dataclass(frozen=True, eq=False)
class SeedGuardReport:
    """One forward pass's answer to the two registered seed risks.

    Attributes:
        entropy: H(a) in nats, the learned seed's own entropy.
        entropy_backbone: H(a) with every learned term dropped -- the Gaussian backbone at tau 1,
            i.e. the seed as initialised. tau counts as a learned term: a large tau flattens the
            mixture just as surely as a metric that cancels the backbone does.
        entropy_gap: entropy - entropy_backbone; positive means the seed is flattening out.
        sigma: The learned bandwidth.
        sigma_init: The C0 value it was initialised from.
        decades: |log10(sigma / sigma_init)|.
    """

    entropy: float
    entropy_backbone: float
    entropy_gap: float
    sigma: float
    sigma_init: float
    decades: float

    @property
    def ok(self) -> bool:
        """Whether both registered thresholds hold."""
        return self.entropy_gap <= ENTROPY_GUARD_NATS and self.decades <= SIGMA_GUARD_DECADES

    def assert_ok(self) -> "SeedGuardReport":
        """Raise unless both thresholds hold.

        Returns:
            Self, so a caller can log the report it just checked.

        Raises:
            AssertionError: The seed collapsed flat, or sigma ballooned past the backbone. Both are
                stop-and-investigate signals, not results.
        """
        assert self.entropy_gap <= ENTROPY_GUARD_NATS, (
            f"seed entropy gap {self.entropy_gap:.4f} nats > {ENTROPY_GUARD_NATS}: the learned seed "
            "is flatter than its own Gaussian backbone, i.e. it is handing the whole localisation "
            "problem to message passing"
        )
        assert self.decades <= SIGMA_GUARD_DECADES, (
            f"sigma={self.sigma:.5g} is {self.decades:.2f} decades from the C0 value "
            f"{self.sigma_init:.5g} (> {SIGMA_GUARD_DECADES}): the backbone has gone constant and "
            "the MLP is bypassing it"
        )
        return self


class LearnedSeed(nn.Module):
    """Two-tier learned seed: tier-1 candidates via inner product, tier-2 scored by C2.

    Args:
        hidden: Node/query embedding width.
        metric: The shared C2 metric.
        sigma_init: Backbone bandwidth in normalised joint units, from C0.
    """

    def __init__(self, hidden: int, metric: C2Metric, sigma_init: float = schema.KERNEL_BW):
        super().__init__()
        self.hidden = int(hidden)
        self.metric = metric
        self.tier1 = nn.Linear(self.hidden, self.hidden)
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(sigma_init))))
        self.log_tau = nn.Parameter(torch.zeros(()))                     # tau init 1.0
        self.register_buffer("sigma_init", torch.tensor(float(sigma_init)))

    @property
    def sigma(self) -> Tensor:
        """The learned backbone bandwidth, always positive."""
        return self.log_sigma.exp()

    @property
    def tau(self) -> Tensor:
        """The learned per-row softmax temperature, always positive."""
        return self.log_tau.exp()

    def candidates(self, z_h: Tensor, n_v: Tensor, m: int,
                   mask: Tensor | None = None) -> "SeedCandidates":
        """Tier 1: s1(h, v) = <f(z_h), n_v> -> the top-m node ids per window row, and their scores.

        Args:
            z_h: [..., H, hidden] per-row query states.
            n_v: [V, hidden] cached node encodings.
            m: Candidates per row.
            mask: [V] bool nodes excluded from tier 1, or None.

        Returns:
            The candidate ids and their tier-1 scores, [..., H, min(m, V)] each.
        """
        s1 = self.tier1(z_h) @ n_v.transpose(-1, -2)                     # [..., H, V]
        if mask is not None:
            s1 = s1.masked_fill(mask, _NEG_INF)
        top = s1.topk(min(int(m), n_v.shape[0]), dim=-1)
        return SeedCandidates(idx=top.indices, s1=top.values)

    def score(
        self, qh_n: Tensor, query: QueryMetricColumns, q_n: Tensor, cols: NodeMetricColumns,
        cand: "SeedCandidates", *, backbone_only: bool = False,
    ) -> Tensor:
        """Tier 2: residual score over each row's own candidates.

        Args:
            qh_n: [..., H, D] normalised query window.
            query: The query's metric columns.
            q_n: [V, D] normalised node configs.
            cols: The graph's metric columns.
            cand: The tier-1 candidates.
            backbone_only: Drop both learned terms -- the entropy guard's reference.

        Returns:
            [..., H, M] scores.
        """
        feats = query_node_features(qh_n, query, q_n, cols, cand.idx)
        backbone = -feats[..., 0].pow(2) / (2.0 * self.sigma.pow(2))
        if backbone_only:
            return backbone
        # s1 is carried into tier 2, not discarded after selection: it is the only gradient
        # path the tier-1 projection gets.
        return backbone + cand.s1 - self.metric.distance(feats)

    def mixture(
        self, scores: Tensor, idx: Tensor, pool_w: Tensor, n_nodes: int, *, tau: float | None = None,
    ) -> Tensor:
        """a_v = sum_h pool_w_h * softmax_v(score(h, .) / tau), scattered back over V.

        Args:
            scores: [..., H, M] tier-2 scores.
            idx: [..., H, M] candidate ids.
            pool_w: [..., H] window pooling weights, summing to 1 over H.
            n_nodes: V.
            tau: Override for the learned temperature; the guard's reference pins it at 1.0.

        Returns:
            [..., V] seed mass, summing to 1 up to duplicate-candidate overlap.
        """
        p = torch.softmax(scores / (self.tau if tau is None else tau), dim=-1)   # [..., H, M]
        contrib = (pool_w[..., None] * p).reshape(*scores.shape[:-2], -1)
        flat_idx = idx.reshape(*idx.shape[:-2], -1)
        a = torch.zeros(*scores.shape[:-2], n_nodes, dtype=scores.dtype, device=scores.device)
        return a.scatter_add(-1, flat_idx, contrib)

    def forward(
        self, qh_n: Tensor, query: QueryMetricColumns, z_h: Tensor, pool_w: Tensor,
        q_n: Tensor, cols: NodeMetricColumns, n_v: Tensor, *, topk: int, tier1_m: int = SEED_TIER1_M,
        factor: Tensor | None = None, mask: Tensor | None = None,
    ) -> Tensor:
        """Seed amplitudes: two tiers, per-row mixture, additive log tilt, then top-k.

        Args:
            qh_n: [..., H, D] normalised query window.
            query: The query's metric columns.
            z_h: [..., H, hidden] per-row query states (tier-1 keys).
            pool_w: [..., H] window pooling weights.
            q_n: [V, D] normalised node configs.
            cols: The graph's metric columns.
            n_v: [V, hidden] cached node encodings.
            topk: Seed-set size; nodes outside it get exactly zero amplitude.
            tier1_m: Tier-1 width.
            factor: [..., V] strictly positive instruction prior, or None.
            mask: [V] bool nodes kept out of the seed entirely, or None.

        Returns:
            [..., V] amplitudes, zero off each row's own top-k.
        """
        cand = self.candidates(z_h, n_v, tier1_m, mask)
        scores = self.score(qh_n, query, q_n, cols, cand)
        a = self.mixture(scores, cand.idx, pool_w, n_v.shape[0])
        # No alive guard needed: a mixture of softmaxes has row max >= 1/M, so isfinite here
        # only catches a regression that reintroduces a float32 floor.
        assert torch.isfinite(a).all(), "learned seed produced a non-finite mass"

        logits = torch.where(a > 0, a.clamp_min(_EPS).log(), torch.full_like(a, _NEG_INF))
        if factor is not None:
            logits = logits + factor.clamp_min(_EPS).log()               # tilt is ADDITIVE in log space
        # Top-k runs AFTER the tilt, so the instruction prior changes WHICH nodes are seeded.
        k = min(int(topk), n_v.shape[0])
        if k < n_v.shape[0]:
            # Hard indices, soft weights -- the readout's own trick, reused; no new gradient path.
            sel = logits.topk(k, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sel, True)
            logits = torch.where(mask, logits, torch.full_like(logits, _NEG_INF))
        return logits.exp()

    def guard_report(
        self, qh_n: Tensor, query: QueryMetricColumns, z_h: Tensor, pool_w: Tensor,
        q_n: Tensor, cols: NodeMetricColumns, n_v: Tensor, *, tier1_m: int = SEED_TIER1_M,
    ) -> SeedGuardReport:
        """Measure both registered seed guards on one query.

        Args:
            qh_n: [..., H, D] normalised query window.
            query: The query's metric columns.
            z_h: [..., H, hidden] per-row query states.
            pool_w: [..., H] window pooling weights.
            q_n: [V, D] normalised node configs.
            cols: The graph's metric columns.
            n_v: [V, hidden] cached node encodings.
            tier1_m: Tier-1 width.

        Returns:
            The report; call assert_ok to enforce it.
        """
        with torch.no_grad():
            cand = self.candidates(z_h, n_v, tier1_m)
            learned = self.mixture(
                self.score(qh_n, query, q_n, cols, cand), cand.idx, pool_w, n_v.shape[0],
            )
            backbone = self.mixture(
                self.score(qh_n, query, q_n, cols, cand, backbone_only=True), cand.idx, pool_w,
                n_v.shape[0], tau=1.0,
            )
            h_learned, h_backbone = _entropy(learned), _entropy(backbone)
            sigma, sigma_init = float(self.sigma), float(self.sigma_init)
        return SeedGuardReport(
            entropy=h_learned, entropy_backbone=h_backbone,
            entropy_gap=h_learned - h_backbone, sigma=sigma, sigma_init=sigma_init,
            decades=abs(float(np.log10(sigma / sigma_init))),
        )


def _entropy(a: Tensor) -> float:
    """Mean Shannon entropy (nats) of a [..., V] non-negative mass, renormalised per row."""
    p = a / a.sum(-1, keepdim=True).clamp_min(_EPS)
    # xlogy, not p*log(p): in float32 a tail entry can be exactly 0, and 0*log(0) is NaN.
    return float(-torch.special.xlogy(p, p).sum(-1).mean())


def tier1_recall(
    seed: LearnedSeed, qh_n: Tensor, query: QueryMetricColumns, z_h: Tensor, pool_w: Tensor,
    q_n: Tensor, cols: NodeMetricColumns, n_v: Tensor, *, topk: int, tier1_m: int = SEED_TIER1_M,
) -> float:
    """Tier-1 recall: fraction of brute-force top-k that survives tier-1 pruning.

    Args:
        seed: The seed being measured.
        qh_n: [B, H, D] normalised query windows.
        query: Their metric columns.
        z_h: [B, H, hidden] per-row query states.
        pool_w: [B, H] window pooling weights.
        q_n: [V, D] normalised node configs.
        cols: The graph's metric columns.
        n_v: [V, hidden] cached node encodings.
        topk: Seed-set size to compare at.
        tier1_m: Tier-1 width under test.

    Returns:
        Mean per-query recall in [0, 1].
    """
    v = n_v.shape[0]
    with torch.no_grad():
        full = seed.candidates(z_h, n_v, v)
        exact = seed.mixture(seed.score(qh_n, query, q_n, cols, full), full.idx, pool_w, v)
        cand = seed.candidates(z_h, n_v, tier1_m)
        kept = torch.zeros(*exact.shape, dtype=torch.bool, device=exact.device)
        kept.scatter_(-1, cand.idx.reshape(*cand.idx.shape[:-2], -1), True)
    want = exact.topk(min(int(topk), v), dim=-1).indices
    return float(kept.gather(-1, want).to(torch.float64).mean())
