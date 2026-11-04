"""Edge construction for the demonstration graph over the R=13 relation vocabulary.

Relation builders are pure functions of (NodeTable, params) -> RelationEdges.
EdgeSet is the only stateful object: it concatenates relations, derives weights
and the retrieval IDF prior, sorts to CSR-by-dst, and owns save/load/validate.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, fields
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterator, Protocol, Sequence

import numpy as np
import torch

from robot_ood.core import npz
from robot_ood.core.device import resolve_device as _resolve_device
from robot_ood.data import schema
from robot_ood.data.binning import phase_bins

if TYPE_CHECKING:
    from robot_ood.data.nodes import NodeTable

DeviceLike = torch.device | str | int | None

_I4 = np.int32
_I1 = np.int8
_F4 = np.float32

# What to tell a caller whose g_edges.npz is missing.
MISSING_EDGES = ("no edge set. Build the graph with `python -m robot_ood.cli build graph --suite <suite>`.")


class StepLogger(Protocol):
    """Minimal logging surface consumed by EdgeSet.build."""

    def step(self, label: str, detail: str) -> None: ...


# ==== data structures ====
@dataclass(slots=True)
class RelationEdges:
    """A directed edge list for one or more relations, in global node ids.

    Attributes:
        src: int64 source node ids, shape [E].
        dst: int64 destination node ids, shape [E].
        rel: int64 relation ids into schema.RELATIONS, shape [E].
    """

    src: np.ndarray
    dst: np.ndarray
    rel: np.ndarray

    def __len__(self) -> int:
        return int(len(self.src))

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yields src, dst, rel so tuple unpacking keeps working."""
        yield self.src
        yield self.dst
        yield self.rel

    @classmethod
    def empty(cls) -> RelationEdges:
        """Return a zero-edge instance.

        Returns:
            An empty RelationEdges.
        """
        z = np.zeros(0, dtype=np.int64)
        return cls(src=z, dst=z.copy(), rel=z.copy())

    @classmethod
    def concat(cls, parts: Sequence[RelationEdges]) -> RelationEdges:
        """Concatenate edge lists.

        Args:
            parts: Edge lists to join, in order.

        Returns:
            A single RelationEdges holding every edge in parts.
        """
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        return cls(
            src=np.concatenate([p.src for p in parts]).astype(np.int64),
            dst=np.concatenate([p.dst for p in parts]).astype(np.int64),
            rel=np.concatenate([p.rel for p in parts]).astype(np.int64),
        )


@dataclass(slots=True)
class GraphTensors:
    """Device-resident tensor view of an EdgeSet.

    Attributes:
        src: int32 source ids, dst-sorted (storage order), shape [E].
        dst: int32 destination ids, dst-sorted, shape [E].
        rel: int64 relation ids, dst-sorted, shape [E].
        w: float32 edge weights, dst-sorted, shape [E].
        log_idf: float32 per-node retrieval prior, shape [V].
        src_by_rel: src permuted into relation-sorted order, shape [E].
        dst_by_rel: dst permuted into relation-sorted order, shape [E].
        w_by_rel: w permuted into relation-sorted order, shape [E].
        rel_offsets: int64 CSR offsets over the fixed relation vocabulary,
            shape [N_RELATIONS + 1]; slice [r]:[r+1] selects relation r
            inside the *_by_rel arrays. Empty slices are legal -- a single-task
            graph really does have zero align edges.
    """

    src: torch.Tensor
    dst: torch.Tensor
    rel: torch.Tensor
    w: torch.Tensor
    log_idf: torch.Tensor
    src_by_rel: torch.Tensor
    dst_by_rel: torch.Tensor
    w_by_rel: torch.Tensor
    rel_offsets: torch.Tensor

    def __getitem__(self, key: str) -> torch.Tensor:
        """Looks a tensor up by field name.

        Args:
            key: Field name.

        Returns:
            The named tensor.

        Raises:
            KeyError: If key is not a field of this class.
        """
        if key not in {f.name for f in fields(self)}:
            raise KeyError(key)
        return getattr(self, key)


# ==== kNN kernel ====
def knn_masked(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int,
    block: torch.Tensor | None = None,
    bs: int = schema.CDIST_BS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched cdist + topk nearest-neighbour search with a pre-topk mask.

    Args:
        X: Query points, shape [N, D].
        Y: Candidate points, shape [M, D].
        k: Neighbours per query; clamped to M.
        block: Optional bool mask [N, M]; True marks an illegal neighbour.
        bs: Row-block size bounding peak cdist memory.

    Returns:
        (idx, dist), both [N, min(k, M)]. Slots with fewer than k legal
        candidates come back with dist == inf and must be dropped by the caller.
    """
    n, m = X.shape[0], Y.shape[0]
    k = int(min(k, m))
    if n == 0 or k <= 0:
        return (
            torch.zeros((n, max(k, 0)), dtype=torch.int64, device=X.device),
            torch.zeros((n, max(k, 0)), dtype=torch.float32, device=X.device),
        )

    idx_parts: list[torch.Tensor] = []
    dist_parts: list[torch.Tensor] = []
    for i in range(0, n, bs):
        d = torch.cdist(X[i:i + bs], Y)
        if block is not None:
            # NOTE: masking BEFORE topk is exact. Over-fetching then filtering starves
            # any row whose neighbourhood is dominated by illegal candidates.
            d = d.masked_fill(block[i:i + bs], float("inf"))
        dv, iv = torch.topk(d, k, dim=1, largest=False)
        idx_parts.append(iv)
        dist_parts.append(dv.to(torch.float32))
    return torch.cat(idx_parts, dim=0), torch.cat(dist_parts, dim=0)


def _dedup(src: np.ndarray, dst: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop duplicate (src, dst) pairs.

    Args:
        src: Source ids, shape [E].
        dst: Destination ids, shape [E].
        n_nodes: Node count, used as the pair-encoding radix.

    Returns:
        Deduplicated (src, dst).
    """
    if len(src) == 0:
        return src, dst
    code = src.astype(np.int64) * n_nodes + dst.astype(np.int64)
    _, keep = np.unique(code, return_index=True)
    return src[keep], dst[keep]


def _sym_dedup(src: np.ndarray, dst: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Add the reverse of every edge, then deduplicate.

    Args:
        src: Source ids, shape [E].
        dst: Destination ids, shape [E].
        n_nodes: Node count, used as the pair-encoding radix.

    Returns:
        Symmetrised, deduplicated (src, dst).
    """
    if len(src) == 0:
        return src, dst
    return _dedup(np.concatenate([src, dst]), np.concatenate([dst, src]), n_nodes)


def _grip_key(base: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """Pack a label and the binary gripper state into a single bucket key.

    Args:
        base: int64 labels, shape [V].
        grip: Binary gripper state (0/1), shape [V].

    Returns:
        int64 keys unique per (base, grip) pair, shape [V].
    """
    return base.astype(np.int64) * 2 + grip.astype(np.int64)


# The shared engine behind sibling_edges (bucket task+grip, block owner) and align_edges
# (bucket phase_bin+grip, block task): swapping bucket/block between them changes the relation.
def _bucketed_knn(
    q: np.ndarray,
    bucket_key: np.ndarray,
    block_key: np.ndarray,
    k: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """kNN restricted to same-bucket_key pools, masking same-block_key pairs.

    Args:
        q: Joint-space coordinates, shape [V, D].
        bucket_key: int64 pool assignment per node; nodes in different pools are
            never candidates for one another.
        block_key: int64 label per node; equal labels are masked before the topk.
        k: Neighbours per node within its pool.
        device: Torch device for the distance computation.

    Returns:
        Raw (pre-symmetrise, pre-dedup) (src, dst) in global node ids.
    """
    order = np.argsort(bucket_key, kind="stable")
    cuts = np.flatnonzero(np.diff(bucket_key[order]) != 0) + 1

    q_dev = torch.as_tensor(q, dtype=torch.float32, device=device)
    block_dev = torch.as_tensor(block_key.astype(np.int64), device=device)

    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    for pool in np.split(order, cuts):
        if len(pool) < 2:
            continue
        sel = torch.as_tensor(pool, dtype=torch.int64, device=device)
        points = q_dev.index_select(0, sel)
        labels = block_dev.index_select(0, sel)
        block = labels[:, None] == labels[None, :]  # also masks the diagonal
        kk = min(k, len(pool))

        nn_idx, nn_dist = knn_masked(points, points, kk, block=block)
        rows = torch.arange(len(pool), device=device).repeat_interleave(kk)
        valid = torch.isfinite(nn_dist.reshape(-1))
        src_parts.append(sel.index_select(0, rows[valid]))
        dst_parts.append(sel.index_select(0, nn_idx.reshape(-1)[valid]))

    if not src_parts:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy()
    # single device sync for the whole relation
    src = torch.cat(src_parts).cpu().numpy()
    dst = torch.cat(dst_parts).cpu().numpy()
    return src, dst


# ==== relation builders -- pure functions of (NodeTable, params) ====
def temporal_edges(
    nodes: NodeTable, dilations: tuple[int, ...] = schema.DILATIONS
) -> RelationEdges:
    """Build next^d / prev^d edges within each demo.

    Args:
        nodes: Node table supplying demo_ptr and t_idx.
        dilations: Temporal strides to emit.

    Returns:
        The temporal edge list.
    """
    demo_ptr = np.asarray(nodes.demo_ptr)
    t_idx_all = np.asarray(nodes.t_idx)

    parts: list[RelationEdges] = []
    for j in range(len(demo_ptr) - 1):
        lo, hi = int(demo_ptr[j]), int(demo_ptr[j + 1])
        n = hi - lo
        if n < 2:
            continue
        tid = t_idx_all[lo:hi].astype(np.int64)
        for d in dilations:
            if n <= d:
                continue
            # NOTE: matched on t_idx, not position -- a demo's coarse t_idx may have gaps.
            target = tid + d
            pos = np.searchsorted(tid, target)
            valid = pos < n
            valid[valid] &= tid[pos[valid]] == target[valid]
            if not valid.any():
                continue
            g_src = lo + np.nonzero(valid)[0]
            g_dst = lo + pos[valid]
            n_e = len(g_src)
            parts.append(RelationEdges(
                src=g_src, dst=g_dst,
                rel=np.full(n_e, schema.REL_INDEX[f"next{d}"], dtype=np.int64),
            ))
            parts.append(RelationEdges(
                src=g_dst, dst=g_src,
                rel=np.full(n_e, schema.REL_INDEX[f"prev{d}"], dtype=np.int64),
            ))
    return RelationEdges.concat(parts)


def sibling_edges(
    nodes: NodeTable, k: int = schema.K_SIBLING, device: DeviceLike = None
) -> RelationEdges:
    """Build sibling edges: different demo, same arm, same task, same gripper state.

    Args:
        nodes: Node table supplying q, task_id, grip, owner.
        k: Neighbours per node within its (task, grip) pool.
        device: Torch device for the distance computation.

    Returns:
        The symmetrised, deduplicated sibling edge list.
    """
    n = _n_real(nodes)
    if n == 0:
        return RelationEdges.empty()
    bucket = _grip_key(_arm_key(nodes, np.asarray(nodes.task_id)[:n]), np.asarray(nodes.grip)[:n])
    owner = np.asarray(nodes.owner)[:n].astype(np.int64)

    src, dst = _bucketed_knn(np.asarray(nodes.q)[:n], bucket, owner, k, _resolve_device(device))
    src, dst = _sym_dedup(src, dst, n)
    return RelationEdges(
        src=src, dst=dst,
        rel=np.full(len(src), schema.REL_INDEX["sibling"], dtype=np.int64),
    )


def align_edges(
    nodes: NodeTable,
    k: int = schema.K_ALIGN,
    nbins: int = schema.NBINS_ALIGN,
    device: DeviceLike = None,
) -> RelationEdges:
    """Build align edges: different task, same arm, same phase bin, same gripper state.

    Args:
        nodes: Node table supplying q, task_id, grip, phase.
        k: Neighbours per node within its (phase_bin, grip) pool.
        nbins: Phase-bin count.
        device: Torch device for the distance computation.

    Returns:
        The symmetrised, deduplicated align edge list.
    """
    n = _n_real(nodes)
    if n == 0:
        return RelationEdges.empty()
    pbin = phase_bins(np.asarray(nodes.phase)[:n], nbins)
    bucket = _grip_key(_arm_key(nodes, pbin), np.asarray(nodes.grip)[:n])
    task_id = np.asarray(nodes.task_id)[:n].astype(np.int64)

    src, dst = _bucketed_knn(np.asarray(nodes.q)[:n], bucket, task_id, k, _resolve_device(device))
    src, dst = _sym_dedup(src, dst, n)
    return RelationEdges(
        src=src, dst=dst,
        rel=np.full(len(src), schema.REL_INDEX["align"], dtype=np.int64),
    )


def _bimanual(nodes: NodeTable, n: int) -> bool:
    """Whether the table carries both arms of a robot."""
    return bool(_arm_of(nodes, n).max(initial=0) > 0)


def _arm_of(nodes: NodeTable, n: int) -> np.ndarray:
    """[n] i8 which arm each node belongs to; all zeros for a table that predates the column."""
    arm = getattr(nodes, "arm", None)
    return np.zeros(n, dtype=np.int64) if arm is None else np.asarray(arm)[:n].astype(np.int64)


def _arm_key(nodes: NodeTable, base: np.ndarray) -> np.ndarray:
    """base, made unique per arm, so a within-arm relation never crosses the robot's two arms."""
    return base.astype(np.int64) * 2 + _arm_of(nodes, len(base))


def partner_edges(nodes: NodeTable) -> RelationEdges:
    """Build partner edges: the other arm's nodes that were recorded while this one was.

    Args:
        nodes: Node table supplying pair, arm, t_raw and n_members.

    Returns:
        The symmetrised, deduplicated partner edge list; empty without paired arms.
    """
    # Partners share an episode and an overlapping raw span: the only relation that says two
    # configurations were true of one robot at one moment.
    n = _n_real(nodes)
    pair = np.asarray(getattr(nodes, "pair", np.full(n, -1)))[:n].astype(np.int64)
    arm = _arm_of(nodes, n)
    if n == 0 or not np.any(pair >= 0):
        return RelationEdges.empty()
    # Spans are compared inside one episode, so they must be demo-local: t_raw counts from the
    # start of the whole corpus, and the two arms of an episode are two different demos.
    demo_start = np.asarray(nodes.raw_ptr)[np.asarray(nodes.owner)[:n].astype(np.int64)].astype(np.int64)
    t_raw = np.asarray(nodes.t_raw)[:n].astype(np.int64) - demo_start
    lo, hi = t_raw, t_raw + np.asarray(nodes.n_members)[:n].astype(np.int64) - 1
    ids = np.arange(n, dtype=np.int64)
    left = ids[(arm == 0) & (pair >= 0)]
    right = ids[(arm == 1) & (pair >= 0)]
    src_parts, dst_parts = [], []
    order = right[np.argsort(pair[right], kind="stable")]
    starts = np.searchsorted(pair[order], pair[left], side="left")
    ends = np.searchsorted(pair[order], pair[left], side="right")
    for node, start, end in zip(left, starts, ends):
        candidates = order[start:end]
        if len(candidates) == 0:
            continue
        # spans overlap when neither ends before the other begins
        hit = candidates[(lo[candidates] <= hi[node]) & (hi[candidates] >= lo[node])]
        src_parts.append(np.full(len(hit), node, dtype=np.int64))
        dst_parts.append(hit)
    if not src_parts:
        return RelationEdges.empty()
    src, dst = _sym_dedup(np.concatenate(src_parts), np.concatenate(dst_parts), n)
    return RelationEdges(
        src=src, dst=dst,
        rel=np.full(len(src), schema.REL_INDEX["partner"], dtype=np.int64),
    )


def cross_arm_edges(
    nodes: NodeTable, k: int = schema.K_SIBLING, device: DeviceLike = None
) -> RelationEdges:
    """Build cross-arm edges: the other arm's nearest configurations of the same task and grip.

    Args:
        nodes: Node table supplying q, task_id, grip, arm.
        k: Neighbours per node within its (task, grip) pool.
        device: Torch device for the distance computation.

    Returns:
        The symmetrised, deduplicated edge list; empty without both arms.
    """
    # Where partner says "at the same moment", this says "in the same posture".
    n = _n_real(nodes)
    if n == 0 or not _bimanual(nodes, n):
        return RelationEdges.empty()
    bucket = _grip_key(np.asarray(nodes.task_id)[:n], np.asarray(nodes.grip)[:n])
    arm = _arm_of(nodes, n)
    src, dst = _bucketed_knn(np.asarray(nodes.q)[:n], bucket, arm, k, _resolve_device(device))
    src, dst = _sym_dedup(src, dst, n)
    return RelationEdges(
        src=src, dst=dst,
        rel=np.full(len(src), schema.REL_INDEX["cross_arm"], dtype=np.int64),
    )


def cross_arm_align_edges(
    nodes: NodeTable, k: int = schema.K_ALIGN, device: DeviceLike = None
) -> RelationEdges:
    """Build cross-arm align edges: the other arm's nearest configurations of a DIFFERENT task.

    Args:
        nodes: Node table supplying q, task_id, grip, arm.
        k: Neighbours per node within its grip pool.
        device: Torch device for the distance computation.

    Returns:
        The symmetrised, deduplicated edge list; empty without both arms.
    """
    n = _n_real(nodes)
    if n == 0 or not _bimanual(nodes, n):
        return RelationEdges.empty()
    grip = np.asarray(nodes.grip)[:n].astype(np.int64)
    arm = _arm_of(nodes, n)
    task = np.asarray(nodes.task_id)[:n].astype(np.int64)
    src, dst = _bucketed_knn(np.asarray(nodes.q)[:n], grip, arm, k, _resolve_device(device))
    keep = task[src] != task[dst]        # the same-arm pairs are already masked by the block key
    src, dst = _sym_dedup(src[keep], dst[keep], n)
    return RelationEdges(
        src=src, dst=dst,
        rel=np.full(len(src), schema.REL_INDEX["cross_arm_align"], dtype=np.int64),
    )


def approach_edges(nodes: NodeTable) -> RelationEdges:
    """Build approach edges: virtual -> the real entry node each was perturbed from.

    Args:
        nodes: Node table supplying is_virtual, owner, t_idx and demo_ptr.

    Returns:
        The approach edge list; empty when the table has no virtual nodes.
    """
    virtual = _virtual_mask(nodes)
    src = np.nonzero(virtual)[0].astype(np.int64)
    if not len(src):
        return RelationEdges.empty()
    demo_ptr = np.asarray(nodes.demo_ptr).astype(np.int64)
    owner = np.asarray(nodes.owner).astype(np.int64)
    t_idx = np.asarray(nodes.t_idx).astype(np.int64)
    dst = demo_ptr[owner[src]] + t_idx[src]
    return RelationEdges(
        src=src, dst=dst,
        rel=np.full(len(src), schema.REL_INDEX["approach"], dtype=np.int64),
    )


def _n_real(nodes: NodeTable) -> int:
    """Length of the real-node prefix, which every demo-derived relation is built over.

    Args:
        nodes: Node table to read.

    Returns:
        demo_ptr[-1].
    """
    return int(np.asarray(nodes.demo_ptr)[-1])


def _virtual_mask(nodes: NodeTable) -> np.ndarray:
    """[V] bool virtual-node mask. A node table predating the column reads as all-real.

    Args:
        nodes: Node table to read.

    Returns:
        The mask.
    """
    flag = getattr(nodes, "is_virtual", None)
    if flag is None:
        return np.zeros(len(nodes), dtype=bool)
    return np.asarray(flag).astype(bool)


def node_idf(nodes: NodeTable, edges: RelationEdges) -> np.ndarray:
    """Compute the structural IDF prior log(1 + reached / reachable) per node.

    Args:
        nodes: Node table supplying task_id, grip, owner.
        edges: Full edge list; only sibling edges contribute.

    Returns:
        float32 per-node prior, shape [V].
    """
    n = len(nodes)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    owner = np.asarray(nodes.owner).astype(np.int64)
    max_owner = int(owner.max()) + 1
    bucket = _grip_key(np.asarray(nodes.task_id), np.asarray(nodes.grip))
    max_bucket = int(bucket.max()) + 1

    uniq_bo = np.unique(bucket * max_owner + owner)
    pool_size = np.bincount(uniq_bo // max_owner, minlength=max_bucket)
    n_strands = np.maximum(1, pool_size[bucket] - 1).astype(np.float64)  # exclude own strand

    sib = edges.rel == schema.REL_INDEX["sibling"]
    s, d = edges.src[sib], edges.dst[sib]
    if len(s):
        uniq_key = np.unique(s.astype(np.int64) * max_owner + owner[d])
        n_reached = np.bincount(uniq_key // max_owner, minlength=n).astype(np.float64)
    else:
        n_reached = np.zeros(n, dtype=np.float64)

    return np.log1p(n_reached / n_strands).astype(np.float32)


# ==== EdgeSet ====
@dataclass
class EdgeSet:
    """The full demonstration-graph edge list, CSR-sorted by dst.

    Attributes:
        src: int32 source ids, shape [E].
        dst: int32 destination ids, shape [E].
        rel: int8 relation ids, shape [E].
        w: float32 edge weights, shape [E].
        indptr: int32 CSR offsets over nodes, shape [V + 1].
        log_idf: float32 per-node retrieval prior, shape [V].
        n_nodes: Node count V.
    """

    src: np.ndarray
    dst: np.ndarray
    rel: np.ndarray
    w: np.ndarray
    indptr: np.ndarray
    log_idf: np.ndarray
    n_nodes: int

    # ---- construction ----------------------------------------------------------------
    @classmethod
    def build(
        cls,
        nodes: NodeTable,
        k_sib: int = schema.K_SIBLING,
        k_align: int = schema.K_ALIGN,
        dilations: tuple[int, ...] = schema.DILATIONS,
        device: DeviceLike = None,
        logger: StepLogger | None = None,
        metric: Callable[[NodeTable, RelationEdges], np.ndarray] | None = None,
    ) -> EdgeSet:
        """Build every relation, derive weights and log_idf, sort to CSR, validate.

        Args:
            nodes: Node table to build from.
            k_sib: Neighbours per node for the sibling relation.
            k_align: Neighbours per node for the align relation.
            dilations: Temporal hop lengths to emit; each must name a vocab relation.
            device: Torch device for distance computations.
            logger: Optional logger receiving one step per relation.
            metric: Edge-weight hook (robot_ood.data.metrics.C2EdgeWeights), which weights EVERY
                relation including temporal. None keeps the Gaussian default, which pins temporal
                edges at 1.0 -- i.e. says next16 is next1.

        Returns:
            A validated EdgeSet.
        """
        n = len(nodes)
        dev = _resolve_device(device)
        builders = [
            ("temporal", partial(temporal_edges, nodes, tuple(dilations))),
            ("sibling", partial(sibling_edges, nodes, k=k_sib, device=dev)),
            ("align", partial(align_edges, nodes, k=k_align, device=dev)),
            ("approach", partial(approach_edges, nodes)),
            # Empty on a single-arm table, so one build path serves both robots. Cross-arm fan-out
            # is a quarter of the within-arm one: the other arm hints at posture, it is not the belief's home.
            ("partner", partial(partner_edges, nodes)),
            ("cross_arm", partial(cross_arm_edges, nodes, k=max(1, k_sib // 4), device=dev)),
            ("cross_arm_align", partial(cross_arm_align_edges, nodes, k=max(1, k_align // 4), device=dev)),
        ]

        parts: list[RelationEdges] = []
        for label, build_fn in builders:
            t0 = time.perf_counter()
            part = build_fn()
            if logger is not None:
                logger.step(label, f"E={len(part):_} t={time.perf_counter() - t0:.2f}s")
            parts.append(part)

        edges = RelationEdges.concat(parts)
        w = cls._kernel_weights(nodes, edges) if metric is None else metric(nodes, edges)
        log_idf = node_idf(nodes, edges)
        edges, w, indptr = cls._csr_sort(edges, w, n)

        es = cls(
            src=edges.src.astype(_I4),
            dst=edges.dst.astype(_I4),
            rel=edges.rel.astype(_I1),
            w=w.astype(_F4),
            indptr=indptr,
            log_idf=log_idf.astype(_F4),
            n_nodes=n,
        )
        es.validate(nodes)
        return es

    @staticmethod
    def _kernel_weights(nodes: NodeTable, edges: RelationEdges) -> np.ndarray:
        """Derive per-edge weights: 1.0 for temporal and partner, a Gaussian kernel otherwise.

        Args:
            nodes: Node table supplying q.
            edges: Edge list to weight.

        Returns:
            float32 weights, shape [E].
        """
        w = np.ones(len(edges), dtype=np.float32)
        # partner keeps weight 1.0 with temporal: it states that two configurations were true at
        # one moment, which no distance between them can strengthen or weaken.
        kernel = np.isin(
            edges.rel,
            [schema.REL_INDEX[r] for r in
             ("sibling", "align", "approach", "cross_arm", "cross_arm_align")],
        )
        if kernel.any():
            q = np.asarray(nodes.q)
            dq = q[edges.src[kernel]] - q[edges.dst[kernel]]
            d2 = np.einsum("ij,ij->i", dq, dq)
            w[kernel] = np.exp(-d2 / (2.0 * schema.KERNEL_BW ** 2)).astype(np.float32)
        return w

    @staticmethod
    def _csr_sort(
        edges: RelationEdges, w: np.ndarray, n_nodes: int
    ) -> tuple[RelationEdges, np.ndarray, np.ndarray]:
        """Sort edges by dst and derive the CSR offset array.

        Args:
            edges: Unsorted edge list.
            w: Per-edge weights aligned with edges.
            n_nodes: Node count V.

        Returns:
            (sorted_edges, sorted_w, indptr) with indptr of length V + 1.
        """
        order = np.argsort(edges.dst, kind="stable")
        sorted_edges = RelationEdges(
            src=edges.src[order], dst=edges.dst[order], rel=edges.rel[order]
        )
        indptr = np.concatenate(
            [[0], np.bincount(sorted_edges.dst, minlength=n_nodes).cumsum()]
        ).astype(_I4)
        return sorted_edges, w[order], indptr

    # ---- (de)serialisation -----------------------------------------------------------
    def save(self, path: str | os.PathLike) -> str:
        """Write the edge set to a compressed npz file.

        Args:
            path: Destination .npz, or the directory to write EDGES_NPZ into.

        Returns:
            The path written to.
        """
        return os.fspath(npz.write_arrays(path, schema.EDGES_NPZ, {
            "src": self.src.astype(_I4),
            "dst": self.dst.astype(_I4),
            "rel": self.rel.astype(_I1),
            "w": self.w.astype(_F4),
            "indptr": self.indptr.astype(_I4),
            "log_idf": self.log_idf.astype(_F4),
        }, compress=True))

    @classmethod
    def load(cls, path: str | os.PathLike, strict: bool = True) -> EdgeSet:
        """Read an edge set back from an npz file.

        Args:
            path: Source .npz, or the directory holding EDGES_NPZ.
            strict: Run the structural half of validate after loading.

        Returns:
            The loaded EdgeSet. n_nodes is recovered as len(indptr) - 1.

        Raises:
            FileNotFoundError: Nothing is at the resolved path.
        """
        _, d = npz.read(path, schema.EDGES_NPZ, MISSING_EDGES)
        indptr = d["indptr"].astype(_I4)
        es = cls(
            src=d["src"].astype(_I4),
            dst=d["dst"].astype(_I4),
            rel=d["rel"].astype(_I1),
            w=d["w"].astype(_F4),
            indptr=indptr,
            log_idf=d["log_idf"].astype(_F4),
            n_nodes=int(len(indptr) - 1),
        )
        if strict:
            es.validate()
        return es

    # ---- derived views ---------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        """Count edges per relation name.

        Returns:
            Mapping from relation name to edge count; values sum to E.
        """
        hist = np.bincount(self.rel.astype(np.int64), minlength=schema.N_RELATIONS)
        return {name: int(hist[i]) for i, name in enumerate(schema.RELATIONS)}

    def weight_report(
        self, w: np.ndarray | None = None, dead_below: float = 1e-3
    ) -> dict[str, dict[str, float]]:
        """Summarise the edge-weight distribution per relation.

        Args:
            w: [E] weights to summarise, or None for this edge set's own.
            dead_below: Weight under which an edge counts as carrying nothing.

        Returns:
            Relation name -> {n, mean, std, p01, p50, p99, frac_dead}. Relations with no edges are
            reported with n = 0 and NaN statistics rather than omitted.
        """
        # A passed-in w must be in the same dst-sorted order as self.w (e.g. learned weights),
        # so it lines up with self.rel positionally, not just in length.
        values = self.w if w is None else np.asarray(w, dtype=np.float64)
        if len(values) != len(self.rel):
            raise ValueError(f"weight_report: w has {len(values)} entries, E={len(self.rel)}")
        rel = self.rel.astype(np.int64)

        out: dict[str, dict[str, float]] = {}
        for index, name in enumerate(schema.RELATIONS):
            block = np.asarray(values[rel == index], dtype=np.float64)
            if not len(block):
                out[name] = {
                    "n": 0.0, "mean": float("nan"), "std": float("nan"),
                    "p01": float("nan"), "p50": float("nan"), "p99": float("nan"),
                    "frac_dead": float("nan"),
                }
                continue
            p01, p50, p99 = (float(x) for x in np.percentile(block, [1, 50, 99]))
            out[name] = {
                "n": float(len(block)), "mean": float(block.mean()), "std": float(block.std()),
                "p01": p01, "p50": p50, "p99": p99,
                "frac_dead": float((block < dead_below).mean()),
            }
        return out

    def next_successors(self, n_nodes: int) -> np.ndarray:
        """[V, n_next] the destination of each next relation from every node, the node itself where none."""
        names = schema.DEFAULTS.vocab.next_relations
        out = np.repeat(np.arange(n_nodes, dtype=np.int64)[:, None], len(names), axis=1)
        rel = np.asarray(self.rel, dtype=np.int64)
        src, dst = np.asarray(self.src, dtype=np.int64), np.asarray(self.dst, dtype=np.int64)
        for j, name in enumerate(names):
            m = rel == schema.REL_INDEX[name]
            out[src[m], j] = dst[m]
        return out

    def torch(self, device: DeviceLike = None) -> GraphTensors:
        """Materialise the edge set as device-resident tensors.

        Args:
            device: Target torch device.

        Returns:
            A GraphTensors holding both the dst-sorted storage order and a
            relation-grouped permutation for training-time throughput.
        """
        dev = _resolve_device(device)
        rel_np = self.rel.astype(np.int64)
        order = np.argsort(rel_np, kind="stable")
        # NOTE: offsets are over the FIXED relation vocabulary, not over node ids the way
        # indptr is. A relation with zero edges yields an empty slice, which is legal.
        rel_offsets = np.searchsorted(
            rel_np[order], np.arange(schema.N_RELATIONS + 1)
        ).astype(np.int64)

        order_t = torch.as_tensor(order, dtype=torch.int64, device=dev)
        src_t = torch.as_tensor(self.src, dtype=torch.int32, device=dev)
        dst_t = torch.as_tensor(self.dst, dtype=torch.int32, device=dev)
        w_t = torch.as_tensor(self.w, dtype=torch.float32, device=dev)
        return GraphTensors(
            src=src_t,
            dst=dst_t,
            rel=torch.as_tensor(self.rel, dtype=torch.int64, device=dev),
            w=w_t,
            log_idf=torch.as_tensor(self.log_idf, dtype=torch.float32, device=dev),
            src_by_rel=src_t.index_select(0, order_t),
            dst_by_rel=dst_t.index_select(0, order_t),
            w_by_rel=w_t.index_select(0, order_t),
            rel_offsets=torch.as_tensor(rel_offsets, dtype=torch.int64, device=dev),
        )

    # ---- validation ------------------------------------------------------------------
    def validate(self, nodes: NodeTable | None = None) -> None:
        """Check every structural and (optionally) semantic invariant.

        Args:
            nodes: Node table enabling the per-relation semantic checks. Structural
                checks always run.

        Raises:
            ValueError: On the first violated invariant, naming it and a few
                offending edge indices.
        """
        v, e = self.n_nodes, len(self.src)
        if not (len(self.dst) == e and len(self.rel) == e and len(self.w) == e):
            raise ValueError(
                f"EdgeSet: array length mismatch src={e} dst={len(self.dst)} "
                f"rel={len(self.rel)} w={len(self.w)}"
            )
        if len(self.indptr) != v + 1:
            raise ValueError(f"EdgeSet: indptr length {len(self.indptr)} != n_nodes+1 ({v + 1})")
        if nodes is not None and v != len(nodes):
            raise ValueError(f"EdgeSet: n_nodes={v} != len(nodes)={len(nodes)}")

        if e:
            if int(self.src.min()) < 0 or int(self.src.max()) >= v:
                raise ValueError(f"EdgeSet: src out of range [0,{v}) -- max={int(self.src.max())}")
            if int(self.dst.min()) < 0 or int(self.dst.max()) >= v:
                raise ValueError(f"EdgeSet: dst out of range [0,{v}) -- max={int(self.dst.max())}")
            self_loop = self.src == self.dst
            if self_loop.any():
                off = np.nonzero(self_loop)[0][:5]
                raise ValueError(f"EdgeSet: self-loop(s) at node(s) {self.src[off].tolist()}")
            if int(self.rel.min()) < 0 or int(self.rel.max()) >= schema.N_RELATIONS:
                raise ValueError(
                    f"EdgeSet: rel out of range [0,{schema.N_RELATIONS}) -- "
                    f"min={int(self.rel.min())} max={int(self.rel.max())}"
                )
            if np.any(np.isnan(self.w)):
                raise ValueError("EdgeSet: NaN in w")

            # NOTE: semantic checks run BEFORE the CSR-order check on purpose -- a
            # corrupted edge usually breaks both, and the specific message must win.
            if nodes is not None:
                self._validate_semantics(nodes)

            if np.any(np.diff(self.dst.astype(np.int64)) < 0):
                raise ValueError("EdgeSet: dst is not sorted non-decreasing (broken CSR ordering)")

        if self.indptr[0] != 0:
            raise ValueError(f"EdgeSet: indptr[0]={int(self.indptr[0])} != 0")
        if int(self.indptr[-1]) != e:
            raise ValueError(f"EdgeSet: indptr[-1]={int(self.indptr[-1])} != E={e}")
        if np.any(np.diff(self.indptr.astype(np.int64)) < 0):
            raise ValueError("EdgeSet: indptr is not non-decreasing")
        if len(self.log_idf) != v:
            raise ValueError(f"EdgeSet: log_idf length {len(self.log_idf)} != n_nodes {v}")
        if np.any(np.isnan(self.log_idf)):
            raise ValueError("EdgeSet: NaN in log_idf")

    def _validate_semantics(self, nodes: NodeTable) -> None:
        """Check the per-relation equality and offset constraints.

        Args:
            nodes: Node table supplying the labels each relation constrains.

        Raises:
            ValueError: Naming the violated constraint and a few offending edges.
        """
        owner = np.asarray(nodes.owner)
        task_id = np.asarray(nodes.task_id)
        grip = np.asarray(nodes.grip)
        t_idx = np.asarray(nodes.t_idx).astype(np.int64)
        phase = np.asarray(nodes.phase)
        s = self.src.astype(np.int64)
        d = self.dst.astype(np.int64)
        rel = self.rel

        def check(mask: np.ndarray, ok: np.ndarray, label: str) -> None:
            bad = mask & ~ok
            if bad.any():
                off = np.nonzero(bad)[0][:5]
                raise ValueError(
                    f"EdgeSet.validate: '{label}' violated at {int(bad.sum())} edge(s), "
                    f"e.g. idx={off.tolist()} src={s[off].tolist()} dst={d[off].tolist()}"
                )

        m = rel == schema.REL_INDEX["sibling"]
        check(m, owner[s] != owner[d], "sibling: owner[src] != owner[dst]")
        check(m, task_id[s] == task_id[d], "sibling: task_id[src] == task_id[dst]")
        check(m, grip[s] == grip[d], "sibling: grip[src] == grip[dst]")

        m = rel == schema.REL_INDEX["align"]
        check(m, task_id[s] != task_id[d], "align: task_id[src] != task_id[dst]")
        check(m, grip[s] == grip[d], "align: grip[src] == grip[dst]")
        check(
            m,
            phase_bins(phase[s], schema.NBINS_ALIGN) == phase_bins(phase[d], schema.NBINS_ALIGN),
            "align: phase_bin[src] == phase_bin[dst]",
        )

        virtual = _virtual_mask(nodes)
        m = rel == schema.REL_INDEX["approach"]
        check(m, virtual[s], "approach: src is virtual")
        check(m, ~virtual[d], "approach: dst is real")
        check(m, owner[s] == owner[d], "approach: owner[src] == owner[dst]")

        for dl in schema.DILATIONS:
            m = rel == schema.REL_INDEX[f"next{dl}"]
            check(m, owner[s] == owner[d], f"next{dl}: owner[src] == owner[dst]")
            check(m, (t_idx[d] - t_idx[s]) == dl, f"next{dl}: t_idx[dst] - t_idx[src] == +{dl}")

            m = rel == schema.REL_INDEX[f"prev{dl}"]
            check(m, owner[s] == owner[d], f"prev{dl}: owner[src] == owner[dst]")
            check(m, (t_idx[d] - t_idx[s]) == -dl, f"prev{dl}: t_idx[dst] - t_idx[src] == -{dl}")
