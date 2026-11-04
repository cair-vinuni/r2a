"""Parameter objects for robot_ood.models.retrieval.gnn's multi-argument methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import Tensor

from robot_ood.data import schema
from robot_ood.data.metrics import NodeMetricColumns, QueryMetricColumns, SEED_TIER1_M

__all__ = [
    "QueryWindow",
    "EdgeIndex",
    "GraphEdges",
    "MoveCost",
    "PipelineParams",
    "NodeCloud",
    "SeedQuery",
    "SeedLanguage",
]


def _as_int_list(offsets: Tensor | Iterable[int] | None) -> list[int] | None:
    """Normalize relation offsets to a plain Python list.

    Args:
        offsets: Offsets as a tensor, any int iterable, or None.

    Returns:
        The offsets as list[int], or None if none were given.
    """
    if offsets is None:
        return None
    if isinstance(offsets, Tensor):
        return offsets.tolist()
    return list(offsets)


def _both_or_neither(name_a: str, a: object, name_b: str, b: object) -> None:
    """Raise if exactly one of two co-required fields is set.

    Args:
        name_a: Field name of the first value.
        a: First value.
        name_b: Field name of the second value.
        b: Second value.

    Raises:
        ValueError: Exactly one of the two was given.
    """
    if (a is None) != (b is None):
        missing, given = (name_a, name_b) if a is None else (name_b, name_a)
        raise ValueError(f"{given} was given without {missing}; pass both or neither.")


@dataclass(frozen=True, slots=True)
class QueryWindow:
    """One query's history window, as taken by QueryEncoder and GraphRetrieverNet.seed.

    Attributes:
        qh: Joint positions, [H, dim] or [B, H, dim], row 0 oldest.
        vh: Joint velocities, same shape as qh.
        gh: Gripper states, [H] or [B, H].
    """

    qh: Tensor
    vh: Tensor
    gh: Tensor


@dataclass(frozen=True, slots=True)
class GraphEdges:
    """Graph-level tensors for forward, including the optional relation-grouped view.

    Attributes:
        node_x: Raw node rows [V, 2*dim+2] = [q, qdot, grip, t_frac].
        src: Edge sources [E], dst-sorted CSR order.
        dst: Edge destinations [E].
        rel: Edge relation ids [E].
        log_idf: Per-node log-IDF prior [V].
        src_by_rel: Relation-grouped sources [E], or None.
        dst_by_rel: Relation-grouped destinations [E], or None.
        rel_offsets: CSR offsets over the relation vocabulary, length n_rel+1, or None. May be
            passed as the raw int64 tensor; stored as list[int].
        task_id: [V] int64 per-node task id, or None.
        e_task: [n_tasks, instr_dim] task instruction embeddings, or None.
        action: [V, chunk, 7] each node's demonstrated action chunk, or None.
        succ: [V, n_next] successor along each next relation (self where none), or None.
    """

    node_x: Tensor
    src: Tensor
    dst: Tensor
    rel: Tensor
    log_idf: Tensor
    src_by_rel: Tensor | None = None
    dst_by_rel: Tensor | None = None
    rel_offsets: list[int] | None = None
    task_id: Tensor | None = None
    e_task: Tensor | None = None
    action: Tensor | None = None
    succ: Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rel_offsets", _as_int_list(self.rel_offsets))
        _both_or_neither("src_by_rel", self.src_by_rel, "dst_by_rel", self.dst_by_rel)


@dataclass(frozen=True, slots=True)
class EdgeIndex:
    """Per-edge index tensors for one message-passing layer, plus the optional grouped view.

    Attributes:
        src: Edge sources [E].
        dst: Edge destinations [E].
        rel: Edge relation ids [E], int64 (indexes the gate table).
        src_by_rel: Relation-grouped sources [E], or None.
        dst_by_rel: Relation-grouped destinations [E], or None.
        rel_offsets: CSR offsets over the relation vocabulary, length n_rel+1, or None.
        adj: Per relation, its [V, V] sparse adjacency (destination by source), or None to send
            messages edge by edge instead. Sum aggregation reads it and never builds a per-edge
            tensor; an empty relation is None in the list.
    """

    src: Tensor
    dst: Tensor
    rel: Tensor
    src_by_rel: Tensor | None = None
    dst_by_rel: Tensor | None = None
    rel_offsets: list[int] | None = None
    adj: list[Tensor | None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rel_offsets", _as_int_list(self.rel_offsets))
        _both_or_neither("src_by_rel", self.src_by_rel, "dst_by_rel", self.dst_by_rel)


@dataclass(frozen=True, slots=True)
class MoveCost:
    """Optional movement-cost feature concatenated onto the readout MLP.

    Attributes:
        q_nodes: Normalized node configs [V, dim], or None.
        q_now: The query's current normalized config [dim] or [B, dim], or None.
    """

    q_nodes: Tensor | None = None
    q_now: Tensor | None = None

    def __post_init__(self) -> None:
        _both_or_neither("q_nodes", self.q_nodes, "q_now", self.q_now)


@dataclass(frozen=True, slots=True)
class PipelineParams:
    """Scalar knobs forward passes down to seed, logits and propagate.

    Attributes:
        idf_w: Weight on the per-node log-IDF prior in the readout.
        topk: Seed-set size; nodes outside it get exactly zero h^0.
        tier1_m: Tier-1 candidates per window row in the two-tier seed.
        checkpoint: Recompute each layer's forward on backward instead of retaining activations.
    """

    idf_w: float = 0.0
    topk: int = schema.SEED_TOPK
    tier1_m: int = SEED_TIER1_M
    checkpoint: bool = True


@dataclass(frozen=True, slots=True)
class NodeCloud:
    """Per-node, query-independent tensors seed takes.

    Attributes:
        q_nodes_n: Every node's normalized joint config [V, dim], the metric backbone's own space.
        n_v: node_encode's output [V, hidden], modulating the seeded h^0 and keying tier 1.
        columns: The remaining per-node columns C2 reads (velocity, phase, grip, task, pose),
            as robot_ood.data.metrics.NodeMetricColumns.
    """

    q_nodes_n: Tensor
    n_v: Tensor
    columns: NodeMetricColumns


@dataclass(frozen=True, slots=True)
class SeedQuery:
    """The query side of the learned seed: what the encoder produced plus what the metric reads.

    Attributes:
        e_q: Pooled query embedding [hidden] or [B, hidden]; scales the seeded h^0.
        z_h: Per-step encoder states [H, hidden] or [B, H, hidden]; tier-1 keys.
        pool_w: Pooling weights [H] or [B, H]; the PER-ROW mixture's weights.
        qh_n: Normalized query window [H, dim] or [B, H, dim], row 0 oldest.
        columns: The query's own metric columns (velocity, grip, optional pose).
        instr: [instr_low] or [B, instr_low] the projected instruction, or None.
        policy: [chunk, 7] or [B, chunk, 7] the policy chunk just issued, or None.
        instr_p: [n_tasks] or [B, n_tasks] the instruction's posterior over the graph's tasks, or
            None; the seed's language prior and the readout's log-prior column read it.
    """
    e_q: Tensor
    z_h: Tensor
    pool_w: Tensor
    qh_n: Tensor
    columns: QueryMetricColumns
    instr: Tensor | None = None
    policy: Tensor | None = None
    has_instr: Tensor | None = None
    has_policy: Tensor | None = None
    instr_p: Tensor | None = None
@dataclass(frozen=True, slots=True)
class SeedLanguage:
    """Optional instruction prior on the seed, as an already-computed per-node multiplier.

    Attributes:
        factor: [V] or [B, V] strictly positive multipliers on the seeding score, applied BEFORE the
            top-k truncation so the prior can change which nodes are seeded. None disables it, and
            disabled is bit-identical to the seed without this term -- not a multiply by ones.
    """

    factor: Tensor | None = None
