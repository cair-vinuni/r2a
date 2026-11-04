"""The graph moved to a device, shared read-only by every epoch and batch of a run."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robot_ood.core.config import GraphConfig
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable
from robot_ood.models.retrieval.gnn import CHUNK
from robot_ood.models.types import GraphEdges
from robot_ood.train.data import _node_feature_matrix
from robot_ood.train.target import TargetBuilder
from robot_ood.train.types import GraphTensors

__all__ = ["prepare_tensors", "graph_edges"]


def prepare_tensors(
    nodes: NodeTable, edges: EdgeSet, dev: Any, *, cfg: GraphConfig | None = None,
    language: Any = None, chunk: int = CHUNK,
) -> GraphTensors:
    """Move the graph to dev and bind the run's label rule to it.

    Args:
        nodes: Node table to prepare.
        edges: Its edge set.
        dev: Target device.
        cfg: Graph configuration; None leaves the label rule on its defaults.
        language: robot_ood.models.language.encoder.TaskEmbeddings, or None.
        chunk: Rows per node action chunk; must match the net's policy channel width.

    Returns:
        The bundle threaded through every epoch and batch of one run.
    """
    node_x_t = torch.as_tensor(_node_feature_matrix(nodes), device=dev)
    et = edges.torch(dev)
    adj = relation_adjacencies(len(nodes), et["src_by_rel"].long(), et["dst_by_rel"].long(),
                               et["rel_offsets"], dev)
    return GraphTensors(
        nodes=nodes, edges=edges, node_x_t=node_x_t,
        src=et["src"].long(), dst=et["dst"].long(), rel=et["rel"], log_idf=et["log_idf"],
        phase_t=torch.as_tensor(nodes.phase.astype(np.float32), device=dev),
        src_by_rel=et["src_by_rel"].long(), dst_by_rel=et["dst_by_rel"].long(),
        rel_offsets=et["rel_offsets"], adj=adj,
        targets=TargetBuilder(nodes) if cfg is None else TargetBuilder(nodes, joint_tol=cfg.joint_tol),
        task_id_t=torch.as_tensor(np.asarray(nodes.task_id, dtype=np.int64), device=dev),
        e_task_t=(None if language is None else
                  torch.as_tensor(np.asarray(language.e_task, dtype=np.float32), device=dev)),
        action_t=torch.as_tensor(nodes.action_chunks(chunk), device=dev),
        succ_t=torch.as_tensor(edges.next_successors(len(nodes)), device=dev),
        owner_t=torch.as_tensor(np.asarray(nodes.owner, dtype=np.int64), device=dev),
    )


def graph_edges(gt: GraphTensors) -> GraphEdges:
    """The forward pass's graph bundle off a prepared GraphTensors."""
    return GraphEdges(
        node_x=gt.node_x_t, src=gt.src, dst=gt.dst, rel=gt.rel, log_idf=gt.log_idf,
        src_by_rel=gt.src_by_rel, dst_by_rel=gt.dst_by_rel, rel_offsets=gt.rel_offsets,
        task_id=gt.task_id_t, e_task=gt.e_task_t, action=gt.action_t, succ=gt.succ_t,
    )


def relation_adjacencies(n_nodes: int, src_by_rel: torch.Tensor, dst_by_rel: torch.Tensor,
                         rel_offsets: Any, dev: Any) -> list[torch.Tensor | None]:
    """One sparse [V, V] adjacency per relation, destinations by sources.

    Args:
        n_nodes: V.
        src_by_rel: Relation-grouped edge sources.
        dst_by_rel: Their destinations.
        rel_offsets: Where each relation's block starts and ends.
        dev: Torch device.

    Returns:
        Per relation, its adjacency, or None where the relation has no edge.
    """
    # Sum aggregation multiplies by these instead of gathering every edge, which is the same
    # arithmetic and never holds an [E, hidden] tensor (robot_ood.models.retrieval.gnn).
    offsets = [int(x) for x in rel_offsets]
    out: list[torch.Tensor | None] = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if hi <= lo:
            out.append(None)
            continue
        idx = torch.stack([dst_by_rel[lo:hi], src_by_rel[lo:hi]])
        values = torch.ones(hi - lo, dtype=torch.float32, device=dev)
        out.append(torch.sparse_coo_tensor(idx, values, (n_nodes, n_nodes)).coalesce().to_sparse_csr())
    return out
