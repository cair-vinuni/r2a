"""Parameter objects for the training curriculum's multi-argument functions.

Field names match the corresponding parameters in loop.py, loss.py and data.py, so a mis-wiring at
a call site is still visible in a diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from robot_ood.core.config import GraphConfig
    from robot_ood.data.edges import EdgeSet
    from robot_ood.data.nodes import NodeTable


@dataclass(frozen=True)
class GraphTensors:
    """The graph prepared once by robot_ood.train.loop.prepare_tensors, shared read-only by every run.

    Attributes:
        nodes: The node table.
        edges: Its edge set.
        node_x_t: Node feature matrix [V, 2*dim+2] on device.
        src: Edge sources [E], on device.
        dst: Edge destinations [E].
        rel: Edge relation ids [E].
        log_idf: Per-node log-IDF prior [V].
        phase_t: Per-node phase [V] float32.
        src_by_rel: Relation-grouped sources [E], or None.
        dst_by_rel: Relation-grouped destinations [E], or None.
        rel_offsets: CSR offsets over the relation vocabulary, or None.
        adj: Per relation, its sparse [V, V] adjacency, built once for the run.
        targets: The run-scoped robot_ood.train.target.TargetBuilder, typed Any to keep this
            module import-free of the modules that import it.
        task_id_t: [V] int64 per-node task id on device.
        e_task_t: [n_tasks, instr_dim] task embeddings on device, or None.
        action_t: [V, chunk, 7] per-node action chunks on device.
        succ_t: [V, n_next] next-relation successors on device.
        owner_t: [V] int64 per-node demo on device.
    """

    nodes: "NodeTable"
    edges: "EdgeSet"
    node_x_t: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    rel: torch.Tensor
    log_idf: torch.Tensor
    phase_t: torch.Tensor
    src_by_rel: torch.Tensor | None = None
    dst_by_rel: torch.Tensor | None = None
    rel_offsets: torch.Tensor | None = None
    adj: Any = None
    targets: Any = None
    task_id_t: torch.Tensor | None = None
    e_task_t: torch.Tensor | None = None
    action_t: torch.Tensor | None = None
    succ_t: torch.Tensor | None = None
    owner_t: torch.Tensor | None = None


@dataclass(frozen=True)
class QueryBatch:
    """One training step's slice of the query mapping, sliced to qbatch rows and moved to device.

    Attributes:
        qh: Joint positions [B, H, D].
        vh: Joint velocities [B, H, D].
        gh: Gripper states [B, H].
        tgt: Target node ids [B].
        a_policy: [B, K, 7] the frozen policy's own chunk, or None.
        a_corr: [B, K, 6] the pose-derived correction chunk, or None.
        ee_now: [B, 6] each window's current end-effector pose, or None.
        ee_ref: [B, K, 6] the label node's reference segment, or None.
        corrupt_node: [B] i64 node whose segment replaces the posterior's, -1 to keep it, or None.
        corrupt_offset: [B, 6] f32 Cartesian displacement (metres, rotation vector) of the reference segment, or None.
        corrupt_kind: [B] i8 which corruption branch each row was assigned.
        e_lang: [B, lang_dim] a paraphrase embedding of the row's task name, or None.
        seg_id: [B] i64 the row's own C1 segment id (SEGMENT queries) or -1, or None.
        neg_segments: [B, n] i64 mined hard-negative segment ids, -1 padded, or None.
        pos_segments: [B] i64 the cross-demo positive segment (ROBOT_OOD_POS=xdemo), or None.
        a_hist: [B, H, ACTION_DIM] the demo's recorded actions ending at this row's window, or None.
        phase_hat: [B] f32 the phase each row's QueryContext carries, or None.
    """

    qh: torch.Tensor
    vh: torch.Tensor
    gh: torch.Tensor
    tgt: np.ndarray
    a_policy: torch.Tensor | None = None
    a_corr: torch.Tensor | None = None
    ee_now: torch.Tensor | None = None
    ee_ref: torch.Tensor | None = None
    corrupt_node: torch.Tensor | None = None
    corrupt_offset: torch.Tensor | None = None
    corrupt_kind: np.ndarray | None = None
    policy_kind: np.ndarray | None = None
    e_lang: torch.Tensor | None = None
    seg_id: torch.Tensor | None = None
    neg_segments: torch.Tensor | None = None
    pos_segments: torch.Tensor | None = None
    a_hist: torch.Tensor | None = None
    task_p: torch.Tensor | None = None
    phase_hat: torch.Tensor | None = None


@dataclass(frozen=True)
class QuerySpec:
    """The keyword-only options robot_ood.train.data.make_queries mixes n queries with.

    Attributes:
        advance: Raw frames from the window end to the target; None resolves to schema.ADVANCE.
        window: Query window length; None resolves to schema.HIST_H.
        entry_frac: Share of ENTRY queries; None resolves to data.ENTRY_FRAC.
        entry_sigma_range: Entry perturbation radius range (rad); None derives it from the corpus.
        entry_static_frac: Share of ENTRY queries that are ENTRY_STATIC instead; 0.0 is opt-out.
        drift_frac: Share of the remaining queries that are DRIFT; 0.0 is opt-out.
        segments: The C1 segment table SEGMENT queries are drawn against, or None to draw none.
        segment_frac: Share of the remaining queries that are SEGMENT; ignored without segments.
        segment_entry_frac: Share of SEGMENT rows anchored in the entry stratum; 0.0 is opt-out.
        segment_entry_band: Phase ceiling of the entry stratum; None resolves to
            robot_ood.core.constants.ENTRY_STATIC_BAND.
        phase_residual: A robot_ood.train.augment.BeliefPhaseResidual putting the belief filter's
            noisy phase in phase_hat, or None for the ground truth.
    """

    advance: int | None = None
    window: int | None = None
    entry_frac: float | None = None
    entry_sigma_range: tuple[float, float] | None = None
    entry_static_frac: float = 0.0
    drift_frac: float = 0.0
    segments: Any = None
    segment_frac: float = 0.0
    segment_entry_frac: float = 0.0
    segment_entry_band: float | None = None
    phase_residual: Any = None


@dataclass(frozen=True)
class TrainGraphSpec:
    """The keyword-only options robot_ood.train.loop.train_graph runs the curriculum with.

    Attributes:
        stages: Curriculum stages to run; only (1,) is legal.
        epochs: Epochs per stage.
        seed: Seeds both the numpy stream and torch.
        device: Torch device; None resolves to the project default.
        save: Whether to write g_head.npz at the end.
        cfg: Graph/training configuration; None builds a default GraphConfig.
        logger: Run logger, or None for stdout.
        n_query: Training queries per stage; None resolves to loop.N_QUERY.
        n_eval: Queries per evaluation set; None resolves to loop.N_EVAL, not a literal copy --
            importing loop.N_EVAL directly here would invert the module dependency.
        task: The robot_ood.train.loop.TASKS key naming what to train. Model selection reads the chunk-blend
            artifacts (a_pi_raw.npz, action_scale.json) when they sit beside the graph, so a run
            picks its epoch on held-out mse(a_exec) whenever they are present.
    """

    stages: tuple[int, ...] = (1,)
    epochs: int = 40
    seed: int = 0
    device: Any = None
    save: bool = True
    cfg: "GraphConfig | None" = None
    logger: Any = None
    n_query: int | None = None
    n_eval: int | None = None
    task: str = "locator"
    no_memory: bool = False
