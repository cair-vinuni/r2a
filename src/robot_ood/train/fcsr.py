"""Run options, constants and graph helpers shared by the FCSR objective."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_ood.core.constants import ENTRY_STATIC_BAND, HELDOUT_STRIDE
from robot_ood.core.config import Ablation, GraphConfig
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable
from robot_ood.data.store import ArtifactStore

N_QUERY = 4_096
N_EVAL = 256
N_SEG_NEG = 8
QBATCH = 256
DEFAULT_PHASE_BAND = 0.05
DEFAULT_DTW_THRESH = 0.05
LR = 1e-3
WEIGHT_DECAY = 1e-5
# The corrector reads the retriever's shortlist, so it joins once tier 1 has settled; the segment
# weight then halves so the two terms share the optimiser.
CORRECTOR_START_EPOCH = 5
SEG_WEIGHT_LATE = 0.5


@dataclass(frozen=True)
class FcsrTrainSpec:
    """Run options for the joint FCSR stage.

    Attributes:
        epochs: Epochs to train.
        seed: RNG/torch seed.
        device: Torch device; None resolves to the project default.
        cfg: Graph configuration; None resolves from the environment.
        logger: Run logger, or None for stdout.
        n_query: Training queries per epoch; None resolves to N_QUERY.
        n_eval: Held-out queries; None resolves to N_EVAL.
        save: Whether to write checkpoints.
        qbatch: Rows per optimiser step; None resolves to QBATCH.
        lr: Optimiser learning rate; None resolves to LR.
        language: Whether the retriever reads the episode's instruction -- the query encoder's
            language branch and the scorer's task_affinity feature. False trains a graph's
            geometry alone, which is all a graph built without build language can offer.
        ablation: Which deploy mechanisms this run trains without (robot_ood.core.config.Ablation);
            graph drops the locator's phase, task prior and gate, memory reads the locator cold.
    """

    epochs: int = 30
    seed: int = 0
    device: Any = None
    cfg: "GraphConfig | None" = None
    logger: Any = None
    n_query: int | None = None
    n_eval: int | None = None
    save: bool = True
    qbatch: int | None = None
    lr: float | None = None
    language: bool = True
    ablation: "Ablation" = Ablation()


def lambdas(epoch: int) -> tuple[float, float]:
    """The (segment, corrector) loss weights at epoch."""
    if epoch < CORRECTOR_START_EPOCH:
        return 1.0, 0.0
    return SEG_WEIGHT_LATE, 1.0


def log_line(logger: Any, msg: str) -> None:
    """Emit one progress line through logger, or to stdout."""
    (logger.log(msg) if logger is not None else print(msg))


def entry_stratum() -> tuple[float, float | None]:
    """The calibrated entry-stratum oversample, or no oversample when none was calibrated."""
    from robot_ood.core.config import calibrated_constants, warn_uncalibrated

    constants = calibrated_constants()
    if constants is None or "seg_entry_stratum_frac" not in constants:
        warn_uncalibrated("seg_entry_stratum_frac", 0.0)
    if constants is None or "seg_entry_band" not in constants:
        warn_uncalibrated("seg_entry_band", ENTRY_STATIC_BAND)
    if constants is None:
        return 0.0, None
    return (
        float(constants.value("seg_entry_stratum_frac", 0.0)),
        float(constants.value("seg_entry_band", ENTRY_STATIC_BAND)),
    )


def hard_negative_geometry() -> tuple[float, float]:
    """The calibrated hard-negative (dtw_thresh, phase_band), or the module defaults."""
    from robot_ood.core.config import calibrated_constants, warn_uncalibrated

    constants = calibrated_constants()
    if constants is None or "seg_hard_neg_thresh" not in constants:
        warn_uncalibrated("seg_hard_neg_thresh", DEFAULT_DTW_THRESH)
    if constants is None or "seg_hard_neg_phase_band" not in constants:
        warn_uncalibrated("seg_hard_neg_phase_band", DEFAULT_PHASE_BAND)
    if constants is None:
        return DEFAULT_DTW_THRESH, DEFAULT_PHASE_BAND
    return (
        float(constants.value("seg_hard_neg_thresh", DEFAULT_DTW_THRESH)),
        float(constants.value("seg_hard_neg_phase_band", DEFAULT_PHASE_BAND)),
    )


def split_owners(nodes: NodeTable) -> tuple[np.ndarray, np.ndarray]:
    """The stride-HELDOUT_STRIDE by-demo train/held-out split."""
    from robot_ood.train.loop import heldout_demos

    heldout_owners = heldout_demos(nodes.n_demos)
    train_owners = np.setdiff1d(np.arange(nodes.n_demos, dtype=np.int64), heldout_owners)
    if len(train_owners) == 0 or len(heldout_owners) == 0:
        raise ValueError(
            f"robot_ood.train.fcsr: n_demos={nodes.n_demos} too small for a stride-{HELDOUT_STRIDE} "
            "by-demo held-out split (need >= 2 demos)"
        )
    return train_owners, heldout_owners


def load_graph(graph_dir: Path) -> tuple[NodeTable, EdgeSet, Any]:
    """The node table, edge set and segment table beside graph_dir."""
    store = ArtifactStore(graph_dir)
    nodes = store.nodes()
    edges = store.edges()
    edges.validate(nodes)
    return nodes, edges, store.segments()


def retrieve_candidates(retriever: Any, computer: Any, seg_t: Any, qb: Any,
                        exclude_owner: Any = None) -> tuple[Any, Any]:
    """The deployed tier-1/tier-2 shortlist for qb, and the query's observed phase.

    Args:
        retriever: The SegmentRetriever.
        computer: A SegmentLossComputer, used for its context helper.
        seg_t: Segment tensors for this run's segment table.
        qb: This step's queries.
        exclude_owner: [B] demo id each row may not retrieve from, or None.

    Returns:
        (BatchedSegmentCandidates, [B] query phase), both detached from the retriever.
    """
    import torch

    with torch.no_grad():
        ctx = computer.context(seg_t, qb)
        return retriever.retrieve_batch(ctx, exclude_owner=exclude_owner), ctx.phase


def realize_column0(seg_t: Any, qb: Any, cand_idx: Any, ref_for: Any, cart: Any = None) -> tuple[Any, ...]:
    """Apply the row's retrieval corruption to the candidate set.

    WRONG_TASK and WRONG_PHASE swap column 0's segment. OFFSET moves column 0's start, end and every
    reference row by one displacement: one candidate from elsewhere. TARGET is the scene that moved:
    every column keeps its start and carries the same displacement on every later reference row and
    on its end, so the retrieved demos still agree with each other and the consensus reference
    follows them to where the object used to be.

    Args:
        seg_t: Segment tensors for this run's segment table.
        qb: This step's queries.
        cand_idx: [B, K] candidate segment ids.
        ref_for: [B, K] segment ids -> [B, K, C, 6] their pose reference (ee_ref_for), read after
            the swap so a swapped column 0 carries its own segment's reference.
        cart: The arm's robot_ood.core.kinematics.Arx5Kinematics on a joint-space lane, through
            whose Jacobian the Cartesian displacement becomes a joint displacement, or None on a
            Cartesian lane, where it applies as drawn.

    Returns:
        (cand_idx, cand_ee_start, cand_ee_end, cand_actions, ee_ref, realized_kind).
    """
    import torch

    from robot_ood.core.kinematics import lane_displacement
    from robot_ood.train.augment import CLEAN, OFFSET, TARGET

    idx = cand_idx.clone()
    realized = None if qb.corrupt_kind is None else np.asarray(qb.corrupt_kind, dtype=np.int8).copy()
    if qb.corrupt_node is not None:
        node = qb.corrupt_node
        want_swap = node >= 0
        if bool(want_swap.any()):
            seg_for_node = seg_t.node_to_segment[node.clamp_min(0)]
            ok = want_swap & (seg_for_node >= 0)
            idx[:, 0] = torch.where(ok, seg_for_node, idx[:, 0])
            if realized is not None:
                realized[(want_swap & ~ok).detach().cpu().numpy()] = CLEAN

    ee_start = seg_t.ee_start[idx]
    ee_end = seg_t.ee_end[idx]
    ee_ref = ref_for(idx)
    if qb.corrupt_offset is not None and realized is not None:
        kind = torch.as_tensor(realized, device=idx.device)
        is_offset, is_target = kind == OFFSET, kind == TARGET
        if bool((is_offset | is_target).any()):
            delta = lane_displacement(qb.corrupt_offset.to(ee_start.dtype), ee_start[:, 0], cart)
            zero = torch.zeros_like(delta)
            whole = torch.where(is_offset[:, None], delta, zero)
            moved = torch.where(is_target[:, None], delta, zero)
            after_start = torch.ones(ee_ref.shape[2], device=idx.device, dtype=ee_ref.dtype)
            after_start[0] = 0.0
            ee_ref = ee_ref.clone()
            ee_ref[:, 0] = ee_ref[:, 0] + whole[:, None, :]
            ee_ref = ee_ref + after_start[None, None, :, None] * moved[:, None, None, :]
            ee_start = ee_start.clone()
            ee_start[:, 0] = ee_start[:, 0] + whole
            ee_end = ee_end.clone()
            ee_end[:, 0] = ee_end[:, 0] + whole
            ee_end = ee_end + moved[:, None, :]

    return idx, ee_start, ee_end, seg_t.action_chunk[idx], ee_ref, realized



def ee_ref_for(table: Any, node_ee: np.ndarray, cand_idx: Any, chunk_len: int, raw_ref: Any = None) -> Any:
    """[B, K] candidate ids -> [B, K, chunk_len, 6] pose reference.

    raw_ref (a robot_ood.models.fusion.target.RawPoseRef) reads the candidates' own raw frames;
    None interpolates node ids (node_pose_ref). Deploy (CorrectionAuthority.apply) makes the same
    choice off ROBOT_OOD_REF, so the corrector sees one construction at both ends.
    """
    import torch

    from robot_ood.models.fusion.target import node_pose_ref

    idx_np = cand_idx.detach().cpu().numpy()
    if raw_ref is not None:
        ref = raw_ref(table.start_node[idx_np], chunk_len)
    else:
        ref = node_pose_ref(node_ee, table.start_node[idx_np], table.end_node[idx_np], chunk_len)
    return torch.as_tensor(ref, device=cand_idx.device)
