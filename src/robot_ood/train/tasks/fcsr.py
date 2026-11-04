"""The joint FCSR objective: one forward over retriever and corrector, one backward."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robot_ood.data import schema
from robot_ood.data.geometry import CONSTANTS_JSON
from robot_ood.data.metrics import ee_pose_delta_norm
from robot_ood.train.fcsr import (
    CORRECTOR_START_EPOCH, LR, N_EVAL, N_QUERY, N_SEG_NEG, QBATCH, FcsrTrainSpec, ee_ref_for,
    entry_stratum, hard_negative_geometry, lambdas, load_graph, log_line, realize_column0,
    retrieve_candidates, split_owners,
)
from robot_ood.train.negatives import HardNegativeMiner, segment_tensors
from robot_ood.train.tasks.query import QuerySet
from robot_ood.train.types import QuerySpec

TIER1_RECALL_CUTS: tuple[int, ...] = (8, 128, 512, 2048)
_DOSE_GRID = 41   # constant-gain grid the learned corrector is measured against


def load_segment_language(graph_dir: Any, logger: Any = None, *, enabled: bool = True) -> Any:
    """The graph's instruction embeddings: the query encoder's language branch and task_affinity."""
    if not enabled:
        log_line(logger, "[fcsr] language OFF by configuration -- the scorer sees no instruction")
        return None
    from robot_ood.models.language.encoder import TaskEmbeddings

    try:
        language = TaskEmbeddings.load(graph_dir)
    except FileNotFoundError:
        log_line(logger, f"[fcsr] no g_lang.npz in {graph_dir} -- training without the instruction "
                         "(build one with `build language`)")
        return None
    log_line(logger, f"[fcsr] language on: {language.n_tasks} tasks, "
                     f"dim {language.e_task.shape[1]}, encoder {language.model_id}")
    return language


def language_dim(language: Any) -> int:
    """The width of a TaskEmbeddings table, or 0 when there is none."""
    return 0 if language is None else int(language.e_task.shape[1])


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based ROC AUC of score against a binary label (ties averaged); nan when one class is absent."""
    pos = label > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = sums[inv] / counts[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def state_bytes(module: Any) -> dict[str, np.ndarray]:
    """A module's state_dict as a {name: array} mapping."""
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def corrector_report(stats: dict[str, float]) -> str:
    """One line of the numbers that say whether the corrector is worth deploying."""
    floor = min(stats["corr_mse_policy"], stats["corr_mse_full"], stats["corr_mse_best_const"])
    verdict = "BETTER" if stats["corr_mse_learned"] < floor else "NO GAIN OVER THE BEST CONSTANT"
    return (
        f"[fcsr] corrector mse: policy {stats['corr_mse_policy']:.6f}  full {stats['corr_mse_full']:.6f}  "
        f"best-const {stats['corr_mse_best_const']:.6f} (at {stats['corr_best_const']:.2f})  "
        f"learned {stats['corr_mse_learned']:.6f}  oracle {stats['corr_mse_oracle']:.6f}  "
        f"| trust auc {stats['trust_auc']:.3f} acc {stats['trust_acc']:.3f} "
        f"pos-rate {stats['trust_pos_rate']:.3f}  {verdict}"
    )


class FcsrTask:
    """Fit the segment retriever and the corrector as one staged objective.

    Attributes:
        name: The task's registry name.
    """

    name = "fcsr"

    def __init__(self, spec: FcsrTrainSpec | None = None) -> None:
        """Bind the task to its run options; nothing is read until build.

        Args:
            spec: Run options; None is FcsrTrainSpec()'s default.
        """
        self.spec = spec if spec is not None else FcsrTrainSpec()
        self.ctx: Any = None
        self.epoch = 0

    def requires(self) -> Any:
        """The graph, its segment memory and the chunk-blend artifacts."""
        from robot_ood.models.fusion.target import ACTIONS_NPZ
        from robot_ood.train.task import Requirements

        # HEAD_NPZ because the phase residual rides inside it: train locator embeds its own
        # replay report there.
        return Requirements(artifacts=(
            schema.NODES_NPZ, schema.EDGES_NPZ, schema.SEGMENTS_NPZ, schema.HEAD_NPZ,
            ACTIONS_NPZ, schema.ACTION_SCALE_JSON,
        ))

    def build(self, ctx: Any) -> Any:
        """Construct the retriever and the corrector, and draw the held-out set.

        Args:
            ctx: The run's store, config, device and generator.

        Returns:
            All five modules trained together; this objective freezes nothing.
        """
        from robot_ood.core.action import ActionScale
        from robot_ood.core.kinematics import cartesian_for, is_joint_space, kinematics_for
        from robot_ood.core.protocol import suite_chunk
        from robot_ood.core.schemas import ACTION_EMBED_DIM
        from robot_ood.train.data import entry_sigma_range
        from robot_ood.models.fusion.corrector import Corrector
        from robot_ood.models.fusion.target import ChunkTargetBuilder, PolicyActionCache
        from robot_ood.models.fcsr_net import FcsrNet, SegEncoders
        from robot_ood.models.retrieval.retriever import (
            SegmentRetriever,
        )
        from robot_ood.models.retrieval.encoders import QueryEncoder, SegmentKeyEncoder
        from robot_ood.models.retrieval.scorer import ActionEncoder, SegmentScorer
        from robot_ood.data.segments import graph_plan_rows, node_ee_table, segment_pause
        from robot_ood.models.fusion.corrector import N_GATE
        from robot_ood.data.store import recorded_suite
        from robot_ood.train.augment import BeliefPhaseResidual, CopyPlan, PolicyCorruptor, RetrievalCorruptor
        from robot_ood.train.loss import CorrectorLoss, SegmentLossComputer
        from robot_ood.train.task import TrainUnit

        self.ctx = ctx
        self.epoch = 0
        dev = ctx.device
        self.cfg = self.spec.cfg or ctx.cfg
        graph_dir = ctx.store.dir
        suite = recorded_suite(graph_dir)
        self._joint_space = is_joint_space(suite)
        kinematics = kinematics_for(suite)
        # The arm's chain on a joint-space lane: the corrector's reference is read through it.
        self.cart = cartesian_for(suite)

        self.nodes, self.edges, self.table = load_graph(graph_dir)
        self._sigma_range = entry_sigma_range(self.nodes)
        # The rows one deployed plan covers: the corrector is trained on chunks of that length, so
        # its reference carries the same travel per row that deploy will hand it.
        self.plan_rows = graph_plan_rows(graph_dir, suite_chunk(suite))
        # Resolved before any consumer reads a constant: the retriever encoders and the entry
        # stratum below pull calibrated values in their constructors.
        self.constants = self._resolve_constants()
        self.node_ee = node_ee_table(self.nodes)
        from robot_ood.models.fusion.target import raw_pose_ref_for
        self.raw_ref = raw_pose_ref_for(self.nodes)
        if self.raw_ref is not None:
            log_line(getattr(ctx, "logger", None), "[fcsr] reference: raw demo frames (ROBOT_OOD_REF=raw)")
        self.seg_t = segment_tensors(self.table, dev)
        self.seg_pause = torch.as_tensor(segment_pause(self.nodes, self.table), device=dev)
        self.train_owners, self.heldout_owners = split_owners(self.nodes)
        self.entry_frac, self.entry_band = entry_stratum()
        self.dtw_thresh, self.phase_band = hard_negative_geometry()
        self.miner = HardNegativeMiner(self.table, dtw_thresh=self.dtw_thresh,
                                       phase_band=self.phase_band)
        self.ablation = self.spec.ablation
        self.phase_residual = None if self.ablation.graph else BeliefPhaseResidual(graph_dir)
        self.gate_dim = N_GATE
        self.language = load_segment_language(graph_dir, getattr(ctx, "logger", None),
                                              enabled=bool(self.spec.language))

        chunk_len = int(self.table.action_chunk.shape[1])
        self.net = FcsrNet(
            SegEncoders(
                QueryEncoder(self.nodes.dim, self.cfg.hist, self.cfg.hidden,
                             future_len=chunk_len, lang_dim=language_dim(self.language)),
                SegmentKeyEncoder(self.cfg.hidden),
                ActionEncoder(chunk_len, out_dim=ACTION_EMBED_DIM),
            ),
            SegmentScorer(),
            Corrector(self.plan_rows, k=self.cfg.seg_topk),
        ).to(dev)
        seg = self.net.seg
        self.retriever = SegmentRetriever(
            seg.query, seg.key, seg.action, self.net.scorer, self.table, cfg=self.cfg, device=dev,
        )
        self.corrector = self.net.corrector
        self.computer = SegmentLossComputer(self.retriever,
                                            kinematics,
                                            false_neg_band=self.phase_band)
        self.corr_loss = CorrectorLoss(self.corrector)
        if not self.ablation.graph:
            self._build_gate(graph_dir, dev)

        policy = PolicyActionCache.load(graph_dir).check_against(self.nodes)
        self.scale = ActionScale.from_json(graph_dir / schema.ACTION_SCALE_JSON)
        self.chunk_builder = ChunkTargetBuilder(
            nodes=self.nodes, scale=self.scale, policy=policy,
            kinematics=kinematics,
            chunk_len=self.plan_rows, cart=self.cart,
        )
        self.corruptor = RetrievalCorruptor(self.nodes)
        self.policy_corruptor = PolicyCorruptor(self.nodes, self.corruptor)
        self.copies = CopyPlan(self.cfg.query_copies, self.spec.seed)

        self.held = self._query_set(ctx.rng, self._n_eval, self.heldout_owners,
                                    entry=False, copies=None)
        self._fit_scorer_stats()
        return TrainUnit(trained={"fcsr": self.net}, param_groups=[
            {"params": [*seg.parameters(), *self.net.scorer.parameters()], "lr": self._lr},
            {"params": list(self.corrector.parameters()), "lr": self._lr},
        ])

    def _resolve_constants(self) -> Any:
        """The C0 calibration this run trains under, or None when nothing resolves one.

        Returns:
            A robot_ood.core.constants.Constants, or None (unit tests on synthetic graphs). The
            calibration is embedded in g_fcsr.npz at save, so deploy reads the identical bytes.
        """
        from robot_ood.core.config import calibrated_constants

        constants = calibrated_constants()
        if constants is not None:
            log_line(getattr(self.ctx, "logger", None),
                     f"[fcsr] constants: {constants.content_hash[:19]}")
        return constants

    def batches(self, ctx: Any, rng: Any) -> Any:
        """A freshly drawn training set each epoch, shuffled into steps.

        Args:
            ctx: The run's context.
            rng: The generator the draw comes from.

        Yields:
            One QueryBatch per optimiser step.
        """
        # An epoch is a row budget: on several ranks each draws its share, so the epoch is the
        # same amount of data whatever the world size (robot_ood.train.parallel).
        ranks = max(1, getattr(ctx.parallel, "world_size", 1))
        train_set = self._query_set(rng, max(1, self._n_query // ranks), self.train_owners,
                                    entry=True, copies=self.copies)
        return train_set.shuffled_batches(self._qbatch)

    def loss(self, unit: Any, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """The scheduled sum of the segment and corrector terms over one batch.

        Args:
            unit: The modules being trained.
            batch: One QueryBatch.

        Returns:
            (the weighted total, the two terms and this step's shortlist miss rate).
        """
        w_seg, w_corr = lambdas(self.epoch)
        batch, gate = self._with_belief(batch)
        l_seg = self.computer(self.seg_t, batch)
        total = w_seg * l_seg
        stats = {"l_seg": float(l_seg.detach()),
                 "tier1_miss_frac": float(self.computer.last_tier1_miss_frac)}
        if w_corr:
            parts = self._shortlist(batch, gate)
            out = self._corrector_forward(parts, batch)
            l_corr = self.corr_loss.loss_from(out, batch.a_policy, batch.a_corr, parts["qb"])
            total = total + w_corr * l_corr
            stats["l_corr"] = float(l_corr.detach())
        return total, stats

    def on_epoch(self, unit: Any, epoch: int) -> None:
        """Advance the loss schedule, rebuild tier 1, and stamp the corrector's feature statistics the epoch it joins."""
        self.epoch = epoch
        self.retriever.refresh_keys()
        if epoch == CORRECTOR_START_EPOCH:
            self._fit_corrector_stats()

    def evaluate(self, unit: Any, evalset: Any) -> dict[str, float]:
        """Held-out retrieval and corrector metrics.

        Args:
            unit: The modules being trained.
            evalset: Unread; the held-out set was drawn in build.

        Returns:
            The two metric families, each under its own prefix.
        """
        self.retriever.refresh_keys()
        metrics = self._segment_metrics()
        metrics.update(self._corrector_metrics())
        log_line(self.ctx.logger, corrector_report(metrics))
        return metrics

    def select(self, metrics: dict[str, float]) -> float | None:
        """No epoch is selected: the stage keeps its final weights."""
        return None

    def save(self, unit: Any, store: Any) -> None:
        """Write the retriever, scorer and corrector checkpoints under one segment graph hash.

        Args:
            unit: The trained modules.
            store: Where they go.
        """
        from robot_ood.models.retrieval.retriever import segment_graph_hash

        import json

        from robot_ood.core import npz

        graph_dir = store.dir
        # The stamp covers the exact calibration bytes this run trained under; embedding the same
        # bytes in the archive lets the loader recompute the stamp from what it carries.
        blob = (json.dumps(self.constants.to_dict(), indent=2, sort_keys=True) + "\n"
                if self.constants is not None else None)
        gh = segment_graph_hash(
            self.nodes, self.edges, blob if blob is not None else graph_dir / CONSTANTS_JSON,
            segment_len=int(self.table.action_chunk.shape[1]), seg_topk=int(self.cfg.seg_topk),
            scorer_state=state_bytes(unit.trained["fcsr"].scorer),
        )
        net = unit.trained["fcsr"]
        net.future_temp = self._fit_future_temp()
        path = net.save(graph_dir, graph_hash=gh)
        if blob is not None:
            arrays = dict(np.load(path, allow_pickle=False))
            arrays[schema.CONSTANTS_KEY] = np.array(blob)
            npz.write_arrays(path, schema.FCSR_NPZ, arrays)
        future_temp = net.future_temp
        log_line(self.ctx.logger,
                 f"[fcsr] saved graph_hash={gh} future_temp={future_temp:.6g}"
                 + ("" if blob is None else " constants=embedded"))

    # ---- batches ---------------------------------------------------------------------------------
    def _query_set(self, rng: Any, n: int, owners: Any, *, entry: bool, copies: Any) -> QuerySet:
        """One epoch's device-resident query set, carrying both objectives' extras."""
        from robot_ood.train.data import make_queries

        raw = make_queries(self.nodes, rng, n, owners, QuerySpec(
            window=self.cfg.hist, segments=self.table,
            segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0,
            entry_sigma_range=self._sigma_range,
            segment_entry_frac=self.entry_frac if entry else 0.0,
            segment_entry_band=self.entry_band, phase_residual=self.phase_residual,
        ), joint_space=self._joint_space)
        return QuerySet.from_queries(
            raw, self.ctx.device, self.chunk_builder, self.corruptor, rng,
            language=self.language, task_id=self.nodes.task_id,
            policy_corruptor=self.policy_corruptor, copies=copies,
            seg_miner=self.miner, n_seg_neg=N_SEG_NEG,
        )

    def _build_gate(self, graph_dir: Any, dev: Any) -> None:
        """The frozen locator and the nearest-node index behind the trust head's gate features."""
        from robot_ood.data.geometry import NeighborIndex
        from robot_ood.models.retrieval.gnn import GraphRetrieverNet
        from robot_ood.train.loss import NodeState
        from robot_ood.train.tensors import prepare_tensors

        self.locator = GraphRetrieverNet.load_npz(graph_dir / schema.HEAD_NPZ).to(dev).eval()
        for p in self.locator.parameters():
            p.requires_grad_(False)
        self.gt = prepare_tensors(self.nodes, self.edges, dev, cfg=self.cfg,
                                  language=self.language, chunk=self.locator.chunk)
        self.node_state = NodeState(self.locator, self.gt)
        self.neighbors = NeighborIndex.from_nodes(self.nodes, with_owner=True)

    @torch.no_grad()
    def _locator_view(self, qb: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """(gate [B, N_GATE], task_p [B, n_tasks]) from one belief pass, as deploy computes them.

        Args:
            qb: The batch, for its window, instruction and policy chunk.

        Returns:
            The trust head's gate columns and the belief's task marginal -- what the locator already
            concluded, handed down instead of re-derived. The off-manifold distance leaves the
            query's own demonstration out of the node set, as retrieval does: at deploy the
            trajectory being corrected is never in the graph, so its nearest node is another
            demonstration's, not its own.
        """
        from robot_ood.train.loss import forward_check
        from robot_ood.train.tasks.locator import truncate

        ones = torch.ones(qb.qh.shape[0], device=qb.qh.device)
        kw = dict(
            instr=qb.e_lang, has_instr=None if qb.e_lang is None else ones,
            policy=None if qb.a_policy is None else qb.a_policy[:, : self.locator.chunk],
            has_policy=None if qb.a_policy is None else ones,
        )
        cold = forward_check(self.locator, self.cfg, self.gt, self.node_state, qb.qh, qb.vh, qb.gh, **kw)
        if self.ablation.memory:
            belief = cold.belief()
        else:
            warm = forward_check(self.locator, self.cfg, self.gt, self.node_state, qb.qh, qb.vh, qb.gh,
                                 memory=truncate(cold.belief()), has_memory=ones, **kw)
            belief = warm.belief()
        entropy = -(belief.clamp_min(1e-12).log() * belief).sum(dim=-1)
        q_now = qb.qh[:, -1, :].detach().cpu().numpy().astype(np.float64)
        own = np.asarray(self.nodes.owner[np.asarray(qb.tgt, dtype=np.int64)], dtype=np.int64)
        off = torch.as_tensor(self.neighbors.min_distance(q_now, point_owner=own), dtype=qb.qh.dtype,
                              device=qb.qh.device)
        gate = torch.stack([off, entropy.to(qb.qh.dtype)], dim=-1)
        n_tasks = int(self.nodes.n_tasks)
        task_p = torch.zeros(belief.shape[0], n_tasks, dtype=belief.dtype, device=belief.device)
        task_p.index_add_(1, self.gt.task_id_t, belief)
        return gate, task_p.to(qb.qh.dtype)

    def _with_belief(self, qb: Any) -> tuple[Any, torch.Tensor]:
        """The batch with the locator's task marginal attached, and the gate columns, computed once."""
        import dataclasses

        if self.ablation.graph:
            return qb, torch.zeros(qb.qh.shape[0], self.gate_dim, dtype=qb.qh.dtype, device=qb.qh.device)
        gate, task_p = self._locator_view(qb)
        return dataclasses.replace(qb, task_p=task_p), gate

    def _shortlist(self, qb: Any, gate: torch.Tensor | None = None) -> dict[str, Any]:
        """The deployed shortlist for one batch, with column 0's corruption realised."""
        import dataclasses

        from robot_ood.train.loss import align_candidates

        if gate is None or (qb.task_p is None and not self.ablation.graph):
            qb, gate = self._with_belief(qb)
        task_p = qb.task_p
        # Leave-one-demo-out: at deploy the trajectory being corrected is never in the memory.
        own = torch.as_tensor(self.nodes.owner[np.asarray(qb.tgt, dtype=np.int64)], device=self.ctx.device)
        cand, query_phase = retrieve_candidates(self.retriever, self.computer, self.seg_t, qb, exclude_owner=own)
        chunk_len = qb.a_policy.shape[-2]
        cand_idx, cand_ee_start, cand_ee_end, cand_actions, ee_ref, realized = realize_column0(
            self.seg_t, qb, cand.idx,
            lambda idx: ee_ref_for(self.table, self.node_ee, idx, chunk_len, self.raw_ref), self.cart,
        )
        if realized is not None:
            qb = dataclasses.replace(qb, corrupt_kind=realized)
        return {
            "qb": qb, "cand_idx": cand_idx, "cand_score": cand.score.detach(),
            "cand_phase": self.seg_t.phase_start[cand_idx], "cand_actions": cand_actions,
            "cand_ee_start": cand_ee_start, "cand_ee_end": cand_ee_end,
            "query_phase": query_phase, "realized": realized,
            "aligned": align_candidates(cand_actions, chunk_len),
            "ee_ref": ee_ref,
            "gate": gate,
            "cand_task_id": self.seg_t.task_id[cand_idx],
            "cand_task_p": None if task_p is None else task_p.gather(-1, self.seg_t.task_id[cand_idx].to(torch.int64)),
            "cand_pause": self.seg_pause[cand_idx],
        }

    def _corrector_forward(self, parts: dict[str, Any], qb: Any) -> Any:
        """One corrector pass over a realised shortlist."""
        return self.corr_loss.forward(
            parts["aligned"], parts["cand_score"], parts["cand_phase"], qb.a_policy,
            parts["query_phase"], parts["ee_ref"], qb.ee_now, self.scale,
            parts["cand_ee_start"], parts["cand_ee_end"], parts["gate"],
            parts["cand_task_p"], parts["cand_task_id"], parts["cand_pause"], self.cart,
        )

    # ---- metrics ---------------------------------------------------------------------------------
    @torch.no_grad()
    def _fit_scorer_stats(self) -> None:
        """Measure the scorer's input standardisation on real shortlist features, before step one."""
        from robot_ood.models.retrieval import scorer as scorer_mod

        retriever = self.retriever
        feats = []
        for qb in self.held.ordered_batches(self._qbatch):
            qb, _ = self._with_belief(qb)
            ctx = self.computer.context(self.seg_t, qb)
            cand_idx = self.computer._candidates(qb, self.seg_t.n_segments)
            seg = retriever.gather(cand_idx)
            feats.append(scorer_mod.segment_features(
                ee_query=ctx.ee,
                e_future_q=retriever.action_encoder(retriever.query_encoder.predict_future(ctx)),
                phase_query=ctx.phase,
                seg_ee_start=seg.ee_start,
                e_future_seg=retriever.action_encoder(seg.action_chunk),
                seg_phase=seg.phase_start,
                cand_task_p=retriever.candidate_task_p(ctx, cand_idx),
            ).reshape(-1, scorer_mod.SEG_FEATURE_DIM))
        self.computer.scorer.fit_feature_stats(torch.cat(feats))

    @torch.no_grad()
    def _fit_future_temp(self) -> float:
        """The softmax temperature the trained scorer's own held-out logits imply."""
        from robot_ood.build.measure import bisect_temperature
        from robot_ood.core.schemas import SEG_TOPK

        self.retriever.refresh_keys()
        scores = []
        for qb in self.held.ordered_batches(self._qbatch):
            qb, _ = self._with_belief(qb)
            ctx = self.computer.context(self.seg_t, qb)
            cand = self.retriever.retrieve_batch(ctx)
            scores.append(cand.score.detach().cpu().numpy())
        score = np.concatenate(scores, axis=0).astype(np.float64)
        temp, _perplexity = bisect_temperature(score, float(SEG_TOPK))
        return temp

    @torch.no_grad()
    def _segment_metrics(self) -> dict[str, float]:
        """Held-out tier-1 recall, deploy hit rate, rank-1 and future-chunk MSE."""
        self.computer.score_temp = self.cfg.future_temp
        retriever = self.computer.retriever
        e_future_all = retriever.action_encoder(retriever.action_chunks)
        seg_keys = retriever.key_encoder(retriever._ee_start, retriever._phase_start, e_future_all)
        s = seg_keys.shape[0]
        tier1_m = min(int(retriever.cfg.seg_tier1_m), s)
        max_cut = min(max(TIER1_RECALL_CUTS), s)
        seg_topk = int(retriever.cfg.seg_topk)

        tier1_hits, future_mse, deploy_top1_hits, geom_top1_hits = 0.0, 0.0, 0.0, 0.0
        cut_hits = {m: 0.0 for m in TIER1_RECALL_CUTS}
        deploy_hits = {m: 0.0 for m in TIER1_RECALL_CUTS}
        n = 0
        for qb in self.held.ordered_batches(self._qbatch):
            qb, _ = self._with_belief(qb)
            out = self.computer.forward(self.seg_t, qb)
            true_chunk = self.seg_t.action_chunk[qb.seg_id.clamp_min(0)]
            future_mse += float(((out.pred_future - true_chunk) ** 2).mean(dim=(1, 2)).sum())

            ctx = self.computer.context(self.seg_t, qb)
            e_q = retriever.query_encoder(ctx)
            sim = e_q @ seg_keys.T
            target = qb.seg_id.clamp_min(0)[:, None]

            _, top_idx = sim.topk(tier1_m, dim=-1)
            tier1_hits += float((top_idx == target).any(dim=-1).sum())

            top1_score = retriever.score(ctx, e_q, top_idx)
            top1_pick = top_idx.gather(-1, top1_score.argmax(dim=-1, keepdim=True))
            deploy_top1_hits += float((top1_pick == target).sum())

            cand_ee = retriever._ee_start[top_idx]
            geom_pick = top_idx.gather(-1, ee_pose_delta_norm(
                ctx.ee[:, None, :], cand_ee,
            ).argmin(dim=-1, keepdim=True))
            geom_top1_hits += float((geom_pick == target).sum())

            _, ranked = sim.topk(max_cut, dim=-1)
            found_by = (ranked == target).float().cumsum(dim=-1) > 0
            for m in TIER1_RECALL_CUTS:
                k = min(m, max_cut)
                cut_hits[m] += float(found_by[:, k - 1].sum())
                shortlist = ranked[:, :k]
                score = retriever.score(ctx, e_q, shortlist)
                _, order = score.topk(min(seg_topk, k), dim=-1)
                deploy_hits[m] += float((shortlist.gather(-1, order) == target).any(dim=-1).sum())
            n += out.scores.shape[0]
        n = max(n, 1)
        metrics = {
            "seg_tier1_recall": tier1_hits / n, "seg_future_mse": future_mse / n,
            "seg_deploy_top1": deploy_top1_hits / n, "seg_tier1_miss_frac": 1.0 - tier1_hits / n,
            "seg_geom_top1": geom_top1_hits / n,
        }
        metrics.update({f"seg_tier1_recall@{m}": cut_hits[m] / n for m in TIER1_RECALL_CUTS})
        metrics.update({f"seg_deploy_hit@{m}": deploy_hits[m] / n for m in TIER1_RECALL_CUTS})
        return metrics

    @torch.no_grad()
    def _fit_corrector_stats(self) -> None:
        """Stamp the corrector's feature standardisation off the held-out shortlist."""
        from robot_ood.models.fusion.corrector import corrector_features

        z_row, z_chk = [], []
        for qb in self.held.ordered_batches(self._qbatch):
            parts = self._shortlist(qb)
            feats = corrector_features(
                qb.a_policy, parts["aligned"], parts["cand_score"], parts["cand_phase"],
                parts["query_phase"], parts["ee_ref"], qb.ee_now, self.scale,
                parts["cand_ee_start"], parts["cand_ee_end"], parts["gate"],
                parts["cand_task_p"], parts["cand_task_id"], cart=self.cart,
            )
            z_row.append(feats.z_row)
            z_chk.append(feats.z_chk)
        self.corrector.fit_feature_stats(torch.cat(z_row), torch.cat(z_chk))
        log_line(self.ctx.logger, "[fcsr] corrector feature statistics stamped")

    @torch.no_grad()
    def _corrector_metrics(self) -> dict[str, float]:
        """Held-out chunk MSE (policy, full reference, best constant, learned, oracle), trust AUC/acc, per-branch means."""
        from robot_ood.models.fusion.corrector import Corrector, optimal_dose
        from robot_ood.train.augment import KIND_NAMES, POLICY_KIND_NAMES
        from robot_ood.train.loss import ref_helps_label, trust_bce

        self.corrector.eval()
        pol, ref, exec_, corr, trust, shape, y_int, kinds, pkinds = ([] for _ in range(9))
        for qb in self.held.ordered_batches(self._qbatch):
            parts = self._shortlist(qb)
            out = self._corrector_forward(parts, qb)
            pol.append(qb.a_policy.cpu())
            ref.append(out.features.a_ref.cpu())
            exec_.append(out.a_exec.cpu())
            corr.append(qb.a_corr.cpu())
            trust.append(out.trust.cpu())
            shape.append(out.shape.mean(dim=(1, 2)).cpu())
            y_int.append(ref_helps_label(qb.a_policy, out.features.a_ref, qb.a_corr).cpu())
            kinds.append(parts["realized"])
            pkinds.append(qb.policy_kind)
        self.corrector.train()

        pi, ar, ex, ac = torch.cat(pol), torch.cat(ref), torch.cat(exec_), torch.cat(corr)
        ta = ac
        d = ta.shape[-1]

        def mse(chunk: torch.Tensor, y: torch.Tensor) -> float:
            return float(((chunk[..., :d] - y) ** 2).mean())

        def at_gain(c: torch.Tensor) -> torch.Tensor:
            return Corrector.blend(pi, ar, c.reshape(-1, 1, 1).expand(pi.shape[0], pi.shape[1], d))

        grid = {float(c): mse(at_gain(torch.full((pi.shape[0],), float(c))), ta)
                for c in np.linspace(0.0, 1.0, _DOSE_GRID)}
        best_const = min(grid, key=grid.__getitem__)
        t, y, sh = torch.cat(trust), torch.cat(y_int), torch.cat(shape)
        out = {
            "corr_mse_policy": mse(pi, ta), "corr_mse_full": mse(ar, ta),
            "corr_best_const": best_const, "corr_mse_best_const": grid[best_const],
            "corr_mse_learned": mse(ex, ta),
            "corr_mse_oracle": mse(at_gain(optimal_dose(pi, ar, ta)), ta),
            "trust_bce": float(trust_bce(t, y)),
            "trust_acc": float(((t > 0.5).float() == y).float().mean()),
            "trust_auc": _auc(t.numpy(), y.numpy()),
            "trust_pos_rate": float(y.mean()),
            "trust_mean_pos": float(t[y > 0.5].mean()) if bool((y > 0.5).any()) else float("nan"),
            "trust_mean_neg": float(t[y <= 0.5].mean()) if bool((y <= 0.5).any()) else float("nan"),
            "shape_mean": float(sh.mean()),
        }
        if all(k is not None for k in kinds) and all(k is not None for k in pkinds):
            kind, pkind = np.concatenate(kinds), np.concatenate(pkinds)
            t_np, sh_np = t.numpy(), sh.numpy()
            for names, ids, tag in ((KIND_NAMES, kind, "ret"), (POLICY_KIND_NAMES, pkind, "pol")):
                for k, name in enumerate(names):
                    m = ids == k
                    out[f"trust[{tag}:{name}]"] = float(t_np[m].mean()) if m.any() else float("nan")
                    out[f"shape[{tag}:{name}]"] = float(sh_np[m].mean()) if m.any() else float("nan")
        return out

    # ---- resolved run options --------------------------------------------------------------------
    @property
    def _qbatch(self) -> int:
        """Rows per optimiser step."""
        return QBATCH if self.spec.qbatch is None else self.spec.qbatch

    @property
    def _lr(self) -> float:
        """Learning rate for every parameter group."""
        return LR if self.spec.lr is None else self.spec.lr

    @property
    def _n_query(self) -> int:
        """Training queries drawn per epoch."""
        return N_QUERY if self.spec.n_query is None else self.spec.n_query

    @property
    def _n_eval(self) -> int:
        """Held-out queries."""
        return N_EVAL if self.spec.n_eval is None else self.spec.n_eval
