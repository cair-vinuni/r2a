"""Training-step losses: the locator's check loss, and FCSR's segment and corrector terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F

from robot_ood.core.config import GraphConfig
from robot_ood.data.metrics import NodeMetricColumns
from robot_ood.models.retrieval.gnn import GraphRetrieverNet
from robot_ood.models.types import MoveCost, NodeCloud, PipelineParams, QueryWindow
from robot_ood.train.target import Target
from robot_ood.train.tensors import graph_edges
from robot_ood.train.types import GraphTensors, QueryBatch

POINT_W = 0.3
RANK_W = 0.7

# The corrector's weight on the trust BCE, beside the chunk MSE.
TRUST_BCE_W = 1.0

_EPS = 1e-8


@dataclass(frozen=True)
class _Forward:
    """One check's batched forward.

    Attributes:
        logits: [B, V] node logits, -inf where unreached.
        reached: [B, V] bool reachability mask.
        e_q: [B, hidden] query embedding.
    """

    logits: torch.Tensor
    reached: torch.Tensor
    e_q: torch.Tensor

    def belief(self) -> torch.Tensor:
        return torch.nan_to_num(F.softmax(self.logits, dim=-1), nan=0.0)


class NodeState:
    """The query-independent node tensors of one graph, encoded once per batch."""

    def __init__(self, net: GraphRetrieverNet, gt: GraphTensors) -> None:
        node_x_t = gt.node_x_t
        self.n_v = net.node_repr(node_x_t, gt.e_task_t, gt.task_id_t)
        self.q_nodes_n = net.normalize_q(node_x_t[:, : net.dim])
        self.cloud = NodeCloud(
            q_nodes_n=self.q_nodes_n, n_v=self.n_v,
            columns=NodeMetricColumns(
                qdot=node_x_t[:, net.dim : 2 * net.dim], phase=gt.phase_t,
                grip=node_x_t[:, 2 * net.dim], task_id=gt.task_id_t,
            ),
        )
        self.edges = graph_edges(gt)


def forward_check(
    net: GraphRetrieverNet, cfg: GraphConfig, gt: GraphTensors, state: NodeState,
    qh: torch.Tensor, vh: torch.Tensor, gh: torch.Tensor, *,
    instr: torch.Tensor | None = None, has_instr: torch.Tensor | None = None,
    policy: torch.Tensor | None = None, has_policy: torch.Tensor | None = None,
    memory: torch.Tensor | None = None, has_memory: torch.Tensor | None = None,
    mask: torch.Tensor | None = None, instr_p: torch.Tensor | None = None,
) -> _Forward:
    """Run one check over a batch of windows.

    Args:
        net: The retriever being trained.
        cfg: Graph/training configuration.
        gt: Graph tensors shared by the batch.
        state: The batch's encoded node state.
        qh: [B, H, D] joint window.
        vh: [B, H, D] velocities.
        gh: [B, H] gripper.
        instr: [B, instr_dim] instruction embeddings, or None.
        has_instr: [B] float mask over instr, or None for all present.
        policy: [B, chunk, 7] policy chunks, or None.
        has_policy: [B] float mask over policy, or None for all present.
        memory: [B, V] the previous check's belief, or None.
        has_memory: [B] float mask over memory, or None for all present.
        mask: [V] bool nodes hidden from the query -- never seeded, reached or scored -- or None.
        instr_p: [B, n_tasks] the instruction's task posterior, or None.

    Returns:
        The forward.
    """
    query = net.query(QueryWindow(qh, vh, gh), instr, policy, has_instr, has_policy, instr_p=instr_p)
    h0, seeded = net.seed(
        query, state.cloud, PipelineParams(topk=cfg.seed_topk, tier1_m=cfg.seed_tier1_m),
        memory=memory, succ=gt.succ_t, has_memory=has_memory, mask=mask,
    )
    h_l, reached = net.propagate(h0, state.n_v, query.e_q, state.edges, checkpoint=False,
                                 seeded=seeded, mask=mask)
    logits = net.logits(
        h_l, state.n_v, reached, gt.log_idf, cfg.idf,
        MoveCost(q_nodes=state.q_nodes_n, q_now=query.qh_n[:, -1, :]),
        net.extra_features(query, state.edges),
    )
    return _Forward(logits=logits, reached=reached, e_q=query.e_q)


def _forward_batch(net: GraphRetrieverNet, cfg: GraphConfig, gt: GraphTensors, qb: QueryBatch) -> _Forward:
    """One check over a plain window batch: no instruction, no policy chunk, no memory."""
    return forward_check(net, cfg, gt, NodeState(net, gt), qb.qh, qb.vh, qb.gh)


def check_loss(node_logit: torch.Tensor, target: Target, reachable: torch.Tensor) -> torch.Tensor:
    """POINT_W * point_loss + RANK_W * rank_loss for one check.

    Args:
        node_logit: [V] node logits, -inf where unreached.
        target: The row's target set and weights.
        reachable: [V] bool.

    Returns:
        The scalar loss; zero when no target node was reached.
    """
    dev = node_logit.device
    tgt = torch.as_tensor(target.nodes, device=dev, dtype=torch.long)
    w = torch.as_tensor(target.weight, device=dev, dtype=node_logit.dtype)
    hit = reachable[tgt]
    if not bool(hit.any()):
        return 0.0 * node_logit[reachable].sum()
    tgt, w = tgt[hit], w[hit]
    w = w / w.sum()
    neg = reachable.clone()
    neg[tgt] = False
    point = -(w * F.logsigmoid(node_logit[tgt])).sum()
    if bool(neg.any()):
        point = point - F.logsigmoid(-node_logit[neg]).mean()
    rank = -(w * F.log_softmax(node_logit, dim=0)[tgt]).sum()
    return POINT_W * point + RANK_W * rank


def batch_loss(out: _Forward, gt: GraphTensors, tgt: np.ndarray, has_instr: np.ndarray) -> torch.Tensor:
    """Mean check_loss over the rows of one check."""
    rows = [check_loss(out.logits[bi], gt.targets(int(tgt[bi]), bool(has_instr[bi])), out.reached[bi])
            for bi in range(out.logits.shape[0])]
    return torch.stack(rows).mean()


class QueryLossComputer:
    """One training step over a plain window batch, mean check_loss over the rows."""

    def __init__(self, net: GraphRetrieverNet, cfg: GraphConfig) -> None:
        self.net, self.cfg = net, cfg

    def __call__(self, gt: GraphTensors, qb: QueryBatch) -> torch.Tensor:
        out = _forward_batch(self.net, self.cfg, gt, qb)
        return batch_loss(out, gt, qb.tgt, np.ones(len(qb.tgt), dtype=bool))


# ==== FCSR loss-term weights: the segment and fusion halves of the one objective ====
SEGMENT_PHASE_CE_W: Final[float] = 0.3
SEGMENT_INFONCE_W: Final[float] = 1.0
SEGMENT_FUTURE_W: Final[float] = 0.5
SEGMENT_SHORTLIST_CE_W: Final[float] = 0.5
SEGMENT_SCORE_TEMP: Final[float] = 1.0
SEGMENT_PHASE_TEMP: Final[float] = 0.1
SEGMENT_N_FILLER: Final[int] = 4
SEGMENT_INFONCE_TEMP: Final[float] = 1.0
# ASSUMED, not measured: reused from HardNegativeMiner's own same-magnitude fallback.
SEGMENT_FALSE_NEG_PHASE_BAND: Final[float] = 0.05
# Pose half-width of the in-batch false-negative mask (unscaled pose units); 0 keeps the mask on
# task and phase alone. Set together with ROBOT_OOD_HARDNEG_STATE, see robot_ood.train.negatives.
NEG_POSE_BAND: Final[float] = float(__import__("os").environ.get("ROBOT_OOD_NEG_POSE_BAND", "0"))


@dataclass(frozen=True, slots=True)
class SegmentOutputs:
    """One SEGMENT forward pass, kept whole so the loss and any diagnostic read the same numbers.

    Attributes:
        scores: [B, K] SegmentScorer compatibility scores over the candidate set.
        weights: [B, K] softmax(scores / temp) -- the ONLY differentiable path back to the segment
            memory; candidate ACTIONS themselves are gathered, not produced by the encoder.
        cand_idx: [B, K] int64 segment ids the candidate set was built from. Column 0 is always the
            row's own true continuation.
        pred_future: [B, SEGMENT_LEN, ACTION_DIM] weights-weighted mixture of the candidates' own
            action chunks -- the chunk future_loss regresses towards the true continuation.
        e_q: [B, hidden] this row's fused query embedding -- infonce_global's query side, computed
            once here so it need not be recomputed from ctx a second time.
    """

    scores: torch.Tensor
    weights: torch.Tensor
    cand_idx: torch.Tensor
    pred_future: torch.Tensor
    e_q: torch.Tensor


class SegmentLossComputer:
    """SEGMENT loss: 0.3*phase_ce + 1.0*infonce_global + 0.5*future_loss + 0.5*shortlist_ce."""

    def __init__(
        self, retriever: Any, kinematics: Any, *,
        score_temp: float = SEGMENT_SCORE_TEMP,
        phase_temp: float = SEGMENT_PHASE_TEMP, phase_ce_w: float = SEGMENT_PHASE_CE_W,
        infonce_w: float = SEGMENT_INFONCE_W, future_w: float = SEGMENT_FUTURE_W,
        shortlist_ce_w: float = SEGMENT_SHORTLIST_CE_W, infonce_temp: float = SEGMENT_INFONCE_TEMP,
        false_neg_band: float = SEGMENT_FALSE_NEG_PHASE_BAND,
    ) -> None:
        """Bind the computer to the retriever, kinematics and the SEGMENT loss-term weights.

        Args:
            retriever: The SegmentRetriever (trainable), scored through so its four encoders take a
                gradient together and _deploy_shortlist reads the cfg deploy resolves.
            kinematics: robot_ood.core.kinematics.PandaKinematics, mapping joints to the [B, 6]
                end-effector pose QueryContext.ee needs.
            score_temp: Softmax temperature over the candidate scores.
            phase_temp: Decay length of phase_ce's soft target.
            phase_ce_w: Weight of the phase term.
            infonce_w: Weight of infonce_global.
            future_w: Weight of future_loss.
            shortlist_ce_w: Weight of shortlist_ce.
            infonce_temp: Temperature over infonce_global's logits.
            false_neg_band: Phase half-width within which a same-task in-batch candidate is masked
                out of infonce_global's denominator instead of counting as a negative.
        """
        self.retriever = retriever
        self.kinematics = kinematics
        self.score_temp, self.phase_temp = score_temp, phase_temp
        self.phase_ce_w, self.infonce_w, self.future_w, self.shortlist_ce_w = (
            phase_ce_w, infonce_w, future_w, shortlist_ce_w
        )
        self.infonce_temp, self.false_neg_band = infonce_temp, false_neg_band
        # Set by _deploy_shortlist on every call -- the tier-1 miss diagnostic, log-only.
        self.last_tier1_miss_frac: float = float("nan")

    @property
    def query_encoder(self) -> Any:
        """The QueryEncoder, for callers that train or inspect it directly."""
        return self.retriever.query_encoder

    @property
    def action_encoder(self) -> Any:
        """The ActionEncoder, likewise."""
        return self.retriever.action_encoder

    @property
    def scorer(self) -> Any:
        """The SegmentScorer, likewise."""
        return self.retriever.fine_check

    def context(self, gt: Any, qb: QueryBatch) -> Any:
        """This batch's QueryContext.

        Returns:
            QueryContext with ee observed (never estimated) and phase from belief filter or ground truth.
        """
        from robot_ood.core.schemas import QueryContext

        # PandaKinematics.pose is numpy in/out (no gradient by design).
        q_now = qb.qh[:, -1, :].detach().cpu().numpy()
        ee = torch.as_tensor(self.kinematics.pose(q_now), dtype=qb.qh.dtype, device=qb.qh.device)
        # Use belief filter estimate when available, else ground truth (deployed path never has truth).
        phase = (
            gt.phase_start[qb.seg_id.clamp_min(0)] if qb.phase_hat is None
            else qb.phase_hat.to(dtype=qb.qh.dtype)
        )
        return QueryContext(
            qh=qb.qh, vh=qb.vh, gh=qb.gh, e_lang=qb.e_lang, a_hist=qb.a_hist, ee=ee, phase=phase,
            task_p=qb.task_p,
        )

    def _candidates(self, qb: QueryBatch, n_seg: int) -> torch.Tensor:
        """Column 0 = own true segment, then the mined negatives, rest = in-batch fillers.

        Args:
            qb: This step's queries; seg_id and neg_segments are required.
            n_seg: S, the segment table's size -- fillers wrap modulo this so an index is always
                legal even on a tiny synthetic table.

        Returns:
            [B, K] int64 candidate segment ids.
        """
        b = qb.seg_id.shape[0]
        cols = [qb.seg_id[:, None]]
        if qb.neg_segments is not None:
            cols.append(qb.neg_segments)
        filler_n = min(SEGMENT_N_FILLER, max(b - 1, 0))
        if filler_n > 0:
            shifts = torch.arange(1, filler_n + 1, device=qb.seg_id.device)
            fillers = torch.stack(
                [qb.seg_id[(torch.arange(b, device=qb.seg_id.device) + s) % b] for s in shifts],
                dim=1,
            )
            cols.append(fillers)
        return torch.cat(cols, dim=1).clamp_min(0) if len(cols) > 1 else cols[0].clamp_min(0)

    def forward(self, gt: Any, qb: QueryBatch, ctx: Any = None) -> SegmentOutputs:
        """The scored candidate set and the differentiable predicted-future mixture.

        Args:
            gt: robot_ood.train.negatives.SegmentTensors for this run's segment table.
            qb: This step's queries; seg_id and neg_segments are required.
            ctx: The batch's QueryContext, if the caller already built one; None builds it here.

        Returns:
            The forward pass.
        """
        ctx = self.context(gt, qb) if ctx is None else ctx
        cand_idx = self._candidates(qb, gt.n_segments)
        # Computed once more here so e_q is available to infonce_global.
        e_q = self.retriever.query_encoder(ctx)
        scores = self.retriever.score_batch(ctx, cand_idx)
        weights = F.softmax(scores / max(self.score_temp, _EPS), dim=-1)
        cand_chunks = gt.action_chunk[cand_idx]                          # [B, K, SEGMENT_LEN, 7]
        pred_future = (weights[..., None, None] * cand_chunks).sum(dim=1)
        return SegmentOutputs(
            scores=scores, weights=weights, cand_idx=cand_idx, pred_future=pred_future, e_q=e_q,
        )

    def _key(self, gt: Any, seg_id: torch.Tensor) -> torch.Tensor:
        """Shortlist key for seg_id: matches SegmentRetriever.score exactly.

        Args:
            gt: robot_ood.train.negatives.SegmentTensors.
            seg_id: [...] int64 segment ids, any shape.

        Returns:
            [..., hidden].
        """
        e_future = self.retriever.action_encoder(gt.action_chunk[seg_id])
        return self.retriever.key_encoder(gt.ee_start[seg_id], gt.phase_start[seg_id], e_future)

    def _infonce_global(self, gt: Any, qb: QueryBatch, e_q: torch.Tensor) -> torch.Tensor:
        """Global InfoNCE: e_q vs true key against in-batch and mined negatives.

        Args:
            gt: robot_ood.train.negatives.SegmentTensors.
            qb: This step's queries; seg_id is required (neg_segments optional).
            e_q: [B, hidden] this row's query embedding, from SegmentOutputs.e_q.

        Returns:
            The scalar mean cross-entropy, label 0 (the true continuation) always column 0.
        """
        own_id = qb.seg_id.clamp_min(0)
        pos_id = own_id if qb.pos_segments is None else qb.pos_segments.clamp_min(0)
        pos_key = self._key(gt, pos_id)                                    # [B, hidden], grad-carrying
        pos_logit = (e_q * pos_key).sum(-1, keepdim=True)                  # [B, 1]

        task_i, phase_i = gt.task_id[pos_id], gt.phase_start[pos_id]       # [B], [B]
        b = e_q.shape[0]

        # ---- in-batch: every OTHER row's own true continuation --------------------------------
        batch_sim = e_q @ pos_key.T                                        # [B, B]
        self_mask = torch.eye(b, dtype=torch.bool, device=batch_sim.device)
        false_neg = (task_i[:, None] == task_i[None, :]) & (
            (phase_i[:, None] - phase_i[None, :]).abs() <= self.false_neg_band
        )
        if NEG_POSE_BAND > 0.0:
            # ROBOT_OOD_NEG_POSE_BAND: a same-task, same-phase row is only a false negative when its
            # start pose is also within the band; a pose-far one is a legitimate negative, so the
            # query embedding has to carry the arm's configuration and not just task and phase.
            ee_i = gt.ee_start[pos_id]
            false_neg = false_neg & ((ee_i[:, None, :] - ee_i[None, :, :]).norm(dim=-1) <= NEG_POSE_BAND)
        if qb.pos_segments is not None:
            # another row's positive may be this row's own segment (or its positive): never a negative
            false_neg = false_neg | (pos_id[None, :] == own_id[:, None]) | (pos_id[None, :] == pos_id[:, None])
        batch_sim = batch_sim.masked_fill(self_mask | false_neg, float("-inf"))

        # ==== mined hard negatives: ADDED to the denominator, not swapped in for it ====
        if qb.neg_segments is not None and qb.neg_segments.shape[1] > 0:
            neg_valid = qb.neg_segments >= 0
            neg_key = self._key(gt, qb.neg_segments.clamp_min(0))          # [B, n_neg, hidden]
            hard_sim = (e_q[:, None, :] * neg_key).sum(-1).masked_fill(~neg_valid, float("-inf"))
        else:
            hard_sim = batch_sim.new_full((b, 0), float("-inf"))

        logits = torch.cat([pos_logit, batch_sim, hard_sim], dim=1)
        logits = logits / max(self.infonce_temp, _EPS)
        labels = torch.zeros(b, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def _deploy_shortlist(
        self, gt: Any, qb: QueryBatch, e_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tier-1 top-M shortlist with mined negatives and shuffled-column true positive.

        Args:
            gt: robot_ood.train.negatives.SegmentTensors.
            qb: This step's queries; seg_id required, neg_segments optional.
            e_q: [B, hidden] query embedding (no_grad for tier-1 pick only).

        Returns:
            (cand_idx [B, K], label [B], tier1_hit [B]).
        """
        pos_id = qb.seg_id.clamp_min(0) if qb.pos_segments is None else qb.pos_segments.clamp_min(0)
        b, dev = pos_id.shape[0], pos_id.device
        with torch.no_grad():
            e_future_all = self.retriever.action_encoder(gt.action_chunk)
            seg_keys = self.retriever.key_encoder(gt.ee_start, gt.phase_start, e_future_all)
            s = seg_keys.shape[0]
            # M is retriever.cfg.seg_tier1_m, the SAME cfg deploy resolves -- not a training constant.
            m = min(int(self.retriever.cfg.seg_tier1_m), s)
            sim = e_q.detach() @ seg_keys.T                                # [B, S]
            _, top_idx = sim.topk(m, dim=-1)                               # [B, m] -- deploy's own cut

            tier1_hit = (top_idx == pos_id[:, None]).any(dim=-1)           # [B] -- BEFORE forcing

            pos_col = torch.randint(0, m, (b,), device=dev)
            shortlist = top_idx.scatter(-1, pos_col[:, None], pos_id[:, None])

            if qb.neg_segments is not None and qb.neg_segments.shape[1] > 0:
                cand_idx = torch.cat([shortlist, qb.neg_segments.clamp_min(0)], dim=-1)
            else:
                cand_idx = shortlist

            k = cand_idx.shape[1]
            shuffle = torch.argsort(torch.rand(b, k, device=dev), dim=-1)
            cand_idx = cand_idx.gather(-1, shuffle)
            label = (cand_idx == pos_id[:, None]).float().argmax(dim=-1)
        return cand_idx, label, tier1_hit

    def _shortlist_ce(self, gt: Any, qb: QueryBatch, e_q: torch.Tensor, ctx: Any) -> torch.Tensor:
        """Tier-2 cross-entropy over deployed M-shortlist: ranks true segment.

        Args:
            gt: robot_ood.train.negatives.SegmentTensors.
            qb: This step's queries.
            e_q: [B, hidden] this row's query embedding, for _deploy_shortlist's tier-1 cut.
            ctx: The batch's QueryContext, built once by __call__.

        Returns:
            The scalar mean cross-entropy, label = the true continuation's (shuffled) column.
        """
        cand_idx, label, tier1_hit = self._deploy_shortlist(gt, qb, e_q)
        self.last_tier1_miss_frac = float((~tier1_hit).to(torch.float32).mean())
        scores = self.retriever.score_batch(ctx, cand_idx)
        return F.cross_entropy(scores, label)

    def __call__(self, gt: Any, qb: QueryBatch) -> torch.Tensor:
        """Run one training step's forward pass and reduce it to a scalar loss.

        Args:
            gt: robot_ood.train.negatives.SegmentTensors.
            qb: This step's queries; seg_id and neg_segments are required.

        Returns:
            The MEAN loss over the batch, ready for .backward().
        """
        ctx = self.context(gt, qb)
        out = self.forward(gt, qb, ctx)
        true_chunk = gt.action_chunk[qb.seg_id.clamp_min(0)]

        infonce = self._infonce_global(gt, qb, out.e_q)
        future = F.mse_loss(out.pred_future, true_chunk)
        shortlist_ce = self._shortlist_ce(gt, qb, out.e_q, ctx)

        phase_ce = self._phase_ce(gt, qb, out)
        loss = (
            self.phase_ce_w * phase_ce + self.infonce_w * infonce + self.future_w * future
            + self.shortlist_ce_w * shortlist_ce
        )
        return loss

    def _phase_ce(self, gt: Any, qb: QueryBatch, out: SegmentOutputs) -> torch.Tensor:
        """CE between candidate weights and a soft target peaked at the phase-nearest candidates.

        Args:
            gt: robot_ood.train.negatives.SegmentTensors.
            qb: This step's queries.
            out: This step's forward pass.

        Returns:
            The scalar mean CE.
        """
        phase_query = gt.phase_start[qb.seg_id.clamp_min(0)][:, None]
        cand_phase = gt.phase_start[out.cand_idx]
        y = torch.exp(-(cand_phase - phase_query).abs() / max(self.phase_temp, _EPS))
        y = y / y.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        log_p = torch.log(out.weights.clamp_min(_EPS))
        return -(y * log_p).sum(dim=-1).mean()


def align_candidates(cand_actions: torch.Tensor, chunk_len: int) -> torch.Tensor:
    """Resample candidate chunks to policy time axis via interpolation.

    Args:
        cand_actions: [B, K, T, ACTION_DIM].
        chunk_len: The policy's chunk length.

    Returns:
        [B, K, chunk_len, ACTION_DIM]; the input unchanged when T == chunk_len.
    """
    from robot_ood.models.fusion.corrector import resample_torch

    if cand_actions.shape[-2] == chunk_len:
        return cand_actions
    b, k = cand_actions.shape[:2]
    flat = resample_torch(cand_actions.reshape(b * k, *cand_actions.shape[2:]), chunk_len)
    return flat.reshape(b, k, *flat.shape[1:])


# y_int = the policy chunk was corrupted and the retrieval was not: y is a_corr there and a_policy
# elsewhere, so a clean row and a wide reference both pull the gain to 0.
class CorrectorLoss:
    """mse(a_exec, a_corr) + trust_w * bce(trust, ref_helps): the corrector's one objective.

    a_corr is the query's own demo continued from the live pose, on every row: where the retrieval
    was corrupted the reference sits far from it and the dose is pushed down, where the policy was
    corrupted the reference sits on it and the dose is pushed up, and on a clean row the dose
    settles at what the neighbouring demos are worth. The trust head is also asked directly whether
    the reference lies closer to a_corr than the policy does, as a plain BCE: its answer is the
    probability that the reference helps, class prior included, because a deployed check is
    usually one where it does not and the dose is trust times shape.
    """

    def __init__(self, corrector: Any, *, trust_w: float = TRUST_BCE_W) -> None:
        """Bind the loss to the corrector and its trust weight.

        Args:
            corrector: robot_ood.models.fusion.corrector.Corrector (trainable).
            trust_w: Weight of the trust BCE.
        """
        self.corrector, self.trust_w = corrector, trust_w

    def forward(
        self, cand_actions: torch.Tensor, cand_score: torch.Tensor, cand_phase: torch.Tensor,
        a_policy: torch.Tensor, query_phase: torch.Tensor, ee_ref: torch.Tensor,
        ee_now: torch.Tensor, scale: Any, cand_ee_start: torch.Tensor, cand_ee_end: torch.Tensor,
        gate: torch.Tensor | None = None, cand_task_p: torch.Tensor | None = None,
        cand_task_id: torch.Tensor | None = None, cand_pause: torch.Tensor | None = None,
        cart: Any = None,
    ) -> Any:
        """Run the corrector once on a shortlist, aligning the candidates to the policy's chunk.

        Args:
            cand_actions: [B, K, T, 7] frozen-retriever candidate action chunks.
            cand_score: [B, K] frozen-retriever candidate scores.
            cand_phase: [B, K] frozen-retriever candidate phases.
            a_policy: [B, C, 7] the frozen policy's own chunk.
            query_phase: [B] the query's own current phase estimate.
            ee_ref: [B, K, C, 6] each candidate's own node-interpolated pose trajectory.
            ee_now: [B, 6] the query window's own live end-effector pose.
            scale: Metres/radians per action unit (robot_ood.core.action.ActionScale).
            cand_ee_start: [B, K, 6] each candidate's own ee_start, post realize_column0.
            cand_ee_end: [B, K, 6] each candidate's own ee_end, likewise.
            gate: [B, N_GATE] the locator's view of the check, or None.
            cand_task_p: [B, K] the belief's mass on each candidate's task, or None.
            cand_task_id: [B, K] each candidate's task id, or None.
            cand_pause: [B, K] whether each candidate is a pause, or None.
            cart: The arm's chain on a joint-space lane, or None; see corrector_features.

        Returns:
            The robot_ood.models.fusion.corrector.CorrectorOutputs forward pass.
        """
        cand_actions = align_candidates(cand_actions, a_policy.shape[-2])
        return self.corrector(
            a_policy, cand_actions, cand_score, cand_phase, query_phase, ee_ref, ee_now, scale,
            cand_ee_start, cand_ee_end, gate, cand_task_p, cand_task_id, cand_pause, cart,
        )

    def loss_from(self, out: Any, a_policy: torch.Tensor, a_corr: torch.Tensor, qb: QueryBatch) -> torch.Tensor:
        """The scalar loss for an already-computed forward pass.

        Args:
            out: What forward() returned for this batch.
            a_policy: [B, C, 7] the frozen policy's own chunk.
            a_corr: [B, C, 6] the pose-derived correction chunk.
            qb: This step's queries; unread, kept so the call site reads like FcsrTask.loss.

        Returns:
            The MEAN loss over the batch, ready for .backward().
        """
        mse = ((out.a_exec[..., : a_corr.shape[-1]] - a_corr) ** 2).mean()
        helps = ref_helps_label(a_policy, out.features.a_ref, a_corr)
        return mse + self.trust_w * trust_bce(out.trust, helps)


def ref_helps_label(a_policy: torch.Tensor, a_ref: torch.Tensor, a_corr: torch.Tensor) -> torch.Tensor:
    """The reference chunk is closer to the correction than the policy's own, as a float [B] target.

    Args:
        a_policy: [B, C, 7] the policy's chunk.
        a_ref: [B, C, 6] the reference chunk the corrector would mix toward.
        a_corr: [B, C, 6] the pose-derived correction chunk.

    Returns:
        [B] float32 in {0, 1}; a tie is 0.
    """
    d = a_corr.shape[-1]
    err_policy = ((a_policy[..., :d] - a_corr) ** 2).flatten(1).mean(1)
    err_ref = ((a_ref.detach()[..., :d] - a_corr) ** 2).flatten(1).mean(1)
    return (err_ref < err_policy).to(torch.float32)


def trust_bce(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean BCE(p, y), so p is calibrated to the label's own rate.

    Args:
        p: [B] probabilities in (0, 1).
        y: [B] float targets in {0, 1}.

    Returns:
        The scalar loss.
    """
    return F.binary_cross_entropy(p.clamp(_EPS, 1.0 - _EPS), y)
