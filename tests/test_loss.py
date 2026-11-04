"""Tests for FCSR's SegmentLossComputer and CorrectorLoss -- robot_ood.train.loss.

These tests stub robot_ood.models.retrieval.scorer's one free function, segment_features, by
monkeypatching sys.modules -- the same reason SegmentLossComputer/SegmentRetriever import it lazily.
The retriever itself is real: SegmentLossComputer scores through SegmentRetriever.score_batch (the
match term carried in additively), so a hand-rolled stand-in would not exercise the thing these
tests are for -- only QueryEncoder/ActionEncoder/SegmentScorer stay small fakes.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot_ood.core.action import ActionScale
from robot_ood.core.config import GraphConfig
from robot_ood.core.schemas import SEG_FEATURE_DIM, ACTION_EMBED_DIM, SegmentTable
from robot_ood.models.retrieval.encoders import SegmentKeyEncoder
from robot_ood.models.retrieval.retriever import SegmentRetriever
from robot_ood.train.negatives import SegmentTensors, segment_tensors
from robot_ood.train.loss import (
    CorrectorLoss, SegmentLossComputer, TRUST_BCE_W, align_candidates, trust_bce,
    ref_helps_label,
)
from robot_ood.train.types import QueryBatch

SEGMENT_LEN = 4
ACTION_DIM = 7
HIDDEN = 8
EMBED = ACTION_EMBED_DIM   # SegmentKeyEncoder's in_dim is fixed to contracts.ACTION_EMBED_DIM


# ======================================================================================================
# stand-ins for the scorer/retriever modules, built strictly against the documented signatures
# ======================================================================================================
class _FakeQueryEncoder(nn.Module):
    def __init__(self, dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.future = nn.Linear(dim, SEGMENT_LEN * ACTION_DIM)

    def forward(self, ctx):
        self.last_ctx = ctx
        return self.proj(ctx.qh[:, -1, :])

    def predict_future(self, ctx):
        b = ctx.qh.shape[0]
        return self.future(ctx.qh[:, -1, :]).reshape(b, SEGMENT_LEN, ACTION_DIM)


class _FakeActionEncoder(nn.Module):
    def __init__(self, chunk_len: int, out_dim: int = EMBED):
        super().__init__()
        self.proj = nn.Linear(chunk_len * ACTION_DIM, out_dim)

    def forward(self, chunk):
        return self.proj(chunk.reshape(*chunk.shape[:-2], -1))


class _FakeSegmentScorer(nn.Module):
    def __init__(self, feature_dim: int = SEG_FEATURE_DIM):
        super().__init__()
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, feats):
        return self.head(feats).squeeze(-1)


class _FakeKinematics:
    def pose(self, q):
        d = q.shape[-1]
        return q[..., :6] if d >= 6 else np.pad(q, ((0, 0), (0, 6 - d)))


def _fake_segment_features(
    *, ee_query, e_future_q, phase_query, seg_ee_start, e_future_seg, seg_phase, cand_task_p,
):
    """A real, differentiable implementation matching robot_ood.core.schemas.SEG_FEATURES's order,
    to exercise SegmentLossComputer's plumbing without the real scorer module."""
    d_state = (ee_query[:, None, :] - seg_ee_start).norm(dim=-1)
    d_future = 1.0 - F.cosine_similarity(e_future_q[:, None, :], e_future_seg, dim=-1, eps=1e-8)
    phase_diff = phase_query[:, None] - seg_phase
    return torch.stack([d_state, d_future, phase_diff, cand_task_p.to(d_state.dtype)], dim=-1)


@pytest.fixture(autouse=True)
def _stub_fcsr_scorer(monkeypatch):
    fake = types.ModuleType("robot_ood.models.retrieval.scorer")
    fake.segment_features = _fake_segment_features
    fake.ActionEncoder = _FakeActionEncoder
    fake.SegmentScorer = _FakeSegmentScorer
    monkeypatch.setitem(sys.modules, "robot_ood.models.retrieval.scorer", fake)


# ======================================================================================================
# fixtures
# ======================================================================================================
def _seg_table(s: int = 12) -> SegmentTable:
    """The numpy SegmentTable both gt (via segment_tensors) and the test SegmentRetriever are built
    from -- in production both always come from the SAME C1 table, so the test fixtures do too."""
    g = torch.Generator().manual_seed(0)
    action_chunk = torch.randn(s, SEGMENT_LEN, ACTION_DIM, generator=g).numpy().astype(np.float32)
    phase_start = torch.rand(s, generator=g).numpy().astype(np.float32)
    task_id = torch.randint(0, 3, (s,), generator=g).numpy().astype(np.int16)
    ee_start = torch.randn(s, 6, generator=g).numpy().astype(np.float32)
    return SegmentTable(
        start_node=np.arange(s, dtype=np.int32), end_node=np.arange(s, dtype=np.int32),
        owner=np.arange(s, dtype=np.int32), task_id=task_id,
        phase_start=phase_start, phase_end=phase_start.copy(),
        action_chunk=action_chunk, ee_start=ee_start, ee_end=ee_start.copy(),node_to_segment=np.arange(s, dtype=np.int32),
    )


def _seg_tensors(s: int = 12) -> SegmentTensors:
    return segment_tensors(_seg_table(s), "cpu")


def _query_batch(b: int, dim: int = 5, s: int = 12, n_neg: int = 3, window: int = 3) -> QueryBatch:
    g = torch.Generator().manual_seed(1)
    return QueryBatch(
        qh=torch.randn(b, window, dim, generator=g), vh=torch.zeros(b, window, dim),
        gh=torch.zeros(b, window),
        tgt=np.arange(b),
        seg_id=torch.randint(0, s, (b,)),
        neg_segments=torch.randint(0, s, (b, n_neg)),
    )


def _segment_computer(dim: int = 5, s: int = 12) -> SegmentLossComputer:
    retriever = SegmentRetriever(
        _FakeQueryEncoder(dim), SegmentKeyEncoder(HIDDEN), _FakeActionEncoder(SEGMENT_LEN),
        _FakeSegmentScorer(), _seg_table(s), cfg=GraphConfig(),
    )
    for m in (retriever.query_encoder, retriever.key_encoder, retriever.action_encoder,
              retriever.fine_check):
        m.train()
    return SegmentLossComputer(retriever, _FakeKinematics())


# ======================================================================================================
# SegmentLossComputer
# ======================================================================================================
def test_forward_candidate_set_has_the_true_segment_in_column_zero():
    gt, qb = _seg_tensors(), _query_batch(b=6)
    out = _segment_computer().forward(gt, qb)
    assert torch.equal(out.cand_idx[:, 0], qb.seg_id)


def test_forward_weights_sum_to_one_and_are_nonnegative():
    gt, qb = _seg_tensors(), _query_batch(b=6)
    out = _segment_computer().forward(gt, qb)
    assert torch.allclose(out.weights.sum(dim=-1), torch.ones(out.weights.shape[0]), atol=1e-5)
    assert bool((out.weights >= 0).all())


def test_call_returns_a_finite_scalar_with_gradient_to_every_trained_module():
    computer = _segment_computer()
    gt, qb = _seg_tensors(), _query_batch(b=6)
    loss = computer(gt, qb)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for module in (computer.query_encoder, computer.action_encoder, computer.scorer):
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())


def test_hard_negative_loss_pushes_true_score_above_mined_negatives():
    """A few optimiser steps on a toy problem should raise scores[:, 0] relative to the mined
    negative columns -- the margin term's whole purpose."""
    torch.manual_seed(0)
    computer = _segment_computer()
    opt = torch.optim.Adam(
        list(computer.query_encoder.parameters()) + list(computer.action_encoder.parameters())
        + list(computer.scorer.parameters()), lr=0.05,
    )
    gt, qb = _seg_tensors(), _query_batch(b=8)
    with torch.no_grad():
        gap0 = computer.forward(gt, qb).scores[:, 0].mean() - computer.forward(gt, qb).scores[:, 1].mean()
    for _ in range(50):
        opt.zero_grad()
        computer(gt, qb).backward()
        opt.step()
    with torch.no_grad():
        gap1 = computer.forward(gt, qb).scores[:, 0].mean() - computer.forward(gt, qb).scores[:, 1].mean()
    assert gap1 > gap0


def test_candidates_column_count_bounded_by_hard_negatives_plus_fillers():
    gt, qb = _seg_tensors(), _query_batch(b=6, n_neg=3)
    out = _segment_computer().forward(gt, qb)
    # 1 true + 3 mined + up to SEGMENT_N_FILLER(4) in-batch fillers.
    assert out.cand_idx.shape[1] <= 1 + 3 + 4


def test_no_neg_segments_still_produces_a_valid_forward_pass():
    qb = _query_batch(b=6)
    qb_no_neg = QueryBatch(**{**qb.__dict__, "neg_segments": None})
    out = _segment_computer().forward(_seg_tensors(), qb_no_neg)
    assert torch.isfinite(out.scores).all()


def test_context_threads_the_batchs_a_hist_into_the_query_encoder():
    """Gap A: a_hist must reach QueryEncoder.forward, not be dropped on the floor -- the bug that
    made GR_ACTION_HIST=0 indistinguishable from GR_ACTION_HIST=1."""
    computer = _segment_computer()
    qb = _query_batch(b=6)
    qb = QueryBatch(**{**qb.__dict__, "a_hist": torch.randn(6, 3, ACTION_DIM)})
    computer.forward(_seg_tensors(), qb)
    seen = computer.query_encoder.last_ctx.a_hist
    assert seen is not None
    assert torch.equal(seen, qb.a_hist)


def test_context_a_hist_is_none_when_the_batch_carries_none():
    computer = _segment_computer()
    qb = _query_batch(b=6)
    assert qb.a_hist is None
    computer.forward(_seg_tensors(), qb)
    assert computer.query_encoder.last_ctx.a_hist is None


# ======================================================================================================
# infonce_global -- the fix for the measured shortcut (docs/method-runtime.md's "The measurement":
# pool-local
# segment_nce read recall@5=1.000/guard_mrr=0.996 from epoch 1 while the real tier1_recall decayed to
# 0.047, chance being 0.018)
# ======================================================================================================
def test_infonce_global_key_matches_deploys_own_match_expression():
    """SegmentLossComputer._key must be the SAME expression SegmentRetriever.retrieve/score build
    their match term from -- not a re-derivation that happens to agree today. At construction time
    the retriever's cached key table is built from this exact table's ee_start/phase_start/
    action_chunk with the SAME (untrained) key_encoder/action_encoder weights _key would use, so the
    two must match to float precision."""
    computer = _segment_computer(s=12)
    gt = _seg_tensors(s=12)
    seg_id = torch.arange(12)
    fresh = computer._key(gt, seg_id)
    cached = computer.retriever._key_table
    assert torch.allclose(fresh, cached, atol=1e-5)


# ======================================================================================================
# CorrectorLoss
# ======================================================================================================
def _corrector_batch(b: int = 6, k: int = 4, pose_dim: int = 6):
    g = torch.Generator().manual_seed(2)
    cand_actions = torch.randn(b, k, SEGMENT_LEN, ACTION_DIM, generator=g)
    cand_score = torch.randn(b, k, generator=g)
    cand_phase = torch.rand(b, k, generator=g)
    query_phase = torch.rand(b, generator=g)
    a_policy = torch.randn(b, SEGMENT_LEN, ACTION_DIM, generator=g)
    a_corr = torch.randn(b, SEGMENT_LEN, pose_dim, generator=g)
    ee_ref = torch.randn(b, k, SEGMENT_LEN, pose_dim, generator=g) * 0.1
    ee_now = torch.randn(b, pose_dim, generator=g) * 0.1
    cand_ee_start = torch.randn(b, k, pose_dim, generator=g) * 0.1
    cand_ee_end = torch.randn(b, k, pose_dim, generator=g) * 0.1
    scale = ActionScale(pos=0.0116, rot=0.1092, r2_pos=1.0, r2_rot=1.0)
    qb = QueryBatch(
        qh=torch.zeros(b, 1, 1), vh=torch.zeros(b, 1, 1), gh=torch.zeros(b, 1),
        tgt=np.arange(b),
        corrupt_kind=np.array([0, 1, 2, 3, 0, 0], dtype=np.int8),
        policy_kind=np.array([0, 1, 1, 2, 3, 0], dtype=np.int8),
    )
    return (cand_actions, cand_score, cand_phase, a_policy, a_corr, query_phase, qb, ee_ref, ee_now,
            scale, cand_ee_start, cand_ee_end)


def _corrector_loss(loss, cand_actions, cand_score, cand_phase, a_policy, a_corr, query_phase, qb,
                    ee_ref, ee_now, scale, cand_ee_start, cand_ee_end):
    """forward + loss_from, exactly how FcsrTask.loss drives the computer."""
    out = loss.forward(cand_actions, cand_score, cand_phase, a_policy, query_phase, ee_ref, ee_now,
                       scale, cand_ee_start, cand_ee_end)
    return out, loss.loss_from(out, a_policy, a_corr, qb)


def test_trust_bce_is_the_plain_mean_so_trust_keeps_the_class_prior():
    """A batch that is one quarter positive must pull a constant trust toward one quarter, not one
    half: the deployed dose is trust times shape, and a check where the reference does not help is
    the usual one."""
    p = torch.tensor([0.9, 0.1, 0.1, 0.1])
    y = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert float(trust_bce(p, y)) == pytest.approx(-np.log(0.9), abs=1e-5)
    quarter, half = torch.full((4,), 0.25), torch.full((4,), 0.5)
    assert float(trust_bce(quarter, y)) < float(trust_bce(half, y))


def test_corrector_loss_is_finite_and_moves_both_heads():
    from robot_ood.models.fusion.corrector import Corrector

    net = Corrector(chunk_len=SEGMENT_LEN, k=4, hidden=8, trust_hidden=8)
    loss = CorrectorLoss(net)
    batch = _corrector_batch(k=4)
    _out, total = _corrector_loss(loss, *batch)
    assert torch.isfinite(total)
    total.backward()
    assert torch.any(net.shape[-1].weight.grad != 0)
    assert torch.any(net.trust[-1].weight.grad != 0)


def test_corrector_loss_is_the_mse_to_the_correction_plus_the_ref_helps_bce():
    from robot_ood.models.fusion.corrector import Corrector

    net = Corrector(chunk_len=SEGMENT_LEN, k=4, hidden=8, trust_hidden=8)
    loss = CorrectorLoss(net)
    (cand_actions, cand_score, cand_phase, a_policy, a_corr, query_phase, qb, ee_ref, ee_now, scale,
     cand_ee_start, cand_ee_end) = _corrector_batch(k=4)
    out, total = _corrector_loss(loss, cand_actions, cand_score, cand_phase, a_policy, a_corr,
                                 query_phase, qb, ee_ref, ee_now, scale, cand_ee_start, cand_ee_end)
    helps = ref_helps_label(a_policy, out.features.a_ref, a_corr)
    expected = ((out.a_exec[..., :6] - a_corr) ** 2).mean() + TRUST_BCE_W * trust_bce(out.trust, helps)
    assert float(total) == pytest.approx(float(expected))


def _fit(loss, net, batch, a_corr, steps: int = 20):
    """A few Adam steps of the corrector loss against a_corr; the mean shape before and after."""
    (cand_actions, cand_score, cand_phase, a_policy, _a_corr, query_phase, qb, ee_ref, ee_now, scale,
     cand_ee_start, cand_ee_end) = batch
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    before = None
    for _ in range(steps):
        out, total = _corrector_loss(loss, cand_actions, cand_score, cand_phase, a_policy, a_corr,
                                     query_phase, qb, ee_ref, ee_now, scale, cand_ee_start, cand_ee_end)
        before = float(out.shape.mean()) if before is None else before
        opt.zero_grad(); total.backward(); opt.step()
    return before, float(out.shape.mean())


def test_the_dose_falls_where_the_policy_already_is_the_correction():
    """With a_corr equal to the policy's own pose block the loss is minimised at gain 0, so a few
    steps must lower the mean shape from its 0.27 init."""
    from robot_ood.models.fusion.corrector import Corrector

    net = Corrector(chunk_len=SEGMENT_LEN, k=4, hidden=8, trust_hidden=8)
    batch = _corrector_batch(k=4)
    before, after = _fit(CorrectorLoss(net), net, batch, batch[3][..., :6].clone())
    assert after < before


def test_the_dose_rises_where_the_reference_is_the_correction():
    """With a_corr equal to the reference the loss is minimised at gain 1, so the mean shape rises."""
    from robot_ood.models.fusion.corrector import Corrector

    net = Corrector(chunk_len=SEGMENT_LEN, k=4, hidden=8, trust_hidden=8)
    loss = CorrectorLoss(net)
    batch = _corrector_batch(k=4)
    (cand_actions, cand_score, cand_phase, a_policy, _a_corr, query_phase, qb, ee_ref, ee_now, scale,
     cand_ee_start, cand_ee_end) = batch
    with torch.no_grad():
        a_ref = loss.forward(cand_actions, cand_score, cand_phase, a_policy, query_phase, ee_ref, ee_now,
                             scale, cand_ee_start, cand_ee_end).features.a_ref.clone()
    before, after = _fit(loss, net, batch, a_ref)
    assert after > before
