"""Tests for robot_ood.data.metrics's LearnedSeed -- the two-tier learned seed.

The invariants pinned here are the ones the deleted hand-written kernel used to hold implicitly and
the ones its replacement is only correct because of: the mixture is PER ROW, the language tilt is
additive in log space and the top-k runs AFTER it, the seed can no longer underflow to empty, and
both registered guards fire.

tier1_recall is exercised for correctness, not as the >= 0.99 gate: tier 1 is a LEARNED projection,
so on an untrained head its recall is the chance level M/V. The gate is an S1 post-training
measurement -- this file checks the diagnostic that measures it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_ood.data.metrics import (
    ENTROPY_GUARD_NATS, SIGMA_GUARD_DECADES, C2Metric, LearnedSeed, NodeMetricColumns,
    QueryMetricColumns, tier1_recall,
)
from robot_ood.data import schema

DIM = 7
HIDDEN = 16


def _seed_module(sigma_init: float = schema.KERNEL_BW) -> LearnedSeed:
    torch.manual_seed(0)
    return LearnedSeed(HIDDEN, C2Metric(), sigma_init=sigma_init)


def _graph(v: int, seed: int = 1, scale: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    q_n = scale * torch.randn(v, DIM, generator=g)
    n_v = torch.randn(v, HIDDEN, generator=g)
    cols = NodeMetricColumns(
        qdot=torch.randn(v, DIM, generator=g), phase=torch.rand(v, generator=g),
        grip=torch.zeros(v), task_id=torch.zeros(v, dtype=torch.int64),
    )
    return q_n, n_v, cols


def _query(h: int, q_n: torch.Tensor, rows: torch.Tensor | None = None, seed: int = 2):
    """A window whose rows sit on given node configs (or at random), plus its metric columns."""
    g = torch.Generator().manual_seed(seed)
    qh_n = q_n[rows] if rows is not None else torch.randn(h, DIM, generator=g)
    return qh_n, QueryMetricColumns(qdot=torch.randn(h, DIM, generator=g), grip=torch.zeros(h))


# ---- 1. shape, sparsity, finiteness ----------------------------------------------------------------
def test_seed_is_zero_off_its_own_topk():
    v, h, topk = 200, 8, 12
    seed = _seed_module()
    q_n, n_v, cols = _graph(v)
    qh_n, query = _query(h, q_n)
    z_h = torch.randn(h, HIDDEN, generator=torch.Generator().manual_seed(3))
    pool_w = torch.full((h,), 1.0 / h)

    a = seed(qh_n, query, z_h, pool_w, q_n, cols, n_v, topk=topk, tier1_m=64)
    assert a.shape == (v,)
    assert torch.isfinite(a).all()
    assert int((a > 0).sum()) == topk


def test_seed_never_empties_however_small_the_prior():
    """The alive guard is gone STRUCTURALLY: a is a mixture of softmaxes, so there is no float32
    floor left to fall through, and a uniformly tiny prior only shifts the log-space score."""
    v, h, topk = 150, 8, 16
    seed = _seed_module()
    q_n, n_v, cols = _graph(v, seed=4)
    qh_n, query = _query(h, q_n, seed=5)
    z_h = torch.randn(h, HIDDEN, generator=torch.Generator().manual_seed(6))
    pool_w = torch.full((h,), 1.0 / h)

    factor = torch.full((v,), 1e-38)
    a = seed(qh_n, query, z_h, pool_w, q_n, cols, n_v, topk=topk, tier1_m=64, factor=factor)
    assert torch.isfinite(a).all()
    assert int((a > 0).sum()) == topk


def test_batched_seed_matches_the_looped_single_query():
    v, h, topk, b = 120, 8, 10, 3
    seed = _seed_module()
    q_n, n_v, cols = _graph(v, seed=7)
    g = torch.Generator().manual_seed(8)
    qh_n = torch.randn(b, h, DIM, generator=g)
    query = QueryMetricColumns(qdot=torch.randn(b, h, DIM, generator=g), grip=torch.zeros(b, h))
    z_h = torch.randn(b, h, HIDDEN, generator=g)
    pool_w = torch.softmax(torch.randn(b, h, generator=g), dim=-1)

    batched = seed(qh_n, query, z_h, pool_w, q_n, cols, n_v, topk=topk, tier1_m=48)
    for i in range(b):
        one = seed(
            qh_n[i], QueryMetricColumns(qdot=query.qdot[i], grip=query.grip[i]), z_h[i], pool_w[i],
            q_n, cols, n_v, topk=topk, tier1_m=48,
        )
        assert torch.allclose(batched[i], one, atol=1e-6)


# ---- 2. the per-row mixture -------------------------------------------------------------------------
def test_one_clean_row_still_seeds_when_the_other_rows_are_dirty():
    """The mixture is per ROW for exactly this case: a single softmax over the whole [H, V] block
    would let seven dirty rows swallow the clean one's mass."""
    v, h, topk = 300, 8, 8
    seed = _seed_module()
    q_n, n_v, cols = _graph(v, seed=9)
    clean_node = 42
    rows = torch.full((h,), 7, dtype=torch.int64)      # seven rows parked on a far-away node
    rows[-1] = clean_node
    qh_n, query = _query(h, q_n, rows=rows, seed=10)
    z_h = torch.zeros(h, HIDDEN)                       # tier 1 identical per row: tier 2 decides
    pool_w = torch.zeros(h)
    pool_w[-1] = 1.0                                   # the clean row carries all the weight

    a = seed(qh_n, query, z_h, pool_w, q_n, cols, n_v, topk=topk, tier1_m=v)
    assert int(a.argmax()) == clean_node


# ---- 3. the tilt ------------------------------------------------------------------------------------
def _untilted_topk(seed, qh_n, query, z_h, pool_w, q_n, cols, n_v, topk, m):
    a = seed(qh_n, query, z_h, pool_w, q_n, cols, n_v, topk=topk, tier1_m=m)
    return set(torch.nonzero(a).flatten().tolist())


def test_prior_changes_which_nodes_are_seeded_not_just_their_weights():
    """top-k AFTER the tilt is a load-bearing invariant: a prior that only reweighted an
    already-chosen set could never rescue a node the untilted score ranked out."""
    v, h, topk, m = 200, 8, 8, 200
    seed = _seed_module()
    # a tight cloud, so the score spread stays in the same decade as the prior it is compared with
    q_n, n_v, cols = _graph(v, seed=11, scale=0.02)
    qh_n, query = _query(h, q_n, seed=12)
    z_h = torch.randn(h, HIDDEN, generator=torch.Generator().manual_seed(13))
    pool_w = torch.full((h,), 1.0 / h)

    before = _untilted_topk(seed, qh_n, query, z_h, pool_w, q_n, cols, n_v, topk, m)
    outsider = next(i for i in range(v) if i not in before)
    factor = torch.full((v,), 1e-6)
    factor[outsider] = 1.0
    after = torch.nonzero(
        seed(qh_n, query, z_h, pool_w, q_n, cols, n_v, topk=topk, tier1_m=m, factor=factor)
    ).flatten().tolist()
    assert outsider in after


def test_tilt_is_additive_in_log_space():
    """a_tilted == a * factor exactly where both are supported -- the log-space add is the same
    quantity, not a second scoring rule."""
    v, h, topk, m = 100, 8, 100, 100
    seed = _seed_module()
    q_n, n_v, cols = _graph(v, seed=14)
    qh_n, query = _query(h, q_n, seed=15)
    z_h = torch.randn(h, HIDDEN, generator=torch.Generator().manual_seed(16))
    pool_w = torch.full((h,), 1.0 / h)
    factor = torch.rand(v, generator=torch.Generator().manual_seed(17)) + 0.5

    plain = seed(qh_n, query, z_h, pool_w, q_n, cols, n_v, topk=topk, tier1_m=m)
    tilted = seed(qh_n, query, z_h, pool_w, q_n, cols, n_v, topk=topk, tier1_m=m, factor=factor)
    support = plain > 0
    assert torch.allclose(tilted[support], (plain * factor)[support], rtol=1e-5)


# ---- 4. sigma and tau ---------------------------------------------------------------------------------
def test_sigma_and_tau_are_learned_and_initialised_from_the_calibrator():
    seed = _seed_module()
    assert float(seed.sigma) == pytest.approx(schema.KERNEL_BW, rel=1e-6)
    assert float(seed.tau) == pytest.approx(1.0, rel=1e-6)
    assert float(seed.sigma_init) == pytest.approx(schema.KERNEL_BW, rel=1e-6)
    assert seed.log_sigma.requires_grad and seed.log_tau.requires_grad


def test_gradient_reaches_sigma_tau_and_the_tier1_projection():
    """Hard indices, soft weights -- the readout's trick. f only gets a gradient because its own
    score is carried into tier 2 for the candidates it selected."""
    v, h, topk = 80, 8, 8
    seed = _seed_module()
    q_n, n_v, cols = _graph(v, seed=18)
    qh_n, query = _query(h, q_n, seed=19)
    z_h = torch.randn(h, HIDDEN, generator=torch.Generator().manual_seed(20))
    pool_w = torch.full((h,), 1.0 / h)

    seed.zero_grad()
    seed(qh_n, query, z_h, pool_w, q_n, cols, n_v, topk=topk, tier1_m=32).sum().backward()
    missing = [n for n, p in seed.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"no gradient reached: {missing}"


# ---- 5. the two registered guards ----------------------------------------------------------------------
def _guard_inputs(v: int = 200, h: int = 8):
    q_n, n_v, cols = _graph(v, seed=21)
    qh_n, query = _query(h, q_n, seed=22)
    z_h = torch.randn(h, HIDDEN, generator=torch.Generator().manual_seed(23))
    pool_w = torch.full((h,), 1.0 / h)
    return qh_n, query, z_h, pool_w, q_n, cols, n_v


def test_guards_pass_at_initialisation():
    seed = _seed_module()
    report = seed.guard_report(*_guard_inputs(), tier1_m=200)
    assert report.decades == pytest.approx(0.0, abs=1e-5)   # float32 log/exp round trip
    assert report.entropy_gap <= ENTROPY_GUARD_NATS
    report.assert_ok()


def test_sigma_guard_fires_when_the_mlp_bypasses_the_backbone():
    seed = _seed_module()
    with torch.no_grad():
        seed.log_sigma.fill_(float(np.log(schema.KERNEL_BW * 100.0)))
    report = seed.guard_report(*_guard_inputs(), tier1_m=200)
    assert report.decades > SIGMA_GUARD_DECADES
    assert not report.ok
    with pytest.raises(AssertionError, match="decades from the C0 value"):
        report.assert_ok()


def test_entropy_guard_fires_when_the_seed_collapses_flat():
    """The degenerate solution the residual form exists to block: a seed so flat that message
    passing carries the whole localisation problem. A runaway tau produces it exactly."""
    seed = _seed_module()
    inputs = _guard_inputs()
    with torch.no_grad():
        seed.log_tau.fill_(float(np.log(1e6)))
    report = seed.guard_report(*inputs, tier1_m=200)
    assert report.entropy > report.entropy_backbone
    assert report.entropy_gap > ENTROPY_GUARD_NATS
    with pytest.raises(AssertionError, match="seed entropy gap"):
        report.assert_ok()


# ---- 6. the two-tier gate's diagnostic ---------------------------------------------------------------
def test_tier1_recall_is_one_when_tier_one_keeps_everything():
    v, h, topk = 300, 8, 32
    seed = _seed_module()
    q_n, n_v, cols = _graph(v, seed=24)
    g = torch.Generator().manual_seed(25)
    qh_n = torch.randn(4, h, DIM, generator=g)
    query = QueryMetricColumns(qdot=torch.randn(4, h, DIM, generator=g), grip=torch.zeros(4, h))
    z_h = torch.randn(4, h, HIDDEN, generator=g)
    pool_w = torch.softmax(torch.randn(4, h, generator=g), dim=-1)

    args = (seed, qh_n, query, z_h, pool_w, q_n, cols, n_v)
    assert tier1_recall(*args, topk=topk, tier1_m=v) == 1.0
    narrow = tier1_recall(*args, topk=topk, tier1_m=32)
    assert 0.0 <= narrow <= 1.0
    assert narrow <= tier1_recall(*args, topk=topk, tier1_m=128)
