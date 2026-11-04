"""Tests for robot_ood.runtime.readout — synthetic-only, runs in .venv-core (numpy + torch, no HDF5,
no trained GNN). The retrieval distribution p(v|Q) is hand-built for every test; the module under
test only cares about aggregating a GIVEN p, never about how p was produced.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_ood.data.nodes import NodeTable
from robot_ood.data.segments import ee_segment_table, expected_ee_segment
from robot_ood.runtime.readout import (
    TOPK_K_MAX,
    TOPK_K_MIN,
    ReadoutContext,
    read_out,
    weighted_topk,
)

D = 5     # joint-space dim used throughout


# ======================================================================================================
# synthetic-context helpers
# ======================================================================================================
def _ee_from_q(q: np.ndarray) -> np.ndarray:
    """[T, 6] end-effector track built as q's first 3 columns twice, so a posterior-weighted ee_seg
    is comparable column-for-column against the q_seg carrying the same weights."""
    return np.concatenate([q[:, :3], q[:, :3]], axis=1).astype(np.float32)


def _declare_ee_base(nodes: NodeTable) -> NodeTable:
    """Stand in for NodeTable.ee_base on a corpus that is not a 7-DoF Panda.

    ee_base is normally forward-kinematicked from q_raw, which needs a real robot; these demos are
    D=5 random walks. The arms under test only ever ask whether the SAME table flows through the
    same support with the same weights, so the demos' own declared ee is the right stand-in."""
    nodes.ee_base = nodes.ee_raw
    return nodes


def _make_demo(t: int, seed: int, task_id: int = 0, offset: float = 0.0) -> dict:
    """A deterministic, smoothly-moving demo: single stage, gripper open throughout (keeps
    coarsen=1 node count exactly equal to frame count, so node indices are hand-traceable)."""
    rng = np.random.RandomState(seed)
    steps = rng.randn(t, D).astype(np.float32) * 0.01
    q = (np.cumsum(steps, axis=0) + offset).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    grip = np.zeros(t, dtype=np.uint8)
    stage = np.zeros(t, dtype=np.int16)
    return {"q": q, "qdot": qdot, "grip": grip, "stage": stage, "task_id": task_id,
            "ee": _ee_from_q(q)}


def _flat_context(n_demos: int = 3, t: int = 12, topm: int = 32) -> tuple[NodeTable, ReadoutContext]:
    """A plain multi-demo table."""
    demos = [_make_demo(t, seed=i) for i in range(n_demos)]
    nodes = _declare_ee_base(NodeTable.from_demos(demos, task_names=["taskA"], coarsen=1))
    return nodes, ReadoutContext(nodes=nodes, topm=topm)


def _fork_context(fork: int = 4, t: int = 12, topm: int = 32) -> tuple[NodeTable, ReadoutContext, int, int]:
    """Two demos sharing an IDENTICAL trunk (t=0..fork) then diverging in joint space
    (t=fork+1..T-1) — enough geometric structure to exercise "mass spread over two distinct configs"
    without needing any latent embedding (the arms that consumed one -- hyp/euc_embed -- are
    gone; see the module docstring).

    Returns (nodes, ctx, tip0, tip1) — the two branch-tip node indices mass gets put on.
    """
    demos = [_make_demo(t, seed=0, offset=0.0), _make_demo(t, seed=1, offset=5.0)]
    nodes = _declare_ee_base(NodeTable.from_demos(demos, task_names=["taskA"], coarsen=1))
    ctx = ReadoutContext(nodes=nodes, topm=topm)
    tip0 = int(nodes.demo_ptr[0]) + t - 1
    tip1 = int(nodes.demo_ptr[1]) + t - 1
    return nodes, ctx, tip0, tip1


def _concentrated_p(v: int, n: int) -> np.ndarray:
    """Mass ~entirely on node v, with a tiny positive floor everywhere else so weighted_topk's
    k_min filler slots are well-defined (zero weight, but not literally absent probabilities)."""
    p = np.full(n, 1e-9)
    p[v] = 1.0
    return p / p.sum()


def test_euc_raw_concentrated_mass_recovers_the_node():
    """When p concentrates entirely on one node, the p-weighted mean IS that node's own q (up
    to the tiny floor mass on every other node) -- the free-form blend degenerates to the single-strand
    answer, same as any other aggregation would under concentrated mass."""
    nodes, ctx, tip0, _ = _fork_context()
    p = _concentrated_p(tip0, len(nodes))

    res = read_out(p, ctx)
    np.testing.assert_allclose(res.q_star, np.asarray(nodes.q[tip0], dtype=np.float64), atol=1e-6)
    assert res.depth == pytest.approx(float(nodes.phase[tip0]), abs=1e-6)


# ======================================================================================================
# weighted_topk truncation honoured
# ======================================================================================================
def test_weighted_topk_truncation_honoured_euc_raw():
    nodes, ctx, tip0, tip1 = _fork_context()
    n = len(nodes)
    p = np.full(n, 1e-9)
    p[tip0] = 0.5
    p[tip1] = 0.5
    p = p / p.sum()

    res = read_out(p, ctx)
    # conf is the support's TOP-1 mass, not the mass it retained: a two-mode posterior reports ~0.5,
    # where the retained mass would have reported weighted_topk's own 0.99 target for every query.
    assert res.conf == pytest.approx(0.5, abs=1e-6)
    assert TOPK_K_MIN <= res.info["K"] <= min(ctx.topm, TOPK_K_MAX)


def test_conf_varies_with_how_concentrated_the_posterior_is():
    """N4: the retained mass was ~0.99 by construction for every query, so it carried no signal."""
    nodes, ctx, tip0, tip1 = _fork_context()
    n = len(nodes)
    sharp = np.full(n, 1e-9)
    sharp[tip0] = 1.0
    flat = np.full(n, 1.0)                      # the ENTRY regime: uniform over the whole pool
    sharp_conf = read_out(sharp / sharp.sum(), ctx).conf
    flat_conf = read_out(flat / flat.sum(), ctx).conf
    assert sharp_conf > 0.99 > flat_conf
    assert flat_conf == pytest.approx(1.0 / min(ctx.topm, n), rel=1e-9)


# ======================================================================================================
# determinism
# ======================================================================================================
def test_determinism():
    nodes, ctx, tip0, tip1 = _fork_context()
    ctx.q_now = np.asarray(nodes.q[0], dtype=np.float64)
    n = len(nodes)
    p = np.full(n, 1e-9)
    p[tip0] = 0.5
    p[tip1] = 0.5
    p = p / p.sum()

    r1, r2 = read_out(p, ctx), read_out(p, ctx)
    assert r1.depth == r2.depth
    assert r1.conf == r2.conf
    assert r1.off_manifold == r2.off_manifold
    np.testing.assert_array_equal(r1.q_star, r2.q_star)
    np.testing.assert_array_equal(r1.q_seg, r2.q_seg)
    np.testing.assert_array_equal(r1.qdot_seg, r2.qdot_seg)


# ======================================================================================================
# the differentiable torch expectation — must optimise the SAME quantity the numpy arms deploy
# ======================================================================================================
def test_ee_segment_table_rows_match_node_table_including_demo_boundaries():
    """Every row of the materialised table must be NodeTable.ee_segment exactly -- same clip at
    raw_ptr[owner+1], same edge padding. The final nodes of each demo are the ones that clip, so the
    test asserts at least one row really was padded rather than passing vacuously."""
    nodes, ctx = _flat_context()
    table = ee_segment_table(nodes, ctx.seg_k)
    assert table.shape == (len(nodes), ctx.seg_k, 6)

    for v in range(len(nodes)):
        np.testing.assert_array_equal(
            table[v].double().numpy(), nodes.ee_segment(v, ctx.seg_k)
        )
    padded = [v for v in range(len(nodes))
              if (nodes.raw_ptr[nodes.owner[v] + 1] - nodes.t_raw[v]) < ctx.seg_k]
    assert padded, "no node's segment was clipped: the boundary case is untested"


def test_expected_ee_segment_matches_a_weighted_topk_numpy_aggregate():
    """The torch expectation must equal the p-weighted numpy blend over the same truncated
    support, or Stage 2 optimises something the belief rollout never computes. p is built so
    weighted_topk's 0.99-mass support is EXACTLY K nodes, which is the only condition under
    which the two supports differ -- the torch arm uses a fixed k by design."""
    k = 10
    nodes, ctx = _flat_context()
    # Distinct masses, so no sort/topk tie-break can pick a different support; the top k sum to
    # 0.9995 while the top k-1 sum to 0.9000, bracketing weighted_topk's 0.99 target at exactly k.
    p = np.full(len(nodes), 1e-9)
    p[:k] = 0.0995 + 1e-4 * np.arange(k)
    p = p / p.sum()

    idx, w = weighted_topk(torch.as_tensor(p, dtype=torch.float64), k_max=ctx.topm).numpy()
    assert len(idx) == k, "supports differ; the comparison below would be meaningless"
    ee_k = np.stack([np.asarray(nodes.ee_segment(int(v), ctx.seg_k), np.float64) for v in idx])
    expected = np.tensordot(w, ee_k, axes=(0, 0))

    got = expected_ee_segment(torch.as_tensor(p), ee_segment_table(nodes, ctx.seg_k), k=k)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-9, atol=1e-12)


def test_expected_ee_segment_gradient_reaches_p():
    """The whole point of the torch arm: an action-space loss must be able to move p. The topk
    INDICES are a hard selection, so nodes outside the support correctly get exactly zero grad."""
    v, seg_k, k = 6, 3, 3
    table = torch.arange(v * seg_k * 6, dtype=torch.float64).reshape(v, seg_k, 6)
    p = torch.tensor([0.30, 0.25, 0.20, 0.13, 0.08, 0.04], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(lambda x: expected_ee_segment(x, table, k=k), (p,))

    expected_ee_segment(p, table, k=k).sum().backward()
    assert p.grad[:k].abs().sum() > 0
    assert torch.count_nonzero(p.grad[k:]) == 0
