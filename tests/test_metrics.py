"""Tests for robot_ood.data.metrics's C2 -- the shared learned pair metric.

Hand-built tensors only, no dataset. The load-bearing checks are the FEATURE SLOT CONTRACT (a site
that cannot supply a feature zero-fills it, and the vector never changes width), the inductive
guarantee (no parameter dimension in V or n_tasks), and the edge-weight hook actually weighting
temporal edges, which the Gaussian default pins at 1.0.
"""

from __future__ import annotations

import numpy as np
import torch

from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.metrics import (
    N_METRIC_FEATURES, C2EdgeWeights, C2Metric, NodeMetricColumns, QueryMetricColumns,
    ee_pose_delta_norm, node_pair_features, query_node_features,
)
from robot_ood.data.nodes import NodeTable

DIM = 7


def _columns(v: int, *, seed: int = 0, with_optional: bool = False) -> NodeMetricColumns:
    g = torch.Generator().manual_seed(seed)
    return NodeMetricColumns(
        qdot=torch.randn(v, DIM, generator=g),
        phase=torch.rand(v, generator=g),
        grip=(torch.rand(v, generator=g) > 0.5).float(),
        task_id=torch.randint(0, 3, (v,), generator=g),
        ee=torch.randn(v, 6, generator=g) if with_optional else None,
    )


def _demo(n_frames: int = 31, seed: int = 3) -> dict:
    rng = np.random.RandomState(seed)
    direction = rng.randn(DIM)
    direction /= np.linalg.norm(direction)
    t = np.arange(n_frames, dtype=np.float32)
    q = (rng.randn(DIM)[None, :] * 0.01 + 0.05 * t[:, None] * direction[None, :]).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[1:] = q[1:] - q[:-1]
    grip = np.zeros(n_frames, dtype=np.uint8)
    grip[n_frames // 2:] = 1
    return {"q": q, "qdot": qdot, "grip": grip, "stage": grip.astype(np.int16), "task_id": 0}


# ---- 1. the feature slot contract ----------------------------------------------------------------
def test_node_pair_features_width_is_fixed_whatever_columns_exist():
    v, e = 20, 40
    q_n = torch.randn(v, DIM)
    g = torch.Generator().manual_seed(1)
    src = torch.randint(0, v, (e,), generator=g)
    dst = torch.randint(0, v, (e,), generator=g)

    bare = node_pair_features(q_n, _columns(v), src, dst)
    full = node_pair_features(q_n, _columns(v, with_optional=True), src, dst)
    assert bare.shape == full.shape == (e, N_METRIC_FEATURES)


def test_absent_columns_zero_fill_their_own_slots_only():
    """A corpus with no poses and no DINO features must zero slots 1, 2 and 7 and change nothing
    else -- that is what lets one trained metric serve a corpus that ships fewer columns."""
    v, e = 15, 30
    q_n = torch.randn(v, DIM)
    g = torch.Generator().manual_seed(2)
    src = torch.randint(0, v, (e,), generator=g)
    dst = torch.randint(0, v, (e,), generator=g)
    cols = _columns(v, with_optional=True)
    bare = NodeMetricColumns(qdot=cols.qdot, phase=cols.phase, grip=cols.grip, task_id=cols.task_id)

    f_full = node_pair_features(q_n, cols, src, dst)
    f_bare = node_pair_features(q_n, bare, src, dst)
    optional = [1, 2, 7]
    assert torch.equal(f_bare[:, optional], torch.zeros(e, len(optional)))
    kept = [i for i in range(N_METRIC_FEATURES) if i not in optional]
    assert torch.equal(f_full[:, kept], f_bare[:, kept])
    assert f_full[:, optional].abs().sum() > 0


def test_query_features_zero_the_three_slots_the_query_cannot_have():
    """dphase, same_task and cos_DINO are node-side facts. The query has none of them, and in
    particular it never touches vision here."""
    v, h, m = 30, 4, 6
    q_n = torch.randn(v, DIM)
    qh_n = torch.randn(h, DIM)
    idx = torch.randint(0, v, (h, m), generator=torch.Generator().manual_seed(4))
    query = QueryMetricColumns(qdot=torch.randn(h, DIM), grip=torch.zeros(h))

    feats = query_node_features(qh_n, query, q_n, _columns(v, with_optional=True), idx)
    assert feats.shape == (h, m, N_METRIC_FEATURES)
    assert torch.equal(feats[..., 4], torch.zeros(h, m))
    assert torch.equal(feats[..., 6], torch.zeros(h, m))
    assert torch.equal(feats[..., 7], torch.zeros(h, m))
    # slot 0 is the same normalised ||dq|| the seed backbone uses
    assert torch.allclose(feats[..., 0], (qh_n[:, None, :] - q_n[idx]).norm(dim=-1))


# ---- 1b. ee_pose_delta_norm, the segment scorer's reused pose distance -----------------------------
def test_ee_pose_delta_norm_matches_a_direct_norm_over_position_and_rotation():
    a = torch.randn(4, 6)
    b = torch.randn(4, 6)
    assert torch.allclose(ee_pose_delta_norm(a, b), (a - b).norm(dim=-1))


def test_ee_pose_delta_norm_is_zero_for_identical_poses():
    a = torch.randn(3, 6)
    assert torch.allclose(ee_pose_delta_norm(a, a), torch.zeros(3))


def test_ee_pose_delta_norm_broadcasts():
    a = torch.randn(5, 1, 6)
    b = torch.randn(5, 8, 6)
    out = ee_pose_delta_norm(a, b)
    assert out.shape == (5, 8)


# ---- 2. the metric itself -------------------------------------------------------------------------
def test_weight_is_positive_and_decreasing_in_distance():
    metric = C2Metric()
    feats = torch.randn(64, N_METRIC_FEATURES)
    d = metric.distance(feats)
    w = metric.weight(feats)
    assert (w > 0).all(), "softplus(-d) must never delete an edge, only silence it"
    order_d = torch.argsort(d)
    order_w = torch.argsort(w, descending=True)
    assert torch.equal(order_d, order_w)


def test_metric_is_inductive_no_parameter_dimension_in_v_or_n_tasks():
    metric = C2Metric()
    n_params = sum(p.numel() for p in metric.parameters())
    assert n_params < 1000, f"C2 is meant to be a few hundred parameters, got {n_params}"
    for name, p in metric.named_parameters():
        assert all(dim <= 32 for dim in p.shape), f"{name} has a suspicious dimension {tuple(p.shape)}"


def test_metric_output_is_shape_agnostic_across_call_sites():
    """The same weights must score an [E, 8] edge list and an [H, M, 8] seed block."""
    metric = C2Metric()
    edge_like = metric.weight(torch.randn(50, N_METRIC_FEATURES))
    seed_like = metric.weight(torch.randn(8, 16, N_METRIC_FEATURES))
    assert edge_like.shape == (50,)
    assert seed_like.shape == (8, 16)


# ---- 3. the edge-weight hook -----------------------------------------------------------------------
def test_edge_weight_hook_weights_temporal_edges_too():
    """The Gaussian default pins every temporal edge at exactly 1.0 -- i.e. next16 == next1. The
    whole point of routing edge weights through C2 is that this stops being true."""
    nodes = NodeTable.from_demos([_demo()], task_names=["taskA"], coarsen=3)
    default = EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu")
    hooked = EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu",
                           metric=C2EdgeWeights(C2Metric(), device=torch.device("cpu")))

    temporal = default.rel == schema.REL_INDEX["next1"]
    assert bool(temporal.any())
    assert np.allclose(default.w[temporal], 1.0)
    assert not np.allclose(hooked.w[temporal], 1.0)
    assert hooked.w.dtype == np.float32 and hooked.w.shape == default.w.shape
    assert (hooked.w > 0).all()


def test_edge_weight_hook_default_is_bit_identical_to_the_gaussian():
    """metric=None must leave EdgeSet.build exactly as it was: the hook is opt-in."""
    nodes = NodeTable.from_demos([_demo(seed=5)], task_names=["taskA"], coarsen=3)
    a = EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu")
    b = EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu", metric=None)
    assert np.array_equal(a.w, b.w)
