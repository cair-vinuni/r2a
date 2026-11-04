"""Tests for robot_ood.runtime.retrieve — the runtime demo-graph retriever.

No dataset dependency: every test builds a small, synthetic, fully in-memory graph (straight-line toy
demos through robot_ood.data.nodes.NodeTable.from_demos + robot_ood.data.edges.EdgeSet.build) and
a randomly-initialised robot_ood.models.retrieval.gnn.GraphRetrieverNet — this subsystem is data-coupling-free
by design (see src/robot_ood/graph/gnn.py's module docstring).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_ood.core.config import GraphConfig
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.models.retrieval.gnn import GraphRetrieverNet
from robot_ood.data.nodes import NodeTable
from robot_ood.runtime.retrieve import GraphRetriever
from robot_ood.data.geometry import CONSTANTS_JSON, graph_hash

DIM = 4        # small synthetic joint dim -- any works, node_encode/edges don't care
# 31 raw frames, coarsen=3: the gripper-close split at n//2=15 leaves a 16-frame second stage, whose
# last coarse group is a single frame (16 % 3 == 1) landing exactly on the final raw frame -- required
# so the table-wide phase max hits >0.99 (NodeTable.validate's phase-convention invariant; an evenly
# divisible frame count would leave every node's phase < 1.0 and fail that check).
N_FRAMES = 31
COARSEN = 3
HIDDEN = 16


# ======================================================================================================
# synthetic graph fixture
# ======================================================================================================
def _make_demo(task_id: int, seed: int, n: int = N_FRAMES, dim: int = DIM) -> dict:
    """A straight-line synthetic demo: joint config drifts linearly along a random unit direction, the
    gripper closes halfway through (one gripper-state change, exercising the node-splitting rule and
    the grip-bucketed sibling/align relations without any real robot data)."""
    rng = np.random.RandomState(seed)
    start = rng.randn(dim) * 0.01
    direction = rng.randn(dim)
    direction /= np.linalg.norm(direction)
    t = np.arange(n, dtype=np.float32)
    q = (start[None, :] + 0.05 * t[:, None] * direction[None, :]).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[1:] = q[1:] - q[:-1]
    grip = np.zeros(n, dtype=np.uint8)
    grip[n // 2 :] = 1
    stage = grip.astype(np.int16)
    return {"q": q, "qdot": qdot, "grip": grip, "stage": stage, "task_id": task_id}


def _declare_ee_base(nodes: NodeTable) -> NodeTable:
    """Stand in for NodeTable.ee_base on a corpus that is not a 7-DoF Panda.

    ee_base is normally forward-kinematicked from q_raw, which needs a real robot; these demos are
    straight lines in a D=5 joint space. No test here reads an end-effector value, only the readout
    path that gathers one, so the demos' own declared ee is the right stand-in."""
    nodes.ee_base = nodes.ee_raw
    return nodes


def _toy_nodes_edges(n_demos: int = 4, dim: int = DIM) -> tuple[NodeTable, EdgeSet]:
    demos = [_make_demo(task_id=i % 2, seed=100 + i, dim=dim) for i in range(n_demos)]
    nodes = NodeTable.from_demos(demos, task_names=["taskA", "taskB"], coarsen=COARSEN)
    edges = EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu")
    return nodes, edges


def _toy_net(nodes: NodeTable, hidden: int = HIDDEN, layers: int = 2) -> GraphRetrieverNet:
    net = GraphRetrieverNet(dim=nodes.dim, hidden=hidden, layers=layers, n_rel=schema.N_RELATIONS, agg="sum")
    mean, std = nodes.feature_stats()
    net.set_stats(mean, std)
    net.eval()
    return net


def _stamp(nodes: NodeTable, edges: EdgeSet, net: GraphRetrieverNet, graph_dir) -> str:
    """graph_hash v2 as GraphRetriever.load recomputes it: geometry, C0's constants (absent
    here) and the head's own C2 tensors."""
    return graph_hash(nodes, edges, constants=graph_dir / CONSTANTS_JSON,
                      c2_state=net.c2_state())


def _write_graph_dir(tmp_path, nodes: NodeTable, edges: EdgeSet, net: GraphRetrieverNet, bad_hash: str | None = None):
    d = tmp_path / "graph"
    d.mkdir(exist_ok=True)
    nodes.save(d)
    edges.save(d / schema.EDGES_NPZ)
    head = dict(net.state_npz_dict())
    stamp = bad_hash if bad_hash is not None else _stamp(nodes, edges, net, d)
    head["graph_hash"] = np.array(stamp)
    np.savez(d / schema.HEAD_NPZ, **head)
    return d


def _loaded_retriever(tmp_path, **cfg_kwargs) -> GraphRetriever:
    nodes, edges = _toy_nodes_edges()
    net = _toy_net(nodes)
    d = _write_graph_dir(tmp_path, nodes, edges, net)
    retriever = GraphRetriever.load(d, cfg=GraphConfig(device="cpu", **cfg_kwargs))
    _declare_ee_base(retriever.nodes)
    return retriever


# ======================================================================================================
# 1. load round-trip
# ======================================================================================================
def test_load_round_trips_from_saved_artifacts(tmp_path):
    nodes, edges = _toy_nodes_edges()
    net = _toy_net(nodes)
    d = _write_graph_dir(tmp_path, nodes, edges, net)

    retriever = GraphRetriever.load(d, cfg=GraphConfig(device="cpu"))
    assert len(retriever.nodes) == len(nodes)
    assert retriever.net.dim == nodes.dim
    assert retriever.graph_hash == _stamp(nodes, edges, net, d)


# ======================================================================================================
# 2. graph_hash mismatch raises
# ======================================================================================================
def test_load_rejects_graph_hash_mismatch(tmp_path):
    nodes, edges = _toy_nodes_edges()
    net = _toy_net(nodes)
    d = _write_graph_dir(tmp_path, nodes, edges, net, bad_hash="not-the-right-hash")

    with pytest.raises(ValueError, match="graph_hash mismatch"):
        GraphRetriever.load(d, cfg=GraphConfig(device="cpu"))


def test_load_rejects_missing_graph_hash(tmp_path):
    """A head with no graph_hash key at all (predates the stamp) must also be rejected, not silently
    trusted -- see the module docstring on why this is the single worst failure mode to let slide."""
    nodes, edges = _toy_nodes_edges()
    net = _toy_net(nodes)
    d = tmp_path / "graph"
    d.mkdir()
    nodes.save(d)
    edges.save(d / schema.EDGES_NPZ)
    np.savez(d / schema.HEAD_NPZ, **net.state_npz_dict())    # no graph_hash key

    with pytest.raises(ValueError, match="graph_hash mismatch"):
        GraphRetriever.load(d, cfg=GraphConfig(device="cpu"))


# ======================================================================================================
# 3. well-formed GraphRetrieval
# ======================================================================================================
def test_retrieve_returns_well_formed_result(tmp_path):
    retriever = _loaded_retriever(tmp_path)
    nodes = retriever.nodes
    q0 = nodes.q[5].astype(np.float64)

    result = retriever.retrieve(np.array([q0]))

    assert result.p_node.shape == (len(nodes),)
    assert np.isclose(result.p_node.sum(), 1.0)
    assert not np.any(np.isnan(result.p_node))
    assert np.all(result.p_node >= 0.0)
    assert result.p_phase.shape == (schema.NBINS_ALIGN,)
    assert np.isclose(result.p_phase.sum(), 1.0)
    assert result.q_seg.shape == (retriever.cfg.seg_k, nodes.dim)
    assert result.qdot_seg.shape == (retriever.cfg.seg_k, nodes.dim)
    assert result.q_star.shape == (nodes.dim,)
    assert 0.0 <= result.depth <= 1.0
    assert np.isfinite(result.conf) and np.isfinite(result.entropy)
    assert np.isfinite(result.off_manifold) and result.off_manifold >= 0.0
    # euc_raw (this helper's default arm, see _loaded_retriever) is free-form: node == -1 always
    # The readout aggregates rather than snapping, so there is no single node to read an nll off of.
    assert result.node == -1
    assert np.isnan(result.nll)


# ======================================================================================================
# 4. determinism
# ======================================================================================================
def test_retrieve_deterministic_bit_identical(tmp_path):
    retriever = _loaded_retriever(tmp_path)
    q0 = retriever.nodes.q[3].astype(np.float64)

    r1 = retriever.retrieve(np.array([q0]))
    r2 = retriever.retrieve(np.array([q0]))

    assert r1.node == r2.node
    assert np.array_equal(r1.p_node, r2.p_node)
    assert np.array_equal(r1.p_phase, r2.p_phase)
    assert np.array_equal(r1.q_seg, r2.q_seg)
    assert np.array_equal(r1.qdot_seg, r2.qdot_seg)
    assert r1.entropy == r2.entropy
    # euc_raw is free-form (node == -1 always), so nll is nan by construction -- nan != nan under
    # plain ==, so this must be checked the same nan-safe way as every other nll comparison in this
    # file (e.g. test_entry_grip_open_is_bit_identical_to_g_none).
    assert r1.nll == r2.nll or (np.isnan(r1.nll) and np.isnan(r2.nll))
    assert r1.off_manifold == r2.off_manifold


# ======================================================================================================
# 5. short / single-frame history is edge-padded
# ======================================================================================================
def test_retrieve_short_history_edge_padded(tmp_path):
    retriever = _loaded_retriever(tmp_path)
    nodes = retriever.nodes
    q0 = nodes.q[0].astype(np.float64)

    r_single = retriever.retrieve(np.array([q0]))         # T=1: "no history yet" (episode start)
    assert r_single.info["padded"] is True
    assert r_single.info["hist_len"] == 1
    assert not np.any(np.isnan(r_single.p_node))

    q_short = nodes.q[:3].astype(np.float64)               # 3 < cfg.hist (8)
    r_short = retriever.retrieve(q_short)
    assert r_short.info["padded"] is True
    assert r_short.info["hist_len"] == 3
    assert not np.any(np.isnan(r_short.p_node))

    q_full = nodes.q[:retriever.cfg.hist].astype(np.float64)
    r_full = retriever.retrieve(q_full)
    assert r_full.info["padded"] is False


def test_retrieve_rejects_truly_empty_history(tmp_path):
    retriever = _loaded_retriever(tmp_path)
    with pytest.raises(ValueError, match="at least"):
        retriever.retrieve(np.zeros((0, retriever.nodes.dim)))


# ======================================================================================================
# 6. exclude_owner is a genuine leave-one-demo-out exclusion
# ======================================================================================================
def test_exclude_owner_excludes_that_demos_nodes(tmp_path):
    retriever = _loaded_retriever(tmp_path)
    nodes = retriever.nodes
    owner0_nodes = nodes.demo_nodes(0)
    q0 = nodes.q[owner0_nodes[len(owner0_nodes) // 2]].astype(np.float64)
    q_hist = np.repeat(q0[None, :], retriever.cfg.hist, axis=0)

    excluded = retriever.retrieve(q_hist, exclude_owner=0)

    assert np.all(excluded.p_node[owner0_nodes] == 0.0)
    assert np.isclose(excluded.p_node.sum(), 1.0)          # renormalized over the survivors
    assert excluded.owner != 0                              # can't have picked a node it has zero mass on


# ======================================================================================================
# 9. caching: node_encode runs once across many retrieve() calls
# ======================================================================================================
def test_node_encode_cached_across_retrieves(tmp_path, monkeypatch):
    nodes, edges = _toy_nodes_edges()
    net = _toy_net(nodes)
    d = _write_graph_dir(tmp_path, nodes, edges, net)

    calls = {"n": 0}
    orig = GraphRetrieverNet.node_encode

    def counting(self, x):
        calls["n"] += 1
        return orig(self, x)

    monkeypatch.setattr(GraphRetrieverNet, "node_encode", counting)

    # : GraphConfig's own bare default ("argmax") is no longer a registered arm --
    # see _loaded_retriever's docstring.
    retriever = GraphRetriever.load(d, cfg=GraphConfig(device="cpu"))
    _declare_ee_base(retriever.nodes)
    assert calls["n"] == 1        # exactly once, at load

    q0 = nodes.q[1].astype(np.float64)
    for _ in range(5):
        retriever.retrieve(np.array([q0]))
    assert calls["n"] == 1        # retrieve() must never rebuild it


# ======================================================================================================
# 13. real robot_ood.runtime.readout delegation
# ======================================================================================================
def test_readout_delegates_correctly(tmp_path):
    """'euc_raw' also needs no latent artifact and always reports node == -1 (a free-form blend, not a
    single demo frame) -- exercises the node<0 branch of the delegation."""
    retriever = _loaded_retriever(tmp_path)
    q0 = retriever.nodes.q[4].astype(np.float64)

    result = retriever.retrieve(np.array([q0]))

    assert result.node == -1
    assert result.q_star.shape == (retriever.nodes.dim,)
    assert result.q_seg.shape == (retriever.cfg.seg_k, retriever.nodes.dim)
    assert "target_off_manifold" in result.info
    # node == -1 is the free-form sentinel -- bookkeeping that would otherwise silently negative-index
    # the LAST node (owner/task_id/stage/t_idx) must all report -1, and there is no single node to
    # read an nll off of.
    assert (result.owner, result.task_id, result.stage, result.t_idx) == (-1, -1, -1, -1)
    assert np.isnan(result.nll)


# ======================================================================================================
# 14. move_cost (Divergence A) -- see robot_ood.runtime.retrieve's module docstring, "DIVERGENCE A"
# ======================================================================================================
# Divergence A (the fix that threads GraphConfig.move_cost's channel into net.logits) is exercised
# implicitly by every test in this file that calls retrieve() with the default move_cost=True.


# ======================================================================================================
# 15. observe() -- the PURE CODE MOVE extraction (see robot_ood.runtime.retrieve's module docstring)
# ======================================================================================================
def test_retrieve_matches_observe_plus_aggregation(tmp_path):
    """GraphRetriever.observe is retrieve()'s network forward pass, extracted verbatim; the
    softmax / top-M truncation / readout / aggregation block below it is UNCHANGED. Rebuild retrieve()'s
    result BY HAND from observe()'s robot_ood.runtime.retrieve.Observation and compare field-by-field
    with np.testing.assert_array_equal (not allclose) -- a genuine behaviour change introduced
    during the split would show up here as an exact mismatch, not just "close enough"."""
    retriever = _loaded_retriever(tmp_path)
    nodes = retriever.nodes
    q_hist = nodes.q[2 : 2 + retriever.cfg.hist].astype(np.float64)

    actual = retriever.retrieve(q_hist)

    # ---- by-hand reconstruction, mirroring retrieve()'s own body exactly ----------------------------
    obs = retriever.observe(q_hist)
    q_last = obs.window.q_now                                # retrieve() reads it off the observation

    with torch.no_grad():
        logits64 = obs.logits64
        p_full = torch.softmax(logits64, dim=0)

        finite = torch.isfinite(logits64)
        topm = min(retriever.cfg.topm, int(finite.sum().item()))
        topv, topidx = torch.topk(p_full, topm, largest=True, sorted=True)
        p_node_t = torch.zeros_like(p_full)
        p_node_t[topidx] = topv
        p_node_t = p_node_t / p_node_t.sum().clamp_min(1e-300)

    with torch.no_grad():
        outcome = retriever.aggregator.readout(p_full, q_now=q_last)
    node, q_star, q_seg = outcome.node, outcome.q_star, outcome.q_seg
    qdot_seg, depth, conf = outcome.qdot_seg, outcome.depth, outcome.conf

    p_node = p_node_t.detach().cpu().numpy()
    p_full_np = p_full.detach().cpu().numpy()
    if node >= 0:
        owner, task_id = int(nodes.owner[node]), int(nodes.task_id[node])
        stage, t_idx = int(nodes.stage[node]), int(nodes.t_idx[node])
        nll = float(-np.log(max(p_full_np[node], 1e-300)))
    else:
        owner = task_id = stage = t_idx = -1
        nll = float("nan")
    p_phase = retriever.aggregator.phase_hist(p_node)
    nz = p_node > 0
    entropy = float(-(p_node[nz] * np.log(p_node[nz])).sum())

    # ---- compare against the ACTUAL retrieve() output, field by field, EXACT equality ---------------
    np.testing.assert_array_equal(actual.p_node, p_node)
    np.testing.assert_array_equal(actual.p_phase, p_phase)
    assert actual.node == node
    assert (actual.owner, actual.task_id, actual.stage, actual.t_idx) == (owner, task_id, stage, t_idx)
    np.testing.assert_array_equal(actual.q_star, q_star)
    np.testing.assert_array_equal(actual.q_seg, q_seg)
    np.testing.assert_array_equal(actual.qdot_seg, qdot_seg)
    assert actual.depth == depth
    assert actual.conf == conf
    assert actual.entropy == entropy
    assert actual.nll == nll or (np.isnan(actual.nll) and np.isnan(nll))
    assert actual.off_manifold == obs.off_manifold
    assert actual.info["hist_len"] == obs.hist_len
    assert actual.info["padded"] == obs.padded


def test_retrieve_deterministic_bit_identical_still_passes_through_observe(tmp_path):
    """Non-regression pin: the pre-existing determinism guarantee
    (test_retrieve_deterministic_bit_identical) must survive the observe() extraction
    unchanged -- two calls through the now-split pipeline are still bit-identical."""
    retriever = _loaded_retriever(tmp_path)
    q0 = retriever.nodes.q[3].astype(np.float64)

    r1 = retriever.retrieve(np.array([q0]))
    r2 = retriever.retrieve(np.array([q0]))

    assert r1.node == r2.node
    assert np.array_equal(r1.p_node, r2.p_node)
    assert np.array_equal(r1.p_phase, r2.p_phase)
    assert np.array_equal(r1.q_seg, r2.q_seg)
    assert np.array_equal(r1.qdot_seg, r2.qdot_seg)
    assert r1.entropy == r2.entropy
    # euc_raw is free-form (node == -1 always), so nll is nan by construction -- nan != nan under
    # plain ==, so this must be checked the same nan-safe way as every other nll comparison in this
    # file (e.g. test_entry_grip_open_is_bit_identical_to_g_none).
    assert r1.nll == r2.nll or (np.isnan(r1.nll) and np.isnan(r2.nll))
    assert r1.off_manifold == r2.off_manifold


# ======================================================================================================
# 16. divergence C -- the gripper channel (see robot_ood.runtime.retrieve's module docstring, "DIVERGENCE C")
# ======================================================================================================
def test_entry_grip_open_is_bit_identical_to_g_none(tmp_path):
    """g_hist=None defaults to an all-OPEN gripper channel — *correct* pre-grasp — so an explicit
    all-open g_hist (a scalar 0.0, or an all-zero array) must be BIT-IDENTICAL to it. This is the
    offline PROOF that the ENTRY path (the project's best-measured result, ~89%) is not disturbed by
    threading g_hist through retrieve() -- see robot_ood.data.schema.grip_flag's docstring."""
    retriever = _loaded_retriever(tmp_path)
    q0 = retriever.nodes.q[0].astype(np.float64)
    q_hist = np.repeat(q0[None, :], retriever.cfg.hist, axis=0)

    r_none = retriever.retrieve(q_hist, g_hist=None)
    r_scalar = retriever.retrieve(q_hist, g_hist=0.0)
    r_array = retriever.retrieve(q_hist, g_hist=np.zeros(retriever.cfg.hist))

    for r in (r_scalar, r_array):
        assert r.node == r_none.node
        assert np.array_equal(r.p_node, r_none.p_node)
        assert np.array_equal(r.q_seg, r_none.q_seg)
        assert np.array_equal(r.qdot_seg, r_none.qdot_seg)
        assert r.entropy == r_none.entropy
        assert r.nll == r_none.nll or (np.isnan(r.nll) and np.isnan(r_none.nll))


def test_grip_hist_is_constant_within_window(tmp_path):
    """QueryPreprocessor.build takes a SINGLE current gripper flag (a bare scalar, or an array's LAST entry) and
    broadcasts it across the WHOLE padded window -- matching training's own convention exactly
    (train.py's make_queries/_queries_from_drift, both np.repeat): the trained encoder
    has never seen a window whose grip channel changes mid-window (see QueryPreprocessor.build)."""
    retriever = _loaded_retriever(tmp_path)
    q_hist = retriever.nodes.q[:3].astype(np.float64)          # shorter than cfg.hist -> edge-padded too

    g_scalar = retriever.preprocessor.build(q_hist, 1.0).grip
    assert g_scalar.shape == (retriever.cfg.hist,)
    assert np.all(g_scalar == 1.0)

    # an array's LAST entry is "the current flag" -- mirrors q_hist's own row(-1)=current convention
    g_array = retriever.preprocessor.build(q_hist, np.array([0.0, 0.0, 1.0])).grip
    assert np.all(g_array == 1.0)

    g_none = retriever.preprocessor.build(q_hist, None).grip
    assert np.all(g_none == 0.0)
