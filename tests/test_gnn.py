"""Pure-torch tests for robot_ood.models.retrieval.gnn — the query-dependent demonstration-graph retriever.

No dataset dependency: every test builds tiny hand-constructed tensors (this subsystem is "PURE
TORCH with no data coupling" by design — see src/robot_ood/graph/gnn.py's module docstring for what the
network computes and why). Test 1 (golden propagation) is the load-bearing one: it pins down message
direction, the explicit reached mask, and the scatter mechanics that every other test relies on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from robot_ood.data import schema
from robot_ood.data.metrics import NodeMetricColumns, QueryMetricColumns
from robot_ood.models.retrieval.gnn import GraphRetrieverNet, QueryEncoder
from robot_ood.models.types import GraphEdges, MoveCost, NodeCloud, PipelineParams, QueryWindow, SeedQuery

# propagate only reads src/dst/rel(/src_by_rel/dst_by_rel/rel_offsets) off the
# GraphEdges they're given (node_x/log_idf pass through unused -- see that class's docstring), so this
# zero-size placeholder stands in for both at every _prop call site below rather than threading real
# node_x/log_idf through every golden/parity test in this file that never otherwise needs them.
_UNUSED = torch.zeros(0)

torch.manual_seed(0)

DIM = 7        # Panda joint dim (matches LIBERO)
HIDDEN = 16    # small for fast tests; schema.HIDDEN=64 is the production value
N_PSI = 4


# ---- helpers ---------------------------------------------------------------------------------------
def _small_net(layers: int = 3, agg: str = "sum", n_rel: int = schema.N_RELATIONS,
               hidden: int = HIDDEN, dim: int = DIM, n_psi: int = N_PSI) -> GraphRetrieverNet:
    """A tiny net with identity normalization stats (mean=0, std=1), so raw and normalized joint
    values coincide and tests can hand-build tensors without touching the network's internals.

    Returned in eval mode, which is the mode every invariant below is stated in: the query encoder's
    frame dropout is a training-only draw, so a train-mode net makes two forwards of one window
    legitimately disagree. A test about the dropout itself calls .train() on its own encoder."""
    net = GraphRetrieverNet(dim=dim, hidden=hidden, layers=layers, n_rel=n_rel, agg=agg, n_psi=n_psi)
    net.set_stats(np.zeros(2 * dim + 1, np.float32), np.ones(2 * dim + 1, np.float32))
    return net.eval()


def _columns(v: int, dim: int = DIM) -> NodeMetricColumns:
    """Metric columns for a hand-built graph: the slots C2 reads beyond the configs themselves."""
    g = torch.Generator().manual_seed(11)
    return NodeMetricColumns(
        qdot=torch.randn(v, dim, generator=g), phase=torch.rand(v, generator=g),
        grip=torch.zeros(v), task_id=torch.zeros(v, dtype=torch.int64),
    )


def _fwd(net, qh, vh, gh, node_x, src, dst, rel, log_idf, idf_w=0.0,
         topk=schema.SEED_TOPK, checkpoint=True, *,
         src_by_rel=None, dst_by_rel=None, rel_offsets=None):
    """Test-only shim over GraphRetrieverNet.forward's grouped
    (QueryWindow, GraphEdges, PipelineParams, NodeMetricColumns) signature -- rebuilds the parameter
    objects from the same flat arguments this file's tests already pass."""
    qw = QueryWindow(qh=qh, vh=vh, gh=gh)
    ge = GraphEdges(node_x=node_x, src=src, dst=dst, rel=rel, log_idf=log_idf,
                     src_by_rel=src_by_rel, dst_by_rel=dst_by_rel, rel_offsets=rel_offsets)
    params = PipelineParams(idf_w=idf_w, topk=topk, checkpoint=checkpoint)
    return net.forward(qw, ge, params, _columns(node_x.shape[0], net.dim))


def _seed(net, e_q, qh_n, w, q_nodes_n, n_v, topk=schema.SEED_TOPK, *,
          vh=None, gh=None, columns=None, tier1_m=None):
    """Test-only shim over GraphRetrieverNet.seed's grouped (SeedQuery, NodeCloud,
    PipelineParams) signature. z_h is e_q broadcast over the window, so tier 1 ranks identically for
    every row and the tests below measure tier 2 alone."""
    zeros_h = torch.zeros(*qh_n.shape[:-1])
    query = SeedQuery(
        e_q=e_q, z_h=e_q[..., None, :].expand(*qh_n.shape[:-1], e_q.shape[-1]),
        pool_w=torch.softmax(w.clamp_min(1e-12).log(), dim=-1), qh_n=qh_n,
        columns=QueryMetricColumns(qdot=zeros_h[..., None].expand_as(qh_n) if vh is None else vh,
                                   grip=zeros_h if gh is None else gh),
    )
    nodes = NodeCloud(q_nodes_n=q_nodes_n, n_v=n_v,
                      columns=_columns(q_nodes_n.shape[0], q_nodes_n.shape[-1])
                      if columns is None else columns)
    params = PipelineParams(topk=topk, tier1_m=q_nodes_n.shape[0] if tier1_m is None else tier1_m)
    return net.seed(query, nodes, params)[0]


def _logits(net, h, n_v, reached, log_idf, idf_w, *, q_nodes=None, q_now=None):
    """Test-only shim over GraphRetrieverNet.logits's grouped move_cost:
    MoveCost signature -- same flat-argument call shape this file's tests already pass."""
    move_cost = MoveCost(q_nodes=q_nodes, q_now=q_now)
    return net.logits(h, n_v, reached, log_idf, idf_w, move_cost)


def _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False, *,
          src_by_rel=None, dst_by_rel=None, rel_offsets=None):
    """Test-only shim over GraphRetrieverNet.propagate's grouped
    ge: GraphEdges signature -- same flat-argument call shape this file's tests already pass.
    node_x/log_idf are unused by propagate (see _UNUSED's comment)."""
    ge = GraphEdges(node_x=_UNUSED, src=src, dst=dst, rel=rel, log_idf=_UNUSED,
                     src_by_rel=src_by_rel, dst_by_rel=dst_by_rel, rel_offsets=rel_offsets)
    return net.propagate(h0, n_v, e_q, ge, checkpoint=checkpoint)


def _chain_edges(n: int, rel_id: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """offset -> offset+1 -> ... -> offset+n-1, all edges of relation rel_id."""
    src = torch.arange(offset, offset + n - 1, dtype=torch.int32)
    dst = torch.arange(offset + 1, offset + n, dtype=torch.int32)
    rel = torch.full((n - 1,), rel_id, dtype=torch.int8)
    return src, dst, rel


def _rand_edges(v: int, e: int, n_rel: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    src = torch.as_tensor(rng.randint(0, v, e), dtype=torch.int32)
    dst = torch.as_tensor(rng.randint(0, v, e), dtype=torch.int32)
    rel = torch.as_tensor(rng.randint(0, n_rel, e), dtype=torch.int8)
    return src, dst, rel


def _node_x(v: int, dim: int = DIM, seed: int = 0) -> torch.Tensor:
    """[V, 2*dim+2] raw node rows: [q, qdot, grip, t_frac]."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(v, dim, generator=g)
    qdot = torch.randn(v, dim, generator=g)
    grip = torch.rand(v, 1, generator=g)
    t_frac = torch.rand(v, 1, generator=g)
    return torch.cat([q, qdot, grip, t_frac], dim=-1)


def _group_by_rel(
    src: torch.Tensor, dst: torch.Tensor, rel: torch.Tensor, n_rel: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirrors robot_ood.data.edges.EdgeSet.torch's relation-grouping exactly (permute edges into
    relation order, CSR offsets over the FIXED n_rel-size vocabulary), reimplemented inline here
    rather than imported so this test module stays "PURE TORCH, no dataset coupling" (its own module
    docstring's contract) -- building a real EdgeSet needs a full, validated
    NodeTable."""
    rel64 = rel.to(torch.int64)
    order = torch.argsort(rel64, stable=True)
    rel_offsets = torch.searchsorted(rel64[order], torch.arange(n_rel + 1, dtype=torch.int64))
    return src[order].long(), dst[order].long(), rel_offsets


def _query(h: int, dim: int = DIM, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    qh = torch.randn(h, dim, generator=g)
    vh = torch.randn(h, dim, generator=g)
    gh = torch.rand(h, generator=g)
    w = torch.rand(h, generator=g) + 0.1                       # keep strictly > 0
    return qh, vh, gh, w


# ---- 1. golden propagation test (load-bearing) ------------------------------------------------------
def test_golden_propagation_direction_and_reach():
    """Chain 0->1->...->9, only next1 edges. Force the relation gate to identity and the update MLP
    to pass the scattered message through unchanged (isolating scatter/gather from the LEARNED update,
    which is not what this test verifies). Seed one-hot at node 0. After L layers, reached must be
    EXACTLY {0,...,L} and unreached nodes' h must be EXACTLY zero -- this single test catches: swapped
    src/dst (message flows backward -- the chain is near-symmetric so a model would still train, it
    would just retrieve the PAST instead of the FUTURE), wrong dilation indexing, index_add_ mistakes,
    and a broken reached mask.
    """
    v, layers = 10, 4
    rel_id = schema.REL_INDEX["next1"]
    src, dst, rel = _chain_edges(v, rel_id)
    net = _small_net(layers=layers, agg="sum")

    net.rel.forward = lambda e_q: torch.ones(net.n_rel, net.hidden)             # identity relation gate
    net.upd.forward = lambda x: x[:, net.hidden : 2 * net.hidden]               # pass out through raw

    h0 = torch.zeros(v, net.hidden)
    h0[0] = torch.randn(net.hidden)
    n_v = torch.zeros(v, net.hidden)                                            # unused (upd ignores it)
    e_q = torch.zeros(net.hidden)                                               # unused (rel ignores it)

    h, reached = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)

    expect_reached = torch.zeros(v, dtype=torch.bool)
    expect_reached[: layers + 1] = True
    assert torch.equal(reached, expect_reached), (reached, expect_reached)

    assert torch.equal(h[layers + 1 :], torch.zeros(v - layers - 1, net.hidden))
    reached_h = h[expect_reached]
    assert reached_h.abs().sum(dim=-1).min().item() > 1e-6                      # every reached row nonzero
    assert torch.allclose(reached_h, reached_h[0].expand_as(reached_h), atol=1e-5)  # traveling wave


def test_golden_propagation_checkpoint_matches_no_checkpoint():
    """Same golden setup, but confirms the checkpointed path reproduces the eager path."""
    v, layers = 10, 4
    rel_id = schema.REL_INDEX["next1"]
    src, dst, rel = _chain_edges(v, rel_id)
    net = _small_net(layers=layers, agg="sum")
    net.rel.forward = lambda e_q: torch.ones(net.n_rel, net.hidden)
    net.upd.forward = lambda x: x[:, net.hidden : 2 * net.hidden]

    h0 = torch.zeros(v, net.hidden)
    h0[0] = torch.randn(net.hidden)
    n_v = torch.zeros(v, net.hidden)
    e_q = torch.zeros(net.hidden)

    h_a, r_a = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)
    h_b, r_b = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=True)
    assert torch.equal(r_a, r_b)
    assert torch.allclose(h_a, h_b, atol=1e-6)


# ---- 1b. grouped relation path (the perf fix) matches the fused fallback -----------------------------
# The gate g_rel[rel]'s backward is a scatter-add of [E, hidden] into only R=12 destination rows --
# measured ~165x slower on backward than forward on the real long suite graph (E=767k). The fix:
# group edges by relation and broadcast the gate over each contiguous block instead of gathering it
# per edge. index_add_'s accumulation order differs between "one fused scatter over all E edges" and
# "R separate per-block scatters", so exact bit-identity is not achievable (floating-point
# reassociation) -- these tests use atol=1e-5, rtol=1e-4: loose enough to absorb reassociation noise on
# a few hundred edges, tight enough that an actual indexing bug (wrong src/dst, off-by-one block
# boundary, a dropped relation) would still fail it by many orders of magnitude.
GROUP_ATOL, GROUP_RTOL = 1e-5, 1e-4


def test_grouped_relation_path_matches_fused_forward_sum():
    v, e = 200, 900
    net = _small_net(layers=3, agg="sum")
    src, dst, rel = _rand_edges(v, e, schema.N_RELATIONS, seed=123)
    src_g, dst_g, rel_offsets = _group_by_rel(src, dst, rel, schema.N_RELATIONS)

    g = torch.Generator().manual_seed(1230)          # seeded: this comparison's tolerance is
    h0 = torch.randn(v, net.hidden, generator=g)   # tight enough to care which draw it sees
    n_v = torch.randn(v, net.hidden, generator=g)
    e_q = torch.randn(net.hidden, generator=g)

    with torch.no_grad():
        h_fused, r_fused = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)
        h_grouped, r_grouped = _prop(net,
            h0, n_v, e_q, src, dst, rel, checkpoint=False,
            src_by_rel=src_g, dst_by_rel=dst_g, rel_offsets=rel_offsets,
        )
    assert torch.equal(r_fused, r_grouped)
    assert torch.allclose(h_fused, h_grouped, atol=GROUP_ATOL, rtol=GROUP_RTOL)


def test_grouped_relation_path_matches_fused_forward_logsumexp():
    """Same parity check under agg="logsumexp" -- the two-pass max/exp-sum stabiliser is the part
    of the grouped path most at risk of silently changing semantics (see gnn.py's _layer
    docstring: the max/sum MUST be taken over all relations landing at a node, not per relation
    block), so this is the test that would catch getting that wrong."""
    v, e = 200, 900
    net = _small_net(layers=3, agg="logsumexp")
    src, dst, rel = _rand_edges(v, e, schema.N_RELATIONS, seed=124)
    src_g, dst_g, rel_offsets = _group_by_rel(src, dst, rel, schema.N_RELATIONS)

    g = torch.Generator().manual_seed(1240)          # seeded: this comparison's tolerance is
    h0 = torch.randn(v, net.hidden, generator=g)   # tight enough to care which draw it sees
    n_v = torch.randn(v, net.hidden, generator=g)
    e_q = torch.randn(net.hidden, generator=g)

    with torch.no_grad():
        h_fused, r_fused = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)
        h_grouped, r_grouped = _prop(net,
            h0, n_v, e_q, src, dst, rel, checkpoint=False,
            src_by_rel=src_g, dst_by_rel=dst_g, rel_offsets=rel_offsets,
        )
    assert torch.equal(r_fused, r_grouped)
    assert not torch.isnan(h_grouped).any()
    assert torch.allclose(h_fused, h_grouped, atol=GROUP_ATOL, rtol=GROUP_RTOL)


def test_grouped_relation_path_gradients_match_fused():
    """Forward parity alone would not catch a grouping bug that only manifests in the backward graph
    (e.g. a block boundary that drops an edge's gradient contribution) -- this checks EVERY parameter's
    gradient, not just the output value."""
    v, e = 150, 700
    net = _small_net(layers=2, agg="sum")
    src, dst, rel = _rand_edges(v, e, schema.N_RELATIONS, seed=321)
    src_g, dst_g, rel_offsets = _group_by_rel(src, dst, rel, schema.N_RELATIONS)

    g = torch.Generator().manual_seed(3210)          # seeded: this comparison's tolerance is
    h0 = torch.randn(v, net.hidden, generator=g)   # tight enough to care which draw it sees
    n_v = torch.randn(v, net.hidden, generator=g)
    e_q = torch.randn(net.hidden, generator=g)

    net.zero_grad()
    h_fused, _ = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)
    h_fused.pow(2).sum().backward()
    grads_fused = {name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None}

    net.zero_grad()
    h_grouped, _ = _prop(net,
        h0, n_v, e_q, src, dst, rel, checkpoint=False,
        src_by_rel=src_g, dst_by_rel=dst_g, rel_offsets=rel_offsets,
    )
    h_grouped.pow(2).sum().backward()
    grads_grouped = {name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None}

    assert set(grads_fused) == set(grads_grouped) and grads_fused           # nonempty: grad DID flow
    for name in grads_fused:
        assert torch.allclose(grads_fused[name], grads_grouped[name], atol=GROUP_ATOL, rtol=GROUP_RTOL), name


def test_grouped_relation_path_golden_direction_and_reach():
    """The grouped path's own golden test: same chain-0->1->...->9 setup as
    test_golden_propagation_direction_and_reach, but driven through src_by_rel/dst_by_rel/
    rel_offsets instead of the fused fallback -- pins message direction and the explicit reached
    mask INDEPENDENTLY on the new code path, not just by agreement with the old one."""
    v, layers = 10, 4
    rel_id = schema.REL_INDEX["next1"]
    src, dst, rel = _chain_edges(v, rel_id)
    src_g, dst_g, rel_offsets = _group_by_rel(src, dst, rel, schema.N_RELATIONS)
    net = _small_net(layers=layers, agg="sum")

    net.rel.forward = lambda e_q: torch.ones(net.n_rel, net.hidden)
    net.upd.forward = lambda x: x[:, net.hidden : 2 * net.hidden]

    h0 = torch.zeros(v, net.hidden)
    h0[0] = torch.randn(net.hidden)
    n_v = torch.zeros(v, net.hidden)
    e_q = torch.zeros(net.hidden)

    h, reached = _prop(net,
        h0, n_v, e_q, src, dst, rel, checkpoint=False,
        src_by_rel=src_g, dst_by_rel=dst_g, rel_offsets=rel_offsets,
    )

    expect_reached = torch.zeros(v, dtype=torch.bool)
    expect_reached[: layers + 1] = True
    assert torch.equal(reached, expect_reached), (reached, expect_reached)
    assert torch.equal(h[layers + 1 :], torch.zeros(v - layers - 1, net.hidden))
    reached_h = h[expect_reached]
    assert reached_h.abs().sum(dim=-1).min().item() > 1e-6
    assert torch.allclose(reached_h, reached_h[0].expand_as(reached_h), atol=1e-5)


def test_grouped_relation_path_empty_relation_block_runs_forward_and_backward():
    """A relation with zero edges must be an empty, harmless block in the grouped path -- exactly the
    single-task-graph case (zero align edges) the design has to support (see edges.py's
    EdgeSet.torch docstring)."""
    v = 15
    net = _small_net(layers=2, agg="sum")
    used_rels = [schema.REL_INDEX["next1"], schema.REL_INDEX["next2"], schema.REL_INDEX["sibling"]]
    rng = np.random.RandomState(11)
    e = 30
    src = torch.as_tensor(rng.randint(0, v, e), dtype=torch.int32)
    dst = torch.as_tensor(rng.randint(0, v, e), dtype=torch.int32)
    rel = torch.as_tensor(rng.choice(used_rels, e), dtype=torch.int8)
    src_g, dst_g, rel_offsets = _group_by_rel(src, dst, rel, schema.N_RELATIONS)
    # confirm the test actually exercises >=1 empty block (e.g. align) before trusting the result
    assert bool((rel_offsets[1:] == rel_offsets[:-1]).any())

    node_x = _node_x(v)
    log_idf = torch.zeros(v)
    qh, vh, gh, w = _query(schema.HIST_H)

    logit = _fwd(
        net, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=False,
        src_by_rel=src_g, dst_by_rel=dst_g, rel_offsets=rel_offsets,
    )
    mask = torch.isfinite(logit)
    assert mask.any()
    logit[mask].sum().backward()      # must not raise


# ---- 2. logsumexp + disconnected node: no NaN (the deg>0 guard) -------------------------------------
def test_logsumexp_no_nan_with_disconnected_node():
    v = 8
    rel_id = schema.REL_INDEX["next1"]
    src, dst, rel = _chain_edges(6, rel_id)                    # edges among nodes 0..5; 6,7 isolated
    net = _small_net(layers=3, agg="logsumexp")

    h0 = torch.zeros(v, net.hidden)
    h0[0] = torch.randn(net.hidden)
    n_v = torch.randn(v, net.hidden)                            # nonzero: exercise the real update MLP
    e_q = torch.randn(net.hidden)

    h, reached = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)
    assert not torch.isnan(h).any()
    assert not torch.isinf(h).any()
    assert bool(reached[0])
    assert not bool(reached[6]) and not bool(reached[7])


# ---- 3. inductivity ----------------------------------------------------------------------------------
def test_inductive_state_dict_has_no_v_sized_dim():
    net = _small_net(layers=2)
    d = net.state_npz_dict()
    # Arbitrary node-count sentinels, chosen to avoid colliding with any of THIS net's own
    # hyperparameters (dim=7, hidden=16, n_rel=schema.N_RELATIONS=12, n_psi=4, layers=2, N_PROTO=32,
    # and the encoder's derived widths hidden+1=17, 2*hidden=32, 3*hidden+1=49) -- a sentinel that
    # happens to equal a legitimate designed dimension would make this test fail on a correct
    # network. 15497 mirrors this checkout's real object-suite V (a realistic "large" graph).
    for v in (19, 15497):
        for k, arr in d.items():
            assert v not in np.shape(arr), f"{k} has a dim == V={v}: shape {np.shape(arr)}"


def test_inductive_param_count_independent_of_graph_size():
    net_small = _small_net(layers=2)
    net_large = _small_net(layers=2)
    n_small = sum(p.numel() for p in net_small.parameters())
    n_large = sum(p.numel() for p in net_large.parameters())
    assert n_small == n_large > 0


def test_inductive_same_state_dict_runs_on_both_graph_sizes():
    net = _small_net(layers=2, agg="sum")
    state = net.state_npz_dict()

    for v, e in ((12, 20), (3000, 9000)):
        net_v = GraphRetrieverNet(dim=DIM, hidden=HIDDEN, layers=2, n_rel=schema.N_RELATIONS,
                                   agg="sum", n_psi=N_PSI)
        for name, mod in net_v._named_modules():
            mod.load_state_dict({k.split(".", 1)[1]: torch.tensor(val)
                                 for k, val in state.items() if k.startswith(name + ".")})
        net_v.set_stats(state["mean"], state["std"])
        net_v.eval()

        node_x = _node_x(v)
        src, dst, rel = _rand_edges(v, e, schema.N_RELATIONS, seed=v)
        log_idf = torch.zeros(v)
        qh, vh, gh, w = _query(schema.HIST_H)

        with torch.no_grad():
            logit = _fwd(net_v, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=False)
        assert logit.shape == (v,)
        assert torch.isfinite(logit[torch.isfinite(logit)]).all()


# ---- 4. empty relation ---------------------------------------------------------------------------
def test_empty_relation_runs_forward_and_backward():
    """A graph where some relations (e.g. align, on a suite too small for any cross-task phase-bin
    match) have zero edges must still run. No special-casing is needed: the relation-gate MLP is
    applied densely to all n_rel rows regardless of which relations edges actually use."""
    v = 15
    net = _small_net(layers=2, agg="sum")
    node_x = _node_x(v)
    used_rels = [schema.REL_INDEX["next1"], schema.REL_INDEX["next2"], schema.REL_INDEX["sibling"]]
    rng = np.random.RandomState(11)
    e = 30
    src = torch.as_tensor(rng.randint(0, v, e), dtype=torch.int32)
    dst = torch.as_tensor(rng.randint(0, v, e), dtype=torch.int32)
    rel = torch.as_tensor(rng.choice(used_rels, e), dtype=torch.int8)
    log_idf = torch.zeros(v)
    qh, vh, gh, w = _query(schema.HIST_H)

    logit = _fwd(net, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=False)
    mask = torch.isfinite(logit)
    assert mask.any()
    logit[mask].sum().backward()      # must not raise


# ---- 5. checkpoint round-trip ----------------------------------------------------------------------
def test_load_npz_state_npz_dict_roundtrip(tmp_path):
    net = _small_net(layers=2, agg="logsumexp")
    net.eval()
    v = 20
    node_x = _node_x(v)
    src, dst, rel = _rand_edges(v, 40, schema.N_RELATIONS, seed=22)
    log_idf = torch.zeros(v)
    qh, vh, gh, w = _query(schema.HIST_H)

    with torch.no_grad():
        logit_a = _fwd(net, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=False)

    path = tmp_path / "g_head.npz"
    np.savez(path, **net.state_npz_dict())
    net2 = GraphRetrieverNet.load_npz(path)

    assert set(net2.state_npz_dict().keys()) == set(net.state_npz_dict().keys())
    assert net2.dim == net.dim and net2.hidden == net.hidden
    assert net2.layers == net.layers and net2.agg == net.agg

    with torch.no_grad():
        logit_b = _fwd(net2, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=False)
    assert torch.equal(logit_a, logit_b)


# ---- 6. seeding ---------------------------------------------------------------------------------
def test_seed_zero_off_topk_and_topk_respected():
    v, h_win, topk = 50, 4, 7
    net = _small_net(layers=1)
    n_v = torch.randn(v, net.hidden)
    e_q = torch.randn(net.hidden)
    qh_n = torch.randn(h_win, DIM)
    w = torch.rand(h_win) + 0.1
    q_nodes_n = torch.randn(v, DIM)

    h0 = _seed(net, e_q, qh_n, w, q_nodes_n, n_v, topk=topk)
    nonzero = h0.abs().sum(-1) > 0
    assert nonzero.sum().item() == topk


def test_seed_larger_sigma_spreads_mass():
    """With e_Q and n_v held constant across nodes, h0's row-sum is exactly proportional to the seed
    mass a, so its concentration (max/sum over the surviving top-k) measures how peaked the learned
    backbone is. sigma is a parameter now, not a call argument."""
    v, h_win, topk = 30, 1, 10
    net = _small_net(layers=1)
    n_v = torch.ones(v, net.hidden)
    e_q = torch.ones(net.hidden)
    qh_n = torch.zeros(h_win, DIM)
    w = torch.ones(h_win)
    q_nodes_n = 0.3 * torch.randn(v, DIM, generator=torch.Generator().manual_seed(3))

    def concentration(sigma: float) -> float:
        with torch.no_grad():
            net.seed_head.log_sigma.fill_(float(np.log(sigma)))
        h0 = _seed(net, e_q, qh_n, w, q_nodes_n, n_v, topk=topk)
        a = h0[:, 0]
        a = a[a > 0]
        return (a.max() / a.sum()).item()

    assert concentration(0.05) > concentration(5.0)


# ---- 7. gradients + checkpoint parity ------------------------------------------------------------
def test_gradients_flow_to_every_param_including_checkpointed_path():
    v = 25
    net = _small_net(layers=3, agg="sum")
    node_x = _node_x(v)
    src, dst, rel = _rand_edges(v, 60, schema.N_RELATIONS, seed=44)
    log_idf = torch.zeros(v)
    qh, vh, gh, w = _query(schema.HIST_H)

    net.zero_grad()
    logit_ckpt = _fwd(net, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=True)
    mask = torch.isfinite(logit_ckpt)
    assert mask.any()
    logit_ckpt[mask].pow(2).sum().backward()

    # The instruction, policy and memory channels are absent from a plain forward.
    absent = ("instr_proj.", "instr_q.", "node_lang.", "policy_proj.", "mem.")
    missing = [n for n, p in net.named_parameters()
               if not n.startswith(absent) and (p.grad is None or p.grad.abs().sum().item() == 0)]
    assert not missing, f"no gradient reached: {missing}"

    with torch.no_grad():
        logit_no_ckpt = _fwd(net, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=False)
    assert torch.equal(mask, torch.isfinite(logit_no_ckpt))
    assert torch.allclose(logit_ckpt.detach()[mask], logit_no_ckpt[mask], atol=1e-5)


# ---- 8. determinism -----------------------------------------------------------------------------
def test_determinism_same_inputs_bit_identical():
    v = 18
    net = _small_net(layers=2, agg="sum")
    net.eval()
    node_x = _node_x(v)
    src, dst, rel = _rand_edges(v, 30, schema.N_RELATIONS, seed=55)
    log_idf = torch.zeros(v)
    qh, vh, gh, w = _query(schema.HIST_H)

    with torch.no_grad():
        a = _fwd(net, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=False)
        b = _fwd(net, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=False)
    assert torch.equal(a, b)


# ---- 9. reached / -inf / softmax-zero on an unreachable component -----------------------------------
def test_reached_unreachable_component_gives_zero_softmax():
    v, off = 12, 6
    rel_id = schema.REL_INDEX["next1"]
    src_a, dst_a, rel_a = _chain_edges(6, rel_id, offset=0)     # component A: nodes 0..5 (seeded)
    src_b, dst_b, rel_b = _chain_edges(6, rel_id, offset=off)   # component B: nodes 6..11 (never seeded)
    src, dst, rel = torch.cat([src_a, src_b]), torch.cat([dst_a, dst_b]), torch.cat([rel_a, rel_b])

    net = _small_net(layers=3, agg="sum")
    h0 = torch.zeros(v, net.hidden)
    h0[0] = torch.randn(net.hidden)
    n_v = torch.randn(v, net.hidden)
    e_q = torch.randn(net.hidden)
    log_idf = torch.zeros(v)

    h, reached = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)
    assert not reached[off:].any()

    logit = _logits(net, h, n_v, reached, log_idf, idf_w=0.0)
    assert torch.isneginf(logit[off:]).all()

    p = F.softmax(logit, dim=0)
    assert torch.equal(p[off:], torch.zeros(v - off))
    assert torch.isclose(p.sum(), torch.tensor(1.0))


# ---- 10. query-dependence ------------------------------------------------------------------------
def test_query_dependence_gate_and_logits_differ():
    v = 20
    net = _small_net(layers=2, agg="sum")
    node_x = _node_x(v)
    src, dst, rel = _rand_edges(v, 40, schema.N_RELATIONS, seed=66)
    log_idf = torch.zeros(v)

    qh1, vh1, gh1, w1 = _query(schema.HIST_H, seed=10)
    qh2, vh2, gh2, w2 = _query(schema.HIST_H, seed=20)

    with torch.no_grad():
        e_q1, e_q2 = net.enc(qh1, vh1, gh1), net.enc(qh2, vh2, gh2)
        assert not torch.allclose(e_q1, e_q2)
        assert not torch.allclose(net.rel(e_q1), net.rel(e_q2))

        logit1 = _fwd(net, qh1, vh1, gh1, node_x, src, dst, rel, log_idf, checkpoint=False)
        logit2 = _fwd(net, qh2, vh2, gh2, node_x, src, dst, rel, log_idf, checkpoint=False)
        mask = torch.isfinite(logit1) & torch.isfinite(logit2)
        assert mask.any()
        assert not torch.allclose(logit1[mask], logit2[mask])


# ---- 10b. movement-cost readout feature (Track GR change 3) ---------------------------------------
def test_logits_movement_cost_defaults_to_zero_and_stays_backward_compatible():
    """q_nodes/q_now default to None -> the movement-cost feature is EXACTLY zero, so an
    existing caller (notably robot_ood.runtime.retrieve.GraphRetriever, which predates this feature and
    calls net.logits(h, n_v, reached, log_idf, idf_w) with the default (all-None) MoveCost)
    keeps working unchanged.
    Omitting the kwargs entirely must be bit-identical to passing q_nodes=None, q_now=None
    explicitly."""
    v = 12
    net = _small_net(layers=2, agg="sum")
    h = torch.randn(v, net.hidden)
    n_v = torch.randn(v, net.hidden)
    reached = torch.ones(v, dtype=torch.bool)
    log_idf = torch.zeros(v)

    with torch.no_grad():
        a = _logits(net, h, n_v, reached, log_idf, idf_w=0.0)
        b = _logits(net, h, n_v, reached, log_idf, idf_w=0.0, q_nodes=None, q_now=None)
    assert torch.equal(a, b)


def test_logits_movement_cost_feature_changes_output_when_supplied():
    """Supplying q_nodes/q_now must actually reach the readout MLP (GraphRetrieverNet.
    out) as a real input, not a no-op -- two different q_now positions (with the SAME h/n_v)
    must generally produce different logits, since the per-node movement cost ||q_v - q_now||
    differs between them."""
    v, dim = 12, DIM
    net = _small_net(layers=2, agg="sum", dim=dim)
    h = torch.randn(v, net.hidden)
    n_v = torch.randn(v, net.hidden)
    reached = torch.ones(v, dtype=torch.bool)
    log_idf = torch.zeros(v)
    q_nodes = torch.randn(v, dim)
    q_now_a = torch.randn(dim)
    q_now_b = q_now_a + 5.0        # far away -> a very different movement-cost column

    with torch.no_grad():
        logit_a = _logits(net, h, n_v, reached, log_idf, idf_w=0.0, q_nodes=q_nodes, q_now=q_now_a)
        logit_b = _logits(net, h, n_v, reached, log_idf, idf_w=0.0, q_nodes=q_nodes, q_now=q_now_b)
        logit_zero = _logits(net, h, n_v, reached, log_idf, idf_w=0.0)
    assert not torch.allclose(logit_a, logit_b)
    assert not torch.allclose(logit_a, logit_zero)


def test_logits_gradient_flows_through_movement_cost_column():
    """Gradient must reach the readout MLP's weight column dedicated to the movement-cost feature (the
    LAST column of net.out[0].weight, since self.out takes [h, n_v, move] concatenated) --
    proof the feature is actually wired into the trainable path, not merely computed and discarded."""
    v, dim = 10, DIM
    net = _small_net(layers=2, agg="sum", dim=dim)
    h = torch.randn(v, net.hidden)
    n_v = torch.randn(v, net.hidden)
    reached = torch.ones(v, dtype=torch.bool)
    log_idf = torch.zeros(v)
    q_nodes = torch.randn(v, dim)
    q_now = torch.randn(dim)

    net.zero_grad()
    logit = _logits(net, h, n_v, reached, log_idf, idf_w=0.0, q_nodes=q_nodes, q_now=q_now)
    logit.pow(2).sum().backward()

    move_col_grad = net.out[0].weight.grad[:, 2 * net.hidden]
    assert torch.isfinite(move_col_grad).all()
    assert float(move_col_grad.abs().sum()) > 0.0


def test_logits_batched_movement_cost_matches_looped_single_query():
    """Batched-input parity, movement-cost feature included: logits(..., q_nodes=q_nodes,
    q_now=q_now_batch) must match B independent logits calls, each with its own
    q_now row and the SAME (graph-only, unbatched) q_nodes."""
    b, v, dim = 4, 15, DIM
    net = _small_net(layers=2, agg="sum", dim=dim)
    h = torch.randn(b, v, net.hidden)
    n_v = torch.randn(v, net.hidden)
    reached = torch.ones(b, v, dtype=torch.bool)
    log_idf = torch.zeros(v)
    q_nodes = torch.randn(v, dim)
    q_now = torch.randn(b, dim)

    with torch.no_grad():
        single = torch.stack([
            _logits(net, h[i], n_v, reached[i], log_idf, idf_w=0.0, q_nodes=q_nodes, q_now=q_now[i])
            for i in range(b)
        ])
        batched = _logits(net, h, n_v, reached, log_idf, idf_w=0.0, q_nodes=q_nodes, q_now=q_now)
    assert torch.allclose(single, batched, atol=1e-5, rtol=1e-4)


def test_forward_auto_wires_movement_cost_from_qh():
    """forward() ALWAYS threads the movement-cost feature through internally
    (q_now = qh_n[-1], the window's own most recent row) -- so a plain forward() call must
    differ from a hand-built logits() call that leaves q_nodes/q_now at their zero-feature
    default, confirming the auto-wiring actually happens rather than silently staying off."""
    v = 20
    net = _small_net(layers=2, agg="sum")
    node_x = _node_x(v)
    src, dst, rel = _rand_edges(v, 40, schema.N_RELATIONS, seed=77)
    log_idf = torch.zeros(v)
    qh, vh, gh, w = _query(schema.HIST_H, seed=5)

    with torch.no_grad():
        logit_auto = _fwd(net, qh, vh, gh, node_x, src, dst, rel, log_idf, checkpoint=False)

        # replicate forward()'s own pipeline up to logits(), but WITHOUT the movement-cost feature
        query = net.query(QueryWindow(qh=qh, vh=vh, gh=gh))
        e_q = query.e_q
        n_v = net.node_encode(node_x)
        q_nodes_n = net.normalize_q(node_x[:, : net.dim])
        h0, _ = net.seed(
            query, NodeCloud(q_nodes_n=q_nodes_n, n_v=n_v, columns=_columns(v, net.dim)),
            PipelineParams(topk=schema.SEED_TOPK),
        )
        h_l, reached = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)
        logit_no_move = _logits(net, h_l, n_v, reached, log_idf, 0.0)

    mask = torch.isfinite(logit_auto) & torch.isfinite(logit_no_move)
    assert mask.any()
    assert not torch.allclose(logit_auto[mask], logit_no_move[mask])


# ---- QueryEncoder in isolation ---------------------------------------------------------------------
def test_query_encoder_shape():
    enc = QueryEncoder(DIM, hidden=HIDDEN)
    qh, vh, gh, w = _query(schema.HIST_H)
    assert enc(qh, vh, gh).shape == (HIDDEN,)


def test_query_encoder_pool_weights_are_a_distribution_over_the_window():
    """pool_w is what LearnedSeed.mixture spreads its seed mass by, so it has to be a genuine
    distribution over the H window rows, not merely non-negative."""
    h_win = 5
    enc = QueryEncoder(DIM, hidden=HIDDEN)
    enc.eval()
    g = torch.Generator().manual_seed(7)
    qh = torch.randn(h_win, DIM, generator=g)
    vh = torch.randn(h_win, DIM, generator=g)
    gh = torch.rand(h_win, generator=g)

    with torch.no_grad():
        out = enc.encode(qh, vh, gh)
    assert out.pool_w.shape == (h_win,)
    assert torch.all(out.pool_w >= 0.0)
    assert torch.isclose(out.pool_w.sum(), torch.tensor(1.0), atol=1e-6)


def test_query_encoder_weights_a_frame_by_its_content_not_its_position():
    """The pooling head reads each frame's own features, so perturbing one row's q must move
    pool_w. A position-only rule (recency) would leave the weights untouched."""
    h_win = 5
    enc = QueryEncoder(DIM, hidden=HIDDEN)
    enc.eval()
    g = torch.Generator().manual_seed(7)
    qh = torch.randn(h_win, DIM, generator=g)
    vh = torch.randn(h_win, DIM, generator=g)
    gh = torch.rand(h_win, generator=g)
    qh_perturbed = qh.clone()
    qh_perturbed[2] = torch.randn(DIM, generator=g) * 100

    with torch.no_grad():
        w_clean = enc.encode(qh, vh, gh).pool_w
        w_dirty = enc.encode(qh_perturbed, vh, gh).pool_w
    assert not torch.allclose(w_clean, w_dirty, atol=1e-6)


def test_frame_dropout_fires_only_in_training_mode():
    """Frame dropout is the parameter-free control that stops the pooling softmax collapsing onto
    one row. It must leave eval-mode inference deterministic."""
    h_win = 8
    enc = QueryEncoder(DIM, hidden=HIDDEN, frame_dropout=0.9)
    g = torch.Generator().manual_seed(3)
    qh = torch.randn(h_win, DIM, generator=g)
    vh = torch.randn(h_win, DIM, generator=g)
    gh = torch.rand(h_win, generator=g)

    enc.eval()
    with torch.no_grad():
        assert torch.equal(enc.encode(qh, vh, gh).pool_w, enc.encode(qh, vh, gh).pool_w)

    enc.train()
    torch.manual_seed(0)
    draws = [enc.encode(qh, vh, gh).pool_w for _ in range(8)]
    assert any(not torch.allclose(draws[0], d) for d in draws[1:])
    for d in draws:                                   # one row always survives the mask
        assert torch.isclose(d.sum(), torch.tensor(1.0), atol=1e-6)






# ======================================================================================================
# 11. BATCHED PATH — a batch of B queries per forward/backward instead of one at a time (pure
# throughput change; see src/robot_ood/graph/gnn.py's "BATCHED PATH" banner and robot_ood.train.loop's "BATCHED
# QUERIES" module docstring section for the full motivation). Every test below compares the batched
# method against B independent calls of its single-query sibling, which the tests ABOVE this banner
# have already pinned down; the single-query methods themselves are untouched by this addition.
#
# torch.equal bit-identity across batched vs. one-at-a-time is not achievable: index_add_/
# index_reduce_ accumulation order changes when the same edges scatter into a [B, V, hidden]
# tensor instead of B separate [V, hidden] ones (floating-point reassociation) -- exactly the same
# situation as the grouped-vs-fused relation path above (see GROUP_ATOL/GROUP_RTOL's comment), so
# these reuse the identical tolerance rationale: 1e-4 absolute/relative is loose enough to absorb
# reassociation noise on a few hundred edges and a handful of layers, tight enough that a real bug --
# batch rows mixed together, a transposed dim, a dropped edge -- would still fail by orders of
# magnitude.
# ======================================================================================================
BATCH_ATOL, BATCH_RTOL = 1e-4, 1e-4


def _query_batch(b: int, h: int, dim: int = DIM, seed0: int = 1):
    """b independent single-query tuples (see _query), stacked into a batch."""
    qs = [_query(h, dim=dim, seed=seed0 + i) for i in range(b)]
    qh = torch.stack([q[0] for q in qs])
    vh = torch.stack([q[1] for q in qs])
    gh = torch.stack([q[2] for q in qs])
    w = torch.stack([q[3] for q in qs])
    return qh, vh, gh, w


def test_golden_propagation_direction_and_reach_batched():
    """Batched sibling of test_golden_propagation_direction_and_reach -- same chain 0->1->...->9,
    but THREE queries seeded at three DIFFERENT nodes in ONE batched propagate call. This is
    the load-bearing batched test: a dim mix-up in _layer_batch (e.g. scattering along dim 0
    instead of dim 1, or gathering h[s_r] instead of h[:, s_r]) would blend one batch row's
    messages into another's, which a batch of IDENTICAL queries (or B=1) could never catch -- three
    DIFFERENT seed nodes make any cross-row leakage visible as a wrong reached window or nonzero mass
    where a row's own chain distance says there should be none.
    """
    v, layers = 10, 4
    rel_id = schema.REL_INDEX["next1"]
    src, dst, rel = _chain_edges(v, rel_id)
    net = _small_net(layers=layers, agg="sum")

    net.rel.forward = lambda e_q: torch.ones(e_q.shape[0], net.n_rel, net.hidden)  # identity gate
    net.upd.forward = lambda x: x[..., net.hidden : 2 * net.hidden]                       # pass out raw

    seeds = [0, 2, 5]                            # three DIFFERENT seed nodes, one per batch row
    b = len(seeds)
    h0 = torch.zeros(b, v, net.hidden)
    for i, s in enumerate(seeds):
        h0[i, s] = torch.randn(net.hidden)
    n_v = torch.zeros(v, net.hidden)                                            # unused (upd ignores it)
    e_q = torch.zeros(b, net.hidden)                                            # unused (rel ignores it)

    h, reached = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)

    for i, s in enumerate(seeds):
        lo, hi = s, min(s + layers + 1, v)
        expect_reached = torch.zeros(v, dtype=torch.bool)
        expect_reached[lo:hi] = True
        assert torch.equal(reached[i], expect_reached), (i, s, reached[i], expect_reached)

        assert torch.equal(h[i, :lo], torch.zeros(lo, net.hidden))              # nothing BEHIND the seed
        assert torch.equal(h[i, hi:], torch.zeros(v - hi, net.hidden))          # nothing beyond L hops
        reached_h = h[i, expect_reached]
        assert reached_h.abs().sum(dim=-1).min().item() > 1e-6                  # every reached row nonzero
        assert torch.allclose(reached_h, reached_h[0].expand_as(reached_h), atol=1e-5)  # traveling wave


def test_grouped_relation_path_golden_direction_and_reach_batched():
    """Batched sibling of test_grouped_relation_path_golden_direction_and_reach -- pins message
    direction and the explicit reached mask on the batched GROUPED path independently, not just by
    agreement with the batched fused path."""
    v, layers = 10, 4
    rel_id = schema.REL_INDEX["next1"]
    src, dst, rel = _chain_edges(v, rel_id)
    src_g, dst_g, rel_offsets = _group_by_rel(src, dst, rel, schema.N_RELATIONS)
    net = _small_net(layers=layers, agg="sum")

    net.rel.forward = lambda e_q: torch.ones(e_q.shape[0], net.n_rel, net.hidden)
    net.upd.forward = lambda x: x[..., net.hidden : 2 * net.hidden]

    seeds = [1, 4]
    b = len(seeds)
    h0 = torch.zeros(b, v, net.hidden)
    for i, s in enumerate(seeds):
        h0[i, s] = torch.randn(net.hidden)
    n_v = torch.zeros(v, net.hidden)
    e_q = torch.zeros(b, net.hidden)

    h, reached = _prop(
        net, h0, n_v, e_q, src, dst, rel, checkpoint=False,
        src_by_rel=src_g, dst_by_rel=dst_g, rel_offsets=rel_offsets,
    )

    for i, s in enumerate(seeds):
        lo, hi = s, min(s + layers + 1, v)
        expect_reached = torch.zeros(v, dtype=torch.bool)
        expect_reached[lo:hi] = True
        assert torch.equal(reached[i], expect_reached), (i, s, reached[i], expect_reached)
        assert torch.equal(h[i, :lo], torch.zeros(lo, net.hidden))
        assert torch.equal(h[i, hi:], torch.zeros(v - hi, net.hidden))
        reached_h = h[i, expect_reached]
        assert reached_h.abs().sum(dim=-1).min().item() > 1e-6
        assert torch.allclose(reached_h, reached_h[0].expand_as(reached_h), atol=1e-5)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("agg", ["sum", "logsumexp"])
def test_batched_forward_matches_looped_single_query_forward(agg, grouped):
    """CORRECTNESS REQUIREMENT 1 (numerical parity): batched output for B queries must match the
    single-query path run B times, to the tolerance declared above (BATCH_ATOL/BATCH_RTOL)."""
    v, e, b = 200, 900, 5
    net = _small_net(layers=3, agg=agg)
    src, dst, rel = _rand_edges(v, e, schema.N_RELATIONS, seed=123)
    kwargs = {}
    if grouped:
        src_g, dst_g, rel_offsets = _group_by_rel(src, dst, rel, schema.N_RELATIONS)
        kwargs = dict(src_by_rel=src_g, dst_by_rel=dst_g, rel_offsets=rel_offsets)

    node_x = _node_x(v)
    log_idf = torch.zeros(v)
    qh_b, vh_b, gh_b, w_b = _query_batch(b, schema.HIST_H, seed0=100)

    with torch.no_grad():
        single = torch.stack([
            _fwd(net, qh_b[i], vh_b[i], gh_b[i], node_x, src, dst, rel, log_idf,
                 checkpoint=False, **kwargs)
            for i in range(b)
        ])
        batched = _fwd(net, qh_b, vh_b, gh_b, node_x, src, dst, rel, log_idf,
                        checkpoint=False, **kwargs)

    assert torch.equal(torch.isfinite(single), torch.isfinite(batched)), "reached set differs"
    mask = torch.isfinite(single)
    assert mask.any()
    assert torch.allclose(single[mask], batched[mask], atol=BATCH_ATOL, rtol=BATCH_RTOL), \
        (single[mask] - batched[mask]).abs().max()


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("agg", ["sum", "logsumexp"])
def test_batched_gradient_matches_looped_single_query_gradient(agg, grouped):
    """CORRECTNESS REQUIREMENT 2 (gradient parity): the gradient from one batch of B must match the
    ACCUMULATED gradient from B single-query steps -- the single-query reference here takes the MEAN
    over B independent .backward() calls ((loss_i / b).backward() each), mirroring
    robot_ood.train.loop's own mean-over-batch convention (see that module's docstring, "THE MEAN, NOT
    THE SUM"), so this test also stands as a check that the training loop's gradient-averaging story is
    consistent with what the network itself produces batched vs. looped one query at a time."""
    v, e, b = 150, 700, 4
    net = _small_net(layers=2, agg=agg)
    src, dst, rel = _rand_edges(v, e, schema.N_RELATIONS, seed=321)
    kwargs = {}
    if grouped:
        src_g, dst_g, rel_offsets = _group_by_rel(src, dst, rel, schema.N_RELATIONS)
        kwargs = dict(src_by_rel=src_g, dst_by_rel=dst_g, rel_offsets=rel_offsets)
    node_x = _node_x(v)
    log_idf = torch.zeros(v)
    qh_b, vh_b, gh_b, w_b = _query_batch(b, schema.HIST_H, seed0=200)

    net.zero_grad()
    for i in range(b):
        logit = _fwd(net, qh_b[i], vh_b[i], gh_b[i], node_x, src, dst, rel, log_idf,
                     checkpoint=True, **kwargs)
        mask = torch.isfinite(logit)
        (logit[mask].pow(2).sum() / b).backward()
    grads_single = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    net.zero_grad()
    logit_b = _fwd(net, qh_b, vh_b, gh_b, node_x, src, dst, rel, log_idf,
                   checkpoint=True, **kwargs)
    mask_b = torch.isfinite(logit_b)
    (logit_b[mask_b].pow(2).sum() / b).backward()
    grads_batched = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    assert set(grads_single) == set(grads_batched) and grads_single       # nonempty: grad DID flow
    for name in grads_single:
        gs, gb = grads_single[name], grads_batched[name]
        assert torch.allclose(gs, gb, atol=BATCH_ATOL, rtol=1e-3), (name, (gs - gb).abs().max())


def test_batched_logsumexp_no_nan_with_disconnected_node():
    """CORRECTNESS REQUIREMENT 4: agg="logsumexp"'s deg > 0 guard (module docstring's
    -1e30 + log(0) -> NaN note) must still work batched -- a disconnected node stays disconnected
    for EVERY batch row (it is a property of the graph, not the query), and must produce neither NaN
    nor Inf for any row."""
    v, b = 8, 3
    rel_id = schema.REL_INDEX["next1"]
    src, dst, rel = _chain_edges(6, rel_id)                    # edges among nodes 0..5; 6,7 isolated
    net = _small_net(layers=3, agg="logsumexp")

    h0 = torch.zeros(b, v, net.hidden)
    for i in range(b):
        h0[i, 0] = torch.randn(net.hidden)
    n_v = torch.randn(v, net.hidden)                            # nonzero: exercise the real update MLP
    e_q = torch.randn(b, net.hidden)

    h, reached = _prop(net, h0, n_v, e_q, src, dst, rel, checkpoint=False)
    assert not torch.isnan(h).any()
    assert not torch.isinf(h).any()
    assert bool(reached[:, 0].all())
    assert not reached[:, 6].any() and not reached[:, 7].any()


def test_batched_inductive_param_count_independent_of_batch_size():
    """CORRECTNESS REQUIREMENT 5 (inductivity), extended along the new batch axis: the batched path
    introduces no parameter, and running batched forward at several DIFFERENT batch sizes on the SAME
    net must not change its parameter count or values (no lazily-created, B-shaped weight) -- batching
    is a throughput change to how many queries share one forward pass, never a per-B specialisation of
    the network. state_npz_dict's own |V|-free checks (test_inductive_state_dict_has_no_v_
    sized_dim) already cover the batch-independent module weights this reuses unmodified."""
    net = _small_net(layers=2)
    n_params_before = sum(p.numel() for p in net.parameters())
    state_before = {k: v.clone() for k, v in net.state_dict().items()}

    v, e = 30, 80
    node_x = _node_x(v)
    src, dst, rel = _rand_edges(v, e, schema.N_RELATIONS, seed=7)
    log_idf = torch.zeros(v)

    for b in (1, 3, 7, 16):
        qh_b, vh_b, gh_b, w_b = _query_batch(b, schema.HIST_H, seed0=500)
        with torch.no_grad():
            logit = _fwd(net, qh_b, vh_b, gh_b, node_x, src, dst, rel, log_idf,
                         checkpoint=False)
        assert logit.shape == (b, v)

    n_params_after = sum(p.numel() for p in net.parameters())
    assert n_params_after == n_params_before
    for k in state_before:
        assert torch.equal(state_before[k], net.state_dict()[k])


def test_seed_batch_zero_off_topk_and_topk_respected_per_row():
    """Batched sibling of test_seed_zero_off_topk_and_topk_respected -- EACH batch row must respect
    its OWN top-topk sparsification independently (not, e.g., a shared top-k across the whole
    batch, which would silently couple unrelated queries' seed sets)."""
    v, h_win, topk, b = 50, 4, 7, 3
    net = _small_net(layers=1)
    n_v = torch.randn(v, net.hidden)
    e_q = torch.randn(b, net.hidden)
    qh_n = torch.randn(b, h_win, DIM)
    w = torch.rand(b, h_win) + 0.1
    q_nodes_n = torch.randn(v, DIM)

    h0 = _seed(net, e_q, qh_n, w, q_nodes_n, n_v, topk=topk)
    assert h0.shape == (b, v, net.hidden)
    for i in range(b):
        nonzero = h0[i].abs().sum(-1) > 0
        assert nonzero.sum().item() == topk, (i, nonzero.sum().item())


def test_query_encoder_batched_input_matches_looped_single_query():
    """QueryEncoder on a batched window must match forward run once per batch row -- the pooling
    reduction's axis moves from the sole dim ([H]) to the middle dim ([B,H]), the one detail a
    batching bug in this class could get wrong."""
    b = 6
    enc = QueryEncoder(DIM, hidden=HIDDEN)
    enc.eval()
    qh_b, vh_b, gh_b, w_b = _query_batch(b, schema.HIST_H, seed0=10)

    with torch.no_grad():
        single = torch.stack([enc(qh_b[i], vh_b[i], gh_b[i]) for i in range(b)])
        batched = enc(qh_b, vh_b, gh_b)
    assert batched.shape == (b, HIDDEN)
    assert torch.allclose(single, batched, atol=1e-5, rtol=1e-4)


# ======================================================================================================
# 12. TRAIN/SERVE PARITY -- "the quantity measured was not the quantity deployed", made un-reintroducible
# ======================================================================================================
# Six separate bugs in this project came from exactly that pattern (see robot_ood.runtime.retrieve's module
# docstring, "DIVERGENCE A", for the most recent and most expensive one: retrieve() silently zeroed the
# readout's movement-cost channel for a full session while robot_ood.train.loss._forward_batch always passed it
# -- 22x worse phase error before the fix was even noticed). This ONE test needs a real (toy)
# NodeTable/EdgeSet graph, not the bare hand-built tensors the rest of this module uses (this module's
# own "no dataset dependency" contract, see the module docstring) -- the point is to exercise
# GraphRetriever's ACTUAL cached deploy-time pipeline, not a hand-rebuilt stand-in for it, so a toy graph
# (still no real dataset on disk) is the smallest fixture that lets both sides run unmodified code.
def test_retrieve_logits_match_train_query_loss_batch():
    """Build ONE window on a small toy graph and assert robot_ood.runtime.retrieve.GraphRetriever.observe
    's logits (retrieve()'s ACTUAL deploy-time pipeline) agree, to < 1e-6, with a hand-inlined copy of
    robot_ood.train.loss.QueryLossComputer's own pipeline (_forward_batch: enc -> node_encode ->
    normalize_q -> seed -> propagate -> logits, WITH q_nodes=/q_now=) run on the exact SAME
    net/graph/window. A future edit that re-diverges either copy (e.g. someone "simplifies" one call site
    and drops the movement-cost kwargs) fails HERE, at unit-test speed, instead of surfacing as a
    multi-x phase-error regression measured against a trained checkpoint days later.
    """
    from robot_ood.core.config import GraphConfig
    from robot_ood.data.edges import EdgeSet
    from robot_ood.data.nodes import NodeTable
    from robot_ood.runtime.retrieve import GraphRetriever
    from robot_ood.data.geometry import finite_diff_vel

    # ---- the smallest toy demo graph this invariant needs: one straight-line demo, real NodeTable/
    # EdgeSet machinery (not hand-built tensors) so GraphRetriever's ACTUAL __init__ caching runs.
    # N_FRAMES/COARSEN mirror tests/test_retrieve.py's fixture (_make_demo/_toy_nodes_edges) -- the
    # same combination the phase-convention invariant (NodeTable.validate) is already known to accept.
    n_frames, coarsen = 31, 3
    rng = np.random.RandomState(9)
    start = rng.randn(DIM) * 0.01
    direction = rng.randn(DIM)
    direction /= np.linalg.norm(direction)
    t = np.arange(n_frames, dtype=np.float32)
    q = (start[None, :] + 0.05 * t[:, None] * direction[None, :]).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[1:] = q[1:] - q[:-1]
    grip = np.zeros(n_frames, dtype=np.uint8)
    grip[n_frames // 2 :] = 1
    demo = {"q": q, "qdot": qdot, "grip": grip, "stage": grip.astype(np.int16), "task_id": 0}

    nodes = NodeTable.from_demos([demo], task_names=["taskA"], coarsen=coarsen)
    edges = EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu")

    net = GraphRetrieverNet(dim=nodes.dim, hidden=HIDDEN, layers=2, n_rel=schema.N_RELATIONS, agg="sum", n_psi=N_PSI)
    mean, std = nodes.feature_stats()
    net.set_stats(mean, std)
    net.eval()

    cfg = GraphConfig(device="cpu")
    retriever = GraphRetriever(net, nodes, edges, cfg=cfg, device="cpu")

    # a real clean window, exactly cfg.hist raw frames -- no padding, so BOTH pipelines see byte-for-byte
    # the same raw input (no need to separately reproduce edge_pad_hist's behaviour here).
    hist = cfg.hist
    assert len(nodes.q_raw) >= hist
    q_hist = nodes.q_raw[:hist].astype(np.float64)

    # ---- side 1: retrieve()'s ACTUAL deploy-time pipeline (GraphRetriever.observe, unmodified) -------
    obs = retriever.observe(q_hist)
    logits_deploy = obs.logits64

    # ---- side 2: robot_ood.train.loss._forward_batch's pipeline, hand-inlined ---------
    grip_col = nodes.grip.astype(np.float32)[:, None]
    t_frac_col = nodes.t_frac.astype(np.float32)[:, None]
    node_x = np.concatenate([nodes.q, nodes.qdot, grip_col, t_frac_col], axis=1).astype(np.float32)
    node_x_t = torch.as_tensor(node_x)
    et = edges.torch("cpu")
    src_t, dst_t, rel_t, log_idf_t = et["src"], et["dst"], et["rel"], et["log_idf"]

    v_hist = finite_diff_vel(q_hist)
    g_hist = np.zeros(hist, dtype=np.float32)

    qh = torch.as_tensor(q_hist, dtype=torch.float32)
    vh = torch.as_tensor(v_hist, dtype=torch.float32)
    gh = torch.as_tensor(g_hist, dtype=torch.float32)

    with torch.no_grad():
        enc = net.enc.encode(qh, vh, gh)
        e_q = enc.e_q
        n_v = net.node_encode(node_x_t)
        q_nodes_n = net.normalize_q(node_x_t[:, : net.dim])
        qh_n = net.normalize_q(qh)
        # The metric columns the TRAIN path builds -- off node_x_t plus phase/task, which are not
        # node features -- must equal the ones GraphRetriever caches off the node table.
        columns = NodeMetricColumns(
            qdot=node_x_t[:, net.dim : 2 * net.dim], grip=node_x_t[:, 2 * net.dim],
            phase=torch.as_tensor(nodes.phase, dtype=torch.float32),
            task_id=torch.as_tensor(nodes.task_id.astype(np.int64)),
        )
        h0, _ = net.seed(
            SeedQuery(e_q=e_q, z_h=enc.z, pool_w=enc.pool_w, qh_n=qh_n,
                      columns=QueryMetricColumns(qdot=vh, grip=gh)),
            NodeCloud(q_nodes_n=q_nodes_n, n_v=n_v, columns=columns),
            PipelineParams(topk=cfg.seed_topk, tier1_m=cfg.seed_tier1_m),
        )
        h_l, reached = _prop(net, h0, n_v, e_q, src_t, dst_t, rel_t, checkpoint=False)
        logits_train = _logits(
            net, h_l, n_v, reached, log_idf_t, cfg.idf, q_nodes=q_nodes_n, q_now=qh_n[-1],
        ).double()

    finite_deploy = torch.isfinite(logits_deploy)
    finite_train = torch.isfinite(logits_train)
    assert torch.equal(finite_deploy, finite_train), "reached set differs between retrieve() and train's pipeline"
    assert bool(finite_deploy.any())
    assert torch.allclose(logits_deploy[finite_deploy], logits_train[finite_train], atol=1e-6)
