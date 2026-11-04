"""Tests for robot_ood.data.edges — the demonstration-graph edge-construction module.

robot_ood.data.nodes.NodeTable (the sibling module this builds on) is being written in parallel, so
these tests build a small, fully hand-enumerable stand-in (_toy_nodes) that exposes exactly
the NodeTable contract edges.py is written against: q/qdot [V,D] f32, grip [V] u1 (1 =
closed), stage/t_idx/t_frac/t_raw/n_members/owner/task_id/phase [V],
demo_ptr [J+1] i4, q_raw/qdot_raw/raw_ptr, task_names, coarsen, plus
__len__/.dim/.n_demos/.demo_nodes(owner).

stage is kept on the toy table (it is still a diagnostic field on the real NodeTable, see
schema.py) but edge construction must not read it — test_edges_ignore_corrupted_stage
below is the machine-checkable version of that claim.

TOY GRAPH LAYOUT (15 nodes, 3 demos, 2 tasks, both gripper states)::

    demo A  owner=0  task=0   nodes  0.. 4  grip=[0,0,0,1,1]  t_idx=0..4
    demo B  owner=1  task=0   nodes  5..10  grip=[0,0,0,1,1,1]  t_idx=0..5
    demo C  owner=2  task=1   nodes 11..14  grip=[0,0,1,1]     t_idx=0..3

Every relation below is verified against a reference computed with a SEPARATE, obviously-correct
definition (plain nested loops over the same node attributes) rather than by re-deriving the
implementation's own masking logic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from robot_ood.data import schema
from robot_ood.data.edges import (
    EdgeSet,
    RelationEdges,
    align_edges,
    knn_masked,
    node_idf,
    sibling_edges,
    temporal_edges,
)

LARGE_K = 100   # >> any toy bucket size, so kNN degenerates to "connect every legal pair" (deterministic)


# ======================================================================================================
# toy NodeTable stand-in
# ======================================================================================================
@dataclasses.dataclass
class _ToyNodeTable:
    q: np.ndarray
    qdot: np.ndarray
    grip: np.ndarray
    stage: np.ndarray
    t_idx: np.ndarray
    t_frac: np.ndarray
    t_raw: np.ndarray
    n_members: np.ndarray
    owner: np.ndarray
    task_id: np.ndarray
    phase: np.ndarray
    demo_ptr: np.ndarray
    q_raw: np.ndarray
    qdot_raw: np.ndarray
    raw_ptr: np.ndarray
    task_names: list
    coarsen: int = 1

    def __len__(self) -> int:
        return len(self.q)

    @property
    def dim(self) -> int:
        return self.q.shape[1]

    @property
    def n_demos(self) -> int:
        return len(self.demo_ptr) - 1

    def demo_nodes(self, owner: int) -> np.ndarray:
        return np.arange(int(self.demo_ptr[owner]), int(self.demo_ptr[owner + 1]))


def _toy_nodes() -> _ToyNodeTable:
    """Build the 15-node / 3-demo / 2-task toy graph described in the module docstring.

    stage is carried on the table only because the real NodeTable still carries it as a
    diagnostic field (schema.py) -- it is populated with a plausible-looking (but, per the change
    under test, entirely IRRELEVANT to edge construction) release-count pattern, never read by any
    function under test here except in test_edges_ignore_corrupted_stage, which corrupts it
    on purpose.
    """
    stage = np.array(
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int16
    )
    grip = stage.astype(np.uint8).copy()          # open pre-grasp, closed post-grasp, per demo

    owner = np.array([0] * 5 + [1] * 6 + [2] * 4, dtype=np.int32)
    task_id = np.array([0] * 5 + [0] * 6 + [1] * 4, dtype=np.int16)
    t_idx = np.array(list(range(5)) + list(range(6)) + list(range(4)), dtype=np.int32)
    phase = np.array(
        [i / 4 for i in range(5)] + [i / 5 for i in range(6)] + [i / 3 for i in range(4)],
        dtype=np.float32,
    )
    demo_ptr = np.array([0, 5, 11, 15], dtype=np.int32)

    V, D = 15, 3
    rng = np.random.RandomState(0)
    # small joint-space-realistic scale (std ~ schema.KERNEL_BW) so sibling/align kernel weights
    # (exp(-d^2/(2*bw^2))) don't uniformly float32-underflow to 0 -- distinctness is all that
    # matters for the topology tests, but the weight-sanity test wants a physically plausible spread.
    q = (rng.randn(V, D) * schema.KERNEL_BW).astype(np.float32)
    qdot = rng.randn(V, D).astype(np.float32) * 0.1

    return _ToyNodeTable(
        q=q, qdot=qdot, grip=grip, stage=stage, t_idx=t_idx,
        t_frac=(t_idx.astype(np.float32)), t_raw=t_idx.astype(np.int32),
        n_members=np.ones(V, dtype=np.int16), owner=owner, task_id=task_id, phase=phase,
        demo_ptr=demo_ptr, q_raw=q.copy(), qdot_raw=qdot.copy(),
        raw_ptr=np.arange(V + 1, dtype=np.int32), task_names=["task0", "task1"], coarsen=1,
    )


def _single_task_nodes() -> _ToyNodeTable:
    """9-node / 2-demo / SINGLE-task toy graph (demo A + demo B of _toy_nodes, both task 0).

    align is defined as cross-TASK; with only one task in the whole table it has no legal partner
    for any node, so it legitimately builds to zero edges. This is the new home of the "some relation
    is empty but the graph must still build/validate/save/load" case -- previously exercised by
    stage_next/stage_prev on a single-stage graph, which no longer exist.
    """
    stage = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int16)
    grip = stage.astype(np.uint8).copy()
    owner = np.array([0] * 5 + [1] * 6, dtype=np.int32)
    task_id = np.zeros(11, dtype=np.int16)
    t_idx = np.array(list(range(5)) + list(range(6)), dtype=np.int32)
    phase = np.array([i / 4 for i in range(5)] + [i / 5 for i in range(6)], dtype=np.float32)
    demo_ptr = np.array([0, 5, 11], dtype=np.int32)

    V, D = 11, 3
    rng = np.random.RandomState(7)
    q = (rng.randn(V, D) * schema.KERNEL_BW).astype(np.float32)
    qdot = rng.randn(V, D).astype(np.float32) * 0.1

    return _ToyNodeTable(
        q=q, qdot=qdot, grip=grip, stage=stage, t_idx=t_idx,
        t_frac=(t_idx.astype(np.float32)), t_raw=t_idx.astype(np.int32),
        n_members=np.ones(V, dtype=np.int16), owner=owner, task_id=task_id, phase=phase,
        demo_ptr=demo_ptr, q_raw=q.copy(), qdot_raw=qdot.copy(),
        raw_ptr=np.arange(V + 1, dtype=np.int32), task_names=["task0"], coarsen=1,
    )


def _edge_set(src, dst, rel) -> set:
    return set(zip(src.tolist(), dst.tolist(), rel.tolist()))


# ======================================================================================================
# knn_masked — brute-force agreement (with and without a pre-topk mask)
# ======================================================================================================
def test_knn_masked_matches_bruteforce_no_mask():
    rng = np.random.RandomState(1)
    X = rng.randn(500, 5).astype(np.float32)
    Y = rng.randn(300, 5).astype(np.float32)
    k = 7

    idx, dist = knn_masked(torch.as_tensor(X), torch.as_tensor(Y), k)

    d_bf = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    order = np.argsort(d_bf, axis=1, kind="stable")[:, :k]
    for i in range(len(X)):
        assert set(idx[i].tolist()) == set(order[i].tolist()), i
        np.testing.assert_allclose(
            np.sort(dist[i]), np.sort(d_bf[i, order[i]]), atol=1e-4, err_msg=f"row {i}"
        )


def test_knn_masked_matches_bruteforce_with_mask():
    """The pre-topk-masking correctness test: block half of Y (by parity) and confirm the returned
    neighbours are always the true nearest LEGAL ones, never a masked-out point."""
    rng = np.random.RandomState(2)
    X = rng.randn(500, 5).astype(np.float32)
    Y = rng.randn(300, 5).astype(np.float32)
    k = 7

    row_parity = (np.arange(500) % 2)[:, None]
    col_parity = (np.arange(300) % 2)[None, :]
    block_np = row_parity == col_parity              # block same-parity columns (half of Y, >= k left)
    block = torch.as_tensor(block_np)

    idx, dist = knn_masked(torch.as_tensor(X), torch.as_tensor(Y), k, block=block)
    assert np.isfinite(dist).all(), "every row has >= k legal candidates, no slot should be inf"

    d_bf = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    d_bf_masked = np.where(block_np, np.inf, d_bf)
    order = np.argsort(d_bf_masked, axis=1, kind="stable")[:, :k]
    for i in range(len(X)):
        assert set(idx[i].tolist()) == set(order[i].tolist()), i
        # none of the returned neighbours may be a masked (same-parity) column
        assert not np.any(block_np[i, idx[i]]), i


def test_knn_masked_starved_row_reports_inf():
    """Fewer legal candidates than k -> the leftover slots come back with dist == inf, not a
    silently-wrong nearest-illegal-neighbour."""
    X = torch.zeros(1, 3)
    Y = torch.randn(5, 3)
    block = torch.ones(1, 5, dtype=torch.bool)
    block[0, 0] = False   # exactly one legal neighbour
    idx, dist = knn_masked(X, Y, k=4, block=block)
    assert np.isfinite(dist[0]).sum() == 1
    assert idx[0, np.isfinite(dist[0])][0] == 0


# ======================================================================================================
# temporal_edges
# ======================================================================================================
def test_temporal_edges_exact():
    nodes = _toy_nodes()
    src, dst, rel = temporal_edges(nodes)
    got = _edge_set(src, dst, rel)

    # independent reference: plain nested loop over each demo's contiguous t_idx run
    expected = set()
    for lo, hi in zip(nodes.demo_ptr[:-1], nodes.demo_ptr[1:]):
        n = hi - lo
        for d in schema.DILATIONS:
            for i in range(n - d):
                expected.add((lo + i, lo + i + d, schema.REL_INDEX[f"next{d}"]))
                expected.add((lo + i + d, lo + i, schema.REL_INDEX[f"prev{d}"]))

    assert got == expected
    # per demo: next-count = sum_d max(0, n-d); demo A(n=5)=4+3+1=8, demo B(n=6)=5+4+2=11,
    # demo C(n=4)=3+2=5 -> 24 next edges, mirrored by 24 prev edges = 48 total.
    assert len(expected) == 48

    # spot checks by hand
    assert (0, 4, schema.REL_INDEX["next4"]) in got     # demo A: 5 nodes, next4 spans the whole demo
    assert (4, 0, schema.REL_INDEX["prev4"]) in got
    assert (2, 3, schema.REL_INDEX["next1"]) in got      # demo A next1: (0,1),(1,2),(2,3),(3,4)
    assert (2, 4, schema.REL_INDEX["next2"]) in got      # demo A next2: (0,2),(1,3),(2,4)
    assert (10, 6, schema.REL_INDEX["prev4"]) in got     # demo B: node10 - 4 = node6
    # no dilation reaches across the owner=0/owner=1 boundary (node 4 -> node 5)
    assert not any(s == 4 and d == 5 for s, d, _ in got)


def test_temporal_edges_dilation8_16_empty_on_short_demos():
    """Every toy demo has <= 6 nodes, so next8/next16/prev8/prev16 must be entirely empty — this is
    exactly the 'dilation is load-bearing' warning: a demo shorter than the dilation has no such hop."""
    nodes = _toy_nodes()
    _, _, rel = temporal_edges(nodes)
    for d in (8, 16):
        assert not np.any(rel == schema.REL_INDEX[f"next{d}"])
        assert not np.any(rel == schema.REL_INDEX[f"prev{d}"])


# ======================================================================================================
# sibling_edges
# ======================================================================================================
def test_sibling_edges_exact():
    nodes = _toy_nodes()
    src, dst, rel = sibling_edges(nodes, k=LARGE_K)
    got = _edge_set(src, dst, rel)

    owner, task_id, grip = nodes.owner, nodes.task_id, nodes.grip
    sib = schema.REL_INDEX["sibling"]
    expected = {
        (i, j, sib)
        for i in range(len(nodes))
        for j in range(len(nodes))
        if i != j and owner[i] != owner[j] and task_id[i] == task_id[j] and grip[i] == grip[j]
    }
    assert got == expected
    assert len(expected) == 30   # {0,1,2}x{5,6,7} bipartite (18) + {3,4}x{8,9,10} bipartite (12)

    # task 1 (demo C) is the ONLY demo of its task -> zero sibling edges for nodes 11-14
    assert not any(11 <= s <= 14 or 11 <= d <= 14 for s, d, _ in got)


def test_sibling_never_crosses_owner_task_or_gripper():
    nodes = _toy_nodes()
    src, dst, rel = sibling_edges(nodes, k=schema.K_SIBLING)
    assert len(src) > 0
    assert np.all(nodes.owner[src] != nodes.owner[dst])
    assert np.all(nodes.task_id[src] == nodes.task_id[dst])
    assert np.all(nodes.grip[src] == nodes.grip[dst])
    assert np.all(rel == schema.REL_INDEX["sibling"])


# ======================================================================================================
# align_edges
# ======================================================================================================
def test_align_edges_exact():
    nodes = _toy_nodes()
    src, dst, rel = align_edges(nodes, k=LARGE_K, nbins=schema.NBINS_ALIGN)
    got = _edge_set(src, dst, rel)

    align = schema.REL_INDEX["align"]

    def pbin(p: float) -> int:
        return min(int(p * schema.NBINS_ALIGN), schema.NBINS_ALIGN - 1)

    task_id, grip, phase = nodes.task_id, nodes.grip, nodes.phase
    bins = np.array([pbin(p) for p in phase])
    expected = {
        (i, j, align)
        for i in range(len(nodes))
        for j in range(len(nodes))
        if i != j and task_id[i] != task_id[j] and grip[i] == grip[j] and bins[i] == bins[j]
    }
    assert got == expected
    # hand-derived: node11(bin0,grip0) <-> node0,node5 ; node14(bin19,grip1) <-> node4,node10
    assert expected == {
        (11, 0, align), (0, 11, align),
        (11, 5, align), (5, 11, align),
        (14, 4, align), (4, 14, align),
        (14, 10, align), (10, 14, align),
    }
    # node12 (bin6, grip0) and node13 (bin13, grip1) have NO same-bin cross-task partner
    assert not any(s in (12, 13) or d in (12, 13) for s, d, _ in got)


def test_align_never_crosses_task_gripper_or_phase_bin():
    nodes = _toy_nodes()
    src, dst, rel = align_edges(nodes, k=schema.K_ALIGN)
    assert len(src) > 0
    assert np.all(nodes.task_id[src] != nodes.task_id[dst])
    assert np.all(nodes.grip[src] == nodes.grip[dst])
    b_s = np.clip((nodes.phase[src] * schema.NBINS_ALIGN).astype(np.int64), 0, schema.NBINS_ALIGN - 1)
    b_d = np.clip((nodes.phase[dst] * schema.NBINS_ALIGN).astype(np.int64), 0, schema.NBINS_ALIGN - 1)
    assert np.all(b_s == b_d)
    assert np.all(rel == schema.REL_INDEX["align"])


def test_align_never_within_task():
    """A single-task graph has no legal align partner for anyone -- see _single_task_nodes,
    the new home of the empty-relation end-to-end test."""
    nodes = _single_task_nodes()
    src, dst, rel = align_edges(nodes, k=schema.K_ALIGN)
    assert len(src) == len(dst) == len(rel) == 0


# ======================================================================================================
# EdgeSet.build — weights, structure, empty relation end-to-end, determinism
# ======================================================================================================
def test_edgeset_build_weights():
    nodes = _toy_nodes()
    es = EdgeSet.build(nodes)
    kernel = (es.rel == schema.REL_INDEX["sibling"]) | (es.rel == schema.REL_INDEX["align"])
    assert np.all(es.w[~kernel] == 1.0)
    assert np.all((es.w[kernel] > 0) & (es.w[kernel] <= 1.0))
    assert not np.any(np.isnan(es.w))


def test_edgeset_build_full_toy_counts():
    nodes = _toy_nodes()
    es = EdgeSet.build(nodes)
    counts = es.counts()
    assert counts["sibling"] == 30
    assert counts["align"] == 8
    assert sum(counts.values()) == len(es.src)
    es.validate(nodes)   # must not raise


def test_edgeset_empty_relation_builds_validates_saves_loads(tmp_path):
    nodes = _single_task_nodes()
    es = EdgeSet.build(nodes)
    assert es.counts()["align"] == 0
    es.validate(nodes)

    path = es.save(tmp_path / "g_edges.npz")
    loaded = EdgeSet.load(path, strict=True)
    loaded.validate(nodes)
    assert loaded.counts()["align"] == 0


def test_edgeset_build_deterministic():
    nodes = _toy_nodes()
    a = EdgeSet.build(nodes)
    b = EdgeSet.build(nodes)
    assert np.array_equal(a.src, b.src)
    assert np.array_equal(a.dst, b.dst)
    assert np.array_equal(a.rel, b.rel)
    assert np.array_equal(a.w, b.w)
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.log_idf, b.log_idf)


# ======================================================================================================
# EdgeSet.torch() — the relation-grouped derived view (the GNN backward perf fix; see gnn.py)
# ======================================================================================================
def test_torch_rel_offsets_is_csr_over_fixed_relation_vocabulary():
    """rel_offsets has length N_RELATIONS+1 regardless of which relations this particular
    graph happens to use, is non-decreasing, starts at 0 and ends at E -- the CSR shape contract that
    lets a relation with zero edges be a legitimately empty (lo == hi) block."""
    nodes = _toy_nodes()
    es = EdgeSet.build(nodes)
    et = es.torch(device="cpu")
    offs = et["rel_offsets"]
    assert offs.shape == (schema.N_RELATIONS + 1,)
    assert int(offs[0]) == 0
    assert int(offs[-1]) == len(es.src)
    assert bool((offs[1:] >= offs[:-1]).all())


def test_torch_rel_offsets_empty_relation_is_empty_block():
    """The single-task graph's align relation has zero edges (see _single_task_nodes) --
    its CSR slice must be an empty lo == hi block, not an error and not a nonzero range into the
    wrong relation's edges."""
    nodes = _single_task_nodes()
    es = EdgeSet.build(nodes)
    assert es.counts()["align"] == 0
    et = es.torch(device="cpu")
    offs = et["rel_offsets"]
    r = schema.REL_INDEX["align"]
    assert int(offs[r]) == int(offs[r + 1])


def test_torch_grouped_view_is_a_permutation_of_the_same_edges():
    """{src,dst,w}_by_rel must be exactly the same multiset of (src, dst, w) triples as
    {src,dst,w}, just reordered -- and every edge inside block r must actually carry relation
    r (catches an off-by-one in the rel_offsets/permutation construction that silently
    misattributes edges to the wrong relation's gate)."""
    nodes = _toy_nodes()
    es = EdgeSet.build(nodes)
    et = es.torch(device="cpu")
    src, dst, w = et["src"], et["dst"], et["w"]
    src_g, dst_g, w_g, offs = et["src_by_rel"], et["dst_by_rel"], et["w_by_rel"], et["rel_offsets"]

    def _triples(s, d, ww):
        return sorted(zip(s.tolist(), d.tolist(), ww.tolist()))

    assert _triples(src, dst, w) == _triples(src_g, dst_g, w_g)

    rel_np = es.rel.astype(np.int64)
    for r in range(schema.N_RELATIONS):
        lo, hi = int(offs[r]), int(offs[r + 1])
        if hi > lo:
            block_src, block_dst = src_g[lo:hi], dst_g[lo:hi]
            # every (src,dst) pair in this block must be an actual relation-r edge in the original set
            block_pairs = set(zip(block_src.tolist(), block_dst.tolist()))
            true_pairs = set(zip(es.src[rel_np == r].tolist(), es.dst[rel_np == r].tolist()))
            assert block_pairs == true_pairs, r


# ======================================================================================================
# stage independence — THE point of this change: edge construction must not depend on the fragile
# stage label at all (see schema.py's note under RELATIONS for the measurement this encodes).
# ======================================================================================================
def _corrupt_stage_shuffled(nodes: _ToyNodeTable) -> _ToyNodeTable:
    """Same multiset of stage values, scrambled across nodes (breaks the "stage is monotone within a
    demo" structure entirely, not just one label)."""
    rng = np.random.RandomState(99)
    shuffled = nodes.stage.copy()
    rng.shuffle(shuffled)
    return dataclasses.replace(nodes, stage=shuffled.astype(nodes.stage.dtype))


def _corrupt_stage_spurious_transition(nodes: _ToyNodeTable) -> _ToyNodeTable:
    """One extra mid-episode stage bump inserted into demo A (node 1, which is nowhere near the real
    demo-A release at node 2..3) -- the single-misread-flicker failure mode described in schema.py."""
    bumped = nodes.stage.copy()
    bumped[1] = bumped[1] + 1
    return dataclasses.replace(nodes, stage=bumped.astype(nodes.stage.dtype))


@pytest.mark.parametrize(
    "corrupt_fn", [_corrupt_stage_shuffled, _corrupt_stage_spurious_transition],
    ids=["shuffled", "spurious_transition"],
)
def test_edges_ignore_corrupted_stage(corrupt_fn):
    """A graph built from a table whose stage field has been corrupted (shuffled, or with a
    spurious mid-episode transition inserted) must produce a BYTE-IDENTICAL EdgeSet (src/dst/rel and,
    since neither depends on stage either, w/indptr/log_idf too) to one built from the clean table --
    the machine-checkable statement that edge construction no longer depends on the fragile,
    cumsum-amplified stage label at all."""
    clean = _toy_nodes()
    corrupted = corrupt_fn(clean)
    assert not np.array_equal(clean.stage, corrupted.stage), "corruption fixture didn't corrupt anything"
    # nothing else about the table changed
    for field in ("q", "qdot", "grip", "t_idx", "owner", "task_id", "phase"):
        assert np.array_equal(getattr(clean, field), getattr(corrupted, field)), field

    es_clean = EdgeSet.build(clean)
    es_corrupt = EdgeSet.build(corrupted)

    assert np.array_equal(es_clean.src, es_corrupt.src)
    assert np.array_equal(es_clean.dst, es_corrupt.dst)
    assert np.array_equal(es_clean.rel, es_corrupt.rel)
    assert np.array_equal(es_clean.w, es_corrupt.w)
    assert np.array_equal(es_clean.indptr, es_corrupt.indptr)
    assert np.array_equal(es_clean.log_idf, es_corrupt.log_idf)


# ======================================================================================================
# save / load round-trip
# ======================================================================================================
def test_save_load_roundtrip_exact(tmp_path):
    nodes = _toy_nodes()
    es = EdgeSet.build(nodes)
    path = es.save(tmp_path / "g_edges.npz")
    loaded = EdgeSet.load(path)

    for name in ("src", "dst", "rel", "w", "indptr", "log_idf"):
        a, b = getattr(es, name), getattr(loaded, name)
        assert a.dtype == b.dtype, name
        assert np.array_equal(a, b), name
    assert es.n_nodes == loaded.n_nodes


def test_counts_sum_to_num_edges():
    nodes = _toy_nodes()
    es = EdgeSet.build(nodes)
    assert sum(es.counts().values()) == len(es.src)


# ======================================================================================================
# node_idf
# ======================================================================================================
def test_node_idf_larger_for_shared_node():
    nodes = _toy_nodes()
    es = EdgeSet.build(nodes)
    # node 0 (task0/grip0) is sibling-connected to 3 strands' worth of nodes (owner1's 5,6,7);
    # node 11 (task1) is the sole demo of its task and has ZERO sibling edges -> zero reach.
    assert es.log_idf[0] > es.log_idf[11]
    assert es.log_idf[11] == pytest.approx(0.0)


def test_node_idf_direct_call_matches_edgeset():
    nodes = _toy_nodes()
    es = EdgeSet.build(nodes)
    idf = node_idf(nodes, RelationEdges(
        src=es.src.astype(np.int64), dst=es.dst.astype(np.int64), rel=es.rel.astype(np.int64)
    ))
    np.testing.assert_allclose(idf, es.log_idf, atol=1e-6)


# ======================================================================================================
# validate() — structural + semantic corruption catches
# ======================================================================================================
def _fresh_edgeset(nodes):
    return EdgeSet.build(nodes)


def test_validate_raises_on_sibling_task_mismatch():
    nodes = _toy_nodes()
    es = _fresh_edgeset(nodes)
    i = int(np.nonzero(es.rel == schema.REL_INDEX["sibling"])[0][0])
    bad_dst = es.dst.copy()
    # node 12 is task1; the sibling edge at i connects two task0 nodes -> now mismatched
    bad_dst[i] = 12
    corrupt = dataclasses.replace(es, dst=bad_dst)
    with pytest.raises(ValueError, match="sibling"):
        corrupt.validate(nodes)


def test_validate_raises_on_flipped_gripper():
    nodes = _toy_nodes()
    es = _fresh_edgeset(nodes)
    i = int(np.nonzero(es.rel == schema.REL_INDEX["sibling"])[0][0])
    d_node = int(es.dst[i])
    bad_grip = nodes.grip.copy()
    bad_grip[d_node] = 1 - bad_grip[d_node]
    bad_nodes = dataclasses.replace(nodes, grip=bad_grip)
    with pytest.raises(ValueError, match="sibling"):
        es.validate(bad_nodes)


def test_validate_raises_on_align_task_match():
    nodes = _toy_nodes()
    es = _fresh_edgeset(nodes)
    i = int(np.nonzero(es.rel == schema.REL_INDEX["align"])[0][0])
    src_node = int(es.src[i])
    # retarget dst to another node that shares src's task (an align edge must be CROSS-task)
    same_task = np.nonzero(nodes.task_id == nodes.task_id[src_node])[0]
    same_task = same_task[same_task != src_node][0]
    bad_dst = es.dst.copy()
    bad_dst[i] = same_task
    corrupt = dataclasses.replace(es, dst=bad_dst)
    with pytest.raises(ValueError, match="align"):
        corrupt.validate(nodes)


def test_validate_raises_on_unsorted_dst():
    nodes = _toy_nodes()
    es = _fresh_edgeset(nodes)
    src, dst = es.src, es.dst
    n = len(nodes)
    swapped = None
    for i in range(len(dst)):
        for j in range(len(dst) - 1, i, -1):
            if dst[i] != dst[j] and src[i] != dst[j] and src[j] != dst[i]:
                bad = dst.copy()
                bad[i], bad[j] = bad[j], bad[i]
                swapped = bad
                break
        if swapped is not None:
            break
    assert swapped is not None, "could not find a safe (non-self-loop) dst swap in the toy graph"
    corrupt = dataclasses.replace(es, dst=swapped)
    with pytest.raises(ValueError, match="sorted"):
        corrupt.validate()
    assert n > 0   # (n unused beyond the loop bound above; keeps flake happy about intent)


def test_validate_raises_on_bad_indptr_total():
    nodes = _toy_nodes()
    es = _fresh_edgeset(nodes)
    bad_indptr = es.indptr.copy()
    bad_indptr[-1] += 1
    corrupt = dataclasses.replace(es, indptr=bad_indptr)
    with pytest.raises(ValueError, match="indptr"):
        corrupt.validate()


def test_validate_raises_on_self_loop():
    nodes = _toy_nodes()
    es = _fresh_edgeset(nodes)
    bad_src = es.src.copy()
    bad_src[0] = es.dst[0]
    corrupt = dataclasses.replace(es, src=bad_src)
    with pytest.raises(ValueError, match="self-loop"):
        corrupt.validate()


def test_validate_raises_on_out_of_range_node_id():
    nodes = _toy_nodes()
    es = _fresh_edgeset(nodes)
    bad_dst = es.dst.copy()
    bad_dst[0] = es.n_nodes + 5
    corrupt = dataclasses.replace(es, dst=bad_dst)
    with pytest.raises(ValueError, match="out of range"):
        corrupt.validate()
