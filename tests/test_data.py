"""Tests for FCSR's C6 data-engine additions: HardNegativeMiner and the SEGMENT query class.

Small, fully synthetic fixtures throughout (mirrors tests/test_train.py's own style), since neither
piece needs a real HDF5 corpus or C1's SegmentBuilder -- robot_ood.core.schemas.SegmentTable is a plain
frozen dataclass and these tests build it by hand, with the geometry chosen so the expected mined
negatives are unambiguous rather than dependent on tie-breaking.
"""

from __future__ import annotations


import numpy as np
import pytest

from robot_ood.core.schemas import SegmentTable
from robot_ood.data.nodes import NodeTable
from robot_ood.train.data import make_queries
from robot_ood.train.negatives import HardNegativeMiner
from robot_ood.train.types import QuerySpec

D = 4
SEGMENT_LEN = 16
ACTION_DIM = 7


# ======================================================================================================
# HardNegativeMiner fixtures and tests
# ======================================================================================================
_N_FILLER = 70    # >= HardNegativeMiner's _STATE_POOL(64), so the phase-band widening loop's first
                   # stop condition (pool already full) fires without ever reaching segment 4 --
                   # what test_same_state_different_future_respects_the_phase_band depends on.


def _mining_table() -> SegmentTable:
    """Six segments of interest, plus _N_FILLER same-phase/same-task/identical-future padding
    segments so the table is realistically dense. Geometry of the first six is chosen so each
    bucket's answer is exactly one segment:

        0  anchor: phase=0.5 task=0, action chunk all-zero, ee_start at the origin.
        1  same phase, same task, ee_start at the origin (state-close), action chunk far from 0's
           (DTW clears any sane threshold) -- the ONLY same_state_different_future hit.
        2  same phase, same task, ee_start at the origin, action chunk IDENTICAL to 0's (DTW == 0)
           -- a same-state, SAME-future decoy: state-close but must NOT be mined.
        3  same phase, DIFFERENT task, action chunk identical to 0's -- the ONLY
           same_phase_wrong_task hit (its DTW is irrelevant to that bucket).
        4  phase far outside the band (0.9 vs 0.5) -- excluded from both buckets regardless of
           task or action chunk.
        5  same phase, same task, ee_start at the origin, action chunk identical to 0's -- a
           second same-state decoy.

    The _N_FILLER padding segments sit at the SAME phase as the anchor but far away in ee_start
    (state-DISTANT) with an action chunk identical to the anchor's (DTW == 0), so they are never
    themselves mined -- their only job is bulking out the in-band pool.
    """
    s = 6 + _N_FILLER
    phase_start = np.concatenate([[0.5, 0.5, 0.5, 0.5, 0.9, 0.5], np.full(_N_FILLER, 0.5)])
    task_id = np.concatenate([[0, 0, 0, 1, 2, 0], np.zeros(_N_FILLER, dtype=np.int64)]).astype(np.int64)
    owner = np.arange(s, dtype=np.int64)
    chunk = np.zeros((s, SEGMENT_LEN, ACTION_DIM), dtype=np.float32)
    chunk[1] = 5.0                                            # the one genuinely divergent future
    ee_start = np.zeros((s, 6), dtype=np.float32)
    ee_start[6:] = 100.0                                      # fillers are state-DISTANT
    return SegmentTable(
        start_node=np.arange(s, dtype=np.int32), end_node=np.arange(s, dtype=np.int32), owner=owner,
        task_id=task_id, phase_start=phase_start, phase_end=phase_start.copy(), action_chunk=chunk,
        ee_start=ee_start, ee_end=np.zeros((s, 6), dtype=np.float32),node_to_segment=np.arange(s, dtype=np.int32),
    )


@pytest.fixture()
def miner() -> HardNegativeMiner:
    return HardNegativeMiner(_mining_table(), dtw_thresh=1.0, phase_band=0.3)


def test_same_state_different_future_finds_only_the_divergent_chunk(miner):
    assert set(miner.same_state_different_future(0, n=5).tolist()) == {1}


def test_same_state_different_future_excludes_the_anchor_and_identical_futures(miner):
    negs = miner.same_state_different_future(0, n=5)
    assert 0 not in negs
    assert 2 not in negs and 5 not in negs      # identical-chunk decoys, DTW == 0 < dtw_thresh


def test_same_state_different_future_respects_the_phase_band(miner):
    """Segment 4 is a phase 0.9 outlier with a divergent-looking setup only by coincidence (it is
    zero, same as the anchor) -- but the point of this test is that a segment OUTSIDE the band never
    even enters the pool, so widening dtw_thresh to 0 (which would otherwise flag every non-anchor
    row) still must not surface it."""
    zero_thresh = HardNegativeMiner(_mining_table(), dtw_thresh=-1.0, phase_band=0.3)
    assert 4 not in zero_thresh.same_state_different_future(0, n=5)


def test_same_phase_wrong_task_finds_only_the_other_task_segment(miner):
    assert set(miner.same_phase_wrong_task(0, n=5).tolist()) == {3}


def test_same_phase_wrong_task_respects_the_phase_band(miner):
    """Segment 4's task (2) differs from the anchor's (0), but it sits outside the phase band."""
    assert 4 not in miner.same_phase_wrong_task(0, n=5)


def test_negatives_unions_both_buckets_and_excludes_the_anchor(miner):
    negs = miner.negatives(0, n=5)
    assert set(negs.tolist()) == {1, 3}
    assert 0 not in negs
    assert len(negs) == len(set(negs.tolist()))         # unique


def test_negatives_caps_at_n(miner):
    assert len(miner.negatives(0, n=1)) <= 1


def test_negatives_deterministic_across_calls(miner):
    """No RNG anywhere in the miner -- see the module docstring."""
    assert miner.negatives(0, n=5).tolist() == miner.negatives(0, n=5).tolist()


def test_negatives_zero_n_is_empty(miner):
    assert miner.negatives(0, n=0).shape == (0,)


def test_hard_negatives_dtw_cost_agrees_with_align_dtw_cost():
    """hard_negatives._dtw_cost must be the SAME metric robot_ood.build.measure.SegmentHardNegatives fits
    seg_hard_neg_thresh against (robot_ood.train.align.dtw_cost's mean-per-step, translation-only
    distance), not a second DP implementation with its own normalisation.

    Before the fix, hard_negatives._dtw_cost returned the raw cumulative DP cost over all 7 action
    columns (translation + rotation + gripper), unnormalised by path length -- roughly path_length
    (SEGMENT_LEN=16) times larger than align.dtw_cost's mean warped distance on the same pair, and
    on the wrong units besides. This test pins them to agree exactly (both computed from the same
    torch DP), which the old raw-sum implementation could not have passed.
    """
    import torch

    from robot_ood.train.align import dtw_cost
    from robot_ood.train.negatives import _dtw_cost

    rng = np.random.RandomState(0)
    anchor = rng.normal(size=(SEGMENT_LEN, ACTION_DIM)).astype(np.float64)
    candidates = rng.normal(size=(5, SEGMENT_LEN, ACTION_DIM)).astype(np.float64)

    got = _dtw_cost(anchor, candidates)

    anchor_t = torch.as_tensor(
        np.broadcast_to(anchor[None, :, :3], (5, SEGMENT_LEN, 3)), dtype=torch.float64
    )
    cand_t = torch.as_tensor(candidates[:, :, :3], dtype=torch.float64)
    want = dtw_cost(anchor_t, cand_t).numpy()

    np.testing.assert_allclose(got, want, rtol=1e-10)
    # Sanity: for non-trivial paths the mean-per-step distance is well below SEGMENT_LEN times the
    # raw cumulative cost the old buggy implementation returned, i.e. this is really the normalised
    # metric and not accidentally still the raw one.
    assert np.all(got < SEGMENT_LEN)


def _entry_mining_table() -> SegmentTable:
    """Five segments for entry_same_state_different_task, entry_band=0.05:

        0  anchor: phase=0.02 (entry), task=0, ee_start at the origin.
        1  entry (phase=0.01), DIFFERENT task, ee_start at the origin -- nearest hit.
        2  entry (phase=0.03), DIFFERENT task, ee_start far away -- second-nearest hit.
        3  entry (phase=0.04), SAME task as the anchor -- excluded by task, however close in state.
        4  OUTSIDE the entry band (phase=0.5), DIFFERENT task, ee_start at the origin -- excluded by
           phase alone.
    """
    s = 5
    phase_start = np.array([0.02, 0.01, 0.03, 0.04, 0.5], dtype=np.float32)
    task_id = np.array([0, 1, 1, 0, 1], dtype=np.int64)
    ee_start = np.zeros((s, 6), dtype=np.float32)
    ee_start[2] = 10.0
    return SegmentTable(
        start_node=np.arange(s, dtype=np.int32), end_node=np.arange(s, dtype=np.int32),
        owner=np.arange(s, dtype=np.int64), task_id=task_id, phase_start=phase_start,
        phase_end=phase_start.copy(), action_chunk=np.zeros((s, SEGMENT_LEN, ACTION_DIM), dtype=np.float32),
        ee_start=ee_start, ee_end=np.zeros((s, 6), dtype=np.float32),node_to_segment=np.arange(s, dtype=np.int32),
    )


@pytest.fixture()
def entry_miner() -> HardNegativeMiner:
    return HardNegativeMiner(_entry_mining_table(), dtw_thresh=1.0, phase_band=0.3, entry_band=0.05)


def test_entry_same_state_different_task_orders_by_ee_start_distance(entry_miner):
    assert entry_miner.entry_same_state_different_task(0, n=5).tolist() == [1, 2]


def test_entry_same_state_different_task_excludes_same_task(entry_miner):
    assert 3 not in entry_miner.entry_same_state_different_task(0, n=5)


def test_entry_same_state_different_task_respects_the_entry_band(entry_miner):
    assert 4 not in entry_miner.entry_same_state_different_task(0, n=5)


def test_entry_same_state_different_task_mines_nothing_for_a_mid_demo_anchor(entry_miner):
    """Segment 4 sits outside the entry band -- as an ANCHOR it has nothing to learn from this
    bucket, which exists only for episode-start queries."""
    assert entry_miner.entry_same_state_different_task(4, n=5).shape == (0,)


def test_negatives_includes_the_entry_bucket_only_for_an_entry_region_anchor(entry_miner):
    entry_negs = entry_miner.negatives(0, n=5)
    assert 1 in entry_negs or 2 in entry_negs
    mid_negs = entry_miner.negatives(4, n=5)
    assert set(mid_negs.tolist()) <= {0, 1, 2, 3}   # never mines from entry_same_state_different_task


def test_negatives_unaffected_for_non_entry_anchors(miner):
    """miner's anchor (segment 0) sits at phase=0.5, well outside the default entry_band=0.05, so
    adding entry_same_state_different_task must not change this pre-existing result."""
    assert set(miner.negatives(0, n=5).tolist()) == {1, 3}


def test_mine_batch_pads_with_minus_one_and_skips_negative_anchors(miner):
    out = miner.mine_batch(np.array([0, -1, 0]), n=3)
    assert out.shape == (3, 3)
    assert np.all(out[1] == -1)                     # anchor -1 mines nothing
    assert set(out[0][out[0] >= 0].tolist()) == {1, 3}
    assert out[0].tolist() == out[2].tolist()        # same anchor, same result


def test_single_segment_table_mines_nothing():
    table = _mining_table()
    trivial = SegmentTable(
        start_node=table.start_node[:1], end_node=table.end_node[:1], owner=table.owner[:1],
        task_id=table.task_id[:1], phase_start=table.phase_start[:1], phase_end=table.phase_end[:1],
        action_chunk=table.action_chunk[:1], ee_start=table.ee_start[:1], ee_end=table.ee_end[:1],
        node_to_segment=table.node_to_segment[:1],
    )
    m = HardNegativeMiner(trivial, dtw_thresh=1.0, phase_band=0.3)
    assert m.negatives(0, n=5).shape == (0,)


# ======================================================================================================
# SEGMENT query class
# ======================================================================================================
def _make_demo(task_id: int, seed: int, n: int, dim: int = D, base: np.ndarray | None = None) -> dict:
    rng = np.random.RandomState(seed)
    start = (base if base is not None else np.zeros(dim)) + rng.randn(dim) * 0.01
    direction = rng.randn(dim)
    direction /= np.linalg.norm(direction)
    t = np.arange(n, dtype=np.float32)
    q = (start[None, :] + 0.05 * t[:, None] * direction[None, :]).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    qdot[-1] = qdot[-2] if n > 1 else 0.0
    grip = np.zeros(n, dtype=np.uint8)
    stage = np.zeros(n, dtype=np.int16)
    return {"q": q, "qdot": qdot, "grip": grip, "stage": stage, "task_id": task_id}


def _toy_nodes(n_tasks: int = 2, demos_per_task: int = 4, length: int = 20, seed: int = 0) -> NodeTable:
    rng = np.random.RandomState(seed)
    demos = []
    for ti in range(n_tasks):
        base = rng.randn(D) * 2.0
        for di in range(demos_per_task):
            demos.append(_make_demo(ti, seed=1000 * ti + di + seed, n=length, dim=D, base=base))
    return NodeTable.from_demos(demos, task_names=[f"task{i}" for i in range(n_tasks)], coarsen=1)


def _segment_table_from_nodes(nodes: NodeTable) -> SegmentTable:
    """One segment per node, so a SEGMENT row drawn against any node is always legal."""
    v = len(nodes)
    return SegmentTable(
        start_node=np.arange(v, dtype=np.int32), end_node=np.arange(v, dtype=np.int32),
        owner=np.asarray(nodes.owner, dtype=np.int32), task_id=np.asarray(nodes.task_id, dtype=np.int16),
        phase_start=np.asarray(nodes.phase, dtype=np.float32),
        phase_end=np.asarray(nodes.phase, dtype=np.float32),
        action_chunk=np.zeros((v, SEGMENT_LEN, ACTION_DIM), dtype=np.float32),
        ee_start=np.zeros((v, 6), dtype=np.float32), ee_end=np.zeros((v, 6), dtype=np.float32),node_to_segment=np.arange(v, dtype=np.int32),
    )


def test_segment_frac_zero_is_a_strict_noop_on_seg_id():
    nodes = _toy_nodes()
    rng = np.random.RandomState(0)
    out = make_queries(nodes, rng, 64, np.arange(nodes.n_demos), QuerySpec())
    assert "seg_id" in out and out["seg_id"].shape == (64,)
    assert np.all(out["seg_id"] == -1)


def test_segment_class_targets_its_own_segments_start_node():
    nodes = _toy_nodes()
    table = _segment_table_from_nodes(nodes)
    rng = np.random.RandomState(0)
    out = make_queries(
        nodes, rng, 64, np.arange(nodes.n_demos),
        QuerySpec(segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0),
    )
    assert np.all(out["seg_id"] >= 0)
    assert np.array_equal(out["tgt_node"], table.start_node[out["seg_id"]].astype(np.int64))
    assert np.array_equal(out["src_node"], out["tgt_node"])


def test_segment_class_window_is_a_single_perturbed_frame_repeated():
    """Same shape as ENTRY_STATIC: identically zero finite-differenced velocity."""
    nodes = _toy_nodes()
    table = _segment_table_from_nodes(nodes)
    rng = np.random.RandomState(0)
    out = make_queries(
        nodes, rng, 32, np.arange(nodes.n_demos),
        QuerySpec(segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0),
    )
    assert np.allclose(out["qdot_hist"], 0.0)
    for row in out["q_hist"]:
        assert np.allclose(row, row[0])


def test_segment_class_perturbation_is_smaller_than_entrys_own():
    """SEGMENT_SIGMA_FRAC keeps the row inside the segment's own basin."""
    nodes = _toy_nodes()
    table = _segment_table_from_nodes(nodes)
    owners = np.arange(nodes.n_demos)
    seg_radii = make_queries(
        nodes, np.random.RandomState(1), 200, owners,
        QuerySpec(segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0),
    )["perturb_radius"]
    entry_radii = make_queries(
        nodes, np.random.RandomState(1), 200, owners,
        QuerySpec(segments=None, segment_frac=0.0, entry_frac=1.0, entry_static_frac=0.0, drift_frac=0.0),
    )["perturb_radius"]
    assert seg_radii.mean() < entry_radii.mean()


def test_segment_class_restricted_to_owners_in_the_pool():
    nodes = _toy_nodes(n_tasks=2, demos_per_task=4)
    table = _segment_table_from_nodes(nodes)
    train_owners = np.array([0, 1])                # excludes every other demo's segments
    rng = np.random.RandomState(0)
    out = make_queries(
        nodes, rng, 64, train_owners,
        QuerySpec(segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0),
    )
    assert np.all(np.isin(out["src_owner"], train_owners))


def test_segment_entry_frac_zero_is_a_strict_noop_on_the_rng_stream():
    """0.0 (the literal default) must reproduce the exact stream segment_entry_frac never existed
    for -- the same short-circuit-at-zero convention entry_static_frac/drift_frac already follow."""
    nodes = _toy_nodes()
    table = _segment_table_from_nodes(nodes)
    owners = np.arange(nodes.n_demos)
    spec_kwargs = dict(segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0)
    baseline = make_queries(nodes, np.random.RandomState(0), 64, owners, QuerySpec(**spec_kwargs))
    zero = make_queries(
        nodes, np.random.RandomState(0), 64, owners, QuerySpec(segment_entry_frac=0.0, **spec_kwargs),
    )
    assert np.array_equal(baseline["seg_id"], zero["seg_id"])


def test_segment_entry_frac_shifts_the_phase_histogram_toward_zero():
    """The measured, not asserted, shift make_queries' docstring promises: report a strata histogram
    before (uniform) and after (segment_entry_frac=1.0) so the curriculum change is a number."""
    nodes = _toy_nodes(n_tasks=2, demos_per_task=4, length=40)
    table = _segment_table_from_nodes(nodes)
    owners = np.arange(nodes.n_demos)
    band = 0.05

    before = make_queries(
        nodes, np.random.RandomState(0), 2000, owners,
        QuerySpec(segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0),
    )
    after = make_queries(
        nodes, np.random.RandomState(0), 2000, owners,
        QuerySpec(
            segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0,
            segment_entry_frac=1.0, segment_entry_band=band,
        ),
    )
    phase = np.asarray(table.phase_start)
    before_share = float(np.mean(phase[before["seg_id"]] <= band))
    after_share = float(np.mean(phase[after["seg_id"]] <= band))
    print(f"entry stratum (phase<={band}) share: before={before_share:.3f} after={after_share:.3f}")
    assert after_share > before_share
    assert after_share == pytest.approx(1.0, abs=1e-6)


def test_segment_entry_frac_falls_back_to_the_full_pool_when_the_stratum_is_empty():
    nodes = _toy_nodes()
    table = _segment_table_from_nodes(nodes)
    # No node has phase this low, so the entry stratum is empty for every owner.
    with pytest.warns(UserWarning, match="segment_entry_frac"):
        out = make_queries(
            nodes, np.random.RandomState(0), 32, np.arange(nodes.n_demos),
            QuerySpec(
                segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0,
                segment_entry_frac=1.0, segment_entry_band=-1.0,
            ),
        )
    assert np.all(out["seg_id"] >= 0)


def test_a_hist_reads_the_demos_own_recorded_actions_edge_padded():
    """Gap A, training side: a_hist must come from nodes.a_raw (the demo's OWN recorded actions),
    sliced per row's own (owner, t_raw_now) window and edge-padded exactly like q_hist -- not left
    None/zero, which is what made the fcsr_noact ablation indistinguishable from fcsr."""
    length = 20
    demo0 = _make_demo(0, seed=0, n=length)
    demo0["a"] = np.arange(length * ACTION_DIM, dtype=np.float32).reshape(length, ACTION_DIM)
    demo1 = _make_demo(0, seed=1, n=length)
    demo1["a"] = np.zeros((length, ACTION_DIM), dtype=np.float32)
    nodes = NodeTable.from_demos([demo0, demo1], task_names=["task0"], coarsen=1)
    rng = np.random.RandomState(0)
    out = make_queries(
        nodes, rng, 16, np.array([0]),
        QuerySpec(entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0),
    )
    assert "a_hist" in out
    window = out["a_hist"].shape[1]
    assert out["a_hist"].shape == (16, window, ACTION_DIM)
    assert not np.allclose(out["a_hist"], 0.0)
    for i in range(len(out["a_hist"])):
        t = int(out["t_raw_now"][i])
        lo = max(0, t - window + 1)
        expected = demo0["a"][lo : t + 1]
        if expected.shape[0] < window:
            pad = np.repeat(expected[:1], window - expected.shape[0], axis=0)
            expected = np.concatenate([pad, expected], axis=0)
        np.testing.assert_allclose(out["a_hist"][i], expected)


def test_segment_frac_forced_to_zero_when_owner_pool_holds_no_segment():
    nodes = _toy_nodes(n_tasks=2, demos_per_task=4)
    table = _segment_table_from_nodes(nodes)
    lone_owner = np.array([0])
    # Restrict the table itself to demos NOT in lone_owner, so the intersection is empty.
    other = np.flatnonzero(table.owner != 0)
    restricted = SegmentTable(
        start_node=table.start_node[other], end_node=table.end_node[other], owner=table.owner[other],
        task_id=table.task_id[other], phase_start=table.phase_start[other],
        phase_end=table.phase_end[other], action_chunk=table.action_chunk[other],
        ee_start=table.ee_start[other], ee_end=table.ee_end[other],
        node_to_segment=table.node_to_segment,
    )
    rng = np.random.RandomState(0)
    with pytest.warns(UserWarning, match="segment_frac"):
        out = make_queries(
            nodes, rng, 32, lone_owner,
            QuerySpec(segments=restricted, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0,
                      drift_frac=0.0),
        )
    assert np.all(out["seg_id"] == -1)
