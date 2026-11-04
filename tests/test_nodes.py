"""Tests for robot_ood.data.nodes — synthetic demos only (no HDF5), so this runs in .venv-core."""

import numpy as np
import pytest

from robot_ood.data import schema
from robot_ood.data.nodes import NodeTable, coarsen_segments, episode_phase, psi

# ==================================================================================================
# synthetic-demo helpers
# ==================================================================================================
D = 3


def _make_demo(t: int, task_id: int = 0, grip_flip: int | None = None, stage_flip: int | None = None,
                seed: int = 0) -> dict:
    """A deterministic synthetic demo: a smooth random walk in q, matching finite-difference qdot,
    a gripper that flips open->closed at grip_flip, a stage that flips at stage_flip.

    ee is concat(q, q), i.e. D=3 twice, so an ee_segment can be checked column-for-column against
    the q it was decoded alongside."""
    rng = np.random.RandomState(seed)
    steps = rng.randn(t, D).astype(np.float32) * 0.01
    q = np.cumsum(steps, axis=0).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    qdot[-1] = qdot[-2] if t > 1 else 0.0
    grip = np.zeros(t, dtype=np.uint8)
    if grip_flip is not None:
        grip[grip_flip:] = 1
    stage = np.zeros(t, dtype=np.int16)
    if stage_flip is not None:
        stage[stage_flip:] = 1
    ee = np.concatenate([q, q], axis=1).astype(np.float32)
    a = np.zeros((t, 7), dtype=np.float32)
    return {"q": q, "qdot": qdot, "grip": grip, "stage": stage, "task_id": task_id, "ee": ee, "a": a}


def _tiny_table(coarsen: int = 3) -> NodeTable:
    """Two short demos with hand-traceable coarsening, for exact segment()/qdot arithmetic checks.

    demo0 uses T=7 with coarsen=3 (7 = 2*3 + 1) so its trailing group has size 1 and its last node's
    phase lands at exactly 1.0 (t_raw=6, T-1=6) — satisfying the table-wide phase.max()>0.99 check
    without needing hundreds of raw frames.
    """
    demos = [
        _make_demo(7, task_id=0, seed=1),                 # T=7, coarsen=3 -> nodes of size 3,3,1
        _make_demo(4, task_id=1, grip_flip=2, seed=2),     # T=4, grip flips mid -> nodes 2,2
    ]
    return NodeTable.from_demos(demos, task_names=["taskA", "taskB"], coarsen=coarsen)


# ==================================================================================================
# episode_phase
# ==================================================================================================
@pytest.mark.parametrize("t", [1, 2, 3, 150, 517])
def test_episode_phase_endpoints(t):
    ph = episode_phase(np.linspace(0.0, 1.0, t, dtype=np.float32)[:, None].repeat(D, 1))
    assert ph.shape == (t,)
    assert ph[0] == 0.0
    if t > 1:
        assert ph[-1] == 1.0
    assert np.all(ph >= 0.0) and np.all(ph <= 1.0)
    assert np.all(np.diff(ph) >= 0)          # monotone non-decreasing


def test_episode_phase_does_not_advance_while_the_arm_holds_still():
    q = np.repeat(np.arange(4, dtype=np.float32)[:, None], D, 1)
    frozen = np.concatenate([q[:2], np.repeat(q[1:2], 20, 0), q[2:]])
    ph = episode_phase(frozen)
    assert ph[2] == pytest.approx(ph[22])
    assert ph[2] == pytest.approx(2.0 / 3.0)


def test_psi_deterministic():
    t_frac = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    a = psi(t_frac)
    b = psi(t_frac)
    np.testing.assert_array_equal(a, b)


def test_psi_endpoints_exact():
    # sin(0) = 0, cos(0) = 1 at t_frac=0 for every frequency
    out = psi(np.array([0.0], dtype=np.float32), n_freq=4)
    np.testing.assert_allclose(out[0, 0::2], 0.0, atol=1e-6)   # sin columns
    np.testing.assert_allclose(out[0, 1::2], 1.0, atol=1e-6)   # cos columns


# ==================================================================================================
# coarsen_segments
# ==================================================================================================
def test_coarsen_segments_no_boundary():
    grip = np.zeros(12, dtype=np.uint8)
    ids = coarsen_segments(grip, coarsen=5)
    assert list(ids) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    assert ids.dtype == np.int32


def test_coarsen_segments_never_crosses_gripper_change():
    # T=17, coarsen=5, grip flips at an "awkward" mid-group offset (7)
    t = 17
    grip = np.zeros(t, dtype=np.uint8)
    grip[7:] = 1
    ids = coarsen_segments(grip, coarsen=5)

    assert ids.shape == (t,)
    assert ids[0] == 0
    assert list(np.unique(ids)) == list(range(int(ids.max()) + 1))     # contiguous from 0

    # no node id spans the gripper boundary (index 6 -> 7)
    assert ids[6] != ids[7]

    # every node's members share one grip value
    for node_id in np.unique(ids):
        mask = ids == node_id
        assert len(np.unique(grip[mask])) == 1

    # group sizes never exceed coarsen
    counts = np.bincount(ids)
    assert np.all(counts <= 5) and np.all(counts >= 1)


def test_coarsen_segments_group_sizes_and_contiguous_ids():
    grip = np.zeros(23, dtype=np.uint8)
    ids = coarsen_segments(grip, coarsen=4)
    counts = np.bincount(ids)
    assert np.all(counts <= 4)
    assert list(np.unique(ids)) == list(range(len(counts)))


def test_coarsen_segments_ignores_stage_changes():
    """The new, looser contract: coarsening never crosses a gripper change but MAY span a stage
    change -- stage is diagnostic-only (see schema.py's note under RELATIONS) and is no longer a
    coarsening boundary. A demo whose stage flips mid-group must coarsen IDENTICALLY to the same
    demo with a constant stage, and the resulting table must contain a node whose raw span
    straddles the (ignored) stage boundary."""
    # T=11=2*5+1 so the trailing group has size 1 and phase.max() lands exactly at 1.0 -- the same
    # trick _tiny_table uses above, needed to satisfy NodeTable.validate()'s phase-convention check.
    t = 11
    demo_flip = _make_demo(t, task_id=0, stage_flip=6, seed=3)     # grip constant, stage flips at 6
    demo_flat = dict(demo_flip, stage=np.zeros(t, dtype=np.int16))

    table_flip = NodeTable.from_demos([demo_flip], task_names=["t"], coarsen=5)
    table_flat = NodeTable.from_demos([demo_flat], task_names=["t"], coarsen=5)

    np.testing.assert_array_equal(table_flip.t_raw, table_flat.t_raw)
    np.testing.assert_array_equal(table_flip.n_members, table_flat.n_members)

    # a node whose raw span covers both index 5 (stage 0) and index 6 (stage 1) proves it straddled
    # the boundary rather than being cut there
    ends = table_flip.t_raw.astype(np.int64) + table_flip.n_members.astype(np.int64)
    spans_flip = np.any((table_flip.t_raw <= 5) & (ends > 6))
    assert spans_flip, "expected a coarse node to straddle the (ignored) stage boundary at index 6"

    # meanwhile grip is constant in this demo, so the raw coarsen_segments() call agrees directly
    ids = coarsen_segments(np.asarray(demo_flip["grip"]), coarsen=5)
    assert list(ids) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]


# ==================================================================================================
# from_demos / validate / sizing
# ==================================================================================================
def test_from_demos_builds_valid_table():
    table = _tiny_table()
    table.validate()          # must not raise
    assert len(table) == table.q.shape[0]
    assert table.dim == D
    assert table.n_demos == 2
    assert table.n_tasks == 2


def test_from_demos_requires_nonempty():
    with pytest.raises(ValueError, match="non-empty"):
        NodeTable.from_demos([], task_names=[])


def test_from_demos_realistic_sizes():
    demos = [_make_demo(150, task_id=0, grip_flip=80, stage_flip=None, seed=i) for i in range(4)]
    demos += [_make_demo(517, task_id=1, grip_flip=300, seed=10 + i) for i in range(3)]
    table = NodeTable.from_demos(demos, task_names=["a", "b"], coarsen=schema.COARSEN)
    table.validate()
    assert table.n_demos == 7
    assert table.n_tasks == 2
    assert len(table) > 0


# ==================================================================================================
# save / load round-trip
# ==================================================================================================
def test_save_load_roundtrip(tmp_path):
    table = _tiny_table()
    out = table.save(tmp_path)
    assert out.name == schema.NODES_NPZ

    loaded = NodeTable.load(tmp_path, strict=True)
    loaded.validate()

    for key in schema.NODES_KEYS:
        a, b = getattr(table, key), getattr(loaded, key)
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
    assert loaded.task_names == table.task_names
    assert loaded.coarsen == table.coarsen


def test_save_load_roundtrip_explicit_file(tmp_path):
    table = _tiny_table()
    path = tmp_path / "custom_nodes.npz"
    out = table.save(path)
    assert out == path
    loaded = NodeTable.load(path)
    assert len(loaded) == len(table)


def test_load_strict_false_skips_validation(tmp_path):
    table = _tiny_table()
    table.save(tmp_path)
    loaded = NodeTable.load(tmp_path, strict=False)
    assert len(loaded) == len(table)


# ==================================================================================================
# segment()
# ==================================================================================================
def test_segment_returns_exactly_k_frames_no_padding_needed():
    table = _tiny_table(coarsen=3)
    q_seg, qdot_seg = table.segment(0, k=3)
    assert q_seg.shape == (3, D)
    assert qdot_seg.shape == (3, D)
    assert q_seg.dtype == np.float64


def test_segment_last_node_edge_pads_and_never_crosses_demo_boundary():
    table = _tiny_table(coarsen=3)
    # last node of demo 0 (owner==0)
    demo0_nodes = table.demo_nodes(0)
    last_v = int(demo0_nodes[-1])
    demo0_end = int(table.raw_ptr[1])          # exclusive raw end of demo 0

    q_seg, qdot_seg = table.segment(last_v, k=10)     # ask for way more than remains
    assert q_seg.shape == (10, D)
    assert qdot_seg.shape == (10, D)

    start = int(table.t_raw[last_v])
    n_avail = demo0_end - start
    expected_q = table.q_raw[start:demo0_end].astype(np.float64)
    expected_q = np.concatenate([expected_q, np.repeat(expected_q[-1:], 10 - n_avail, axis=0)], axis=0)
    # expected_q is built purely from demo 0's own raw frames (edge-padded with its own last frame),
    # so this equality itself proves segment() never bled into demo 1's raw frames.
    np.testing.assert_array_equal(q_seg, expected_q)


def test_ee_segment_clips_and_pads_exactly_like_segment():
    """ee_segment must share segment()'s raw_ptr clipping and edge padding, checked at the demo
    boundary where they matter: ee_base is declared here as concat(q, q), so the decoded [k, 6] must
    be the decoded [k, D] q twice over, padding included.

    Declared rather than forward-kinematicked because this table is a D=3 toy, not a Panda; the
    clipping this test is about is a property of raw_ptr and is the same either way.
    """
    table = _tiny_table(coarsen=3)
    table.ee_base = table.ee_raw
    last_v = int(table.demo_nodes(0)[-1])

    q_seg, _ = table.segment(last_v, k=10)          # spills past demo 0's end -> edge-padded
    ee_seg = table.ee_segment(last_v, k=10)

    assert ee_seg.shape == (10, 6)
    assert ee_seg.dtype == np.float64
    np.testing.assert_array_equal(ee_seg, np.concatenate([q_seg, q_seg], axis=1))

    with pytest.raises(ValueError, match="k must be"):
        table.ee_segment(last_v, k=0)


def test_load_refuses_an_artifact_missing_the_new_columns(tmp_path):
    """A pre-ee_raw g_nodes.npz must fail loudly, naming the rebuild command -- zero-filling would
    hand the FK-vs-recorded check a dead pose column with nothing reporting an error."""
    table = _tiny_table()
    path = table.save(tmp_path)
    with np.load(path) as archive:
        stale = {k: archive[k] for k in archive.files if k not in ("ee_raw", "a_raw")}
    np.savez_compressed(path, **stale)

    with pytest.raises(ValueError, match=r"ee_raw, a_raw.*python -m robot_ood\.cli build graph --suite"):
        NodeTable.load(tmp_path)


def test_segment_hand_computed_values():
    # single demo, coarsen=2, D=2, T=5, q is a simple arange so displacement is hand-checkable
    q = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.]], dtype=np.float32)
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    qdot[-1] = qdot[-2]
    demo = {"q": q, "qdot": qdot, "grip": np.zeros(5, np.uint8), "stage": np.zeros(5, np.int16), "task_id": 0}
    table = NodeTable.from_demos([demo], task_names=["t"], coarsen=2)
    # nodes: [0,1] (size2), [2,3] (size2), [4] (size1) -> t_raw = [0,2,4]
    np.testing.assert_array_equal(table.t_raw, np.array([0, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(table.n_members, np.array([2, 2, 1], dtype=np.int16))

    q_seg, qdot_seg = table.segment(0, k=2)
    np.testing.assert_array_equal(q_seg, q[0:2].astype(np.float64))

    # last node (size 1, t_raw=4) requesting k=3 must edge-pad with q[4] three times
    q_seg, qdot_seg = table.segment(2, k=3)
    expected = np.repeat(q[4:5], 3, axis=0).astype(np.float64)
    np.testing.assert_array_equal(q_seg, expected)


# ==================================================================================================
# coarse qdot correctness (the xcoarsen bug)
# ==================================================================================================
def test_coarse_qdot_is_endpoint_displacement_not_mean_of_deltas():
    # hand-built demo: q increases by 1.0 per raw step over 7 steps, coarsen=3 (T=7=2*3+1 so the
    # trailing group has size 1 and phase.max() hits exactly 1.0, satisfying validate())
    q = np.stack([np.arange(7, dtype=np.float32)] * D, axis=1)       # [7,3], each dim = 0..6
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    qdot[-1] = qdot[-2]
    demo = {"q": q, "qdot": qdot, "grip": np.zeros(7, np.uint8), "stage": np.zeros(7, np.int16), "task_id": 0}
    table = NodeTable.from_demos([demo], task_names=["t"], coarsen=3)

    # node 0 spans raw frames [0,1,2]: correct coarse qdot = q[2]-q[0] = 2.0 (NOT mean-of-deltas = 1.0)
    np.testing.assert_allclose(table.qdot[0], np.full(D, 2.0, dtype=np.float32))
    # the WRONG (mean-of-per-step-deltas) value would be 1.0 -- assert we are NOT that
    assert not np.allclose(table.qdot[0], np.full(D, 1.0, dtype=np.float32))


# ==================================================================================================
# features()
# ==================================================================================================
def test_features_width_and_no_task_or_stage_leak():
    table = _tiny_table()
    feats = table.features()
    expected_width = 2 * D + 1 + 2 * schema.PSI_FREQS
    assert feats.shape == (len(table), expected_width)

    # two tables identical except task_id must produce IDENTICAL features
    demos_a = [_make_demo(9, task_id=0, seed=5), _make_demo(7, task_id=0, grip_flip=3, seed=6)]
    demos_b = [_make_demo(9, task_id=7, seed=5), _make_demo(7, task_id=9, grip_flip=3, seed=6)]
    table_a = NodeTable.from_demos(demos_a, task_names=["x"], coarsen=3)
    table_b = NodeTable.from_demos(demos_b, task_names=["x"] * 10, coarsen=3)
    assert not np.array_equal(table_a.task_id, table_b.task_id)   # sanity: task_id really differs
    np.testing.assert_array_equal(table_a.features(), table_b.features())


def test_feature_stats_shape():
    table = _tiny_table()
    mean, std = table.feature_stats()
    assert mean.shape == (2 * D + 1,)
    assert std.shape == (2 * D + 1,)
    assert np.all(std > 0)


# ==================================================================================================
# validate() corruption detection
# ==================================================================================================
def test_validate_raises_on_truncated_phase():
    table = _tiny_table()
    table.phase = table.phase.copy()
    table.phase[:] = np.clip(table.phase, 0.0, 0.44).astype(np.float32)   # truncated phase
    with pytest.raises(ValueError, match="phase convention"):
        table.validate()


def test_validate_raises_on_shuffled_owner():
    table = _tiny_table()
    assert table.owner[0] != table.owner[-1]      # sanity: owner really is non-constant here
    table.owner = table.owner[::-1].copy()         # reverse -> guaranteed non-decreasing violation
    with pytest.raises(ValueError, match="owner"):
        table.validate()


def test_validate_raises_on_t_raw_past_demo_end():
    table = _tiny_table()
    table.t_raw = table.t_raw.copy()
    last_of_demo0 = table.demo_nodes(0)[-1]
    table.t_raw[last_of_demo0] += 1          # shift past demo 0's raw end -> spills into demo 1
    # depending on which structural check trips first this is caught either as a broken
    # owner-contiguous chain or as an out-of-range raw pointer -- both messages name "t_raw" and
    # both are exactly the "decodes frames from the WRONG DEMONSTRATION" failure mode being guarded.
    with pytest.raises(ValueError, match="t_raw"):
        table.validate()


def test_validate_passes_on_clean_table():
    _tiny_table().validate()


# ==================================================================================================
# demo_nodes / torch
# ==================================================================================================
def test_demo_nodes_order():
    table = _tiny_table()
    nodes0 = table.demo_nodes(0)
    nodes1 = table.demo_nodes(1)
    np.testing.assert_array_equal(np.sort(nodes0), nodes0)     # already in order
    assert set(nodes0.tolist()) & set(nodes1.tolist()) == set()
    np.testing.assert_array_equal(table.owner[nodes0], 0)
    np.testing.assert_array_equal(table.owner[nodes1], 1)


def test_torch_accessor():
    torch = pytest.importorskip("torch")
    table = _tiny_table()
    td = table.torch()
    assert set(td) == set(schema.NODES_KEYS)
    assert td["q"].dtype == torch.float32
    assert td["owner"].dtype == torch.int64
    assert td["q"].shape == tuple(table.q.shape)
