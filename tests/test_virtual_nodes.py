"""Tests for the synthesised entry cloud: the is_virtual column, the approach relation, and the
readout mask that makes a virtual node seedable, message-passing, and never selectable."""

import numpy as np
import pytest

from robot_ood.build.graph import VirtualNodeSynthesizer
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet, approach_edges
from robot_ood.data.nodes import NodeTable
from robot_ood.runtime.readout import ReadoutContext, read_out

D = 3
BAND = schema.VIRTUAL_PHASE_BAND


def _demo(t: int, task_id: int, seed: int) -> dict:
    """A smooth synthetic demo whose qdot is the exact finite difference of its q."""
    rng = np.random.RandomState(seed)
    q = np.cumsum(rng.randn(t, D).astype(np.float32) * 0.01, axis=0).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    qdot[-1] = qdot[-2]
    return {
        "q": q, "qdot": qdot,
        "grip": np.zeros(t, dtype=np.uint8), "stage": np.zeros(t, dtype=np.int16),
        "task_id": task_id, "ee": np.concatenate([q, q], axis=1).astype(np.float32),
        "a": np.zeros((t, 7), dtype=np.float32),
    }


def _table() -> NodeTable:
    """Four demos over two tasks, long enough that phase <= BAND is a small stratum.

    T = 5k + 1 so the last node starts on the final frame and phase reaches exactly 1.0, which is
    what NodeTableValidator.check_phase_convention demands.
    """
    demos = [_demo(61, task_id=j // 2, seed=j + 1) for j in range(4)]
    return NodeTable.from_demos(demos, task_names=["taskA", "taskB"], coarsen=5)


def _grown(per_entry: int = 3, seed: int = 0) -> tuple[NodeTable, NodeTable]:
    real = _table()
    return real, VirtualNodeSynthesizer(per_entry=per_entry, seed=seed).grow(real)


# ==================================================================================================
# 1. the relation vocabulary grew by APPENDING
# ==================================================================================================
def test_approach_is_appended_last():
    # The bimanual relations were appended after it, so every earlier index is unchanged.
    assert schema.RELATIONS[12] == "approach" and schema.RELATIONS[-1] == "cross_arm_align"
    assert schema.N_RELATIONS == 16
    # Every pre-existing relation keeps the index its trained embedding row was learned at.
    assert schema.REL_INDEX["sibling"] == 10 and schema.REL_INDEX["align"] == 11
    assert schema.REL_INDEX["approach"] == 12


# ==================================================================================================
# 2. synthesis
# ==================================================================================================
def test_grow_appends_m_virtual_per_entry_node_and_leaves_the_real_prefix_alone():
    real, grown = _grown(per_entry=3)
    n_entry = int((np.asarray(real.phase) <= BAND).sum())
    assert n_entry > 0
    assert len(grown) == len(real) + 3 * n_entry
    assert grown.n_real == len(real)
    for name in ("q", "qdot", "owner", "task_id", "t_idx", "t_raw", "phase"):
        np.testing.assert_array_equal(getattr(grown, name)[: len(real)], getattr(real, name))


def test_virtual_flag_is_a_suffix_and_the_table_validates():
    _, grown = _grown()
    flag = grown.virtual_mask
    assert not flag[: grown.n_real].any() and flag[grown.n_real :].all()
    grown.validate()


def test_virtual_configs_are_offsets_of_their_parent_within_the_calibrated_radius():
    real, grown = _grown(per_entry=2)
    parents = grown.demo_ptr[grown.owner[grown.n_real :]] + grown.t_idx[grown.n_real :]
    offsets = np.linalg.norm(grown.q[grown.n_real :] - grown.q[parents], axis=1)
    assert np.all(offsets > 0.0)
    assert np.all(np.asarray(grown.phase)[grown.n_real :] <= BAND)


def test_synthesis_is_seed_deterministic():
    _, a = _grown(per_entry=2, seed=7)
    _, b = _grown(per_entry=2, seed=7)
    _, c = _grown(per_entry=2, seed=8)
    np.testing.assert_array_equal(a.q, b.q)
    assert not np.array_equal(a.q, c.q)


def test_per_entry_zero_is_the_identity():
    real, grown = _grown(per_entry=0)
    assert grown is real


def test_growing_twice_is_refused():
    _, grown = _grown()
    with pytest.raises(ValueError, match="already carries virtual"):
        VirtualNodeSynthesizer(per_entry=1).grow(grown)


# ==================================================================================================
# 3. save / load, including artifacts written before the column existed
# ==================================================================================================
def test_is_virtual_survives_a_save_load_round_trip(tmp_path):
    _, grown = _grown()
    grown.save(tmp_path)
    back = NodeTable.load(tmp_path)
    np.testing.assert_array_equal(back.is_virtual, grown.is_virtual)
    np.testing.assert_array_equal(back.q, grown.q)
    assert back.n_real == grown.n_real


def test_artifact_without_the_column_loads_as_all_real(tmp_path):
    table = _table()
    table.save(tmp_path)
    src = tmp_path / schema.NODES_NPZ
    with np.load(src) as archive:
        payload = {k: archive[k] for k in archive.files if k != "is_virtual"}
    assert "is_virtual" not in payload
    np.savez_compressed(src, **payload)

    back = NodeTable.load(tmp_path)
    assert back.is_virtual.dtype == np.uint8
    assert not back.virtual_mask.any()
    assert back.n_real == len(back)


# ==================================================================================================
# 4. the approach relation
# ==================================================================================================
def test_approach_edges_run_virtual_to_the_parent_only():
    _, grown = _grown(per_entry=2)
    edges = approach_edges(grown)
    assert len(edges) == len(grown) - grown.n_real
    assert np.array_equal(np.sort(edges.src), np.arange(grown.n_real, len(grown)))
    expected = grown.demo_ptr[grown.owner[edges.src]] + grown.t_idx[edges.src]
    np.testing.assert_array_equal(edges.dst, expected)
    assert np.all(edges.rel == schema.REL_INDEX["approach"])


def test_approach_is_absent_from_a_real_only_graph():
    assert len(approach_edges(_table())) == 0


def test_virtual_nodes_change_no_other_relation():
    real, grown = _grown(per_entry=2)
    before = EdgeSet.build(real, device="cpu").counts()
    after = EdgeSet.build(grown, device="cpu").counts()
    assert after.pop("approach") == len(grown) - grown.n_real
    before.pop("approach")
    assert after == before


def test_edge_set_with_virtual_nodes_validates_and_round_trips(tmp_path):
    _, grown = _grown(per_entry=2)
    edges = EdgeSet.build(grown, device="cpu")
    edges.validate(grown)
    path = edges.save(tmp_path / schema.EDGES_NPZ)
    EdgeSet.load(path, strict=True).validate(grown)


# ==================================================================================================
# 5. the readout mask -- seedable, propagating, never selectable
# ==================================================================================================
def _readout_on(table: NodeTable, p: np.ndarray):
    # D=3 here, so supply the base-frame poses directly rather than through the 7-DoF Panda FK.
    table.ee_base = np.concatenate([table.q_raw, table.q_raw], axis=1)
    return read_out(p, ReadoutContext(nodes=table, topm=32))


def test_virtual_mass_never_reaches_the_target():
    _, grown = _grown(per_entry=2)
    p = np.zeros(len(grown))
    p[: grown.n_real] = 1.0
    p_clean = p / p.sum()

    # The same real distribution, plus a mountain of mass on the synthesised cloud.
    p_dirty = p.copy()
    p_dirty[grown.n_real :] = 1e3
    p_dirty /= p_dirty.sum()

    clean = _readout_on(grown, p_clean)
    dirty = _readout_on(grown, p_dirty)
    np.testing.assert_allclose(dirty.q_star, clean.q_star, rtol=0, atol=1e-12)


def test_all_virtual_mass_is_an_error_not_a_virtual_target():
    _, grown = _grown(per_entry=2)
    p = np.zeros(len(grown))
    p[grown.n_real :] = 1.0
    with pytest.raises(ValueError, match="is_virtual mask"):
        _readout_on(grown, p / p.sum())


def test_off_manifold_is_measured_against_real_nodes_only():
    real, grown = _grown(per_entry=4)
    q = np.asarray(grown.q[grown.n_real], dtype=np.float64)     # sits ON a virtual node
    ctx_real = ReadoutContext(nodes=real)
    ctx_grown = ReadoutContext(nodes=grown)
    assert ctx_grown.nn_distance(q) == pytest.approx(ctx_real.nn_distance(q))
    assert ctx_grown.nn_distance(q) > 0.0
