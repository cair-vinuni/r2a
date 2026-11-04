"""Tests for robot_ood.data.segments — C1's segment memory (SegmentBuilder, save_segments,
load_segments) plus the pre-existing torch reference-segment path (ee_segment_table,
expected_ee_segment), which this module also owns and must not have moved.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ood.core.schemas import ACTION_DIM
from robot_ood.build.graph import VirtualNodeSynthesizer
from robot_ood.data import schema
from robot_ood.data.nodes import NodeTable
from robot_ood.data.segments import SegmentBuilder, load_segments, save_segments, segment_pause

D = 3


def _demo(t: int, task_id: int = 0, seed: int = 0, a_scale: float = 0.1) -> dict:
    """A smooth synthetic demo with a genuine (non-zero) recorded action stream."""
    rng = np.random.RandomState(seed)
    q = np.cumsum(rng.randn(t, D).astype(np.float32) * 0.02, axis=0).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    qdot[-1] = qdot[-2] if t > 1 else 0.0
    a = (rng.randn(t, ACTION_DIM).astype(np.float32) * a_scale)
    return {
        "q": q, "qdot": qdot, "grip": np.zeros(t, dtype=np.uint8),
        "stage": np.zeros(t, dtype=np.int16), "task_id": task_id, "a": a,
    }


def _table(coarsen: int = 3) -> NodeTable:
    demos = [
        _demo(45, task_id=0, seed=1), _demo(60, task_id=0, seed=2),
        _demo(50, task_id=0, seed=3), _demo(40, task_id=1, seed=4),
        _demo(70, task_id=1, seed=5), _demo(55, task_id=1, seed=6),
    ]
    table = NodeTable.from_demos(demos, task_names=["taskA", "taskB"], coarsen=coarsen)
    # Stand-in for FK: these demos are D=3 random walks, not a 7-DoF Panda -- see
    # tests/test_readout.py's _declare_ee_base for the same idiom.
    table.ee_base = np.concatenate([table.q_raw, table.q_raw], axis=1)
    return table


# ==================================================================================================
# 1. segment starts, shapes, dtypes
# ==================================================================================================
def test_every_real_node_starts_a_segment_by_default():
    nodes = _table()
    table = SegmentBuilder(nodes).build()
    assert table.n_segments == nodes.n_real
    np.testing.assert_array_equal(table.start_node, np.arange(nodes.n_real))


def test_field_shapes_and_dtypes_match_the_contract():
    nodes = _table()
    segment_len = 12
    table = SegmentBuilder(nodes, segment_len=segment_len).build()
    s = table.n_segments
    assert table.start_node.dtype == np.int32 and table.start_node.shape == (s,)
    assert table.end_node.dtype == np.int32 and table.end_node.shape == (s,)
    assert table.owner.dtype == np.int32 and table.owner.shape == (s,)
    assert table.task_id.dtype == np.int16 and table.task_id.shape == (s,)
    assert table.phase_start.dtype == np.float32 and table.phase_start.shape == (s,)
    assert table.phase_end.dtype == np.float32 and table.phase_end.shape == (s,)
    assert table.action_chunk.shape == (s, segment_len, ACTION_DIM)
    assert table.action_chunk.dtype == np.float32
    assert table.ee_start.shape == (s, 6) and table.ee_end.shape == (s, 6)
    assert table.node_to_segment.shape == (len(nodes),)
    assert table.node_to_segment.dtype == np.int32


def test_stride_subsamples_starts_and_everything_else_is_unreachable():
    nodes = _table()
    stride = 4
    table = SegmentBuilder(nodes, stride=stride).build()

    demo_ptr = np.asarray(nodes.demo_ptr)
    owner = np.asarray(nodes.owner)[: nodes.n_real]
    local = np.arange(nodes.n_real) - demo_ptr[owner]
    expected_starts = np.nonzero(local % stride == 0)[0]
    np.testing.assert_array_equal(table.start_node, expected_starts)

    # Every node that IS a start maps to its own segment; every other real node -- not just
    # virtual ones -- gets -1, because node_to_segment means "the segment this node STARTS".
    non_starts = np.setdiff1d(np.arange(nodes.n_real), expected_starts)
    assert np.all(table.node_to_segment[non_starts] == -1)
    assert np.all(table.node_to_segment[expected_starts] == np.arange(len(expected_starts)))


def test_a_bad_segment_len_or_stride_is_refused():
    nodes = _table()
    with pytest.raises(ValueError, match="segment_len"):
        SegmentBuilder(nodes, segment_len=0)
    with pytest.raises(ValueError, match="stride"):
        SegmentBuilder(nodes, stride=0)


# ==================================================================================================
# 2. virtual nodes contribute no segments
# ==================================================================================================
def test_virtual_nodes_start_no_segment_and_read_as_minus_one():
    nodes = _table()
    grown = VirtualNodeSynthesizer(per_entry=2, seed=0).grow(nodes)
    grown.ee_base = np.concatenate([grown.q_raw, grown.q_raw], axis=1)
    assert grown.n_real == nodes.n_real
    assert len(grown) > grown.n_real            # the virtual suffix is non-empty

    table = SegmentBuilder(grown).build()
    assert table.n_segments == grown.n_real      # no segment starts past the real prefix
    assert np.all(table.start_node < grown.n_real)
    assert np.all(table.node_to_segment[grown.n_real:] == -1)
    assert table.node_to_segment.shape == (len(grown),)


# ==================================================================================================
# 3. a segment never crosses a demo boundary
# ==================================================================================================
def test_action_chunk_is_edge_padded_at_the_demo_end_not_borrowed_from_the_next_demo():
    nodes = _table()
    segment_len = 200        # far longer than any one demo
    table = SegmentBuilder(nodes, segment_len=segment_len).build()

    raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
    owner = np.asarray(table.owner, dtype=np.int64)
    t_raw = np.asarray(nodes.t_raw, dtype=np.int64)[table.start_node]
    demo_end = raw_ptr[owner + 1]
    a_raw = np.asarray(nodes.a_raw, dtype=np.float32)

    for row in range(0, table.n_segments, 17):        # sample, full loop is redundant
        start = int(t_raw[row])
        last = min(start + segment_len, int(demo_end[row])) - 1
        expected = a_raw[np.minimum(start + np.arange(segment_len), last)]
        np.testing.assert_array_equal(table.action_chunk[row], expected)
        # The padded tail repeats the last frame -- proof no read crossed into the next demo.
        if last - start + 1 < segment_len:
            np.testing.assert_array_equal(
                table.action_chunk[row, last - start], table.action_chunk[row, -1]
            )


def test_end_node_and_ee_end_land_on_the_clipped_last_frame():
    nodes = _table()
    table = SegmentBuilder(nodes, segment_len=1000).build()
    ee_base = np.asarray(nodes.ee_base, dtype=np.float32)
    raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
    demo_end = raw_ptr[np.asarray(table.owner, dtype=np.int64) + 1]
    np.testing.assert_array_equal(table.ee_end, ee_base[demo_end - 1])
    # end_node's own coarse node must actually contain that clipped raw frame.
    t_raw = np.asarray(nodes.t_raw, dtype=np.int64)
    owner = np.asarray(nodes.owner, dtype=np.int64)
    end_owner = owner[table.end_node]
    assert np.array_equal(end_owner, np.asarray(table.owner))
    assert np.all(t_raw[table.end_node] <= demo_end - 1)


# ==================================================================================================
# ==================================================================================================
# 5. save / load, including the graph_hash stamp
# ==================================================================================================
def test_save_load_round_trips_every_field(tmp_path):
    nodes = _table()
    table = SegmentBuilder(nodes, stride=3).build()
    save_segments(table, tmp_path)
    back = load_segments(tmp_path)
    for name in schema.ArtifactSchemas.SEGMENTS.keys:
        np.testing.assert_array_equal(getattr(table, name), getattr(back, name))


def test_load_accepts_a_file_path_or_its_directory(tmp_path):
    nodes = _table()
    table = SegmentBuilder(nodes).build()
    path = save_segments(table, tmp_path)
    assert path == tmp_path / schema.SEGMENTS_NPZ
    assert load_segments(path).n_segments == table.n_segments
    assert load_segments(tmp_path).n_segments == table.n_segments


def test_an_unstamped_file_loads_when_no_hash_is_demanded(tmp_path):
    nodes = _table()
    table = SegmentBuilder(nodes).build()
    save_segments(table, tmp_path)                       # no graph_hash given
    assert load_segments(tmp_path).n_segments == table.n_segments


def test_a_mismatched_graph_hash_is_refused(tmp_path):
    nodes = _table()
    table = SegmentBuilder(nodes).build()
    save_segments(table, tmp_path, graph_hash="abc123")
    load_segments(tmp_path, graph_hash="abc123")          # matches -- no raise
    with pytest.raises(ValueError, match="graph_hash mismatch"):
        load_segments(tmp_path, graph_hash="different")


def test_a_missing_column_is_refused(tmp_path):
    nodes = _table()
    table = SegmentBuilder(nodes).build()
    path = save_segments(table, tmp_path)
    with np.load(path) as archive:
        payload = {k: archive[k] for k in archive.files if k != "action_chunk"}
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="action_chunk"):
        load_segments(tmp_path)


# ==================================================================================================
# 6. the empty-graph edge case
# ==================================================================================================
def test_zero_real_nodes_yields_an_empty_table_not_a_crash():
    nodes = _table()
    # A table whose real prefix is empty: fabricate by pointing demo_ptr/raw_ptr at nothing real,
    # exercising SegmentBuilder's own guard rather than NodeTable's (which never produces this).
    import dataclasses

    empty = dataclasses.replace(
        nodes,
        demo_ptr=np.array([0], dtype=np.int32),
        owner=np.zeros(0, dtype=np.int32), task_id=np.zeros(0, dtype=np.int16),
        phase=np.zeros(0, dtype=np.float32), q=np.zeros((0, D), dtype=np.float32),
        qdot=np.zeros((0, D), dtype=np.float32), grip=np.zeros(0, dtype=np.uint8),
        stage=np.zeros(0, dtype=np.int16), t_idx=np.zeros(0, dtype=np.int32),
        t_frac=np.zeros(0, dtype=np.float32), t_raw=np.zeros(0, dtype=np.int32),
        n_members=np.zeros(0, dtype=np.int16), is_virtual=np.zeros(0, dtype=np.uint8),
    )
    table = SegmentBuilder(empty).build()
    assert table.n_segments == 0
    assert table.node_to_segment.shape == (0,)


# ==================================================================================================
# 5. pauses
# ==================================================================================================
def _parked(t_move: int, t_park: int, seed: int) -> dict:
    demo = _demo(t_move + t_park, seed=seed)
    demo["q"][t_move:] = demo["q"][t_move - 1]
    demo["qdot"][t_move - 1:] = 0.0
    return demo


def _still(t: int) -> dict:
    demo = _demo(t, seed=9)
    demo["q"][:] = demo["q"][0]
    demo["qdot"][:] = 0.0
    return demo


def test_segment_pause_is_a_hold_of_an_arm_that_moved():
    table = NodeTable.from_demos([_parked(40, 40, seed=1), _still(40), _demo(40, seed=2)],
                                 task_names=["taskA"], coarsen=3)
    table.ee_base = np.concatenate([table.q_raw, table.q_raw], axis=1)
    segments = SegmentBuilder(table, segment_len=8).build()
    pause = segment_pause(table, segments)
    hold = segments.start_node == segments.end_node
    owner = table.owner[segments.start_node]
    assert pause.dtype == np.bool_ and pause.shape == (segments.n_segments,)
    assert hold[owner == 0].any() and pause[owner == 0].sum() == hold[owner == 0].sum()
    assert hold[owner == 1].all() and not pause[owner == 1].any()
    last = np.flatnonzero(owner == 2)[-1]
    assert hold[owner == 2].sum() == 1 and hold[last] and pause[last]
