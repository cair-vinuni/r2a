"""Tests for robot_ood.build.graph — the HDF5 -> demonstration-graph pipeline.

Runs entirely against a fabricated hdf5 fixture (_write_fixture) that matches the REAL
LIBERO key layout (obs/joint_states, obs/gripper_states, actions, one file per task, one
group per demo) so it exercises the exact same code path the real 500-demos-per-suite builds do,
without requiring the real dataset. Tests that DO need the real data are marked with
skip_no_data, following tests/test_parity.py's skip_no_field pattern.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_ood.core.paths import Paths
from robot_ood.data import schema
from robot_ood.build.graph import (
    DemoLoader,
    EdgeBuilder,
    GraphBuilder,
    LoadLimits,
    NodeBuilder,
    measured_grip,
    stage_labels,
)
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable

# ==================================================================================================
# real-data skip marker (tests/test_parity.py's skip_no_field pattern)
# ==================================================================================================
_PATHS = Paths.resolve()


def _has_real_hdf5(name: str = "libero_object") -> bool:
    try:
        _PATHS.hdf5(name)
        return True
    except FileNotFoundError:
        return False


HAS_REAL_DATA = _has_real_hdf5()
skip_no_data = pytest.mark.skipif(not HAS_REAL_DATA, reason="no real LIBERO hdf5 data found")


# ==================================================================================================
# stage adapters -- the builder API is class-per-stage (DemoLoader/NodeBuilder/EdgeBuilder), but each
# assertion below is about ONE stage's output, so these three helpers name that stage and hand back
# the plain value the assertions read. They add no logic: each is a construct-and-call.
# ==================================================================================================
def load_demos(hdf5_dir, *, limit_tasks=None, limit_demos=None):
    """(per-demo payload dicts, task names) -- what NodeTable.from_demos consumes."""
    batch = DemoLoader(hdf5_dir, limits=LoadLimits(tasks=limit_tasks, demos=limit_demos)).load()
    return batch.as_payloads(), batch.task_names


def build_nodes(hdf5_dir, out_dir, *, coarsen):
    """HDF5 dir -> written g_nodes.npz path, as a str."""
    batch = DemoLoader(hdf5_dir).load()
    builder = NodeBuilder(coarsen=coarsen)
    return str(builder.save(builder.build(batch), Path(out_dir)))


def build_graph(suite, *, limit_tasks=None, limit_demos=None, **kw):
    """Full chain -> the metrics dict (GraphArtifacts.to_dict())."""
    limits = LoadLimits(tasks=limit_tasks, demos=limit_demos)
    return GraphBuilder(suite, limits=limits, **kw).build().to_dict()


def build_edges(nodes_path_or_table, out_dir):
    """NodeTable (live or on disk) -> written g_edges.npz path, as a str."""
    table = (nodes_path_or_table if isinstance(nodes_path_or_table, NodeTable)
             else NodeTable.load(nodes_path_or_table))
    builder = EdgeBuilder()
    return str(builder.save(builder.build(table), Path(out_dir)))


# ==================================================================================================
# every build_graph(logger=None) call below owns a fresh StageLogger, which ALWAYS writes its run
# directory under Paths.outputs() regardless of build_graph's own out_dir argument (that argument
# only controls where g_nodes.npz/g_edges.npz land, not the run/manifest bookkeeping dir) -- redirect
# it into this test's own tmp_path so the suite never pollutes the real repo outputs/ tree, or worse,
# repoints outputs/<real-suite-name>/latest (e.g. "object") away from an actual build.
# ==================================================================================================
@pytest.fixture(autouse=True)
def _isolate_stagelogger_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOT_OOD_OUTPUTS", str(tmp_path / "_stagelogger_outputs"))


# ==================================================================================================
# fixture: a tiny hdf5 tree matching the real key layout
# ==================================================================================================
D_JOINT = 7


def _synth_demo(t: int, *, has_gripper_states: bool = True, seed: int = 0, lag: int = 2) -> dict:
    """One synthetic demo's raw arrays, matching the real per-demo hdf5 layout. Scripts ONE clean
    gripper close/open cycle (a rise then a fall, roughly the middle third of the episode) so both
    grip (measured-channel) and stage (command-channel release-count) transitions are exercised
    by every demo. The measured gripper_states channel is built to LAG the commanded
    actions[:, 6] by lag frames — mirroring the real command-leads-measurement offset the
    module docstring documents — and to sit on the correct side of schema.GRIP_OPEN_THR in each
    state (near 0 when closed, comfortably above threshold when open).
    """
    rng = np.random.RandomState(seed)
    q = np.cumsum(rng.randn(t, D_JOINT).astype(np.float64) * 0.01, axis=0)

    rise, fall = t // 3, (2 * t) // 3
    cmd = np.full(t, -1.0)
    cmd[rise:fall] = 1.0

    actions = np.zeros((t, 7), dtype=np.float64)
    actions[:, :6] = rng.randn(t, 6).astype(np.float64) * 0.05
    actions[:, 6] = cmd

    closed = np.zeros(t, dtype=bool)
    closed[min(t, rise + lag):min(t, fall + lag)] = True
    finger = np.where(closed, 0.01, 4 * schema.GRIP_OPEN_THR)   # closed near 0; open well above thr
    # force grip[-1] != grip[-2]: coarsen_segments() then starts a brand-new (size-1) node at the
    # very last raw frame regardless of coarsen, so that node's phase hits exactly 1.0 -- the
    # _tiny_table trick in test_nodes.py achieves the same NodeTable.validate() phase-convention
    # requirement (table-wide phase.max() > 0.99) via a hand-picked T instead.
    finger[-1] = 4 * schema.GRIP_OPEN_THR if closed[-2] else 0.01
    gripper_states = np.stack([finger, -finger], axis=1)

    demo = {
        "joint_states": q,
        "actions": actions,
        "ee_pos": q[:, :3] * 0.1,
        "ee_ori": q[:, 3:6] * 0.1,
        "states": np.zeros((t, 110), dtype=np.float64),
    }
    if has_gripper_states:
        demo["gripper_states"] = gripper_states
    return demo


def _write_fixture(
    root: Path, *, n_tasks: int = 2, n_demos: int = 3, base_t: int = 40,
    missing_gripper: tuple[int, int] | None = None,
) -> Path:
    """Write n_tasks hdf5 files (task{i}_demo.hdf5), each holding n_demos demo groups.
    missing_gripper=(task_idx, demo_idx) omits obs/gripper_states for exactly that one demo,
    to exercise robot_ood.build.graph.measured_grip's command-channel fallback."""
    import h5py

    root.mkdir(parents=True, exist_ok=True)
    for ti in range(n_tasks):
        fp = root / f"task{ti}_demo.hdf5"
        with h5py.File(fp, "w") as f:
            grp = f.create_group("data")
            for di in range(n_demos):
                t = base_t + 3 * di + ti     # varying lengths -- tests must not assume a fixed T
                has_gs = (ti, di) != missing_gripper
                d = _synth_demo(t, has_gripper_states=has_gs, seed=ti * 100 + di)
                dgrp = grp.create_group(f"demo_{di}")
                dgrp.create_dataset("actions", data=d["actions"])
                dgrp.create_dataset("states", data=d["states"])
                obs = dgrp.create_group("obs")
                obs.create_dataset("joint_states", data=d["joint_states"])
                obs.create_dataset("ee_pos", data=d["ee_pos"])
                obs.create_dataset("ee_ori", data=d["ee_ori"])
                if "gripper_states" in d:
                    obs.create_dataset("gripper_states", data=d["gripper_states"])
    return root


# ==================================================================================================
# stage_labels — pure array unit tests (no HDF5). DIAGNOSTIC ONLY (see schema.py's note under
# RELATIONS): a plain, undebounced gripper-release count, with no debounce and no trailing fold —
# both were tried and retired because the label is a cumulative sum over a channel that really does
# flicker on real data, and no threshold can clean that without also erasing genuine short stages
# elsewhere. These tests lock in the current, deliberately simple behaviour.
# ==================================================================================================
def test_stage_labels_two_cycles_exact():
    """Two full close/open cycles (2 releases) -> stage == [0]*n + [1]*m + [2]*k."""
    n, m, k = 12, 9, 7
    # stage0: open(a) -> closed(b) -> [RELEASE] stage1: open(c) -> closed(d) -> [RELEASE] stage2: open(e)
    a, b = n // 2, n - n // 2
    c, d = m // 2, m - m // 2
    gb = np.concatenate([np.zeros(a), np.ones(b), np.zeros(c), np.ones(d), np.zeros(k)])
    cmd = np.where(gb > 0.5, 1.0, -1.0)

    stage = stage_labels(cmd)
    expected = np.array([0] * n + [1] * m + [2] * k, dtype=np.int16)
    np.testing.assert_array_equal(stage, expected)
    assert stage.dtype == np.int16


def test_stage_labels_no_release_is_all_zero():
    cmd = np.concatenate([np.full(5, -1.0), np.full(20, 1.0)])   # grasps, never releases
    stage = stage_labels(cmd)
    np.testing.assert_array_equal(stage, np.zeros(25, dtype=np.int16))


def test_stage_labels_all_open_is_all_zero():
    cmd = np.full(10, -1.0)
    stage = stage_labels(cmd)
    assert stage.dtype == np.int16
    np.testing.assert_array_equal(stage, np.zeros(10, dtype=np.int16))


def test_stage_labels_no_debounce_short_flicker_still_counts():
    """The retired debounce would have erased a short release/regrasp blip; the current plain count
    does NOT -- it counts every diff == -1 transition, flicker or genuine, by design (see the
    module docstring's Decision 2 and schema.py's note under RELATIONS for why: no threshold can
    separate flicker from genuine transitions across every suite, so the label stopped trying to)."""
    hold = 40
    flicker = 3       # short blip; would have been erased by the retired debounce
    gb = np.concatenate([
        np.zeros(10), np.ones(hold // 2), np.zeros(flicker), np.ones(hold // 2), np.zeros(5),
    ])
    cmd = np.where(gb > 0.5, 1.0, -1.0)
    stage = stage_labels(cmd)
    assert int(stage.max()) == 2, "a plain release count must see BOTH releases, flicker or not"


def test_stage_labels_dtype_and_length():
    cmd = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    stage = stage_labels(cmd)
    assert stage.dtype == np.int16
    assert stage.shape == (6,)


# ==================================================================================================
# measured_grip
# ==================================================================================================
def test_measured_grip_uses_gripper_states():
    gs = np.array([[0.01, -0.01], [0.05, -0.05], [0.001, -0.001]])
    cmd = np.array([1.0, 1.0, 1.0])    # command disagrees with measurement on row 1 -- must be ignored
    grip = measured_grip(gs, cmd)
    np.testing.assert_array_equal(grip, np.array([1, 0, 1], dtype=np.uint8))
    assert grip.dtype == np.uint8


def test_measured_grip_falls_back_without_gripper_states():
    cmd = np.array([-1.0, 1.0, 1.0, -1.0])
    grip = measured_grip(None, cmd)
    np.testing.assert_array_equal(grip, np.array([0, 1, 1, 0], dtype=np.uint8))


# ==================================================================================================
# load_demos
# ==================================================================================================
def test_load_demos_shapes_dtypes_and_task_names(tmp_path):
    _write_fixture(tmp_path, n_tasks=2, n_demos=3, base_t=40)
    demos, task_names = load_demos(tmp_path)

    assert task_names == ["task0", "task1"]
    assert len(demos) == 6
    for d in demos:
        t = d["q"].shape[0]
        assert d["q"].shape == (t, D_JOINT) and d["q"].dtype == np.float32
        assert d["qdot"].shape == (t, D_JOINT) and d["qdot"].dtype == np.float32
        assert d["grip"].shape == (t,) and d["grip"].dtype == np.uint8
        assert d["stage"].shape == (t,) and d["stage"].dtype == np.int16
        assert d["ee"].shape == (t, 6) and d["ee"].dtype == np.float32
        assert d["a"].shape == (t, 7) and d["a"].dtype == np.float32
        assert set(np.unique(d["grip"]).tolist()) <= {0, 1}
        assert d["task_id"] in (0, 1)

    # first 3 demos (list order) belong to task0, next 3 to task1 (file-per-task, in-order)
    assert [d["task_id"] for d in demos[:3]] == [0, 0, 0]
    assert [d["task_id"] for d in demos[3:]] == [1, 1, 1]


def test_load_demos_qdot_matches_finite_difference(tmp_path):
    _write_fixture(tmp_path, n_tasks=1, n_demos=1, base_t=37)
    demos, _ = load_demos(tmp_path)
    q = demos[0]["q"]
    expected = np.zeros_like(q)
    expected[:-1] = q[1:] - q[:-1]
    expected[-1] = expected[-2]
    np.testing.assert_allclose(demos[0]["qdot"], expected, atol=1e-6)


def test_load_demos_grip_transitions_exercised(tmp_path):
    """The fixture scripts a real close/open cycle -- grip must show BOTH 0 and 1 in every demo."""
    _write_fixture(tmp_path, n_tasks=1, n_demos=2, base_t=45)
    demos, _ = load_demos(tmp_path)
    for d in demos:
        vals = set(np.unique(d["grip"]).tolist())
        assert vals == {0, 1}, f"expected both grip states, got {vals}"


def test_load_demos_grip_from_measured_channel_not_command(tmp_path):
    """The measured channel LAGS the command by lag frames (scripted in the fixture) -- grip must
    follow the lagged measurement, not the instantaneous command."""
    lag = 2
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    import h5py

    t = 50
    d = _synth_demo(t, has_gripper_states=True, seed=0, lag=lag)
    fp = root / "task0_demo.hdf5"
    with h5py.File(fp, "w") as f:
        grp = f.create_group("data")
        dgrp = grp.create_group("demo_0")
        dgrp.create_dataset("actions", data=d["actions"])
        dgrp.create_dataset("states", data=d["states"])
        obs = dgrp.create_group("obs")
        obs.create_dataset("joint_states", data=d["joint_states"])
        obs.create_dataset("ee_pos", data=d["ee_pos"])
        obs.create_dataset("ee_ori", data=d["ee_ori"])
        obs.create_dataset("gripper_states", data=d["gripper_states"])

    demos, _ = load_demos(root)
    grip = demos[0]["grip"]
    cmd_closed = d["actions"][:, 6] > 0.5
    # in the lag window right after the command rises, the command says "closed" but the measured
    # channel (and hence grip) has not caught up yet
    rise = t // 3
    assert cmd_closed[rise] and grip[rise] == 0, "grip must lag the command, not track it instantly"
    # matches the measured signal directly (recomputed independently here, not via measured_grip)
    gs = d["gripper_states"]
    expected = (np.abs(gs).mean(axis=1) < schema.GRIP_OPEN_THR).astype(np.uint8)
    np.testing.assert_array_equal(grip, expected)


def test_load_demos_missing_gripper_states_falls_back(tmp_path):
    _write_fixture(tmp_path, n_tasks=1, n_demos=2, base_t=40, missing_gripper=(0, 1))
    demos, _ = load_demos(tmp_path)
    # demo_0 (has gripper_states) and demo_1 (missing) are both present, list-ordered by h5py key
    # iteration (not necessarily numeric) -- identify the fallback demo by re-deriving it directly.
    import h5py

    with h5py.File(tmp_path / "task0_demo.hdf5", "r") as f:
        keys = list(f["data"].keys())
        idx_missing = keys.index("demo_1")
        act = np.asarray(f["data"]["demo_1"]["actions"])
    expected = (act[:, 6] > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(demos[idx_missing]["grip"], expected)


def test_load_demos_limit_tasks_and_demos(tmp_path):
    _write_fixture(tmp_path, n_tasks=3, n_demos=4, base_t=40)
    demos, task_names = load_demos(tmp_path, limit_tasks=2, limit_demos=2)
    assert task_names == ["task0", "task1"]
    assert len(demos) == 4


def test_load_demos_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        load_demos("/nonexistent/definitely/not/a/real/path")


# ==================================================================================================
# build_nodes
# ==================================================================================================
def test_build_nodes_writes_valid_npz(tmp_path):
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=2, n_demos=3, base_t=40)
    out_dir = tmp_path / "artifacts"
    path = build_nodes(fixture, out_dir, coarsen=3)
    assert path.endswith(schema.NODES_NPZ)
    table = NodeTable.load(path)
    table.validate()
    assert table.n_demos == 6
    assert table.n_tasks == 2


def test_build_nodes_coarsen_reduces_node_count(tmp_path):
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=1, n_demos=2, base_t=100)
    out1 = build_nodes(fixture, tmp_path / "c1", coarsen=1)
    out5 = build_nodes(fixture, tmp_path / "c5", coarsen=5)
    t1, t5 = NodeTable.load(out1), NodeTable.load(out5)
    assert len(t5) < len(t1)


# ==================================================================================================
# build_edges
# ==================================================================================================
def test_build_edges_accepts_table_or_path(tmp_path):
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=2, n_demos=3, base_t=40)
    nodes_path = build_nodes(fixture, tmp_path / "artifacts", coarsen=3)
    table = NodeTable.load(nodes_path)

    p_from_table = build_edges(table, tmp_path / "from_table")
    p_from_path = build_edges(nodes_path, tmp_path / "from_path")

    es1 = EdgeSet.load(p_from_table)
    es2 = EdgeSet.load(p_from_path)
    es1.validate(table)
    es2.validate(table)
    assert es1.counts() == es2.counts()


# ==================================================================================================
# build_graph — full pipeline
# ==================================================================================================
def test_build_graph_roundtrip_validates(tmp_path):
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=2, n_demos=3, base_t=45)
    out_dir = tmp_path / "artifacts"

    metrics = build_graph("unit_test_suite", hdf5_dir=fixture, out_dir=out_dir, coarsen=3)

    assert metrics["n_nodes"] > 0
    assert metrics["n_demos"] == 6
    assert metrics["n_tasks"] == 2
    assert metrics["n_edges"] == sum(metrics["edge_counts"].values())

    table = NodeTable.load(metrics["nodes_path"])
    table.validate()
    es = EdgeSet.load(metrics["edges_path"])
    es.validate(table)


def test_build_graph_run_dir_has_provenance_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOT_OOD_OUTPUTS", str(tmp_path / "outputs"))
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=2, n_demos=2, base_t=40)

    metrics = build_graph("unit_test_suite2", hdf5_dir=fixture, coarsen=3)
    assert metrics["n_nodes"] > 0

    run_dirs = sorted((tmp_path / "outputs" / "unit_test_suite2").glob("graph_*"))
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]

    for fname in ("config.json", "manifest.json", "metrics.json", "log.txt"):
        assert (run_dir / fname).exists(), fname

    import json

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    recorded = set(manifest["inputs"].keys())
    expected_inputs = {str(p) for p in sorted(fixture.glob("*.hdf5"))}
    assert recorded == expected_inputs

    metrics_json = json.loads((run_dir / "metrics.json").read_text())
    assert metrics_json["n_nodes"] == metrics["n_nodes"]

    assert (run_dir / "artifacts" / schema.NODES_NPZ).exists()
    assert (run_dir / "artifacts" / schema.EDGES_NPZ).exists()


def test_build_graph_deterministic(tmp_path):
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=2, n_demos=3, base_t=40)

    m1 = build_graph("det_suite", hdf5_dir=fixture, out_dir=tmp_path / "run1", coarsen=3)
    m2 = build_graph("det_suite", hdf5_dir=fixture, out_dir=tmp_path / "run2", coarsen=3)

    b1_nodes = Path(m1["nodes_path"]).read_bytes()
    b2_nodes = Path(m2["nodes_path"]).read_bytes()
    assert b1_nodes == b2_nodes, "g_nodes.npz must be byte-identical across repeated builds"

    b1_edges = Path(m1["edges_path"]).read_bytes()
    b2_edges = Path(m2["edges_path"]).read_bytes()
    assert b1_edges == b2_edges, "g_edges.npz must be byte-identical across repeated builds"


def test_build_graph_single_grasp_suite_has_no_boundary_relation(tmp_path):
    """A single-grasp suite (no genuine stage transitions) must build normally. There is no
    boundary/stage-handoff relation at all any more to assert zero-edges on -- edge_counts simply
    never has a stage_next/stage_prev key (schema.RELATIONS dropped both; see the long comment
    there)."""
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=2, n_demos=3, base_t=40)
    metrics = build_graph("single_grasp_suite", hdf5_dir=fixture, out_dir=tmp_path / "artifacts", coarsen=3)
    assert set(metrics["edge_counts"]) == set(schema.RELATIONS)
    assert "stage_next" not in metrics["edge_counts"]
    assert "stage_prev" not in metrics["edge_counts"]


def test_build_graph_multi_release_stage_is_diagnostic_only(tmp_path):
    """A suite whose demos genuinely release the gripper twice must still report n_stages
    (DIAGNOSTIC ONLY now, see schema.py) counting those releases -- the field still works end-to-end
    through the real HDF5 pipeline -- but produces no boundary-type edges: that relation no longer
    exists at all, regardless of how many stages a demo has."""
    import h5py

    root = tmp_path / "hdf5"
    root.mkdir(parents=True, exist_ok=True)
    fp = root / "task0_demo.hdf5"
    n_demos = 3
    with h5py.File(fp, "w") as f:
        grp = f.create_group("data")
        for di in range(n_demos):
            rng = np.random.RandomState(di)
            seg = 60 + di
            gb = np.concatenate([
                np.zeros(seg), np.ones(seg),      # stage0 -> release 1
                np.zeros(seg), np.ones(seg),      # stage1 -> release 2
                np.zeros(30),                     # stage2 tail
            ])
            t = len(gb)
            q = np.cumsum(rng.randn(t, D_JOINT).astype(np.float64) * 0.01, axis=0)
            actions = np.zeros((t, 7), dtype=np.float64)
            actions[:, :6] = rng.randn(t, 6).astype(np.float64) * 0.05
            actions[:, 6] = np.where(gb > 0.5, 1.0, -1.0)
            finger = np.where(gb > 0.5, 0.01, 4 * schema.GRIP_OPEN_THR)
            # force a size-1 last coarse node (see _synth_demo's identical trick) so this demo's
            # phase reaches exactly 1.0 -- satisfies NodeTable.validate()'s phase-convention check.
            finger[-1] = 0.01 if finger[-2] > schema.GRIP_OPEN_THR else 4 * schema.GRIP_OPEN_THR
            gripper_states = np.stack([finger, -finger], axis=1)

            dgrp = grp.create_group(f"demo_{di}")
            dgrp.create_dataset("actions", data=actions)
            dgrp.create_dataset("states", data=np.zeros((t, 110)))
            obs = dgrp.create_group("obs")
            obs.create_dataset("joint_states", data=q)
            obs.create_dataset("ee_pos", data=q[:, :3] * 0.1)
            obs.create_dataset("ee_ori", data=q[:, 3:6] * 0.1)
            obs.create_dataset("gripper_states", data=gripper_states)

    metrics = build_graph("multistage_suite", hdf5_dir=root, out_dir=tmp_path / "artifacts", coarsen=5)
    assert metrics["n_stages"] == 3        # diagnostic: 2 releases seen -> 3 distinct stage values
    assert set(metrics["edge_counts"]) == set(schema.RELATIONS)
    assert "stage_next" not in metrics["edge_counts"]
    assert "stage_prev" not in metrics["edge_counts"]

    table = NodeTable.load(metrics["nodes_path"])
    table.validate()
    es = EdgeSet.load(metrics["edges_path"])
    es.validate(table)


def test_build_graph_given_logger_does_not_reenter(tmp_path, monkeypatch):
    """Passing an already-active logger must not create a second run directory."""
    monkeypatch.setenv("ROBOT_OOD_OUTPUTS", str(tmp_path / "outputs"))
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=1, n_demos=2, base_t=40)

    from robot_ood.core.report import StageLogger

    with StageLogger("graph", "given_logger_suite", config={}) as log:
        metrics = build_graph("given_logger_suite", hdf5_dir=fixture, coarsen=3, logger=log)
        assert metrics["n_nodes"] > 0

    run_dirs = sorted((tmp_path / "outputs" / "given_logger_suite").glob("graph_*"))
    assert len(run_dirs) == 1


# ==================================================================================================
# real-data smoke test
# ==================================================================================================
@skip_no_data
def test_build_graph_real_object_suite_smoke(tmp_path):
    """A tiny, fast slice of the real object suite (1 task, 2 demos) -- just enough to prove the
    pipeline runs end-to-end against real hdf5 without pulling in all 500 demos."""
    metrics = build_graph(
        "object", out_dir=tmp_path / "artifacts", coarsen=5, limit_tasks=1, limit_demos=2,
    )
    assert metrics["n_nodes"] > 0
    table = NodeTable.load(metrics["nodes_path"])
    table.validate()
    es = EdgeSet.load(metrics["edges_path"])
    es.validate(table)
