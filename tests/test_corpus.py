"""Tests for robot_ood.data.corpus — the corpus-measured-rate module (numpy only; no torch, no
mujoco; see the module's own docstring for why that constraint matters).

Two families:
  * SYNTHETIC (_make_demo / _synthetic_table, mirroring tests/test_nodes.py's
    _make_demo/_tiny_table idiom) — always run, no real artifacts required.
  * REAL-GRAPH (@skip_no_real_graph) — pin every measured number in robot_ood.data.corpus's
    docstrings against the fixed reference artifact outputs/long/frozen_learn1/artifacts (500
    demos, 28,476 nodes, 138,090 raw frames). A FIXED path is used deliberately (not Paths.graph("long"), which resolves to whatever run is
    currently latest — a moving target while other graph-build runs land in the same
    outputs/long/ tree). Skips cleanly if that directory is absent, following
    tests/test_builders.py's skip_no_data pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_ood.data import schema
from robot_ood.data.corpus import measure_inter_demo_spacing
from robot_ood.data.nodes import NodeTable

# ==================================================================================================
# synthetic-demo helpers (mirrors tests/test_nodes.py's _make_demo / _tiny_table)
# ==================================================================================================
D = 3


def _make_demo(t: int, task_id: int = 0, grip_flip: int | None = None, seed: int = 0) -> dict:
    """A deterministic synthetic demo: a smooth random walk in q, matching finite-difference qdot, an
    optional gripper open->closed flip at grip_flip."""
    rng = np.random.RandomState(seed)
    steps = rng.randn(t, D).astype(np.float32) * 0.02
    q = np.cumsum(steps, axis=0).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    qdot[-1] = qdot[-2] if t > 1 else 0.0
    grip = np.zeros(t, dtype=np.uint8)
    if grip_flip is not None:
        grip[grip_flip:] = 1
    stage = np.zeros(t, dtype=np.int16)
    return {"q": q, "qdot": qdot, "grip": grip, "stage": stage, "task_id": task_id}


def _synthetic_table(coarsen: int = 3) -> NodeTable:
    """6 demos over 2 tasks (3 demos/task), T in [40, 70], half with a mid-demo grip flip."""
    demos = [
        _make_demo(45, task_id=0, grip_flip=20, seed=1),
        _make_demo(60, task_id=0, grip_flip=30, seed=2),
        _make_demo(50, task_id=0, seed=3),
        _make_demo(40, task_id=1, grip_flip=15, seed=4),
        _make_demo(70, task_id=1, seed=5),
        _make_demo(55, task_id=1, grip_flip=25, seed=6),
    ]
    return NodeTable.from_demos(demos, task_names=["taskA", "taskB"], coarsen=coarsen)


@pytest.fixture(scope="module")
def synth() -> NodeTable:
    return _synthetic_table()


# ==================================================================================================
# real-graph skip marker (tests/test_builders.py's skip_no_data pattern), fixed path (see module
# docstring for why this is NOT Paths.graph("long"))
# ==================================================================================================
_REPO = Path(__file__).resolve().parents[1]
_REAL_GRAPH_DIR = _REPO / "outputs" / "long" / "frozen_learn1" / "artifacts"
HAS_REAL_GRAPH = (_REAL_GRAPH_DIR / schema.NODES_NPZ).exists()
skip_no_real_graph = pytest.mark.skipif(not HAS_REAL_GRAPH, reason=f"no real graph at {_REAL_GRAPH_DIR}")


@pytest.fixture(scope="module")
def real_nodes() -> NodeTable:
    return NodeTable.load(_REAL_GRAPH_DIR)


@pytest.fixture(scope="module")
def real_spacing(real_nodes: NodeTable) -> float:
    """The expensive O(V^2) computation, cached once per test session."""
    return measure_inter_demo_spacing(real_nodes)


def _assert_json_roundtrip(obj) -> None:
    dumped = json.dumps(obj)
    assert json.loads(dumped) is not None


# ==================================================================================================
# synthetic: shape / dtype / JSON-serialisability
# ==================================================================================================
def test_spacing_is_positive_scalar(synth):
    s = measure_inter_demo_spacing(synth)
    assert isinstance(s, float)
    assert s > 0.0
    _assert_json_roundtrip({"spacing": s})


@skip_no_real_graph
def test_spacing_reproduces_documented_value(real_spacing):
    """Measured 0.0883 rad on the real long graph (500 demos, 28,476 nodes) -- see
    measure_inter_demo_spacing's own docstring."""
    assert real_spacing == pytest.approx(0.0883, abs=1e-3)


@skip_no_real_graph
def test_all_real_graph_outputs_json_serialisable(real_spacing):
    _assert_json_roundtrip({"inter_demo_spacing": real_spacing})
