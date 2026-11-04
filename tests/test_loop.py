"""Tests for FCSR's C6 loop/orchestration additions:

  * robot_ood.train.loop.QuerySet threading seg_id/neg_segments end to end (from_queries -> batch).
  * robot_ood.models.retrieval.retriever.segment_graph_hash covering the segment configuration.
  * robot_ood.cli.train.TrainCommand accepting/dispatching the new segment/fusion objectives.

No HDF5/suite dependency -- small synthetic graphs, mirroring tests/test_train.py's own fixtures.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from robot_ood.core.schemas import SegmentTable
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.geometry import CONSTANTS_JSON
from robot_ood.data.nodes import NodeTable
from robot_ood.train.data import make_queries
from robot_ood.train.negatives import HardNegativeMiner
from robot_ood.train.tasks.query import QuerySet
from robot_ood.train.types import QuerySpec

D = 4
SEGMENT_LEN = 4
ACTION_DIM = 7


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


def _toy_graph(n_tasks: int = 2, demos_per_task: int = 4, length: int = 24, seed: int = 0):
    rng = np.random.RandomState(seed)
    demos = []
    for ti in range(n_tasks):
        base = rng.randn(D) * 2.0
        for di in range(demos_per_task):
            demos.append(_make_demo(ti, seed=1000 * ti + di + seed, n=length, dim=D, base=base))
    nodes = NodeTable.from_demos(demos, task_names=[f"task{i}" for i in range(n_tasks)], coarsen=1)
    edges = EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu")
    return nodes, edges


def _segment_table_from_nodes(nodes: NodeTable) -> SegmentTable:
    v = len(nodes)
    return SegmentTable(
        start_node=np.arange(v, dtype=np.int32), end_node=np.arange(v, dtype=np.int32),
        owner=np.asarray(nodes.owner, dtype=np.int32), task_id=np.asarray(nodes.task_id, dtype=np.int16),
        phase_start=np.asarray(nodes.phase, dtype=np.float32),
        phase_end=np.asarray(nodes.phase, dtype=np.float32),
        action_chunk=np.random.RandomState(0).randn(v, SEGMENT_LEN, ACTION_DIM).astype(np.float32),
        ee_start=np.zeros((v, 6), dtype=np.float32), ee_end=np.zeros((v, 6), dtype=np.float32),node_to_segment=np.arange(v, dtype=np.int32),
    )


# ======================================================================================================
# QuerySet.seg_id / neg_segments
# ======================================================================================================
def test_query_set_without_a_seg_miner_carries_all_minus_one_seg_id():
    nodes, _edges = _toy_graph()
    rng = np.random.RandomState(0)
    raw = make_queries(nodes, rng, 32, np.arange(nodes.n_demos), QuerySpec())
    qs = QuerySet.from_queries(raw, "cpu")
    assert qs.seg_id is not None
    assert np.all(qs.seg_id == -1)
    assert qs.neg_segments is None


def test_query_set_with_a_seg_miner_mines_negatives_for_segment_rows():
    nodes, _edges = _toy_graph()
    table = _segment_table_from_nodes(nodes)
    miner = HardNegativeMiner(table, dtw_thresh=0.0, phase_band=0.3)
    rng = np.random.RandomState(0)
    raw = make_queries(
        nodes, rng, 32, np.arange(nodes.n_demos),
        QuerySpec(segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0),
    )
    qs = QuerySet.from_queries(raw, "cpu", seg_miner=miner, n_seg_neg=5)
    assert qs.neg_segments.shape == (32, 5)
    assert np.all(qs.seg_id >= 0)


def test_batch_uploads_seg_id_and_neg_segments_as_device_tensors_sliced_to_the_batch():
    nodes, _edges = _toy_graph()
    table = _segment_table_from_nodes(nodes)
    miner = HardNegativeMiner(table, dtw_thresh=0.0, phase_band=0.3)
    rng = np.random.RandomState(0)
    raw = make_queries(
        nodes, rng, 16, np.arange(nodes.n_demos),
        QuerySpec(segments=table, segment_frac=1.0, entry_frac=0.0, entry_static_frac=0.0, drift_frac=0.0),
    )
    qs = QuerySet.from_queries(raw, "cpu", seg_miner=miner, n_seg_neg=4)
    idx = np.array([0, 2, 5])
    qb = qs.batch(idx)
    assert isinstance(qb.seg_id, torch.Tensor) and qb.seg_id.shape == (3,)
    assert isinstance(qb.neg_segments, torch.Tensor) and qb.neg_segments.shape == (3, 4)
    assert torch.equal(qb.seg_id, torch.as_tensor(qs.seg_id[idx]))


def test_ordinary_retrieval_query_set_is_unaffected_by_the_new_fields():
    """A run that never touches segments still uploads a plain, small seg_id=-1 column rather than
    changing shape or dtype of anything pre-existing."""
    nodes, _edges = _toy_graph()
    rng = np.random.RandomState(0)
    raw = make_queries(nodes, rng, 16, np.arange(nodes.n_demos), QuerySpec())
    qs = QuerySet.from_queries(raw, "cpu")
    qb = qs.batch(np.arange(4))
    assert qb.a_policy is None
    assert qb.seg_id.shape == (4,)
    assert torch.all(qb.seg_id == -1)


# ======================================================================================================
# segment_graph_hash
# ======================================================================================================
def test_segment_graph_hash_changes_with_segment_len(tmp_path):
    from robot_ood.models.retrieval.retriever import segment_graph_hash

    nodes, edges = _toy_graph()
    scorer_state = {"w": np.array([1.0, 2.0], dtype=np.float32)}
    constants_path = tmp_path / CONSTANTS_JSON
    h1 = segment_graph_hash(
        nodes, edges, constants_path, segment_len=16, seg_topk=8, scorer_state=scorer_state,
    )
    h2 = segment_graph_hash(
        nodes, edges, constants_path, segment_len=32, seg_topk=8, scorer_state=scorer_state,
    )
    h3 = segment_graph_hash(
        nodes, edges, constants_path, segment_len=16, seg_topk=16, scorer_state=scorer_state,
    )
    assert len({h1, h2, h3}) == 3


def test_segment_graph_hash_changes_with_the_scorer_checkpoint(tmp_path):
    from robot_ood.models.retrieval.retriever import segment_graph_hash

    nodes, edges = _toy_graph()
    seg_kw = dict(segment_len=16, seg_topk=8)
    constants_path = tmp_path / CONSTANTS_JSON
    h1 = segment_graph_hash(
        nodes, edges, constants_path, **seg_kw, scorer_state={"w": np.zeros(2, dtype=np.float32)},
    )
    h2 = segment_graph_hash(
        nodes, edges, constants_path, **seg_kw, scorer_state={"w": np.ones(2, dtype=np.float32)},
    )
    assert h1 != h2


def test_segment_graph_hash_deterministic():
    from robot_ood.models.retrieval.retriever import segment_graph_hash

    nodes, edges = _toy_graph()
    seg_kw = dict(segment_len=16, seg_topk=8)
    state = {"w": np.array([1.0], dtype=np.float32)}
    h1 = segment_graph_hash(nodes, edges, None, **seg_kw, scorer_state=state)
    h2 = segment_graph_hash(nodes, edges, None, **seg_kw, scorer_state=state)
    assert h1 == h2


# ======================================================================================================
# cli/train.py verb wiring
# ======================================================================================================
def _train_args(tmp_path, task, **over):
    """An argparse.Namespace as `train <task>` builds one."""
    ns = dict(
        suite="object", task=task, out=str(tmp_path / "outputs"), name="", seed=None, device=None,
        quiet=True, dry_run=False, force=False, resume=None,
    )
    return argparse.Namespace(**{**ns, **over})


def test_every_train_task_validates_against_a_built_graph(tmp_path):
    from robot_ood.cli.train import TRAIN_TASKS, TrainCommand

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    np.savez(graph_dir / schema.NODES_NPZ, dummy=np.zeros(1))
    for task in TRAIN_TASKS:
        TrainCommand(_train_args(tmp_path, task)).validate(
            "object", {"graph_dir": str(graph_dir)})


def test_an_unknown_train_task_is_an_argparse_error():
    """The verb list is argparse's, not a hand-checked string: a typo cannot reach resolve_config."""
    from robot_ood.cli.main import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "bogus", "--suite", "object"])


def test_train_artifacts_present_checks_that_task_own_outputs(tmp_path):
    from robot_ood.cli.train import TRAIN_TASKS, TrainCommand

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    for task, spec in TRAIN_TASKS.items():
        cmd = TrainCommand(_train_args(tmp_path, task))
        config = {"graph_dir": str(graph_dir)}
        assert cmd.artifacts_present("object", config) is False, task
        for name in spec.outputs:
            (graph_dir / name).write_bytes(b"\x00")
        assert cmd.artifacts_present("object", config) is True, task
        for name in spec.outputs:
            (graph_dir / name).unlink()


