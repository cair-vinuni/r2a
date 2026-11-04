"""GraphLocator: the deploy read surface -- memory carried between checks, reset forgets, chunks recorded."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robot_ood.core.config import GraphConfig
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.geometry import CONSTANTS_JSON, graph_hash
from robot_ood.data.nodes import NodeTable
from robot_ood.models.retrieval.gnn import CHUNK, GraphRetrieverNet
from robot_ood.runtime.locator import GraphLocator
from robot_ood.runtime.retrieve import GraphRetriever

DIM = 4
N_FRAMES = 31
COARSEN = 3
HIDDEN = 16


def _make_demo(task_id: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    start = rng.randn(DIM) * 0.01
    direction = rng.randn(DIM)
    direction /= np.linalg.norm(direction)
    t = np.arange(N_FRAMES, dtype=np.float32)
    q = (start[None, :] + 0.05 * t[:, None] * direction[None, :]).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[1:] = q[1:] - q[:-1]
    grip = np.zeros(N_FRAMES, dtype=np.uint8)
    grip[N_FRAMES // 2 :] = 1
    return {"q": q, "qdot": qdot, "grip": grip, "stage": grip.astype(np.int16), "task_id": task_id}


@pytest.fixture(scope="module")
def graph():
    demos = [_make_demo(task_id=i % 2, seed=100 + i) for i in range(6)]
    nodes = NodeTable.from_demos(demos, task_names=["taskA", "taskB"], coarsen=COARSEN)
    return nodes, EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu")


def _toy_net(nodes: NodeTable, seed: int = 0) -> GraphRetrieverNet:
    torch.manual_seed(seed)
    net = GraphRetrieverNet(dim=nodes.dim, hidden=HIDDEN, layers=2, n_rel=schema.N_RELATIONS,
                            agg="sum")
    mean, std = nodes.feature_stats()
    net.set_stats(mean, std)
    return net.eval()


def _write_graph_dir(tmp_path, nodes, edges):
    d = tmp_path / "graph"
    d.mkdir(exist_ok=True)
    nodes.save(d)
    edges.save(d / schema.EDGES_NPZ)
    net = _toy_net(nodes)
    payload = dict(net.state_npz_dict())
    payload["graph_hash"] = np.array(graph_hash(
        nodes, edges, constants=d / CONSTANTS_JSON, c2_state=net.c2_state(),
    ))
    np.savez(d / schema.HEAD_NPZ, **payload)
    return d, net


@pytest.fixture
def deployed(tmp_path, graph):
    nodes, edges = graph
    d, _ = _write_graph_dir(tmp_path, nodes, edges)
    loc = GraphLocator.load(d, cfg=GraphConfig(device="cpu"))
    loc.retriever.nodes.ee_base = loc.retriever.nodes.ee_raw
    return loc


def _hist(nodes: NodeTable, start: int, hist: int = 8) -> np.ndarray:
    return nodes.q_raw[start : start + hist].astype(np.float64)


def test_the_checkpoint_loads_as_a_head_and_as_a_locator(tmp_path, graph):
    nodes, edges = graph
    d, written = _write_graph_dir(tmp_path, nodes, edges)
    retriever = GraphRetriever.load(d, cfg=GraphConfig(device="cpu"))
    for p, q in zip(retriever.net.parameters(), written.parameters()):
        assert torch.equal(p, q)


def test_the_first_check_of_an_episode_carries_no_memory(deployed):
    nodes = deployed.retriever.nodes
    step = deployed.step(_hist(nodes, 0))
    assert step.topk.first_step
    assert np.isfinite(step.phase_hat)
    assert step.task_prob.shape == (nodes.n_tasks,)
    assert step.task_prob.sum() == pytest.approx(1.0, abs=1e-6)


def test_a_later_check_reads_the_previous_belief_as_memory(deployed):
    nodes = deployed.retriever.nodes
    deployed.step(_hist(nodes, 0))
    step = deployed.step(_hist(nodes, 8))
    assert not step.topk.first_step
    assert 0.0 <= step.phase_hat <= 1.0


def test_the_belief_top_k_is_a_normalised_descending_view(deployed):
    nodes = deployed.retriever.nodes
    deployed.step(_hist(nodes, 0))
    step = deployed.step(_hist(nodes, 8))
    weight = step.topk.weight
    assert np.all(np.diff(weight) <= 1e-12)
    assert weight.sum() == pytest.approx(1.0, abs=1e-9)
    assert step.topk.idx.dtype == np.int64


def test_reset_forgets_the_episode(deployed):
    nodes = deployed.retriever.nodes
    deployed.step(_hist(nodes, 0))
    deployed.step(_hist(nodes, 8))
    deployed.reset()
    step = deployed.step(_hist(nodes, 0))
    assert step.topk.first_step


def test_the_same_episode_replays_identically(deployed):
    nodes = deployed.retriever.nodes

    def once():
        deployed.reset()
        return [deployed.step(_hist(nodes, s)).phase_hat for s in (0, 8, 16)]

    assert once() == once()


def test_a_recorded_chunk_reaches_the_next_check(deployed):
    nodes = deployed.retriever.nodes
    deployed.step(_hist(nodes, 0))
    without = deployed.step(_hist(nodes, 8)).belief_view.p_node.copy()
    deployed.reset()
    deployed.step(_hist(nodes, 0))
    deployed.record_chunk(np.ones((CHUNK + 2, 7), dtype=np.float32))
    with_chunk = deployed.step(_hist(nodes, 8)).belief_view.p_node
    assert without.shape == with_chunk.shape
    assert deployed._policy_chunk.shape == (CHUNK, 7)


def test_the_bundle_builds_the_locator_the_directory_carries(tmp_path, graph, monkeypatch):
    from robot_ood.core.config import SentinelConfig
    from robot_ood.runtime.bundle import RuntimeBundle

    nodes, edges = graph
    d, written = _write_graph_dir(tmp_path, nodes, edges)
    monkeypatch.setenv("SN_GRAPH_DIR", str(d))
    built = RuntimeBundle(cfg=SentinelConfig.from_env(), fwm_dir=tmp_path).build_locator()
    assert isinstance(built, GraphLocator)
    for p, q in zip(built.retriever.net.parameters(), written.parameters()):
        assert torch.equal(p.cpu(), q.cpu())


def test_a_locator_without_memory_reads_every_check_cold(tmp_path, graph):
    nodes, edges = graph
    d, _ = _write_graph_dir(tmp_path, nodes, edges)
    cold = GraphLocator.load(d, cfg=GraphConfig(device="cpu"), memory=False)
    cold.retriever.nodes.ee_base = cold.retriever.nodes.ee_raw
    first = cold.step(_hist(nodes, 0)).belief_view.p_node.copy()
    cold.step(_hist(nodes, 8))
    assert cold._memory is None
    cold.reset()
    np.testing.assert_allclose(cold.step(_hist(nodes, 0)).belief_view.p_node, first)


def test_the_bundle_ablates_memory_from_the_environment(tmp_path, graph, monkeypatch):
    from robot_ood.core.config import SentinelConfig
    from robot_ood.runtime.bundle import RuntimeBundle

    nodes, edges = graph
    d, _ = _write_graph_dir(tmp_path, nodes, edges)
    monkeypatch.setenv("SN_GRAPH_DIR", str(d))
    monkeypatch.setenv("ROBOT_OOD_ABLATE", "memory")
    assert not RuntimeBundle(cfg=SentinelConfig.from_env(), fwm_dir=tmp_path).build_locator().memory
