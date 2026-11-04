"""The joint FCSR objective as a Task: what it declares, and that the runner acts on it."""

from __future__ import annotations

import pytest
import torch

from robot_ood.data.store import ArtifactStore
from robot_ood.data import schema
from robot_ood.train.fcsr import CORRECTOR_START_EPOCH, FcsrTrainSpec, lambdas
from robot_ood.train.runner import MissingRequirement, Runner, RunSpec
from robot_ood.train.task import Task
from robot_ood.train.tasks.fcsr import FcsrTask


def _run(task, store):
    return Runner(RunSpec(epochs=1, save=False)).run(
        task, store, cfg=None, device=torch.device("cpu"))


def test_it_satisfies_the_task_protocol():
    assert isinstance(FcsrTask(), Task)


def test_it_declares_the_graph_its_segment_memory_and_the_chunk_blend_inputs():
    from robot_ood.models.fusion.target import ACTIONS_NPZ

    artifacts = set(FcsrTask().requires().artifacts)
    # HEAD_NPZ, not the skew sidecar: the phase residual rides inside the locator checkpoint.
    assert {schema.NODES_NPZ, schema.EDGES_NPZ, schema.SEGMENTS_NPZ, schema.HEAD_NPZ,
            ACTIONS_NPZ, schema.ACTION_SCALE_JSON} == artifacts


def test_it_has_no_upstream_task_left_to_wait_for():
    assert FcsrTask().requires().upstream == ()


def test_it_names_every_missing_input_before_it_loads_anything(tmp_path):
    with pytest.raises(MissingRequirement) as exc:
        _run(FcsrTask(), ArtifactStore(tmp_path))
    for name in (schema.NODES_NPZ, schema.SEGMENTS_NPZ, schema.ACTION_SCALE_JSON):
        assert name in str(exc.value)


def test_it_selects_no_epoch():
    assert FcsrTask().select({"corr_mse_learned": 0.25}) is None


def test_the_schedule_admits_the_corrector_once_tier1_has_settled():
    assert lambdas(0) == (1.0, 0.0)
    assert lambdas(CORRECTOR_START_EPOCH - 1) == (1.0, 0.0)
    assert lambdas(CORRECTOR_START_EPOCH) == (0.5, 1.0)


def test_it_carries_the_spec_it_was_built_with():
    assert FcsrTask(FcsrTrainSpec(epochs=3)).spec.epochs == 3


def test_it_carries_the_ablation_it_was_built_with():
    from robot_ood.core.config import Ablation

    assert FcsrTask().spec.ablation == Ablation()
    assert FcsrTask(FcsrTrainSpec(ablation=Ablation(memory=True))).spec.ablation.memory


def test_without_the_graph_the_batch_reads_no_belief_and_a_zero_gate():
    from robot_ood.core.config import Ablation
    from robot_ood.models.fusion.corrector import N_GATE
    from robot_ood.train.types import QueryBatch

    task = FcsrTask(FcsrTrainSpec(ablation=Ablation(graph=True)))
    task.ablation, task.gate_dim = task.spec.ablation, N_GATE
    task._locator_view = lambda qb: pytest.fail("locator read")
    qh = torch.zeros(5, 4, 3)
    batch = QueryBatch(qh=qh, vh=qh, gh=qh[..., 0], tgt=None)
    out, gate = task._with_belief(batch)
    assert out is batch and out.task_p is None
    assert gate.shape == (5, N_GATE) and torch.all(gate == 0)


def test_the_gate_leaves_the_querys_own_demonstration_out_of_the_node_set():
    """The off-manifold distance handed to the trust head must skip the query's own demo, as
    retrieval does: with it in, every training window sits a node spacing from the graph
    (0.03 rad on g4) while a deployed one sits where another demonstration is (0.06 median,
    0.25 at the 95th percentile), and the head has never seen the values it is asked to read.

    Asserted on the source because the failure is a MISSING argument, and a behavioural test would
    have to build a graph and run the locator to see it.
    """
    import inspect

    from robot_ood.train.tasks.fcsr import FcsrTask

    assert "with_owner=True" in inspect.getsource(FcsrTask._build_gate)
    assert "point_owner=own" in inspect.getsource(FcsrTask._locator_view)
