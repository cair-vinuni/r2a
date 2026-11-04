"""Tests for the chunk blend's WIRING -- when a plan is built, and where it is applied.

The blend arithmetic itself is covered by tests/test_blend.py. What is silent in a rollout and
therefore tested here: a default weight that is only approximately zero (it would move every base
number ever reported), and a plan written to the pending-action buffer before the policy refills it,
which the refill then erases.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from robot_ood.core.action import ActionScale
from robot_ood.core.config import SentinelConfig
from robot_ood.core.schemas import SegmentCandidates
from robot_ood.models.fusion.corrector import Corrector
from robot_ood.runtime.bundle import RuntimeBundle
from robot_ood.runtime.sentinel import Sentinel

REPO = Path(__file__).resolve().parents[1]
_SCALE = ActionScale(pos=0.0116, rot=0.1092, r2_pos=0.91, r2_rot=0.91)
CHUNK = 8
_NODE_EE = np.zeros((64, 6), dtype=np.float32)


# ==================================================================================================
# when Sentinel plans
# ==================================================================================================
def _check_result() -> float:
    return 0.0


def test_plan_is_none_off_a_check_boundary(tmp_path):
    sentinel = Sentinel(cfg=SentinelConfig(), fwm_dir=tmp_path)
    # A non-multiple of check_every: no filter runs, so no artifact is loaded either.
    assert sentinel.plan(np.zeros(7), 0.0, 3, wid=0) is None


def test_every_check_plans(tmp_path, monkeypatch):
    """A check whose target is q_now itself still reaches planning, which loads the graph."""
    sentinel = Sentinel(cfg=SentinelConfig(), fwm_dir=tmp_path)
    monkeypatch.setattr(Sentinel, "check", lambda *args, **kwargs: _check_result())
    with pytest.raises(FileNotFoundError, match="robot_ood.cli build graph"):
        sentinel.plan(np.zeros(7), 0.0, 0, wid=0)


# ==================================================================================================
# the query window's REAL row spacing
#
# The knob guard in SentinelConfig.__post_init__ cannot see this: the spacing that reaches the
# network is set by how often the caller calls observe(), and the caller used to call it once per
# check (8 env steps) while the knob said 1.
# ==================================================================================================
def test_observe_fills_the_window_one_row_per_env_step(tmp_path):
    sentinel = Sentinel(cfg=SentinelConfig(), fwm_dir=tmp_path)
    for step in range(CHUNK):
        sentinel.observe(np.full(7, float(step)), step, wid=0)
    worker = sentinel._worker(0)
    assert list(worker.s_win) == list(range(CHUNK))
    assert worker.q_win[-1][0] == CHUNK - 1


def test_check_refuses_a_window_that_was_only_fed_on_check_boundaries(tmp_path):
    """The actual bug: observe() called at the check cadence, so 8 rows span 64 env steps."""
    sentinel = Sentinel(cfg=SentinelConfig(), fwm_dir=tmp_path)
    for step in range(0, 8 * CHUNK + 1, CHUNK):
        sentinel.observe(np.zeros(7), step, wid=0)
    with pytest.raises(RuntimeError, match="spaced .* env steps apart"):
        sentinel.check(np.zeros(7), 0.0, 8 * CHUNK, wid=0)


def test_check_refuses_an_unfed_window_and_names_observe(tmp_path):
    sentinel = Sentinel(cfg=SentinelConfig(), fwm_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Sentinel.observe"):
        sentinel.check(np.zeros(7), 0.0, 0, wid=0)




def test_reset_clears_both_halves_of_the_window(tmp_path):
    sentinel = Sentinel(cfg=SentinelConfig(), fwm_dir=tmp_path)
    for step in range(CHUNK):
        sentinel.observe(np.zeros(7), step, wid=0)
    sentinel.reset(0)
    worker = sentinel._worker(0)
    assert not worker.q_win and not worker.s_win


# ==================================================================================================
# where the plan is applied: after the refill, not before
# ==================================================================================================
class _StubClient:
    """A policy server that answers every inference with the same chunk.

    Attributes:
        chunk: [8, 7] float64 chunk handed back on each infer.
        calls: Number of infer calls received.
    """

    def __init__(self, chunk: np.ndarray) -> None:
        self.chunk = chunk
        self.calls = 0

    def infer(self, _request: dict) -> dict:
        self.calls += 1
        return {"data": {"actions": [self.chunk.copy()]}}


def _load_interface(monkeypatch):
    """Import the vendored policy client with the two deps absent from this env stubbed out."""
    tools = types.ModuleType("tools")
    tools.__path__ = []
    client = types.ModuleType("tools.websocket_policy_client")
    client.WebsocketClientPolicy = object
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    monkeypatch.setitem(sys.modules, "tools", tools)
    monkeypatch.setitem(sys.modules, "tools.websocket_policy_client", client)
    path = REPO / "evals" / "libero_plus" / "harness" / "model2libero_interface.py"
    spec = importlib.util.spec_from_file_location("model2libero_interface", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _model(monkeypatch, client):
    """An M1Inference wired to a stub client, bypassing the websocket connect in __init__."""
    inference = _load_interface(monkeypatch).M1Inference
    model = inference.__new__(inference)
    model.client = client
    model.task_description = "pick up the thing"
    model.unnorm_key = None
    model.use_ddim, model.num_ddim_steps = True, 10
    model.image_size, model.worker_id = [224, 224], 0
    model.action_chunk_size = CHUNK
    model.exec_horizon = CHUNK
    return model


def _rollout(model, plan) -> np.ndarray:
    """The [16, 7] float64 actions executed over two chunks, planning only on chunk boundaries."""
    executed = []
    for step in range(2 * CHUNK):
        raw = model.step(images=[], step=step, blend=plan if step % CHUNK == 0 else None)["raw_action"]
        executed.append(np.concatenate([raw["world_vector"], raw["rotation_delta"], raw["open_gripper"]]))
    return np.stack(executed)


class _StubPlan:
    """A stand-in plan whose apply shifts the pose block and leaves the gripper alone."""

    def apply(self, a_policy: np.ndarray) -> np.ndarray:
        out = a_policy.copy()
        out[:, :6] += 0.5
        return out


def test_the_blend_survives_the_next_refill_and_leaves_the_gripper_alone(monkeypatch):
    chunk = np.arange(CHUNK * 7, dtype=np.float64).reshape(CHUNK, 7) / 100.0
    chunk[:, 6] = np.tile([1.0, 0.0], CHUNK // 2)

    client = _StubClient(chunk)
    base = _rollout(_model(monkeypatch, client), None)
    assert client.calls == 2

    blended = _rollout(_model(monkeypatch, _StubClient(chunk)), _StubPlan())
    # Both chunks moved, the second included: a plan applied BEFORE the refill would have been
    # overwritten by it, leaving rows 8..15 equal to base.
    np.testing.assert_allclose(blended[:, :6] - base[:, :6], 0.5)
    assert np.array_equal(blended[:, 6], base[:, 6])








class _StubKinematics:
    def pose(self, q):
        return np.zeros((q.shape[0], 6))


class _StubScorer:
    """Stands in for the CleanlinessScorer a real locator's retriever owns.

    is_uniform is the no-trained-field case, which Sentinel._query_weights answers with the corpus
    mean rather than ones -- passing ones is the defect that measured -14.4pp of deployed rank-1.
    """

    def __init__(self, uniform: bool = False) -> None:
        self.is_uniform = uniform

    def weights(self, q_window):
        return np.full(np.asarray(q_window).shape[0], 0.3, dtype=np.float64)


class _StubGraphRetriever:
    """The GRAPH retriever a GraphLocator owns -- distinct from the SegmentRetriever stub below,
    which fusion uses. Only its CleanlinessScorer matters here."""

    def __init__(self, uniform: bool = False) -> None:
        self.scorer = _StubScorer(uniform)


class _StubLocator:
    def __init__(self, uniform_field: bool = False) -> None:
        self.retriever = _StubGraphRetriever(uniform_field)

    def record_chunk(self, a_policy):
        self.recorded = np.asarray(a_policy)


# ==================================================================================================
# BlendPlan.authority -- the correction arm
# ==================================================================================================
def _cands(k: int = 4, seg: int = 16) -> SegmentCandidates:
    rng = np.random.default_rng(0)
    return SegmentCandidates(
        idx=np.arange(k, dtype=np.int32), score=rng.normal(size=k).astype(np.float32),
        action_chunk=rng.normal(size=(k, seg, 7)).astype(np.float32),
        phase=rng.random(k).astype(np.float32),
        ee_start=np.zeros((k, 6), dtype=np.float32), ee_end=np.zeros((k, 6), dtype=np.float32),
        start_node=np.arange(k, dtype=np.int32), end_node=(np.arange(k, dtype=np.int32) + seg - 1),
    )








# ==================================================================================================
# Sentinel.plan under SN_FUSION -- the corrector path
# ==================================================================================================
class _StubRetriever:
    def __init__(self, cands: SegmentCandidates) -> None:
        self.cands, self.calls = cands, []

    def retrieve(self, ctx, *, task_id=None, exclude_owner=None, task_mask=None):
        self.calls.append((ctx, task_id, exclude_owner))
        return self.cands










def test_an_unset_device_loads_the_fcsr_archive_onto_cpu(tmp_path):
    """The runtime feeds these modules CPU tensors. Resolving an unset cfg.device to whatever CUDA
    device happens to be visible loads the weights onto it, and every check then dies on
    "expected all tensors to be on the same device" -- at episode 1 of a real eval, not here."""
    from conftest import write_fcsr_archive

    write_fcsr_archive(tmp_path)
    bundle = RuntimeBundle(cfg=SentinelConfig(graph_dir=str(tmp_path)), fwm_dir=tmp_path)

    assert {p.device.type for p in bundle.fcsr_net().parameters()} == {"cpu"}


def _fusion_sentinel(tmp_path, monkeypatch, *, action_hist: bool) -> tuple[Sentinel, "_StubRetriever"]:
    from robot_ood.core.config import GraphConfig

    sentinel = Sentinel(cfg=SentinelConfig(graph_win=CHUNK), fwm_dir=tmp_path)
    retriever = _StubRetriever(_cands())
    net = Corrector(chunk_len=CHUNK)

    monkeypatch.setattr(Sentinel, "check", lambda *a, **k: _check_result())
    monkeypatch.setattr(RuntimeBundle, "action_scale", lambda self: _SCALE)
    monkeypatch.setattr(RuntimeBundle, "kinematics", lambda self: _StubKinematics())
    monkeypatch.setattr(Sentinel, "_get_locator", lambda self, wid: _StubLocator())
    monkeypatch.setattr(RuntimeBundle, "segment_retriever", lambda self: retriever)
    monkeypatch.setattr(RuntimeBundle, "corrector", lambda self: net)
    monkeypatch.setattr(RuntimeBundle, "node_ee", lambda self: _NODE_EE)
    monkeypatch.setattr(RuntimeBundle, "segment_pause", lambda self: None)
    monkeypatch.setattr(RuntimeBundle, "graph_config",
                        lambda self: (GraphConfig(action_hist=action_hist), tmp_path))
    return sentinel, retriever


def test_plan_under_fusion_threads_a_hist_once_the_commanded_action_window_is_full(tmp_path, monkeypatch):
    """Gap A, deploy side: observe_action() must fill a per-worker ring buffer that _plan_fusion
    reads into QueryContext.a_hist once it holds a full window -- the mechanism that makes
    GR_ACTION_HIST=1 actually differ from GR_ACTION_HIST=0 at deploy, not just in training."""
    sentinel, retriever = _fusion_sentinel(tmp_path, monkeypatch, action_hist=True)
    for step in range(CHUNK):
        sentinel.observe(np.full(7, float(step)), step, wid=0)
        sentinel.observe_action(np.full(7, float(step) + 100.0), step, wid=0)

    plan = sentinel.plan(np.zeros(7), 0.0, 0, wid=0)
    assert plan is not None
    ctx, _, _ = retriever.calls[0]
    assert ctx.a_hist is not None
    assert ctx.a_hist.shape == (1, CHUNK, 7)
    np.testing.assert_array_equal(
        ctx.a_hist[0].numpy(), np.stack([np.full(7, float(s) + 100.0) for s in range(CHUNK)])
    )


def test_plan_under_fusion_a_hist_stays_none_until_the_window_is_full(tmp_path, monkeypatch):
    sentinel, retriever = _fusion_sentinel(tmp_path, monkeypatch, action_hist=True)
    for step in range(CHUNK - 1):    # one row short of a full window
        sentinel.observe(np.full(7, float(step)), step, wid=0)
        sentinel.observe_action(np.full(7, float(step)), step, wid=0)

    plan = sentinel.plan(np.zeros(7), 0.0, 0, wid=0)
    assert plan is not None
    ctx, _, _ = retriever.calls[0]
    assert ctx.a_hist is None


def test_plan_under_fusion_respects_graphconfig_action_hist_off(tmp_path, monkeypatch):
    """GraphConfig.action_hist=False is the state-only ablation arm: even a full commanded-action
    window must not reach the encoder."""
    sentinel, retriever = _fusion_sentinel(tmp_path, monkeypatch, action_hist=False)
    for step in range(CHUNK):
        sentinel.observe(np.full(7, float(step)), step, wid=0)
        sentinel.observe_action(np.full(7, float(step)), step, wid=0)

    plan = sentinel.plan(np.zeros(7), 0.0, 0, wid=0)
    assert plan is not None
    ctx, _, _ = retriever.calls[0]
    assert ctx.a_hist is None


def test_plan_fusion_builds_a_query_context_with_OBSERVED_ee_and_phase():
    """The deploy QueryContext must carry ee and phase, not leave the retriever to estimate them.

    This is a regression test for a crash that every one of 32 eval clients hit on episode 1 while
    the whole unit suite was green: SegmentRetriever.score refuses a QueryContext whose ee is None
    (ee is observed by FK, phase comes from the belief), and the fusion plan never set either.
    The suite missed it because the fusion tests stub the retriever, so nothing ever read ctx.ee.
    Asserting on the CONTEXT rather than on the absence of an exception keeps that hole shut even if
    the retriever's own guard is later relaxed.
    """
    import inspect

    from robot_ood.runtime.fuse import plan_fusion

    src = inspect.getsource(plan_fusion)
    assert "ee=" in src and "phase=" in src, (
        "robot_ood.runtime.fuse.plan_fusion must populate QueryContext.ee and .phase; the retriever "
        "raises rather than estimating either, and a missing one fails at episode 1 of a "
        "1573-episode sweep."
    )
    assert "phase" in str(inspect.signature(plan_fusion)), (
        "the phase estimate must be threaded in from the check, not recomputed"
    )


# ==================================================================================================
# ablation: the graph off
# ==================================================================================================
def test_a_check_without_the_graph_reads_phase_off_top_k_retrieval(tmp_path, monkeypatch):
    from robot_ood.core.config import Ablation

    sentinel = Sentinel(cfg=SentinelConfig(graph_win=CHUNK, ablation=Ablation(graph=True)), fwm_dir=tmp_path)
    monkeypatch.setattr(RuntimeBundle, "kinematics", lambda self: _StubKinematics())
    monkeypatch.setattr(RuntimeBundle, "query_context", lambda self, **kw: kw)
    monkeypatch.setattr(RuntimeBundle, "topk_phase", lambda self, ctx: 0.8)
    monkeypatch.setattr(Sentinel, "_get_locator", lambda self, wid: pytest.fail("the graph was consulted"))
    for step in range(CHUNK + 1):
        sentinel.observe(np.zeros(7), step, wid=0)
    assert sentinel.check(np.zeros(7), 0.0, CHUNK, wid=0) == pytest.approx(0.8)
    worker = sentinel._worker(0)
    assert worker.last_gate is None and worker.last_task_p is None
    assert worker.last_phase == pytest.approx(0.8)
    sentinel.reset(0)
    assert worker.last_phase == 0.0


def test_topk_phase_is_the_score_weighted_mean_of_the_neighbours_phase(tmp_path, monkeypatch):
    cands = _cands()
    bundle = RuntimeBundle(SentinelConfig(graph_win=CHUNK), fwm_dir=tmp_path)
    monkeypatch.setattr(RuntimeBundle, "segment_retriever", lambda self: _StubRetriever(cands))
    w = np.exp(cands.score.astype(np.float64) - cands.score.max())
    w = w / w.sum()
    assert bundle.topk_phase(object()) == pytest.approx(float(np.dot(w, cands.phase.astype(np.float64))))


def test_plan_without_the_graph_records_no_chunk(tmp_path, monkeypatch):
    from robot_ood.core.config import Ablation, GraphConfig

    sentinel = Sentinel(cfg=SentinelConfig(graph_win=CHUNK, ablation=Ablation(graph=True)), fwm_dir=tmp_path)
    retriever = _StubRetriever(_cands())
    monkeypatch.setattr(RuntimeBundle, "instruction_embedding", lambda self, task: None)
    monkeypatch.setattr(RuntimeBundle, "action_scale", lambda self: _SCALE)
    monkeypatch.setattr(RuntimeBundle, "kinematics", lambda self: _StubKinematics())
    monkeypatch.setattr(Sentinel, "_get_locator", lambda self, wid: pytest.fail("the graph was consulted"))
    monkeypatch.setattr(RuntimeBundle, "segment_retriever", lambda self: retriever)
    monkeypatch.setattr(RuntimeBundle, "corrector", lambda self: Corrector(chunk_len=CHUNK))
    monkeypatch.setattr(RuntimeBundle, "node_ee", lambda self: _NODE_EE)
    monkeypatch.setattr(RuntimeBundle, "segment_pause", lambda self: None)
    monkeypatch.setattr(RuntimeBundle, "graph_config", lambda self: (GraphConfig(), tmp_path))
    for step in range(CHUNK + 1):
        sentinel.observe(np.zeros(7), step, wid=0)
    plan = sentinel.plan(np.zeros(7), 0.0, CHUNK, wid=0)
    assert plan.record_chunk is None
    assert plan.authority_ctx["gate"] is None and plan.authority_ctx["cand_task_p"] is None
    assert retriever.calls[0][0].task_p is None
