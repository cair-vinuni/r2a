"""Tests for robot_ood.build.calibration — C0, the Calibrator.

Two families, mirroring tests/test_corpus.py:
  * SYNTHETIC (_synthetic_table, the same _make_demo idiom) — the measuring functions and the
    constants.json contract, always run, no real artifacts required.
  * REAL-GRAPH (@skip_no_real_graph) — the two gates of the plan's section 3.4 against the fixed
    reference artifacts: constants.json regenerates BIT-FOR-BIT from the same corpus, and the
    numbers already known for this corpus (the action-unit scales) come back inside rounding.
    The seed/posterior support measurement is not exercised there: it costs ~2 minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_ood.core.constants import ASSUMED, MEASURED, SCHEMA_VERSION, Constant, Constants
from robot_ood.data.libero import DemoCorpus
from robot_ood.build.measure import (
    CalibrationSpec,
    DilationReach,
    EntryPerturbation,
    FutureTemp,
    NeighbourSpacing,
    SeedSupport,
    SegmentEntryStratum,
    SegmentHardNegatives,
    SegmentLen,
    SegTier1M,
)
from robot_ood.core.schemas import ACTION_DIM, SEG_TIER1_M, SEG_TOPK, SEGMENT_LEN
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.geometry import CONSTANTS_JSON, graph_hash
from robot_ood.data.nodes import NodeTable
from robot_ood.models.retrieval.gnn import GraphRetrieverNet

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GRAPH = REPO_ROOT / "outputs" / "long" / "artifacts"
skip_no_real_graph = pytest.mark.skipif(
    not (REAL_GRAPH / schema.NODES_NPZ).exists(),
    reason=f"needs the reference graph at {REAL_GRAPH}",
)

D = 3


# ==================================================================================================
# synthetic corpus (tests/test_corpus.py's _make_demo / _synthetic_table)
# ==================================================================================================
def _make_demo(t: int, task_id: int = 0, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    q = np.cumsum(rng.randn(t, D).astype(np.float32) * 0.02, axis=0).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[:-1] = q[1:] - q[:-1]
    qdot[-1] = qdot[-2] if t > 1 else 0.0
    return {
        "q": q, "qdot": qdot, "grip": np.zeros(t, dtype=np.uint8),
        "stage": np.zeros(t, dtype=np.int16), "task_id": task_id,
    }


@pytest.fixture(scope="module")
def synth() -> NodeTable:
    demos = [
        _make_demo(45, task_id=0, seed=1), _make_demo(60, task_id=0, seed=2),
        _make_demo(50, task_id=0, seed=3), _make_demo(40, task_id=1, seed=4),
        _make_demo(70, task_id=1, seed=5), _make_demo(55, task_id=1, seed=6),
    ]
    return NodeTable.from_demos(demos, task_names=["taskA", "taskB"], coarsen=3)


def _make_demo_with_actions(t: int, task_id: int = 0, seed: int = 0) -> dict:
    """_make_demo plus a genuine (non-zero) recorded action stream, for the FCSR measurements below
    -- the plain fixture's all-zero 'a' would make SegmentLen/FutureTemp/SegmentHardNegatives
    degenerate rather than exercised."""
    demo = _make_demo(t, task_id=task_id, seed=seed)
    rng = np.random.RandomState(seed + 100)
    demo["a"] = (rng.randn(t, ACTION_DIM).astype(np.float32) * 0.1)
    return demo


@pytest.fixture(scope="module")
def synth_seg() -> NodeTable:
    demos = [
        _make_demo_with_actions(45, task_id=0, seed=1), _make_demo_with_actions(60, task_id=0, seed=2),
        _make_demo_with_actions(50, task_id=0, seed=3), _make_demo_with_actions(40, task_id=1, seed=4),
        _make_demo_with_actions(70, task_id=1, seed=5), _make_demo_with_actions(55, task_id=1, seed=6),
    ]
    table = NodeTable.from_demos(demos, task_names=["taskA", "taskB"], coarsen=3)
    table.ee_base = np.concatenate([table.q_raw, table.q_raw], axis=1)
    return table


def _constants(**values: float) -> Constants:
    return Constants(
        entries={k: Constant(v, MEASURED, "test") for k, v in values.items()},
        corpus={"suite": "synthetic"},
    )


# ==================================================================================================
# the constants.json contract
# ==================================================================================================
def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        Constant(1.0, "guessed", "test")


def test_the_content_hash_ignores_the_corpus_block():
    """Two machines calibrating the same corpus differ in their paths and nothing else."""
    a = Constants(entries={"x": Constant(1.0, MEASURED, "p")}, corpus={"graph_dir": "/a"})
    b = Constants(entries={"x": Constant(1.0, MEASURED, "p")}, corpus={"graph_dir": "/b"})
    assert a.content_hash == b.content_hash
    assert a.content_hash != _constants(x=1.0, y=2.0).content_hash


def test_the_content_hash_moves_with_a_value():
    assert _constants(x=1.0).content_hash != _constants(x=1.0000001).content_hash


def test_round_trip_through_json_is_byte_identical(tmp_path):
    written = _constants(kernel_bw=0.0883, topm=32.0).to_json(tmp_path)
    first = written.read_bytes()
    reloaded = Constants.from_json(tmp_path)
    assert reloaded.content_hash == json.loads(first)["content_hash"]
    assert reloaded.to_json(tmp_path).read_bytes() == first


def test_a_hand_edited_file_is_refused(tmp_path):
    path = _constants(kernel_bw=0.0883).to_json(tmp_path)
    data = json.loads(path.read_text())
    data["constants"]["kernel_bw"]["value"] = 0.15          # the edit a hash exists to catch
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="content_hash mismatch"):
        Constants.from_json(path)


def test_an_unknown_schema_version_is_refused(tmp_path):
    path = _constants(kernel_bw=0.0883).to_json(tmp_path)
    data = json.loads(path.read_text())
    data["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema_version"):
        Constants.from_json(path)


def test_of_kind_names_what_is_still_a_hand_constant():
    entries = {
        **SegTier1M().constants(),
        "kernel_bw": Constant(value=0.0883, kind=MEASURED, procedure="lodo median"),
    }
    constants = Constants(entries=entries)
    assert constants.of_kind(ASSUMED) == ("seg_tier1_m",)


# ==================================================================================================
# the measuring functions
# ==================================================================================================
def test_neighbour_spacing_is_the_corpus_lodo_median(synth):
    from robot_ood.data.corpus import measure_inter_demo_spacing

    entries = NeighbourSpacing(synth).constants()
    assert entries["kernel_bw"].value == measure_inter_demo_spacing(synth)
    # One measurement, both consumers: the seed kernel and the sibling/align Gaussian.
    assert entries["edge_sigma_q"].value == entries["kernel_bw"].value


def test_entry_perturbation_is_the_multiplier_of_its_own_floor(synth):
    from robot_ood.data.geometry import entry_perturbation_floor

    spec = CalibrationSpec()
    entries = EntryPerturbation(synth, spec).constants()
    stats = entries["entry_perturb_lo"].stats
    assert entries["entry_perturb_lo"].value == pytest.approx(spec.entry_lo_mult * stats["p50"])
    assert stats["p50"] == pytest.approx(entry_perturbation_floor(synth)["p50"])
    assert 0.0 < entries["entry_perturb_lo"].value


def test_the_dilation_reach_constraint_is_checked_not_chosen(synth):
    """A 45-70 frame synthetic demo is well inside the shipped schedule's 240-frame reach."""
    reach = DilationReach(synth)
    assert reach.satisfied
    stats = reach.constants()["reach_raw_frames"].stats
    assert stats["satisfied"] is True
    assert reach.constants()["reach_raw_frames"].value == schema.DEFAULTS.reach_raw_frames


def test_the_dilation_reach_constraint_fails_a_schedule_that_cannot_cross_a_demo(synth):
    import dataclasses

    shallow = dataclasses.replace(
        schema.DEFAULTS, network=dataclasses.replace(schema.DEFAULTS.network, layers=1),
        build=dataclasses.replace(schema.DEFAULTS.build, coarsen=1),
    )
    assert not DilationReach(synth, shallow).satisfied


# ==================================================================================================
# C0 additions: FCSR (C1 segment memory) calibration
# ==================================================================================================
def test_segment_len_is_measured_not_defaulted(synth_seg):
    spec = CalibrationSpec()
    entry = SegmentLen(synth_seg, spec).constants()["segment_len"]
    assert entry.kind == MEASURED
    assert entry.stats["architectural_default"] == SEGMENT_LEN
    assert 1 <= entry.value <= spec.direction_max_lag


def test_segment_len_reflects_the_coherence_threshold(synth_seg):
    loose = CalibrationSpec(direction_coherence_thresh=-1.0)  # never decorrelates
    entry = SegmentLen(synth_seg, loose).constants()["segment_len"]
    assert entry.value == loose.direction_max_lag


def test_future_temp_hits_its_target_perplexity(synth_seg):
    spec = CalibrationSpec()
    entry = FutureTemp(synth_seg, spec).constants()["future_temp"]
    assert entry.kind == MEASURED
    assert entry.value > 0.0
    assert entry.stats["achieved_perplexity"] == pytest.approx(entry.stats["target_perplexity"], rel=1e-3)


def test_future_temp_target_is_seg_topk_by_default(synth_seg):
    entry = FutureTemp(synth_seg, CalibrationSpec()).constants()["future_temp"]
    assert entry.stats["target_perplexity"] == float(SEG_TOPK)


def test_seg_hard_neg_thresh_is_the_p90_of_cross_task_same_phase_dtw(synth_seg):
    spec = CalibrationSpec()
    entry = SegmentHardNegatives(synth_seg, spec).constants()["seg_hard_neg_thresh"]
    assert entry.kind == MEASURED
    assert entry.value == pytest.approx(entry.stats["p90"])
    assert entry.stats["p10"] <= entry.stats["p50"] <= entry.stats["p90"] <= entry.stats["p99"]
    assert entry.value >= 0.0


def test_seg_entry_stratum_frac_is_oversampled_natural_share(synth_seg):
    from robot_ood.data.segments import SegmentBuilder

    spec = CalibrationSpec()
    entry = SegmentEntryStratum(synth_seg, spec).constants()["seg_entry_stratum_frac"]
    natural = entry.stats["natural_share"]
    segments = SegmentBuilder(synth_seg).build()
    assert natural == pytest.approx(
        float(np.mean(segments.phase_start.astype(np.float64) <= spec.seg_entry_band))
    )
    assert entry.value == pytest.approx(min(spec.seg_entry_frac_cap, spec.seg_entry_oversample * natural))


def test_seg_entry_stratum_frac_is_capped(synth_seg):
    loud = CalibrationSpec(seg_entry_oversample=1000.0, seg_entry_frac_cap=0.3)
    entry = SegmentEntryStratum(synth_seg, loud).constants()["seg_entry_stratum_frac"]
    assert entry.value == pytest.approx(0.3)


GRAPH_DIM = 4  # small synthetic joint dim, matching tests/test_retrieve.py's toy-graph idiom
GRAPH_HIDDEN = 16


def _make_seed_demo(task_id: int, seed: int, n: int = 31, dim: int = GRAPH_DIM) -> dict:
    """Straight-line synthetic demo -- the same idiom tests/test_retrieve.py uses to build a graph
    with no dataset dependency: a gripper close halfway through, so NodeTable's phase invariant and
    the grip-bucketed relations both have something to key on."""
    rng = np.random.RandomState(seed)
    start = rng.randn(dim) * 0.01
    direction = rng.randn(dim)
    direction /= np.linalg.norm(direction)
    t = np.arange(n, dtype=np.float32)
    q = (start[None, :] + 0.05 * t[:, None] * direction[None, :]).astype(np.float32)
    qdot = np.zeros_like(q)
    qdot[1:] = q[1:] - q[:-1]
    grip = np.zeros(n, dtype=np.uint8)
    grip[n // 2 :] = 1
    return {"q": q, "qdot": qdot, "grip": grip, "stage": grip.astype(np.int16), "task_id": task_id}


@pytest.fixture
def seed_support_corpus(tmp_path) -> DemoCorpus:
    """A tiny on-disk graph WITH a trained head -- the one combination that makes SeedSupport.
    measurable True and actually runs robot_ood.models.retrieval.gnn.GraphRetrieverNet.seed(), the exact code
    path that crashed under commit cb79161b for a day with no test noticing (calibrate never got
    an end-to-end regression test that carries a head, because doing it against the real corpus
    costs minutes -- see test_the_fcsr_measuring_functions_run_against_the_real_graph's docstring).
    The head is randomly initialised, never trained: SeedSupport only needs the seed/candidates
    pipeline to RUN and return a well-formed distribution, not a good one.
    """
    demos = [_make_seed_demo(task_id=i % 2, seed=100 + i) for i in range(4)]
    nodes = NodeTable.from_demos(demos, task_names=["taskA", "taskB"], coarsen=3)
    # Stand in for the real forward-kinematics ee_base: these demos are straight lines in a
    # GRAPH_DIM-joint space, not a 7-DoF Panda, so PandaKinematics().pose() would reject them.
    # FutureTemp/SegmentBuilder only need SOME ee_base to key phase off of.
    nodes.ee_base = nodes.ee_raw
    edges = EdgeSet.build(nodes, k_sib=4, k_align=2, device="cpu")
    net = GraphRetrieverNet(
        dim=nodes.dim, hidden=GRAPH_HIDDEN, layers=2, n_rel=schema.N_RELATIONS, agg="sum",
    )
    mean, std = nodes.feature_stats()
    net.set_stats(mean, std)
    net.eval()

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    nodes.save(graph_dir)
    edges.save(graph_dir / schema.EDGES_NPZ)
    head = dict(net.state_npz_dict())
    head["graph_hash"] = np.array(
        graph_hash(nodes, edges, constants=graph_dir / CONSTANTS_JSON, c2_state=net.c2_state())
    )
    np.savez(graph_dir / schema.HEAD_NPZ, **head)

    return DemoCorpus("synthetic", hdf5_dir=tmp_path, graph_dir=graph_dir)


def test_seed_support_runs_the_deployed_seed_without_crashing(seed_support_corpus):
    """Regression test for the cb79161b outage: SeedSupport.run() calls the DEPLOYED
    GraphRetrieverNet.seed() (via the all-ones NodeCloud/e_q trick), not a reimplementation, so a
    shape break in that call is exactly what silently killed every calibrate run for a day. This
    must not raise, and its answer must be a genuine node count -- bounded by the graph's own V, not
    by V * hidden (the flattened-tensor symptom the outage actually produced)."""
    spec = CalibrationSpec(n_queries=4, support_mass=0.99, support_quantile=0.95)
    seed_support = SeedSupport(seed_support_corpus, spec, kernel_bw=0.05)
    assert seed_support.measurable

    entries = seed_support.constants()

    n_nodes = len(seed_support_corpus.nodes)
    assert entries["seed_topk"].kind == MEASURED
    assert entries["topm"].kind == MEASURED
    assert 1 <= entries["seed_topk"].value <= n_nodes
    assert 1 <= entries["topm"].value <= n_nodes




def test_seg_tier1_m_is_marked_type_a_unmeasurable():
    """ASSUMED here means "not a per-suite measurement this class computes", not "no measurement
    exists anywhere" -- robot_ood.core.schemas.SEG_TIER1_M's own comment carries the deployed
    own-in-top-8 curve; this class is a snapshot of that one checkpoint's number, not a recompute."""
    entry = SegTier1M().constants()["seg_tier1_m"]
    assert entry.kind == ASSUMED
    assert entry.value == SEG_TIER1_M
    assert "Snapshot" in entry.procedure


def test_measuring_the_four_fcsr_constants_together_lands_in_one_constants_object(synth_seg):
    """Calibrator.measure() just runs every procedure and updates one dict with the result -- this
    pins that the four FCSR additions coexist without a name collision, without standing up a full
    DemoCorpus (that wiring is covered end-to-end below against the real graph)."""
    spec = CalibrationSpec()
    entries = {}
    for measurement in (SegmentLen(synth_seg, spec), FutureTemp(synth_seg, spec),
                        SegmentHardNegatives(synth_seg, spec), SegTier1M()):
        entries.update(measurement.constants())
    constants = Constants(entries=entries, corpus={"suite": "synthetic"})
    for name in ("segment_len", "future_temp", "seg_hard_neg_thresh", "seg_tier1_m"):
        assert name in constants


@skip_no_real_graph
def test_the_fcsr_measuring_functions_run_against_the_real_graph(real_nodes):
    """Calibrator.measure() itself also fits the action-scale (reads every demo HDF5) and the seed
    support (runs the trained head), both several minutes on the real corpus and already exercised
    by ActionScaleFit's/SeedSupport's own tests -- so this checks only the FCSR additions that DO
    converge against the real geometry, the same scope as this file's other @skip_no_real_graph
    test. future_temp is checked separately below (it does NOT converge on this corpus -- that is
    the point of the bug this class now guards against)."""
    spec = CalibrationSpec()
    for name, measurement in (
        ("segment_len", SegmentLen(real_nodes, spec)),
        ("seg_hard_neg_thresh", SegmentHardNegatives(real_nodes, spec)),
        ("seg_tier1_m", SegTier1M()),
    ):
        entry = measurement.constants()[name]
        assert entry.value is not None


@skip_no_real_graph
def test_future_temp_raises_rather_than_report_the_bracket_endpoint_on_the_real_graph(real_nodes):
    """Regression for the FutureTemp bug found 2026-08-14: on the real g_segments.npz corpus the
    proxy-score bisection does NOT converge (achieved perplexity ~42.7 against a target of 8) and
    used to silently emit the T=1e-4 bracket floor as if it were a genuine MEASURED fit. Against the
    OLD code this call returned a Constant (no exception) with
    stats["achieved_perplexity"] far from stats["target_perplexity"] -- a fabricated number this
    test would not have caught without asserting the raise itself."""
    spec = CalibrationSpec()
    with pytest.raises(ValueError, match="did NOT converge"):
        FutureTemp(real_nodes, spec).constants()


# ==================================================================================================
# consumers: robot_ood.core.config reads a calibration only when pointed at one
# ==================================================================================================
def test_graph_config_keeps_its_literals_without_onf_constants(monkeypatch):
    from robot_ood.core import config

    monkeypatch.delenv("ROBOT_OOD_CONSTANTS", raising=False)
    config.calibrated_constants.cache_clear()
    cfg = config.GraphConfig.from_env()
    assert cfg.seed_topk == schema.SEED_TOPK
    assert cfg.topm == 32
    config.calibrated_constants.cache_clear()


def test_uncalibrated_fallback_warns_once_per_key_not_per_call(monkeypatch):
    """Part 2 of the cb79161b outage fix: constants.json stays deliberately opt-in (see
    robot_ood.data.store.REQUIRED_ARTIFACTS's comment), but silently substituting a hardcoded default is
    exactly how a dead C0 calibration went unnoticed for a day. Every fallback must warn, and must
    warn only ONCE per key so a hot path (this one runs per GraphConfig.from_env() call) does not
    become log spam."""
    import warnings

    from robot_ood.core import config

    monkeypatch.delenv("ROBOT_OOD_CONSTANTS", raising=False)
    config.calibrated_constants.cache_clear()
    config._uncalibrated_warned.clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config.calibrated("seed_topk", 256)
            config.calibrated("seed_topk", 256)   # repeat call, same key -- must NOT warn again
            config.calibrated("topm", 32)          # different key -- must warn
        messages = [str(w.message) for w in caught]
        assert len(messages) == 2
        assert any("seed_topk" in m and "256" in m for m in messages)
        assert any("topm" in m and "32" in m for m in messages)
    finally:
        config._uncalibrated_warned.clear()
        config.calibrated_constants.cache_clear()


def test_graph_config_reads_a_calibration_it_is_pointed_at(monkeypatch, tmp_path):
    from robot_ood.core import config

    path = _constants(kernel_bw=0.0883, seed_topk=99, topm=2724).to_json(tmp_path)
    monkeypatch.setenv("ROBOT_OOD_CONSTANTS", str(path))
    monkeypatch.delenv("GR_TOPM", raising=False)
    config.calibrated_constants.cache_clear()
    try:
        cfg = config.GraphConfig.from_env()
        assert cfg.seed_topk == 99
        assert cfg.topm == 2724
        # kernel_bw is deliberately absent: C2 made the seed bandwidth a LEARNED sigma, so the
        # calibrated value is that parameter's init, not a config field. A calibration carrying
        # kernel_bw is still valid -- nothing in GraphConfig reads it any more.
        assert not hasattr(cfg, "kernel_bw")
        # The env variable still wins over the calibration.
        monkeypatch.setenv("GR_TOPM", "32")
        assert config.GraphConfig.from_env().topm == 32
    finally:
        config.calibrated_constants.cache_clear()


# ==================================================================================================
# section 3.4's two gates, on the real corpus
# ==================================================================================================
@pytest.fixture(scope="module")
def real_nodes() -> NodeTable:
    return NodeTable.load(REAL_GRAPH / schema.NODES_NPZ)


@skip_no_real_graph
def test_the_geometric_constants_regenerate_bit_for_bit(real_nodes):
    """Determinism: the same corpus measured twice is the same bytes, hash included."""
    spec = CalibrationSpec()
    def measure() -> Constants:
        entries = {}
        for measurement in (NeighbourSpacing(real_nodes), DilationReach(real_nodes),
                            EntryPerturbation(real_nodes, spec)):
            entries.update(measurement.constants())
        return Constants(entries=entries, corpus={"suite": "long"})

    first, second = measure(), measure()
    assert first.content_hash == second.content_hash
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(second.to_dict(), sort_keys=True)


@skip_no_real_graph
def test_the_known_numbers_come_back(real_nodes):
    """Section 3.4's second gate, on the constants this corpus already has published values for."""
    assert NeighbourSpacing(real_nodes).spacing == pytest.approx(0.0883, abs=5e-5)
    reach = DilationReach(real_nodes).constants()["reach_raw_frames"]
    assert reach.stats["median_raw_frames_per_demo"] == 263.0
    assert reach.value == 240
