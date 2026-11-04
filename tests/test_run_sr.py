"""Tests for the success-rate eval driver (robot_ood.eval, fronted by scripts/run_sr.py).

NO GPU, no real eval: every test either exercises the driver's pure logic (yaml parsing, gate
math, LPT scheduling, resume/completeness decisions) against fabricated fixtures, or
monkeypatches the one external dependency that would otherwise require hardware (torch.cuda)
/ a subprocess (subprocess.Popen, forced to raise so a --dry-run that accidentally launches
something fails loudly instead of silently doing the wrong thing).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import socket
import subprocess
from pathlib import Path

import pytest
import torch

from robot_ood.eval import driver, ladder, launcher, preflight, provenance, runner, scheduler, scoring
from robot_ood.eval.runner import CellRunner

REPO = Path(__file__).resolve().parents[1]


# One shared LadderConfig for every test that only READS the ladder. Tests that need a different
# results root build their own with _config(results_root=...).
CONFIG = ladder.LadderConfig.load()

LADDER = CONFIG.doc
PATHS = CONFIG.paths
SUITES = CONFIG.suites
AXES = CONFIG.axes


def _config(results_root: Path | None = None) -> "ladder.LadderConfig":
    return ladder.LadderConfig.load(results_root=results_root)


def _mkcell(name: str, expected_n: int, **kw) -> "ladder.Cell":
    """A test cell whose tag is derived the way production derives it. name only needs to make the
    cell DISTINCT from its neighbours, so it rides in as the variant suffix; read the real tag off
    cell.tag, never off name."""
    kw.setdefault("suite", "object")
    kw.setdefault("axis", "Objects_Layout")
    kw.setdefault("mode", "base")
    kw.setdefault("policy", "t")
    kw.setdefault("base_task_suite", CONFIG.bench.task_suites[kw["suite"]])
    return ladder.Cell(variant=name, expected_n=expected_n, **kw)


def _score(cell, successes: int, actual_n: int) -> "ladder.CellScore":
    return ladder.CellScore(cell=cell, count=scoring.DirCount(successes, actual_n))


def _verdict(score, allow_partial: bool) -> "ladder.GateVerdict":
    """The production verdict path: completeness first, then the cell's own gate."""
    return runner.CellScorer(CONFIG).verdict(score, allow_partial)


def _iter_declared_cells():
    """(rung_name, cell_doc) for every hand-declared cell (skips the auto_generate: grid rungs)."""
    for rung_name, rung_doc in LADDER["rungs"].items():
        if rung_doc.get("auto_generate"):
            continue
        for cell_doc in rung_doc["cells"]:
            yield rung_name, cell_doc


# ======================================================================================================
# ladder yaml: parses, every cell names a real suite/axis, expected counts match the task's table
# ======================================================================================================
def test_ladder_yaml_parses_and_has_all_rungs():
    # R1 smokes the harness; L4, L3 and R4 are the reporting ladder in widening-scope order and G3
    # is RoboTwin Hard. A wider set (R2/R3/R5/E*/F*/H* ...) was the pre-reduction ablation ladder;
    # asserting the CURRENT set pins this down instead of passing against a stale, larger yaml.
    assert set(LADDER["rungs"]) == {"R1", "L4", "L3", "R4", "G3"}


def test_ladder_paths_block_declares_every_artifact_the_driver_drives():
    """LadderPaths is the single place the driver learns where its external artifacts live, so a
    missing entry must fail loudly at load rather than as a KeyError deep inside a launch."""
    with pytest.raises(KeyError, match="paths block is missing"):
        ladder.LadderPaths.from_doc({"paths": {"score_py": "x"}})


def test_every_declared_cell_names_a_real_suite_and_axis():
    for rung_name, cell_doc in _iter_declared_cells():
        assert cell_doc["suite"] in SUITES, f"{rung_name}: unknown suite {cell_doc['suite']!r}"
        assert cell_doc["axis"] in AXES, f"{rung_name}: unknown axis {cell_doc['axis']!r}"


def test_every_declared_cell_mode_is_a_known_mode():
    for rung_name, cell_doc in _iter_declared_cells():
        mode = cell_doc.get("mode")
        if mode is not None:
            assert mode in ladder.MODES, f"{rung_name} {cell_doc['tag']}: unknown mode {mode!r}"


def test_expected_episode_counts_match_the_task_table():
    expected = {
        "object": {"Background_Textures": 248, "Camera_Viewpoints": 396, "Language_Instructions": 354,
                   "Light_Conditions": 296, "Objects_Layout": 403, "Robot_Initial_States": 398, "Sensor_Noise": 422},
        "spatial": {"Background_Textures": 258, "Camera_Viewpoints": 376, "Language_Instructions": 390,
                    "Light_Conditions": 291, "Objects_Layout": 385, "Robot_Initial_States": 350, "Sensor_Noise": 351},
        "goal": {"Background_Textures": 281, "Camera_Viewpoints": 408, "Language_Instructions": 410,
                 "Light_Conditions": 278, "Objects_Layout": 425, "Robot_Initial_States": 409, "Sensor_Noise": 379},
        "long": {"Background_Textures": 289, "Camera_Viewpoints": 419, "Language_Instructions": 383,
                 "Light_Conditions": 273, "Objects_Layout": 312, "Robot_Initial_States": 393, "Sensor_Noise": 449},
    }
    assert LADDER["expected_episodes"] == expected


def test_unfiltered_cell_expected_n_matches_expected_episodes_table():
    """A cell without its own instance-id filter must inherit the full suite x axis denominator."""
    for rung_name, cell_doc in _iter_declared_cells():
        if cell_doc.get("filter") not in (None,) and cell_doc["axis"] != "Light_Conditions":
            continue   # e.g. R1's 20-instance smoke filter legitimately shrinks expected_n
        suite, axis = cell_doc["suite"], cell_doc["axis"]
        assert cell_doc["expected_n"] == LADDER["expected_episodes"][suite][axis], (rung_name, cell_doc["tag"])


# ======================================================================================================
# modes — one word per arm, exported to the child as $MODE
# ======================================================================================================
@pytest.mark.parametrize("mode", sorted(ladder.MODES))
def test_mode_env_matches_contract(mode):
    env = launcher.EnvBuilder(CONFIG).for_cell(_mkcell("m", 10, mode=mode))
    assert env["MODE"] == mode


def test_mode_env_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown MODE"):
        launcher.EnvBuilder(CONFIG).for_cell(_mkcell("m", 10, mode="not_a_real_mode"))


# ======================================================================================================
# cell resolution / env composition
# ======================================================================================================
def test_compose_cell_env_extra_env_override_wins():
    """A cell's per-cell env: (yaml) / extra_env override must win over the builder's own defaults."""
    cell = _mkcell("t", 10, extra_env={"LOAD_VISION": "1"})
    env = launcher.EnvBuilder(CONFIG).for_cell(cell)
    assert env["LOAD_VISION"] == "1"         # extra_env override, not the builder's own default


def test_compose_cell_env_sets_qndf_dir_per_suite():
    builder = launcher.EnvBuilder(CONFIG)
    for cell in CONFIG.cells_for_rung("R4"):
        env = builder.for_cell(cell)
        assert env["QNDF_DIR"] == str(SUITES[cell.suite].fwm_dir)


def test_resolve_graph_dir_matches_onf_config_paths_graph():
    from robot_ood.core.paths import default_paths
    for suite in SUITES:
        assert preflight.resolve_graph_dir(suite) == default_paths().graph(suite)


def test_graph_artifact_status_reports_missing_files_not_ok(tmp_path):
    status = preflight.graph_artifact_status(tmp_path / "no_such_graph_dir")
    assert not status.ok
    assert status.graph_hash is None
    assert status.g_head_mtime is None
    assert len(status.missing) == 1   # g_head.npz


def test_graph_artifact_status_ok_when_the_locator_is_present(tmp_path):
    import numpy as np
    from robot_ood.data import schema as graph_schema

    d = tmp_path / "graph_dir"
    d.mkdir()
    np.savez(d / graph_schema.HEAD_NPZ, graph_hash=np.array("deadbeef"))
    status = preflight.graph_artifact_status(d)
    assert status.ok
    assert status.missing == ()
    assert status.graph_hash == "deadbeef"
    assert status.g_head_mtime is not None
    assert status.as_record()["graph_hash"] == "deadbeef"


# ======================================================================================================
# 1f: log-directory naming -- shared by both drivers, UTC-stamped, encodes policy/mode/suite/axis so
# two cells that differ only by one of those can never collide/truncate each other's shard logs.
# ======================================================================================================
def test_make_log_dir_embeds_policy_mode_suite_axis_and_normalizes_axis_spaces():
    d = ladder.make_log_dir("stablevla", "sentinel", "long", "Sensor Noise", root=Path("/x"))
    # the log dir name is exactly the cell tag, so logs and results are findable from each other
    assert d.parent == Path("/x/logs")
    assert d.name.endswith("_stablevla_sentinel_long_Sensor_Noise")
    # UTC timestamp prefix: YYYYmmdd_HHMMSS_...
    assert re.match(r"^\d{8}_\d{6}_stablevla_sentinel_long_Sensor_Noise$", d.name), d.name


# ======================================================================================================
# 1e: run.json provenance -- every result dir gets one, naming policy/mode/suite/axis/expected_n/
# resolved graph dir/graph_hash/g_head mtime/git commit/the full mode env.
# ======================================================================================================
def test_write_run_json_round_trips_expected_fields(tmp_path):
    out = provenance.write_run_json(
        tmp_path / "result_dir", policy="stablevla", mode="sentinel", suite="long",
        axis="Sensor_Noise", expected_n=449, graph_dir="/g", graph_hash="abc123",
        g_head_mtime="2026-08-01T00:00:00+00:00", git_commit="deadbeef", mode_env={"MODE": "fcsr"},
    )
    assert out == tmp_path / "result_dir" / "run.json"
    payload = json.loads(out.read_text())
    assert payload["policy"] == "stablevla"
    assert payload["mode"] == "sentinel"
    assert payload["suite"] == "long"
    assert payload["axis"] == "Sensor_Noise"
    assert payload["expected_n"] == 449
    assert payload["graph_dir"] == "/g"
    assert payload["graph_hash"] == "abc123"
    assert payload["g_head_mtime"] == "2026-08-01T00:00:00+00:00"
    assert payload["git_commit"] == "deadbeef"
    assert payload["mode_env"] == {"MODE": "fcsr"}


def test_update_run_json_merges_outcome_into_identity(tmp_path):
    provenance.write_run_json(tmp_path / "result_dir", policy="stablevla", expected_n=449)
    out = provenance.update_run_json(
        tmp_path / "result_dir", actual_n=449, successes=394, status="complete", duration_s=120.5)
    payload = json.loads(out.read_text())
    assert payload["policy"] == "stablevla"          # identity written pre-launch survives
    assert payload["expected_n"] == 449
    assert payload["actual_n"] == 449
    assert payload["successes"] == 394
    assert payload["status"] == "complete"
    assert payload["duration_s"] == 120.5
    assert payload["updated_at"]
    assert "written_at" in payload


def test_light_conditions_axis_auto_resolves_per_suite_filter():
    cell = CONFIG.cell_from_doc({"suite": "goal", "axis": "Light_Conditions", "mode": "base",
                                           "tag": "t", "expected_n": 278})
    assert cell.filter_file == SUITES["goal"].light_filter


def test_non_light_axis_has_no_filter_by_default():
    cell = CONFIG.cell_from_doc({"suite": "goal", "axis": "Background_Textures", "mode": "base",
                                           "tag": "t", "expected_n": 281})
    assert cell.filter_file is None


def test_cell_category_value_is_axis_with_spaces():
    cell = _mkcell("t", 10, axis="Robot_Initial_States")
    assert cell.category_value == "Robot Initial States"


def test_cell_result_dir_layout():
    cell = _mkcell("sr_test", 10, suite="goal", axis="Background_Textures")
    d = cell.result_dir(Path("/results"))
    assert d == Path("/results/plus_libero_goal/Background_Textures") / cell.tag


# ======================================================================================================
# 1a: a rung tag must bake in its resolved MODE, and overriding --mode must re-key the tag's mode
# suffix along with it -- NOT leave the tag naming the original mode. That mismatch is the actual
# wrong-number hazard from the task: --rung R4 then --rung R4 --mode base --resume used to write
# into the SAME result directory (tag had no mode in it) and the ledger's tag-only lookup then reported
# the FIRST run's numbers under the SECOND run's mode name.
# ======================================================================================================
def test_cell_tag_embeds_policy_mode_suite_axis():
    assert ladder.cell_tag("stablevla", "base", "object", "Robot_Initial_States") \
        == "stablevla_base_object_Robot_Initial_States"
    # the optional variant separates a filtered smoke pool from the full axis it subsets
    assert ladder.cell_tag("gr00t", "probe", "long", "Sensor_Noise", "smoke") \
        == "gr00t_probe_long_Sensor_Noise_smoke"


def test_r1_cell_tag_bakes_in_its_declared_mode():
    cells = CONFIG.cells_for_rung("R1")
    assert cells[0].mode == "base"
    assert "_base_" in cells[0].tag


def test_r4_generated_tags_bake_in_their_axis_default_mode():
    for c in CONFIG.cells_for_rung("R4"):
        assert f"_{c.mode}_" in c.tag, c.tag


def test_r1_mode_override_re_keys_the_tag_not_just_the_mode():
    cells = CONFIG.cells_for_rung("R1")
    assert cells[0].mode == "base"
    overridden = [c.with_mode("probe") for c in cells]
    assert overridden[0].mode == "probe"
    assert overridden[0].tag != cells[0].tag                 # THE FIX: tag changes with mode
    assert "_probe_" in overridden[0].tag
    assert "_base_" not in overridden[0].tag                  # no stale mode fragment left behind
    # everything else about the cell (expected_n, gate, filter) survives the swap
    assert overridden[0].expected_n == cells[0].expected_n
    assert overridden[0].gate == cells[0].gate


def test_cell_with_mode_round_trips_through_two_overrides():
    """Overriding mode twice (e.g. two separate CLI invocations against the same rung) must never
    leave a stale mode fragment baked into the tag from an EARLIER override."""
    cell = ladder.Cell(suite="object", axis="Robot_Initial_States", base_task_suite="libero_object",
                        mode="probe", expected_n=398)
    once = cell.with_mode("other")
    assert once.tag == "stablevla_other_object_Robot_Initial_States"
    twice = once.with_mode("base")
    assert twice.tag == "stablevla_base_object_Robot_Initial_States"


def test_cell_is_frozen_so_a_tag_cannot_drift_from_its_mode():
    cell = _mkcell("t", 10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cell.mode = "sentinel"


def test_resolve_cells_with_mode_override_produces_mode_bearing_tags():
    """End-to-end through the same path main() uses for --rung ... --mode ...."""
    default_cells = _driver(["--rung", "R1", "--gpus", "0"])._resolve_cells()
    assert "_base_" in default_cells[0].tag

    overridden_cells = _driver(["--rung", "R1", "--gpus", "0", "--mode", "probe"])._resolve_cells()
    assert overridden_cells[0].mode == "probe"
    assert "_probe_" in overridden_cells[0].tag
    assert overridden_cells[0].tag != default_cells[0].tag


def _driver(argv, ledger=None) -> "driver.SRDriver":
    return driver.SRDriver(driver.build_argparser().parse_args(argv), ledger=ledger)


# ======================================================================================================
# gate math
# ======================================================================================================
def test_gate_from_dict_rejects_an_unknown_kind():
    """GateKind turns a typo'd yaml gate into a load-time error instead of a silent no-op."""
    with pytest.raises(ValueError):
        ladder.Gate.from_dict({"kind": "not_a_real_gate", "min_successes": 1})


def test_gate_min_successes():
    cell = _mkcell("t", 11, gate=ladder.Gate(kind=ladder.GateKind.MIN_SUCCESSES, min_successes=8))
    assert _verdict(_score(cell, 8, 11), allow_partial=False).passed
    assert not _verdict(_score(cell, 7, 11), allow_partial=False).passed


def test_gate_none_always_passes():
    cell = _mkcell("t", 10, gate=None)
    assert _verdict(_score(cell, 1, 10), allow_partial=False).passed


# ======================================================================================================
# completeness: mp4 count vs expected_n
# ======================================================================================================
def test_completeness_rule_marks_short_cell_incomplete_and_excludes_by_default():
    cell = _mkcell("t", 100, gate=ladder.Gate(kind=ladder.GateKind.MIN_SUCCESSES, min_successes=50))
    verdict = _verdict(_score(cell, 90, 90), allow_partial=False)
    assert not verdict.passed
    assert "INCOMPLETE" in verdict.message


def test_completeness_rule_allow_partial_falls_through_to_the_real_gate():
    cell = _mkcell("t", 100, gate=ladder.Gate(kind=ladder.GateKind.MIN_SUCCESSES, min_successes=50))
    assert _verdict(_score(cell, 90, 90), allow_partial=True).passed


def test_score_cell_against_fabricated_results_tree_marks_short_dir_incomplete(tmp_path):
    results_root = tmp_path / "results"
    cell = _mkcell("sr_test", 10)
    d = results_root / "plus_libero_object" / "Objects_Layout" / cell.tag
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"rollout_task_episode{i}_success.mp4").write_bytes(b"x")
    score = runner.CellScorer(_config(results_root)).score(cell)
    assert (score.successes, score.actual_n, score.complete) == (3, 3, False)


# ======================================================================================================
# ledger — append/read round-trip
# ======================================================================================================
def _rec(**kw) -> "provenance.LedgerRecord":
    kw.setdefault("suite", "object")
    kw.setdefault("base_task_suite", "libero_object")
    kw.setdefault("axis", "Objects_Layout")
    kw.setdefault("mode", "base")
    return provenance.LedgerRecord(**kw)


def test_ledger_append_read_round_trip(tmp_path):
    ledger = provenance.Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_rec(tag="a", status=provenance.RunStatus.RUNNING, ts="t0"))
    ledger.append(_rec(tag="a", status=provenance.RunStatus.COMPLETE, ts="t1", actual_n=10))
    records = ledger.read()
    assert [(r.tag, r.status, r.actual_n) for r in records] == [
        ("a", provenance.RunStatus.RUNNING, None),
        ("a", provenance.RunStatus.COMPLETE, 10),
    ]
    assert ledger.latest("a").actual_n == 10
    assert ledger.latest("a", records=records).actual_n == 10
    assert ledger.latest("nope") is None


def test_ledger_record_stamps_ts_when_the_caller_leaves_it_blank():
    payload = _rec(tag="a", status=provenance.RunStatus.RUNNING).to_dict()
    assert payload["ts"]
    assert payload["status"] == "running"       # the VALUE, not the enum repr


def test_ledger_record_from_dict_tolerates_an_older_row(tmp_path):
    """Rows written before a schema change must still read: unknown keys are dropped, missing
    identity fields become "", and an unrecognised status reads as crashed rather than crashing."""
    row = {"tag": "x", "status": "some_status_we_retired", "who_knows": 1, "actual_n": 3}
    rec = provenance.LedgerRecord.from_dict(row)
    assert rec.tag == "x" and rec.suite == "" and rec.actual_n == 3
    assert rec.status is provenance.RunStatus.CRASHED


# ======================================================================================================
# 1a (ledger end): Ledger.latest must ALSO compare mode when given one, not tag alone -- the other
# half of the --resume wrong-number fix (tags now bake mode in via cell_tag, but the ledger lookup
# itself is defense in depth against a record written before that fix, or any other tag collision).
# ======================================================================================================
def test_latest_matching_record_requires_matching_mode_when_given(tmp_path):
    ledger = provenance.Ledger(tmp_path / "l.jsonl")
    records = [
        _rec(tag="x", mode="base", status=provenance.RunStatus.COMPLETE, successes=1),
        _rec(tag="x", mode="sentinel", status=provenance.RunStatus.COMPLETE, successes=2),
    ]
    assert ledger.latest("x", "base", records).successes == 1
    assert ledger.latest("x", "sentinel", records).successes == 2
    # tag-only (mode=None, the default) still returns the LAST matching record, unfiltered by mode --
    # exact prior behaviour, so a caller that doesn't pass mode is unaffected.
    assert ledger.latest("x", None, records).successes == 2


def test_latest_matching_record_mode_mismatch_is_not_a_match(tmp_path):
    ledger = provenance.Ledger(tmp_path / "l.jsonl")
    records = [_rec(tag="x", mode="base", status=provenance.RunStatus.COMPLETE)]
    assert ledger.latest("x", "sentinel", records) is None


def test_read_ledger_missing_file_returns_empty(tmp_path):
    assert provenance.Ledger(tmp_path / "nope.jsonl").read() == []


# ======================================================================================================
# resume: ResumePlanner.plan is a PURE decision; .apply does the actual purge.
# ======================================================================================================
def _planner(results_root: Path, ledger_path: Path) -> "runner.ResumePlanner":
    return runner.ResumePlanner(_config(results_root), provenance.Ledger(ledger_path))


def test_plan_cell_action_no_existing_dir_runs(tmp_path):
    cell = _mkcell("x", 5)
    plan = _planner(tmp_path, tmp_path / "l.jsonl").plan(cell, resume=False, records=[])
    assert plan.action is runner.CellAction.RUN


def test_plan_cell_action_existing_dir_without_resume_errors(tmp_path):
    cell = _mkcell("x", 5)
    d = tmp_path / "plus_libero_object" / "Objects_Layout" / cell.tag
    d.mkdir(parents=True)
    (d / "rollout_t_episode0_success.mp4").write_bytes(b"x")
    plan = _planner(tmp_path, tmp_path / "l.jsonl").plan(cell, resume=False, records=[])
    assert plan.action is runner.CellAction.ERROR


def test_plan_cell_action_resume_skips_ledger_confirmed_complete(tmp_path):
    cell = _mkcell("x", 5)   # _mkcell defaults mode="base" -- the ledger record's mode must match it.
    d = tmp_path / "plus_libero_object" / "Objects_Layout" / cell.tag
    d.mkdir(parents=True)
    for i in range(5):
        (d / f"rollout_t_episode{i}_success.mp4").write_bytes(b"x")
    records = [_rec(tag=cell.tag, mode="base", status=provenance.RunStatus.COMPLETE)]
    plan = _planner(tmp_path, tmp_path / "l.jsonl").plan(cell, resume=True, records=records)
    assert plan.action is runner.CellAction.SKIP
    assert d.exists()   # untouched


def test_plan_cell_action_resume_does_not_skip_on_a_different_modes_ledger_record(tmp_path):
    """The 1a regression this whole fix targets: a complete ledger record for the SAME tag but a
    DIFFERENT mode (e.g. left over from a --rung R4 run, now re-invoked as --rung R4 --mode base)
    must NOT be read as confirming completeness for this cell's mode -- that is exactly how the first
    run's numbers used to get silently reported under the second run's mode name."""
    cell = _mkcell("x", 5, mode="base")
    d = tmp_path / "plus_libero_object" / "Objects_Layout" / cell.tag
    d.mkdir(parents=True)
    for i in range(5):
        (d / f"rollout_t_episode{i}_success.mp4").write_bytes(b"x")
    records = [_rec(tag=cell.tag, mode="sentinel", status=provenance.RunStatus.COMPLETE)]  # different mode
    plan = _planner(tmp_path, tmp_path / "l.jsonl").plan(cell, resume=True, records=records,
                                                         rerun_incomplete=True)
    assert plan.action is runner.CellAction.PURGE   # NOT skip


def test_plan_cell_action_resume_purges_partial_dir_despite_stale_complete_ledger(tmp_path):
    """The hazard from the task: a ledger record claiming 'complete' must NOT be trusted blindly --
    the mp4 count on disk is re-verified against expected_n every time."""
    cell = _mkcell("x", 5)
    d = tmp_path / "plus_libero_object" / "Objects_Layout" / cell.tag
    d.mkdir(parents=True)
    (d / "rollout_t_episode0_success.mp4").write_bytes(b"x")   # only 1 of expected 5
    records = [_rec(tag=cell.tag, status=provenance.RunStatus.COMPLETE)]   # stale/wrong ledger claim
    planner = _planner(tmp_path, tmp_path / "l.jsonl")
    plan = planner.plan(cell, resume=True, records=records, rerun_incomplete=True)
    assert plan.action is runner.CellAction.PURGE
    assert d.exists()   # plan is pure -- nothing deleted yet
    planner.apply(plan)
    assert not d.exists()


def test_plan_cell_action_resume_fills_a_partial_dir_without_deleting_it(tmp_path):
    cell = _mkcell("x", 5)
    d = tmp_path / "plus_libero_object" / "Objects_Layout" / cell.tag
    d.mkdir(parents=True)
    (d / "rollout_t_episode0_success.mp4").write_bytes(b"x")
    plan = _planner(tmp_path, tmp_path / "l.jsonl").plan(cell, resume=True, records=[])
    assert plan.action is runner.CellAction.RUN
    assert d.exists()


def test_plan_cell_action_rerun_incomplete_purges_without_ledger_record(tmp_path):
    cell = _mkcell("x", 5)
    d = tmp_path / "plus_libero_object" / "Objects_Layout" / cell.tag
    d.mkdir(parents=True)
    (d / "rollout_t_episode0_success.mp4").write_bytes(b"x")
    plan = _planner(tmp_path, tmp_path / "l.jsonl").plan(cell, resume=False, records=[],
                                                         rerun_incomplete=True)
    assert plan.action is runner.CellAction.PURGE


def test_apply_cell_action_is_a_noop_for_run_and_skip(tmp_path):
    cell = _mkcell("x", 5)
    d = tmp_path / "plus_libero_object" / "Objects_Layout" / cell.tag
    d.mkdir(parents=True)
    (d / "rollout_t_episode0_success.mp4").write_bytes(b"x")
    planner = _planner(tmp_path, tmp_path / "l.jsonl")
    planner.apply(runner.CellPlan(cell, runner.CellAction.RUN, ""))
    planner.apply(runner.CellPlan(cell, runner.CellAction.SKIP, ""))
    assert d.exists()


# ======================================================================================================
# LPT scheduling
# ======================================================================================================
def _assign(cells, gpus):
    return scheduler.Schedule.build(cells, gpus).assignments


def test_lpt_schedule_exact_greedy_assignment():
    cells = [_mkcell("a", 100), _mkcell("b", 90), _mkcell("c", 50), _mkcell("d", 10), _mkcell("e", 10)]
    assign = _assign(cells, [0, 1])
    assert [c.variant for c in assign[0]] == ["a", "d", "e"]     # 100 + 10 + 10 = 120
    assert [c.variant for c in assign[1]] == ["b", "c"]          # 90 + 50 = 140
    totals = {g: sum(c.expected_n for c in cs) for g, cs in assign.items()}
    assert totals == {0: 120, 1: 140}


def test_lpt_schedule_processes_largest_first_on_a_single_gpu():
    assign = _assign([_mkcell("small", 1), _mkcell("big", 1000)], [0])
    assert [c.variant for c in assign[0]] == ["big", "small"]


def test_lpt_schedule_with_two_slots_per_gpu_balances_over_four_tracks():
    cells = [_mkcell("a", 100), _mkcell("b", 90), _mkcell("c", 50), _mkcell("d", 40), _mkcell("e", 10)]
    schedule = scheduler.Schedule.build(cells, [0, 1], per_gpu=2)
    assert [(t.gpu, t.slot) for t in schedule.tracks] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    by_track = {(t.gpu, t.slot): [c.variant for c in t.cells] for t in schedule.tracks}
    assert by_track == {(0, 0): ["a"], (0, 1): ["b"], (1, 0): ["c"], (1, 1): ["d", "e"]}
    assert {g: [c.variant for c in cs] for g, cs in schedule.assignments.items()} == {
        0: ["a", "b"], 1: ["c", "d", "e"],
    }
    with pytest.raises(ValueError, match="per_gpu"):
        scheduler.Schedule.build(cells, [0], per_gpu=0)


def test_a_lone_cell_is_handed_the_idle_gpus_for_its_servers():
    schedule = scheduler.Schedule.build([_mkcell("only", 500)], [0, 1, 2, 3])
    busy = [t for t in schedule.tracks if t.cells]
    assert [(t.gpu, t.gpus) for t in busy] == [(0, (0, 1, 2, 3))]
    two = scheduler.Schedule.build([_mkcell("a", 500), _mkcell("b", 100)], [0, 1, 2, 3])
    assert {t.gpu: t.gpus for t in two.tracks if t.cells} == {0: (0, 2), 1: (1, 3)}
    slotted = scheduler.Schedule.build([_mkcell("a", 500)], [0, 1], per_gpu=2)
    assert all(t.gpus == (t.gpu,) for t in slotted.tracks)     # slots never borrow GPUs


def test_shards_round_robin_over_servers_and_render_on_their_server_gpu():
    assert CellRunner.shard_targets(4, [14090], [0]) == [(14090, 0)] * 4
    assert CellRunner.shard_targets(5, [1, 2, 3, 4], [0, 1, 2, 3]) == [
        (1, 0), (2, 1), (3, 2), (4, 3), (1, 0)]
    assert CellRunner.shard_targets(3, [1, 2], [0]) == [(1, 0), (2, 0), (1, 0)]


def test_a_cell_with_zero_counts_is_placed_by_its_policy_over_the_gpus_it_gets():
    from dataclasses import replace
    from robot_ood.eval.runner import CellRunner
    cell = replace(_mkcell("p", 500), policy="gr00t", servers=0, num_clients=0)
    placed = CellRunner.place(cell, 4)
    assert (placed.servers, placed.num_clients) == (8, 32)
    assert (CellRunner.place(cell, 1).servers, CellRunner.place(cell, 1).num_clients) == (2, 8)
    explicit = CellRunner.place(replace(cell, servers=3, num_clients=5), 4)
    assert (explicit.servers, explicit.num_clients) == (3, 5)
    only_servers = CellRunner.place(replace(cell, servers=4), 4)
    assert (only_servers.servers, only_servers.num_clients) == (4, 16)


def test_lpt_schedule_covers_every_cell_exactly_once():
    cells = [_mkcell(str(i), i + 1) for i in range(9)]
    assign = _assign(cells, [0, 1, 2])
    scheduled = [c.variant for cs in assign.values() for c in cs]
    assert sorted(scheduled, key=int) == sorted((c.variant for c in cells), key=int)




# ======================================================================================================
# --cells parsing
# ======================================================================================================
def test_parse_cells_arg_builds_cells_in_order_with_explicit_mode():
    cells = CONFIG.cells_from_tokens("goal:Background_Textures,long:Camera_Viewpoints", "sentinel")
    assert [(c.suite, c.axis, c.mode) for c in cells] == [
        ("goal", "Background_Textures", "sentinel"),
        ("long", "Camera_Viewpoints", "sentinel"),
    ]
    assert cells[0].tag == "stablevla_sentinel_goal_Background_Textures"


def test_parse_cells_arg_defaults_mode_from_axis_when_omitted():
    cells = CONFIG.cells_from_tokens("goal:Robot_Initial_States", None)
    # Read the expectation from the ladder rather than restating it: the whole point of this test is
    # that an omitted --mode falls through to axis_default_mode, not that the default is any one value.
    assert cells[0].mode == CONFIG.default_mode("Robot_Initial_States")


def test_parse_cells_arg_unknown_axis_raises():
    with pytest.raises(ValueError):
        CONFIG.cells_from_tokens("goal:NotAnAxis", "base")


def test_parse_cells_arg_unknown_suite_raises():
    with pytest.raises(ValueError):
        CONFIG.cells_from_tokens("notasuite:Background_Textures", "base")


# ======================================================================================================
# R4 (full grid) auto-generation
# ======================================================================================================
def test_r4_generates_the_full_28_cell_grid():
    cells = CONFIG.cells_for_rung("R4")
    assert len(cells) == 28
    assert {c.suite for c in cells} == {"object", "spatial", "goal", "long"}
    assert {c.axis for c in cells} == set(AXES)
    for c in cells:
        assert c.mode == CONFIG.default_mode(c.axis)


def test_cells_for_rung_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown rung"):
        CONFIG.cells_for_rung("NOPE")


# ======================================================================================================
# GPU / load preflight
# ======================================================================================================
def _preflight(min_free_gib=12.0, max_loadavg_factor=2.5) -> "preflight.Preflight":
    config = _config()
    config.limits = dataclasses.replace(
        config.limits, min_free_gib=min_free_gib, max_loadavg_factor=max_loadavg_factor)
    return preflight.Preflight(config)


def test_gpu_report_refuses_busy_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (5 * 1024**3, 140 * 1024**3))
    statuses = _preflight(min_free_gib=12.0).gpus([0])
    assert len(statuses) == 1
    assert not statuses[0].ok
    assert "GiB free" in statuses[0].reason


def test_gpu_report_allows_free_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (50 * 1024**3, 140 * 1024**3))
    assert _preflight(min_free_gib=12.0).gpus([0])[0].ok


def test_gpu_report_nonexistent_index_is_not_ok(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (50 * 1024**3, 140 * 1024**3))
    assert not _preflight(min_free_gib=12.0).gpus([5])[0].ok


def test_gpu_report_defaults_to_every_visible_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (50 * 1024**3, 140 * 1024**3))
    assert [s.index for s in _preflight().gpus()] == [0, 1, 2]


def test_load_report_ok_and_busy(monkeypatch):
    monkeypatch.setattr(os, "getloadavg", lambda: (1.0, 1.0, 1.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    ok = _preflight(max_loadavg_factor=1.5).load()
    assert ok.ok and ok.threshold == 12.0

    monkeypatch.setattr(os, "getloadavg", lambda: (20.0, 20.0, 20.0))
    assert not _preflight(max_loadavg_factor=1.5).load().ok


def test_check_benchmark_reports_missing_tree(monkeypatch, tmp_path):
    monkeypatch.setenv("LIBERO_HOME", str(tmp_path / "does_not_exist"))
    result = _preflight().benchmark()
    assert not result.ok
    assert "missing/empty" in result.message


# ======================================================================================================
# --dry-run: no subprocess, no filesystem mutation, even with --resume against a partial dir
# ======================================================================================================
def _no_launch(monkeypatch, tmp_path, n_gpus=2):
    def _boom(*_a, **_k):
        raise AssertionError("subprocess.Popen must not be called during --dry-run")
    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpus)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (100 * 1024**3, 140 * 1024**3))
    monkeypatch.setattr(os, "getloadavg", lambda: (0.1, 0.1, 0.1))
    return provenance.Ledger(tmp_path / "ledger.jsonl")


def test_dry_run_never_calls_subprocess_popen(monkeypatch, tmp_path):
    ledger = _no_launch(monkeypatch, tmp_path)
    results_root = tmp_path / "results"
    rc = _driver(["--rung", "R1", "--gpus", "0,1", "--dry-run",
                  "--results-root", str(results_root)], ledger=ledger).run()
    assert rc == runner.ExitCode.OK


def test_dry_run_with_rerun_incomplete_does_not_delete_a_partial_dir(monkeypatch, tmp_path):
    """Regression guard: --dry-run --rerun-incomplete must PREVIEW the purge, never perform it."""
    ledger = _no_launch(monkeypatch, tmp_path, n_gpus=1)
    results_root = tmp_path / "results"
    cells = _config(results_root).cells_for_rung("R1")
    partial_dir = cells[0].result_dir(results_root)
    partial_dir.mkdir(parents=True)
    (partial_dir / "rollout_t_episode0_success.mp4").write_bytes(b"x")   # 1 of expected 11

    rc = _driver(["--rung", "R1", "--gpus", "0", "--dry-run", "--rerun-incomplete",
                  "--results-root", str(results_root)], ledger=ledger).run()
    assert rc == runner.ExitCode.OK
    assert partial_dir.exists()   # NOT purged -- dry-run is read-only


def test_dry_run_refuses_when_every_gpu_is_busy(monkeypatch, tmp_path):
    _no_launch(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (1 * 1024**3, 140 * 1024**3))
    rc = _driver(["--rung", "R1", "--gpus", "0,1", "--dry-run",
                  "--results-root", str(tmp_path / "r")],
                 ledger=provenance.Ledger(tmp_path / "l.jsonl")).run()
    assert rc == runner.ExitCode.ERROR


# ======================================================================================================
# exit codes (main() wiring)
# ======================================================================================================
R1_TAG = "stablevla_base_object_Robot_Initial_States_smoke"   # rung R1's first cell, tag derived by
                                                                # cell_tag(policy, mode, suite, axis, variant)


def test_main_exit_code_2_when_gate_not_met(tmp_path):
    results_root = tmp_path / "results"
    cell_dir = results_root / "plus_libero_object" / "Robot_Initial_States" / R1_TAG
    cell_dir.mkdir(parents=True)
    # 1/11 successes vs the min_successes=8 gate -- must fail
    (cell_dir / "rollout_t_episode0_success.mp4").write_bytes(b"x")
    for i in range(1, 11):
        (cell_dir / f"rollout_t_episode{i}_failure.mp4").write_bytes(b"x")
    rc = driver.main(["--score", "--rung", "R1", "--results-root", str(results_root)])
    assert rc == 2


def test_main_exit_code_0_when_gate_met(tmp_path):
    results_root = tmp_path / "results"
    cell_dir = results_root / "plus_libero_object" / "Robot_Initial_States" / R1_TAG
    cell_dir.mkdir(parents=True)
    # 9/11 successes -- meets the min_successes=8 gate
    for i in range(9):
        (cell_dir / f"rollout_t_episode{i}_success.mp4").write_bytes(b"x")
    for i in range(9, 11):
        (cell_dir / f"rollout_t_episode{i}_failure.mp4").write_bytes(b"x")
    rc = driver.main(["--score", "--rung", "R1", "--results-root", str(results_root)])
    assert rc == 0


def test_main_exit_code_1_when_nothing_scored(tmp_path):
    rc = driver.main(["--score", "--rung", "R1", "--results-root", str(tmp_path / "results")])
    assert rc == 1


def test_main_requires_rung_or_cells(tmp_path):
    rc = driver.main(["--results-root", str(tmp_path / "results")])
    assert rc == 1


def test_main_rejects_rung_and_cells_together():
    with pytest.raises(SystemExit):
        driver.main(["--rung", "R1", "--cells", "object:Objects_Layout", "--gpus", "0"])


def test_main_unknown_rung_raises():
    with pytest.raises(SystemExit):
        driver.main(["--rung", "NOPE", "--gpus", "0"])


def test_main_requires_gpus_to_launch(tmp_path):
    rc = driver.main(["--rung", "R1", "--results-root", str(tmp_path / "results")])
    assert rc == 1


def test_parse_gpus_dedups_and_keeps_order():
    parse = driver.SRDriver._parse_gpus
    assert parse(None) == [] and parse("") == []
    assert parse("0,1") == [0, 1]
    assert parse("3,2") == [3, 2]
    assert parse(" 1 , 2 ") == [1, 2]
    # a repeated index would otherwise open two schedule tracks against the same device
    assert parse("0,0,1") == [0, 1]


# ======================================================================================================
# Port claiming — concurrent run_sr invocations must never share a policy-server port.
#
# Why this has its own test: the failure is SILENT and it mis-scores. Two run_sr processes both start
# their port counter at defaults.base_port, so the second one's sim clients connect to the FIRST one's
# already-running policy server -- a different CHECKPOINT -- and write ordinary-looking mp4s. It was
# caught in practice with a goal Robot-Init cell whose clients reached a long-checkpoint server and
# produced 121 episodes under the wrong policy, with nothing in any log to say so.
# ======================================================================================================
def test_claim_port_skips_a_bound_port(tmp_path):
    import socket as _s
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as held:
        held.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
        held.bind(("0.0.0.0", 0))
        held.listen(1)
        busy = held.getsockname()[1]
        got = scheduler.PortAllocator(busy, tmp_path / ".ports").claim()
        assert got != busy, "claim handed out a port something is already listening on"


def test_claim_port_reservation_blocks_a_concurrent_process(tmp_path):
    """A reservation held by a LIVE pid must not be handed out again. This is the half a bind-probe
    cannot cover: a policy server spends minutes loading its checkpoint before it binds, so a second
    process probing during that window still sees the port as free and claims it too."""
    base = 15900
    reservations = tmp_path / ".ports"
    reservations.mkdir(parents=True)
    (reservations / str(base)).write_text(f"{os.getpid()}\n")     # us: definitely alive
    got = scheduler.PortAllocator(base, reservations).claim()
    assert got != base, "a reservation held by a live process was handed out anyway"


def test_claim_port_reclaims_a_reservation_whose_owner_died(tmp_path):
    """A run killed mid-flight leaves its reservation files behind; they must not wedge the port
    range forever, so an unreachable owner pid is treated as stale and taken over."""
    base = 15910
    reservations = tmp_path / ".ports"
    reservations.mkdir(parents=True)
    dead_pid = 999_999                              # far above /proc/sys/kernel/pid_max on this box
    (reservations / str(base)).write_text(f"{dead_pid}\n")
    assert scheduler.PortAllocator(base, reservations).claim() == base, \
        "a reservation from a dead process was not reclaimed"


def test_claim_port_release_all_drops_every_marker(tmp_path):
    reservations = tmp_path / ".ports"
    allocator = scheduler.PortAllocator(16100, reservations)
    ports = [allocator.claim() for _ in range(3)]
    assert all((reservations / str(p)).exists() for p in ports)
    allocator.release_all()
    assert not any((reservations / str(p)).exists() for p in ports)


def test_claim_port_gives_out_a_distinct_port_every_call(tmp_path):
    allocator = scheduler.PortAllocator(16200, tmp_path / ".ports")
    ports = [allocator.claim() for _ in range(5)]
    assert len(set(ports)) == 5


# ======================================================================================================
# --rung + --cells subsetting
# ======================================================================================================
def test_rung_plus_cells_selects_a_subset_and_keeps_the_rungs_own_filter_and_n():
    """--rung R --cells suite:axis must SUBSET the rung, not rebuild the cell ad hoc: an ad hoc cell
    carries the full-axis expected_n and no instance filter, so a 12-episode smoke would silently
    become a full run."""
    cells = _driver(["--rung", "R1", "--cells", "long:Sensor_Noise", "--gpus", "0"])._resolve_cells()
    assert [(c.suite, c.axis) for c in cells] == [("long", "Sensor_Noise")]
    full = [c for c in CONFIG.cells_for_rung("R1") if (c.suite, c.axis) == ("long", "Sensor_Noise")][0]
    assert cells[0].expected_n == full.expected_n
    assert cells[0].filter_file == full.filter_file is not None


def test_rung_plus_cells_rejects_a_cell_the_rung_does_not_declare():
    driver = _driver(["--rung", "R1", "--cells", "goal:Camera_Viewpoints", "--gpus", "0"])
    with pytest.raises(SystemExit):
        driver._resolve_cells()


# =====================================================================================================
# CellLock — one live driver per result dir.
# =====================================================================================================
def test_cell_lock_refuses_a_dir_another_live_driver_holds(tmp_path):
    held = runner.CellLock(tmp_path / "cell")
    held.acquire("stablevla_fcsr_long_Camera_Viewpoints")
    with pytest.raises(runner.CellLockedError):
        runner.CellLock(tmp_path / "cell").acquire("stablevla_fcsr_long_Camera_Viewpoints")
    held.release()
    runner.CellLock(tmp_path / "cell").acquire("stablevla_fcsr_long_Camera_Viewpoints")


def test_cell_lock_takes_over_a_claim_whose_owner_died(tmp_path):
    stale = {"pid": 4194303, "host": os.uname().nodename, "tag": "x", "started": "x"}
    (tmp_path / "cell").mkdir()
    (tmp_path / "cell" / runner.CellLock.NAME).write_text(json.dumps(stale))
    assert runner.CellLock.live_owner(tmp_path / "cell") is None
    runner.CellLock(tmp_path / "cell").acquire("x")
    owner = json.loads((tmp_path / "cell" / runner.CellLock.NAME).read_text())
    assert owner["pid"] == os.getpid()


def test_cell_lock_never_steals_a_claim_from_another_host(tmp_path):
    remote = {"pid": 1, "host": "someotherbox", "tag": "x", "started": "x"}
    (tmp_path / "cell").mkdir()
    (tmp_path / "cell" / runner.CellLock.NAME).write_text(json.dumps(remote))
    assert runner.CellLock.live_owner(tmp_path / "cell") is not None
    with pytest.raises(runner.CellLockedError):
        runner.CellLock(tmp_path / "cell").acquire("x")


def test_cell_lock_file_is_not_counted_as_an_episode(tmp_path):
    runner.CellLock(tmp_path / "cell").acquire("x")
    config = ladder.LadderConfig.load(results_root=tmp_path)
    assert config.scorer.count_dir(tmp_path / "cell").n == 0


# =====================================================================================================
# --num-clients — shard count override.
# =====================================================================================================
def _cells_for(argv):
    return driver.SRDriver(driver.build_argparser().parse_args(argv))._resolve_cells()


def test_num_clients_defaults_to_zero_so_the_policy_places_the_cell():
    cells = _cells_for(["--cells", "long:Camera_Viewpoints", "--mode", "fcsr"])
    assert [c.num_clients for c in cells] == [0]
    assert [c.servers for c in cells] == [0]


def test_num_clients_override_applies_to_every_cell():
    cells = _cells_for(["--cells", "long:Camera_Viewpoints,long:Language_Instructions",
                        "--mode", "fcsr", "--num-clients", "12"])
    assert [c.num_clients for c in cells] == [12, 12]


def test_num_clients_override_applies_to_a_rung():
    cells = _cells_for(["--rung", "L4", "--mode", "fcsr", "--num-clients", "12"])
    assert cells and all(c.num_clients == 12 for c in cells)


def test_num_clients_rejects_zero():
    with pytest.raises(SystemExit):
        _cells_for(["--cells", "long:Camera_Viewpoints", "--mode", "fcsr", "--num-clients", "0"])


# =====================================================================================================
# expected_episodes — the completeness denominators, held to the installed LIBERO-Plus.
# =====================================================================================================
def test_expected_episodes_match_the_installed_checkout():
    """A stale denominator marks a full cell INCOMPLETE and drops it from every total.

    Light_Conditions is the one axis a run does NOT see whole: every suite's light_filter
    (configs/robot_ood.yaml) drops the bad instance id that crashes benchmark construction, so the
    filter file's own length is the denominator there, not the checkout's category count.
    """
    import collections

    import yaml

    home = os.environ.get("LIBERO_HOME")
    if not home:
        pytest.skip("LIBERO_HOME unset")
    spec = Path(home) / "libero" / "libero" / "benchmark" / "task_classification.json"
    if not spec.exists():
        pytest.skip(f"no task_classification.json at {spec}")

    doc = json.loads(spec.read_text())
    declared = yaml.safe_load((REPO / "configs" / "sr_ladder.yaml").read_text())["expected_episodes"]
    filters = ladder.load_suites_doc(REPO / "configs" / "robot_ood.yaml")["suites"]
    bench = {"object": "libero_object", "spatial": "libero_spatial",
             "goal": "libero_goal", "long": "libero_10"}
    drift = []
    for suite, key in bench.items():
        actual = collections.Counter(row["category"] for row in doc[key])
        filter_file = REPO / filters[suite]["light_filter"]
        for axis, n in declared[suite].items():
            if axis == "Light_Conditions":
                ids = [line for line in filter_file.read_text().split() if line.strip()]
                got, src = len(ids), f"filter {filter_file.name}"
            else:
                got, src = actual[axis.replace("_", " ")], "checkout"
            if got != n:
                drift.append(f"{suite}/{axis}: ladder={n} {src}={got}")
    assert not drift, "sr_ladder.yaml expected_episodes drifted from the checkout:\n  " + "\n  ".join(drift)


def test_run_config_sha_hashes_the_tree_or_is_empty(tmp_path):
    assert provenance.run_config_sha(tmp_path) == ""
    (tmp_path / "run_config.json").write_text('{"meta": {}}')
    assert len(provenance.run_config_sha(tmp_path)) == 64
    assert provenance.run_config_sha(None) == ""


class _FakeClient:
    """A client process that exits after a set number of polls, writing to its log until then."""

    def __init__(self, log, polls, quiet=False):
        self.log, self.polls, self.quiet, self.killed = log, polls, quiet, False
        log.write_text("")

    def poll(self):
        if self.killed:
            return -9
        self.polls -= 1
        if self.polls <= 0:
            return 0
        if not self.quiet:
            with self.log.open("a") as f:
                f.write("step\n")
        return None

    def kill(self):
        self.killed = True


def _wait(procs, logs):
    runner = object.__new__(CellRunner)
    return CellRunner._wait_clients(runner, procs, logs, "cell")


def test_wait_clients_returns_every_clients_code(tmp_path, monkeypatch):
    monkeypatch.setattr(CellRunner, "CLIENT_POLL_S", 0.0)
    logs = [tmp_path / "a.log", tmp_path / "b.log"]
    procs = [_FakeClient(logs[0], 2), _FakeClient(logs[1], 5)]
    assert _wait(procs, logs) == (0, 0)
    assert not any(p.killed for p in procs)


def test_wait_clients_kills_the_one_that_stops_printing(tmp_path, monkeypatch):
    monkeypatch.setattr(CellRunner, "CLIENT_POLL_S", 0.0)
    monkeypatch.setattr(CellRunner, "CLIENT_STALL_S", -1.0)
    logs = [tmp_path / "a.log", tmp_path / "b.log"]
    procs = [_FakeClient(logs[0], 3), _FakeClient(logs[1], 10**6, quiet=True)]
    assert _wait(procs, logs) == (0, -9)
    assert procs[1].killed and not procs[0].killed


def test_wait_clients_kills_every_silent_client_of_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(CellRunner, "CLIENT_POLL_S", 0.0)
    monkeypatch.setattr(CellRunner, "CLIENT_STALL_S", -1.0)
    logs = [tmp_path / f"{n}.log" for n in "abc"]
    procs = [_FakeClient(logs[0], 10**6, quiet=True), _FakeClient(logs[1], 2),
             _FakeClient(logs[2], 10**6, quiet=True)]
    assert _wait(procs, logs) == (-9, 0, -9)


def test_a_lock_held_by_a_zombie_is_stale():
    assert runner.CellLock._stale({"host": socket.gethostname(), "pid": os.getpid()}) is False
    proc = subprocess.Popen(["true"])
    while runner.CellLock._stale({"host": socket.gethostname(), "pid": proc.pid}) is False:
        pass
    proc.wait()
    assert runner.CellLock._stale({"host": socket.gethostname(), "pid": 2 ** 22 - 1}) is True
    assert runner.CellLock._stale({"host": "somewhere-else", "pid": 1}) is False
