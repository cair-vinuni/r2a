"""The build DAG, its hash-based staleness, and the job runner that replaced detach-and-poll.

The point of every test here is the thing `build_artifacts.sh` could not do: notice that an input
changed, and notice that a detached job died.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_ood.pipeline.dag import DEPLOY_ARTIFACTS, PROFILES, STEPS, PipelineSpec, Step
from robot_ood.pipeline.jobs import run_job, spawn_job
from robot_ood.pipeline.manifest import Manifest, StepRecord, config_hash, file_hash, hash_all
from robot_ood.pipeline.runner import explain, plan


# ==================================================================================================
# manifest -- staleness by hash, never by existence
# ==================================================================================================
def test_file_hash_is_empty_for_a_missing_path(tmp_path):
    assert file_hash(tmp_path / "nope.npz") == ""


def test_file_hash_changes_with_content(tmp_path):
    p = tmp_path / "a.npz"
    p.write_bytes(b"one")
    first = file_hash(p)
    p.write_bytes(b"two")
    assert first != file_hash(p) and first.startswith("sha256:")


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def _record(manifest, step, inputs, outputs, cfg):
    manifest.record(StepRecord(step=step, config_hash=cfg, inputs=hash_all(inputs),
                               outputs=hash_all(outputs)))


def test_a_step_that_never_ran_is_stale(tmp_path):
    assert Manifest(tmp_path / "m.json").stale("03_locator", [], [], "c") == "never run"


def test_a_recorded_step_with_unchanged_files_is_not_stale(tmp_path):
    src, out = tmp_path / "in.npz", tmp_path / "out.npz"
    src.write_bytes(b"in"); out.write_bytes(b"out")
    m = Manifest(tmp_path / "m.json")
    _record(m, "s", [src], [out], "cfg")
    assert m.stale("s", [src], [out], "cfg") is None


def test_a_changed_INPUT_makes_a_step_stale_even_though_its_output_exists(tmp_path):
    """The whole reason the manifest exists: `[ -e output ]` cannot see this."""
    src, out = tmp_path / "in.npz", tmp_path / "out.npz"
    src.write_bytes(b"in"); out.write_bytes(b"out")
    m = Manifest(tmp_path / "m.json")
    _record(m, "s", [src], [out], "cfg")
    src.write_bytes(b"in, but different")
    assert m.stale("s", [src], [out], "cfg") == f"input changed: {src}"
    assert out.exists()


def test_a_changed_config_makes_a_step_stale(tmp_path):
    out = tmp_path / "out.npz"
    out.write_bytes(b"out")
    m = Manifest(tmp_path / "m.json")
    _record(m, "s", [], [out], "cfg")
    assert m.stale("s", [], [out], "other") == "config changed since the recorded run"


def test_a_deleted_output_makes_a_step_stale(tmp_path):
    out = tmp_path / "out.npz"
    out.write_bytes(b"out")
    m = Manifest(tmp_path / "m.json")
    _record(m, "s", [], [out], "cfg")
    out.unlink()
    assert m.stale("s", [], [out], "cfg") == f"output is gone: {out}"


def test_a_failed_step_is_stale_however_its_files_look(tmp_path):
    m = Manifest(tmp_path / "m.json")
    m.record(StepRecord(step="s", exit_code=1, config_hash="cfg"))
    assert m.stale("s", [], [], "cfg") == "last run exited 1"


def test_a_pre_registered_exit_code_leaves_a_step_current(tmp_path):
    m = Manifest(tmp_path / "m.json")
    m.record(StepRecord(step="05_calib", exit_code=2, config_hash="cfg"))
    assert m.stale("05_calib", [], [], "cfg", accept=(0, 2)) is None
    assert m.stale("05_calib", [], [], "cfg") == "last run exited 2"


def test_the_planner_carries_a_steps_accepted_codes_into_the_staleness_test():
    step = Step("gated", "build", lambda s: ["run"], lambda s: [], lambda s: [], accept_rc=(0, 2))
    spec = PipelineSpec(suite="long", device="cpu")
    m = Manifest(Path("unused.json"))
    cfg = config_hash({"argv": ["run"], "inputs": [], "outputs": []})
    m.record(StepRecord(step="gated", exit_code=2, config_hash=cfg))
    assert [p.stale for p in plan(spec, (step,), m)] == [False]


def test_manifest_round_trips_through_disk(tmp_path):
    m = Manifest.load(tmp_path)
    m.record(StepRecord(step="s", command="x", exit_code=0, config_hash="cfg",
                        inputs={"a": "sha256:1"}, outputs={"b": "sha256:2"}))
    m.save()
    again = Manifest.load(tmp_path)
    assert again.records["s"].outputs == {"b": "sha256:2"}
    assert json.loads((tmp_path / "pipeline.json").read_text())["steps"]["s"]["command"] == "x"


# ==================================================================================================
# the DAG
# ==================================================================================================
def test_every_step_resolves_its_command_inputs_and_outputs():
    spec = PipelineSpec("long", "cpu")
    for step in STEPS:
        assert step.argv(spec) and isinstance(step.inputs(spec), list)
        assert isinstance(step.outputs(spec), list)


def test_step_names_are_unique_and_ordered():
    names = [s.name for s in STEPS]
    assert names == sorted(names) and len(set(names)) == len(names)


def test_every_step_output_is_produced_before_it_is_consumed():
    """A step may only read what an EARLIER step wrote, or what came with the corpus."""
    spec = PipelineSpec("long", "cpu")
    written: set[str] = set()
    for step in STEPS:
        unmet = [p for p in step.inputs(spec) if p not in written]
        assert not unmet, f"{step.name} reads what nothing wrote yet: {unmet}"
        written |= set(step.outputs(spec))


def test_the_deploy_gate_reads_every_artifact_the_earlier_steps_write():
    spec = PipelineSpec("long", "cpu")
    written = {p for s in STEPS[:-1] for p in s.outputs(spec)}
    assert set(STEPS[-1].inputs(spec)) <= written


def test_deploy_artifacts_are_all_written_by_some_step():
    spec = PipelineSpec("long", "cpu")
    written = {Path(p).name for s in STEPS for p in s.outputs(spec)}
    assert set(DEPLOY_ARTIFACTS) <= written


def test_the_smoke_profile_limits_the_corpus_for_both_step_02_and_step_07():
    """The policy cache refuses a row count differing from NodeTable.q_raw, so
    limiting one side alone is a broken build."""
    spec = PipelineSpec("long", "cpu", "smoke")
    graph = next(s for s in STEPS if s.name == "02_graph").argv(spec)
    cache = next(s for s in STEPS if s.name == "07_apiraw").argv(spec)
    assert "--limit-tasks" in graph and "--limit-tasks" in cache
    assert graph[graph.index("--limit-demos") + 1] == cache[cache.index("--limit-demos") + 1]


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_profile_supplies_every_epoch_knob_a_step_asks_for(profile):
    spec = PipelineSpec("long", "cpu", profile)
    for step in STEPS:
        step.argv(spec)          # a missing knob would KeyError here


def test_no_constants_step_remains():
    """The calibration is embedded in g_fcsr.npz by train fcsr; no step writes a constants.json."""
    assert not any("constants" in s.name for s in STEPS)


def test_the_pipeline_pins_the_artifact_dir_for_every_child():
    spec = PipelineSpec("long", "cpu")
    assert spec.env()["GR_GRAPH_DIR"] == str(spec.artifacts)


# ==================================================================================================
# explain
# ==================================================================================================
def test_explain_names_every_step_and_why_it_is_stale(tmp_path):
    spec = PipelineSpec("long", "cpu", outputs_root=tmp_path)
    lines = explain(spec, plan(spec, STEPS, Manifest.load(tmp_path)))
    text = "\n".join(lines)
    for step in STEPS:
        assert step.name in text
    assert text.count("never run") == len(STEPS)
    assert f"{len(STEPS)} of {len(STEPS)} step(s) would run." in lines[-1]


def test_force_marks_every_step_stale_whatever_the_manifest_says(tmp_path):
    spec = PipelineSpec("long", "cpu", outputs_root=tmp_path)
    m = Manifest.load(tmp_path)
    for p in plan(spec, STEPS, m):
        m.record(StepRecord(step=p.step.name, config_hash=p.cfg_hash,
                            inputs=hash_all(p.inputs), outputs=hash_all(p.outputs)))
    assert all(p.stale for p in plan(spec, STEPS, m, force=True))
    assert all(p.reason == "--force" for p in plan(spec, STEPS, m, force=True))


# ==================================================================================================
# jobs -- a dead job is seen, not waited on
# ==================================================================================================
def test_run_job_records_the_exit_code_and_the_output(tmp_path):
    r = run_job("ok", ["bash", "-c", "echo hi; exit 0"], tmp_path, echo=False)
    assert r.ok and r.exit_code == 0 and "hi" in r.log_path.read_text()


def test_run_job_reports_a_failure_rather_than_raising(tmp_path):
    r = run_job("bad", ["bash", "-c", "echo boom >&2; exit 7"], tmp_path, echo=False)
    assert not r.ok and r.exit_code == 7 and "boom" in r.log_path.read_text()


def test_a_detached_job_that_dies_is_noticed_with_its_exit_code(tmp_path):
    """§17.4 defect 2: `until [ -f out ]` waits forever on a job that died at minute five."""
    job = spawn_job("dies", ["bash", "-c", "exit 9"], tmp_path)
    assert job.wait(poll=0.1, timeout=30).exit_code == 9


def test_a_detached_job_that_finishes_reports_success(tmp_path):
    job = spawn_job("lives", ["bash", "-c", "sleep 0.3; echo done"], tmp_path)
    result = job.wait(poll=0.1, timeout=30)
    assert result.ok and "done" in result.log_path.read_text()


def test_a_detached_job_can_be_stopped(tmp_path):
    job = spawn_job("long", ["bash", "-c", "sleep 300"], tmp_path)
    assert job.alive()
    job.stop()
    assert not job.alive()


def test_waiting_past_the_deadline_raises_instead_of_blocking(tmp_path):
    job = spawn_job("slow", ["bash", "-c", "sleep 60"], tmp_path)
    try:
        with pytest.raises(TimeoutError):
            job.wait(poll=0.1, timeout=1)
    finally:
        job.stop()
