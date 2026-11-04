"""Tests for robot_ood.data.store -- the ARTIFACTS.json provenance manifest and its verify() gate.

All fixtures live under tmp_path; nothing here touches the shared outputs/ tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_ood.data import store
from robot_ood.data.store import (
    HASH_STAMPED_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    ArtifactManifest,
    assert_not_latest,
    validate,
)


def _touch_npz(path: Path, **arrays) -> None:
    np.savez(path, **arrays)


# ==========================================================================================
# assert_not_latest
# ==========================================================================================
def test_assert_not_latest_refuses_a_path_naming_latest(tmp_path):
    real = tmp_path / "graph_20260101_000000_x" / "artifacts"
    real.mkdir(parents=True)
    latest = tmp_path / "latest"
    latest.symlink_to(real.parent)
    with pytest.raises(ValueError, match="latest"):
        assert_not_latest(latest / "artifacts")


def test_assert_not_latest_names_the_resolved_dir_in_the_message(tmp_path):
    real = tmp_path / "graph_20260101_000000_x" / "artifacts"
    real.mkdir(parents=True)
    latest = tmp_path / "latest"
    latest.symlink_to(real.parent)
    with pytest.raises(ValueError, match=str(real)):
        assert_not_latest(latest / "artifacts")


def test_assert_not_latest_accepts_a_plain_directory(tmp_path):
    d = tmp_path / "graph_20260101_000000_x" / "artifacts"
    d.mkdir(parents=True)
    assert assert_not_latest(d) == d.resolve()


# ==========================================================================================
# load: absent manifest is empty, not an error
# ==========================================================================================
def test_load_absent_manifest_is_empty_not_an_error(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    manifest = ArtifactManifest.load(d)
    assert manifest.artifacts == {}
    assert manifest.graph_dir == d.resolve()


def test_load_infers_suite_from_the_shipping_directory_layout(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    assert ArtifactManifest.load(d).suite == "long"


def test_load_infers_suite_from_a_run_directory_too(tmp_path):
    for run in ("graph_20260819_191549", "latest"):
        d = tmp_path / run / "long" / run / "artifacts"
        d.mkdir(parents=True)
        assert ArtifactManifest.load(d).suite == "long"


def test_load_rejects_a_manifest_with_the_wrong_schema(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    (d / "ARTIFACTS.json").write_text(json.dumps({"schema": 999, "artifacts": {}}))
    with pytest.raises(ValueError, match="schema"):
        ArtifactManifest.load(d)


# ==========================================================================================
# record / save round trip
# ==========================================================================================
def test_record_reads_graph_hash_off_the_npz_stamp(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_head.npz", graph_hash=np.array("deadbeef"))
    manifest = ArtifactManifest(d)
    manifest.record(d / "g_head.npz", stage="retrieval", command="python -m robot_ood.cli train")
    assert manifest.artifacts["g_head.npz"].graph_hash == "deadbeef"


def test_record_with_no_stamp_in_the_file_records_none(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_lang.npz", e_task=np.zeros(4))
    manifest = ArtifactManifest(d)
    manifest.record(d / "g_lang.npz", stage="lang", command="python -m robot_ood.cli build language --suite long")
    assert manifest.artifacts["g_lang.npz"].graph_hash is None


def test_record_explicit_graph_hash_overrides_the_files_own_stamp(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_head.npz", graph_hash=np.array("aaa"))
    manifest = ArtifactManifest(d)
    manifest.record(d / "g_head.npz", stage="retrieval", command="c", graph_hash="bbb")
    assert manifest.artifacts["g_head.npz"].graph_hash == "bbb"


def test_record_missing_file_raises():
    manifest = ArtifactManifest(Path("/nonexistent"))
    with pytest.raises(FileNotFoundError):
        manifest.record("/nonexistent/g_head.npz", stage="x", command="x")


def test_save_then_load_round_trips_every_field(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_head.npz", graph_hash=np.array("deadbeef"))
    written = ArtifactManifest(d)
    written.record(d / "g_head.npz", stage="retrieval", command="python -m robot_ood.cli train")
    written.save()

    reloaded = ArtifactManifest.load(d)
    assert reloaded.suite == "long"
    assert reloaded.graph_dir == d.resolve()
    rec = reloaded.artifacts["g_head.npz"]
    assert rec.stage == "retrieval"
    assert rec.command == "python -m robot_ood.cli train"
    assert rec.graph_hash == "deadbeef"
    assert rec.sha256 == written.artifacts["g_head.npz"].sha256


def test_save_writes_the_documented_schema_shape(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_head.npz", graph_hash=np.array("deadbeef"))
    manifest = ArtifactManifest(d)
    manifest.record(d / "g_head.npz", stage="retrieval", command="c")
    out = manifest.save()
    doc = json.loads(out.read_text())
    assert doc["schema"] == 1
    assert doc["suite"] == "long"
    assert doc["graph_dir"] == str(d.resolve())
    assert set(doc["artifacts"]["g_head.npz"]) == {
        "stage", "sha256", "bytes", "mtime", "graph_hash", "git_commit", "command", "written_at",
    }


# ==========================================================================================
# verify: the empty-list-means-clean contract, and every problem kind
# ==========================================================================================
def _built_build_dir(tmp_path) -> tuple[Path, ArtifactManifest]:
    d = tmp_path / "long" / "graph_20260819_191549" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_nodes.npz", q=np.zeros(3))
    _touch_npz(d / "g_edges.npz", src=np.zeros(3))
    _touch_npz(d / "g_segments.npz", graph_hash=np.array("h1"), start_node=np.zeros(2))
    _touch_npz(d / "g_lang.npz", e_task=np.zeros(4))
    _touch_npz(d / "a_pi_raw.npz", a=np.zeros(2))
    (d / "action_scale.json").write_text("{}")
    # constants.json is deliberately NOT recorded here and never asserted on below -- it is not in
    # REQUIRED_ARTIFACTS["build"] (opt-in C0 calibration, see the module docstring), so a healthy
    # directory need not carry one at all. See test_constants_json_is_not_required_by_any_stage.
    manifest = ArtifactManifest(d)
    manifest.record(d / "g_nodes.npz", stage="build", command="python -m robot_ood.cli build graph")
    manifest.record(d / "g_edges.npz", stage="build", command="python -m robot_ood.cli build graph")
    manifest.record(d / "g_segments.npz", stage="segment", command="python -m robot_ood.cli build segments")
    manifest.record(d / "g_lang.npz", stage="lang", command="python -m robot_ood.cli build language")
    manifest.record(d / "a_pi_raw.npz", stage="cache", command="python -m robot_ood.cli cache-policy")
    manifest.record(d / "action_scale.json", stage="calibrate", command="python -m robot_ood.cli build calibrate")
    return d, manifest


def test_verify_empty_list_when_the_build_set_is_complete_and_consistent(tmp_path):
    _, manifest = _built_build_dir(tmp_path)
    assert manifest.verify(REQUIRED_ARTIFACTS["build"]) == []


def test_verify_reports_a_missing_file_naming_it_and_its_command(tmp_path):
    d, manifest = _built_build_dir(tmp_path)
    (d / "g_segments.npz").unlink()
    problems = manifest.verify(REQUIRED_ARTIFACTS["build"])
    assert len(problems) == 1
    assert "g_segments.npz" in problems[0]
    assert "missing" in problems[0]
    assert "build segments" in problems[0]


def test_verify_reports_a_file_present_but_never_recorded(tmp_path):
    d, manifest = _built_build_dir(tmp_path)
    del manifest.artifacts["g_segments.npz"]
    problems = manifest.verify(REQUIRED_ARTIFACTS["build"])
    assert len(problems) == 1
    assert "g_segments.npz" in problems[0]
    assert "no ARTIFACTS.json entry" in problems[0]


def test_verify_reports_sha256_drift_since_recording(tmp_path):
    d, manifest = _built_build_dir(tmp_path)
    _touch_npz(d / "g_nodes.npz", q=np.ones(9))
    problems = manifest.verify(REQUIRED_ARTIFACTS["build"])
    assert len(problems) == 1
    assert "g_nodes.npz" in problems[0]
    assert "sha256 drift" in problems[0]


def test_constants_json_is_not_required_by_any_stage(tmp_path):
    # C0 calibration is opt-in (robot_ood.core.config.calibrated_constants reads constants.json only under
    # ROBOT_OOD_CONSTANTS) and graph_hash takes constants as an explicit argument rather than reading the
    # directory, so a constants.json appearing AFTER training must never be forced into existence
    # just to satisfy this gate -- see the HASH_FAMILIES/REQUIRED_ARTIFACTS module docstring.
    for artifacts in REQUIRED_ARTIFACTS.values():
        assert "constants.json" not in artifacts


def test_verify_reports_a_missing_graph_hash_stamp_on_a_hash_stamped_artifact(tmp_path):
    d, manifest = _built_build_dir(tmp_path)
    _touch_npz(d / "g_segments.npz", start_node=np.zeros(2))   # rewritten with no graph_hash key
    manifest.record(d / "g_segments.npz", stage="segment", command="python -m robot_ood.cli build segments")
    problems = manifest.verify(REQUIRED_ARTIFACTS["build"])
    assert len(problems) == 1
    assert "g_segments.npz" in problems[0]
    assert "no graph_hash stamp" in problems[0]


def test_verify_disagreeing_graph_hash_stamps_ACROSS_families_is_not_a_problem(tmp_path):
    # g_fcsr.npz (family "fcsr") and g_segments.npz (family "base") do NOT stamp the same hash by
    # design, so disagreement between them is what a healthy directory looks like, never a problem.
    # See HASH_FAMILIES.
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_fcsr.npz", graph_hash=np.array("aaa"))
    _touch_npz(d / "g_segments.npz", graph_hash=np.array("bbb"), start_node=np.zeros(2))
    manifest = ArtifactManifest(d)
    manifest.record(d / "g_fcsr.npz", stage="fcsr", command="cmd-fcsr")
    manifest.record(d / "g_segments.npz", stage="segment", command="cmd-segments")
    assert manifest.verify(["g_fcsr.npz", "g_segments.npz"]) == []


def test_verify_reports_disagreeing_graph_hash_stamps_WITHIN_a_family(tmp_path, monkeypatch):
    # Two artifacts in one family MUST agree, so a disagreement between them is a real problem. The
    # family is patched: only one shipped artifact carries the base hash now.
    monkeypatch.setitem(store.HASH_FAMILIES, "g_other.npz", "base")
    monkeypatch.setattr(store, "HASH_STAMPED_ARTIFACTS",
                        frozenset(store.HASH_FAMILIES) | {"g_other.npz"})
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_segments.npz", graph_hash=np.array("aaa"), start_node=np.zeros(2))
    _touch_npz(d / "g_other.npz", graph_hash=np.array("bbb"))
    manifest = ArtifactManifest(d)
    manifest.record(d / "g_segments.npz", stage="segment", command="cmd-segments")
    manifest.record(d / "g_other.npz", stage="other", command="cmd-other")
    problems = manifest.verify(["g_segments.npz", "g_other.npz"])
    assert len(problems) == 1
    assert "graph_hash" in problems[0]
    assert "disagrees" in problems[0]
    assert "'base'" in problems[0]


def test_verify_against_an_expected_graph_hash_flags_every_disagreeing_file(tmp_path):
    # The caller-supplied graph_hash is the BASE family's reference -- what graph_hash(nodes, edges)
    # returns -- so this uses the base-family artifact, not a locator or an fcsr one.
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_segments.npz", graph_hash=np.array("aaa"), start_node=np.zeros(2))
    manifest = ArtifactManifest(d)
    manifest.record(d / "g_segments.npz", stage="segment", command="cmd-segments")
    problems = manifest.verify(["g_segments.npz"], graph_hash="zzz")
    assert len(problems) == 1
    assert "graph_hash" in problems[0]

def test_verify_agreeing_stamps_against_the_expected_hash_is_clean(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    _touch_npz(d / "g_segments.npz", graph_hash=np.array("aaa"), start_node=np.zeros(2))
    manifest = ArtifactManifest(d)
    manifest.record(d / "g_segments.npz", stage="segment", command="cmd")
    assert manifest.verify(["g_segments.npz"], graph_hash="aaa") == []


def test_verify_missing_manifest_entirely_reports_one_problem_per_required_file(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    for name in REQUIRED_ARTIFACTS["build"]:
        (d / name).write_bytes(b"")
    manifest = ArtifactManifest.load(d)   # no ARTIFACTS.json on disk -- empty
    problems = manifest.verify(REQUIRED_ARTIFACTS["build"])
    assert len(problems) == len(REQUIRED_ARTIFACTS["build"])


def test_verify_every_problem_names_a_regeneration_command_even_with_no_manifest(tmp_path):
    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    manifest = ArtifactManifest.load(d)   # nothing on disk, nothing recorded
    problems = manifest.verify(REQUIRED_ARTIFACTS["deploy"])
    assert len(problems) == len(REQUIRED_ARTIFACTS["deploy"])
    assert all("Regenerate with" in p for p in problems)


# ==========================================================================================
# digest
# ==========================================================================================
def test_digest_is_stable_across_reload(tmp_path):
    d, manifest = _built_build_dir(tmp_path)
    manifest.save()
    reloaded = ArtifactManifest.load(d)
    assert reloaded.digest() == manifest.digest()


def test_digest_changes_when_a_recorded_files_content_changes(tmp_path):
    d, manifest = _built_build_dir(tmp_path)
    before = manifest.digest()
    (d / "constants.json").write_text('{"changed": true}')
    manifest.record(d / "constants.json", stage="calibrate", command="c")
    assert manifest.digest() != before


def test_digest_is_short_and_hex():
    empty = ArtifactManifest(Path("/x"))
    d = empty.digest()
    assert len(d) == 16
    int(d, 16)   # raises ValueError if not hex




# ==========================================================================================
# required-set contracts, cross-checked against the real writers
# ==========================================================================================
def test_deploy_is_the_disjoint_union_of_the_four_stage_sets():
    stages = ("build", "locator", "entry", "fcsr")
    sets = [set(REQUIRED_ARTIFACTS[s]) for s in stages]
    assert set(REQUIRED_ARTIFACTS["deploy"]) == set().union(*sets)
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            assert not a & b


def test_hash_stamped_artifacts_are_a_subset_of_the_deploy_set():
    assert HASH_STAMPED_ARTIFACTS <= set(REQUIRED_ARTIFACTS["deploy"])


def test_the_instruction_channel_is_the_task_table_alone():
    # Requiring the deleted g_lang_head.npz would make every graph built since unloadable; requiring
    # neither would let a deploy run with no instruction channel at all.
    deploy = set(REQUIRED_ARTIFACTS["deploy"])
    assert "g_lang.npz" in deploy and "g_lang_head.npz" not in deploy


# ==========================================================================================
# validate: the stage gate, and seed_graph_dir: the provenance-carrying clone
# ==========================================================================================
def test_validate_reports_a_clean_build_directory_as_ok(tmp_path):
    d, manifest = _built_build_dir(tmp_path)
    manifest.save()
    report = validate(d, stage="build")
    assert report.ok and report.problems == () and report.suite == "long"
    assert report.required == REQUIRED_ARTIFACTS["build"]


def test_validate_carries_every_problem_through_without_raising(tmp_path):
    d, manifest = _built_build_dir(tmp_path)
    manifest.save()
    (d / "g_segments.npz").unlink()
    report = validate(d, stage="build")
    assert not report.ok and any("g_segments.npz" in p for p in report.problems)


def test_validate_refuses_an_unknown_stage(tmp_path):
    d, manifest = _built_build_dir(tmp_path)
    manifest.save()
    with pytest.raises(ValueError, match="unknown stage"):
        validate(d, stage="deployy")


def test_validate_refuses_a_latest_path(tmp_path):
    d = tmp_path / "long" / "latest" / "artifacts"
    d.mkdir(parents=True)
    with pytest.raises(ValueError, match="latest"):
        validate(d, stage="build")


def test_seeding_carries_the_sources_records_so_the_copy_verifies(tmp_path):
    from robot_ood.build.graph import seed_graph_dir

    src, manifest = _built_build_dir(tmp_path)
    manifest.save()
    dst = tmp_path / "long" / "artifacts"
    result = seed_graph_dir(src, dst)
    assert set(result.copied) == set(REQUIRED_ARTIFACTS["build"])
    assert validate(dst, stage="build").ok


def test_seeding_withholds_what_the_new_fcsr_run_exists_to_write(tmp_path):
    from robot_ood.build.graph import seed_graph_dir

    src, manifest = _built_build_dir(tmp_path)
    from robot_ood.data import schema as _schema

    owned = (_schema.FCSR_NPZ,)
    for name in owned:
        _touch_npz(src / name, w=np.zeros(2))
        manifest.record(src / name, stage="fcsr", command="python -m robot_ood.cli train fcsr")
    manifest.save()
    seeded = seed_graph_dir(src, tmp_path / "plain")
    assert set(seeded.withheld) == set(owned)
    assert not set(owned) & set(seeded.copied)


def test_seeding_drops_a_cache_the_replacing_graph_does_not_rewrite(tmp_path):
    from robot_ood.build.graph import seed_graph_dir

    dst = tmp_path / "long" / "artifacts"
    dst.mkdir(parents=True)
    stale = dst / "g_stale.npz"
    _touch_npz(stale, z=np.zeros((2, 3)))
    dst_manifest = ArtifactManifest.load(dst)
    dst_manifest.record(stale, stage="build", command="python -m robot_ood.cli build graph")
    dst_manifest.save()

    src, manifest = _built_build_dir(tmp_path)
    manifest.save()
    result = seed_graph_dir(src, dst)

    assert result.dropped == ("g_stale.npz",)
    assert not stale.exists()
    assert "g_stale.npz" not in ArtifactManifest.load(dst).artifacts


def test_seeding_refuses_a_source_with_no_provenance_to_carry(tmp_path):
    from robot_ood.build.graph import seed_graph_dir

    src = tmp_path / "bare"
    src.mkdir()
    _touch_npz(src / "g_nodes.npz", q=np.zeros(3))
    with pytest.raises(ValueError, match="no ARTIFACTS.json entries"):
        seed_graph_dir(src, tmp_path / "dst")
