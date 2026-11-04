"""The one train/deploy decision of which forward kinematics poses a suite's corpus."""

from __future__ import annotations

import json

import pytest

from robot_ood.core.kinematics import (
    PandaKinematics,
    ProxyKinematics,
    kinematics_for,
)
from robot_ood.core.paths import suite_from_artifacts
from robot_ood.data.store import recorded_suite

PROXY = "proxy_suite"


@pytest.fixture
def proxy_tree(monkeypatch):
    """A tree whose corpus poses the end effector off the observation."""
    import robot_ood.core.paths as paths

    real = paths.suite_config
    monkeypatch.setattr(paths, "suite_config",
                        lambda suite: {"ee_from_obs": True} if suite == PROXY else real(suite))


def test_a_suite_without_ee_from_obs_is_posed_by_the_panda_chain():
    assert isinstance(kinematics_for(""), PandaKinematics)
    assert isinstance(kinematics_for("nonexistent-suite"), PandaKinematics)


def test_an_ee_from_obs_tree_is_posed_off_the_observation(proxy_tree):
    assert isinstance(kinematics_for(PROXY), ProxyKinematics)


def test_suite_is_read_off_the_artifact_layout():
    assert suite_from_artifacts("outputs/long/run0/artifacts/g_nodes.npz") == "long"
    assert suite_from_artifacts("outputs/authority_ref/artifacts") == "authority_ref"
    assert suite_from_artifacts("/tmp/somewhere/g_nodes.npz") == "<suite>"


def _manifest(tmp_path, suite):
    graph_dir = tmp_path / "anywhere" / "artifacts"
    graph_dir.mkdir(parents=True)
    (graph_dir / "ARTIFACTS.json").write_text(json.dumps({"schema": 1, "suite": suite, "artifacts": {}}))
    return graph_dir


def test_a_graph_dir_is_posed_by_the_suite_its_manifest_records(tmp_path, proxy_tree):
    assert isinstance(kinematics_for(recorded_suite(_manifest(tmp_path, PROXY))), ProxyKinematics)
    assert isinstance(kinematics_for(recorded_suite(_manifest(tmp_path / "b", "long"))), PandaKinematics)


def test_the_recorded_suite_is_never_guessed_off_the_path(tmp_path):
    graph_dir = tmp_path / "outputs" / PROXY / "artifacts"
    graph_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no recorded suite"):
        recorded_suite(graph_dir)
    (graph_dir / "ARTIFACTS.json").write_text(json.dumps({"schema": 1, "artifacts": {}}))
    with pytest.raises(FileNotFoundError, match="no recorded suite"):
        recorded_suite(graph_dir)


def test_joint_space_predicate_tracks_ee_from_obs(tmp_path, proxy_tree):
    from robot_ood.core.kinematics import is_joint_space

    assert is_joint_space("")
    assert not is_joint_space(PROXY)
    assert is_joint_space(recorded_suite(_manifest(tmp_path, "long")))
    assert not is_joint_space(recorded_suite(_manifest(tmp_path / "f", PROXY)))


def test_clip_bounds_are_the_panda_limits_only_on_a_joint_space_arm():
    import numpy as np

    from robot_ood.train.data import joint_clip_bounds
    from robot_ood.train.drift import PANDA_JOINT_HI, PANDA_JOINT_LO

    lo, hi = joint_clip_bounds(7, joint_space=True)
    np.testing.assert_array_equal(lo, PANDA_JOINT_LO[:7])
    np.testing.assert_array_equal(hi, PANDA_JOINT_HI[:7])
    lo, hi = joint_clip_bounds(7, joint_space=False)
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))


def test_the_joint_space_entry_corrector_is_off_for_a_proxy_robot(proxy_tree):
    from robot_ood.runtime.harness import DeployHooks

    assert DeployHooks._load_entry(PROXY) is None


LANE = "lane_suite"


@pytest.fixture
def lane_tree(monkeypatch):
    """A tree that is one arm of a joint-space robot: its rows are joints, posed as a proxy."""
    import robot_ood.core.paths as paths

    real = paths.suite_config
    monkeypatch.setattr(paths, "suite_config",
                        lambda suite: {"ee_from_obs": True, "arm": "left"} if suite == LANE
                        else {"ee_from_obs": True} if suite == PROXY else real(suite))


def test_the_entry_corrector_walks_a_joint_space_robot_and_one_arm_of_one(lane_tree):
    from robot_ood.core.kinematics import entry_walks

    assert entry_walks("")
    assert entry_walks(LANE)
    assert not entry_walks(PROXY)
    assert isinstance(kinematics_for(LANE), ProxyKinematics)


def test_a_lane_graph_loads_its_entry_nets_beside_the_sentinel(tmp_path, lane_tree, monkeypatch):
    import robot_ood.runtime.harness as harness
    from robot_ood.runtime.harness import DeployHooks

    seen = {}
    monkeypatch.setattr(harness.DeployHooks, "_entry_for", staticmethod(lambda d: (seen.setdefault("dir", d), "entry")[1]))
    monkeypatch.setattr("robot_ood.data.segments.graph_plan_rows", lambda artifacts, chunk: 5)
    monkeypatch.setattr("robot_ood.runtime.sentinel.Sentinel", lambda cfg, fwm_dir: "sentinel")
    assert DeployHooks.for_graph(LANE, tmp_path, 50).entry == "entry" and seen["dir"] == tmp_path
    assert DeployHooks.for_graph(PROXY, tmp_path, 50).entry is None
