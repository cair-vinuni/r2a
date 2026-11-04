"""The StableVLA server rewrites a checkpoint's config.json once, under a lock, never per launch."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def server(monkeypatch):
    fake_utils = types.ModuleType("experiments.robot.openvla_utils")
    fake_utils.calls = []

    def update_auto_map(checkpoint):
        fake_utils.calls.append(checkpoint)
        path = Path(checkpoint) / "config.json"
        doc = json.loads(path.read_text())
        doc["auto_map"] = {"AutoConfig": "configuration_prismatic.OpenVLAConfig",
                           "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"}
        path.write_text(json.dumps(doc))

    fake_utils.update_auto_map = update_auto_map
    pkg = types.ModuleType("experiments"); robot = types.ModuleType("experiments.robot")
    robot.openvla_utils = fake_utils
    monkeypatch.setitem(sys.modules, "experiments", pkg)
    monkeypatch.setitem(sys.modules, "experiments.robot", robot)
    monkeypatch.setitem(sys.modules, "experiments.robot.openvla_utils", fake_utils)
    spec = importlib.util.spec_from_file_location(
        "stablevla_policy_server", Path("evals/stablevla/stablevla_policy_server.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._serialize_auto_map_update()
    return module, fake_utils


def test_a_config_already_pointing_at_openvla_is_left_alone(tmp_path, server):
    module, utils = server
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": module._AUTO_MAP, "x": 1}))
    utils.update_auto_map(str(tmp_path))
    assert utils.calls == []
    assert not (tmp_path / ".config.lock").exists()


def test_a_config_without_the_map_is_rewritten_once(tmp_path, server):
    module, utils = server
    (tmp_path / "config.json").write_text(json.dumps({"x": 1}))
    utils.update_auto_map(str(tmp_path))
    utils.update_auto_map(str(tmp_path))
    assert utils.calls == [str(tmp_path)]
    assert json.loads((tmp_path / "config.json").read_text())["auto_map"] == module._AUTO_MAP


def test_a_missing_checkpoint_dir_is_ignored(tmp_path, server):
    _, utils = server
    utils.update_auto_map(str(tmp_path / "nope"))
    assert utils.calls == []
