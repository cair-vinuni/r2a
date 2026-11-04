"""The benchmark registry's contract: every bench.yaml loads, and libero_plus still says what
scripts/run_sr.py used to hardcode."""

import os
from pathlib import Path

import pytest

from robot_ood.eval.bench import Bench, bench_ids, load_bench, repo_root

# What scripts/run_sr.py and evals/libero_plus/score.py declared before the registry existed.
PLUS_AXES = ["Background_Textures", "Camera_Viewpoints", "Language_Instructions",
             "Light_Conditions", "Objects_Layout", "Robot_Initial_States", "Sensor_Noise"]
PLUS_SUITES = {"object": "libero_object", "spatial": "libero_spatial",
               "goal": "libero_goal", "long": "libero_10"}
PLUS_CLIENT_ARGV = [
    "--args.pretrained-path", "/ckpt", "--args.host", "127.0.0.1", "--args.port", "14090",
    "--args.task-suite-name", "libero_10", "--args.category_value", "Objects Layout",
    "--args.num-trials-per-task", "1",
    "--args.shard_index", "3", "--args.num_shards", "8", "--args.video-out-path", "/out",
    "--args.protocol", "openvla_oft",
]

@pytest.fixture(params=bench_ids())
def bench(request) -> Bench:
    """Every registered benchmark in turn."""
    return Bench.load(request.param)


def test_every_bench_loads(bench):
    assert bench.id and bench.results_prefix and bench.axes and bench.task_suites
    assert bench.client.is_file(), f"{bench.id}: client {bench.client} does not exist"


def test_bench_directory_matches_its_id(bench):
    assert (repo_root() / "evals" / bench.id / "bench.yaml").is_file()


def test_result_namespaces_are_distinct():
    prefixes = [Bench.load(name).results_prefix for name in bench_ids()]
    assert len(set(prefixes)) == len(prefixes), f"two benchmarks share a results prefix: {prefixes}"


def test_plus_reproduces_the_hardcoded_axes_and_suites():
    plus = Bench.load("libero_plus")
    assert list(plus.axes) == PLUS_AXES
    assert plus.task_suites == PLUS_SUITES


def test_plus_reproduces_the_hardcoded_client_argv():
    plus = Bench.load("libero_plus")
    argv = plus.client_argv(checkpoint="/ckpt", host="127.0.0.1", port=14090,
                            task_suite="libero_10", category_value="Objects Layout",
                            trials=1, shard=3, num_shards=8, result_dir="/out",
                            protocol="openvla_oft")
    assert argv == PLUS_CLIENT_ARGV


def test_plus_axes_do_not_suffix_the_task_suite():
    plus = Bench.load("libero_plus")
    assert all(plus.task_suite("long", axis) == "libero_10" for axis in plus.axes)


def test_upstream_libero_is_one_unsuffixed_axis_on_the_shared_harness():
    up = Bench.load("libero")
    assert list(up.axes) == ["Original"]
    assert up.task_suite("long", "Original") == "libero_10"
    assert up.task_suites == PLUS_SUITES
    assert up.axis_dir(Path("/r"), "long", "Original") == Path("/r/libero_libero_10/Original")


def test_result_dirs_are_namespaced_by_the_base_suite():
    plus = Bench.load("libero_plus")
    assert plus.axis_dir(Path("/r"), "long", "Objects_Layout") == Path("/r/plus_libero_10/Objects_Layout")


def test_client_env_evicts_the_other_benchmarks_checkout():
    plus, up = Bench.load("libero_plus"), Bench.load("libero")
    inherited = {"PYTHONPATH": os.pathsep.join([str(plus.root), "/somewhere/else"])}
    path = up.client_env(inherited)["PYTHONPATH"].split(os.pathsep)
    assert str(plus.root) not in path
    assert path[0] == str(up.pythonpath[0])
    assert "/somewhere/else" in path


def test_unknown_bench_names_the_ones_that_exist():
    with pytest.raises(FileNotFoundError, match="libero_plus"):
        Bench.load("nope")


def test_load_bench_honours_the_environment(monkeypatch):
    monkeypatch.setenv("ROBOT_OOD_BENCH", "libero")
    assert load_bench().id == "libero"
    assert load_bench("libero_plus").id == "libero_plus"


# The two clients must drive the deploy chain through the same object. A client that reached for
# Sentinel or ScoreEntryCorrector itself would be a second copy of the wiring, and a Pro-vs-Plus
# delta would stop being attributable to the benchmark.
DEPLOY_NAMES = ("Sentinel(", "ScoreEntryCorrector(", "posture_target", "observe_action")


def test_every_client_drives_the_deploy_chain_through_deployhooks(bench):
    source = bench.client.read_text()
    assert "DeployHooks" in source, f"{bench.id}: client does not use DeployHooks"
    for name in DEPLOY_NAMES:
        assert name not in source, f"{bench.id}: client wires {name} itself instead of DeployHooks"


def test_every_bench_declares_a_setup_script(bench):
    assert bench.setup.is_file(), f"{bench.id}: no setup script at {bench.setup}"
    assert os.access(bench.setup, os.X_OK), f"{bench.id}: {bench.setup} is not executable"


def test_axis_dir_sits_under_the_prefixed_suite_dir(bench):
    root = Path("/results")
    suite = next(iter(bench.task_suites))
    axis = next(iter(bench.axes))
    assert bench.axis_dir(root, suite, axis).parent.name.startswith(bench.results_prefix + "_")
