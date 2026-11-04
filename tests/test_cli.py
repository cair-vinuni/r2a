"""Tests for robot_ood.cli — python -m robot_ood.cli's argument parsing, --dry-run
contract, suite validation, exit codes, and full end-to-end build/train runs through the CLI
against the fabricated tiny HDF5 fixture from tests.test_builders (not real LIBERO data).

Only build and train are supported subcommands.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from pathlib import Path

import pytest
from test_builders import _write_fixture

from robot_ood.data import schema
from robot_ood.cli.suites import load_configured_suites, resolve_suites
from robot_ood.cli.errors import CLIError
from robot_ood.cli.main import build_parser, main
from robot_ood.data.bucket import DEFAULT_BUCKET
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable


# ==================================================================================================
# isolate ROBOT_OOD_OUTPUTS -- every test in this file must never touch the real repo outputs/ tree
# (tests/test_builders.py's own convention, followed here verbatim).
# ==================================================================================================
@pytest.fixture(autouse=True)
def _isolate_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOT_OOD_OUTPUTS", str(tmp_path / "outputs"))


def _suites_with_demos() -> set[str]:
    """Which of the four suites have their demo HDF5 on this box.

    `--suite all` expands to the suites it can actually resolve a dataset for, so a checkout that
    downloaded only `long` (docs/SETUP.md §4 downloads one suite) legitimately prints one plan, not
    four. Guarding on the data is the difference between "not measured here" and "the CLI is broken".
    """
    from robot_ood.core.paths import default_paths
    paths = default_paths()
    found = set()
    for suite in ("object", "spatial", "goal", "long"):
        try:
            if paths.hdf5(suite).exists():
                found.add(suite)
        except FileNotFoundError:
            pass
    return found


skip_not_all_suites = pytest.mark.skipif(
    _suites_with_demos() != {"object", "spatial", "goal", "long"},
    reason="needs all four suites' demo HDF5; this box has: "
            f"{sorted(_suites_with_demos()) or 'none'} -- see docs/SETUP.md §4",
)


@pytest.fixture
def outputs_root(tmp_path):
    return tmp_path / "outputs"


# ==================================================================================================
# robot_ood.yaml / suite resolution
# ==================================================================================================
def test_load_configured_suites_matches_yaml():
    # A bimanual tree owns no corpus of its own, so `all` leaves it out; naming it still works.
    trees = sorted(p.stem for p in Path("configs/trees").glob("*.yaml")
                   if "lanes:" not in p.read_text())
    assert load_configured_suites() == trees


def test_resolve_suites_all_expands():
    configured = ["object", "spatial", "goal", "long"]
    assert resolve_suites("all", configured) == configured


def test_resolve_suites_single_name():
    configured = ["object", "spatial", "goal", "long"]
    assert resolve_suites("spatial", configured) == ["spatial"]


def test_resolve_suites_unknown_raises_cli_error():
    with pytest.raises(CLIError, match="unknown suite"):
        resolve_suites("bogus", ["object", "spatial", "goal", "long"])


# ==================================================================================================
# argparse construction -- every subcommand parses its own documented flags
# ==================================================================================================
def test_build_parser_has_every_subcommand():
    parser = build_parser()
    sub_actions = [a for a in parser._subparsers._group_actions if hasattr(a, "choices")]
    names = set(sub_actions[0].choices.keys())
    assert names == {"artifacts", "bench", "build", "train", "cache-policy", "eval"}


def test_artifacts_has_every_verb():
    parser = build_parser()
    art = [a for a in parser._subparsers._group_actions if hasattr(a, "choices")][0].choices["artifacts"]
    verbs = [a for a in art._subparsers._group_actions if hasattr(a, "choices")][0].choices
    assert set(verbs) == {"check", "seed", "fetch", "push"}


def test_push_and_fetch_name_the_same_bucket_subdirectory():
    parser = build_parser()
    pushed = parser.parse_args(["artifacts", "push", "long", "--suite", "long"])
    fetched = parser.parse_args(["artifacts", "fetch", "long", "--suite", "long"])
    assert pushed.dest == fetched.source
    assert pushed.bucket == fetched.bucket == DEFAULT_BUCKET


@pytest.mark.parametrize("command", ["build", "build graph", "build calibrate", "train",
                                    "cache-policy", "eval", "artifacts"])
def test_subcommand_help_exits_zero_and_is_non_trivial(capsys, command):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([*command.split(), "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert len(out) > 200      # a real, substantive description -- not a one-liner


def test_build_args_parse_all_flags():
    parser = build_parser()
    args = parser.parse_args([
        "build", "graph", "--suite", "long", "--coarsen", "7", "--device", "cuda:0",
        "--limit-demos", "3", "--limit-tasks", "2", "--hdf5-dir", "/tmp/x",
        "--out", "/tmp/o", "--name", "n", "--seed", "5", "--quiet", "--dry-run", "--force",
    ])
    assert args.suite == "long" and args.coarsen == 7 and args.device == "cuda:0"
    assert args.limit_demos == 3 and args.limit_tasks == 2 and args.hdf5_dir == "/tmp/x"
    assert args.out == "/tmp/o" and args.name == "n" and args.seed == 5
    assert args.quiet is True and args.dry_run is True and args.force is True
    assert args.resume is None


def test_missing_required_suite_is_argparse_usage_error():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["build", "graph"])
    assert exc.value.code == 2      # argparse's own usage-error convention


# ==================================================================================================
# --suite all expansion (through main(), not just resolve_suites in isolation)
# ==================================================================================================
@skip_not_all_suites
def test_suite_all_expands_in_dry_run(capsys, outputs_root):
    rc = main(["build", "graph", "--suite", "all", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    for suite in ("object", "spatial", "goal", "long"):
        assert f"build graph[{suite}]" in out
    assert not outputs_root.exists()


# ==================================================================================================
# unknown suite -> clear message, exit code 1
# ==================================================================================================
def test_unknown_suite_fails_with_exit_1(capsys):
    rc = main(["build", "graph", "--suite", "not_a_real_suite", "--dry-run"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown suite" in err
    assert "not_a_real_suite" in err


# ==================================================================================================
# --dry-run: resolved config printed, outputs/ never touched -- every subcommand
# ==================================================================================================
@pytest.mark.parametrize("argv", [
    ["build", "graph", "--suite", "object"],
    ["train", "locator", "--suite", "object"],
])
@skip_not_all_suites
def test_dry_run_touches_nothing_and_prints_config(capsys, outputs_root, argv):
    rc = main([*argv, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fully-resolved config" in out
    assert not outputs_root.exists()


@skip_not_all_suites
def test_dry_run_build_config_has_every_default_materialized(capsys, outputs_root):
    rc = main(["build", "graph", "--suite", "object", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    start = out.index("{")
    end = out.index("planned stages")
    config = json.loads(out[start:end])
    # every GraphConfig field must be present -- "every default materialized", not just the ones the
    # user happened to pass on the command line
    for field in ("coarsen", "k_sibling", "k_align", "hidden", "layers", "agg",
                  "temp", "topm"):
        assert field in config, field
    assert config["suite"] == "object"
    assert config["hdf5_dir"]      # resolved, non-empty, never a bare placeholder


# ==================================================================================================
# real, tiny, end-to-end build through the CLI (fabricated fixture, no real LIBERO data)
# ==================================================================================================
def test_build_end_to_end_leaves_well_formed_run_dir(tmp_path, outputs_root, capsys):
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=2, n_demos=3, base_t=40)

    rc = main(["build", "graph", "--suite", "object", "--hdf5-dir", str(fixture), "--coarsen", "3"])
    assert rc == 0

    run_dirs = sorted((outputs_root / "object").glob("graph_*"))
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]

    for fname in ("config.json", "manifest.json", "metrics.json", "log.txt"):
        assert (run_dir / fname).exists(), fname
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"

    nodes_path = run_dir / "artifacts" / schema.NODES_NPZ
    edges_path = run_dir / "artifacts" / schema.EDGES_NPZ
    assert nodes_path.exists() and edges_path.exists()
    table = NodeTable.load(nodes_path)
    table.validate()
    es = EdgeSet.load(edges_path)
    es.validate(table)
    assert table.n_demos == 6 and table.n_tasks == 2

    # latest was repointed at this build (build's StageLogger uses update_latest=True)
    assert (outputs_root / "object" / "latest").resolve() == run_dir.resolve()

    out = capsys.readouterr().out
    assert "built" in out


def test_build_skips_identical_config_unless_forced(tmp_path, outputs_root, capsys):
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=1, n_demos=2, base_t=40)
    argv = ["build", "graph", "--suite", "object", "--hdf5-dir", str(fixture), "--coarsen", "3"]

    assert main(argv) == 0
    capsys.readouterr()
    run_dirs_after_first = sorted((outputs_root / "object").glob("graph_*"))
    assert len(run_dirs_after_first) == 1

    assert main(argv) == 0     # identical config -> SKIP, no second run dir
    out = capsys.readouterr().out
    assert "SKIP" in out
    run_dirs_after_second = sorted((outputs_root / "object").glob("graph_*"))
    assert len(run_dirs_after_second) == 1

    # --force -> a genuine rebuild, not a skip (StageLogger's run-dir timestamp is second-resolution
    # -- see robot_ood.core.report.TIMESTAMP_FMT's own docstring note -- so back-to-back calls in a fast
    # test CAN collide on the same directory name; what --force actually guarantees is "did not skip",
    # which the printed status -- not the directory count -- is the reliable way to check here).
    assert main([*argv, "--force"]) == 0
    out = capsys.readouterr().out
    assert "SKIP" not in out
    assert "built" in out


def test_a_changed_input_re_runs_a_config_that_would_otherwise_skip(tmp_path, outputs_root, capsys):
    """A config match alone does not certify a prior run: rewriting an input the run recorded makes
    it stale, which is what lets a retrained upstream artifact propagate."""
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=1, n_demos=2, base_t=40)
    argv = ["build", "graph", "--suite", "object", "--hdf5-dir", str(fixture), "--coarsen", "3"]

    assert main(argv) == 0
    capsys.readouterr()
    manifest = json.loads(
        (sorted((outputs_root / "object").glob("graph_*"))[0] / "manifest.json").read_text())
    recorded = manifest.get("inputs") or {}
    assert recorded, "the run recorded no input digests, so drift cannot be detected"

    for path in recorded:
        pathlib.Path(path).write_bytes(pathlib.Path(path).read_bytes() + b"\0")
        break

    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "SKIP" not in out
    assert "changed since" in out


def test_build_suite_all_builds_every_configured_suite_against_same_fixture(tmp_path, outputs_root):
    """--suite all combined with --hdf5-dir points every configured suite at the SAME tiny
    fixture -- a cheap way to exercise the multi-suite loop end-to-end without four real datasets."""
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=1, n_demos=2, base_t=40)

    rc = main(["build", "graph", "--suite", "all", "--hdf5-dir", str(fixture), "--coarsen", "3"])
    assert rc == 0
    for suite in ("object", "spatial", "goal", "long"):
        run_dirs = sorted((outputs_root / suite).glob("graph_*"))
        assert len(run_dirs) == 1, (suite, run_dirs)


# ==================================================================================================
# train: requires a build first
# ==================================================================================================
def test_train_requires_graph_built_first(outputs_root):
    rc = main(["train", "locator", "--suite", "object", "--epochs", "1"])
    assert rc == 1


def test_train_end_to_end_writes_checkpoint_next_to_graph(tmp_path, outputs_root):
    fixture = tmp_path / "hdf5"
    _write_fixture(fixture, n_tasks=2, n_demos=3, base_t=40)
    assert main(["build", "graph", "--suite", "object", "--hdf5-dir", str(fixture), "--coarsen", "3"]) == 0

    graph_run_dir = sorted((outputs_root / "object").glob("graph_*"))[0]
    rc = main(["train", "entry", "--suite", "object", "--epochs", "1"])
    assert rc == 0

    # checkpoint lands in the canonical artifacts dir build graph seeded, not the train run's
    # own -- update_latest=False on train's StageLogger must leave latest pointed at the build.
    from robot_ood.models.entry.score import SCORE_NPZ
    checkpoint = outputs_root / "object" / "artifacts" / SCORE_NPZ
    assert checkpoint.exists()
    assert (outputs_root / "object" / "latest").resolve() == graph_run_dir.resolve()

    train_run_dirs = sorted((outputs_root / "object").glob("train_*"))
    assert len(train_run_dirs) == 1


# ==================================================================================================
# __main__.py -- the actual python -m robot_ood.cli entry point (subprocess, light coverage)
# ==================================================================================================
def test_python_dash_m_onf_graph_help(monkeypatch):
    env = dict(**{"PYTHONPATH": "src"})
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    full_env.setdefault("OMP_NUM_THREADS", "4")
    full_env.setdefault("MKL_NUM_THREADS", "4")
    full_env.setdefault("OPENBLAS_NUM_THREADS", "4")
    result = subprocess.run(
        [sys.executable, "-m", "robot_ood.cli", "--help"],
        cwd=str(Path(__file__).resolve().parents[1]), env=full_env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "build" in result.stdout and "train" in result.stdout


# ==================================================================================================
# small helper-level unit tests (no I/O)
# ==================================================================================================
def test_pick_prefers_explicit_then_resumed_then_default():
    from robot_ood.cli.runs import pick

    assert pick(5, {"x": 1}, "x", 0) == 5
    assert pick(None, {"x": 1}, "x", 0) == 1
    assert pick(None, {}, "x", 0) == 0


def test_train_resolved_config_keeps_the_resolved_graph_dir(capsys, outputs_root):
    """GraphConfig carries its own graph_dir field (robot_ood.core.config's GR_GRAPH_DIR override), which
    collides with the key train resolves. Splatting the dataclass last used to overwrite the
    resolved path with GraphConfig's default "", so config.json recorded an empty graph_dir and the
    --force/skip guard compared runs on a key that was always blank."""
    rc = main(["train", "locator", "--suite", "object", "--graph-dir", "/tmp/some_graph", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    config = json.loads(out[out.index("{"):out.index("planned:")])
    assert config["graph_dir"] == "/tmp/some_graph"


def test_fmt_metric_shows_ci_spread():
    from robot_ood.cli.render import fmt_metric

    s = fmt_metric({"lo": 0.1, "point": 0.2, "hi": 0.3})
    assert "0.2" in s and "0.1" in s and "0.3" in s


def test_train_task_outputs_match_their_modules():
    """TRAIN_TASKS spells its artifact filenames as literals so argparse can build the help without
    loading a task module. This is the check that keeps those literals equal to the real constants."""
    from robot_ood.models.entry.invdyn import INVDYN_NPZ
    from robot_ood.models.entry.score import SCORE_NPZ
    from robot_ood.data import schema

    from robot_ood.cli.train import TRAIN_TASKS

    assert {name: spec.outputs for name, spec in TRAIN_TASKS.items()} == {
        "locator": (schema.HEAD_NPZ,),
        "fcsr": (schema.FCSR_NPZ,),
        "entry": (SCORE_NPZ, INVDYN_NPZ),
    }


def test_every_train_task_is_a_subcommand_of_train():
    """The eight verbs come from the registry, so a new tasks/<name>.py row IS a new command."""
    from robot_ood.cli.main import build_parser
    from robot_ood.cli.train import TRAIN_TASKS

    for name in TRAIN_TASKS:
        args = build_parser().parse_args(["train", name, "--suite", "object"])
        assert args.task == name


def test_build_verb_outputs_match_their_modules():
    """Same contract as TRAIN_TASKS: the registry's filename literals must equal the real constants."""
    from robot_ood.core.constants import CONSTANTS_JSON
    from robot_ood.core.schemas import SEGMENTS_NPZ
    from robot_ood.data import schema
    from robot_ood.models.language.encoder import LANG_NPZ

    from robot_ood.cli.build import BUILD_VERBS

    assert {name: verb.outputs for name, verb in BUILD_VERBS.items()} == {
        "graph": (schema.NODES_NPZ, schema.EDGES_NPZ),
        "segments": (SEGMENTS_NPZ,),
        "calibrate": (schema.ACTION_SCALE_JSON,),
        "language": (LANG_NPZ,),
        "constants": (CONSTANTS_JSON,),
    }


def test_every_build_verb_is_a_subcommand_of_build():
    """One registry row per artifact family, and each row IS the command that writes it."""
    from robot_ood.cli.build import BUILD_VERBS
    from robot_ood.cli.main import build_parser

    for name, verb in BUILD_VERBS.items():
        args = build_parser().parse_args(["build", name, "--suite", "object"])
        assert args.verb == name and args.command_cls is verb.command_cls


def test_bare_build_dry_run_lists_the_build_and_cache_steps(capsys, tmp_path):
    from robot_ood.pipeline.dag import STEPS

    rc = main(["build", "--suite", "object", "--dry-run", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    for step in STEPS:
        if step.stage in ("build", "cache"):
            assert step.name in out
        else:
            assert step.name not in out
    assert not list(tmp_path.iterdir())


def test_bare_train_dry_run_lists_the_train_and_check_steps(capsys, tmp_path):
    rc = main(["train", "--suite", "object", "--dry-run", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "09_locator" in out and "13_entry" in out and "14b_fcsr" in out and "15_check" in out
    assert "02_graph" not in out and "07_apiraw" not in out


def test_the_smoke_profile_builds_into_its_own_tree(capsys):
    rc = main(["build", "--suite", "object", "--profile", "smoke", "--dry-run"])
    assert rc == 0 and "outputs_smoke" in capsys.readouterr().out


def _eval_args(**over):
    import argparse
    base = dict(suite="long", mode="fcsr", axis=None, cell=False, rung=None, bench=None,
                trials=None, variant=None, gpus="0", ladder=None, results_root=None,
                dry_run=False, resume=False, score=False, check_env=False, allow_partial=False,
                artifacts=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_eval_artifacts_flag_wins_over_suite_default(tmp_path):
    from robot_ood.cli.evaluate import EvalCommand

    cmd = EvalCommand(_eval_args(artifacts=str(tmp_path / "tree")))
    assert cmd._artifacts_dir(tmp_path) == tmp_path / "tree"


def test_eval_env_graph_dir_is_left_alone(tmp_path, monkeypatch):
    from robot_ood.cli.evaluate import EvalCommand

    monkeypatch.setenv("GR_GRAPH_DIR", str(tmp_path / "elsewhere"))
    cmd = EvalCommand(_eval_args())
    assert cmd._artifacts_dir(tmp_path) is None


def test_eval_gate_refuses_an_incomplete_fcsr_tree(tmp_path):
    from robot_ood.cli.errors import CLIError
    from robot_ood.cli.evaluate import EvalCommand

    d = tmp_path / "long" / "artifacts"
    d.mkdir(parents=True)
    cmd = EvalCommand(_eval_args())
    with pytest.raises(CLIError, match="deploy gate"):
        cmd._gate(d)


def test_eval_gate_skips_base(tmp_path):
    from robot_ood.cli.evaluate import EvalCommand

    cmd = EvalCommand(_eval_args(mode="base"))
    cmd._gate(tmp_path / "nowhere")


def test_eval_cell_needs_an_axis(capsys):
    rc = main(["eval", "--suite", "long", "--mode", "base", "--cell"])
    assert rc == 1 and "--axis" in capsys.readouterr().err


def test_train_resolved_config_records_the_ablation(capsys, outputs_root, monkeypatch):
    monkeypatch.setenv("ROBOT_OOD_ABLATE", "graph")
    rc = main(["train", "fcsr", "--suite", "object", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    config = json.loads(out[out.index("{"):out.index("planned:")])
    assert config["ablation"] == "graph"
