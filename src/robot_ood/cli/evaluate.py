"""eval: the success-rate driver, and --cell for the one-cell smoke that used to be a shell script.

scripts/run_sr.py stays the backend verbatim (§5). What moves here is the cell selection that used
to be a 25-line python heredoc inside run_one_cell.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from robot_ood.cli.errors import CLIError
from robot_ood.core.paths import repo_root
from robot_ood.eval.bench import Bench, load_bench

__all__ = ["EvalCommand", "one_cell_ladder"]

SMOKE_RUNG = "R1"
ONE_CELL_YAML = Path(".local") / "ladder_one_cell.yaml"
BENCH_FOR_SUITE = {"object": "libero_object", "spatial": "libero_spatial",
                   "goal": "libero_goal", "long": "libero_10"}
TASK_CLASSIFICATION = Path("external/LIBERO-plus/libero/libero/benchmark/task_classification.json")
_CLASSIFICATION_ENV = "LIBERO_PLUS_TASK_CLASSIFICATION"
_LIBERO_HOME_ENV = "LIBERO_HOME"
DRIVER = Path("scripts") / "run_sr.py"


def _root() -> Path:
    """The repository root."""
    return repo_root()


def _task_classification() -> Path:
    """Where LIBERO-Plus's task table actually is.

    Returns:
        The path named by LIBERO_PLUS_TASK_CLASSIFICATION, else the one under LIBERO_HOME, else the
        repo-local checkout. scripts/run_sr.py resolves it the same way, so a shared LIBERO-Plus
        outside the repo works for both.
    """
    named = os.environ.get(_CLASSIFICATION_ENV)
    if named:
        return Path(named)
    home = os.environ.get(_LIBERO_HOME_ENV)
    if home:
        return Path(home) / "libero" / "libero" / "benchmark" / "task_classification.json"
    return _root() / TASK_CLASSIFICATION


def _expected_n(cell: dict) -> int:
    """Recount a cell's instances from the installed LIBERO-plus, not from the ladder's number.

    Args:
        cell: The ladder cell, carrying suite, axis and filter.

    Returns:
        How many tasks of that axis carry an id in the cell's filter.

    Raises:
        CLIError: The filter file or the task classification is missing.
    """
    root = _root()
    try:
        ids = {int(t) for t in (root / cell["filter"]).read_text().split()}
        classification = json.loads(_task_classification().read_text())
    except (OSError, ValueError) as err:
        raise CLIError(f"cannot recount {cell['suite']}/{cell['axis']}: {err}") from err
    axis = cell["axis"].replace("_", " ")
    bench = BENCH_FOR_SUITE[cell["suite"]]
    return sum(1 for e in classification[bench] if e["id"] in ids and e["category"] == axis)


def one_cell_ladder(suite: str, axis: str, mode: str, dest: Path, bench: Bench,
                    trials: int | None = None, source: Path | None = None) -> dict:
    """Write a ladder holding exactly one cell of the smoke rung, with its mode forced.

    Args:
        suite: Suite short-name.
        axis: The perturbation axis, underscored as the ladder spells it.
        mode: The arm to run.
        dest: Where to write the reduced ladder.
        bench: The benchmark the cell runs against.
        trials: Episodes per task, for a benchmark that counts its own tasks.
        source: The ladder file to reduce; None reads the benchmark's own ladder.

    Returns:
        The chosen cell, with expected_n recounted.

    Raises:
        CLIError: The smoke rung holds no such cell.
    """
    root = _root()
    doc = yaml.safe_load((source or root / "configs" / "sr_ladder.yaml").read_text())
    cell = _smoke_cell(doc, suite, axis, bench, trials)
    cell["mode"] = mode
    doc["rungs"] = {SMOKE_RUNG: {"label": f"one-cell smoke ({mode})",
                                 "description": "install check", "cells": [cell]}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(doc, sort_keys=False))
    return cell


def _smoke_cell(doc: dict, suite: str, axis: str, bench: Bench, trials: int | None) -> dict:
    """One cell for the smoke ladder, counted the way this benchmark can count.

    Args:
        doc: The parsed ladder.
        suite: Suite short-name.
        axis: The perturbation axis.
        bench: The benchmark the cell runs against.
        trials: Episodes per task, for a benchmark that counts its own tasks.

    Returns:
        A cells: entry with expected_n filled in.

    Raises:
        CLIError: The smoke rung holds no such cell for a benchmark that cannot count its own.
    """
    counted = bench.expected_episodes(suite, axis, trials)
    if counted is not None:
        return {"suite": suite, "axis": axis, "expected_n": counted}
    cells = [c for rung in doc["rungs"].values() for c in rung.get("cells", [])
             if (c["suite"], c["axis"]) == (suite, axis)]
    if not cells:
        have = sorted({(c["suite"], c["axis"]) for rung in doc["rungs"].values()
                       for c in rung.get("cells", [])})
        raise CLIError(f"the ladder has no cell {(suite, axis)}; it has: {have}")
    cell = dict(cells[0])
    cell.pop("comparators", None)
    cell.pop("gate", None)
    # expected_n in sr_ladder.yaml is the author's LIBERO-plus. Against the public checkout the
    # filter selects a different count, and a stale number marks the cell INCOMPLETE.
    if "filter" in cell:
        cell["expected_n"] = _expected_n(cell)
    return cell


class EvalCommand:
    """Dispatch to scripts/run_sr.py, building the one-cell ladder first under --cell."""

    label = "eval"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> int:
        """Gate the artifact tree, then spawn the driver with the argv it expects.

        Returns:
            The driver's exit code.

        Raises:
            CLIError: --artifacts was given for a tree that runs one graph per arm.
        """
        from robot_ood.core.paths import lane_graph_dirs

        # A subprocess, not an import: scripts/ sits on top of onf and is not importable from it
        # (tests/test_layering.py::test_library_modules_do_not_import_entry_points).
        root = _root()
        env = dict(os.environ)
        env["ROBOT_OOD_VIDEO"] = getattr(self.args, "video", None) or "none"
        lanes = lane_graph_dirs(self.args.suite)
        argv = [sys.executable, str(root / DRIVER), *self._driver_argv(root)]
        if lanes:
            if getattr(self.args, "artifacts", None):
                raise CLIError(f"--artifacts names one tree, but {self.args.suite} runs one graph "
                               f"per arm ({', '.join(sorted(lanes))}); drop the flag.")
            # No GR_GRAPH_DIR: each arm's chain resolves the artifacts of its own lane tree.
            for arm in sorted(lanes):
                self._gate(lanes[arm])
            artifacts = lanes[sorted(lanes)[0]]
        else:
            artifacts = self._artifacts_dir(root)
            if artifacts is not None:
                self._gate(artifacts)
                env["GR_GRAPH_DIR"] = str(artifacts)
        if artifacts is not None:
            # The tree only names the policy it was built against; an explicit --policy is the
            # run's own choice and is already in argv, so it wins.
            policy = self._tree_policy(artifacts)
            if policy and not getattr(self.args, "policy", None):
                argv += ["--policy", policy]
        return subprocess.call(argv, cwd=root, env=env)

    @staticmethod
    def _tree_policy(artifacts: Path) -> str:
        """The policy id run_config.json recorded for this tree, "" for the default."""
        try:
            meta = json.loads((artifacts / "run_config.json").read_text()).get("meta") or {}
        except (OSError, ValueError):
            return ""
        policy = str(meta.get("policy") or "")
        return "" if policy == "stablevla" else policy

    def _artifacts_dir(self, root: Path) -> Path | None:
        """The artifact tree this run evaluates.

        Args:
            root: The repository root.

        Returns:
            --artifacts, else $GR_GRAPH_DIR untouched (None: the driver resolves it), else the
            suite's configured artifacts directory.
        """
        from robot_ood.core.paths import default_paths, suite_config

        named = getattr(self.args, "artifacts", None)
        if named:
            return (root / named).resolve() if not Path(named).is_absolute() else Path(named)
        if os.environ.get("GR_GRAPH_DIR"):
            return None
        configured = suite_config(self.args.suite).get("artifacts")
        if not configured:
            return None
        return Path(default_paths().outputs(configured))

    def _gate(self, artifacts: Path) -> None:
        """Refuse to launch an fcsr arm against a tree that fails the deploy gate.

        Args:
            artifacts: The resolved artifact directory.

        Raises:
            CLIError: The gate reports problems. base skips the gate: it never reads the graph.
        """
        if self.args.mode == "base" or getattr(self.args, "dry_run", False):
            return
        from robot_ood.core.kinematics import entry_walks
        from robot_ood.data.store import validate

        # A corpus whose rows are not joints has nothing for the entry corrector to walk, so its
        # deploy set is the one without the two t = 0 nets -- the rule the driver applies too.
        stage = "deploy" if entry_walks(self.args.suite) else "deploy_no_entry"
        try:
            report = validate(artifacts, stage=stage)
        except ValueError as exc:
            raise CLIError(str(exc)) from exc
        if not report.ok:
            problems = "\n".join(f"  - {p}" for p in report.problems)
        else:
            print(f"[eval] deploy gate OK: {artifacts} artifact_digest={report.digest}")
            return
        raise CLIError(f"{artifacts} failed the deploy gate:\n{problems}")

    def _driver_argv(self, root: Path) -> list[str]:
        """The argv scripts/run_sr.py expects for this invocation.

        Args:
            root: The repository root.

        Returns:
            The driver's argument list.
        """
        a = self.args
        named = getattr(a, "bench", None)
        if not named and getattr(a, "ladder", None):
            named = yaml.safe_load(Path(a.ladder).read_text()).get("bench")
        bench = load_bench(named)
        trials = getattr(a, "trials", None)
        if not a.axis and not (a.rung or bench.default_rung) and len(bench.axes) == 1:
            a.axis = next(iter(bench.axes))
        if a.check_env:
            return ["--check-env", "--bench", bench.id]
        if a.cell and not a.axis:
            raise CLIError("--cell selects one (suite, axis) cell, so it needs --axis")
        if a.cell:
            ladder = root / ONE_CELL_YAML
            cell = one_cell_ladder(a.suite, a.axis, a.mode, ladder, bench, trials,
                                   Path(a.ladder) if a.ladder else None)
            print(f"cell: {cell['suite']} {cell['axis']} {cell['mode']} n = {cell['expected_n']}")
            argv = ["--rung", SMOKE_RUNG, "--mode", a.mode, "--ladder", str(ladder)]
        else:
            argv = ["--mode", a.mode]
            rung = a.rung or bench.default_rung
            if rung:
                argv += ["--rung", rung]
            if a.axis:
                argv += ["--cells", f"{a.suite}:{a.axis}"]
            elif not rung:
                raise CLIError(f"{bench.id} declares no default rung, so a run needs --axis")
            if a.ladder:
                argv += ["--ladder", a.ladder]
        argv += ["--bench", bench.id, "--gpus", str(a.gpus)]
        if getattr(a, "per_gpu", 1) != 1:
            argv += ["--per-gpu", str(a.per_gpu)]
        if getattr(a, "num_clients", None):
            argv += ["--num-clients", str(a.num_clients)]
        if getattr(a, "servers_per_cell", None):
            argv += ["--servers-per-cell", str(a.servers_per_cell)]
        if getattr(a, "policy", None):
            argv += ["--policy", a.policy]
        if trials:
            argv += ["--trials", str(trials)]
        if getattr(a, "variant", None):
            argv += ["--variant", a.variant]
        if getattr(a, "seed", None) is not None:
            argv += ["--seed", str(a.seed)]
        for flag in ("dry_run", "resume", "rerun_incomplete", "score", "allow_partial"):
            if getattr(a, flag, False):
                argv.append("--" + flag.replace("_", "-"))
        if a.results_root:
            argv += ["--results-root", a.results_root]
        return argv
