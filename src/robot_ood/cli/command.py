"""The base every subcommand shares: resolve config per suite, log, execute, report.

A subcommand declares what it needs (resolve_config, stage_inputs, validate) and what it does
(execute); the loop over suites, the run record and the summary table are here once.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from robot_ood.cli.errors import CLIError
from robot_ood.cli.render import echo_config, print_table, read_json
from robot_ood.cli.runs import find_matching_run, load_resume_config, pick
from robot_ood.cli.suites import load_configured_suites, resolve_suites
from robot_ood.core.paths import Paths, default_paths
from robot_ood.core.report import StageLogger, sha256_file

__all__ = ["SuiteCommand", "SuiteResult"]


@dataclass
class SuiteResult:
    """What one subcommand did for one suite.

    Attributes:
        suite: Suite short-name.
        status: The result status (dry-run, skip, done, failed, etc).
        run_dir: The matching run directory if status is skip.
        metrics: The metrics dict returned by execute.
    """

    suite: str
    status: str
    run_dir: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether this suite counts toward a zero exit code."""
        return self.status != "failed"



class SuiteCommand:
    """The per-suite lifecycle every subcommand shares."""

    # StageLogger stage label, and the <stage>_* run-dir prefix the skip guard scans.
    stage: ClassVar[str] = ""
    # Human-facing subcommand name used in progress/error lines.
    label: ClassVar[str] = ""
    # The summary-table status for a suite that actually ran.
    done_status: ClassVar[str] = "done"
    # Whether this stage's StageLogger repoints <suite>/latest (see the module docstring).
    update_latest: ClassVar[bool] = True
    # Summary-table header row; row must return this many cells.
    table_headers: ClassVar[tuple[str, ...]] = ()

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.paths: Paths = default_paths()
        self._resumed = load_resume_config(args.resume) if getattr(args, "resume", None) else {}
        self.suites = resolve_suites(args.suite, load_configured_suites())
        # Paths.outputs() only RESOLVES a path (it never mkdirs), so computing this eagerly is still
        # compatible with --dry-run's "touches NOTHING under outputs/" contract.
        self.out_root = Path(args.out) if args.out else self.paths.outputs()

    # ---- resolution ------------------------------------------------------------------------------
    def opt(self, key: str, default: Any) -> Any:
        """One knob, resolved: explicit CLI flag > --resumed value > default (see pick)."""
        return pick(getattr(self.args, key, None), self._resumed, key, default)

    def require_graph(self, suite: str, config: dict) -> None:
        """Refuse to run against a suite whose graph has not been built yet."""
        from robot_ood.data import schema

        if not (Path(config["graph_dir"]) / schema.NODES_NPZ).exists():
            raise CLIError(f"no built graph for suite {suite!r} at {config['graph_dir']} -- run "
                           f"`python -m robot_ood.cli build graph --suite {suite}` first")

    # ---- driver ----------------------------------------------------------------------------------
    def run(self) -> int:
        """Run every selected suite, then print the summary table.

        Returns:
            The process exit code (0 on success, 1 on failure).

        Raises:
            CLIError: A config-level precondition failed.
        """
        results: list[SuiteResult] = []
        for suite in self.suites:
            try:
                results.append(self.run_suite(suite))
            except CLIError:
                raise
            except Exception as e:
                traceback.print_exc()
                print(f"[{self.label}] {suite}: FAILED -- {type(e).__name__}: {e}", file=sys.stderr)
                results.append(SuiteResult(suite, "failed"))

        if self.args.dry_run:
            return 0
        print_table(list(self.table_headers), [self.row(r) for r in results])
        return 0 if all(r.ok for r in results) else 1

    def run_suite(self, suite: str) -> SuiteResult:
        """Resolve, guard, and execute this subcommand for one suite."""
        config = self.resolve_config(suite)

        if self.args.dry_run:
            echo_config(f"{self.label}[{suite}]", config)
            print(self.plan_line(suite, config))
            return SuiteResult(suite, "dry-run")

        self.validate(suite, config)

        if not self.args.force:
            hit = find_matching_run(self.out_root, suite, self.stage, config)
            if hit is not None and self.artifacts_present(suite, config):
                drifted = self._drifted_inputs(hit, suite, config)
                if drifted:
                    print(f"[{self.label}] {suite}: re-running -- {', '.join(drifted)} changed "
                          f"since {hit.name}")
                else:
                    print(f"[{self.label}] {suite}: SKIP -- {self.skip_note(hit, config)} "
                          f"(pass --force to re-run)")
                    return SuiteResult(suite, "skip", run_dir=hit)

        echo_config(f"{self.label}[{suite}]", config)
        with self._logger(suite, config) as log:
            metrics = self.execute(suite, config, log)
        return SuiteResult(suite, self.done_status, metrics=metrics)

    def _drifted_inputs(self, hit: Path, suite: str, config: dict) -> list[str]:
        """Input filenames whose bytes no longer match what the matched run recorded."""
        # Retraining an upstream artifact leaves every downstream config identical, so a config
        # match alone cannot certify the prior run.
        recorded = read_json(hit / "manifest.json").get("inputs") or {}
        drifted = []
        for path in self.stage_inputs(suite, config):
            want = recorded.get(path)
            if want is not None and want != sha256_file(path):
                drifted.append(Path(path).name)
        return drifted

    def _logger(self, suite: str, config: dict) -> StageLogger:
        """This run's StageLogger, built from the shared flags plus the per-command hooks."""
        return StageLogger(
            self.stage, suite, name=self.args.name, root=self.args.out, config=config,
            n_stages=self.n_stages(config), inputs=self.stage_inputs(suite, config),
            quiet=self.args.quiet, update_latest=self.update_latest,
        )

    # ---- hooks -----------------------------------------------------------------------------------
    def resolve_config(self, suite: str) -> dict:
        """The fully-resolved config for one suite."""
        raise NotImplementedError

    def execute(self, suite: str, config: dict, log: StageLogger) -> dict:
        """Do the actual work inside an entered StageLogger. Returns the summary-row metrics."""
        raise NotImplementedError

    def row(self, result: SuiteResult) -> list[str]:
        """One SuiteResult -> one summary-table row matching table_headers."""
        raise NotImplementedError

    def plan_line(self, suite: str, config: dict) -> str:
        """The --dry-run "planned stages" line printed under the resolved config."""
        return f"planned stages ({self.n_stages(config)})"

    def n_stages(self, config: dict) -> int:
        """How many StageLogger.step calls this run is expected to make."""
        return 0

    def stage_inputs(self, suite: str, config: dict) -> list[str]:
        """Input artifact paths to fingerprint into the run manifest."""
        return []

    def validate(self, suite: str, config: dict) -> None:
        """Fail loudly, before any expensive work, when a precondition is unmet."""

    def artifacts_present(self, suite: str, config: dict) -> bool:
        """Whether the on-disk artifacts a matching prior run left still exist."""
        return True

    def skip_note(self, hit: Path, config: dict) -> str:
        """The middle of the SKIP line: what was found, and where."""
        return f"identical config already run at {hit}"
