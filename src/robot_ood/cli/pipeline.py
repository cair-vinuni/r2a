"""The all-steps runs behind bare build and bare train, with hash-based staleness."""

from __future__ import annotations

from pathlib import Path

from robot_ood.cli.command import SuiteCommand, SuiteResult
from robot_ood.cli.errors import CLIError
from robot_ood.pipeline.dag import PROFILES, STEPS, PipelineSpec
from robot_ood.pipeline.manifest import Manifest
from robot_ood.pipeline.runner import explain, plan, run_pipeline

__all__ = ["BuildAllCommand", "TrainAllCommand"]


class StageRunCommand(SuiteCommand):
    """One verb group's steps of the offline build, with hash-based staleness."""

    stages: tuple[str, ...] = ()
    stage = "pipeline"
    label = "pipeline"
    done_status = "built"
    update_latest = False               # the steps make their own runs; this one only orders them
    table_headers = ("suite", "status", "ran", "skipped", "failed")

    def spec(self, suite: str) -> PipelineSpec:
        """This run's resolved pipeline configuration.

        Args:
            suite: Suite short-name.

        Returns:
            The spec every step resolves against.
        """
        profile = self.opt("profile", "full")
        if profile not in PROFILES:
            raise CLIError(f"unknown profile {profile!r}; have: {sorted(PROFILES)}")
        root = self.opt("out", None)
        if root is None and profile == "smoke":
            root = str(Path(self.paths.outputs()).parent / "outputs_smoke")
        return PipelineSpec(
            suite=suite, device=self.args.device or "cuda:0", profile=profile,
            port=self.opt("port", 14190), gpu=self.opt("gpu", None) or "", paths=self.paths,
            outputs_root=Path(root) if root else None,
        )

    def resolve_config(self, suite: str) -> dict:
        spec = self.spec(suite)
        return {
            "command": self.label, "suite": suite, "profile": spec.profile,
            "device": spec.device, "port": spec.port, "gpu": spec.gpu,
            "artifacts": str(spec.artifacts),
            "name": self.args.name,
        }

    def _steps(self):
        """This verb group's steps, in build order."""
        return tuple(s for s in STEPS if s.stage in self.stages)

    def run_suite(self, suite: str) -> SuiteResult:
        """Plan, then either explain or run -- the pipeline supervises steps that log for themselves.

        Args:
            suite: Suite short-name.

        Returns:
            What the run did.
        """
        spec, steps = self.spec(suite), self._steps()
        if self.args.dry_run:
            manifest = Manifest.load(spec.outputs / suite)
            for line in explain(spec, plan(spec, steps, manifest, force=self.args.force)):
                print(line)
            return SuiteResult(suite, "dry-run")

        outcomes = run_pipeline(spec, steps, force=self.args.force)
        counts = {k: sum(1 for o in outcomes if o.status == k) for k in ("run", "skip", "failed")}
        status = "failed" if counts["failed"] else self.done_status
        return SuiteResult(suite, status, metrics={"ran": counts["run"],
                                                   "skipped": counts["skip"],
                                                   "failed": counts["failed"]})

    def row(self, result: SuiteResult) -> list[str]:
        m = result.metrics
        return [result.suite, result.status, str(m.get("ran", "-")), str(m.get("skipped", "-")),
                str(m.get("failed", "-"))]


class BuildAllCommand(StageRunCommand):
    """Bare build: every build-stage step, in order, stale ones only."""

    stages = ("build", "cache")
    stage = "build"
    label = "build"


class TrainAllCommand(StageRunCommand):
    """Bare train: locator, entry, fcsr, then the deploy gate."""

    stages = ("train", "check")
    stage = "train"
    label = "train"
