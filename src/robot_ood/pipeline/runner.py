"""Walk the declared steps: decide what is stale, run it, record what it wrote."""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from robot_ood.pipeline.dag import PipelineSpec, Step
from robot_ood.pipeline.jobs import JobResult, run_job, spawn_job
from robot_ood.pipeline.manifest import Manifest, StepRecord, config_hash, git_sha, hash_all

__all__ = ["StepPlan", "StepOutcome", "explain", "run_pipeline"]


@dataclass(frozen=True)
class StepPlan:
    """What one step would do, before it does it.

    Attributes:
        step: The step.
        argv: Its resolved command.
        inputs: Its resolved input paths.
        outputs: Its resolved output paths.
        reason: Why it must run, or None when it is current.
        cfg_hash: Hash of the resolved command and paths.
    """

    step: Step
    argv: list[str]
    inputs: list[str]
    outputs: list[str]
    reason: str | None
    cfg_hash: str

    @property
    def stale(self) -> bool:
        """Whether this step must run."""
        return self.reason is not None

    @property
    def run_argv(self) -> list[str]:
        """argv as executed: a stale forceable step carries --force."""
        # Kept off argv, which cfg_hash is taken from, or the flag would make every step stale.
        if not self.stale or not self.step.forceable or "--force" in self.argv:
            return list(self.argv)
        return [*self.argv, "--force"]


@dataclass(frozen=True)
class StepOutcome:
    """What one step actually did.

    Attributes:
        name: The step name.
        status: run | skip | failed | dry-run.
        seconds: Wall-clock duration.
        exit_code: The process exit code, or 0 for a skip.
        log_path: Where its output went, when it ran.
        reason: Why it ran, or why it was skipped.
    """

    name: str
    status: str
    seconds: float = 0.0
    exit_code: int = 0
    log_path: Path | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Whether this outcome counts toward a zero exit code."""
        return self.status != "failed"


def plan(spec: PipelineSpec, steps: tuple[Step, ...], manifest: Manifest, *,
         force: bool = False) -> list[StepPlan]:
    """Resolve every step and decide, by hash, which ones are stale.

    Args:
        spec: The resolved pipeline configuration.
        steps: The steps to consider, in build order.
        manifest: The record of what ran before.
        force: Treat every step as stale.

    Returns:
        One plan per step, in build order.
    """
    out: list[StepPlan] = []
    for step in steps:
        argv, inputs, outputs = step.argv(spec), step.inputs(spec), step.outputs(spec)
        cfg = config_hash({"argv": argv, "inputs": inputs, "outputs": outputs})
        reason = "--force" if force else manifest.stale(
            step.name, inputs, outputs, cfg, accept=step.accept_rc)
        out.append(StepPlan(step, argv, inputs, outputs, reason, cfg))
    return out


def explain(spec: PipelineSpec, plans: list[StepPlan]) -> list[str]:
    """The --dry-run report: the DAG, and for each step whether it is stale and why.

    Args:
        spec: The resolved pipeline configuration.
        plans: What plan() decided.

    Returns:
        The lines, in print order.
    """
    lines = [f"suite      : {spec.suite}", f"profile    : {spec.profile}",
             f"device     : {spec.device}", f"artifacts  : {spec.artifacts}",
             f"logs       : {spec.log_dir}", ""]
    for p in plans:
        mark = "STALE" if p.stale else "  ok "
        lines.append(f"[{mark}] {p.step.name:<13} {p.reason or 'inputs and outputs unchanged'}")
        lines.append(f"          $ {shlex.join(p.run_argv)}")
        for path in p.outputs:
            lines.append(f"          -> {path}")
        if p.step.note:
            lines.append(f"          note: {p.step.note}")
        lines.append("")
    stale = sum(1 for p in plans if p.stale)
    lines.append(f"{stale} of {len(plans)} step(s) would run.")
    return lines


def _execute(spec: PipelineSpec, p: StepPlan) -> JobResult:
    """Run one step, detached or in the foreground as it declares.

    Args:
        spec: The resolved pipeline configuration.
        p: The step's plan.

    Returns:
        The job's result.
    """
    if not p.step.detach:
        return run_job(p.step.name, p.run_argv, spec.log_dir, env=spec.env())
    job = spawn_job(p.step.name, p.run_argv, spec.log_dir, env=spec.env())
    print(f"  detached pid {job.pid}, log {job.log_path}")
    return job.wait(poll=30.0)


def run_pipeline(spec: PipelineSpec, steps: tuple[Step, ...], *, force: bool = False,
                 ) -> list[StepOutcome]:
    """Run every stale step in order, recording each one before moving on.

    Args:
        spec: The resolved pipeline configuration.
        steps: The steps to consider, in build order.
        force: Run every step regardless of what the manifest says.

    Returns:
        One outcome per step, in build order.
    """
    manifest = Manifest.load(spec.outputs / spec.suite)
    plans = plan(spec, steps, manifest, force=force)

    outcomes: list[StepOutcome] = []
    for p in plans:
        if not p.stale:
            print(f"=== {p.step.name}: SKIP -- inputs and outputs unchanged")
            outcomes.append(StepOutcome(p.step.name, "skip", reason="unchanged"))
            continue
        print(f"\n=== {p.step.name} ({p.reason})\n$ {shlex.join(p.run_argv)}")
        t0 = time.monotonic()
        inputs = hash_all(p.inputs)
        result = _execute(spec, p)
        ok = result.exit_code in p.step.accept_rc
        manifest.record(StepRecord(
            step=p.step.name, command=shlex.join(p.run_argv), exit_code=result.exit_code,
            seconds=result.seconds, git_sha=git_sha(), config_hash=p.cfg_hash,
            inputs=inputs, outputs=hash_all(p.outputs) if ok else {},
        ))
        manifest.save()
        print(f"--- {p.step.name}: exit {result.exit_code} in {time.monotonic() - t0:.0f}s "
              f"({result.log_path})")
        if not ok:
            outcomes.append(StepOutcome(p.step.name, "failed", result.seconds, result.exit_code,
                                        result.log_path, p.reason or ""))
            print(f"!!! {p.step.name} failed -- see {result.log_path}")
            break
        if result.exit_code != 0:
            print(f"!!! {p.step.name}: exit {result.exit_code} is ACCEPTED here -- {p.step.note}")
        outcomes.append(StepOutcome(p.step.name, "run", result.seconds, result.exit_code,
                                    result.log_path, p.reason or ""))
    return outcomes
