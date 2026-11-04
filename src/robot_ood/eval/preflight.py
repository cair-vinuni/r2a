"""GPU, load, benchmark and artifact checks that run before anything launches."""

from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


from robot_ood.eval.bench import Bench
from robot_ood.eval.ladder import LadderConfig
from robot_ood.pipeline.interpreter import resolve_env_python


def resolve_graph_dir(suite: str, env: Mapping[str, str] | None = None) -> Path:
    """The graph-artifact dir (g_nodes.npz / g_edges.npz / g_head.npz) for suite."""
    from robot_ood.core.paths import default_paths
    override = (env or {}).get("GR_GRAPH_DIR", "")
    return Path(override) if override else default_paths().graph(suite)


@dataclass(frozen=True)
class GraphArtifacts:
    """What a resolved graph dir actually contains, for the pre-launch check and the run record."""

    graph_dir: Path
    ok: bool
    missing: tuple[str, ...]
    graph_hash: str | None = None
    g_head_mtime: str | None = None
    artifact_digest: str | None = None

    def as_record(self) -> dict[str, Any]:
        """The subset that goes into run.json / the ledger row."""
        return {
            "graph_dir": str(self.graph_dir),
            "graph_hash": self.graph_hash,
            "g_head_mtime": self.g_head_mtime,
            "artifact_digest": self.artifact_digest,
        }


def graph_artifact_status(graph_dir: Path) -> GraphArtifacts:
    """Inspect a resolved graph dir for the artifact the sentinel needs."""
    import numpy as np
    from robot_ood.data import schema as graph_schema

    graph_dir = Path(graph_dir)
    head = graph_dir / graph_schema.HEAD_NPZ
    missing = [str(head)] if not head.exists() else []
    graph_hash = head_mtime = None

    if head.exists():
        try:
            with np.load(head, allow_pickle=False) as raw:
                if "graph_hash" in raw.files:
                    graph_hash = str(raw["graph_hash"])
        except Exception as exc:   # noqa: BLE001 -- report, never crash a status probe
            missing.append(f"{head} (unreadable: {exc})")
        head_mtime = _iso_mtime(head)
    return GraphArtifacts(
        graph_dir=graph_dir, ok=not missing, missing=tuple(missing),
        graph_hash=graph_hash, g_head_mtime=head_mtime,
    )


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class GpuStatus:
    index: int
    free_gib: float
    total_gib: float
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class LoadStatus:
    load1: float
    ncpu: int
    threshold: float

    @property
    def ok(self) -> bool:
        return self.load1 <= self.threshold


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str


class Preflight:
    """The "is this box ready, and are we being polite" checks."""

    def __init__(self, config: LadderConfig) -> None:
        self._config = config
        self._limits = config.limits

    def gpus(self, indices: Sequence[int] | None = None) -> list[GpuStatus]:
        """Free-memory status per GPU. indices=None reports every visible device."""
        import torch

        device_count = torch.cuda.device_count()
        wanted = list(indices) if indices is not None else list(range(device_count))
        statuses = []
        for index in wanted:
            if not 0 <= index < device_count:
                statuses.append(GpuStatus(
                    index, 0.0, 0.0, False,
                    f"GPU {index} does not exist (device_count={device_count})",
                ))
                continue
            free, total = torch.cuda.mem_get_info(index)
            free_gib, total_gib = free / 1024**3, total / 1024**3
            ok = free_gib >= self._limits.min_free_gib
            reason = "" if ok else (
                f"only {free_gib:.1f} GiB free (< {self._limits.min_free_gib:.1f} GiB threshold)"
            )
            statuses.append(GpuStatus(index, free_gib, total_gib, ok, reason))
        return statuses

    def load(self) -> LoadStatus:
        ncpu = os.cpu_count() or 1
        return LoadStatus(load1=os.getloadavg()[0], ncpu=ncpu,
                          threshold=ncpu * self._limits.max_loadavg_factor)

    def benchmark(self, timeout_s: float = 120.0) -> CheckResult:
        """Verify the benchmark checkout resolves and holds the episodes the ladder expects."""
        bench = self._config.bench
        if not bench.installed:
            return CheckResult(False, (
                f"{bench.id} checkout missing/empty at {bench.root} (no {bench.marker}) -- run "
                f"python -m robot_ood.cli bench setup {bench.id}"
            ))
        if bench.task_glob or bench.task_list:
            return self._task_file_counts(bench)
        python = resolve_env_python(bench.conda_env)
        if python is None:
            return CheckResult(False, f"cannot locate the `{bench.conda_env}` conda env's python")
        snippet = ("from libero_plus_bridge import get_ids_by_category as g; "
                   "d = g('Robot Initial States'); print({k: len(v[0]) for k, v in d.items()})")
        env = dict(os.environ)
        env.update(bench.client_env())
        from robot_ood.eval.launcher import BASE_CHILD_ENV
        env.update(BASE_CHILD_ENV)
        try:
            proc = subprocess.run([str(python), "-c", snippet], cwd=self._config.root, env=env,
                                   capture_output=True, text=True, timeout=timeout_s)
        except Exception as exc:   # noqa: BLE001 -- report, don't crash --check-env
            return CheckResult(False, f"{bench.id} check crashed: {exc}")
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            return CheckResult(False, f"{bench.id} check FAILED (rc={proc.returncode}):\n{tail}")

        counts = self._parse_counts(proc.stdout)
        if counts is None:
            return CheckResult(False, f"{bench.id} check: could not parse output: {proc.stdout!r}")
        # The ladder's own expected_episodes table, keyed by benchmark name; the probe only
        # samples the Robot_Initial_States axis -- the bridge exposes one category per call.
        expected = {
            self._config.bench.task_suites[suite]: table["Robot_Initial_States"]
            for suite, table in self._config.doc["expected_episodes"].items()
        }
        if any(counts.get(name) != n for name, n in expected.items()):
            return CheckResult(False, f"{bench.id} counts mismatch: got {counts}, expected {expected}")
        return CheckResult(True, f"counts OK: {counts}")

    def _task_file_counts(self, bench: Bench) -> CheckResult:
        """Count the tasks per (suite, axis) for a benchmark that ships them as files or a list."""
        counts = {(suite, axis): bench.expected_episodes(suite, axis) or 0
                  for suite in sorted(bench.task_suites) for axis in bench.axes}
        report = "  ".join(f"{suite}/{axis}={n}" for (suite, axis), n in counts.items() if n)
        missing = sorted(f"{suite}/{axis}" for (suite, axis), n in counts.items() if not n)
        if not report:
            return CheckResult(False, f"no task files installed at all -- run "
                               f"python -m robot_ood.cli bench setup {bench.id}")
        # A missing dimension is not a broken install: it is one cell that cannot run, and the
        # launch of that cell refuses on its own.
        note = f"\n  not installed, those cells cannot run: {missing}" if missing else ""
        return CheckResult(True, f"episodes per cell: {report}{note}")

    @staticmethod
    def _parse_counts(stdout: str) -> dict[str, int] | None:
        lines = stdout.strip().splitlines()
        match = re.search(r"\{.*\}", lines[-1]) if lines else None
        if not match:
            return None
        try:
            return ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError):
            return None
