"""Uniform, reproducible run bookkeeping for a pipeline stage.

Run directories are <root>/<suite>/<stage>_<YYYYmmdd_HHMMSS>[_<name>]/ with fixed filenames
(config.json, manifest.json, metrics.json, log.txt, artifacts/, figures/) — hardcoded in
downstream code and tests. The latest symlink per suite lets stages find the newest run."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import numpy as np

from robot_ood.core.paths import default_paths

# robot-ood/src/robot_ood/core/report.py  ->  parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]

# ---- run-directory layout -------------------------------------------------------------------------
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
ARTIFACTS_DIRNAME = "artifacts"
FIGURES_DIRNAME = "figures"
LOG_FILENAME = "log.txt"
CONFIG_FILENAME = "config.json"
MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.json"
LATEST_NAME = "latest"

# ---- step() column widths -- fixed so log.txt reads as an aligned table ---------------------------
STEP_LABEL_W = 24
STEP_DETAIL_W = 24


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file.

    Args:
        path: File to fingerprint.
        chunk_size: Read-block size, so a multi-GB npz/hdf5 input never has to fit in memory.

    Returns:
        The hex digest. Used to fingerprint the input artifacts a run consumed.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(obj: Any) -> Any:
    """json.dump(..., default=_json_default) -- numpy scalars/arrays and Path -> plain JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return os.fspath(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


# ---- environment probes ----
def _run_git(*args: str) -> str | None:
    """Execute git <args> from repo root; None on any failure."""
    try:
        out = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _git_sha() -> str | None:
    return _run_git("rev-parse", "HEAD") or None


def _git_dirty() -> bool | None:
    status = _run_git("status", "--porcelain")
    return None if status is None else bool(status)


def _torch_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.__version__


# ---- run directory ----
class RunDirectory:
    """One run's <root>/<suite>/<stage>_<ts>[_<name>]/ tree, created on construction."""

    def __init__(self, root: Path, suite: str, stage: str, name: str = "") -> None:
        """Create the run directory and its artifacts/ / figures/ subdirectories.

        Args:
            root: Run-bookkeeping root (outputs/).
            suite: Suite short-name -- the directory this run is grouped under.
            stage: Stage label, the run-dir name prefix.
            name: Optional label appended to the run-dir name.
        """
        timestamp = datetime.now().strftime(TIMESTAMP_FMT)
        dirname = f"{stage}_{timestamp}" + (f"_{name}" if name else "")
        self.path = Path(root) / suite / dirname
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ARTIFACTS_DIRNAME).mkdir(exist_ok=True)
        (self.path / FIGURES_DIRNAME).mkdir(exist_ok=True)

    @property
    def parent(self) -> Path:
        """<root>/<suite>/ -- where the latest pointer lives."""
        return self.path.parent

    def artifact_path(self, filename: str) -> str:
        """<run>/artifacts/<filename> -- where this stage should write its npz/ckpt outputs."""
        return os.fspath(self.path / ARTIFACTS_DIRNAME / filename)

    def figure_path(self, filename: str) -> str:
        """<run>/figures/<filename> -- where this stage should write its png/mp4 diagnostics."""
        return os.fspath(self.path / FIGURES_DIRNAME / filename)

    def write_json(self, filename: str, obj: dict) -> None:
        """Serialize obj to <run>/<filename>."""
        with open(self.path / filename, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)

    def read_json(self, filename: str) -> dict | None:
        """Read <run>/<filename>, or None when it does not exist."""
        path = self.path / filename
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def open_log(self) -> TextIO:
        """Open <run>/log.txt for append."""
        return open(self.path / LOG_FILENAME, "a")

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __str__(self) -> str:
        return os.fspath(self.path)


# ---- the latest pointer ----
class LatestPointer:
    """<root>/<suite>/latest -- the symlink downstream stages resolve through."""

    def __init__(self, parent: Path) -> None:
        """Args: parent: <root>/<suite>/ -- the directory the pointer lives in."""
        self.parent = Path(parent)

    @property
    def path(self) -> Path:
        return self.parent / LATEST_NAME

    def point_at(self, target: str) -> None:
        """Repoint at target (run-dir NAME relative to parent), atomically.

        Raises:
            RuntimeError: latest exists as directory (not symlink). No symlink support falls
                back to latest.txt instead (intended). Raises here because fallback doesn't work.
        """
        tmp = self.parent / f".{LATEST_NAME}.{os.getpid()}.tmp"
        try:
            tmp.unlink(missing_ok=True)
            tmp.symlink_to(target, target_is_directory=True)
        except OSError:
            # No symlink support on this filesystem at all -- the documented, intentional fallback.
            tmp.unlink(missing_ok=True)
            self._write_text_fallback(target)
            return
        try:
            tmp.replace(self.path)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"StageLogger._update_latest: cannot repoint {self.path} at {target!r} -- "
                f"{self.path} exists and is not a symlink, so the atomic replace is refused "
                f"by the OS. Paths.graph() resolves through this pointer, so it is now STUCK reading "
                f"stale content until a human removes or replaces {self.path} by hand "
                f"(the latest.txt fallback is only for filesystems without symlinks -- _write_text_fallback). "
                f"Original error: {e}"
            ) from e

    def _write_text_fallback(self, target: str) -> None:
        """Write latest.txt -- the intentional fallback for filesystems without symlinks."""
        txt_tmp = self.parent / f".{LATEST_NAME}.{os.getpid()}.txt.tmp"
        txt_tmp.write_text(target)
        txt_tmp.replace(self.parent / f"{LATEST_NAME}.txt")


# ==== the manifest record ====
class RunManifest:
    """manifest.json: what produced this run, and how it ended."""

    def __init__(self, stage: str, suite: str, name: str) -> None:
        self.stage = stage
        self.suite = suite
        self.name = name

    def opening(self, inputs: list[str | os.PathLike]) -> dict:
        """The manifest as written on entry, with every input artifact fingerprinted."""
        record = self.base()
        record["inputs"] = {os.fspath(p): sha256_file(p) for p in inputs}
        return record

    def base(self) -> dict:
        """The identity/environment block, probed fresh. Every probe degrades to None."""
        return {
            "stage": self.stage, "suite": self.suite, "name": self.name,
            "git_sha": _git_sha(), "git_dirty": _git_dirty(),
            "python": sys.version.split()[0], "numpy": np.__version__, "torch": _torch_version(),
            "hostname": socket.gethostname(),
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
        }

    @staticmethod
    def close(record: dict, elapsed: float, exc: tuple) -> dict:
        """Stamp record with the outcome.

        Args:
            record: The manifest read back from disk (or a fresh base).
            elapsed: Wall seconds since __enter__.
            exc: (exc_type, exc_val, exc_tb) as handed to __exit__.

        Returns:
            record, mutated in place and returned for chaining.
        """
        exc_type, exc_val, exc_tb = exc
        record["end_time"] = datetime.now().isoformat(timespec="seconds")
        record["elapsed_s"] = elapsed
        if exc_type is None:
            record["status"] = "complete"
        else:
            record["status"] = "failed"
            record["traceback"] = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
        return record


# ==== the context manager callers actually use ====
class StageLogger:
    """Uniform run-directory + progress log for one pipeline stage."""

    def __init__(self, stage: str, suite: str, *, name: str = "", root: str | os.PathLike | None = None,
                 config: dict | None = None, n_stages: int = 0,
                 inputs: list[str | os.PathLike] | None = None, quiet: bool = False,
                 update_latest: bool = True):
        """Initialize a run's logger with the given configuration.

        Args:
            stage: Stage label (graph / train / ...), the run-dir name prefix.
            suite: Suite short-name this run belongs to.
            name: Optional label appended to the run-dir name.
            root: Run-bookkeeping root; None resolves robot_ood.core.paths.Paths.outputs.
            config: The fully-resolved config, written verbatim to config.json.
            n_stages: Expected step count for progress prefix.
            inputs: Input artifact paths to fingerprint into the manifest.
            quiet: Suppress stdout mirror (log.txt still gets every line).
            update_latest: Whether to repoint latest at this run (False for read-only stages).
        """
        self.stage = stage
        self.suite = suite
        self.name = name
        self.n_stages = int(n_stages)
        self.config = dict(config or {})
        self.inputs = list(inputs or [])
        self.quiet = bool(quiet)
        self.update_latest = bool(update_latest)

        root = Path(root).expanduser() if root is not None else default_paths().outputs()
        self._dir = RunDirectory(root, suite, stage, name)
        self._latest = LatestPointer(self._dir.parent)
        self._manifest = RunManifest(stage, suite, name)

        self._step_i = 0
        self._t_start = 0.0
        self._t_last = 0.0
        self._metrics: dict[str, Any] = {}
        self._log_fh: TextIO | None = None

    # ---- context manager ----
    def __enter__(self) -> "StageLogger":
        """Initialize the run: open log.txt, write config.json and manifest.json."""
        self._t_start = self._t_last = time.time()
        self._log_fh = self._dir.open_log()
        self._dir.write_json(CONFIG_FILENAME, self.config)
        self._dir.write_json(MANIFEST_FILENAME, self._manifest.opening(self.inputs))
        if self.update_latest:
            self._latest.point_at(self._dir.path.name)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        """Stamp the manifest with outcome and close log.txt."""
        elapsed = time.time() - self._t_start
        record = self._dir.read_json(MANIFEST_FILENAME) or self._manifest.base()
        record = RunManifest.close(record, elapsed, (exc_type, exc_val, exc_tb))
        self._dir.write_json(MANIFEST_FILENAME, record)
        self.log(f"[{self.stage}] {record['status']} in {elapsed:.1f}s -> {self._dir.path}")
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    # ---- public logging surface ----
    def step(self, label: str, detail: str = "", **fields: Any) -> None:
        """Print and log one progress line with timing."""
        now = time.time()
        elapsed = now - self._t_last
        self._t_last = now
        self._step_i += 1
        prefix = f"[{self.stage} {self._step_i}/{self.n_stages}]" if self.n_stages else f"[{self.stage} {self._step_i}]"
        self.log(f"{prefix} {label:<{STEP_LABEL_W}} {detail:<{STEP_DETAIL_W}} {elapsed:>6.1f}s")
        if fields:
            self.metric(**fields)

    def log(self, msg: str) -> None:
        """Free-form line, printed and appended to log.txt (printing suppressed when quiet)."""
        if not self.quiet:
            print(msg)
        if self._log_fh is not None:
            self._log_fh.write(msg + "\n")
            self._log_fh.flush()

    def metric(self, **kv: Any) -> None:
        """Accumulate metrics and flush metrics.json immediately."""
        self._metrics.update(kv)
        self._dir.write_json(METRICS_FILENAME, self._metrics)

    def artifact_path(self, filename: str) -> str:
        """<run>/artifacts/<filename> -- where this stage should write its npz/ckpt outputs."""
        return self._dir.artifact_path(filename)

    def figure_path(self, filename: str) -> str:
        """<run>/figures/<filename> -- where this stage should write its png/mp4 diagnostics."""
        return self._dir.figure_path(filename)

    @property
    def run_dir(self) -> str:
        """This run's directory."""
        return os.fspath(self._dir)
