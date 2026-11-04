"""One JSON record per pipeline step, and the hash comparison that decides what is stale.

The record is the whole point: [ -e output ] cannot tell a fresh artifact from one built against
an input that has since changed, so every input and output is hashed and compared, never counted.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = ["MANIFEST_JSON", "Manifest", "StepRecord", "config_hash", "file_hash", "git_sha"]

MANIFEST_JSON = "pipeline.json"
_CHUNK = 1 << 20


def file_hash(path: str | os.PathLike[str]) -> str:
    """Hash one file's bytes, or "" when it is not there.

    Args:
        path: The file to hash.

    Returns:
        "sha256:<hex>", or "" for a missing path.
    """
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def config_hash(config: dict[str, Any]) -> str:
    """Hash a resolved config the way a step will run it.

    Args:
        config: The step's resolved knobs.

    Returns:
        "sha256:<hex>" over the canonical JSON form.
    """
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def git_sha() -> str:
    """The working tree's commit, or "" outside a repository.

    Returns:
        The full sha, with "-dirty" appended when the tree has uncommitted changes.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return sha + ("-dirty" if dirty else "")


@dataclass
class StepRecord:
    """What one step did, once.

    Attributes:
        step: The step name.
        command: The argv that ran, joined for reading.
        exit_code: The process exit code.
        seconds: Wall-clock duration.
        git_sha: The tree the step ran from.
        config_hash: Hash of the step's resolved config.
        seed: The run's RNG seed, when it had one.
        inputs: {path: hash} at the moment the step started.
        outputs: {path: hash} at the moment it finished.
        metrics: Whatever the step chose to record.
    """

    step: str
    command: str = ""
    exit_code: int = 0
    seconds: float = 0.0
    git_sha: str = ""
    config_hash: str = ""
    seed: int | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


def hash_all(paths: Iterable[str | os.PathLike[str]]) -> dict[str, str]:
    """Hash every path, keeping the missing ones as empty strings.

    Args:
        paths: The paths to hash.

    Returns:
        {str(path): hash or ""}.
    """
    return {str(p): file_hash(p) for p in paths}


class Manifest:
    """The step records for one artifact tree, read and written as a single JSON file."""

    def __init__(self, path: Path, records: dict[str, StepRecord] | None = None) -> None:
        self.path = path
        self.records: dict[str, StepRecord] = records or {}

    @classmethod
    def load(cls, root: str | os.PathLike[str]) -> "Manifest":
        """Read the manifest under root, or start an empty one.

        Args:
            root: The directory holding pipeline.json.

        Returns:
            The manifest.
        """
        path = Path(root) / MANIFEST_JSON
        if not path.is_file():
            return cls(path)
        raw = json.loads(path.read_text())
        return cls(path, {k: StepRecord(**v) for k, v in raw.get("steps", {}).items()})

    def save(self) -> Path:
        """Write the manifest, creating its directory.

        Returns:
            The path written.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {"steps": {k: asdict(v) for k, v in self.records.items()}}
        self.path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return self.path

    def record(self, rec: StepRecord) -> None:
        """Store one step's record, replacing any earlier run of the same step."""
        self.records[rec.step] = rec

    def stale(self, step: str, inputs: Iterable[str], outputs: Iterable[str],
              cfg_hash: str, accept: tuple[int, ...] = (0,)) -> str | None:
        """Why this step must run again, or None when its recorded result still holds.

        Args:
            step: The step name.
            inputs: Paths the step reads.
            outputs: Paths the step writes.
            cfg_hash: Hash of the step's resolved config.
            accept: Exit codes this step declares as a finished run, not a failure.

        Returns:
            A one-line reason, or None when the step is current.
        """
        rec = self.records.get(step)
        if rec is None:
            return "never run"
        if rec.exit_code not in accept:
            return f"last run exited {rec.exit_code}"
        if rec.config_hash != cfg_hash:
            return "config changed since the recorded run"
        for path in outputs:
            if not Path(path).is_file():
                return f"output is gone: {path}"
            if rec.outputs.get(str(path), "") != file_hash(path):
                return f"output changed since it was written: {path}"
        for path in inputs:
            now = file_hash(path)
            if not now:
                return f"input is gone: {path}"
            if rec.inputs.get(str(path), "") != now:
                return f"input changed: {path}"
        return None
