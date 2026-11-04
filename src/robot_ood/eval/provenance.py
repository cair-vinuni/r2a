"""The eval ledger, run.json receipts and git state."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


from robot_ood.core.paths import repo_root

REPO_ROOT = repo_root()


class RunStatus(str, enum.Enum):
    """A ledger row's terminal state. RUNNING is written at launch and superseded on completion."""

    RUNNING = "running"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    CRASHED = "crashed"


def default_ledger_path() -> Path:
    from robot_ood.core.paths import default_paths
    return default_paths().outputs("sr", "ledger.jsonl")


@dataclass
class LedgerRecord:
    """One row of ledger.jsonl. Written twice per attempt: running, then the outcome."""

    tag: str
    suite: str
    base_task_suite: str
    axis: str
    mode: str
    status: RunStatus
    ts: str = ""
    gpu: int | None = None
    port: int | None = None
    expected_n: int | None = None
    actual_n: int | None = None
    successes: int | None = None
    rate: float | None = None
    wall_s: float | None = None
    episodes_per_min: float | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    graph_dir: str | None = None
    graph_hash: str | None = None
    g_head_mtime: str | None = None
    result_dir: str | None = None
    log_dir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready payload; stamps ts at write time when the caller left it blank."""
        payload = dataclasses.asdict(self)
        payload["status"] = self.status.value
        payload["ts"] = self.ts or datetime.now(timezone.utc).isoformat(timespec="seconds")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LedgerRecord":
        """Parse one JSONL row, tolerating rows written by an older schema."""
        known = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in payload.items() if k in known}
        for name in ("tag", "suite", "base_task_suite", "axis", "mode"):
            data.setdefault(name, "")
        try:
            data["status"] = RunStatus(data.get("status", "crashed"))
        except ValueError:
            data["status"] = RunStatus.CRASHED
        return cls(**data)


class Ledger:
    """Append-only JSONL run record. Thread-safe: one worker thread per GPU writes to it."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_ledger_path()
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: LedgerRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), default=str)
        with self._lock, open(self._path, "a") as f:
            f.write(line + "\n")

    def read(self) -> list[LedgerRecord]:
        if not self._path.exists():
            return []
        with open(self._path) as f:
            return [LedgerRecord.from_dict(json.loads(line)) for line in f if line.strip()]

    def latest(self, tag: str, mode: str | None = None,
               records: Sequence[LedgerRecord] | None = None) -> LedgerRecord | None:
        """Last (most recent -- JSONL is append-ordered) record for tag, or None."""
        candidates = records if records is not None else self.read()
        match: LedgerRecord | None = None
        for record in candidates:
            if record.tag == tag and (mode is None or record.mode == mode):
                match = record
        return match


def run_config_sha(graph_dir: Path | str | None) -> str:
    """sha256 of the tree's run_config.json, "" when there is none."""
    if not graph_dir:
        return ""
    path = Path(graph_dir) / "run_config.json"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def write_run_json(result_dir: Path, **fields: Any) -> Path:
    """Write <result_dir>/run.json and return its path."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    out = result_dir / "run.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out


def update_run_json(result_dir: Path, **fields: Any) -> Path:
    """Merge fields into an existing <result_dir>/run.json, stamping updated_at."""
    out = Path(result_dir) / "run.json"
    try:
        payload = json.loads(out.read_text())
    except (OSError, ValueError):
        payload = {}
    payload.update(updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out


@dataclass(frozen=True)
class GitState:
    """The repo's commit + dirtiness at launch. Both fields are None when git is unavailable."""

    sha: str | None
    dirty: bool | None

    @classmethod
    def probe(cls, root: Path = REPO_ROOT) -> "GitState":
        return cls(sha=cls._run(["rev-parse", "HEAD"], root), dirty=cls._dirty(root))

    @staticmethod
    def _run(args: Sequence[str], root: Path) -> str | None:
        try:
            proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    @classmethod
    def _dirty(cls, root: Path) -> bool | None:
        out = cls._run(["status", "--porcelain"], root)
        return None if out is None else bool(out)
