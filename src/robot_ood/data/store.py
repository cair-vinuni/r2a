"""One graph directory: the artifacts it holds, and the provenance manifest recording them.

ArtifactStore resolves and loads the three files that ARE the graph. ARTIFACTS.json below records
what every stage wrote -- filenames, sizes, sha256, the graph_hash stamped inside each npz -- so a
later stage can tell a stale artifact from a current one. The locator calibration and the task
embeddings live above this layer; reaching up for them would invert the dependency the layer test
exists to hold.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

import numpy as np

from robot_ood.data import schema
from robot_ood.data import segments as segments_mod
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable

if TYPE_CHECKING:
    from robot_ood.core.schemas import SegmentTable
from robot_ood.core.report import sha256_file



class ArtifactStore:
    """The artifacts beside one graph, loaded on demand and kept.

    Attributes:
        dir: The directory every artifact is resolved against.
    """

    def __init__(self, graph_dir: str | os.PathLike[str]) -> None:
        """Bind a store to a directory. Nothing is read until something is asked for.

        Args:
            graph_dir: The artifact directory the build stage wrote into.
        """
        self.dir = Path(graph_dir)
        self._cache: dict[tuple[str, bool], Any] = {}

    def path(self, filename: str) -> Path:
        """Where filename lives in this store."""
        return self.dir / filename

    def has(self, filename: str) -> bool:
        """Whether filename is on disk. An optional artifact's absence is not an error."""
        return self.path(filename).is_file()

    def nodes(self, *, strict: bool = True) -> NodeTable:
        """The node table.

        Args:
            strict: Verify the stamped phase convention and run full validation.

        Returns:
            The table.
        """
        return self._once("nodes", strict, lambda: NodeTable.load(self.dir, strict=strict))

    def edges(self, *, strict: bool = True) -> EdgeSet:
        """The edge set.

        Args:
            strict: Run the structural half of validate after loading.

        Returns:
            The edges.
        """
        return self._once("edges", strict, lambda: EdgeSet.load(self.path(schema.EDGES_NPZ), strict))

    def segments(self, *, graph_hash: str | None = None) -> "SegmentTable":
        """The segment table.

        Args:
            graph_hash: When given, the stamp the file must carry -- segment ids are positional, so
                a table built off another graph returns confidently wrong candidates.

        Returns:
            The table.
        """
        # Resolved through the module, not bound at import: the seam tests patch is
        # segments.load_segments, and a from-import would freeze past it.
        src = self.path(schema.SEGMENTS_NPZ)
        stamp = {} if graph_hash is None else {"graph_hash": graph_hash}
        return self._once("segments", bool(stamp),
                          lambda: segments_mod.load_segments(src, **stamp))

    def _once(self, name: str, variant: bool, load: Any) -> Any:
        """Return a cached artifact, loading it the first time this (name, variant) is asked for."""
        key = (name, variant)
        if key not in self._cache:
            self._cache[key] = load()
        return self._cache[key]


# ==== provenance: what each stage wrote, recorded beside it ====

__all__ = [
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "REQUIRED_ARTIFACTS",
    "HASH_STAMPED_ARTIFACTS",
    "REGEN_COMMANDS",
    "recorded_suite",
    "ArtifactRecord",
    "ArtifactManifest",
    "assert_not_latest",
    "sha256_file",
    "record_artifacts",
    "StageReport",
    "validate",
]

MANIFEST_FILENAME = "ARTIFACTS.json"
SCHEMA_VERSION = 1

# Hash-stamped artifacts group into families: base (g_segments.npz), head (g_head.npz),
# fcsr (g_fcsr.npz). Agreement required within family only.
HASH_FAMILIES: Mapping[str, str] = {
    "g_segments.npz": "base",
    "g_head.npz": "head",
    "g_fcsr.npz": "fcsr",
}
HASH_STAMPED_ARTIFACTS: frozenset[str] = frozenset(HASH_FAMILIES)

REGEN_COMMANDS: Mapping[str, str] = {
    "g_nodes.npz": "python -m robot_ood.cli build graph --suite <suite>",
    "g_edges.npz": "python -m robot_ood.cli build graph --suite <suite>",
    "action_scale.json": "python -m robot_ood.cli build calibrate --suite <suite>",
    "g_segments.npz": "python -m robot_ood.cli build segments --suite <suite>",
    "g_fcsr.npz": "python -m robot_ood.cli train fcsr --suite <suite>",
    "g_head.npz": "python -m robot_ood.cli train locator --suite <suite>",
    "g_lang.npz": "python -m robot_ood.cli build language --suite <suite>",
    "a_pi_raw.npz": "python -m robot_ood.cli cache-policy --suite <suite>",
    "g_score.npz": "python -m robot_ood.cli train entry --suite <suite>",
    "g_invdyn.npz": "python -m robot_ood.cli train entry --suite <suite>",
}

# g_lang.npz is the whole instruction channel; constants.json is OPT-IN (not required).
_BUILD: tuple[str, ...] = (
    "g_nodes.npz", "g_edges.npz", "g_segments.npz",
    "action_scale.json", "g_lang.npz", "a_pi_raw.npz",
)
_LOCATOR: tuple[str, ...] = ("g_head.npz",)
_ENTRY: tuple[str, ...] = ("g_score.npz", "g_invdyn.npz")
_FCSR: tuple[str, ...] = ("g_fcsr.npz",)
_DEPLOY: tuple[str, ...] = _BUILD + _LOCATOR + _ENTRY + _FCSR
_DEPLOY_NO_ENTRY: tuple[str, ...] = _BUILD + _LOCATOR + _FCSR
REQUIRED_ARTIFACTS: Mapping[str, tuple[str, ...]] = {
    "build": _BUILD, "locator": _LOCATOR, "entry": _ENTRY, "fcsr": _FCSR, "deploy": _DEPLOY,
    "deploy_no_entry": _DEPLOY_NO_ENTRY,
}

_REPO_ROOT = Path(__file__).resolve().parents[3]


def assert_not_latest(graph_dir: str | os.PathLike) -> Path:
    """Resolve graph_dir and refuse it if it IS or resolves THROUGH a latest path component.

    Args:
        graph_dir: The directory to check, as given (not yet symlink-resolved).

    Returns:
        The resolved, absolute directory.

    Raises:
        ValueError: The given path names a latest component. outputs/ is a symlink shared by
            ten git worktrees and <suite>/latest is repointed by whichever one last built a graph,
            so a run that follows it can silently evaluate a different graph than it trained
            against. Names the resolved directory to pass explicitly instead.
    """
    raw = Path(graph_dir)
    resolved = raw.resolve()
    if "latest" in raw.parts:
        raise ValueError(
            f"{raw}: refuses to run against a `latest` path -- outputs/<suite>/latest is repointed "
            f"by any of ten worktrees' builds, so a run that follows it can silently evaluate a "
            f"different graph than the one it trained against. This resolves to {resolved} right "
            f"now; pass that directory explicitly instead."
        )
    return resolved


_RUN_DIR_RE: Final[re.Pattern[str]] = re.compile(r"latest|.+_\d{8}_\d{6}")


def recorded_suite(graph_dir: str | os.PathLike) -> str:
    """The suite <graph_dir>/ARTIFACTS.json records.

    Args:
        graph_dir: The artifact directory.

    Returns:
        The suite name the build stamped there.

    Raises:
        FileNotFoundError: No manifest there, or one that records no suite. The suite decides
            which forward kinematics poses the corpus; guessing it off the path once posed a
            proxy-robot graph's targets through the Panda chain.
    """
    manifest_path = Path(graph_dir).resolve() / MANIFEST_FILENAME
    suite = json.loads(manifest_path.read_text()).get("suite") if manifest_path.is_file() else None
    if not suite:
        raise FileNotFoundError(
            f"{manifest_path}: no recorded suite. Every build stage stamps it; rebuild with "
            f"`python -m robot_ood.cli build graph --suite <suite>`."
        )
    return str(suite)


def _infer_suite(graph_dir: Path) -> str | None:
    """Best-effort suite name off outputs/<suite>/artifacts, or outputs/<suite>/<run>/artifacts.

    Args:
        graph_dir: A resolved artifact directory.

    Returns:
        The suite component, or None when the path does not follow either layout.
    """
    parts = graph_dir.parts
    if graph_dir.name != "artifacts" or len(parts) < 3:
        return None
    # A run directory is <stage>_<timestamp> or latest; anything else is the suite itself.
    return parts[-3] if _RUN_DIR_RE.fullmatch(parts[-2]) else parts[-2]


def _read_graph_hash(path: Path) -> str | None:
    """The graph_hash stamped inside an npz file, or None."""
    if path.suffix != ".npz":
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            return str(archive["graph_hash"]) if "graph_hash" in archive.files else None
    except Exception:   # noqa: BLE001 -- a corrupt/unreadable npz reports as "no stamp", not a crash
        return None


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One artifact's recorded provenance.

    Attributes:
        stage: The pipeline stage that wrote this file (e.g. "build", "segments", "fcsr").
        sha256: Content hash of the file at record time.
        bytes: File size in bytes at record time.
        mtime: The file's own mtime, ISO-8601 UTC.
        graph_hash: The graph_hash stamped inside the file, or None when this artifact's format
            carries no stamp (see HASH_STAMPED_ARTIFACTS) or none was read.
        git_commit: This repo's HEAD when the record was written, or None if git was unavailable.
        command: The command that produced this file, for the message that regenerates it.
        written_at: When this record was written, ISO-8601 UTC.
    """

    stage: str
    sha256: str
    bytes: int
    mtime: str
    graph_hash: str | None
    git_commit: str | None
    command: str
    written_at: str


class ArtifactManifest:
    """ARTIFACTS.json: one entry per artifact file in a graph's artifact directory.

    Attributes:
        graph_dir: The resolved (never symlink) artifact directory this manifest describes.
        suite: The suite name, from the manifest or inferred off the directory layout.
        artifacts: filename -> ArtifactRecord, in whatever order they were recorded.
    """

    def __init__(
        self, graph_dir: str | os.PathLike, *,
        suite: str | None = None, artifacts: Mapping[str, ArtifactRecord] | None = None,
    ) -> None:
        self.graph_dir = Path(graph_dir).resolve()
        self.suite = suite if suite is not None else _infer_suite(self.graph_dir)
        self.artifacts: dict[str, ArtifactRecord] = dict(artifacts or {})

    @classmethod
    def load(cls, graph_dir: str | os.PathLike) -> "ArtifactManifest":
        """Read <graph_dir>/ARTIFACTS.json.

        Args:
            graph_dir: The artifact directory.

        Returns:
            The manifest. An absent ARTIFACTS.json loads as an EMPTY manifest, not an error -- a
            fresh directory has nothing recorded yet and verify() reports that itself.

        Raises:
            ValueError: The file exists but its schema is not SCHEMA_VERSION -- an unreadable
                version is exactly a wrong stamp, never trusted.
        """
        resolved = Path(graph_dir).resolve()
        manifest_path = resolved / MANIFEST_FILENAME
        if not manifest_path.is_file():
            return cls(resolved)
        doc = json.loads(manifest_path.read_text())
        if doc.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"{manifest_path}: schema={doc.get('schema')!r}, this build reads {SCHEMA_VERSION}. "
                f"Regenerate it by re-recording every artifact in this directory."
            )
        artifacts = {name: ArtifactRecord(**rec) for name, rec in doc.get("artifacts", {}).items()}
        return cls(resolved, suite=doc.get("suite"), artifacts=artifacts)

    def record(
        self, path: str | os.PathLike, *, stage: str, command: str, graph_hash: str | None = None,
    ) -> None:
        """Hash path and add (or replace) its manifest entry.

        Args:
            path: The artifact file, inside graph_dir.
            stage: The pipeline stage that wrote it.
            command: The command that regenerates it.
            graph_hash: This artifact's stamp, or None to read it off the file itself (an npz with
                a "graph_hash" key), which is what every real writer's own save() already embeds.

        Raises:
            FileNotFoundError: Nothing is at path.
        """
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"cannot record {target}: no such file")
        stat = target.stat()
        self.artifacts[target.name] = ArtifactRecord(
            stage=stage,
            sha256=sha256_file(target),
            bytes=stat.st_size,
            mtime=_iso(stat.st_mtime),
            graph_hash=graph_hash if graph_hash is not None else _read_graph_hash(target),
            git_commit=_git_commit(),
            command=command,
            written_at=_iso(datetime.now(timezone.utc).timestamp()),
        )

    def forget(self, name: str) -> bool:
        """Drop one artifact's entry.

        Args:
            name: The filename to remove from the manifest.

        Returns:
            Whether an entry was there to remove.
        """
        return self.artifacts.pop(name, None) is not None

    def save(self) -> Path:
        """Write <graph_dir>/ARTIFACTS.json.

        Returns:
            The written path.
        """
        doc = {
            "schema": SCHEMA_VERSION,
            "suite": self.suite,
            "graph_dir": str(self.graph_dir),
            "artifacts": {name: asdict(rec) for name, rec in sorted(self.artifacts.items())},
        }
        out = self.graph_dir / MANIFEST_FILENAME
        out.write_text(json.dumps(doc, indent=2))
        return out

    def verify(self, required: Sequence[str], *, graph_hash: str | None = None) -> list[str]:
        """Check required files for presence, provenance and stamp agreement.

        Args:
            required: Filenames that must all be present, on-disk, unchanged since recording.
            graph_hash: Expected graph_hash for stamped artifacts, or None to only require agreement.

        Returns:
            Human-readable problems, empty list if all requirements pass.
        """
        problems: list[str] = []
        stamped: dict[str, str] = {}
        for name in required:
            path = self.graph_dir / name
            record = self.artifacts.get(name)
            regen = record.command if record else REGEN_COMMANDS.get(name, f"rebuild {name}")
            if not path.is_file():
                problems.append(f"{path}: missing. Regenerate with `{regen}`.")
                continue
            if record is None:
                problems.append(
                    f"{path}: present on disk but has no ARTIFACTS.json entry -- its provenance is "
                    f"unknown and is treated as untrusted, not as clean. Regenerate with `{regen}` "
                    f"(which records it) or record it directly with ArtifactManifest.record."
                )
                continue
            actual = sha256_file(path)
            if actual != record.sha256:
                problems.append(
                    f"{path}: sha256 drift since it was recorded ({record.sha256[:12]}... -> "
                    f"{actual[:12]}...) -- the file changed after ARTIFACTS.json was written. "
                    f"Regenerate with `{regen}` and re-record."
                )
                continue
            if name in HASH_STAMPED_ARTIFACTS:
                if record.graph_hash is None:
                    problems.append(
                        f"{path}: no graph_hash stamp recorded -- predates stamping or was written "
                        f"unstamped, and a missing stamp is treated as a wrong one, never trusted. "
                        f"Regenerate with `{regen}`."
                    )
                else:
                    stamped[name] = record.graph_hash

        by_family: dict[str, dict[str, str]] = {}
        for name, h in stamped.items():
            by_family.setdefault(HASH_FAMILIES[name], {})[name] = h
        for family, members in sorted(by_family.items()):
            if graph_hash is not None and family == "base":
                reference_name, reference_hash = "<expected>", graph_hash
            else:
                reference_name = sorted(members)[0]
                reference_hash = members[reference_name]
            for name, h in sorted(members.items()):
                if h != reference_hash:
                    regen = self.artifacts[name].command
                    problems.append(
                        f"{self.graph_dir / name}: graph_hash={h[:12]}... disagrees with "
                        f"{reference_name}={reference_hash[:12]}... within the '{family}' stamp "
                        f"family -- built from a different graph, constants.json or C2 checkpoint. "
                        f"Regenerate with `{regen}`."
                    )
        return problems

    def digest(self) -> str:
        """A short content hash over the whole recorded set, for run.json's artifact_digest.

        Returns:
            16 hex characters, stable across re-saves as long as no recorded file/hash changes.
        """
        parts = [
            f"{name}:{r.sha256}:{r.graph_hash or ''}" for name, r in sorted(self.artifacts.items())
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def record_artifacts(
    graph_dir: str | os.PathLike, *, stage: str, command: str, filenames: Sequence[str],
) -> None:
    """Load, update and save graph_dir's ARTIFACTS.json after a writer finishes.

    Args:
        graph_dir: The artifact directory the files were just written into.
        stage: The pipeline stage that wrote them.
        command: The command line that produced them.
        filenames: Artifact filenames to record; missing files on disk are skipped.
    """
    resolved = Path(graph_dir)
    try:
        manifest = ArtifactManifest.load(resolved)
        for name in filenames:
            path = resolved / name
            if path.is_file():
                manifest.record(path, stage=stage, command=command)
        manifest.save()
    except Exception as exc:   # noqa: BLE001 -- provenance must never fail a finished run
        print(
            f"[robot_ood.data.store] WARNING: failed to record {stage} artifacts in {resolved}: "
            f"{type(exc).__name__}: {exc}", file=sys.stderr,
        )


@dataclass(frozen=True)
class StageReport:
    """Whether one artifact directory satisfies one stage's required set.

    Attributes:
        graph_dir: The resolved directory checked.
        stage: The REQUIRED_ARTIFACTS key asked for.
        suite: The suite the manifest names, or None.
        digest: The manifest's content digest over everything it records.
        required: The filenames this stage requires.
        problems: One line per problem; empty means the stage is satisfied.
    """

    graph_dir: Path
    stage: str
    suite: str | None
    digest: str
    required: tuple[str, ...]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when every required artifact is present, unchanged and consistently stamped."""
        return not self.problems


def validate(graph_dir: str | os.PathLike, *, stage: str,
             graph_hash: str | None = None) -> StageReport:
    """Check a graph directory against one stage's required, recorded artifact set.

    Args:
        graph_dir: The artifact directory to check.
        stage: build | locator | entry | fcsr | deploy.
        graph_hash: Require every stamped artifact to carry exactly this hash; None only
            requires the stamped artifacts to agree with each other.

    Returns:
        The report. Read-only: this never writes into graph_dir.

    Raises:
        ValueError: Unknown stage, or graph_dir names a latest path (see assert_not_latest).
    """
    if stage not in REQUIRED_ARTIFACTS:
        raise ValueError(
            f"unknown stage {stage!r}: expected one of {', '.join(sorted(REQUIRED_ARTIFACTS))}"
        )
    resolved = assert_not_latest(graph_dir)
    manifest = ArtifactManifest.load(resolved)
    required = REQUIRED_ARTIFACTS[stage]
    return StageReport(
        graph_dir=resolved, stage=stage, suite=manifest.suite, digest=manifest.digest(),
        required=required, problems=tuple(manifest.verify(required, graph_hash=graph_hash)),
    )
