"""HDF5 -> demonstration-graph builders (NodeTable, EdgeSet) for the full episode.

Builds over the full episode, no grasp cut: retrieval must
localize the arm at any point in a rollout, not just up to the grasp.
Semantics, invariants, and grip vs stage channel split documented in docs/method-offline.md.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np

from robot_ood.core import npz
from robot_ood.data import libero
from robot_ood.data.libero import DEFAULT_SUITE, hdf5_name

# The build's stage labels, in run order; the CLI's progress denominator and the library's agree.
GRAPH_STAGE_LABELS: tuple[str, ...] = (
    "load/hdf5", "nodes/coarsen", "nodes/virtual",
    "temporal", "sibling", "align", "approach", "edges/summary",
)
from robot_ood.core.action import ACTION_DIM, POSE_DIM
from robot_ood.core.paths import Paths, default_paths
from robot_ood.core.config import GraphConfig
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable
from robot_ood.core.report import StageLogger
# Virtual entry cloud and ENTRY query class use the same distribution draw:
# log-uniform over the two decades the corpus's 1-NN floor defines (two spellings = two calibrations).
from robot_ood.train.data import entry_perturb, entry_sigma_range, joint_clip_bounds

__all__ = [
    "DemoRecord",
    "DemoBatch",
    "LoadLimits",
    "GraphArtifacts",
    "DemoLoader",
    "NodeBuilder",
    "VirtualNodeSynthesizer",
    "EdgeBuilder",
    "GraphBuilder",
    "measured_grip",
    "stage_labels",
    "build_graph",
    "SeedResult",
    "seed_graph_dir",
]

# Column of actions carrying the commanded gripper signal.
GRIP_ACTION_COL: Final[int] = 6

# Demos shorter than this cannot support a stable finite difference.
MIN_DEMO_FRAMES: Final[int] = 3

# Fallback artifact directory when no logger and no explicit out_dir are given.
_ADHOC_ARTIFACTS: Final[tuple[str, str]] = ("adhoc", "artifacts")


# ==== Logging protocol ====
@runtime_checkable
class LoggerProtocol(Protocol):
    """Structural type for the subset of StageLogger this module uses."""

    def step(self, name: str, detail: str = ...) -> Any: ...

    def metric(self, **kwargs: Any) -> Any: ...

    def log(self, message: str) -> Any: ...

    def artifact_path(self, name: str) -> str: ...


class _NullLogger:
    """No-op logger, so call sites never branch on logger is not None."""

    def step(self, name: str, detail: str = "") -> None:
        return None

    def metric(self, **kwargs: Any) -> None:
        return None

    def log(self, message: str) -> None:
        return None

    def artifact_path(self, name: str) -> str:
        raise NotImplementedError("_NullLogger has no artifact directory")

    @property
    def is_null(self) -> bool:
        return True


def _is_null(logger: LoggerProtocol) -> bool:
    """Whether logger is the no-op placeholder.

    Args:
        logger: Logger to test.

    Returns:
        True if the logger cannot supply an artifact path.
    """
    return getattr(logger, "is_null", False)


# ==== Data structures ====
@dataclass(frozen=True, slots=True)
class LoadLimits:
    """Caps on how much data DemoLoader reads, for tests and smoke runs.

    Attributes:
        tasks: Max number of HDF5 files to read. None means all.
        demos: Max number of demo groups per file. None means all.
    """

    tasks: int | None = None
    demos: int | None = None

    def take_tasks(self, files: Sequence[Path]) -> list[Path]:
        """Apply the task cap.

        Args:
            files: Discovered HDF5 files, in sorted order.

        Returns:
            The capped file list.
        """
        return list(files if self.tasks is None else files[: self.tasks])

    def take_demos(self, keys: Sequence[str]) -> list[str]:
        """Apply the per-file demo cap.

        Args:
            keys: Demo group keys, in read order.

        Returns:
            The capped key list.
        """
        return list(keys if self.demos is None else keys[: self.demos])


@dataclass(slots=True)
class DemoRecord:
    """One demonstration trajectory.

    Attributes:
        q: [T, D] f32 joint positions.
        qdot: [T, D] f32 finite-difference joint velocities.
        grip: [T] u1, 1 = grasping (measured channel).
        stage: [T] i2 release count. Diagnostic only.
        task_id: Sort position of the source HDF5 file.
        ee: [T, 6] f32 concat(obs/ee_pos, obs/ee_ori), the latter an unnormalized axis-angle
            rotation vector.
        a: [T, 7] f32 the demo's own action commands, verbatim.
        arm: Which arm of the robot this demonstration is; 0 on a single-arm robot.
        pair: The episode both arms share, or -1 when this demo has no partner.
    """

    q: np.ndarray
    qdot: np.ndarray
    grip: np.ndarray
    stage: np.ndarray
    task_id: int
    ee: np.ndarray
    a: np.ndarray
    arm: int = 0
    pair: int = -1

    def __len__(self) -> int:
        return int(self.q.shape[0])

    def as_payload(self) -> dict[str, Any]:
        """Shallow dict in the form NodeTable.from_demos expects.

        Returns:
            Dict of the fields, sharing (not copying) the underlying arrays.
        """
        return {
            "q": self.q,
            "qdot": self.qdot,
            "grip": self.grip,
            "stage": self.stage,
            "task_id": self.task_id,
            "ee": self.ee,
            "a": self.a,
            "arm": self.arm,
            "pair": self.pair,
        }


@dataclass(slots=True)
class DemoBatch:
    """All demos loaded from one HDF5 directory.

    Attributes:
        demos: Loaded trajectories, in global demo-id order.
        task_names: Task name per source file, indexed by task_id.
    """

    demos: list[DemoRecord] = field(default_factory=list)
    task_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.demos)

    @property
    def n_frames(self) -> int:
        """Total raw frames across all demos."""
        return sum(len(d) for d in self.demos)

    def as_payloads(self) -> list[dict[str, Any]]:
        """Per-demo dicts for NodeTable.from_demos.

        Returns:
            List of payload dicts, in global demo-id order.
        """
        return [d.as_payload() for d in self.demos]


@dataclass(slots=True)
class GraphArtifacts:
    """Result of a full graph build.

    Attributes:
        suite: Suite short-name that was built.
        hdf5_dir: Resolved input directory.
        nodes_path: Written g_nodes.npz.
        edges_path: Written g_edges.npz.
        n_nodes: Node count after coarsening (includes synthesised entry cloud).
        n_virtual: Synthesised entry nodes.
        n_raw: Raw frame count before coarsening.
        n_demos: Distinct demos.
        n_tasks: Distinct tasks.
        n_stages: Distinct raw stage values (diagnostic only).
        n_edges: Total edges.
        edge_counts: Per-relation edge counts.
    """

    suite: str
    hdf5_dir: Path
    nodes_path: Path
    edges_path: Path
    n_nodes: int
    n_virtual: int
    n_raw: int
    n_demos: int
    n_tasks: int
    n_stages: int
    n_edges: int
    edge_counts: dict[str, int]

    @property
    def coarsen_ratio(self) -> float:
        """Raw frames per surviving node."""
        return self.n_raw / max(1, self.n_nodes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, key-compatible with legacy metrics.json readers.

        Returns:
            Dict with Path fields flattened to str.
        """
        out = dataclasses.asdict(self)
        out["hdf5_dir"] = os.fspath(self.hdf5_dir)
        out["nodes_path"] = os.fspath(self.nodes_path)
        out["edges_path"] = os.fspath(self.edges_path)
        return out


# ==== Pure array labelling ====
def measured_grip(
    gripper_states: np.ndarray | None,
    action_grip_cmd: np.ndarray,
) -> np.ndarray:
    """Per-frame grasp flag from the measured gripper channel.

    Args:
        gripper_states: [T, F] finger positions, or None if unrecorded.
        action_grip_cmd: [T] commanded gripper signal, used as fallback.

    Returns:
        [T] u1 array, 1 = grasping/closed.
    """
    if gripper_states is not None:
        gs = np.asarray(gripper_states, dtype=np.float32)
        return (np.abs(gs).mean(axis=1) < schema.GRIP_OPEN_THR).astype(np.uint8)
    return (np.asarray(action_grip_cmd) > 0.5).astype(np.uint8)


def stage_labels(grip_cmd: np.ndarray) -> np.ndarray:
    """Cumulative gripper-release count. Diagnostic only.

    Args:
        grip_cmd: [T] commanded gripper signal.

    Returns:
        [T] i2 array of releases completed by each frame.
    """
    # Undebounced: cumulative sum over a flicker-prone channel, so one misread shifts every
    # later frame.
    gb = (np.asarray(grip_cmd) > 0.5).astype(np.int8)
    releases = np.cumsum(np.diff(gb) == -1)
    return np.concatenate([[0], releases]).astype(np.int16)


def _finite_difference(q: np.ndarray) -> np.ndarray:
    """Forward difference along axis 0, with the last step duplicated.

    Args:
        q: [T, D] f32 array with T >= 2.

    Returns:
        [T, D] f32 array of per-step deltas.
    """
    qdot = np.empty_like(q)
    np.subtract(q[1:], q[:-1], out=qdot[:-1])
    qdot[-1] = qdot[-2]
    return qdot



# ==== Stage 1 -- DemoLoader ====
class DemoLoader:
    """Reads a directory of LIBERO *.hdf5 files into DemoRecord objects."""

    def __init__(
        self,
        hdf5_dir: str | os.PathLike[str],
        *,
        limits: LoadLimits | None = None,
        logger: LoggerProtocol | None = None,
    ) -> None:
        """Configure a loader.

        Args:
            hdf5_dir: Directory containing *.hdf5 demo files.
            limits: Caps on tasks/demos read. Defaults to no cap.
            logger: Progress sink. Defaults to a no-op logger.
        """
        self._dir = Path(hdf5_dir)
        self._limits = limits or LoadLimits()
        self._logger: LoggerProtocol = logger or _NullLogger()
        # Setup-phase: discover files and derive task names once, never in the read loop.
        self._files: list[Path] = self._discover_files()
        self._task_names: list[str] = [libero.task_name_of(f) for f in self._files]

    @property
    def files(self) -> list[Path]:
        """HDF5 files this loader will read, in task-id order."""
        return list(self._files)

    @property
    def task_names(self) -> list[str]:
        """Task name per file, indexed by task_id."""
        return list(self._task_names)

    def load(self) -> DemoBatch:
        """Read every demo group under the configured directory.

        Returns:
            A DemoBatch. Demos shorter than MIN_DEMO_FRAMES are skipped.
        """
        batch = DemoBatch(task_names=self.task_names)
        for task_id, path in enumerate(self._files):
            batch.demos.extend(self._read_file(path, task_id))
        self._logger.step(
            "load/hdf5", f"{len(batch)} demos, {batch.n_frames:,} frames"
        )
        return batch

    def _discover_files(self) -> list[Path]:
        """Find and cap the HDF5 files under the configured directory.

        Returns:
            Sorted, capped list of file paths.

        Raises:
            FileNotFoundError: If no *.hdf5 files exist.
        """
        return self._limits.take_tasks(libero.demo_files(self._dir))

    def sorted_demo_keys(self, group: Any) -> list[str]:
        """Demo group keys in read order, capped by limits.

        Args:
            group: The data group of an open HDF5 file.

        Returns:
            Ordered, capped demo keys.
        """
        keys = libero.demo_keys(group)
        return self._limits.take_demos(keys)

    def _read_file(self, path: Path, task_id: int) -> Iterator[DemoRecord]:
        """Read all demo groups from one file.

        Args:
            path: HDF5 file to read.
            task_id: Sort position of this file.

        Yields:
            One DemoRecord per usable demo group.
        """
        with libero.open_demos(path) as data:
            for key in self.sorted_demo_keys(data):
                record = self._read_demo(data[key], task_id)
                if record is not None:
                    yield record

    @staticmethod
    def _read_demo(group: Any, task_id: int) -> DemoRecord | None:
        """Convert one HDF5 demo group into a record.

        Args:
            group: A single demo group.
            task_id: Sort position of the owning file.

        Returns:
            The record, or None if the demo is too short to difference.

        Raises:
            ValueError: If actions or the end-effector channels have the wrong shape.
        """
        obs = group["obs"]
        q = np.asarray(obs["joint_states"], dtype=np.float32)
        if len(q) < MIN_DEMO_FRAMES:
            return None

        actions = np.asarray(group["actions"], dtype=np.float32)
        if actions.shape != (len(q), ACTION_DIM):
            raise ValueError(
                f"actions has shape {actions.shape}; expected {(len(q), ACTION_DIM)} with a "
                f"column {GRIP_ACTION_COL} gripper command"
            )
        grip_cmd = actions[:, GRIP_ACTION_COL]

        ee = np.concatenate(
            [
                np.asarray(obs["ee_pos"], dtype=np.float32),
                np.asarray(obs["ee_ori"], dtype=np.float32),
            ],
            axis=1,
        )
        if ee.shape != (len(q), POSE_DIM):
            raise ValueError(
                f"obs/ee_pos + obs/ee_ori give shape {ee.shape}; expected {(len(q), POSE_DIM)}"
            )

        gripper_states = obs["gripper_states"] if "gripper_states" in obs else None

        return DemoRecord(
            q=q,
            qdot=_finite_difference(q),
            grip=measured_grip(gripper_states, grip_cmd),
            stage=stage_labels(grip_cmd),
            task_id=task_id,
            ee=ee,
            a=actions,
        )


# ==== Stage 2 -- NodeBuilder ====
class NodeBuilder:
    """Builds and writes the g_nodes.npz artifact from a DemoBatch."""

    def __init__(
        self,
        *,
        coarsen: int = schema.COARSEN,
        logger: LoggerProtocol | None = None,
    ) -> None:
        """Configure a node builder.

        Args:
            coarsen: Coarsening factor passed to NodeTable.from_demos.
            logger: Progress sink. Defaults to a no-op logger.
        """
        self._coarsen = int(coarsen)
        self._logger: LoggerProtocol = logger or _NullLogger()

    def build(self, batch: DemoBatch) -> NodeTable:
        """Coarsen a batch into a node table.

        Args:
            batch: Loaded demonstrations.

        Returns:
            The in-memory node table.
        """
        # phase is never computed in this module; it comes from NodeTable.from_demos alone.
        table = NodeTable.from_demos(
            batch.as_payloads(),
            task_names=batch.task_names,
            coarsen=self._coarsen,
        )
        self._report(table, n_raw=batch.n_frames)
        return table

    def save(self, table: NodeTable, out_dir: Path) -> Path:
        """Write the node table.

        Args:
            table: Table to persist.
            out_dir: Destination directory or .npz path.

        Returns:
            The written path.
        """
        return Path(table.save(os.fspath(out_dir)))

    def _report(self, table: NodeTable, *, n_raw: int) -> None:
        """Emit node-stage step and metric lines.

        Args:
            table: The built table.
            n_raw: Raw frame count before coarsening.
        """
        n_nodes = len(table)
        ratio = n_raw / max(1, n_nodes)
        n_stages = diagnostic_stage_count(table)
        self._logger.step(
            "nodes/coarsen", f"V={n_nodes:,} ({ratio:.2f}x) stages_diag={n_stages}"
        )
        self._logger.metric(
            n_raw=n_raw,
            n_nodes=n_nodes,
            coarsen_ratio=ratio,
            n_demos=table.n_demos,
            n_tasks=table.n_tasks,
            n_stages=n_stages,
        )


# The readout masks the virtual cloud out, so it can never be a retrieval answer: this turns
# "funnel a lost arm back to the manifold" into an edge, not a candidate result.
class VirtualNodeSynthesizer:
    """Grows the synthesised entry cloud: M perturbed shadows of every entry-stratum node."""

    def __init__(
        self,
        *,
        per_entry: int = schema.VIRTUAL_PER_ENTRY,
        phase_band: float = schema.VIRTUAL_PHASE_BAND,
        seed: int = 0,
        joint_space: bool = True,
        logger: LoggerProtocol | None = None,
    ) -> None:
        """Configure the synthesiser.

        Args:
            per_entry: Virtual nodes per entry node (M). 0 leaves the table untouched.
            phase_band: Phase ceiling defining the entry stratum.
            seed: RNG seed; the cloud is a build-time artifact and must be reproducible.
            joint_space: True clips shadows to the Panda joint limits; False leaves them unclipped.
            logger: Progress sink. Defaults to a no-op logger.

        Raises:
            ValueError: per_entry is negative.
        """
        if per_entry < 0:
            raise ValueError(f"per_entry={per_entry} must be non-negative")
        self._per_entry = int(per_entry)
        self._phase_band = float(phase_band)
        self._seed = int(seed)
        self._joint_space = bool(joint_space)
        self._logger: LoggerProtocol = logger or _NullLogger()

    def grow(self, table: NodeTable) -> NodeTable:
        """Append the cloud to a real-only table.

        Args:
            table: Table to grow. Must not already carry virtual nodes.

        Returns:
            The grown table, or table itself when per_entry is 0 or the stratum is empty.

        Raises:
            ValueError: table already has virtual nodes.
        """
        if table.virtual_mask.any():
            raise ValueError("VirtualNodeSynthesizer: table already carries virtual nodes")
        if self._per_entry == 0:
            return table

        parents = np.nonzero(np.asarray(table.phase) <= self._phase_band)[0]
        if not len(parents):
            self._logger.step("nodes/virtual", f"phase <= {self._phase_band} is empty, skipped")
            return table

        sigma_range = entry_sigma_range(table)
        rng = np.random.RandomState(self._seed)
        dim = table.dim
        q_parent = np.asarray(table.q, dtype=np.float64)
        q_lo, q_hi = joint_clip_bounds(dim, self._joint_space)
        # Parent-major, so adding M does not resample the offsets an earlier M already drew.
        q_virtual = np.stack([
            entry_perturb(q_parent[p], rng, dim, sigma_range, q_lo, q_hi)
            for p in parents for _ in range(self._per_entry)
        ])
        parent_ids = np.repeat(parents, self._per_entry)

        grown = table.with_virtual_nodes(parent_ids, q_virtual)
        self._logger.step(
            "nodes/virtual",
            f"M={self._per_entry} x {len(parents):,} entry nodes -> V={len(grown):,} "
            f"sigma=({sigma_range[0]:.4f}, {sigma_range[1]:.4f}) rad",
        )
        self._logger.metric(
            n_virtual=int(len(grown) - len(table)),
            virtual_per_entry=self._per_entry,
            virtual_phase_band=self._phase_band,
        )
        return grown


def diagnostic_stage_count(table: NodeTable) -> int:
    """Number of distinct raw stage values in a table. Diagnostic only.

    Args:
        table: Node table to inspect.

    Returns:
        max(stage) + 1. Not a structural count -- see schema.RELATIONS.
    """
    return int(table.stage.max()) + 1


# ==== Stage 3 -- EdgeBuilder ====
class EdgeBuilder:
    """Builds and writes the g_edges.npz artifact from a NodeTable."""

    def __init__(
        self,
        *,
        cfg: GraphConfig | None = None,
        device: Any = None,
        logger: LoggerProtocol | None = None,
    ) -> None:
        """Configure an edge builder.

        Args:
            cfg: Relation fan-out configuration. Defaults to GraphConfig.
            device: Torch device for neighbour search. None lets EdgeSet choose.
            logger: Progress sink. Defaults to a no-op logger.
        """
        self._cfg = cfg or GraphConfig()
        self._device = device
        self._logger: LoggerProtocol = logger or _NullLogger()

    def build(self, table: NodeTable) -> EdgeSet:
        """Construct all relations over a node table.

        Args:
            table: Source node table.

        Returns:
            The in-memory edge set.
        """
        edges = EdgeSet.build(
            table,
            k_sib=self._cfg.k_sibling,
            k_align=self._cfg.k_align,
            dilations=tuple(self._cfg.dilations),
            device=self._device,
            logger=None if _is_null(self._logger) else self._logger,
        )
        self._report(edges)
        return edges

    def save(self, edges: EdgeSet, out_dir: Path) -> Path:
        """Write the edge set, creating parent directories as needed.

        Args:
            edges: Edge set to persist.
            out_dir: Destination directory or .npz path.

        Returns:
            The written path.
        """
        # EdgeSet.save does no path resolution of its own; mirrors NodeTable.save's
        # directory-or-file convenience.
        target = out_dir if out_dir.suffix == ".npz" else out_dir / schema.EDGES_NPZ
        target.parent.mkdir(parents=True, exist_ok=True)
        return Path(edges.save(os.fspath(target)))

    def _report(self, edges: EdgeSet) -> None:
        """Emit the aggregate edge summary the acceptance check reads.

        Args:
            edges: The built edge set.
        """
        counts = edges.counts()
        temporal = sum(
            counts[f"{prefix}{d}"]
            for prefix in ("next", "prev")
            for d in schema.DILATIONS
        )
        self._logger.step(
            "edges/summary",
            f"E={len(edges.src):,} temporal={temporal:,} "
            f"sibling={counts['sibling']:,} align={counts['align']:,} "
            f"approach={counts['approach']:,}",
        )
        self._logger.metric(
            edge_counts=counts,
            n_edges=len(edges.src),
            edge_weights=edges.weight_report(),
        )


# ==== Orchestrator ====
class GraphBuilder:
    """Runs the HDF5 -> nodes -> edges chain for one suite."""

    def __init__(
        self,
        suite: str | None = None,
        *,
        hdf5_dir: str | os.PathLike[str] | None = None,
        out_dir: str | os.PathLike[str] | None = None,
        cfg: GraphConfig | None = None,
        coarsen: int | None = None,
        limits: LoadLimits | None = None,
        device: Any = None,
        logger: LoggerProtocol | None = None,
        paths: Paths | None = None,
        seed: int = 0,
    ) -> None:
        """Configure a build.

        Args:
            suite: Suite short-name. None uses DEFAULT_SUITE.
            hdf5_dir: Input directory. None resolves via Paths.hdf5.
            out_dir: Artifact directory. None resolves per-run at build time.
            cfg: Relation fan-out configuration; also carries virtual_per_entry.
            coarsen: Coarsening factor. None uses schema.COARSEN.
            limits: Caps on tasks/demos read.
            device: Torch device for edge construction.
            logger: An already-entered logger. None means build owns one.
            paths: Path resolver. None uses default_paths().
            seed: RNG seed for the virtual entry cloud.
        """
        self._suite = suite or DEFAULT_SUITE
        self._cfg = cfg or GraphConfig()
        self._coarsen = schema.COARSEN if coarsen is None else int(coarsen)
        self._limits = limits or LoadLimits()
        self._seed = int(seed)
        self._device = device
        self._paths = paths or default_paths()
        self._external_logger = logger
        self._logger: LoggerProtocol = logger or _NullLogger()
        self._explicit_out_dir = Path(out_dir) if out_dir is not None else None
        self._hdf5_dir = self._resolve_hdf5_dir(hdf5_dir)

    # -- public API ----------------------------------------------------------------
    @property
    def suite(self) -> str:
        """Suite short-name being built."""
        return self._suite

    @property
    def hdf5_dir(self) -> Path:
        """Resolved input directory."""
        return self._hdf5_dir

    def build(self) -> GraphArtifacts:
        """Run the full chain, validating the on-disk edge artifact.

        Returns:
            The build result.
        """
        # An externally supplied logger is assumed already entered and is never re-entered here;
        # only a self-owned logger goes through the ExitStack.
        with contextlib.ExitStack() as stack:
            if self._external_logger is None:
                self._logger = stack.enter_context(self._make_logger())
            return self._run()

    def build_nodes(self) -> Path:
        """Load demos and write g_nodes.npz.

        Returns:
            The written path.
        """
        batch = DemoLoader(
            self._hdf5_dir, limits=self._limits, logger=self._logger
        ).load()
        builder = NodeBuilder(coarsen=self._coarsen, logger=self._logger)
        from robot_ood.core.paths import suite_config

        joint_space = not bool(suite_config(self._suite).get("ee_from_obs"))
        table = VirtualNodeSynthesizer(
            per_entry=self._cfg.virtual_per_entry, seed=self._seed,
            joint_space=joint_space, logger=self._logger,
        ).grow(builder.build(batch))
        if not joint_space:
            table.ee_base = table.ee_raw
        return builder.save(table, self._resolve_out_dir(schema.NODES_NPZ))

    def build_edges(self, table: NodeTable) -> Path:
        """Build and write g_edges.npz from an in-memory node table.

        Args:
            table: Source node table.

        Returns:
            The written path.
        """
        builder = EdgeBuilder(
            cfg=self._cfg, device=self._device, logger=self._logger
        )
        edges = builder.build(table)
        return builder.save(edges, self._resolve_out_dir(schema.EDGES_NPZ))

    # -- internals -----------------------------------------------------------------
    def _run(self) -> GraphArtifacts:
        """Execute nodes -> edges -> validate against the active logger.

        Returns:
            The build result.
        """
        nodes_path = self.build_nodes()
        # Round-trip through disk: g_nodes.npz is the artifact of record.
        table = NodeTable.load(os.fspath(nodes_path))
        edges_path = self.build_edges(table)

        # Re-load and validate the on-disk artifact: EdgeSet.build already checked the
        # in-memory result, but this catches save/load corruption that check cannot.
        edges = EdgeSet.load(os.fspath(edges_path), strict=True)
        edges.validate(table)

        artifacts = GraphArtifacts(
            suite=self._suite,
            hdf5_dir=self._hdf5_dir,
            nodes_path=nodes_path,
            edges_path=edges_path,
            n_nodes=len(table),
            n_virtual=int(len(table) - table.n_real),
            # q_raw is the pre-coarsening frame cloud: exactly the count DemoBatch
            # reported, recovered from the artifact rather than threaded through.
            n_raw=int(table.q_raw.shape[0]),
            n_demos=table.n_demos,
            n_tasks=table.n_tasks,
            n_stages=diagnostic_stage_count(table),
            n_edges=len(edges.src),
            edge_counts=edges.counts(),
        )
        self._logger.metric(**artifacts.to_dict())
        self._logger.log(
            f"[graph] {self._suite}: V={artifacts.n_nodes:,} "
            f"E={artifacts.n_edges:,} stages_diag={artifacts.n_stages}"
        )
        return artifacts

    def _make_logger(self) -> StageLogger:
        """Construct the run-owned logger with a fully-resolved config snapshot.

        Returns:
            An unentered StageLogger.
        """
        config = {
            "suite": self._suite,
            "hdf5_dir": os.fspath(self._hdf5_dir),
            "coarsen": self._coarsen,
            "seed": self._seed,
            "limits": dataclasses.asdict(self._limits),
            **dataclasses.asdict(self._cfg),
        }
        inputs = [os.fspath(p) for p in sorted(self._hdf5_dir.glob("*.hdf5"))]
        return StageLogger(
            "graph", self._suite, config=config, n_stages=len(GRAPH_STAGE_LABELS), inputs=inputs
        )

    def _resolve_hdf5_dir(self, override: str | os.PathLike[str] | None) -> Path:
        """Resolve the input directory for this suite.

        Args:
            override: Explicit directory, or None to resolve from the suite name.

        Returns:
            The input directory.
        """
        if override is not None:
            return Path(override)
        return Path(self._paths.hdf5(hdf5_name(self._suite)))

    def _resolve_out_dir(self, artifact_name: str) -> Path:
        """Resolve where one artifact should be written.

        Args:
            artifact_name: Artifact filename, e.g. schema.NODES_NPZ.

        Returns:
            An explicit out_dir, the logger's artifact path, or a scratch dir.
        """
        if self._explicit_out_dir is not None:
            return self._explicit_out_dir
        if not _is_null(self._logger):
            return Path(self._logger.artifact_path(artifact_name))
        return Path(self._paths.outputs(*_ADHOC_ARTIFACTS))


# ==== Functional facade ====
def build_graph(
    suite: str,
    *,
    hdf5_dir: str | os.PathLike[str] | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    cfg: GraphConfig | None = None,
    coarsen: int | None = None,
    limits: LoadLimits | None = None,
    device: Any = None,
    logger: LoggerProtocol | None = None,
    seed: int = 0,
) -> GraphArtifacts:
    """Build g_nodes.npz and g_edges.npz for one suite.

    Args:
        suite: Suite short-name.
        hdf5_dir: Input directory. None resolves from the suite name.
        out_dir: Artifact directory. None resolves to the run's artifacts/.
        cfg: Relation fan-out configuration.
        coarsen: Coarsening factor.
        limits: Caps on tasks/demos read.
        device: Torch device for edge construction.
        logger: An already-entered logger, or None to own one.
        seed: RNG seed for the virtual entry cloud.

    Returns:
        The build result.
    """
    return GraphBuilder(
        suite,
        hdf5_dir=hdf5_dir,
        out_dir=out_dir,
        cfg=cfg,
        coarsen=coarsen,
        limits=limits,
        device=device,
        logger=logger,
        seed=seed,
    ).build()



# ==== Seeding a fresh artifact directory ====

# The files an FCSR run owns; never seeded, they are what it exists to produce.
_FCSR_OWNED: Final[frozenset[str]] = frozenset(
    {"g_fcsr.npz"})


@dataclass(frozen=True)
class SeedResult:
    """What seed_graph_dir carried across, and what it deliberately left behind.

    Attributes:
        source: The directory read.
        dest: The directory written.
        copied: Filenames copied, with their provenance records.
        dropped: Filenames removed from the destination as leftovers of the graph replaced.
        withheld: Filenames the new FCSR run is expected to write itself.
        missing: Filenames recorded in the source manifest but absent from its disk.
    """

    source: Path
    dest: Path
    copied: tuple[str, ...]
    dropped: tuple[str, ...]
    withheld: tuple[str, ...]
    missing: tuple[str, ...]


def seed_graph_dir(source_dir: str | os.PathLike[str],
                   dest_dir: str | os.PathLike[str]) -> SeedResult:
    """Clone a graph directory's non-FCSR artifacts into a fresh one, provenance and all.

    Args:
        source_dir: An existing, deploy-ready artifact directory.
        dest_dir: The directory to seed; created if absent.

    Returns:
        What was copied and what was withheld.

    Raises:
        ValueError: The source records no artifacts, so nothing there has a digest to prove
            the copy by, and robot_ood.data.store.validate would reject every hand-copied file.
    """
    from robot_ood.data.store import ArtifactManifest

    src, dst = Path(source_dir).resolve(), Path(dest_dir).resolve()
    src_manifest = ArtifactManifest.load(src)
    if not src_manifest.artifacts:
        raise ValueError(
            f"{src}: no ARTIFACTS.json entries -- nothing to carry provenance from. Refusing to "
            f"hand-copy files with no digest to prove them by."
        )

    dst.mkdir(parents=True, exist_ok=True)
    dst_manifest = ArtifactManifest.load(dst)
    # A cache the replacing build does not rewrite would otherwise outlive its own graph and be
    # read against node ids that no longer mean the same thing.
    stale = sorted(set(dst_manifest.artifacts) - set(src_manifest.artifacts) - _FCSR_OWNED)
    for name in stale:
        (dst / name).unlink(missing_ok=True)
        dst_manifest.forget(name)
    withheld = _FCSR_OWNED
    copied: list[str] = []
    missing: list[str] = []
    for name, record in sorted(src_manifest.artifacts.items()):
        if name in withheld:
            continue
        src_file = src / name
        if not src_file.is_file():
            missing.append(name)
            continue
        dst_file = dst / name
        shutil.copy2(src_file, dst_file)
        # Carrying the source's own record across, not writing a fresh one: the digest check on a
        # byte-identical copy then PROVES the provenance instead of asserting it.
        dst_manifest.record(
            dst_file, stage=record.stage, graph_hash=record.graph_hash,
            command=f"copied byte-identical from {src} (orig: {record.command})",
        )
        copied.append(name)
    dst_manifest.save()
    return SeedResult(source=src, dest=dst, copied=tuple(copied), dropped=tuple(stale),
                      withheld=tuple(sorted(withheld)), missing=tuple(missing))
