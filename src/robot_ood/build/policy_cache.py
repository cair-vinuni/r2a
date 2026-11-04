"""Cache a frozen policy's per-frame actions over a demo corpus, into a_pi_raw.npz.

Every raw frame of every demo HDF5 is replayed against the policy server under the deploy chunk
contract (its protocol's wire image size, execution horizon and gripper reading); the rows align
1:1 with NodeTable.q_raw. FUSION needs the policy's OWN action: its targets are built against what
the policy would have commanded. This module is the client the cache-policy command spawns, one
per shard; a shard replays every num_shards-th demo and the final run assembles the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol as ProtocolType, Sequence

import numpy as np

from robot_ood.build.graph import (
    MIN_DEMO_FRAMES,
    DemoLoader,
    GraphBuilder,
    LoadLimits,
    NodeBuilder,
)
from robot_ood.data.geometry import graph_hash
from robot_ood.core.paths import default_paths
from robot_ood.data import libero
from robot_ood.data import schema
from robot_ood.data.edges import EdgeSet
from robot_ood.data.nodes import NodeTable
from robot_ood.data.store import record_artifacts

__all__ = [
    "ACTIONS_NPZ",
    "ACTION_DIM",
    "GRIP_COL",
    "STUB_CHUNK_ROWS",
    "StubPolicyClient",
    "ChunkReplayer",
    "ActionCacheBuilder",
    "DemoActions",
    "DemoFrames",
    "DemoSource",
    "Replayer",
    "cached_policy_protocol",
    "configure_logging",
    "main",
    "suite_checkpoint",
]

from robot_ood.core.paths import repo_root

_REPO_ROOT: Final[Path] = repo_root()

log = logging.getLogger("robot_ood.build.policy_cache")

# Artifact names written into the output directory.
ACTIONS_NPZ: Final[str] = schema.A_PI_RAW_NPZ
RAW_MANIFEST_JSON: Final[str] = "a_pi_raw.manifest.json"
SHARD_DIRNAME: Final[str] = "a_pi_raw_shards"

# The one place configs/robot_ood.yaml is read for the per-suite policy checkpoint.
SUITES_YAML: Final[Path] = _REPO_ROOT / "configs" / "robot_ood.yaml"

# Queries folded into one server request; the servers batch across clients on top of this.
DEFAULT_BATCH: Final[int] = 16

from robot_ood.core.absolute import deltas_from_targets
from robot_ood.core.action import ACTION_DIM, GRIP_COL
from robot_ood.core.protocol import BINARIZE, NATIVE_SIZE, Protocol, get_protocol, protocol_ids

# Rows in a stubbed chunk. Only the leading action_chunk_size of them are ever indexed, so this
# just has to be at least as long as whatever chunk the client is configured with.
STUB_CHUNK_ROWS: Final[int] = 64


# Width of a chunk row that carries both arms' absolute joint targets.
BIMANUAL_CHUNK_DIM = 14

# The protocol whose chunks are ONE arm's absolute joint targets: a real ALOHA right arm under a
# GR00T N1.7 finetune, whose server un-relativises every row against the state it was handed.
ALOHA_PROTOCOL = "gr00t_aloha"


class StubPolicyClient:
    """Offline websocket stand-in: chunk row i of a query at step s equals s + i, for --dry-run."""

    def __init__(self, host: str = "", port: int | None = None, dim: int = ACTION_DIM) -> None:
        self.host, self.port, self.dim = host, port, dim

    def get_server_metadata(self) -> dict[str, Any]:
        return {"stub": True}

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Deterministic chunks keyed to payload["episode_step"], in the server's response shape."""
        chunks = [np.arange(int(step), int(step) + STUB_CHUNK_ROWS, dtype=np.float32)
                  for step in payload["episode_step"]]
        actions = np.stack(chunks)[:, :, None].repeat(self.dim, axis=2)
        return {"data": {"actions": actions}}


def _websocket_client(host: str, port: int) -> Any:
    """The harness's websocket client, imported from evals/libero_plus/harness."""
    harness = str(_REPO_ROOT / "evals" / "libero_plus" / "harness")
    if harness not in sys.path:
        sys.path.insert(0, harness)
    from tools.websocket_policy_client import WebsocketClientPolicy

    return WebsocketClientPolicy(host, port)


class ChunkReplayer:
    """Queries a websocket policy server once per execution horizon, many queries per request.

    Args:
        protocol: The policy's rollout recipe: wire image size, execution horizon, gripper reading.
        host: Policy server host.
        port: Policy server port.
        worker_id: Stable client id, forwarded like the harness's shard index.
        batch: Queries per request.
        stub: Replace the websocket client with StubPolicyClient.
    """

    def __init__(self, protocol: Protocol, *, host: str, port: int, worker_id: int = 0,
                 batch: int = DEFAULT_BATCH, stub: bool = False, arm: str = "") -> None:
        configure_logging()
        self.protocol = protocol
        self.arm = arm
        if protocol.bimanual and not arm:
            raise ValueError(f"the {protocol.id} protocol answers for both arms; name the lane's arm")
        self._worker_id = int(worker_id)
        self._batch = max(1, int(batch))
        dim = BIMANUAL_CHUNK_DIM if protocol.bimanual else ACTION_DIM
        self._client = StubPolicyClient(host, port, dim) if stub else _websocket_client(host, port)

    @property
    def chunk_size(self) -> int:
        """Frames replayed open loop per query."""
        return self.protocol.exec_horizon

    @property
    def server_metadata(self) -> dict[str, Any]:
        return dict(self._client.get_server_metadata())

    def replay(self, source: DemoSource, frames: DemoFrames) -> DemoActions:
        """The policy's command at every frame of one demo, under the deploy chunk contract.

        Args:
            source: The demo, for its instruction.
            frames: Its observations.

        Returns:
            The commanded actions and the raw gripper channel.
        """
        horizon = self.protocol.exec_horizon
        n = len(frames)
        commanded = np.empty((n, ACTION_DIM), dtype=np.float32)
        grip = np.empty(n, dtype=np.float32)
        queries = list(range(0, n, horizon))
        for start in range(0, len(queries), self._batch):
            steps = queries[start:start + self._batch]
            response = self._client.infer({
                "batch_images": [self._wire_views(frames, t) for t in steps],
                "instructions": [source.instruction] * len(steps),
                "state": [frames.state[t][None] for t in steps],
                "episode_step": list(steps),
                "worker_id": [self._worker_id] * len(steps),
            })
            actions = np.asarray(response["data"]["actions"], dtype=np.float32)
            for t, chunk in zip(steps, actions):
                rows, raw_grip = self._commanded(chunk[:horizon][: n - t], frames, t)
                commanded[t:t + len(rows)] = rows
                grip[t:t + len(rows)] = raw_grip
        return DemoActions(commanded=commanded, grip=grip)

    def _wire_views(self, frames: "DemoFrames", t: int) -> list[np.ndarray]:
        """Frame t's views as the sim client sends them: resized to the wire side, or as rendered."""
        if self.protocol.wire_image_size == NATIVE_SIZE:
            return [view[t] for view in frames.views]
        import cv2 as cv

        size = (self.protocol.wire_image_size, self.protocol.wire_image_size)
        return [cv.resize(view[t], size, interpolation=cv.INTER_AREA) for view in frames.views]

    def _commanded(self, chunk: np.ndarray, frames: "DemoFrames", t: int) -> tuple[np.ndarray, np.ndarray]:
        """A chunk's rows as the executed [ACTION_DIM] commands from frame t, and the raw gripper channel."""
        # A bimanual joint chunk is this lane's absolute joint targets: differenced from the demo's
        # joints at t (a euclidean pose space), the gripper fraction read as the corpus's open flag.
        if self.protocol.bimanual:
            from robot_ood.core.action import EUCLIDEAN
            from robot_ood.data.robotwin import lane_targets, open_flag
            targets, raw = lane_targets(chunk, self.arm)
            rows = np.empty((len(chunk), ACTION_DIM), dtype=np.float32)
            rows[:, :6] = deltas_from_targets(frames.ee[t], targets, EUCLIDEAN)
            rows[:, GRIP_COL] = 1.0 - 2.0 * open_flag(raw)
            return rows, raw.astype(np.float32)
        # A GR00T ALOHA chunk holds absolute joint targets: differenced from the demo's joints at t
        # like the bimanual lane, gripper kept as sent.
        if self.protocol.id == ALOHA_PROTOCOL:
            from robot_ood.core.action import EUCLIDEAN
            from robot_ood.data.lerobot_aloha import open_flag as aloha_open_flag
            joint_chunk = np.asarray(chunk, dtype=np.float64)
            raw = joint_chunk[:, GRIP_COL]
            rows = np.empty((len(chunk), ACTION_DIM), dtype=np.float32)
            rows[:, :6] = deltas_from_targets(frames.ee[t], joint_chunk[:, :6], EUCLIDEAN)
            rows[:, GRIP_COL] = 1.0 - 2.0 * aloha_open_flag(raw)
            return rows, raw.astype(np.float32)
        rows = np.asarray(chunk, dtype=np.float32).copy()
        raw = rows[:, GRIP_COL].copy()
        if self.protocol.gripper == BINARIZE:
            rows[:, GRIP_COL] = 1.0 - 2.0 * (raw > 0.5)
        return rows, raw


# ==== Provenance ====
@dataclass(frozen=True, slots=True)
class CheckpointId:
    """Content fingerprint of a policy checkpoint directory.

    Attributes:
        path: Resolved checkpoint directory.
        stats_sha256: sha256 of dataset_statistics.json, or "" if absent.
        weights_sha256: sha256 over the sorted (name, size) list of weight files.
    """

    path: Path
    stats_sha256: str
    weights_sha256: str

    @classmethod
    def probe(cls, ckpt_dir: Path) -> CheckpointId:
        """Fingerprint a checkpoint directory without reading the weights.

        Args:
            ckpt_dir: Checkpoint directory.

        Returns:
            The identifier.
        """
        stats = ckpt_dir / "dataset_statistics.json"
        stats_sha = (
            hashlib.sha256(stats.read_bytes()).hexdigest() if stats.exists() else ""
        )
        listing = sorted(
            (p.name, p.stat().st_size)
            for p in ckpt_dir.glob("*")
            if p.is_file() and p.suffix in (".safetensors", ".pt", ".bin")
        )
        digest = hashlib.sha256(json.dumps(listing).encode()).hexdigest()
        return cls(path=ckpt_dir, stats_sha256=stats_sha, weights_sha256=digest)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form.

        Returns:
            Dict with the path flattened to str.
        """
        return {
            "path": os.fspath(self.path),
            "stats_sha256": self.stats_sha256,
            "weights_sha256": self.weights_sha256,
        }


def suite_checkpoint(suite: str, key: str | None = None) -> Path:
    """Resolve a suite's policy checkpoint from configs/robot_ood.yaml.

    Args:
        suite: Suite short-name.
        key: The suites.<s>.eval key naming the checkpoint; default <tree policy>_ckpt.

    Returns:
        The checkpoint directory.

    Raises:
        KeyError: The suite is not declared, or carries no such checkpoint key.
    """
    from robot_ood.core.paths import project_config

    suites = project_config().get("suites") or {}
    if suite not in suites:
        raise KeyError(f"suite {suite!r} not in {SUITES_YAML}; have {sorted(suites)}")
    info = suites[suite]
    if key is None:
        key = f"{info.get('policy') or 'stablevla'}_ckpt"
    ckpt = (info.get("eval") or {}).get(key)
    if not ckpt:
        raise KeyError(f"suite {suite!r} declares no {key!r} checkpoint in {SUITES_YAML}")
    from robot_ood.core.paths import resolve_trunk_path

    return resolve_trunk_path(ckpt)


def cached_policy_protocol(graph_dir: str | os.PathLike) -> str | None:
    """The protocol id graph_dir's a_pi_raw.npz was cached under; None without a stamped manifest.

    Args:
        graph_dir: The artifact directory holding a_pi_raw.manifest.json.

    Returns:
        The protocol id, or None when the manifest is absent or predates the stamp.
    """
    path = Path(graph_dir) / RAW_MANIFEST_JSON
    if not path.is_file():
        return None
    protocol = (json.loads(path.read_text()).get("preprocessing") or {}).get("protocol")
    return None if protocol is None else str(protocol)


# ==== Demo index -- the ordering NodeTable.q_raw is built in ====
@dataclass(frozen=True, slots=True)
class DemoSource:
    """Locates one demo inside the corpus, in global demo-id order.

    Attributes:
        demo_id: Global demo id, i.e. the index into NodeTable.raw_ptr.
        hdf5_path: Source file.
        group_key: Demo group name inside data/.
        task_id: Sort position of the source file.
        task_name: Filename-derived task name, as from_demos.DemoLoader reports it.
        instruction: LIBERO language instruction fed to the policy.
        n_frames: Raw frame count T.
    """

    demo_id: int
    hdf5_path: Path
    group_key: str
    task_id: int
    task_name: str
    instruction: str
    n_frames: int

    @property
    def label(self) -> str:
        """Human-readable id used in logs and shard bookkeeping."""
        return f"{self.task_name}/{self.group_key}"


# Subclassing rather than re-walking the directory is what keeps this cache's row order identical
# to the graph's: q_raw concatenates the surviving demos' joint_states in the base class's order.
class DemoIndexer(DemoLoader):
    """DemoLoader that also reports WHICH (file, group) each demo came from."""

    def index(self) -> tuple[DemoSource, ...]:
        """Enumerate every demo the corresponding graph build would have kept.

        Returns:
            Demo sources in global demo-id order.
        """
        sources: list[DemoSource] = []
        for task_id, path in enumerate(self.files):
            task_name = self.task_names[task_id]
            with libero.open_demos(path) as data:
                instruction = _instruction(data, task_name)
                for key in self.sorted_demo_keys(data):
                    n_frames = int(data[key]["obs"]["joint_states"].shape[0])
                    if n_frames < MIN_DEMO_FRAMES:
                        continue
                    sources.append(
                        DemoSource(
                            demo_id=len(sources),
                            hdf5_path=path,
                            group_key=key,
                            task_id=task_id,
                            task_name=task_name,
                            instruction=instruction,
                            n_frames=n_frames,
                        )
                    )
        return tuple(sources)


def _instruction(data: Any, task_name: str) -> str:
    """Read a file's LIBERO language instruction and tie it to the from_demos task name.

    Args:
        data: The data group of an open HDF5 file.
        task_name: Task name reported by DemoLoader for the same file.

    Returns:
        The instruction string.

    Raises:
        ValueError: problem_info is missing, or its instruction is not part of the task name.
    """
    raw = data.attrs.get("problem_info")
    if raw is None:
        raise ValueError(f"{task_name}: HDF5 data group has no problem_info attribute")
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    instruction = str(json.loads(text)["language_instruction"]).strip()
    if instruction.replace(" ", "_").lower() not in task_name.lower():
        raise ValueError(
            f"{task_name}: problem_info instruction {instruction!r} does not match the filename"
        )
    return instruction


# ==== Observations ====
@dataclass(frozen=True, slots=True)
class DemoFrames:
    """Every observation of one demo, in the form the harness hands the policy.

    Attributes:
        views: Per camera, [T, H, W, 3] u8 as the sim client would send it: LIBERO's agent and
            wrist views rotated 180 degrees; RoboTwin's head and two wrist cameras as rendered.
        state: [T, S] f64 the policy's proprio row: LIBERO concat(ee_pos, ee_ori, gripper_states),
            RoboTwin the recorded policy_state.
        ee: [T, 6] f64 this lane's pose, for differencing absolute targets.
        joints: [T, D] f32 joint_states, for the q_raw alignment check.
    """

    views: tuple[np.ndarray, ...]
    state: np.ndarray
    ee: np.ndarray
    joints: np.ndarray

    def __len__(self) -> int:
        return int(self.joints.shape[0])

    @classmethod
    def read(cls, source: DemoSource, protocol: Protocol) -> DemoFrames:
        """Load one demo's observations.

        Args:
            source: Demo to read.
            protocol: The policy's rollout recipe, which decides the cameras and the proprio.

        Returns:
            The observations.

        Raises:
            ValueError: The group's frame count disagrees with the index.
        """
        with libero.open_demos(source.hdf5_path) as data:
            obs = data[source.group_key]["obs"]
            ee = np.concatenate([np.asarray(obs["ee_pos"], dtype=np.float64),
                                 np.asarray(obs["ee_ori"], dtype=np.float64)], axis=1)
            joints = np.asarray(obs["joint_states"], dtype=np.float32)
            if protocol.bimanual:
                from robot_ood.data.robotwin import CAMERA_KEYS
                views = tuple(_decode_jpegs(np.asarray(obs[key])) for key in CAMERA_KEYS)
                state = np.asarray(obs["policy_state"], dtype=np.float64)
            elif protocol.id == ALOHA_PROTOCOL:
                from robot_ood.data.lerobot_aloha import CAMERA_KEYS as ALOHA_CAMERAS
                # No rotation: these are real camera frames, stored the way they were captured.
                views = tuple(_decode_jpegs(np.asarray(obs[key])) for key in ALOHA_CAMERAS)
                # The proprio the checkpoint takes: six joints then the NATIVE gripper, never the
                # corpus's open flag -- its gripper action config is ABSOLUTE over that range.
                state = np.concatenate([joints[:, :6].astype(np.float64),
                                        np.asarray(obs["grip_raw"], dtype=np.float64)], axis=1)
            else:
                # The 180-degree rotation matches eval_libero.py: stored RGB is in the same OpenGL
                # bottom-up convention the live env returns. ee_ori is already axis-angle.
                agentview = _rotate_180(np.asarray(obs["agentview_rgb"]))
                # a single-camera corpus carries no wrist stream; the agent view stands in for it
                wrist_key = "eye_in_hand_rgb" if "eye_in_hand_rgb" in obs else "agentview_rgb"
                views = (agentview, _rotate_180(np.asarray(obs[wrist_key])))
                state = np.concatenate([ee, np.asarray(obs["gripper_states"], dtype=np.float64)], axis=1)
        if joints.shape[0] != source.n_frames:
            raise ValueError(
                f"{source.label}: read {joints.shape[0]} frames, index says {source.n_frames}"
            )
        return cls(views=views, state=state, ee=ee, joints=joints)


def _decode_jpegs(encoded: np.ndarray) -> np.ndarray:
    """[T] jpeg byte strings -> [T, H, W, 3] u8 RGB, as RoboTwin's own reader decodes them."""
    import cv2 as cv

    frames = [cv.imdecode(np.frombuffer(raw, dtype=np.uint8), cv.IMREAD_COLOR) for raw in encoded]
    return np.stack(frames).astype(np.uint8)


def _rotate_180(frames: np.ndarray) -> np.ndarray:
    """Rotate a stack of images 180 degrees, per-frame contiguous.

    Args:
        frames: [T, H, W, 3] u8 stack.

    Returns:
        [T, H, W, 3] u8 rotated stack.
    """
    return np.ascontiguousarray(frames[:, ::-1, ::-1])


# ==== Alignment against the graph's q_raw ====
@dataclass(frozen=True, slots=True)
class Reference:
    """The node table this cache must align to.

    Attributes:
        table: Node table supplying raw_ptr and q_raw.
        origin: Where it came from -- an on-disk artifact path, or "rebuilt".
        graph_hash: Fingerprint of the built graph, or "" when g_edges.npz is absent.
    """

    table: NodeTable
    origin: str
    graph_hash: str


# Two checks, not one: frame counts against raw_ptr catch structure, and joint configurations
# against q_raw catch a demo ordering that happens to preserve every length.
class AlignmentGuard:
    """Refuses to write a cache whose rows would not line up with q_raw."""

    def __init__(self, reference: Reference, sources: Sequence[DemoSource]) -> None:
        """Bind a reference table and the demo order to check.

        Args:
            reference: Node table to align to.
            sources: Demo sources in global demo-id order.
        """
        self._table = reference.table
        self._origin = reference.origin
        self._sources = tuple(sources)
        self._offsets = np.concatenate(
            [[0], np.cumsum([s.n_frames for s in self._sources])]
        ).astype(np.int64)

    @property
    def offsets(self) -> np.ndarray:
        """[n_demos + 1] i64 CSR boundaries this run will write at."""
        return self._offsets

    @property
    def n_frames(self) -> int:
        """Total raw frames across every indexed demo."""
        return int(self._offsets[-1])

    def check_counts(self) -> None:
        """Compare per-demo frame counts against the reference raw_ptr.

        Raises:
            ValueError: The demo count, the total frame count, or any per-demo count differs.
        """
        expected = self._table.raw_ptr.astype(np.int64)
        if expected.shape != self._offsets.shape:
            raise ValueError(
                f"demo count mismatch vs {self._origin}: index has {len(self._sources)} demos, "
                f"raw_ptr has {expected.shape[0] - 1}"
            )
        if not np.array_equal(expected, self._offsets):
            bad = int(np.flatnonzero(expected != self._offsets)[0])
            raise ValueError(
                f"frame-count mismatch vs {self._origin} at demo {max(0, bad - 1)} "
                f"({self._sources[max(0, bad - 1)].label}): raw_ptr={expected[bad]} "
                f"index={self._offsets[bad]}"
            )

    def check_frames(self, demo_id: int, joints: np.ndarray) -> None:
        """Compare one demo's joint configurations against the reference q_raw slice.

        Args:
            demo_id: Global demo id.
            joints: [T, D] f32 joint_states as read from the HDF5.

        Raises:
            ValueError: Any row differs from q_raw.
        """
        lo, hi = int(self._offsets[demo_id]), int(self._offsets[demo_id + 1])
        expected = self._table.q_raw[lo:hi]
        if not np.array_equal(expected, joints):
            raise ValueError(
                f"{self._sources[demo_id].label}: q_raw[{lo}:{hi}] does not match this demo's "
                f"joint_states -- the demo ordering does not reproduce the graph's"
            )


def load_reference(
    suite: str,
    nodes_npz: Path | None,
    loader: DemoIndexer,
    coarsen: int | None,
    *,
    rebuild: bool = False,
) -> Reference:
    """Resolve the node table to align against.

    Args:
        suite: Suite short-name.
        nodes_npz: Explicit artifact path, or None to resolve one for the suite.
        loader: Loader whose caps define the corpus.
        coarsen: Coarsening factor for a rebuild. None uses the schema default.
        rebuild: Skip the artifact and always rebuild.

    Returns:
        The reference.
    """
    path = nodes_npz or (default_paths().graph(suite) / schema.NODES_NPZ)
    if path.exists() and not rebuild:
        table = NodeTable.load(os.fspath(path))
        edges_npz = path.parent / schema.EDGES_NPZ
        fingerprint = ""
        if edges_npz.exists():
            fingerprint = graph_hash(table, EdgeSet.load(os.fspath(edges_npz), strict=False))
        return Reference(table=table, origin=os.fspath(path), graph_hash=fingerprint)

    # No artifact, or caps make the corpus a strict subset of the built graph. A rebuild off the
    # same loader still exercises NodeTable.from_demos' ordering, which is what alignment checks.
    log.warning("not using %s -- rebuilding the reference table in memory", path)
    table = NodeBuilder(coarsen=coarsen or schema.COARSEN).build(loader.load())
    return Reference(table=table, origin="rebuilt", graph_hash="")


# ==== Policy replay ====
@dataclass(frozen=True, slots=True)
class DemoActions:
    """One demo's cached policy output.

    Attributes:
        commanded: [T, 7] f32 action as eval_libero would have sent it, gripper binarized.
        grip: [T] f32 pre-binarization gripper channel.
    """

    commanded: np.ndarray
    grip: np.ndarray

    def __len__(self) -> int:
        return int(self.commanded.shape[0])


# ==== Resumable per-demo shards ====
class ShardStore:
    """One file per replayed demo, so a session death costs one demo and clients never share a file."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def path(self, source: DemoSource) -> Path:
        return self._root / f"{source.task_id:03d}_{source.task_name}" / f"{source.group_key}.npz"

    def load(self, source: DemoSource) -> DemoActions | None:
        """The demo's cached actions, or None when it is not cached or its length changed."""
        path = self.path(source)
        if not path.exists():
            return None
        with np.load(os.fspath(path), allow_pickle=False) as archive:
            actions = DemoActions(commanded=archive["actions"].astype(np.float32),
                                  grip=archive["grip"].astype(np.float32))
        if len(actions) != source.n_frames:
            log.warning("%s: %d cached rows for %d frames -- recomputing", path, len(actions),
                        source.n_frames)
            return None
        return actions

    def save(self, source: DemoSource, actions: DemoActions) -> None:
        """Write one demo's actions atomically."""
        target = self.path(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp.npz")
        np.savez(os.fspath(tmp), actions=actions.commanded, grip=actions.grip)
        os.replace(tmp, target)

    def clear(self) -> None:
        """Delete the shard directory once the final artifact is on disk."""
        shutil.rmtree(self._root, ignore_errors=True)


# ==== Progress ====
class Progress:
    """Frames-per-second and ETA over the frames this process actually computes."""

    def __init__(self, total_frames: int) -> None:
        """Start the clock.

        Args:
            total_frames: Frames still to compute.
        """
        self._total = max(1, int(total_frames))
        self._done = 0
        self._start = time.monotonic()

    def advance(self, frames: int) -> str:
        """Record progress and format a one-line status.

        Args:
            frames: Frames just completed.

        Returns:
            A "done/total (pct) fps eta" status string.
        """
        self._done += int(frames)
        elapsed = max(1e-6, time.monotonic() - self._start)
        fps = self._done / elapsed
        eta_s = (self._total - self._done) / max(1e-6, fps)
        return (
            f"{self._done}/{self._total} ({100.0 * self._done / self._total:.1f}%) "
            f"{fps:.1f} frame/s eta {eta_s / 60.0:.1f} min"
        )



class Replayer(ProtocolType):
    """What ActionCacheBuilder needs from whatever drives the policy."""

    protocol: Protocol

    @property
    def chunk_size(self) -> int:
        """Action-chunk length the client caches and replays open-loop."""

    @property
    def server_metadata(self) -> dict[str, Any]:
        """Whatever the server announced on connect."""

    def replay(self, source: "DemoSource", frames: "DemoFrames") -> "DemoActions":
        """Step the policy over every frame of one demo.

        Args:
            source: The demo being replayed, for its instruction.
            frames: That demo's observations.

        Returns:
            The demo's cached actions.
        """

# ==== Orchestrator ====
class ActionCacheBuilder:
    """Runs index -> align -> replay -> write for one suite."""

    def __init__(
        self,
        suite: str,
        *,
        hdf5_dir: Path,
        out_dir: Path,
        checkpoint: Path,
        reference: Reference,
        sources: Sequence[DemoSource],
        replayer: Replayer,
        host: str,
        port: int,
        dry_run: bool,
        shard: tuple[int, int] | None = None,
    ) -> None:
        """Configure a build.

        Args:
            suite: Suite short-name.
            hdf5_dir: Resolved demo directory.
            out_dir: Where the artifact and manifest are written.
            checkpoint: Policy checkpoint directory.
            reference: Node table to align against.
            sources: Demo sources in global demo-id order.
            replayer: Configured policy driver.
            host: Policy server host, recorded in the manifest.
            port: Policy server port, recorded in the manifest.
            dry_run: Whether the policy client is stubbed.
            shard: (index, count) restricting this run to every count-th demo and stopping
                before the artifact is assembled, or None to replay and write the whole corpus.
        """
        self._suite = suite
        self._hdf5_dir = hdf5_dir
        self._out_dir = out_dir
        self._checkpoint = checkpoint
        self._reference = reference
        self._sources = tuple(sources)
        self._replayer = replayer
        self._host = host
        self._port = port
        self._dry_run = dry_run
        self._shard = shard
        self._guard = AlignmentGuard(reference, self._sources)
        self._shards = ShardStore(out_dir / SHARD_DIRNAME)

    def run(self) -> Path | None:
        """Replay this run's demos, resuming from the shards on disk, then assemble.

        Returns:
            The written a_pi_raw.npz path, or None when this run is one shard of several.
        """
        self._guard.check_counts()
        log.info("aligned: %d demos, %d frames, reference=%s", len(self._sources),
                 self._guard.n_frames, self._reference.origin)
        mine = [s for s in self._sources if self._is_mine(s.demo_id)]
        todo = [s for s in mine if self._shards.load(s) is None]
        log.info("this run: %d demos, %d still to replay", len(mine), len(todo))
        progress = Progress(sum(s.n_frames for s in todo))
        for source in todo:
            self._shards.save(source, self._replay_one(source))
            log.info("%s  %s", source.label, progress.advance(source.n_frames))
        if self._shard is not None:
            index, count = self._shard
            log.info("shard %d/%d done, artifact not assembled", index, count)
            return None
        return self._write()

    def _is_mine(self, demo_id: int) -> bool:
        """Whether this run replays that demo, given its --shard selector."""
        if self._shard is None:
            return True
        index, count = self._shard
        return demo_id % count == index

    def _replay_one(self, source: DemoSource) -> DemoActions:
        """Read, verify and replay one demo.

        Args:
            source: The demo to run.

        Returns:
            Its cached actions.
        """
        frames = DemoFrames.read(source, self._replayer.protocol)
        self._guard.check_frames(source.demo_id, frames.joints)
        return self._replayer.replay(source, frames)

    def _write(self) -> Path:
        """Concatenate every demo's shard in global demo order and write the artifact plus manifest.

        Returns:
            The written a_pi_raw.npz path.

        Raises:
            ValueError: A demo has no shard, or the assembled length disagrees with the reference.
        """
        cached = [(source, self._shards.load(source)) for source in self._sources]
        missing = [source.label for source, actions in cached if actions is None]
        if missing:
            raise ValueError(f"{len(missing)} demos have no shard (first: {missing[:3]}); "
                             "run the missing shards before assembling")
        actions = np.concatenate([a.commanded for _, a in cached], axis=0).astype(np.float32)
        grip = np.concatenate([a.grip for _, a in cached], axis=0).astype(np.float32)
        if actions.shape[0] != self._guard.n_frames:
            raise ValueError(
                f"assembled {actions.shape[0]} rows, expected {self._guard.n_frames}"
            )
        if self._dry_run:
            self._check_stub_alignment(actions)

        self._out_dir.mkdir(parents=True, exist_ok=True)
        target = self._out_dir / ACTIONS_NPZ
        np.savez_compressed(
            os.fspath(target),
            a_pi_raw=actions,
            a_pi_grip=grip,
            raw_ptr=self._guard.offsets.astype(np.int32),
        )
        (self._out_dir / RAW_MANIFEST_JSON).write_text(json.dumps(self._manifest(), indent=2) + "\n")
        self._shards.clear()
        # a_pi_raw.manifest.json records this command's own bookkeeping; ARTIFACTS.json is the
        # directory-wide ledger the deploy gate reads, and nothing else writes this file into it.
        record_artifacts(
            self._out_dir, stage="policy_actions", command=shlex.join(sys.argv),
            filenames=[ACTIONS_NPZ],
        )
        log.info("wrote %s  a_pi_raw=%s", target, actions.shape)
        return target

    def _check_stub_alignment(self, actions: np.ndarray) -> None:
        """Verify the stub's step-numbered chunks came back in per-demo frame order.

        Args:
            actions: [N, 7] assembled cache.

        Raises:
            ValueError: Any row does not carry its own within-demo frame index.
        """
        expected = np.concatenate([np.arange(s.n_frames) for s in self._sources])
        if not np.array_equal(actions[:, 0].astype(np.int64), expected):
            bad = int(np.flatnonzero(actions[:, 0].astype(np.int64) != expected)[0])
            raise ValueError(
                f"chunk/order check failed at row {bad}: got {actions[bad, 0]}, "
                f"expected {expected[bad]}"
            )
        log.info("dry-run: chunk indexing and demo ordering verified over %d rows", len(expected))

    def _manifest(self) -> dict[str, Any]:
        """Everything needed to decide whether this cache may be reused.

        Returns:
            The JSON-serialisable manifest.
        """
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "suite": self._suite,
            "hdf5_dir": os.fspath(self._hdf5_dir),
            "dry_run": self._dry_run,
            "policy": {
                "checkpoint": CheckpointId.probe(self._checkpoint).to_dict(),
                "server_url": f"ws://{self._host}:{self._port}",
                "server_metadata": self._replayer.server_metadata,
                "chunk_size": self._replayer.chunk_size,
            },
            "preprocessing": {
                "image_rotation": "[:, ::-1, ::-1]",
                "protocol": self._replayer.protocol.id,
                "resize": [self._replayer.protocol.wire_image_size] * 2,
                "state": "concat(ee_pos[3], ee_ori[3], gripper_states[2])",
                "source_image_size": 128,
            },
            "reference": {
                "origin": self._reference.origin,
                "graph_hash": self._reference.graph_hash,
                "n_raw": int(self._reference.table.n_raw),
            },
            "n_demos": len(self._sources),
            "n_frames": self._guard.n_frames,
            "arrays": {
                "a_pi_raw": "[N, 7] f32 commanded action; gripper binarized to +-1 like eval_libero",
                "a_pi_grip": "[N] f32 pre-binarization gripper channel",
                "raw_ptr": "[n_demos + 1] i32 CSR boundaries, equal to NodeTable.raw_ptr",
            },
            "demos": [
                {
                    "task_id": s.task_id,
                    "task_name": s.task_name,
                    "group": s.group_key,
                    "instruction": s.instruction,
                    "n_frames": s.n_frames,
                }
                for s in self._sources
            ],
        }


# ==== CLI ====
def _shard(spec: str | None) -> tuple[int, int] | None:
    """Parse an "I/N" demo selector.

    Args:
        spec: The --shard value, or None for the whole corpus.

    Returns:
        (index, count), or None.

    Raises:
        ValueError: The value is not "I/N" with 0 <= I < N.
    """
    if spec is None:
        return None
    index, _, count = spec.partition("/")
    i, n = int(index), int(count or 0)
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"--shard {spec!r}: expected I/N with 0 <= I < N")
    return i, n


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, excluding the program name.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    from robot_ood.core.paths import project_config
    suites = tuple(sorted(project_config().get("suites") or ())) or ("long",)
    parser.add_argument("suite", choices=suites)
    parser.add_argument("--out-dir", default=None, help="default: the suite's graph artifact dir")
    parser.add_argument("--nodes-npz", default=None, help="reference g_nodes.npz")
    parser.add_argument("--ckpt", default=None, help="default: configs/robot_ood.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--arm", default="", help="the lane's arm under a bimanual protocol (left or right)")
    parser.add_argument("--protocol", required=True, choices=protocol_ids(),
                        help="the policy's rollout recipe")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="queries per request")
    parser.add_argument(
        "--shard", default=None, metavar="I/N",
        help="replay only demos with demo_id %% N == I and stop before assembling; a later run "
             "without this flag reads every shard back and writes the artifact",
    )
    parser.add_argument("--coarsen", type=int, default=None)
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--limit-demos", type=int, default=None, help="per task file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stub the policy server and assert the chunk/order contract",
    )
    return parser.parse_args(list(argv))


def configure_logging() -> None:
    """Give this module its own stdout handler, detached from the root logger."""
    # Idempotent, and called again after the policy client is imported: that import's dictConfig
    # disables every logger built before it, swallowing the progress a multi-hour job is steered by.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False
    log.disabled = False


def main(argv: Sequence[str] | None = None) -> int:
    """Index the corpus, align it to the graph, replay it and write a_pi_raw.npz.

    Args:
        argv: Argument list, or None to read sys.argv.

    Returns:
        Process exit code.
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging()

    limits = LoadLimits(tasks=args.limit_tasks, demos=args.limit_demos)
    hdf5_dir = GraphBuilder(args.suite).hdf5_dir
    loader = DemoIndexer(hdf5_dir, limits=limits)
    sources = loader.index()
    log.info("%s: %d demos under %s", args.suite, len(sources), hdf5_dir)

    # Caps make the corpus a strict subset of the built graph, so the on-disk artifact cannot be
    # the reference -- rebuild one from the same capped loader instead.
    capped = limits.tasks is not None or limits.demos is not None
    reference = load_reference(
        args.suite,
        Path(args.nodes_npz) if args.nodes_npz else None,
        loader,
        args.coarsen,
        rebuild=capped,
    )

    checkpoint = Path(args.ckpt) if args.ckpt else suite_checkpoint(args.suite)
    out_dir = Path(args.out_dir) if args.out_dir else default_paths().graph(args.suite)
    replayer = ChunkReplayer(get_protocol(args.protocol), host=args.host, port=args.port, worker_id=args.worker_id,
                             batch=args.batch, stub=args.dry_run, arm=args.arm)
    ActionCacheBuilder(
        args.suite,
        hdf5_dir=hdf5_dir,
        out_dir=out_dir,
        checkpoint=checkpoint,
        reference=reference,
        sources=sources,
        replayer=replayer,
        host=args.host,
        port=args.port,
        dry_run=args.dry_run,
        shard=_shard(args.shard),
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
