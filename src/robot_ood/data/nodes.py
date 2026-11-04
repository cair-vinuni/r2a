"""The node table: one row per short, homogeneous run of consecutive demo frames."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robot_ood.core import npz
from robot_ood.core.action import ACTION_DIM, POSE_DIM
from robot_ood.core.constants import STILL_TOL
from robot_ood.data import schema

if TYPE_CHECKING:
    import torch

__all__ = [
    "DemoRecord",
    "FeatureStats",
    "NodeFeaturizer",
    "NodeTable",
    "NodeTableBuilder",
    "NodeTableCodec",
    "NodeTableValidator",
    "Segment",
    "TimeEncoder",
    "coarsen_segments",
    "episode_phase",
    "psi",
    "travelled_frames",
]

_DTYPE: Final[dict[str, np.dtype]] = {
    "f4": np.dtype(np.float32),
    "u1": np.dtype(np.uint8),
    "i2": np.dtype(np.int16),
    "i4": np.dtype(np.int32),
}

_PER_NODE_FIELDS: Final[tuple[str, ...]] = (
    "q", "qdot", "grip", "stage", "t_idx", "t_frac", "t_raw",
    "n_members", "owner", "task_id", "phase", "is_virtual", "arm", "pair",
)

_FINITE_FIELDS: Final[tuple[str, ...]] = (
    "q", "qdot", "t_frac", "phase", "q_raw", "qdot_raw", "ee_raw", "a_raw",
)

_PHASE_MAX_MIN: Final[float] = 0.99
_PHASE_MIN_MAX: Final[float] = 0.01
_SUMMARY_ATOL: Final[float] = 1e-4
_STD_FLOOR: Final[float] = 1e-6


# ==== Time / phase primitives ====
def travelled_frames(q: ArrayLike | None, n_frames: int) -> NDArray[np.int64]:
    """Counts the moving frames before each frame of one episode.

    Args:
        q: [T, D] per-frame joints, or None to count every frame as moving.
        n_frames: Episode length T.

    Returns:
        [T] i8 array, non-decreasing, starting at 0.
    """
    # A frame whose joints moved less than STILL_TOL adds nothing, so a stretch where the arm
    # holds still while the other one works does not count as progress.
    moved = np.ones(n_frames, dtype=np.int64)
    if q is not None and n_frames > 0:
        q_arr = np.asarray(q, dtype=np.float64).reshape(n_frames, -1)
        moved[1:] = np.linalg.norm(np.diff(q_arr, axis=0), axis=1) >= STILL_TOL
        moved[0] = 1
    return np.cumsum(moved) - moved


def episode_phase(q: ArrayLike) -> NDArray[np.float32]:
    """Computes canonical phase over a full episode.

    Args:
        q: [T, D] per-frame joints.

    Returns:
        [T] f32 array spanning [0, 1].
    """
    # schema.PHASE_CONV: the one definition of "how far along" -- motion covered over motion to
    # cover, never elapsed time, or a parked arm reads as advancing through its task.
    q_arr = np.asarray(q)
    travelled = travelled_frames(q_arr, len(q_arr))
    if len(travelled) == 0:
        return np.zeros(0, dtype=np.float32)
    return (travelled / max(int(travelled[-1]), 1)).astype(np.float32)


class TimeEncoder:
    """Sinusoidal encoder for a normalised time fraction."""

    def __init__(self, n_freq: int = schema.PSI_FREQS) -> None:
        """Initializes the frequency ladder.

        Args:
            n_freq: Number of octaves.

        Raises:
            ValueError: If n_freq is not positive.
        """
        if n_freq < 1:
            raise ValueError(f"n_freq must be >= 1, got {n_freq}")
        self._n_freq = int(n_freq)
        self._freqs = (2.0 ** np.arange(self._n_freq, dtype=np.float64)) * np.pi

    @property
    def width(self) -> int:
        """Returns the number of output columns."""
        return 2 * self._n_freq

    def encode(self, t_frac: ArrayLike) -> NDArray[np.float32]:
        """Encodes time fractions.

        Args:
            t_frac: [N] values, nominally in [0, 1].

        Returns:
            [N, 2 * n_freq] f32 array.
        """
        flat = np.asarray(t_frac, dtype=np.float32).reshape(-1)
        angles = flat[:, None].astype(np.float64) * self._freqs[None, :]
        out = np.empty((flat.shape[0], self.width), dtype=np.float32)
        out[:, 0::2] = np.sin(angles)
        out[:, 1::2] = np.cos(angles)
        return out


@lru_cache(maxsize=8)
def _time_encoder(n_freq: int) -> TimeEncoder:
    """Cached encoder for repeated feature calls.

    Args:
        n_freq: Number of octaves.

    Returns:
        A shared TimeEncoder instance.
    """
    return TimeEncoder(n_freq)


def psi(t_frac: ArrayLike, n_freq: int = schema.PSI_FREQS) -> NDArray[np.float32]:
    """Sinusoidal time encoding wrapper.

    Args:
        t_frac: [N] values in [0, 1].
        n_freq: Number of octaves.

    Returns:
        [N, 2 * n_freq] f32 array.
    """
    return _time_encoder(n_freq).encode(t_frac)


def coarsen_segments(grip: ArrayLike, coarsen: int, q: ArrayLike | None = None) -> NDArray[np.int32]:
    """Assigns coarse-node ids within one demo, cutting only on gripper changes.

    Args:
        grip: [T] u1 per-frame grasp flags.
        coarsen: Maximum moving frames folded into one node.
        q: [T, D] per-frame joints, or None to count every frame as moving.

    Returns:
        [T] i4 node ids, non-decreasing.

    Raises:
        ValueError: If coarsen < 1.
    """
    if coarsen < 1:
        raise ValueError(f"coarsen must be >= 1, got {coarsen}")
    grip_arr = np.asarray(grip)
    n = len(grip_arr)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    idx = np.arange(n)
    changed = np.empty(n, dtype=bool)
    changed[0] = True
    changed[1:] = grip_arr[1:] != grip_arr[:-1]

    last_reset = np.maximum.accumulate(np.where(changed, idx, -1))
    # Only frames that moved spend the coarsen budget: a stretch where the arm holds still folds
    # into one node however long it lasts, so a parked arm is not a thousand copies of one pose.
    travelled = travelled_frames(q, n)
    pos_in_run = travelled - travelled[last_reset]
    bucket = pos_in_run // coarsen
    is_new_node = np.empty(n, dtype=bool)
    is_new_node[0] = True
    is_new_node[1:] = changed[1:] | (bucket[1:] != bucket[:-1])
    return (np.cumsum(is_new_node) - 1).astype(np.int32)


# ==== Value objects ====
def _as_f32_or_none(value: ArrayLike | None) -> NDArray[np.float32] | None:
    """Casts an optional array to f32, passing None through.

    Args:
        value: Array-like, or None.

    Returns:
        The f32 array, or None.
    """
    return None if value is None else np.asarray(value, dtype=np.float32)


@dataclass(slots=True)
class DemoRecord:
    """One demonstration trajectory.

    Attributes:
        q: [T, D] f32 joint positions.
        qdot: [T, D] f32 per-raw-frame velocities.
        grip: [T] u1, 1 = grasping.
        stage: [T] i2 release count.
        task_id: Index into task_names.
        ee: [T, 6] f32 end-effector pose, concat(position, unnormalized axis-angle orientation).
        a: [T, 7] f32 recorded action commands.
        arm: Which arm of the robot this demonstration is; 0 on a single-arm robot.
        pair: The episode both arms share, or -1 when this demo has no partner.
    """
    q: NDArray[np.float32]
    qdot: NDArray[np.float32]
    grip: NDArray[np.uint8]
    stage: NDArray[np.int16]
    task_id: int
    ee: NDArray[np.float32] | None = None
    a: NDArray[np.float32] | None = None
    arm: int = 0
    pair: int = -1

    def __post_init__(self) -> None:
        # Only a synthetic demo reaches here without the two: robot_ood.build.graph reads
        # both off the HDF5 and raises when either is absent.
        if self.ee is None:
            self.ee = np.zeros((len(self), POSE_DIM), dtype=np.float32)
        if self.a is None:
            self.a = np.zeros((len(self), ACTION_DIM), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.q.shape[0])

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DemoRecord:
        """Adapts a legacy per-demo dict to a DemoRecord.

        Args:
            payload: Mapping with keys q, qdot, grip, stage, task_id, and optionally ee and a.

        Returns:
            The DemoRecord instance.
        """
        return cls(
            q=np.asarray(payload["q"], dtype=np.float32),
            qdot=np.asarray(payload["qdot"], dtype=np.float32),
            grip=np.asarray(payload["grip"]),
            stage=np.asarray(payload["stage"]),
            task_id=int(payload["task_id"]),
            arm=int(payload.get("arm", 0)),
            pair=int(payload.get("pair", -1)),
            ee=_as_f32_or_none(payload.get("ee")),
            a=_as_f32_or_none(payload.get("a")),
        )


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous run of raw frames decoded from one node.

    Attributes:
        q: [k, D] f64 positions.
        qdot: [k, D] f64 velocities.
    """
    q: NDArray[np.float64]
    qdot: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.q.shape[0])

    def __iter__(self):
        """Yields q then qdot, so q, qdot = table.segment(...) works."""
        yield self.q
        yield self.qdot


@dataclass(frozen=True, slots=True)
class FeatureStats:
    """Normalisation statistics over the unbounded feature prefix.

    Attributes:
        mean: [2D+1] f32 column means.
        std: [2D+1] f32 column stds, floored away from zero.
    """
    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    def __iter__(self):
        """Yields mean then std, so mean, std = table.feature_stats() works."""
        yield self.mean
        yield self.std

    def normalize(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Applies statistics to the prefix columns of a feature block.

        Args:
            features: [N, F] raw features.

        Returns:
            [N, F] f32 with the prefix normalised.
        """
        out = np.array(features, dtype=np.float32, copy=True)
        width = self.mean.shape[0]
        out[:, :width] = (out[:, :width] - self.mean) / self.std
        return out


@dataclass(slots=True)
class _DemoNodes:
    """Coarse-node arrays for a single demo."""
    q: NDArray[np.float32]
    qdot: NDArray[np.float32]
    grip: NDArray[Any]
    stage: NDArray[Any]
    t_idx: NDArray[np.int32]
    t_frac: NDArray[np.float32]
    t_raw: NDArray[np.int32]
    n_members: NDArray[np.int16]
    owner: NDArray[np.int32]
    task_id: NDArray[np.int16]
    phase: NDArray[np.float32]
    arm: NDArray[np.uint8]
    pair: NDArray[np.int32]

    def __len__(self) -> int:
        return int(self.q.shape[0])


# ==== The table ====
@dataclass(eq=False)
class NodeTable:
    """One row per coarse node, plus every raw frame."""

    # ---- per-node fields (length V) ----
    q: NDArray[np.float32]
    qdot: NDArray[np.float32]
    grip: NDArray[np.uint8]
    stage: NDArray[np.int16]
    t_idx: NDArray[np.int32]
    t_frac: NDArray[np.float32]
    t_raw: NDArray[np.int32]
    n_members: NDArray[np.int16]
    owner: NDArray[np.int32]
    task_id: NDArray[np.int16]
    phase: NDArray[np.float32]

    # ---- CSR-style per-demo pointers (length n_demos + 1) ----
    demo_ptr: NDArray[np.int32]
    raw_ptr: NDArray[np.int32]

    # ---- every raw frame: ee_raw is world-frame [N_raw,6]; a_raw is [N_raw,7] commands ----
    q_raw: NDArray[np.float32]
    qdot_raw: NDArray[np.float32]
    ee_raw: NDArray[np.float32]
    a_raw: NDArray[np.float32]

    # ---- metadata ----
    task_names: list[str]
    coarsen: int

    # [V] u1, 1 = synthesised entry node. Virtual nodes are always the suffix [n_real, V).
    is_virtual: NDArray[np.uint8] | None = None

    # ---- bimanual columns; a single-arm graph is arm 0 and pair -1 throughout ----
    # [V] u1 which arm of the robot the node's configuration belongs to.
    arm: NDArray[np.uint8] | None = None
    # [V] i4 the episode two arms share: two nodes with the same pair and different arm were
    # recorded at the same time on the same robot. -1 when the demo has no partner.
    pair: NDArray[np.int32] | None = None

    def __post_init__(self) -> None:
        self._ee_base: NDArray[np.float32] | None = None
        if self.is_virtual is None:
            self.is_virtual = np.zeros(len(self), dtype=_DTYPE["u1"])
        if self.arm is None:
            self.arm = np.zeros(len(self), dtype=_DTYPE["u1"])
        if self.pair is None:
            self.pair = np.full(len(self), -1, dtype=_DTYPE["i4"])

    @property
    def bimanual(self) -> bool:
        """Whether this table carries both arms of a robot."""
        return bool(np.asarray(self.arm).max(initial=0) > 0)

    def span(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Per node, the first and last RAW frame it covers, in its own demo's frame numbering."""
        lo = np.asarray(self.t_raw, dtype=np.int64)
        return lo.astype(np.int32), (lo + np.asarray(self.n_members, dtype=np.int64) - 1).astype(np.int32)

    def __len__(self) -> int:
        return int(self.q.shape[0])

    @property
    def ee_base(self) -> NDArray[np.float32]:
        """[N_raw, 6] f32 concat(position, unnormalized axis-angle) in robot base frame.

        Returns:
            The pose table.

        Raises:
            ValueError: q_raw is not a 7-DoF Panda configuration. Assign a table instead when it is
                some other robot.
            ImportError: pytorch_kinematics is not installed in this interpreter.
        """
        if self._ee_base is None:
            from robot_ood.core.kinematics import PandaKinematics

            self.ee_base = PandaKinematics().pose(np.asarray(self.q_raw, dtype=np.float64))
        return self._ee_base

    @ee_base.setter
    def ee_base(self, pose: ArrayLike) -> None:
        """Supply the base-frame table directly, for a corpus the Panda FK above does not describe.

        Args:
            pose: [N_raw, 6] base-frame poses.

        Raises:
            ValueError: pose is not [N_raw, 6].
        """
        arr = np.asarray(pose, dtype=np.float32)
        if arr.shape != (self.n_raw, POSE_DIM):
            raise ValueError(
                f"ee_base must be [Nraw,{POSE_DIM}]=[{self.n_raw},{POSE_DIM}], got {arr.shape}"
            )
        self._ee_base = arr

    @property
    def dim(self) -> int:
        return int(self.q.shape[1])

    @property
    def n_demos(self) -> int:
        return int(self.demo_ptr.shape[0] - 1)

    @property
    def n_tasks(self) -> int:
        return len(self.task_names)

    @property
    def n_raw(self) -> int:
        return int(self.q_raw.shape[0])

    @property
    def n_real(self) -> int:
        """Nodes decoded from actual demo frames -- the length of the real prefix."""
        return int(self.demo_ptr[-1])

    @property
    def virtual_mask(self) -> NDArray[np.bool_]:
        """[V] bool view of is_virtual, for masking rather than arithmetic."""
        return np.asarray(self.is_virtual, dtype=bool)

    def with_virtual_nodes(
        self, parent: ArrayLike, q_virtual: ArrayLike
    ) -> NodeTable:
        """Append synthesised nodes shadowing real ones at perturbed configurations.

        Args:
            parent: [M] real node ids each virtual node is grown from.
            q_virtual: [M, D] perturbed configurations, one per parent entry.

        Returns:
            A new table with M rows appended. Self is unchanged.

        Raises:
            ValueError: A parent id is out of range or already virtual, or the shapes disagree.
        """
        # Every column but q is copied from the parent, so downstream lookups (segment,
        # ee_segment, raw_span) resolve a virtual node to its parent's demo frames.
        parent_idx = np.asarray(parent, dtype=np.int64).reshape(-1)
        q_new = np.asarray(q_virtual, dtype=np.float32)
        if q_new.shape != (parent_idx.shape[0], self.dim):
            raise ValueError(
                f"q_virtual must be [M,D]=[{parent_idx.shape[0]},{self.dim}], got {q_new.shape}"
            )
        if parent_idx.size and (
            parent_idx.min() < 0 or parent_idx.max() >= self.n_real
        ):
            raise ValueError("parent ids must index the real prefix [0, n_real)")

        columns = {
            name: np.concatenate([getattr(self, name), getattr(self, name)[parent_idx]], axis=0)
            for name in _PER_NODE_FIELDS
        }
        columns["q"] = np.concatenate([self.q, q_new], axis=0).astype(_DTYPE["f4"])
        columns["is_virtual"] = np.concatenate(
            [self.is_virtual, np.ones(parent_idx.shape[0], dtype=_DTYPE["u1"])]
        )
        table = NodeTable(
            **columns,
            demo_ptr=self.demo_ptr, raw_ptr=self.raw_ptr,
            q_raw=self.q_raw, qdot_raw=self.qdot_raw, ee_raw=self.ee_raw, a_raw=self.a_raw,
            task_names=list(self.task_names), coarsen=self.coarsen,
        )
        table.validate()
        return table

    @classmethod
    def from_demos(
        cls,
        demos: Sequence[DemoRecord | Mapping[str, Any]],
        task_names: Sequence[str],
        coarsen: int = schema.COARSEN,
    ) -> NodeTable:
        """Builds a validated table from loaded demonstrations.

        Args:
            demos: Records, or legacy per-demo mappings.
            task_names: Task name per task_id.
            coarsen: Maximum raw frames per node.

        Returns:
            The built NodeTable.
        """
        return NodeTableBuilder(coarsen=coarsen).build(demos, task_names)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Writes g_nodes.npz.

        Args:
            path: Destination file or parent directory.

        Returns:
            The written path.
        """
        return NodeTableCodec().save(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, strict: bool = True) -> NodeTable:
        """Reads g_nodes.npz.

        Args:
            path: Source file or parent directory.
            strict: Verify the stamped phase_conv and run full validation.

        Returns:
            The loaded table.
        """
        return NodeTableCodec().load(path, strict=strict)

    def validate(self) -> None:
        """Enforces every structural invariant."""
        NodeTableValidator(self).run()

    def features(self) -> NDArray[np.float32]:
        """Generates raw, unnormalised network features.

        Returns:
            [V, 2D+1+2*PSI_FREQS] f32 array.
        """
        return NodeFeaturizer().encode(self)

    def feature_stats(self) -> FeatureStats:
        """Calculates statistics over the unbounded prefix.

        Returns:
            Mean and floored std.
        """
        return NodeFeaturizer().stats(self)

    def raw_span(self, node: int, k: int) -> tuple[int, int]:
        """Resolves the raw-frame slice a k-frame segment starting at a node may read.

        Args:
            node: Node index.
            k: Number of frames requested.

        Returns:
            (start, end) global raw-frame offsets, with end - start <= k.

        Raises:
            ValueError: If k < 1 or t_raw[node] falls outside its demo's range.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        owner = int(self.owner[node])
        start = int(self.t_raw[node])
        demo_start = int(self.raw_ptr[owner])
        demo_end = int(self.raw_ptr[owner + 1])
        if not demo_start <= start < demo_end:
            raise ValueError(
                f"t_raw[{node}]={start} outside demo {owner}'s raw range "
                f"[{demo_start}, {demo_end})"
            )
        # Clipped at the demo's raw end: never reaches past raw_ptr[owner+1] into the next demo.
        return start, min(start + k, demo_end)

    def segment(self, node: int, k: int) -> Segment:
        """Decodes k raw frames starting at a node, edge-padded to exactly k frames.

        Args:
            node: Node index.
            k: Number of frames to return.

        Returns:
            The decoded Segment.

        Raises:
            ValueError: If k < 1 or t_raw[node] falls outside its demo's range.
        """
        start, end = self.raw_span(node, k)
        return Segment(
            q=_edge_pad(self.q_raw[start:end].astype(np.float64), k),
            qdot=_edge_pad(self.qdot_raw[start:end].astype(np.float64), k),
        )

    def ee_segment(self, node: int, k: int) -> NDArray[np.float64]:
        """Decodes k raw end-effector poses in robot base frame, edge-padded to k frames.

        Args:
            node: Node index.
            k: Number of frames to return.

        Returns:
            [k, 6] f64 concat(position, unnormalized axis-angle orientation) off ee_base.

        Raises:
            ValueError: If k < 1 or t_raw[node] falls outside its demo's range.
        """
        start, end = self.raw_span(node, k)
        return _edge_pad(self.ee_base[start:end].astype(np.float64), k)

    def action_chunks(self, k: int) -> NDArray[np.float32]:
        """[V, k, 7] the demo's recorded actions from each node's first frame, edge-padded at the demo end."""
        t_raw = self.t_raw[: self.n_real].astype(np.int64)
        last = self.raw_ptr[self.owner[: self.n_real].astype(np.int64) + 1].astype(np.int64) - 1
        reach = np.minimum(t_raw[:, None] + np.arange(k, dtype=np.int64)[None, :], last[:, None])
        out = np.asarray(self.a_raw, dtype=np.float32)[reach]
        if len(self) > self.n_real:
            out = np.concatenate([out, np.zeros((len(self) - self.n_real, k, out.shape[-1]), np.float32)])
        return out

    def demo_nodes(self, owner: int) -> NDArray[np.int64]:
        """Retrieves node indices belonging to one demo.

        Args:
            owner: Demo id.

        Returns:
            Indices in t_idx order.
        """
        return np.arange(
            int(self.demo_ptr[owner]), int(self.demo_ptr[owner + 1]), dtype=np.int64
        )

    def node_at_raw(self, owner: int, raw_idx: int) -> int:
        """Finds the node on owner containing raw_idx.

        Args:
            owner: Demo id.
            raw_idx: Global raw-frame offset.

        Returns:
            Node index.
        """
        return int(self.node_at_raw_many(owner, raw_idx).reshape(()))

    def node_at_raw_many(
        self, owner: ArrayLike, raw_idx: ArrayLike
    ) -> NDArray[np.int64]:
        """Vectorised form of node_at_raw.

        Args:
            owner: Demo ids, broadcastable against raw_idx.
            raw_idx: Global raw-frame offsets.

        Returns:
            Node indices with the broadcast shape.
        """
        owner_arr, raw_arr = np.broadcast_arrays(
            np.asarray(owner, dtype=np.int64), np.asarray(raw_idx, dtype=np.int64)
        )
        # t_raw is a global, monotone offset over the real prefix (virtual nodes excluded,
        # since they repeat their parent's t_raw), so one searchsorted answers every row.
        t_raw = self.t_raw[: self.n_real].astype(np.int64, copy=False)
        demo_ptr = self.demo_ptr.astype(np.int64, copy=False)

        flat_owner = owner_arr.reshape(-1)
        pos = np.searchsorted(t_raw, raw_arr.reshape(-1), side="right") - 1
        lo = demo_ptr[flat_owner]
        hi = demo_ptr[flat_owner + 1] - 1
        return np.clip(pos, lo, hi).reshape(owner_arr.shape).astype(np.int64)

    def as_torch(self, device: str | None = None) -> dict[str, torch.Tensor]:
        """Returns every schema array as a torch tensor.

        Args:
            device: Target device, or None for CPU.

        Returns:
            Mapping of field names to tensors.
        """
        # f4 columns become float32, id columns become int64; torch is imported lazily so
        # this module stays numpy-only until a caller reaches here.
        torch_mod = _import_torch()
        out: dict[str, torch.Tensor] = {}
        for key, code in schema.NODES_KEYS.items():
            arr = getattr(self, key)
            cast = arr.astype(np.float32) if code == "f4" else arr.astype(np.int64)
            tensor = torch_mod.as_tensor(cast)
            out[key] = tensor.to(device) if device is not None else tensor
        return out

    # Historical name; EdgeSet.torch still spells it this way.
    torch = as_torch


def _import_torch() -> Any:
    """Lazily imports torch."""
    import torch as torch_mod
    return torch_mod


# ==== Builder ====
class NodeTableBuilder:
    """Coarsens loaded demonstrations into a NodeTable."""

    def __init__(self, *, coarsen: int = schema.COARSEN, validate: bool = True) -> None:
        """Configures the builder.

        Args:
            coarsen: Maximum raw frames per node.
            validate: Run NodeTableValidator on the result.

        Raises:
            ValueError: If coarsen < 1.
        """
        if coarsen < 1:
            raise ValueError(f"coarsen must be >= 1, got {coarsen}")
        self._coarsen = int(coarsen)
        self._validate = validate

    def build(
        self,
        demos: Sequence[DemoRecord | Mapping[str, Any]],
        task_names: Sequence[str],
    ) -> NodeTable:
        """Builds a table.

        Args:
            demos: Records, or legacy per-demo mappings.
            task_names: Task name per task_id.

        Returns:
            The built NodeTable.

        Raises:
            ValueError: If demos is empty.
        """
        records = [
            d if isinstance(d, DemoRecord) else DemoRecord.from_mapping(d)
            for d in demos
        ]
        if not records:
            raise ValueError("demos must be non-empty")

        chunks: list[_DemoNodes] = []
        q_raw: list[NDArray[np.float32]] = []
        qdot_raw: list[NDArray[np.float32]] = []
        ee_raw: list[NDArray[np.float32]] = []
        a_raw: list[NDArray[np.float32]] = []
        demo_ptr: list[int] = [0]
        raw_ptr: list[int] = [0]
        raw_offset = 0

        for demo_id, record in enumerate(records):
            chunk = self._coarsen_demo(record, demo_id=demo_id, raw_offset=raw_offset)
            chunks.append(chunk)
            q_raw.append(record.q)
            qdot_raw.append(record.qdot)
            ee_raw.append(record.ee)
            a_raw.append(record.a)
            demo_ptr.append(demo_ptr[-1] + len(chunk))
            raw_ptr.append(raw_ptr[-1] + len(record))
            raw_offset += len(record)

        table = NodeTable(
            **self._concat_chunks(chunks),
            demo_ptr=np.asarray(demo_ptr, dtype=_DTYPE["i4"]),
            raw_ptr=np.asarray(raw_ptr, dtype=_DTYPE["i4"]),
            q_raw=_stack(q_raw, "f4"),
            qdot_raw=_stack(qdot_raw, "f4"),
            ee_raw=_stack(ee_raw, "f4"),
            a_raw=_stack(a_raw, "f4"),
            task_names=list(task_names),
            coarsen=self._coarsen,
        )
        if self._validate:
            table.validate()
        return table

    def _coarsen_demo(
        self, record: DemoRecord, *, demo_id: int, raw_offset: int
    ) -> _DemoNodes:
        """Coarsens one demo into per-node arrays.

        Args:
            record: The demonstration.
            demo_id: Global demo id.
            raw_offset: Running raw-frame offset.

        Returns:
            The demo's node chunk.
        """
        q = np.asarray(record.q, dtype=np.float32)

        node_id = coarsen_segments(record.grip, self._coarsen, q)
        n_nodes = int(node_id[-1]) + 1
        bounds = np.searchsorted(node_id, np.arange(n_nodes + 1))
        starts = bounds[:-1]
        ends = bounds[1:] - 1

        phase = episode_phase(q)
        return _DemoNodes(
            q=q[starts],
            # Coarse-time DISPLACEMENT, never the mean of per-step deltas.
            qdot=q[ends] - q[starts],
            grip=np.asarray(record.grip)[starts],
            stage=np.asarray(record.stage)[starts],
            arm=np.full(n_nodes, record.arm, dtype=_DTYPE["u1"]),
            pair=np.full(n_nodes, record.pair, dtype=_DTYPE["i4"]),
            t_idx=np.arange(n_nodes, dtype=np.int32),
            t_frac=(np.arange(n_nodes) / max(1, n_nodes - 1)).astype(np.float32),
            t_raw=starts.astype(np.int32) + raw_offset,
            n_members=(ends - starts + 1).astype(np.int16),
            owner=np.full(n_nodes, demo_id, dtype=np.int32),
            task_id=np.full(n_nodes, int(record.task_id), dtype=np.int16),
            phase=phase[starts],
        )

    @staticmethod
    def _concat_chunks(chunks: Sequence[_DemoNodes]) -> dict[str, NDArray[Any]]:
        """Concatenates per-demo chunks field-wise.

        Args:
            chunks: Per-demo node arrays.

        Returns:
            Mapping of field names to table-wide arrays.
        """
        codes = {name: schema.NODES_KEYS[name] for name in _PER_NODE_FIELDS}
        return {
            name: _stack([getattr(c, name) for c in chunks], codes[name])
            for name in (f.name for f in fields(_DemoNodes))
        }


def _stack(parts: Sequence[NDArray[Any]], code: str) -> NDArray[Any]:
    """Concatenates along axis 0 and casts to a schema dtype."""
    return np.concatenate(parts, axis=0).astype(_DTYPE[code])


def _edge_pad(block: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Repeats a block's last row until it has exactly k rows.

    Args:
        block: [n, W] f64 rows, n >= 1.
        k: Target row count; n >= k returns block unchanged.

    Returns:
        [k, W] f64 rows.
    """
    pad = k - block.shape[0]
    if pad <= 0:
        return block
    return np.concatenate([block, np.repeat(block[-1:], pad, axis=0)], axis=0)


# ==== Validator ====
class NodeTableValidator:
    """Enforces the structural invariants of a NodeTable."""

    def __init__(self, table: NodeTable) -> None:
        """Binds a table.

        Args:
            table: The table to check.
        """
        self._table = table
        self._n_nodes = len(table)
        self._n_demos = table.n_demos
        self._n_raw = table.n_raw
        # Every layout invariant below is a statement about DEMO frames, so it is checked on the
        # real prefix. Virtual nodes get check_virtual_layout instead.
        self._n_real = table.n_real

    def run(self) -> None:
        """Runs every default check in dependency order.

        Raises:
            TypeError: On a dtype mismatch.
            ValueError: On any shape, layout, or coarsening violation.
        """
        self.check_dtypes()
        self.check_shapes()
        self.check_finite()
        self.check_domains()
        if self._n_nodes == 0:
            return
        self.check_phase_convention()
        self.check_owner_layout()
        self.check_raw_layout()
        self.check_coarsening()
        self.check_virtual_layout()

    def check_dtypes(self) -> None:
        """Verifies schema dtypes."""
        for key, code in schema.NODES_KEYS.items():
            dtype = getattr(self._table, key).dtype
            if dtype != _DTYPE[code]:
                raise TypeError(
                    f"{key}: dtype {dtype} != schema.NODES_KEYS[{key!r}]={_DTYPE[code]}"
                )

    def check_shapes(self) -> None:
        """Verifies array shapes."""
        table, n_nodes, dim = self._table, self._n_nodes, self._table.dim
        for key in _PER_NODE_FIELDS:
            length = getattr(table, key).shape[0]
            if length != n_nodes:
                raise ValueError(f"{key}: length {length} != V={n_nodes}")
        if table.q.shape != (n_nodes, dim) or table.qdot.shape != (n_nodes, dim):
            raise ValueError(f"q/qdot must be [V,D]=[{n_nodes},{dim}]")
        if table.q_raw.shape != (self._n_raw, dim) or table.qdot_raw.shape != (self._n_raw, dim):
            raise ValueError(f"q_raw/qdot_raw must be [Nraw,D]=[{self._n_raw},{dim}]")
        if table.ee_raw.shape != (self._n_raw, POSE_DIM):
            raise ValueError(f"ee_raw must be [Nraw,{POSE_DIM}]=[{self._n_raw},{POSE_DIM}]")
        if table.a_raw.shape != (self._n_raw, ACTION_DIM):
            raise ValueError(f"a_raw must be [Nraw,{ACTION_DIM}]=[{self._n_raw},{ACTION_DIM}]")
        expected_ptr = (self._n_demos + 1,)
        if table.demo_ptr.shape != expected_ptr or table.raw_ptr.shape != expected_ptr:
            raise ValueError("demo_ptr/raw_ptr must have length n_demos+1")

    def check_finite(self) -> None:
        """Verifies float fields contain no NaN or inf."""
        for key in _FINITE_FIELDS:
            if not np.all(np.isfinite(getattr(self._table, key))):
                raise ValueError(f"{key} contains NaN/inf")

    def check_domains(self) -> None:
        """Verifies categorical values."""
        if self._n_nodes == 0:
            return
        grip = self._table.grip
        if not np.all((grip == 0) | (grip == 1)):
            raise ValueError("grip must be 0/1")
        if not np.all(self._table.n_members >= 1):
            raise ValueError("n_members must be >= 1")

    def check_phase_convention(self) -> None:
        """Verifies global phase spans [0, 1]."""
        phase = self._table.phase
        p_max, p_min = float(phase.max()), float(phase.min())
        if not (p_max > _PHASE_MAX_MIN and p_min < _PHASE_MIN_MAX):
            raise ValueError(
                f"phase convention violated: table-wide max={p_max:.4f} min={p_min:.4f}, "
                f"expected max>{_PHASE_MAX_MIN} and min<{_PHASE_MIN_MAX}"
            )

    def check_owner_layout(self) -> None:
        """Verifies owner contiguity over the real prefix."""
        table = self._table
        n_real = self._n_real
        owner = table.owner[:n_real]
        if not np.all(np.diff(owner) >= 0):
            raise ValueError("owner must be non-decreasing")
        node_counts = np.bincount(owner, minlength=self._n_demos)
        if (
            table.demo_ptr[0] != 0
            or table.demo_ptr[-1] != n_real
            or not np.array_equal(np.diff(table.demo_ptr), node_counts)
        ):
            raise ValueError("demo_ptr inconsistent with bincount(owner)")

        expected_t_idx = np.arange(n_real) - table.demo_ptr[owner]
        if not np.array_equal(table.t_idx[:n_real].astype(np.int64), expected_t_idx.astype(np.int64)):
            raise ValueError("t_idx is not contiguous within demo")

    def check_raw_layout(self) -> None:
        """Verifies raw pointers."""
        table = self._table
        n_real = self._n_real
        owner = table.owner[:n_real]
        t_raw = table.t_raw[:n_real]
        n_members = table.n_members[:n_real]
        raw_counts = np.bincount(
            owner, weights=n_members.astype(np.int64), minlength=self._n_demos,
        ).astype(np.int64)
        if (
            table.raw_ptr[0] != 0
            or table.raw_ptr[-1] != self._n_raw
            or not np.array_equal(np.diff(table.raw_ptr).astype(np.int64), raw_counts)
        ):
            raise ValueError("raw_ptr inconsistent with n_members sum")

        if n_real > 1:
            same_owner = owner[:-1] == owner[1:]
            contiguous = t_raw[1:] == (t_raw[:-1] + n_members[:-1])
            if not np.all(~same_owner | contiguous):
                raise ValueError("t_raw is not contiguous within a demo")

        raw_start = table.raw_ptr[table.owner]
        raw_end = table.raw_ptr[table.owner + 1]
        if not (
            np.all(table.t_raw >= raw_start) and np.all(table.t_raw + table.n_members <= raw_end)
        ):
            raise ValueError("t_raw falls outside its own demo's raw range")

    def check_coarsening(self) -> None:
        """Verifies summaries agree with raw frames."""
        table = self._table
        n_real = self._n_real
        t_raw = table.t_raw[:n_real]
        last_idx = t_raw + table.n_members[:n_real].astype(np.int32) - 1
        if not np.allclose(table.q[:n_real], table.q_raw[t_raw], atol=_SUMMARY_ATOL):
            raise ValueError("q does not match q_raw[t_raw]")

        expected_qdot = table.q_raw[last_idx] - table.q_raw[t_raw]
        if not np.allclose(table.qdot[:n_real], expected_qdot, atol=_SUMMARY_ATOL):
            raise ValueError("qdot must be the NET displacement across node's raw span")

    def check_virtual_layout(self) -> None:
        """Verifies is_virtual is a 0/1 suffix whose rows shadow real parents."""
        table = self._table
        flag = np.asarray(table.is_virtual)
        if not np.all((flag == 0) | (flag == 1)):
            raise ValueError("is_virtual must be 0/1")
        n_real = self._n_real
        if flag[:n_real].any() or not np.all(flag[n_real:] == 1):
            raise ValueError(
                f"is_virtual must be all-zero on [0,{n_real}) and all-one on "
                f"[{n_real},{self._n_nodes}) -- virtual nodes are a suffix"
            )
        if n_real == self._n_nodes:
            return
        parent = table.demo_ptr[table.owner[n_real:]] + table.t_idx[n_real:]
        if not np.array_equal(table.t_raw[n_real:], table.t_raw[parent]):
            raise ValueError("a virtual node's (owner, t_idx) does not resolve to its parent's t_raw")


# ==== Codec ====
def _suite_hint(src: Path) -> str:
    """Guesses the suite an artifact belongs to, for a rebuild instruction.

    Args:
        src: Path to a g_nodes.npz, nominally under outputs/<suite>/<run>/artifacts/.

    Returns:
        The suite name, or a "<suite>" placeholder when the layout says nothing.
    """
    from robot_ood.core.paths import suite_from_artifacts

    return suite_from_artifacts(src)


# What to tell a caller whose g_nodes.npz is missing.
MISSING_NODES = "no node table. Build the graph with `python -m robot_ood.cli build graph --suite <suite>`."


class NodeTableCodec:
    """Reads and writes the NodeTable artifact."""

    def save(self, table: NodeTable, path: str | os.PathLike[str]) -> Path:
        """Writes a table.

        Args:
            table: Table to persist.
            path: Destination file or parent directory.

        Returns:
            The written path.
        """
        payload: dict[str, Any] = {k: getattr(table, k) for k in schema.NODES_KEYS}
        payload["task_names"] = np.array(table.task_names)
        payload["coarsen"] = np.array(table.coarsen)
        payload["phase_conv"] = np.array(schema.PHASE_CONV)
        if table._ee_base is not None:
            payload["ee_base"] = table._ee_base
        return npz.write_arrays(path, schema.NODES_NPZ, payload, compress=True)

    def load(self, path: str | os.PathLike[str], *, strict: bool = True) -> NodeTable:
        """Reads a table.

        Args:
            path: Source file or parent directory.
            strict: Verify the stamped phase_conv and run full validation.

        Returns:
            The loaded table.

        Raises:
            FileNotFoundError: Nothing is at the resolved path.
            ValueError: If the artifact predates a schema column, or if strict and its
                phase convention is foreign.
        """
        src, archive = npz.read(path, schema.NODES_NPZ, MISSING_NODES)
        self._check_columns(list(archive), src)
        arrays = {
            key: self._column(archive, key, code)
            for key, code in schema.NODES_KEYS.items()
        }
        task_names = [str(x) for x in archive["task_names"]]
        coarsen = int(archive["coarsen"])
        stamp = str(archive["phase_conv"]) if "phase_conv" in archive else ""

        table = NodeTable(**arrays, task_names=task_names, coarsen=coarsen)
        if "ee_base" in archive:
            table.ee_base = np.asarray(archive["ee_base"], dtype=_DTYPE["f4"])
        if strict:
            self._check_stamp(stamp, src)
            table.validate()
        return table

    @staticmethod
    def _column(archive: Mapping[str, Any], key: str, code: str) -> NDArray[Any]:
        """Reads one node column, zero-filling an absent OPTIONAL one.

        Args:
            archive: The opened npz.
            key: Column name.
            code: Schema dtype code.

        Returns:
            The column, [V] zeros when it is optional and the artifact predates it.
        """
        if key in archive:
            return np.asarray(archive[key]).astype(_DTYPE[code])
        if key == "pair":     # an absent pair column is "no partner", which is -1, not 0
            return np.full(len(archive["q"]), -1, dtype=_DTYPE[code])
        return np.zeros(len(archive["q"]), dtype=_DTYPE[code])

    @staticmethod
    def _check_columns(present: Sequence[str], src: Path) -> None:
        """Refuses an artifact written before a required schema column existed.

        Args:
            present: Column names found in the archive.
            src: Source path, for the error message.

        Raises:
            ValueError: If any required schema.NODES_KEYS column is absent.
        """
        missing = [key for key in schema.ArtifactSchemas.NODES.required_keys() if key not in present]
        if missing:
            raise ValueError(
                f"{src}: g_nodes.npz has no {', '.join(missing)} column -- it predates the "
                f"current schema. Rebuild it with: "
                f"python -m robot_ood.cli build graph --suite {_suite_hint(src)}"
            )

    @staticmethod
    def _check_stamp(stamp: str, src: Path) -> None:
        """Compares an artifact's phase-convention stamp against the schema.

        Args:
            stamp: The stamp read off the archive, or "" if absent.
            src: Source path, for the error message.

        Raises:
            ValueError: If the stamp is missing or foreign.
        """
        if stamp != schema.PHASE_CONV:
            raise ValueError(
                f"{src}: phase_conv={stamp or '<missing>'!r} != "
                f"schema.PHASE_CONV={schema.PHASE_CONV!r}. This file was not built by "
                "NodeTable.save. Rebuild via NodeTable.from_demos."
            )


# ==== Featurizer ====
class NodeFeaturizer:
    """Builds network features from a NodeTable."""

    def __init__(self, n_freq: int = schema.PSI_FREQS) -> None:
        """Configures the featurizer.

        Args:
            n_freq: Octaves in the time encoding.
        """
        self._encoder = _time_encoder(n_freq)

    def width(self, dim: int) -> int:
        """Returns total feature width for a configuration dimension."""
        return 2 * dim + 1 + self._encoder.width

    def encode(self, table: NodeTable) -> NDArray[np.float32]:
        """Builds raw, unnormalised features.

        Args:
            table: Source table.

        Returns:
            [V, 2D+1+2*n_freq] f32 array.
        """
        return np.concatenate(
            [self._prefix(table), self._encoder.encode(table.t_frac)], axis=1
        ).astype(np.float32)

    def stats(self, table: NodeTable) -> FeatureStats:
        """Computes normalisation statistics over the unbounded prefix.

        Args:
            table: Source table.

        Returns:
            Mean and floored std.
        """
        prefix = self._prefix(table)
        std = prefix.std(axis=0)
        std = np.where(std < _STD_FLOOR, 1.0, std)
        return FeatureStats(
            mean=prefix.mean(axis=0).astype(np.float32), std=std.astype(np.float32)
        )

    @staticmethod
    def _prefix(table: NodeTable) -> NDArray[np.float32]:
        """Returns the [q, qdot, grip] block needing normalisation."""
        grip = table.grip.astype(np.float32)[:, None]
        return np.concatenate([table.q, table.qdot, grip], axis=1).astype(np.float32)
