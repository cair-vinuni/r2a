"""The action vector: its layout, the SO(3) algebra on it, and the metres per action unit.

Dims 0:3 are a world-frame position delta (metres), 3:6 a world-frame rotation delta as axis-angle
radians composed as logmap(R_next @ R_now^T) rather than subtracted, and dim 6 a binary gripper
command; poses arrive in the robot base frame, which differs only by a per-scene translation. The
metres per action unit are the DC gain of an impulse response fitted over RESPONSE_TAPS control
steps, since a controller answers a command over more than one (torch twins: robot_ood.core.rotation).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

__all__ = [
    "ACTION_DIM",
    "EUCLIDEAN",
    "POSE_DIM",
    "POSE_SPACES",
    "SO3",
    "pose_space_for",
    "relative_rotation",
    "GRIP_COL",
    "POS_BLOCK",
    "ROT_BLOCK",
    "ACTION_LIMIT",
    "action_limit",
    "DEFAULT_MIN_R2",
    "RESPONSE_TAPS",
    "rotvec_to_matrix",
    "matrix_to_rotvec",
    "relative_rotvec",
    "DemoDeltas",
    "BlockFit",
    "fit_blocks",
    "ActionScale",
]


# Action layout: [dx, dy, dz, drx, dry, drz, gripper].
ACTION_DIM: Final[int] = 7
POSE_DIM: Final[int] = 6

# How dims 3:6 of a pose compose. so3: an axis-angle orientation, differenced by world-frame
# composition. euclidean: three more coordinates (a joint-space proxy), differenced by subtraction.
SO3: Final[str] = "so3"
EUCLIDEAN: Final[str] = "euclidean"
POSE_SPACES: Final[tuple[str, ...]] = (SO3, EUCLIDEAN)
GRIP_COL: Final[int] = 6
POS_BLOCK: Final[slice] = slice(0, 3)
ROT_BLOCK: Final[slice] = slice(3, 6)

# Pre-registered acceptance bar for ActionScale.check.
DEFAULT_MIN_R2: Final[float] = 0.8
# Control steps a command is allowed to act over: the pose change of one transition is regressed
# on the command that opened it and the ones before it, and the scale is the sum of those gains.
RESPONSE_TAPS: Final[int] = 2

# Not a fitted hyper-parameter: the simulator/demo saturation bound every chunk this module
# builds is clipped to (a free-fit run drifts it to ~24.5 chasing a target already clipped).
ACTION_LIMIT: Final[float] = 1.0

# Below this angle (rad) the exp/log trig quotients are evaluated by their Taylor series instead:
# sin(t)/t and (1-cos t)/t^2 both go 0/0, and t/sin(t/2) with them.
_SMALL_ANGLE: Final[float] = 1e-3


# ==== SO(3) ====
def _skew(vectors: np.ndarray) -> np.ndarray:
    """Cross-product matrices of a stack of vectors.

    Args:
        vectors: [..., 3] float64 vectors.

    Returns:
        [..., 3, 3] float64 skew-symmetric matrices.
    """
    zero = np.zeros(vectors.shape[:-1], dtype=np.float64)
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    rows = [
        np.stack([zero, -z, y], axis=-1),
        np.stack([z, zero, -x], axis=-1),
        np.stack([-y, x, zero], axis=-1),
    ]
    return np.stack(rows, axis=-2)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map, vectorised over leading axes.

    Args:
        rotvec: [..., 3] axis-angle rotation vectors, radians.

    Returns:
        [..., 3, 3] float64 rotation matrices.

    Raises:
        ValueError: rotvec's last axis is not length 3.
    """
    r = np.asarray(rotvec, dtype=np.float64)
    if r.shape[-1] != 3:
        raise ValueError(f"rotvec must have last axis 3, got shape {r.shape}")
    theta = np.linalg.norm(r, axis=-1)
    theta2 = theta * theta
    small = theta < _SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    sin_over = np.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, np.sin(safe) / safe)
    cos_over = np.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - np.cos(safe)) / (safe * safe))
    k = _skew(r)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), k.shape)
    return eye + sin_over[..., None, None] * k + cos_over[..., None, None] * (k @ k)


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Shepperd's rotation-matrix to quaternion conversion.

    Args:
        matrix: [N, 3, 3] float64 rotation matrices.

    Returns:
        [N, 4] float64 unit quaternions ordered [x, y, z, w].
    """
    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    # Branch on the largest diagonal entry (or the trace) to stay conditioned near theta=pi,
    # where 1 + trace goes to 0.
    decision = np.stack([matrix[:, 0, 0], matrix[:, 1, 1], matrix[:, 2, 2], trace], axis=-1)
    choice = np.argmax(decision, axis=-1)
    quat = np.empty((matrix.shape[0], 4), dtype=np.float64)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = choice == i
        block = matrix[sel]
        quat[sel, i] = 1.0 - trace[sel] + 2.0 * block[:, i, i]
        quat[sel, j] = block[:, j, i] + block[:, i, j]
        quat[sel, k] = block[:, k, i] + block[:, i, k]
        quat[sel, 3] = block[:, k, j] - block[:, j, k]
    sel = choice == 3
    block = matrix[sel]
    quat[sel, 0] = block[:, 2, 1] - block[:, 1, 2]
    quat[sel, 1] = block[:, 0, 2] - block[:, 2, 0]
    quat[sel, 2] = block[:, 1, 0] - block[:, 0, 1]
    quat[sel, 3] = 1.0 + trace[sel]
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Rodrigues logarithmic map, vectorised over leading axes.

    Args:
        matrix: [..., 3, 3] rotation matrices.

    Returns:
        [..., 3] float64 axis-angle rotation vectors with angle in [0, pi].

    Raises:
        ValueError: matrix's trailing axes are not (3, 3).
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have trailing axes (3, 3), got shape {m.shape}")
    quat = _matrix_to_quat(m.reshape(-1, 3, 3))
    quat = np.where(quat[:, 3:4] < 0.0, -quat, quat)
    angle = 2.0 * np.arctan2(np.linalg.norm(quat[:, :3], axis=-1), quat[:, 3])
    angle2 = angle * angle
    small = angle < _SMALL_ANGLE
    safe = np.where(small, 1.0, angle)
    scale = np.where(small, 2.0 + angle2 / 12.0 + 7.0 * angle2 * angle2 / 2880.0, safe / np.sin(safe / 2.0))
    return (scale[:, None] * quat[:, :3]).reshape(m.shape[:-2] + (3,))


def relative_rotvec(r_to: np.ndarray, r_from: np.ndarray) -> np.ndarray:
    """World-frame rotation from one orientation to another, as a rotation vector.

    Args:
        r_to: [..., 3] destination orientations, axis-angle.
        r_from: [..., 3] source orientations, axis-angle, broadcastable against r_to.

    Returns:
        [..., 3] float64 logmap(R_to @ R_from^T), the delta dims 3:6 of an action command.
    """
    to_mat = rotvec_to_matrix(r_to)
    from_mat = rotvec_to_matrix(r_from)
    return matrix_to_rotvec(to_mat @ np.swapaxes(from_mat, -1, -2))


def relative_rotation(r_to: np.ndarray, r_from: np.ndarray, pose_space: str = SO3) -> np.ndarray:
    """The dims 3:6 delta from one pose to another under the corpus's pose space.

    Args:
        r_to: [..., 3] destination dims 3:6.
        r_from: [..., 3] source dims 3:6, broadcastable against r_to.
        pose_space: SO3 composes orientations; EUCLIDEAN subtracts coordinates.

    Returns:
        [..., 3] float64 delta.

    Raises:
        ValueError: An unknown pose space.
    """
    if pose_space == SO3:
        return relative_rotvec(r_to, r_from)
    if pose_space == EUCLIDEAN:
        return np.asarray(r_to, dtype=np.float64) - np.asarray(r_from, dtype=np.float64)
    raise ValueError(f"unknown pose space {pose_space!r}; known: {POSE_SPACES}")


def pose_space_for(suite: str) -> str:
    """The pose space configs/trees/<suite>.yaml declares (pose_space), SO3 when it says nothing."""
    from robot_ood.core.paths import suite_config

    value = str(suite_config(suite).get("pose_space") or SO3)
    if value not in POSE_SPACES:
        raise ValueError(f"configs/trees/{suite}.yaml: pose_space must be one of {POSE_SPACES}, got {value!r}")
    return value


# ==== Action-unit calibration ====
@dataclass(frozen=True, slots=True)
class DemoDeltas:
    """Per-transition pose changes paired with the commands that could have produced them.

    Attributes:
        action_pos: [M, RESPONSE_TAPS, 3] f64 action dims 0:3; tap k is the command k steps
            before the transition.
        action_rot: [M, RESPONSE_TAPS, 3] f64 action dims 3:6, laid out the same way.
        delta_pos: [M, 3] f64 world-frame position change, metres.
        delta_rot: [M, 3] f64 world-frame relative rotation vector, radians.
    """

    action_pos: np.ndarray
    action_rot: np.ndarray
    delta_pos: np.ndarray
    delta_rot: np.ndarray

    def __len__(self) -> int:
        return int(self.action_pos.shape[0])

    @classmethod
    def from_demo(cls, ee_pos: np.ndarray, ee_ori: np.ndarray, actions: np.ndarray,
                  pose_space: str = SO3) -> "DemoDeltas":
        """Difference one demonstration into transitions.

        Args:
            ee_pos: [T, 3] end-effector positions, metres.
            ee_ori: [T, 3] end-effector orientations, axis-angle radians (or, under EUCLIDEAN,
                three more coordinates).
            actions: [T, 7] unnormalized action commands.
            pose_space: How ee_ori is differenced; see relative_rotation.

        Returns:
            The T - RESPONSE_TAPS transitions that have a full tap history, where the change from
            t to t+1 is paired with action[t], action[t-1], ... action[t-RESPONSE_TAPS+1].

        Raises:
            ValueError: The three arrays disagree on T, T <= RESPONSE_TAPS, or a shape is wrong.
        """
        pos = np.asarray(ee_pos, dtype=np.float64)
        ori = np.asarray(ee_ori, dtype=np.float64)
        act = np.asarray(actions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3 or ori.shape != pos.shape:
            raise ValueError(f"ee_pos and ee_ori must both be [T, 3], got {pos.shape} and {ori.shape}")
        if act.ndim != 2 or act.shape[1] < POSE_DIM:
            raise ValueError(f"actions must be [T, >=6], got {act.shape}")
        if act.shape[0] != pos.shape[0]:
            raise ValueError(f"actions has T={act.shape[0]} but poses have T={pos.shape[0]}")
        if pos.shape[0] <= RESPONSE_TAPS:
            raise ValueError(f"need T > {RESPONSE_TAPS} to difference, got T={pos.shape[0]}")
        first = RESPONSE_TAPS - 1
        taps = np.stack([act[first - k:act.shape[0] - 1 - k] for k in range(RESPONSE_TAPS)], axis=1)
        return cls(
            action_pos=taps[:, :, POS_BLOCK],
            action_rot=taps[:, :, ROT_BLOCK],
            delta_pos=pos[first + 1:] - pos[first:-1],
            delta_rot=relative_rotation(ori[first + 1:], ori[first:-1], pose_space),
        )

    @classmethod
    def concat(cls, parts: Sequence["DemoDeltas"]) -> "DemoDeltas":
        """Pool transitions across demonstrations.

        Args:
            parts: Per-demo transition sets. Concatenating the raw demos instead would
                manufacture one bogus transition across every demo boundary.

        Returns:
            The pooled transitions.

        Raises:
            ValueError: parts is empty.
        """
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of DemoDeltas")
        return cls(
            action_pos=np.concatenate([p.action_pos for p in parts]),
            action_rot=np.concatenate([p.action_rot for p in parts]),
            delta_pos=np.concatenate([p.delta_pos for p in parts]),
            delta_rot=np.concatenate([p.delta_rot for p in parts]),
        )


def _solve_response(action: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares gains of delta on the command taps through the origin, and the R2.

    Args:
        action: [N, RESPONSE_TAPS] commanded values.
        delta: [N] measured values.

    Returns:
        (gains, r2). gains is all zero when the commands are identically zero; r2 is 0.0 when
        the measurements have no variance to explain.
    """
    x = np.asarray(action, dtype=np.float64).reshape(-1, RESPONSE_TAPS)
    y = np.asarray(delta, dtype=np.float64).ravel()
    if not np.any(x):
        gains = np.zeros(RESPONSE_TAPS)
    else:
        gains = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = float(((y - x @ gains) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    return gains, (1.0 - residual / total) if total > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class BlockFit:
    """Regression of one action block's measured delta onto its command taps.

    Attributes:
        scale: Metres (position) or radians (rotation) per action unit: the DC gain of the
            response, fitted over all 3 axes.
        r2: Coefficient of determination of the pooled fit.
        gains: Per-tap gain of the pooled fit, summing to scale.
        axis_scale: Per-axis DC gain, for diagnostics.
        axis_r2: Per-axis R2, for diagnostics.
        n: Number of transitions the fit used.
    """

    scale: float
    r2: float
    gains: tuple[float, ...]
    axis_scale: tuple[float, float, float]
    axis_r2: tuple[float, float, float]
    n: int

    @classmethod
    def solve(cls, action: np.ndarray, delta: np.ndarray) -> "BlockFit":
        """Fit one block.

        Args:
            action: [M, RESPONSE_TAPS, 3] commanded values.
            delta: [M, 3] measured values.

        Returns:
            The pooled and per-axis fit.
        """
        action = np.asarray(action, dtype=np.float64)
        delta = np.asarray(delta, dtype=np.float64)
        gains, r2 = _solve_response(action.transpose(0, 2, 1).reshape(-1, RESPONSE_TAPS), delta)
        per_axis = [_solve_response(action[:, :, i], delta[:, i]) for i in range(3)]
        return cls(
            scale=float(gains.sum()),
            r2=r2,
            gains=tuple(float(g) for g in gains),
            axis_scale=tuple(float(g.sum()) for g, _ in per_axis),
            axis_r2=tuple(v for _, v in per_axis),
            n=int(action.shape[0]),
        )


def action_limit(deltas: DemoDeltas) -> float:
    """The corpus's one-step action amplitude: the largest pose column's p99 of |command|.

    Args:
        deltas: Pooled transitions; tap 0 of each block is the command at the transition.

    Returns:
        A scalar in action units. A corpus whose commands are normalised to [-1, 1] measures
        about 1 (ACTION_LIMIT); a joint-step corpus measures its own typical step.
    """
    command = np.concatenate([deltas.action_pos[:, 0, :], deltas.action_rot[:, 0, :]], axis=1)
    return float(np.percentile(np.abs(command), 99, axis=0).max())


def fit_blocks(deltas: DemoDeltas) -> tuple[BlockFit, BlockFit]:
    """Fit the position and rotation blocks of a pooled transition set.

    Args:
        deltas: Pooled transitions.

    Returns:
        (position fit, rotation fit).
    """
    return (
        BlockFit.solve(deltas.action_pos, deltas.delta_pos),
        BlockFit.solve(deltas.action_rot, deltas.delta_rot),
    )


@dataclass(frozen=True, slots=True)
class ActionScale:
    """Metres/radians per action unit for the two pose blocks of a chunk.

    Attributes:
        pos: Metres per action unit for dims 0:3.
        rot: Radians per action unit for dims 3:6.
        r2_pos: R2 of the position fit.
        r2_rot: R2 of the rotation fit.
        pose_space: How dims 3:6 compose (SO3 or EUCLIDEAN); every consumer of the scale
            differences poses the way the fit did.
        limit: The corpus's own one-step amplitude in action units (action_limit), the bound a
            tracking chunk's row 0 saturates at; ACTION_LIMIT for a scale fitted before the field.
    """

    pos: float
    rot: float
    r2_pos: float
    r2_rot: float
    pose_space: str = SO3
    limit: float = ACTION_LIMIT

    @classmethod
    def from_blocks(cls, pos: BlockFit, rot: BlockFit, pose_space: str = SO3,
                    limit: float = ACTION_LIMIT) -> "ActionScale":
        """Assemble from two already-solved blocks.

        Args:
            pos: Position-block fit.
            rot: Rotation-block fit.
            pose_space: How dims 3:6 compose.
            limit: The one-step amplitude, from action_limit.

        Returns:
            The calibration.
        """
        return cls(pos=pos.scale, rot=rot.scale, r2_pos=pos.r2, r2_rot=rot.r2, pose_space=pose_space,
                   limit=limit)

    @classmethod
    def from_deltas(cls, deltas: DemoDeltas, pose_space: str = SO3) -> "ActionScale":
        """Fit from a pooled transition set.

        Args:
            deltas: Pooled transitions.

        Returns:
            The calibration.
        """
        return cls.from_blocks(*fit_blocks(deltas), pose_space=pose_space, limit=action_limit(deltas))

    @classmethod
    def fit(cls, ee_pos: Any, ee_ori: Any, actions: Any, pose_space: str = SO3) -> "ActionScale":
        """Fit from one demonstration or a sequence of them.

        Args:
            ee_pos: [T, 3] positions, or a sequence of such arrays, one per demo.
            ee_ori: [T, 3] axis-angle orientations, matching ee_pos.
            actions: [T, 7] action commands, matching ee_pos.

        Returns:
            The calibration, fitted over the transitions of every demo pooled.
        """
        if np.ndim(ee_pos[0]) == 2:
            parts = [DemoDeltas.from_demo(p, o, a, pose_space) for p, o, a in zip(ee_pos, ee_ori, actions)]
        else:
            parts = [DemoDeltas.from_demo(ee_pos, ee_ori, actions, pose_space)]
        return cls.from_deltas(DemoDeltas.concat(parts), pose_space)

    def check(self, min_r2: float = DEFAULT_MIN_R2) -> "ActionScale":
        """Enforce the pre-registered fit-quality bar.

        Args:
            min_r2: Lowest acceptable R2 for either block.

        Returns:
            self, so a caller can chain the check onto a fit.

        Raises:
            ValueError: Either block's R2 is below min_r2. The message names the block.
        """
        failed = [
            f"{name} block R2={r2:.4f} < {min_r2}"
            for name, r2 in (("position", self.r2_pos), ("rotation", self.r2_rot))
            if r2 < min_r2
        ]
        if failed:
            raise ValueError(
                "action-scale fit rejected: "
                + "; ".join(failed)
                + ". A rotation block fails this way when the delta is subtracted rather than "
                "composed in the world frame; refit before deploying."
            )
        return self

    def to_dict(self) -> dict[str, float]:
        """JSON-serialisable form.

        Returns:
            Dict of the four fields.
        """
        return {"pos": self.pos, "rot": self.rot, "r2_pos": self.r2_pos, "r2_rot": self.r2_rot,
                "pose_space": self.pose_space, "limit": self.limit}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionScale":
        """Rebuild from a mapping.

        Args:
            data: Mapping carrying pos, rot, r2_pos and r2_rot.

        Returns:
            The calibration.
        """
        return cls(
            pos=float(data["pos"]),
            rot=float(data["rot"]),
            r2_pos=float(data["r2_pos"]),
            r2_rot=float(data["r2_rot"]),
            pose_space=str(data.get("pose_space", SO3)),   # a scale fitted before the field: SO3
            limit=float(data.get("limit", ACTION_LIMIT)),
        )

    def to_json(self, path: str | os.PathLike[str]) -> Path:
        """Write the calibration, creating parent directories as needed.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return target

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "ActionScale":
        """Read a calibration written by to_json.

        Args:
            path: Source file.

        Returns:
            The calibration.
        """
        with open(path) as handle:
            return cls.from_dict(json.load(handle))
