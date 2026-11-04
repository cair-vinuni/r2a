"""Absolute end-pose targets and the sequential deltas the graph, the corrector and a demo speak.

A LIBERO policy commands per-step deltas; a RoboTwin policy commands absolute targets, one per
step. The two meet here: a chunk of targets is the chain of deltas from the live configuration to
the first target and then target to target, and a chain of deltas integrates back to targets. Row
k of either form describes the same step, so a corrector trained on deltas can act on a chunk of
targets. Quaternions are [w, x, y, z], the order RoboTwin's transforms3d writes.
"""

from __future__ import annotations

import numpy as np

from robot_ood.core.action import EUCLIDEAN, POSE_DIM, SO3, matrix_to_rotvec, relative_rotation, rotvec_to_matrix

__all__ = ["deltas_from_targets", "quat_wxyz_to_rotvec", "rotvec_to_quat_wxyz", "targets_from_deltas"]


def quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """[..., 4] unit quaternions [w, x, y, z] -> [..., 3] rotation vectors."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix_to_rotvec(matrix)


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """[..., 3] rotation vectors -> [..., 4] unit quaternions [w, x, y, z], w >= 0."""
    rv = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-12, rv / np.where(angle > 0, angle, 1.0), np.array([1.0, 0.0, 0.0]))
    half = angle / 2.0
    quat = np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)
    return np.where(quat[..., :1] < 0, -quat, quat)


def deltas_from_targets(ee_now: np.ndarray, targets: np.ndarray, pose_space: str = SO3) -> np.ndarray:
    """The step-to-step deltas a chunk of absolute targets commands.

    Args:
        ee_now: [6] the live pose, position then rotation vector.
        targets: [K, 6] absolute target poses, one per step.
        pose_space: How dims 3:6 are differenced (robot_ood.core.action.relative_rotation).

    Returns:
        [K, 6] float64: row k is targets[k] relative to targets[k - 1] (row 0 relative to ee_now):
        the position difference and the world-frame relative rotation vector.
    """
    ee_now = np.asarray(ee_now, dtype=np.float64).reshape(POSE_DIM)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != POSE_DIM:
        raise ValueError(f"targets must be [K, {POSE_DIM}], got {targets.shape}")
    previous = np.concatenate([ee_now[None], targets[:-1]], axis=0)
    return np.concatenate([targets[:, :3] - previous[:, :3],
                           relative_rotation(targets[:, 3:], previous[:, 3:], pose_space)], axis=1)


def targets_from_deltas(ee_now: np.ndarray, deltas: np.ndarray, pose_space: str = SO3) -> np.ndarray:
    """Integrate step deltas from the live pose back into absolute targets; deltas_from_targets' inverse.

    Args:
        ee_now: [6] the live pose, position then rotation vector.
        deltas: [K, 6] per-step deltas as deltas_from_targets lays them out.
        pose_space: How dims 3:6 compose (robot_ood.core.action.relative_rotation).

    Returns:
        [K, 6] float64 absolute target poses.
    """
    ee_now = np.asarray(ee_now, dtype=np.float64).reshape(POSE_DIM)
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] != POSE_DIM:
        raise ValueError(f"deltas must be [K, {POSE_DIM}], got {deltas.shape}")
    out = np.empty_like(deltas)
    out[:, :3] = ee_now[:3] + np.cumsum(deltas[:, :3], axis=0)
    if pose_space == EUCLIDEAN:
        out[:, 3:] = ee_now[3:] + np.cumsum(deltas[:, 3:], axis=0)
        return out
    rotation = rotvec_to_matrix(ee_now[3:])
    for k in range(len(deltas)):
        rotation = rotvec_to_matrix(deltas[k, 3:]) @ rotation
        out[k, 3:] = matrix_to_rotvec(rotation)
    return out
