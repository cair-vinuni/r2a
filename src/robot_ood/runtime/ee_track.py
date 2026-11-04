"""The end-effector tracking chunk: a reference segment walked out as policy actions.

robot_ood.core.action owns the action layout, the SO(3) helpers and the metres-per-unit
calibration; this is only what a rollout does with them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_ood.core.action import POSE_DIM, POS_BLOCK, ROT_BLOCK, ActionScale, relative_rotation

__all__ = ["EeTrack"]

# ==== Tracking chunk ====
@dataclass(frozen=True, slots=True)
class EeTrack:
    """Builds an action-unit chunk that walks the end-effector along a reference segment.

    Attributes:
        scale: Calibration converting metres/radians into action units.
        bound: Saturation amplitude for row 0's action-unit pose error, or None for no
            saturation. Row 0 is otherwise unbounded; rows 1.. are demo-to-demo steps, already
            bounded by the demo's own dynamics, and are never saturated.
    """

    scale: ActionScale
    bound: float | None = None

    @staticmethod
    def predecessors(ee_ref: np.ndarray, ee_now: np.ndarray) -> np.ndarray:
        """The pose each row of the chunk is measured against.

        Args:
            ee_ref: [K, 6] reference segment, concat(position, axis-angle orientation).
            ee_now: [6] live end-effector pose.

        Returns:
            [K, 6] f64 predecessor poses: the live pose for row 0, ee_ref[k-1] for row k.
        """
        ref = np.asarray(ee_ref, dtype=np.float64)
        now = np.asarray(ee_now, dtype=np.float64)
        # Only row 0 closes the loop on the live pose; handing ee_now to every row would
        # command the same standing offset K times over (K-fold overshoot on an 8-step chunk).
        return np.concatenate([now[None, :], ref[:-1]], axis=0)

    def chunk(self, ee_ref: np.ndarray, ee_now: np.ndarray) -> np.ndarray:
        """One tracking chunk: feedback on row 0, demo feedforward after it.

        Args:
            ee_ref: [K, 6] reference segment, concat(position, axis-angle orientation).
            ee_now: [6] live end-effector pose.

        Returns:
            [K, 6] f64 action-unit deltas, in the layout of action dims 0:6.

        Raises:
            ValueError: ee_ref is not [K, 6] with K >= 1, or ee_now is not [6].
        """
        ref = np.asarray(ee_ref, dtype=np.float64)
        now = np.asarray(ee_now, dtype=np.float64)
        if ref.ndim != 2 or ref.shape[1] != POSE_DIM or ref.shape[0] < 1:
            raise ValueError(f"ee_ref must be [K, {POSE_DIM}] with K >= 1, got shape {ref.shape}")
        if now.shape != (POSE_DIM,):
            raise ValueError(f"ee_now must be [{POSE_DIM}], got shape {now.shape}")
        source = self.predecessors(ref, now)
        track = np.empty_like(ref)
        track[:, POS_BLOCK] = (ref[:, POS_BLOCK] - source[:, POS_BLOCK]) / self.scale.pos
        track[:, ROT_BLOCK] = relative_rotation(ref[:, ROT_BLOCK], source[:, ROT_BLOCK], self.scale.pose_space) / self.scale.rot
        if self.bound is not None:
            track[0] = self.bound * np.tanh(track[0] / self.bound)
        return track
