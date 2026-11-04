"""What the VR-H3.1 base and waist accept.

The walk numbers are the robot team's measurements of the patrol policy. The waist numbers are the
waist_yaw command block of vr_rl_lab's deploy.yaml: a joint position command in rad, rate limited in rad/s,
accepted only while the base stands still. Walk axes are (x, y, yaw) in m/s, m/s, rad/s, and the walk gain is
assumed constant between the deadband and the top of the range.
"""

from __future__ import annotations

from typing import Final

import numpy as np

WALK_CMD_LOWER: Final[np.ndarray] = np.array([-0.5, -0.5, -1.0])
WALK_CMD_UPPER: Final[np.ndarray] = np.array([1.2, 0.5, 1.0])
# Smallest command magnitude that keeps the base moving, per sign.
WALK_DEADBAND_POS: Final[np.ndarray] = np.array([0.06, 0.08, 0.14])
WALK_DEADBAND_NEG: Final[np.ndarray] = np.array([0.10, 0.08, 0.14])
# Measured at the top of the range: the command sent and the sustained velocity it gave.
_CMD_AT_MAX_POS: Final[np.ndarray] = np.array([1.18, 0.50, 1.06])
_CMD_AT_MAX_NEG: Final[np.ndarray] = np.array([0.52, 0.50, 1.06])
_VEL_AT_MAX_POS: Final[np.ndarray] = np.array([1.22, 0.40, 0.90])
_VEL_AT_MAX_NEG: Final[np.ndarray] = np.array([0.45, 0.39, 0.91])
WALK_GAIN_POS: Final[np.ndarray] = _VEL_AT_MAX_POS / _CMD_AT_MAX_POS
WALK_GAIN_NEG: Final[np.ndarray] = _VEL_AT_MAX_NEG / _CMD_AT_MAX_NEG

WAIST_RANGE: Final[float] = 1.57
WAIST_RATE: Final[float] = 0.5


def walk_velocity(command: np.ndarray) -> np.ndarray:
    """Sustained velocity under a walk command: clipped, zero below the deadband, scaled by the gain."""
    command = np.clip(command, WALK_CMD_LOWER, WALK_CMD_UPPER)
    positive = command > 0
    deadband = np.where(positive, WALK_DEADBAND_POS, WALK_DEADBAND_NEG)
    gain = np.where(positive, WALK_GAIN_POS, WALK_GAIN_NEG)
    return np.where(np.abs(command) < deadband, 0.0, command * gain)
