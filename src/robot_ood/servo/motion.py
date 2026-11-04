"""How the robot executes a servo waypoint. Two motions, chosen by name:

waist (default)  the legs stay planted and the waist yaw turns, as a rate-limited joint position command
walk             the legs walk under velocity commands (vx, vy, wz): clipped, scaled by the measured gain,
                 and held at the deadband when the wanted speed is below it, so the base creeps

Poses are offsets from where the episode found the robot, in UNIT = (5 cm, 5 cm, 0.1 rad), so both
weigh the same in the search. The control period is in s, the tracker gain in 1/s, the walk's braking in UNIT/s^2.
"""

from __future__ import annotations

from typing import Callable, Final

import numpy as np

from robot_ood.servo.limits import (WAIST_RANGE, WAIST_RATE, WALK_CMD_LOWER, WALK_CMD_UPPER, WALK_DEADBAND_NEG,
                                    WALK_DEADBAND_POS, WALK_GAIN_NEG, WALK_GAIN_POS, walk_velocity)

UNIT: Final[np.ndarray] = np.array([0.05, 0.05, 0.10])
CONTROL_PERIOD: Final[float] = 0.1
TRACKER_GAIN: Final[float] = 3.0
WALK_DECEL: Final[float] = 2.0

Placer = Callable[[np.ndarray], dict]


class SafeBox:
    """Allowed offsets from the start, per axis [lower, upper] in m, m, rad."""

    def __init__(self, lower=(-0.22, -0.18, -0.30), upper=(0.22, 0.18, 0.30)) -> None:
        self.lower = np.asarray(lower, float) / UNIT
        self.upper = np.asarray(upper, float) / UNIT

    def clip(self, pose: np.ndarray) -> np.ndarray:
        return np.clip(pose, self.lower, self.upper)

    def star_centre(self, pose: np.ndarray, probe_size: float) -> np.ndarray:
        """Nearest point to pose whose whole star stays inside the box."""
        lower, upper = self.lower + probe_size, self.upper - probe_size
        return np.where(lower <= upper, np.clip(pose, lower, upper), (self.lower + self.upper) / 2)

    def distance_to_edge(self, start: np.ndarray, direction: np.ndarray) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            steps = np.where(direction > 0, (self.upper - start) / direction,
                             np.where(direction < 0, (self.lower - start) / direction, np.inf))
        return float(max(steps.min(), 0.0))


class Motion:
    """One control tick at a time toward a target pose; place(offset in m, m, rad) returns what the camera sees."""

    blocks: tuple[tuple[int, ...], ...]

    def __init__(self, place: Placer, box: SafeBox) -> None:
        self.place, self.box = place, box
        self.pose, self.velocity = np.zeros(3), np.zeros(3)

    @staticmethod
    def default_box() -> SafeBox:
        raise NotImplementedError

    def step(self, target: np.ndarray) -> dict:
        raise NotImplementedError

    def stopped_at(self, target: np.ndarray) -> bool:
        return bool(np.abs(self.pose - target).max() < 0.01 and np.abs(self.velocity).max() < 0.02)

    def near(self, target: np.ndarray) -> bool:
        return bool(np.abs(self.pose - target).max() < 0.1)

    def hold(self, pose: np.ndarray) -> dict:
        """Stand at pose, at rest."""
        self.pose, self.velocity = pose.copy(), np.zeros(3)
        return self.place(self.pose * UNIT)


class WaistMotion(Motion):
    """Legs planted: x and y never move, the waist yaw follows its target at no more than WAIST_RATE."""

    blocks = ((2,),)

    @staticmethod
    def default_box() -> SafeBox:
        return SafeBox(lower=(0.0, 0.0, -WAIST_RANGE), upper=(0.0, 0.0, WAIST_RANGE))

    def step(self, target: np.ndarray) -> dict:
        rate = WAIST_RATE / UNIT[2]
        self.velocity = np.array([0.0, 0.0, np.clip(TRACKER_GAIN * (target[2] - self.pose[2]), -rate, rate)])
        self.pose = self.box.clip(self.pose + self.velocity * CONTROL_PERIOD)
        return self.place(self.pose * UNIT)


class WalkMotion(Motion):
    """Velocity commands: brake toward the target, creep at the deadband, stop inside the arrival tolerance."""

    blocks = ((0, 1, 2),)

    def __init__(self, place: Placer, box: SafeBox) -> None:
        super().__init__(place, box)
        # Half the largest minimum step of one control period: a creep never overshoots it.
        self.tolerance = 0.5 * np.maximum(WALK_DEADBAND_POS * WALK_GAIN_POS,
                                          WALK_DEADBAND_NEG * WALK_GAIN_NEG) * CONTROL_PERIOD / UNIT
        self.command = np.zeros(3)

    @staticmethod
    def default_box() -> SafeBox:
        # Turns as far as the waist reaches, so both motions cover the same headings.
        return SafeBox(lower=(-0.22, -0.18, -WAIST_RANGE), upper=(0.22, 0.18, WAIST_RANGE))

    def step(self, target: np.ndarray) -> dict:
        error = (target - self.pose) * UNIT
        speed = np.minimum(TRACKER_GAIN * np.abs(error), np.sqrt(2 * WALK_DECEL * UNIT * np.abs(error)))
        wanted = np.sign(error) * speed
        gain = np.where(wanted > 0, WALK_GAIN_POS, WALK_GAIN_NEG)
        deadband = np.where(wanted > 0, WALK_DEADBAND_POS, WALK_DEADBAND_NEG)
        command = np.clip(wanted / gain, WALK_CMD_LOWER, WALK_CMD_UPPER)
        command = np.where(np.abs(command) < deadband, np.sign(command) * deadband, command)
        self.command = np.where(np.abs(target - self.pose) <= self.tolerance, 0.0, command)
        self.velocity = walk_velocity(self.command) / UNIT
        self.pose = self.box.clip(self.pose + self.velocity * CONTROL_PERIOD)
        return self.place(self.pose * UNIT)

    def stopped_at(self, target: np.ndarray) -> bool:
        return bool(np.all(np.abs(self.pose - target) <= self.tolerance) and np.abs(self.velocity).max() < 0.02)


MOTIONS: Final[dict[str, type[Motion]]] = {"waist": WaistMotion, "walk": WalkMotion}
