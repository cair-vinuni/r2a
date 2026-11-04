"""Star search for E(u) = |r(u)|^2 <= tau over the motion's axes, u = offset (forward, left, turn) in UNIT.

Per probe size and axis block: walk a star around the best pose, fit r ~ a + J (u - centre), line-search
along -J^+ a until E has a bracketed minimum, and repeat from the new best while E drops. Every reading is
checked against tau, so the search stops at the first pose that certifies, mid-star or mid-line.
The robot finishes at the lowest reading if it is clearly below the start, else back at the start.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any, Callable, Final

import numpy as np

from robot_ood.servo.motion import CONTROL_PERIOD, MOTIONS, UNIT, Motion, Placer
from robot_ood.servo.prior import PRIOR_JSON, PRIOR_NPZ, ServoPrior

PROBE_SIZES: Final[tuple[float, ...]] = (2.0, 1.0)     # UNIT
TIME_CAP: Final[float] = 60.0                          # s of robot time

FrameCallback = Callable[[np.ndarray], None]


class TimeCapReached(Exception):
    pass


class Certified(Exception):
    pass


class StarServo:
    """The t = 0 entry: turn (or walk) the robot until the policy's view is nominal, then hand over.

    Args:
        prior: The nominal start model.
        client: Has embed(observations, task) -> [B, F] features.
        motion: "waist" or "walk".
        placer: env -> callable placing the robot at an offset; defaults to the LIBERO base placer.
        settle: env -> observation after the move; defaults to one LIBERO hold step.
    """

    def __init__(self, prior: ServoPrior, client: Any, motion: str = "waist",
                 placer: Callable[[Any], Placer] | None = None, settle: Callable[[Any], dict] | None = None,
                 probe_sizes: tuple[float, ...] = PROBE_SIZES, time_cap: float = TIME_CAP) -> None:
        if placer is None or settle is None:
            from robot_ood.servo import libero
            placer, settle = placer or libero.BasePlacer, settle or libero.settle
        self.prior, self.client = prior, client
        self.motion_cls: type[Motion] = MOTIONS[motion]
        self.box = self.motion_cls.default_box()
        self.placer, self.settle = placer, settle
        self.probe_sizes, self.time_cap = probe_sizes, time_cap
        self.last: dict | None = None

    @classmethod
    def load(cls, prior_dir, host: str, port: int, motion: str = "waist") -> "StarServo":
        from robot_ood.servo.libero import FeatureClient

        prior_dir = pathlib.Path(prior_dir)
        meta = json.loads((prior_dir / PRIOR_JSON).read_text())
        return cls(ServoPrior.load(prior_dir / PRIOR_NPZ), FeatureClient(host, port, meta["feature"]), motion)

    def robot_time(self) -> float:
        return len(self.path) * CONTROL_PERIOD

    def best_pose(self) -> np.ndarray:
        return self.poses[int(np.argmin(self.energies))].copy()

    def advance(self, target: np.ndarray) -> dict:
        """One control tick towards target."""
        obs = self.motion.step(self.box.clip(target))
        self.path.append((self.motion.pose * UNIT).tolist())
        if self.frame_cb is not None:
            self.frame_cb(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        return obs

    def read(self, obs: dict) -> None:
        features = self.client.embed([obs], self.task)
        residual = self.prior.residual(features, self.task_index)[0].astype(np.float64)
        self.poses.append(self.motion.pose.copy())
        self.residuals.append(residual)
        self.energies.append(float(residual @ residual))
        if self.searching and self.energies[-1] <= self.prior.tau:
            raise Certified

    def step_and_read(self, target: np.ndarray, read: bool = True) -> None:
        if read and self.robot_time() >= self.time_cap:
            raise TimeCapReached
        obs = self.advance(target)
        if read:
            self.read(obs)

    def move_to(self, target: np.ndarray, stop: bool = True,
                stop_early: Callable[[], bool] | None = None) -> None:
        """Drive straight to target; stop=False passes through it without stopping."""
        target = self.box.clip(target)
        while not (self.motion.stopped_at(target) or (not stop and self.motion.near(target))):
            self.step_and_read(target)
            if stop_early is not None and stop_early():
                return

    def block_centre(self, block: tuple[int, ...], probe_size: float) -> np.ndarray:
        """Best pose, moved on the block's axes only so that the whole star stays inside the box."""
        best = self.best_pose()
        centre = best.copy()
        centre[list(block)] = self.box.star_centre(best, probe_size)[list(block)]
        return centre

    def walk_star(self, centre: np.ndarray, probe_size: float,
                  block: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Visit centre +- probe_size on each block axis; fit r ~ a + J (u - centre). Returns (J, a)."""
        self.move_to(centre, stop=False)
        first = len(self.energies)
        for side in (1.0, -1.0):
            for axis in block:
                self.move_to(centre + side * probe_size * np.eye(3)[axis], stop=False)
        offsets = (np.array(self.poses[first:]) - centre)[:, list(block)]
        design = np.hstack([offsets, np.ones((len(offsets), 1))])
        coefs = np.linalg.lstsq(design, np.array(self.residuals[first:]), rcond=None)[0]
        return coefs[:-1].T, coefs[-1]

    def line_search(self, start: np.ndarray, direction: np.ndarray) -> None:
        """Drive along direction until E beat the best so far by more than noise and then rose by more than noise."""
        self.move_to(start, stop=False)
        best_so_far, first = min(self.energies), len(self.energies)
        end = start + self.box.distance_to_edge(start, direction) * direction
        self.move_to(end, stop_early=lambda: self.bracketed_minimum(best_so_far, first))

    def bracketed_minimum(self, best_so_far: float, first: int) -> bool:
        lowest = min(self.energies[first:])
        return lowest < best_so_far - self.prior.energy_std and self.energies[-1] > lowest + self.prior.energy_std

    def park_at(self, pose: np.ndarray) -> None:
        while not self.motion.stopped_at(pose):
            self.step_and_read(pose, read=False)

    def search_block(self, block: tuple[int, ...], probe_size: float) -> None:
        """Walk a star, then line-search along the block's Gauss-Newton direction while that keeps lowering E."""
        start = self.block_centre(block, probe_size)
        jacobian, residual = self.walk_star(start, probe_size, block)
        while True:
            step = np.zeros(3)
            step[list(block)] = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            best_before = min(self.energies)
            self.line_search(start, step / (np.linalg.norm(step) + 1e-12))
            if min(self.energies) > best_before - self.prior.energy_std or min(self.energies) <= self.prior.tau:
                return
            start = self.best_pose()
            residual = self.residuals[int(np.argmin(self.energies))]

    def search(self) -> str:
        """Every probe size over every block until certified; returns why it stopped."""
        self.searching = True
        try:
            for probe_size in self.probe_sizes:
                for block in self.motion_cls.blocks:
                    self.search_block(block, probe_size)
                    self.trace.append({"t": round(self.robot_time(), 1), "probe": probe_size, "block": list(block),
                                       "best": np.round(self.best_pose() * UNIT, 3).tolist(),
                                       "E": round(min(self.energies), 3)})
        except Certified:
            return "certified"
        except TimeCapReached:
            return "time cap"
        finally:
            self.searching = False
        return "no probe sizes left"

    def run(self, env: Any, obs: dict, t: int, frame_cb: FrameCallback | None = None, *,
            task: str) -> tuple[dict, int]:
        wall_start = time.perf_counter()
        self.task, self.task_index, self.frame_cb = task, self.prior.tasks.index(task), frame_cb
        self.motion = self.motion_cls(self.placer(env), self.box)
        self.path, self.poses, self.residuals, self.energies, self.trace = [], [], [], [], []
        self.searching = False

        self.step_and_read(np.zeros(3))
        start_energy = self.energies[0]
        stop = "already nominal"
        if start_energy > self.prior.tau:
            stop = self.search()
            best = self.best_pose() if min(self.energies) < start_energy - self.prior.energy_std else np.zeros(3)
            self.park_at(best)
            obs = self.motion.hold(best)
        offset = self.motion.pose * UNIT
        if np.any(offset != 0):
            obs = self.settle(env)
            t += 1
        self.last = {"motion": self.motion_cls.__name__, "energy0": start_energy, "energy": min(self.energies),
                     "tau": self.prior.tau, "fired": bool(start_energy > self.prior.tau), "u": offset.tolist(),
                     "reads": len(self.energies), "robot_seconds": self.robot_time(), "path": self.path,
                     "energies": self.energies, "stop": stop, "trace": self.trace,
                     "seconds": time.perf_counter() - wall_start}
        logging.info("[SERVO %s] E %.3f -> %.3f (tau %.3f) offset=%s stop=%s robot %.1fs", self.motion_cls.__name__,
                     start_energy, min(self.energies), self.prior.tau, np.round(offset, 3).tolist(), stop,
                     self.robot_time())
        return obs, t
