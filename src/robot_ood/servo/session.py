"""One waist servo episode: a few reads of the head camera, each at a waist yaw the robot has turned to.

Waist yaw is the waist_yaw_joint position in rad, snapped to a 0.01 grid. The search reads the view where
the waist is and probes both sides. While the best read is the outermost on its side it steps further out;
once reads bracket it, it jumps to the vertex of a parabola through the best read and its two neighbours. It
stops at the first vertex read under tau, when the vertex lands on a pose already read, or after MAX_READS.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from robot_ood.servo.limits import WAIST_RANGE
from robot_ood.servo.prior import ServoPrior

Embed = Callable[[list[np.ndarray], str], np.ndarray]

PROBE: float = 0.25        # rad, first probe on each side of the start
MAX_JUMP: float = 0.30     # rad, furthest a vertex may sit from the best read
RESOLUTION: float = 0.01
MAX_READS: int = 8


def vertex(u: np.ndarray, e: np.ndarray) -> float | None:
    """Minimiser of the parabola through three (u, E) reads, or None when they do not bend upwards."""
    a, b, _ = np.polyfit(u, e, 2)
    return None if a <= 0 else float(-b / (2 * a))


class WaistSearch:
    """next_target() -> waist command to read next, or None when done; add(u, E) records a read there."""

    def __init__(self, tau: float, start: float = 0.0) -> None:
        self.tau, self.start = tau, start
        self.u: list[float] = []
        self.e: list[float] = []
        self.stop: str | None = None
        self.pending: float | None = start
        self.at_vertex = False

    def _snap(self, u: float) -> float:
        return float(np.clip(np.round(u / RESOLUTION) * RESOLUTION, -WAIST_RANGE, WAIST_RANGE))

    def best(self) -> float:
        return self.u[int(np.argmin(self.e))]

    def add(self, u: float, energy: float) -> None:
        self.u.append(u)
        self.e.append(energy)
        if len(self.u) == 1 and energy <= self.tau:
            self.stop = "already nominal"
        elif self.at_vertex and energy <= self.tau:
            self.stop = "certified"
        elif len(self.u) >= MAX_READS:
            self.stop = "read budget"
        self.pending = None if self.stop else self._next()
        if self.pending is not None and min(abs(self.pending - x) for x in self.u) < RESOLUTION / 2:
            self.stop, self.pending = "converged", None

    def _next(self) -> float:
        self.at_vertex = False
        if len(self.u) == 1:
            return self._snap(self.start + PROBE)
        if len(self.u) == 2:
            return self._snap(self.start - PROBE)
        u, e = np.array(self.u), np.array(self.e)
        best = self.best()
        if best == u.max() or best == u.min():
            return self._snap(best + (PROBE if best == u.max() else -PROBE))
        left, right = u[u < best].max(), u[u > best].min()
        top = vertex(np.array([left, best, right]), np.array([e[u == x].min() for x in (left, best, right)]))
        lo, hi = max(left, best - MAX_JUMP), min(right, best + MAX_JUMP)
        self.at_vertex = True
        return self._snap(np.clip(best if top is None else top, lo, hi))

    def next_target(self) -> float | None:
        return self.pending

    def final(self) -> float:
        return self.best()


class ServoSession:
    """reset(start) returns the first target; step(image) reads the view there and returns the next, until done.

    Args:
        prior: The nominal start model; its single task is the instruction the features are read under.
        embed: (images, task) -> [B, F] features.
    """

    def __init__(self, prior: ServoPrior, embed: Embed) -> None:
        self.prior, self.embed = prior, embed
        self.search: WaistSearch | None = None

    def _reply(self) -> dict:
        s = self.search
        reply = {"done": s.next_target() is None, "reads": len(s.u), "tau": self.prior.tau,
                 "energy": s.e[-1] if s.e else None,
                 "best": s.best() if s.u else None, "best_energy": min(s.e) if s.e else None}
        if reply["done"]:
            reply.update(waist_yaw=s.final(), stop=s.stop, energy0=s.e[0])
        else:
            reply.update(waist_yaw=s.next_target())
        return reply

    def reset(self, start: float = 0.0) -> dict:
        self.search = WaistSearch(self.prior.tau, float(start))
        return self._reply()

    def step(self, image: np.ndarray) -> dict:
        if self.search is None or self.search.next_target() is None:
            raise RuntimeError("call reset() first")
        z = self.embed([np.asarray(image)], self.prior.tasks[0])
        energy = float(self.prior.energy(z, 0)[0])
        self.search.add(self.search.next_target(), energy)
        return self._reply()
