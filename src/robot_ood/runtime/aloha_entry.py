"""ALOHA right-arm homing: the ScoreNet entry target, reached by joint interpolation, one chunk at a time."""

from __future__ import annotations

import os
from typing import Final

import numpy as np
import torch

from robot_ood.core.config import EntryConfig, RADIUS_TO_SIGMA
from robot_ood.models.entry.score import ScoreNet, tweedie_denoise
from robot_ood.models.entry.target import EntryTarget
from robot_ood.runtime.aloha_deploy import ARM_DOF, POLICY_CHUNK, proxy_q

__all__ = ["MAX_PASSES", "STEP_CAP_RAD", "AlohaHoming"]

STEP_CAP_RAD: Final[float] = EntryConfig().score_step_cap
MAX_PASSES: Final[int] = 3


class AlohaHoming:
    """Call reset once per episode, then chunk once per request until it returns None."""

    def __init__(self, graph_dir: str | os.PathLike, *, device: str = "",
                 step_cap: float = STEP_CAP_RAD) -> None:
        if not step_cap > 0.0:
            raise ValueError(f"step_cap must be positive, got {step_cap}")
        graph_dir = os.fspath(graph_dir)
        self.target = EntryTarget.from_graph_dir(graph_dir)
        self.score = ScoreNet.load(graph_dir, device or "cpu")
        self.cfg = EntryConfig.from_env(graph_dir)
        self.step_cap = float(step_cap)
        self.reset()

    def reset(self) -> None:
        self._q0: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self._level_hat = 0.0
        self._walk = np.zeros((0, ARM_DOF))
        self._passes = 0
        self._steps = 0
        self._done = False

    def entry_target(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """([7] q_hat, level_hat): the refined entry configuration and its stopping radius."""
        q_hat = torch.as_tensor(q, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(self.cfg.score_refine_steps):
                sigma = min(max(float(self.score.level_estimate(q_hat)) * RADIUS_TO_SIGMA,
                                self.cfg.score_sigma_lo), self.cfg.score_sigma_hi)
                score = self.score(q_hat, torch.as_tensor(sigma, dtype=torch.float32))
                q_hat = tweedie_denoise(score, q_hat, sigma)
            level_hat = float(self.score.level_estimate(q_hat))
        return q_hat.numpy().astype(np.float64), level_hat

    def path(self, now: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """[n, 6] evenly spaced joint rows from now to goal, each leg at most step_cap."""
        now = np.asarray(now, dtype=np.float64).reshape(ARM_DOF)
        goal = np.asarray(goal, dtype=np.float64).reshape(ARM_DOF)
        n = int(np.ceil(np.linalg.norm(goal - now) / self.step_cap))
        if n == 0:
            return np.zeros((0, ARM_DOF))
        return now + (goal - now)[None, :] * (np.arange(1, n + 1) / n)[:, None]

    def chunk(self, joints: np.ndarray, grip_cmd: float,
              rows: int = POLICY_CHUNK) -> np.ndarray | None:
        """[rows, 7] the next leg of the walk, held at its last row; None once home or out of passes."""
        if self._done:
            return None
        joints = np.asarray(joints, dtype=np.float64).reshape(ARM_DOF)
        if self._goal is None:
            self._q0 = proxy_q(joints, grip_cmd)
            q_hat, self._level_hat = self.entry_target(self._q0)
            self._goal = q_hat[:ARM_DOF]
        if len(self._walk) == 0:
            if self.residual(joints) <= self._level_hat or self._passes >= MAX_PASSES:
                self._done = True
                return None
            self._walk = self.path(joints, self._goal)
            self._passes += 1
        leg, self._walk = self._walk[:rows], self._walk[rows:]
        self._steps += len(leg)
        out = np.empty((rows, ARM_DOF + 1), dtype=np.float64)
        out[:len(leg), :ARM_DOF] = leg
        out[len(leg):, :ARM_DOF] = leg[-1]
        out[:, ARM_DOF] = float(grip_cmd)
        return out

    def residual(self, joints: np.ndarray) -> float:
        return float(np.linalg.norm(self._goal - np.asarray(joints, dtype=np.float64).reshape(ARM_DOF)))

    def report(self, joints: np.ndarray, grip_cmd: float) -> dict:
        """Where the walk stands, measured at the joints given."""
        q1 = proxy_q(joints, grip_cmd)
        residual = self.residual(joints)
        return {
            "steps": self._steps,
            "passes": self._passes,
            "arrived": residual <= self._level_hat,
            "delta": float(np.linalg.norm(self._goal - self._q0[:ARM_DOF])),
            "residual": residual,
            "level_hat": self._level_hat,
            "level0": self._level(self._q0),
            "level_now": self._level(q1),
            "err0": self.target.error(self._q0),
            "err1": self.target.error(q1),
        }

    def _level(self, q: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.score.level_estimate(torch.as_tensor(q, dtype=torch.float32)))

    def level(self, joints: np.ndarray, grip_cmd: float) -> float:
        return self._level(proxy_q(joints, grip_cmd))

    def error(self, joints: np.ndarray, grip_cmd: float) -> float:
        return self.target.error(proxy_q(joints, grip_cmd))

    def entry_grip(self) -> float:
        return float(self.target.q_entry[ARM_DOF])
