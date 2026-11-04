"""The corrector behind the policy server: one call per chunk, nothing from the robot between calls."""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from robot_ood.runtime.aloha_deploy import ARM_DOF, AlohaCorrector
from robot_ood.runtime.aloha_entry import AlohaHoming

__all__ = ["AlohaServerLoop"]

ChunkSource = Callable[[], np.ndarray]


class AlohaServerLoop:
    """Homes the arm over the first requests, then corrects every policy chunk off the last chunk's targets."""

    def __init__(self, graph_dir: str | os.PathLike, *, homing: bool = True,
                 device: str = "") -> None:
        self.corrector = AlohaCorrector(graph_dir, device=device)
        self.homing = AlohaHoming(graph_dir, device=device) if homing else None
        self.reset()

    def reset(self) -> None:
        self.corrector.reset()
        if self.homing is not None:
            self.homing.reset()
        self._grip_cmd: float | None = None
        self._prev: tuple[np.ndarray, np.ndarray] | None = None
        self.homing_leg = False
        self.moved = 0.0

    @property
    def grip_cmd(self) -> float | None:
        return self._grip_cmd

    def step(self, joints: np.ndarray, grip_state: float, task: str,
             policy: ChunkSource) -> np.ndarray:
        """[K, 7] the chunk to execute; policy is only called once homing is over."""
        joints = np.asarray(joints, dtype=np.float64).reshape(ARM_DOF)
        if self._grip_cmd is None:
            self._grip_cmd = float(grip_state)
        if self.homing is not None:
            leg = self.homing.chunk(joints, self._grip_cmd, self.corrector.policy_chunk)
            self.homing_leg = leg is not None
            if leg is not None:
                return leg
        if self._prev is None:
            self.corrector.observe(joints, self._grip_cmd)
        else:
            q_prev, out_prev = self._prev
            rows = self.corrector.action_rows(out_prev, q_prev)
            q_path = np.vstack([q_prev[None, :], out_prev[:-1, :ARM_DOF]])
            for q_k, row, grip in zip(q_path, rows, out_prev[:, ARM_DOF]):
                self.corrector.observe(q_k, self._grip_cmd, row)
                self._grip_cmd = float(grip)
        chunk = np.asarray(policy(), dtype=np.float64)
        out = self.corrector.correct(chunk, joints, self._grip_cmd, task=task)
        self.moved = float(np.abs(out - chunk).max())
        self._prev = (joints, out)
        return out
