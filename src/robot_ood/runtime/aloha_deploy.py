"""Sentinel on a real ALOHA right arm: a chunk of absolute joint targets in, the corrected chunk out."""

from __future__ import annotations

import os
from typing import Final

import numpy as np

from robot_ood.core.absolute import deltas_from_targets, targets_from_deltas
from robot_ood.core.action import ACTION_DIM, EUCLIDEAN, GRIP_COL
from robot_ood.core.config import SentinelConfig
from robot_ood.data.lerobot_aloha import open_flag
from robot_ood.data.segments import graph_plan_rows
from robot_ood.runtime.sentinel import Sentinel

__all__ = ["ARM_DOF", "POLICY_CHUNK", "SUITE", "AlohaCorrector", "proxy_q"]

ARM_DOF: Final[int] = 6
SUITE: Final[str] = "aloha_right"
POLICY_CHUNK: Final[int] = 16


def proxy_q(joints: np.ndarray, grip_cmd: float) -> np.ndarray:
    """[7] the corpus configuration: six joints and the open flag of the commanded gripper."""
    q = np.empty(ARM_DOF + 1, dtype=np.float64)
    q[:ARM_DOF] = np.asarray(joints, dtype=np.float64).reshape(ARM_DOF)
    q[ARM_DOF] = float(open_flag(np.asarray([grip_cmd], dtype=np.float64))[0])
    return q


class AlohaCorrector:
    """One arm's Sentinel; call reset, observe once per executed row, correct once per chunk."""

    def __init__(self, graph_dir: str | os.PathLike, *, fwm_dir: str | os.PathLike | None = None,
                 policy_chunk: int = POLICY_CHUNK, device: str = "", wid: int = 0) -> None:
        graph_dir = os.fspath(graph_dir)
        fwm = os.fspath(fwm_dir) if fwm_dir is not None else os.path.join(
            os.path.dirname(os.path.dirname(os.path.normpath(graph_dir))), SUITE)
        if os.path.basename(os.path.normpath(fwm)) != SUITE:
            raise ValueError(f"fwm_dir basename must be {SUITE!r}, got {os.path.basename(os.path.normpath(fwm))!r}")
        cfg = SentinelConfig(policy_chunk=int(policy_chunk),
                             plan_rows=graph_plan_rows(graph_dir, int(policy_chunk)),
                             graph_dir=graph_dir, graph_device=device)
        self.sentinel = Sentinel(cfg, fwm)
        self.plan_rows = cfg.plan_rows
        self.policy_chunk = cfg.policy_chunk
        self.graph_win = cfg.graph_win
        self.wid = int(wid)
        self._step = 0
        self._seen = 0
        self.decision: dict[str, float] = {}

    def reset(self) -> None:
        self.sentinel.reset(self.wid)
        self._step = 0
        self._seen = 0
        self.decision = {}

    def observe(self, joints: np.ndarray, grip_cmd: float,
                action_row: np.ndarray | None = None) -> None:
        q = proxy_q(joints, grip_cmd)
        self.sentinel.observe(q, self._step, wid=self.wid)
        if action_row is not None:
            self.sentinel.observe_action(
                np.asarray(action_row, dtype=np.float32).reshape(ACTION_DIM), self._step,
                wid=self.wid)
        self._step += 1
        self._seen += 1

    def action_rows(self, chunk: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """[K, 7] the chunk as corpus action rows: per-step joint deltas, gripper +1 closed / -1 open."""
        chunk = np.asarray(chunk, dtype=np.float64)
        rows = np.empty((len(chunk), ACTION_DIM), dtype=np.float32)
        rows[:, :ARM_DOF] = deltas_from_targets(
            np.asarray(joints, dtype=np.float64).reshape(ARM_DOF), chunk[:, :ARM_DOF], EUCLIDEAN)
        rows[:, GRIP_COL] = 1.0 - 2.0 * open_flag(chunk[:, GRIP_COL])
        return rows

    def correct(self, chunk: np.ndarray, joints: np.ndarray, grip_cmd: float,
                task: str = "") -> np.ndarray:
        """[K, 7] the chunk to execute; the gripper column comes back exactly as it arrived."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
            raise ValueError(f"chunk must be [K, {ACTION_DIM}], got {chunk.shape}")
        if self._seen < self.graph_win:
            return chunk
        q = proxy_q(joints, grip_cmd)
        grip = 1.0 - float(open_flag(np.asarray([grip_cmd], dtype=np.float64))[0])
        plan = self.sentinel.plan(q, grip, self._step - 1, wid=self.wid, task=task)
        if plan is None:
            return chunk

        k = min(self.plan_rows, len(chunk))
        ee_now = q[:ARM_DOF]
        a_policy = np.empty((k, ACTION_DIM), dtype=np.float64)
        a_policy[:, :ARM_DOF] = deltas_from_targets(ee_now, chunk[:k, :ARM_DOF], EUCLIDEAN)
        a_policy[:, GRIP_COL] = 1.0 - 2.0 * open_flag(chunk[:k, GRIP_COL])

        a_exec = np.asarray(plan.apply(a_policy), dtype=np.float64)
        self.decision = dict(plan.decision)
        out = chunk.copy()
        out[:k, :ARM_DOF] = targets_from_deltas(ee_now, a_exec[:, :ARM_DOF], EUCLIDEAN)
        return out
