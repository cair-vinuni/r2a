from __future__ import annotations

import os
from typing import Final

import numpy as np

from robot_ood.core.absolute import deltas_from_targets, targets_from_deltas
from robot_ood.core.action import ACTION_DIM, EUCLIDEAN, GRIP_COL
from robot_ood.core.config import SentinelConfig
from robot_ood.data.lerobot_vrh31 import ARMS, POLICY_ARM, POLICY_HAND, open_flag
from robot_ood.data.segments import graph_plan_rows
from robot_ood.runtime.sentinel import Sentinel

__all__ = ["ARM_DOF", "CHUNK_DIM", "POLICY_CHUNK", "SUITES", "Vrh31Corrector", "lane_hand",
           "lane_joints", "proxy_q"]

ARM_DOF: Final[int] = 6
CHUNK_DIM: Final[int] = 26
POLICY_CHUNK: Final[int] = 16
SUITES: Final[dict[str, str]] = {"left": "vrh31_bds_pick_left", "right": "vrh31_bds_pick_right"}


def lane_joints(rows: np.ndarray, arm: str) -> np.ndarray:
    """[..., 6] one arm's corpus joints out of 26-dim policy rows."""
    return np.asarray(rows, dtype=np.float64)[..., POLICY_ARM[arm]]


def lane_hand(rows: np.ndarray, arm: str) -> np.ndarray:
    """[..., 6] one hand's joints out of 26-dim policy rows."""
    return np.asarray(rows, dtype=np.float64)[..., POLICY_HAND[arm]]


def proxy_q(joints: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """[7] the corpus configuration: six arm joints and the open flag of the commanded hand."""
    q = np.empty(ARM_DOF + 1, dtype=np.float64)
    q[:ARM_DOF] = np.asarray(joints, dtype=np.float64).reshape(ARM_DOF)
    q[ARM_DOF] = float(open_flag(np.asarray(hand, dtype=np.float64).reshape(1, -1))[0])
    return q


class Vrh31Corrector:
    """One arm's Sentinel; call reset, observe once per executed row, correct once per chunk.

    Args:
        graph_dir: The lane's artifact directory.
        arm: left or right, naming the lane this corrector answers for.
        fwm_dir: The suite directory; default is the lane suite beside graph_dir's grandparent.
        policy_chunk: Rows the policy returns per query.
        device: Torch device for the graph, empty for the default.
        wid: Stable worker id, as the Sentinel keys its per-episode state by.

    Raises:
        ValueError: The arm is not a lane of this robot, or fwm_dir does not name its suite.
    """

    def __init__(self, graph_dir: str | os.PathLike, *, arm: str,
                 fwm_dir: str | os.PathLike | None = None,
                 policy_chunk: int = POLICY_CHUNK, device: str = "", wid: int = 0) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        suite = SUITES[arm]
        graph_dir = os.fspath(graph_dir)
        fwm = os.fspath(fwm_dir) if fwm_dir is not None else os.path.join(
            os.path.dirname(os.path.dirname(os.path.normpath(graph_dir))), suite)
        if os.path.basename(os.path.normpath(fwm)) != suite:
            raise ValueError(
                f"fwm_dir basename must be {suite!r}, got {os.path.basename(os.path.normpath(fwm))!r}")
        cfg = SentinelConfig(policy_chunk=int(policy_chunk),
                             plan_rows=graph_plan_rows(graph_dir, int(policy_chunk)),
                             graph_dir=graph_dir, graph_device=device)
        self.arm = arm
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

    def observe(self, joints: np.ndarray, hand: np.ndarray,
                action_row: np.ndarray | None = None) -> None:
        q = proxy_q(joints, hand)
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
            np.asarray(joints, dtype=np.float64).reshape(ARM_DOF), lane_joints(chunk, self.arm),
            EUCLIDEAN)
        rows[:, GRIP_COL] = 1.0 - 2.0 * open_flag(lane_hand(chunk, self.arm))
        return rows

    def correct(self, chunk: np.ndarray, state: np.ndarray, task: str = "") -> np.ndarray:
        """[K, 26] the chunk to execute; every column this lane does not own comes back untouched.

        Args:
            chunk: [K, 26] the policy's absolute joint targets.
            state: [26] the robot's measured configuration, in the same column order.
            task: The instruction, for the Sentinel's language gate.

        Returns:
            The corrected chunk.

        Raises:
            ValueError: The chunk is not [K, 26].
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != CHUNK_DIM:
            raise ValueError(f"chunk must be [K, {CHUNK_DIM}], got {chunk.shape}")
        state = np.asarray(state, dtype=np.float64).reshape(CHUNK_DIM)
        if self._seen < self.graph_win:
            return chunk
        joints, hand = lane_joints(state, self.arm), lane_hand(state, self.arm)
        q = proxy_q(joints, hand)
        grip = 1.0 - q[ARM_DOF]
        plan = self.sentinel.plan(q, grip, self._step - 1, wid=self.wid, task=task)
        if plan is None:
            return chunk

        k = min(self.plan_rows, len(chunk))
        ee_now = q[:ARM_DOF]
        a_policy = np.empty((k, ACTION_DIM), dtype=np.float64)
        a_policy[:, :ARM_DOF] = deltas_from_targets(ee_now, lane_joints(chunk[:k], self.arm), EUCLIDEAN)
        a_policy[:, GRIP_COL] = 1.0 - 2.0 * open_flag(lane_hand(chunk[:k], self.arm))

        a_exec = np.asarray(plan.apply(a_policy), dtype=np.float64)
        self.decision = dict(plan.decision)
        out = chunk.copy()
        out[np.ix_(np.arange(k), np.asarray(POLICY_ARM[self.arm]))] = targets_from_deltas(
            ee_now, a_exec[:, :ARM_DOF], EUCLIDEAN)
        return out
