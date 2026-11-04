from __future__ import annotations

import os
from typing import Callable, Mapping

import numpy as np

from robot_ood.data.lerobot_vrh31 import ARMS, POLICY_ARM
from robot_ood.runtime.vrh31_deploy import (
    ARM_DOF,
    CHUNK_DIM,
    Vrh31Corrector,
    lane_hand,
    lane_joints,
)
from robot_ood.runtime.vrh31_entry import Vrh31Homing

__all__ = ["Vrh31ServerLoop"]

ChunkSource = Callable[[], np.ndarray]


class _Lane:
    """One arm's corrector, its homing walk and the previous chunk it has yet to observe."""

    def __init__(self, graph_dir: str | os.PathLike, arm: str, homing: bool, device: str) -> None:
        self.arm = arm
        self.columns = np.asarray(POLICY_ARM[arm])
        self.corrector = Vrh31Corrector(graph_dir, arm=arm, device=device)
        self.homing = Vrh31Homing(graph_dir, device=device) if homing else None
        self.reset()

    def reset(self) -> None:
        self.corrector.reset()
        if self.homing is not None:
            self.homing.reset()
        self.prev: tuple[np.ndarray, np.ndarray] | None = None
        self.homing_leg = False

    def observe_history(self, executed: np.ndarray, measured: np.ndarray) -> None:
        """Feed the Sentinel the rows the robot actually ran ([H, 26]) and the joints it measured
        at each of them ([H, 26]), as the client reports them."""
        self.corrector.reset()
        if len(executed) == 0:
            return
        q_meas = lane_joints(measured, self.arm)
        rows = self.corrector.action_rows(executed, q_meas[0])
        for q_k, row, hand in zip(q_meas, rows, lane_hand(executed, self.arm)):
            self.corrector.observe(q_k, hand, row)

    def observe(self, state: np.ndarray) -> None:
        """Feed the Sentinel what the arm did since the last request."""
        if self.prev is None:
            self.corrector.observe(lane_joints(state, self.arm), lane_hand(state, self.arm))
            return
        q_prev, out_prev = self.prev
        rows = self.corrector.action_rows(out_prev, q_prev)
        joints_prev = lane_joints(out_prev, self.arm)
        hands_prev = lane_hand(out_prev, self.arm)
        q_path = np.vstack([q_prev[None, :], joints_prev[:-1]])
        for q_k, row, hand in zip(q_path, rows, hands_prev):
            self.corrector.observe(q_k, hand, row)



class Vrh31ServerLoop:
    """Homes both arms over the first requests, then corrects every policy chunk lane by lane.

    Args:
        graph_dirs: One artifact directory per arm, keyed left and right.
        homing: Walk each arm to its entry target before the policy is called.
        device: Torch device for the graphs and ScoreNets.

    Raises:
        KeyError: graph_dirs does not name both arms.
    """

    def __init__(self, graph_dirs: Mapping[str, str | os.PathLike], *, homing: bool = True,
                 device: str = "") -> None:
        missing = [arm for arm in ARMS if arm not in graph_dirs]
        if missing:
            raise KeyError(f"graph_dirs must name every arm {ARMS}, missing {missing}")
        self.lanes = {arm: _Lane(graph_dirs[arm], arm, homing, device) for arm in ARMS}
        self.policy_chunk = self.lanes[ARMS[0]].corrector.policy_chunk
        self.reset()

    def reset(self) -> None:
        for lane in self.lanes.values():
            lane.reset()
        self.homing_leg = False
        self.moved = 0.0

    @property
    def decisions(self) -> dict[str, dict[str, float]]:
        """What each lane's Sentinel decided on the last corrected chunk."""
        return {arm: dict(lane.corrector.decision) for arm, lane in self.lanes.items()}

    def step(self, state: np.ndarray, task: str, policy: ChunkSource,
             history: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """[K, 26] the chunk to execute; policy is only called once both arms are home.

        Args:
            state: [26] the robot's measured configuration, in the policy's column order.
            task: The instruction, for the Sentinel's language gate.
            policy: Returns the policy's [K, 26] chunk of absolute joint targets.
            history: The client's "executed" [H, 26] and "measured" [H, 26] rows since the episode
                began; given, the server keeps nothing between requests.

        Returns:
            The chunk to execute.
        """
        state = np.asarray(state, dtype=np.float64).reshape(CHUNK_DIM)
        legs = {}
        for arm, lane in self.lanes.items():
            if lane.homing is None:
                continue
            leg = lane.homing.chunk(lane_joints(state, arm), lane_hand(state, arm),
                                    self.policy_chunk)
            lane.homing_leg = leg is not None
            if leg is not None:
                legs[arm] = leg
        self.homing_leg = bool(legs)
        if legs:
            out = np.repeat(state[None, :], self.policy_chunk, axis=0)
            for arm, leg in legs.items():
                out[:, self.lanes[arm].columns] = leg
            return out

        for lane in self.lanes.values():
            if history is None:
                lane.observe(state)
            else:
                lane.observe_history(np.asarray(history["executed"], dtype=np.float64).reshape(-1, CHUNK_DIM),
                                     np.asarray(history["measured"], dtype=np.float64).reshape(-1, CHUNK_DIM))
        chunk = np.asarray(policy(), dtype=np.float64)
        out = chunk
        for arm, lane in self.lanes.items():
            out = lane.corrector.correct(out, state, task=task)
        self.moved = float(np.abs(out - chunk).max())
        for arm, lane in self.lanes.items():
            lane.prev = (lane_joints(state, arm), out)
        return out
