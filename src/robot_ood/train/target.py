"""The one label rule: the physical band around the true node, restricted to its task when the instruction is known."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_ood.core.constants import (
    JOINT_TOL, PHASE_TOL, TRUE_NODE_BOOST, TRUE_TASK_BOOST, WEIGHT_SCALE,
)
from robot_ood.data.nodes import NodeTable

__all__ = ["Target", "TargetBuilder"]


@dataclass(frozen=True, slots=True)
class Target:
    """One row's label.

    Attributes:
        nodes: [K] int64 the target set.
        weight: [K] float32 target_weight, summing to 1.
    """

    nodes: np.ndarray
    weight: np.ndarray

    def __contains__(self, node: int) -> bool:
        return bool(np.any(self.nodes == int(node)))


class TargetBuilder:
    """make_target over one fixed node table."""

    def __init__(self, nodes: NodeTable, *, phase_tol: float = PHASE_TOL,
                 joint_tol: float = JOINT_TOL, weight_scale: float = WEIGHT_SCALE) -> None:
        self.phase_tol = float(phase_tol)
        self.joint_tol = float(joint_tol)
        self.weight_scale = float(weight_scale)
        self._task = np.asarray(nodes.task_id, dtype=np.int64)
        self._phase = np.asarray(nodes.phase, dtype=np.float64)
        self._q = np.asarray(nodes.q, dtype=np.float64)

    def __call__(self, true_node: int, has_instr: bool = True) -> Target:
        """The target set and weights around true_node.

        Args:
            true_node: The node the robot is at.
            has_instr: Whether the row carries an instruction; without one the band spans every task.

        Returns:
            The Target.
        """
        v = int(true_node)
        band = np.flatnonzero(np.abs(self._phase - self._phase[v]) <= self.phase_tol)
        dist = np.linalg.norm(self._q[band] - self._q[v][None, :], axis=1)
        band, dist = band[dist <= self.joint_tol], dist[dist <= self.joint_tol]
        same_task = self._task[band] == self._task[v]
        if has_instr:
            band, dist, same_task = band[same_task], dist[same_task], same_task[same_task]
        w = np.exp(-dist / self.weight_scale)
        w[band == v] *= TRUE_NODE_BOOST
        if not has_instr:
            w[same_task] *= TRUE_TASK_BOOST
        return Target(nodes=band.astype(np.int64), weight=(w / w.sum()).astype(np.float32))
