"""One check's blend: the flow from a check's phase estimate to the plan for the next action chunk.

No file handle, no checkpoint read, no environment lookup -- it all arrives on the bundle.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from robot_ood.runtime.fuse import BlendPlan
from robot_ood.runtime.bundle import RuntimeBundle
from robot_ood.runtime.fuse import plan_fusion


def plan(bundle: RuntimeBundle, record_chunk: "Callable[[np.ndarray], None] | None", phase_hat: float,
         q_now: np.ndarray, *, q_win, a_win, grip: float, task: str,
         gate: "np.ndarray | None" = None, task_p: "np.ndarray | None" = None,
         seg_prior: "np.ndarray | None" = None, stall: "dict | None" = None) -> BlendPlan:
    """Turn this check's answer into the correction plan for the next action chunk.

    Args:
        bundle: The loaded side: the blend head, the kinematics, the action-unit calibration.
        record_chunk: Receives the policy chunk this plan is applied to, or None records nothing.
        phase_hat: This check's phase estimate; q_now: [D] arm config in radians this step.
        q_win: Query-history window, oldest row first; a_win: commanded actions, same gating.
        grip: The grip_flag reading for this step.
        task: Task hint, threaded to the retriever.
        gate: The locator's [N_GATE] view of this check, threaded to the trust head, or None.
        task_p: The belief's task marginal, threaded to retrieval and fusion, or None.
        seg_prior: [S] log prior over the segment table, threaded to retrieval, or None.
        stall: This worker's per-episode stall counter, threaded to the authority, or None.

    Returns:
        The plan.
    """
    ee_now = bundle.kinematics().pose(np.asarray(q_now, dtype=np.float64)[None, :])[0]
    return plan_fusion(bundle, q_win=q_win, a_win=a_win, grip=grip, task=task,
                       record_chunk=record_chunk, ee_now=ee_now, phase=phase_hat,
                       gate=gate, task_p=task_p, seg_prior=seg_prior, stall=stall)
