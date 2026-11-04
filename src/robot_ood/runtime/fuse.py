"""The correction arm of a check: retrieve candidates, hand them to the CorrectionAuthority.

Everything that loads comes off the RuntimeBundle passed in; this module opens nothing itself.
BlendPlan below is what a check hands back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from robot_ood.runtime.bundle import RuntimeBundle

if TYPE_CHECKING:                                          # pragma: no cover -- typing only
    from robot_ood.core.schemas import SegmentCandidates
    from robot_ood.models.fusion.authority import CorrectionAuthority


def plan_fusion(bundle: RuntimeBundle, *, q_win, a_win, grip: float, task: str,
                record_chunk: "Callable[[np.ndarray], None] | None", ee_now: np.ndarray,
                phase: float, gate: "np.ndarray | None" = None,
                task_p: "np.ndarray | None" = None,
                seg_prior: "np.ndarray | None" = None, stall: "dict | None" = None) -> BlendPlan:
    """Build the correction plan for one check.

    Args:
        bundle: The loaded side: the retriever, the authority's learners, the node pose table.
        q_win: This worker's already-filled query-history window, oldest row first.
        a_win: This worker's commanded-action window, same length and stride gating as q_win.
        grip: The grip_flag reading for this check.
        task: Task hint, threaded to the retriever.
        record_chunk: Receives the policy chunk this plan is applied to, so the next check can
            read it; None records nothing.
        ee_now: [6] this check's live base-frame end-effector pose, threaded into BlendPlan.ee and
            reused verbatim by CorrectionAuthority.apply, never recomputed a second way.
        phase: This check's phase estimate, threaded into both ctx.phase and the
            authority's query_phase -- the SAME estimate reaching both.
        gate: The locator's [N_GATE] view of this check for the trust head, or None.
        task_p: The belief's task marginal; retrieval tilts tier 1 with it and fusion reads how much
            of the shortlist sits on the believed task. None leaves both as they were.
        seg_prior: [S] log prior over the segment table for the shortlist cut, or None.
        stall: The worker's per-episode stall counter the authority updates and reads
            (CorrectionAuthority.apply, ctx["stall"]), or None for no stall push.

    Returns:
        The plan.
    """
    ctx = bundle.query_context(q_win=q_win, a_win=a_win, grip=grip, task=task, ee_now=ee_now,
                               phase=phase, task_p=task_p, seg_prior=seg_prior)
    cands = bundle.segment_retriever().retrieve(ctx)
    # Only when the locator handed a task marginal down: without one there is nothing to look up.
    cand_task = (None if task_p is None
                 else bundle.segment_task_ids()[np.asarray(cands.idx, dtype=np.int64)])
    return BlendPlan(
        authority=bundle.authority(),
        cands=cands,
        ee=ee_now,
        authority_ctx={"query_phase": phase, "gate": gate,
                       "cand_task_p": None if cand_task is None else np.asarray(task_p)[cand_task],
                       "cand_task_id": cand_task, "stall": stall},
        record_chunk=record_chunk,
    )


@dataclass(frozen=True, slots=True)
class BlendPlan:
    """The authority and its candidates for one check, applied to the next policy chunk.

    Attributes:
        authority: The trained CorrectionAuthority.
        cands: This check's K retrieved candidates.
        ee: [6] this check's live base-frame end-effector pose.
        authority_ctx: Extra keyword arguments CorrectionAuthority.apply reads (query_phase,
            required; conf, optional).
        record_chunk: Receives the policy chunk this plan is applied to, or None.
        decision: What apply last decided, as trust, the mean shape, the authority and the check
            features (z_chk) the trust head read; empty until it has run.
    """

    authority: "CorrectionAuthority"
    cands: "SegmentCandidates"
    ee: np.ndarray
    authority_ctx: Mapping[str, Any] = field(default_factory=dict)
    record_chunk: "Callable[[np.ndarray], None] | None" = None
    decision: dict[str, Any] = field(default_factory=dict)

    def apply(self, a_policy: np.ndarray) -> np.ndarray:
        """Mix the correction into a policy chunk, gripper untouched.

        Args:
            a_policy: [K, 7] policy chunk.

        Returns:
            [K, 7] the chunk to execute.
        """
        a_policy = np.asarray(a_policy)
        if self.record_chunk is not None:
            self.record_chunk(a_policy)
        out = self.authority.apply(a_policy, self.cands, self.ee, **self.authority_ctx)
        self.decision.update(trust=float(out.trust), shape=float(np.mean(out.shape)),
                             authority=float(out.authority), z_chk=np.asarray(out.z_chk, dtype=np.float64))
        return out.a_exec
