"""CorrectionAuthority: the deploy boundary over the trained Corrector.

It owns no parameters; it turns one check's numpy inputs into the tensors
robot_ood.models.fusion.corrector.Corrector.forward reads, and hands back the chunk to execute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from robot_ood.core.action import ActionScale
from robot_ood.core.env import env_float
from robot_ood.core.schemas import SegmentCandidates
from robot_ood.core.action import POSE_DIM
from robot_ood.models.fusion.corrector import (SCORE_CLAMP, Corrector, corrector_features,
                                                resample_torch)
from robot_ood.models.fusion.target import node_pose_ref, track_chunk

__all__ = ["AuthorityOutput", "CorrectionAuthority", "STALL_CHECKS"]


# Stall push. A policy that hovers (every row of its chunk below STALL_EPS on the pose block) for
# STALL_CHECKS consecutive checks gets the retrieved demos' own displacement from the live pose
# added to its chunk, when the candidates agree on that displacement (their weighted cosine with
# the mean at least _STALL_COS) and it is a real motion (at least _STALL_MIN over the chunk). The
# demos' relative motion at a phase is scene-invariant where their absolute pose is not
# (scratchpad demo_consistency.py: cosine 0.99-1.0 across demos, pose scatter 0.4-0.5 rad).
# 0 disables the push.
STALL_CHECKS = int(env_float("ROBOT_OOD_STALL_PUSH", 0))
# ROBOT_OOD_PUSH_LEAN=1: no counter and no agreement gates; the candidates' displacement at the
# belief's phase is added in proportion to how little the robot moves, (1 - |a_policy| / eps).
_LEAN = int(env_float("ROBOT_OOD_PUSH_LEAN", 0)) == 1
# ROBOT_OOD_STALL_DWELL=1: the still time that counts as a stall is the longer of STALL_CHECKS
# checks and the demos' own pause at the belief's phase (rel_ref.dwell), and the displacement is
# read from past the whole pause. A robot is stalled when it has waited longer than the demos wait.
_DWELL = int(env_float("ROBOT_OOD_STALL_DWELL", 0)) == 1
_DWELL_MASS = 0.1           # a candidate's pause counts when it carries this much belief
_STALL_EPS = env_float("ROBOT_OOD_STALL_EPS", 0.003)
_STALL_GAIN = env_float("ROBOT_OOD_STALL_GAIN", 1.0)
_STALL_COS = 0.7
_STALL_MIN = 0.02
_STALL_PHASE_MIN = 0.1      # an arm that has not started its part of the task is the policy's call
_STALL_PHASE_TOL = 0.1      # the belief may move this much between checks and still name one phase
# Past this phase the demos have only the retreat left, so a still arm there is waiting on the
# task check, not stalled mid-motion. 1.0 pushes at every phase.
_STALL_PHASE_MAX = env_float("ROBOT_OOD_STALL_PHASE_MAX", 1.0)
# ROBOT_OOD_STALL_RETRY=n: an arm still where the demos rest (their mass at this phase below
# 0.5) at a late phase, in an episode that has not ended, has finished the motion without
# finishing the task. Back it off along the demos' own approach, read _RETRY_BACK earlier in
# phase, so the policy re-approaches from a fresh pose; at most n times per episode. 0 disables.
_RETRY = int(env_float("ROBOT_OOD_STALL_RETRY", 0))
_RETRY_BACK = env_float("ROBOT_OOD_STALL_RETRY_BACK", 0.4)   # phase to rewind, along the demos
_RETRY_PHASE = 0.85
_RETRY_STEP = 0.3           # pose units one check may cover while the arm backs off

# Deploy-side trust gate: trust below this floor blends nothing, and above it the blend gain is
# (trust - floor) / (1 - floor). 0 keeps the trained head as it is.
_TRUST_MIN = env_float("ROBOT_OOD_TRUST_MIN", 0.0)
# Phase ceiling for the running correction: past it the blend gain is zero. Retrieval predicts the
# policy's motion only near the start of an episode (audit_oracle.py), so 1 keeps everything.
_TRUST_PHASE_MAX = env_float("ROBOT_OOD_TRUST_PHASE_MAX", 1.0)


@dataclass(frozen=True, slots=True)
class AuthorityOutput:
    """One correction decision, without the batch axis.

    Attributes:
        a_exec: [C, ACTION_DIM] a_policy + (trust * authority * shape) * (a_ref - a_policy),
            gripper verbatim.
        trust: scalar d in (0, 1).
        shape: [C, 6] A in (0, 1).
        a_ref: [C, 6] the reference chunk the correction stepped toward.
        authority: scalar in [0, 1], the candidate mass that is not a pause.
        z_chk: [N_CHK_FEAT] the check features the trust head read, raw (before standardisation).
    """

    a_exec: np.ndarray
    trust: np.ndarray
    shape: np.ndarray
    a_ref: np.ndarray
    authority: np.ndarray
    z_chk: np.ndarray


@dataclass(frozen=True, eq=False)
class CorrectionAuthority:
    """The trained corrector plus what deploy needs to build its inputs.

    Attributes:
        corrector: The trained robot_ood.models.fusion.corrector.Corrector.
        node_ee: [V, 6] this graph's per-node end-effector pose table.
        scale: Metres/radians per action unit.
        pause: [S] bool, which segments are pauses (robot_ood.data.segments.segment_pause), or None
            for none.
        raw_ref: A robot_ood.models.fusion.target.RawPoseRef when ROBOT_OOD_REF=raw (the reference
            is the candidate's own raw frames, as training built it), else None for node_pose_ref.
        direct: Execute the fused reference at full gain, bypassing both trained heads.
        rel_ref: A relative RawPoseRef (the candidates' own displacements from the frame before
            their segment) for the stall push, or None for no push.
        cart: The arm's robot_ood.core.kinematics.Arx5Kinematics on a joint-space lane, so the
            reference follows the candidates only where they agree (see corrector_features), or
            None on a Cartesian lane.
    """

    corrector: Corrector
    node_ee: np.ndarray
    scale: ActionScale
    pause: "np.ndarray | None" = None
    raw_ref: Any = None
    direct: bool = False
    rel_ref: Any = None
    cart: Any = None

    def apply(self, a_policy: np.ndarray, cands: SegmentCandidates, ee: np.ndarray, **ctx: Any) -> AuthorityOutput:
        """One correction decision on one check's numpy inputs, under no_grad.

        Args:
            a_policy: [C, ACTION_DIM] policy chunk.
            cands: This check's K structured candidates.
            ee: [6] this check's live base-frame end-effector pose.
            **ctx: query_phase (required); gate, the locator's [N_GATE] view of this check, and
                cand_task_p / cand_task_id, the belief's mass on each candidate's task (optional);
                stall, the worker's per-episode counter dict this call updates, whose
                partner_still entry the harness sets when every other arm's chunk is still too
                (optional; absent means single-armed).

        Returns:
            AuthorityOutput of float64 numpy arrays; a_exec is the chunk to execute.

        Raises:
            KeyError: ctx carries no query_phase.
        """
        if "query_phase" not in ctx:
            raise KeyError(
                "CorrectionAuthority.apply needs ctx['query_phase'] -- the belief's own phase "
                "estimate this check's a_policy chunk starts from, threaded by the caller building "
                "the plan (see robot_ood.runtime.fuse.plan_fusion)."
            )
        with torch.no_grad():
            policy = torch.as_tensor(np.asarray(a_policy, dtype=np.float32))[None]
            actions = resample_torch(
                torch.as_tensor(np.asarray(cands.action_chunk, dtype=np.float32))[None], policy.shape[-2]
            )
            score = torch.as_tensor(np.asarray(cands.score, dtype=np.float32))[None]
            phase = torch.as_tensor(np.asarray(cands.phase, dtype=np.float32))[None]
            query_phase = torch.as_tensor(np.asarray(ctx["query_phase"], dtype=np.float32)).reshape(1)
            if self.raw_ref is not None:
                ee_ref_np = self.raw_ref(cands.start_node, policy.shape[-2])
            else:
                ee_ref_np = node_pose_ref(self.node_ee, cands.start_node, cands.end_node, policy.shape[-2])
            ee_ref = torch.as_tensor(ee_ref_np.astype(np.float32))[None]
            ee_now = torch.as_tensor(np.asarray(ee, dtype=np.float32))[None]
            cand_ee_start = torch.as_tensor(np.asarray(cands.ee_start, dtype=np.float32))[None]
            cand_ee_end = torch.as_tensor(np.asarray(cands.ee_end, dtype=np.float32))[None]
            gate = ctx.get("gate")
            gate_t = None if gate is None else torch.as_tensor(np.asarray(gate, dtype=np.float32))[None]
            task_p, task_id = ctx.get("cand_task_p"), ctx.get("cand_task_id")
            task_p_t = None if task_p is None else torch.as_tensor(np.asarray(task_p, dtype=np.float32))[None]
            task_id_t = None if task_id is None else torch.as_tensor(np.asarray(task_id, dtype=np.int64))[None]
            pause_t = None if self.pause is None else torch.as_tensor(
                np.asarray(self.pause, dtype=bool)[np.asarray(cands.idx, dtype=np.int64)])[None]
            inputs = (policy, actions, score, phase, query_phase, ee_ref, ee_now, self.scale,
                      cand_ee_start, cand_ee_end, gate_t, task_p_t, task_id_t, pause_t, self.cart)
            if self.direct:
                feats = corrector_features(*inputs)
                trust, shape = torch.ones(1), torch.ones_like(feats.a_ref)
                a_exec = self.corrector.blend(policy, feats.a_ref, feats.authority[:, None, None] * shape)
            else:
                out = self.corrector(*inputs)
                feats, trust, shape, a_exec = out.features, out.trust, out.shape, out.a_exec
                if _TRUST_MIN >= 1.0:                      # the running corrector fully off
                    trust = torch.zeros_like(trust)
                elif _TRUST_MIN > 0.0:
                    trust = ((trust - _TRUST_MIN) / (1.0 - _TRUST_MIN)).clamp(0.0, 1.0)
                if _TRUST_PHASE_MAX < 1.0:                 # correct only while retrieval is predictive
                    trust = trust * (query_phase <= _TRUST_PHASE_MAX).to(trust.dtype)
                if _TRUST_MIN > 0.0 or _TRUST_PHASE_MAX < 1.0:
                    a_exec = self.corrector.blend(
                        policy, feats.a_ref, (trust * feats.authority)[:, None, None] * shape)
            a_ref, authority = feats.a_ref, feats.authority
            push = self._stall_push(a_policy, cands, score, ee_now, float(query_phase), ctx.get("stall"))
            if push is not None:
                a_ref = policy[..., :POSE_DIM] + push
                a_exec = self.corrector.blend(policy, a_ref, torch.ones_like(push))
                trust, shape, authority = torch.ones_like(trust), torch.ones_like(shape), torch.ones_like(authority)
        return AuthorityOutput(
            a_exec=a_exec[0].numpy().astype(np.float64),
            trust=trust[0].numpy().astype(np.float64),
            shape=shape[0].numpy().astype(np.float64),
            a_ref=a_ref[0].numpy().astype(np.float64),
            authority=authority[0].numpy().astype(np.float64),
            z_chk=feats.z_chk[0].numpy().astype(np.float64),
        )

    def _stall_push(self, a_policy: np.ndarray, cands: SegmentCandidates, score: torch.Tensor,
                    ee_now: torch.Tensor, phase: float, stall: "dict | None") -> "torch.Tensor | None":
        """[1, C, 6] the demos' displacement to add to a hovering policy chunk, or None.

        Counts consecutive still chunks in stall["checks"]; from STALL_CHECKS on, each candidate's
        demo is read at the locator's phase (rel_ref.at_phase), the candidates that move there are
        score-weighted, and the chunk tracking the live pose plus their mean displacement comes back
        when they hold most of the mass and agree on the direction.
        """
        if STALL_CHECKS <= 0 or stall is None or self.rel_ref is None:
            return None
        if _LEAN:
            return self._lean_push(a_policy, cands, score, ee_now, phase, stall)
        pending = self._retry_step(a_policy, ee_now, stall)
        if pending is not None:
            return pending
        if phase < _STALL_PHASE_MIN or phase >= _STALL_PHASE_MAX:
            return None
        # Motion in pose units (metres/radians per step), the unit the demo's own pauses are read in.
        unit = np.array([self.scale.pos] * 3 + [self.scale.rot] * 3, dtype=np.float64)
        motion = float(np.abs(np.asarray(a_policy, dtype=np.float64)[:, :POSE_DIM] * unit).mean())
        still = motion < _STALL_EPS and bool(stall.get("partner_still", True))
        stall["checks"] = stall.get("checks", 0) + 1 if still else 0
        # The displacement is read at the locator's phase, so it is only as good as that estimate:
        # a belief that jumped since the last check names no phase to read at.
        stable = stall.get("phase") is not None and abs(phase - stall["phase"]) <= _STALL_PHASE_TOL
        stall["phase"] = phase
        stall["push"] = 0.0
        rows = int(np.asarray(a_policy).shape[0])
        p = torch.softmax(score.clamp(-SCORE_CLAMP, SCORE_CLAMP), dim=-1)                  # [1, K]
        need = STALL_CHECKS * rows
        if _DWELL:
            pause = self.rel_ref.dwell(cands.start_node)                                       # [K] frames
            held = pause[(p[0] >= _DWELL_MASS).numpy()]
            need = stall["dwell"] = max(need, int(held.max()) + rows if len(held) else need)
        if stall["checks"] * rows < need or not stable:
            return None
        disp = torch.as_tensor(self.rel_ref.at_phase(cands.start_node, phase, rows,
                                                     max_skip=None if _DWELL else 30).astype(np.float32))[None]
        end = disp[..., -1, :]                                                                 # [1, K, 6]
        moving = (end.norm(dim=-1) >= _STALL_MIN).to(p.dtype)                                  # [1, K]
        mass = stall["mass"] = float((p * moving).sum())
        if mass < 0.5:                                                                         # the demos pause here too
            return self._retry(cands, p, ee_now, phase, rows, stall)
        w = p * moving / mass
        mean_end = (w[..., None] * end).sum(dim=1)                                             # [1, 6]
        norm = stall["norm"] = float(mean_end.norm())
        cos = (end * mean_end[:, None, :]).sum(-1) / (end.norm(dim=-1) * norm + 1e-6)          # [1, K]
        agree = stall["agree"] = float((w * cos).sum())
        if agree < _STALL_COS or norm < _STALL_MIN:
            return None
        d_mean = (w[..., None, None] * disp).sum(dim=1)                                        # [1, C, 6]
        stall["push"] = norm
        stall["checks"] = 0            # one displacement per stall; the policy gets its dwell back
        logging.info("[STALL] push phase=%.3f dwell=%d norm=%.4f agree=%.2f mass=%.2f",
                     phase, need, norm, agree, mass)
        return _STALL_GAIN * track_chunk(ee_now[:, None, :] + d_mean, ee_now, self.scale, bound=self.scale.limit)

    def _retry(self, cands: SegmentCandidates, p: torch.Tensor, ee_now: torch.Tensor, phase: float,
               rows: int, stall: dict) -> "torch.Tensor | None":
        """[1, C, 6] the first chunk of a back-off along the demos' own path, or None.

        The demos' displacement from _RETRY_BACK before the belief's phase up to it, reversed,
        is the whole plan; _retry_step pays it out over the following checks so the arm retraces
        the approach and the policy starts the manipulation again from where the demos did.
        """
        if _RETRY <= 0 or phase < _RETRY_PHASE or stall.get("retries", 0) >= _RETRY:
            return None
        span = torch.as_tensor(self.rel_ref.between(cands.start_node, phase - _RETRY_BACK, phase)
                               .astype(np.float32))[None]                                  # [1, K, 6]
        plan = -(p[..., None] * span).sum(dim=1)[0]                                            # [6]
        norm = float(plan.norm())
        if norm < _STALL_MIN:
            return None
        stall["retries"] = stall.get("retries", 0) + 1
        stall["plan"] = plan.numpy().astype(np.float64)
        logging.info("[STALL] retry phase=%.3f back=%.2f norm=%.4f n=%d", phase, _RETRY_BACK, norm, stall["retries"])
        return self._retry_step(None, ee_now, stall, rows=rows)

    def _retry_step(self, a_policy: "np.ndarray | None", ee_now: torch.Tensor, stall: "dict | None",
                    rows: "int | None" = None) -> "torch.Tensor | None":
        """[1, C, 6] the next chunk of a pending back-off plan, or None when none is pending.

        Each check covers at most _RETRY_STEP of what remains, ramped linearly over the rows;
        the plan is cleared, and the stall counter with it, when it is spent.
        """
        if stall is None or stall.get("plan") is None:
            return None
        rows = int(np.asarray(a_policy).shape[0]) if rows is None else int(rows)
        remaining = np.asarray(stall["plan"], dtype=np.float64)
        norm = float(np.linalg.norm(remaining))
        if norm < _STALL_MIN:
            stall["plan"] = None
            stall["checks"] = 0
            return None
        step = remaining * min(1.0, _RETRY_STEP / norm)
        stall["plan"] = remaining - step
        stall["push"] = -float(np.linalg.norm(step))
        stall["checks"] = 0
        ramp = torch.as_tensor(step, dtype=torch.float32)[None, None, :] * \
            (torch.arange(1, rows + 1, dtype=torch.float32) / rows)[None, :, None]           # [1, C, 6]
        return _STALL_GAIN * track_chunk(ee_now[:, None, :] + ramp, ee_now, self.scale, bound=self.scale.limit)

    def _lean_push(self, a_policy: np.ndarray, cands: SegmentCandidates, score: torch.Tensor,
                   ee_now: torch.Tensor, phase: float, stall: dict) -> "torch.Tensor | None":
        """[1, C, 6] (1 - s) times the candidates' mean displacement at the belief's phase, or None.

        s = min(1, |a| / eps) is how much the whole robot moves on its own this check, |a| the
        larger of this arm's mean joint step and the partner's (stall["partner_motion"]). A
        moving robot adds nothing; a still one adds the demonstrations' own next displacement,
        which is itself near zero where the demonstrations pause.
        """
        own = float(np.abs(np.asarray(a_policy, dtype=np.float64)[:, :POSE_DIM]).mean())
        motion = max(own, float(stall.get("partner_motion", 0.0)))
        s = min(1.0, motion / _STALL_EPS)
        stall["push"] = 0.0
        if s >= 1.0:
            return None
        rows = int(np.asarray(a_policy).shape[0])
        disp = torch.as_tensor(self.rel_ref.at_phase(cands.start_node, phase, rows).astype(np.float32))[None]
        w = torch.softmax(score.clamp(-SCORE_CLAMP, SCORE_CLAMP), dim=-1)                  # [1, K]
        d_mean = (w[..., None, None] * disp).sum(dim=1)                                        # [1, C, 6]
        stall["push"] = float(d_mean[0, -1].norm()) * (1.0 - s)
        return (1.0 - s) * _STALL_GAIN * track_chunk(ee_now[:, None, :] + d_mean, ee_now, self.scale, bound=self.scale.limit)
