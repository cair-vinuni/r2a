"""The one-shot joint-space move onto the demonstrated entry manifold, plus the posture re-point.

Design writeup: docs/method-runtime.md.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Final

import numpy as np
import torch

from robot_ood.core import kinematics
from robot_ood.core.config import EntryConfig, RADIUS_TO_SIGMA
from robot_ood.models.entry.invdyn import InverseDynamicsNet
from robot_ood.models.entry.score import ScoreNet, tweedie_denoise
from robot_ood.models.entry.target import EntryTarget

# Non-termination guard on the walk, not a budget or a tuned magnitude.
_MAX_WALK: Final[int] = 64

FrameCallback = Callable[[np.ndarray], None]

# Terminal observation-resync action: zero end-effector delta, gripper commanded OPEN.
_RESYNC_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
_GRIPPER_OPEN = -1.0
_RENDER_KW = {"width": 256, "height": 256, "camera_name": "agentview"}


def _frame(env) -> np.ndarray:
    if hasattr(env, "onf_render"):
        return env.onf_render()
    return kinematics.sim(env).render(**_RENDER_KW)[::-1]


class ScoreEntryCorrector:
    """Denoise the current configuration onto the demonstrated entry manifold, walk there, re-point posture.

    Attributes:
        target: EntryTarget, read ONLY for the err0/err1 diagnostics logged in last -- the
            control flow below never branches on it.
        score_net: (q, sigma) -> score plus the level head, trained by train entry.
        invdyn_net: (q, delta_q) -> action, trained by train entry.
        cfg: The level clamps, the refinement count and the step cap.
        last: The previous run's report, or None before the first run.
    """

    def __init__(self, target: EntryTarget, score_net: ScoreNet, invdyn_net: InverseDynamicsNet,
                 cfg: EntryConfig | None = None) -> None:
        """Initialize the entry corrector with target and trained networks.

        Args:
            target: Diagnostic-only reference point.
            score_net: The trained score field.
            invdyn_net: The trained inverse-dynamics action map.
            cfg: Entry knobs; None reads them from the environment.
        """
        self.target = target
        self.score_net = score_net
        self.invdyn_net = invdyn_net
        self.cfg = cfg or EntryConfig.from_env()
        self.last: dict | None = None

    @torch.no_grad()
    def run(self, env: Any, obs: Any, t: int,
            frame_cb: FrameCallback | None = None) -> tuple[Any, int]:
        """Denoise, walk onto the estimate when it lies outside the arrival radius, and hand off.

        A start already inside the level head's radius of its own estimate is left alone: no
        control step, no resync, the env counter untouched. On a scene perturbation the arm starts
        where the demonstrations start (delta ~0.0005 rad against a radius ~0.009), and a step
        emitted there moves nothing yet changes the rollout the policy then produces.

        Args:
            env: The robosuite-style environment.
            obs: Current observation, replaced by the final resync step's own when the arm walked.
            t: Current episode step count.
            frame_cb: Optional per-step callback receiving a rendered RGB frame.

        Returns:
            (obs, t) with t advanced by the emitted control steps plus one resync step, or
            unchanged when the arm did not walk.
        """
        q_now = kinematics.q_arm(env)
        err0 = self.target.error(q_now)
        q0 = torch.as_tensor(q_now, dtype=torch.float32)

        q_hat = q0
        for _ in range(self.cfg.score_refine_steps):
            sigma = min(max(float(self.score_net.level_estimate(q_hat)) * RADIUS_TO_SIGMA,
                            self.cfg.score_sigma_lo), self.cfg.score_sigma_hi)
            score = self.score_net(q_hat, torch.as_tensor(sigma, dtype=torch.float32))
            q_hat = tweedie_denoise(score, q_hat, sigma)
        delta = float((q_hat - q0).norm())

        cap = self.cfg.score_step_cap
        level_hat = float(self.score_net.level_estimate(q_hat))
        level0 = float(self.score_net.level_estimate(q0))
        n_steps = 0
        moved_total = 0.0
        remaining = delta
        commanded = 0.0
        while remaining > level_hat and n_steps < _MAX_WALK:
            q_t = torch.as_tensor(q_now, dtype=torch.float32)
            dq_t = q_hat - q_t
            # cap is a rate limit only; it does not decide when the walk stops.
            dq_norm = dq_t.norm()
            if dq_norm > cap:
                dq_t = dq_t * (cap / dq_norm)
            action = self.invdyn_net(q_t, dq_t).numpy().astype(np.float32).copy()
            action[6] = _GRIPPER_OPEN

            kinematics.set_nullspace_home(env, q_hat.numpy())
            obs, *_ = env.step(action)
            t += 1
            n_steps += 1
            if frame_cb is not None:
                frame_cb(_frame(env))
            q_prev, q_now = q_now, kinematics.q_arm(env)

            moved = float(np.linalg.norm(q_now - q_prev))
            moved_total += moved
            remaining = float((q_hat - torch.as_tensor(q_now, dtype=torch.float32)).norm())
            commanded += float(dq_t.norm())
            logging.debug("[EN-SCORE] walk %d cmd=%.4f moved=%.4f total=%.4f remaining=%.4f "
                          "(arrive %.4f)", n_steps, float(dq_t.norm()), moved, moved_total,
                          remaining, level_hat)

        if n_steps > 0:
            obs, *_ = env.step(_RESYNC_ACTION)
            t += 1
            if frame_cb is not None:
                frame_cb(_frame(env))

        q_end = kinematics.q_arm(env)
        err1 = self.target.error(q_end)
        # The head's reading at the configuration actually handed off, so the SR logs carry both
        # sides of the stopping rule and the deployed arrival radius is measurable, not inferred.
        level_now = float(self.score_net.level_estimate(
            torch.as_tensor(q_end, dtype=torch.float32)))
        realised = (moved_total / commanded) if commanded > 0.0 else float("nan")
        self.last = {"fired": n_steps > 0, "err0": err0, "err1": err1, "steps": n_steps,
                     "delta": delta, "residual": remaining, "realised": realised,
                     "level_now": level_now, "level_hat": level_hat, "level0": level0,
                     "moved_total": moved_total, "walk_capped": n_steps >= _MAX_WALK}
        if n_steps >= _MAX_WALK:
            logging.warning("[EN-SCORE] walk hit the non-termination guard at %d steps; the gap "
                            "never fell inside level_hat=%.4f, so this handoff is unconverged and "
                            "the step count was set by the guard rather than by arrival.",
                            _MAX_WALK, level_hat)
        logging.info("[EN-SCORE] WALK residual=%.4f moved_total=%.4f realised/cmd=%.3f capped=%s",
                     remaining, moved_total, realised, n_steps >= _MAX_WALK)
        logging.info("[EN-SCORE] HANDOFF delta=%.4f level0=%.4f level_hat=%.4f level_now=%.4f "
                     "(rate cap %.4f, K=%d) steps=%d |q_err|_M %.3f -> %.3f", delta,
                     level0, level_hat, level_now, self.cfg.score_step_cap,
                     self.cfg.score_refine_steps, n_steps, err0, err1)
        return obs, t
