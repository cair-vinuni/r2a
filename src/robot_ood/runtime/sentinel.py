"""Sentinel -- the mid-episode locator for a frozen policy.

Env-free by construction: runs on q_now alone, driving both sim and real rollouts with end-effector
pose from forward kinematics. The composition root of the runtime -- it owns the per-worker windows
and delegates loading to robot_ood.runtime.bundle and the check flow to robot_ood.runtime.step.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_ood.core.config import SentinelConfig
from robot_ood.runtime import step as step_flow
from robot_ood.runtime.bundle import RuntimeBundle

if TYPE_CHECKING:                                          # pragma: no cover -- typing only
    from robot_ood.runtime.fuse import BlendPlan
    from robot_ood.runtime.locator import GraphLocator

__all__ = ["Sentinel"]


@dataclasses.dataclass
class _WorkerState:
    """All per-worker mutable state. One instance per parallel rollout; never shared.

    Attributes:
        q_win: Rolling query-history window, bounded by its own maxlen.
        s_win: The env step each q_win row was observed at, same maxlen and pushed in lockstep, so
            a check can measure the window's REAL row spacing instead of trusting the knob that was
            supposed to set it.
        a_win: Rolling COMMANDED-action history window, same maxlen and stride gating as q_win --
            the rows robot_ood.core.schemas.QueryContext.a_hist is built from. "Commanded" means the row
            that actually reached the arm (post blend), the deploy-side analogue of
            robot_ood.train.data's a_hist, which reads the demo's own recorded actions.
        locator: The locator, None until a check needs it -- constructing one loads graph
            artifacts from disk, which must not happen just because reset was called.
    """

    q_win: deque
    s_win: deque
    a_win: deque
    locator: "GraphLocator | None" = None
    last_gate: "np.ndarray | None" = None
    last_task_p: "np.ndarray | None" = None
    last_phase: float = 0.0
    stall: dict = dataclasses.field(default_factory=dict)

    def start_episode(self) -> None:
        """Reset every per-episode channel: window, locator, stall counter."""
        self.q_win.clear()
        self.s_win.clear()
        self.a_win.clear()
        self.last_phase = 0.0
        self.stall.clear()
        if self.locator is not None:
            self.locator.reset()


class Sentinel:
    """Track the arm config against the demo graph, publishing every plan_rows steps."""

    def __init__(self, cfg: SentinelConfig | None = None,
                 fwm_dir: str | os.PathLike | None = None):
        """Initialize this Sentinel.

        Args:
            cfg: Sentinel knobs; None reads them from the environment.
            fwm_dir: Field directory; None resolves robot_ood.core.paths.Paths.fwm.
        """
        self.bundle = RuntimeBundle(cfg, fwm_dir)
        self._workers: dict[int, _WorkerState] = {}

    @property
    def cfg(self) -> SentinelConfig:
        """This Sentinel's knobs."""
        return self.bundle.cfg

    @property
    def plan_rows(self) -> int:
        """Rows one plan covers, and so the period between checks."""
        return self.cfg.plan_rows

    @property
    def fwm_dir(self) -> str:
        """The field directory this Sentinel resolves its suite and artifacts from."""
        return self.bundle.fwm_dir

    # ---- public API ----------------------------------------------------------------------------------
    def reset(self, wid: int) -> None:
        """Reset every per-worker channel for wid."""
        self._worker(wid).start_episode()

    def observe(self, q_now: np.ndarray, step: int, *, wid: int) -> None:
        """Push this step's arm config into the query-history window; call EVERY env step.

        Args:
            q_now: [D] arm config in radians, sampled by the caller this step.
            step: The harness's own step counter.
            wid: Worker id; each gets its own window.
        """
        worker = self._worker(wid)
        q_now = np.asarray(q_now, dtype=np.float64)
        # Window row spacing must stay graph_stride env steps apart, matching training: otherwise
        # velocity channel sees wrong displacement magnitude.
        if step % self.cfg.graph_stride != 0:
            return
        worker.q_win.append(q_now)
        worker.s_win.append(int(step))

    def observe_action(self, a_row: np.ndarray, step: int, *, wid: int) -> None:
        """Push this step's COMMANDED action into the action-history window; call EVERY env step.

        Args:
            a_row: [ACTION_DIM] the executed action this step.
            step: The harness's own step counter.
            wid: Worker id; each gets its own window.
        """
        # a_win rows align 1:1 with q_win: QueryContext's qh/a_hist windows span same episode span.
        if step % self.cfg.graph_stride != 0:
            return
        worker = self._worker(wid)
        worker.a_win.append(np.asarray(a_row, dtype=np.float32))

    def check(self, q_now: np.ndarray, grip: float, step: int, *,
              wid: int, task: str = "") -> float | None:
        """Run the locator once per plan_rows steps over the query-history window.

        Args:
            q_now: [D] arm config in radians, sampled by the caller this step.
            grip: The grip_flag reading for this step.
            step: The harness's own step counter.
            wid: Worker id; each gets its own locator and query window.
            task: The episode's instruction string.

        Returns:
            The locator's phase estimate, or None on a step that was not a check.
        """
        worker = self._worker(wid)
        q_now = np.asarray(q_now, dtype=np.float64)
        if step % self.plan_rows != 0:
            return None
        self._assert_window(worker, step)
        if self.cfg.ablation.graph:
            ee_now = self.bundle.kinematics().pose(q_now[None, :])[0]
            ctx = self.bundle.query_context(q_win=worker.q_win, a_win=worker.a_win, grip=grip,
                                            task=task, ee_now=ee_now, phase=worker.last_phase)
            phase_hat = self.bundle.topk_phase(ctx)
            worker.last_phase = phase_hat
            worker.last_gate = None
            worker.last_task_p = None
        else:
            q_hist = np.asarray(worker.q_win, np.float64) if worker.q_win else q_now[None, :]
            g_hist = np.full(q_hist.shape[0], grip, dtype=np.float64)
            result = self._get_locator(wid).step(q_hist, g_hist=g_hist, task=task)
            phase_hat = float(result.phase_hat)
            worker.last_gate = result.gate
            worker.last_task_p = result.task_p
        logging.info("[SN] w=%d step=%d phase_hat=%.3f", wid, step, phase_hat)
        return phase_hat

    def plan(self, q_now: np.ndarray, grip: float, step: int, *,
             wid: int, task: str = "", seg_prior: "np.ndarray | None" = None) -> "BlendPlan | None":
        """Run one check and turn its reference segment into a blend for the next action chunk.

        Args:
            q_now: [D] arm config in radians, sampled by the caller this step.
            grip: The grip_flag reading for this step.
            step: The harness's own step counter.
            wid: Worker id; each gets its own locator and query window.
            task: The task string, the locator's instruction channel.
            seg_prior: [S] log prior over the segment table for retrieval's shortlist cut, or None.

        Returns:
            The blend, or None on a step that was not a check.
        """
        # Live pose from FK(q_now) closes the loop: no world-frame reading, no per-scene base offset.
        phase_hat = self.check(q_now, grip, step, wid=wid, task=task)
        if phase_hat is None:
            return None
        return self.plan_checked(q_now, grip, phase_hat, wid=wid, task=task, seg_prior=seg_prior)

    def plan_checked(self, q_now: np.ndarray, grip: float, phase_hat: float, *,
                     wid: int, task: str = "", seg_prior: "np.ndarray | None" = None) -> "BlendPlan":
        """The second half of plan(): the blend for a check that check() has already run.

        Split out so a two-armed harness can run both arms' checks, read both queries, and only
        then hand each arm the prior the other arm's query implies.

        Args:
            q_now: [D] arm config in radians, the same rows check() read.
            grip: The grip_flag reading for this step.
            phase_hat: What check() returned for this step.
            wid: Worker id.
            task: The task string.
            seg_prior: [S] log prior over the segment table for retrieval's shortlist cut, or None.

        Returns:
            The blend.
        """
        worker = self._worker(wid)
        record_chunk = None if self.cfg.ablation.graph else self._get_locator(wid).record_chunk
        return step_flow.plan(self.bundle, record_chunk, phase_hat, np.asarray(q_now, dtype=np.float64),
                              q_win=worker.q_win, a_win=worker.a_win, grip=grip, task=task,
                              gate=worker.last_gate, task_p=worker.last_task_p, seg_prior=seg_prior,
                              stall=worker.stall)

    def query_context(self, q_now: np.ndarray, grip: float, phase_hat: float, *, wid: int, task: str = "") -> Any:
        """The QueryContext this check's blend would hand the retriever, without retrieving.

        Args:
            q_now: [D] arm config in radians.
            grip: The grip_flag reading for this step.
            phase_hat: What check() returned for this step.
            wid: Worker id.
            task: The task string.

        Returns:
            The context, B == 1, carrying the locator's task marginal and no seg_prior.
        """
        worker = self._worker(wid)
        q_now = np.asarray(q_now, dtype=np.float64)
        ee_now = self.bundle.kinematics().pose(q_now[None, :])[0]
        return self.bundle.query_context(q_win=worker.q_win, a_win=worker.a_win, grip=grip, task=task,
                                         ee_now=ee_now, phase=phase_hat, task_p=worker.last_task_p)

    # ---- per-worker state ----------------------------------------------------------------------------
    def _assert_window(self, worker: _WorkerState, step: int) -> None:
        """Refuse to run a check on a window whose rows are not graph_stride env steps apart.

        Args:
            worker: The worker whose window is about to be read.
            step: The step of the check, for the error message.

        Raises:
            RuntimeError: The window is empty (observe() is not wired up), or its rows are spaced
                by anything other than graph_stride and allow_stride_mismatch is off.
        """
        # Check actual spacing, not just config: caller's observe() frequency determines real spacing.
        c = self.cfg
        if not worker.s_win or worker.s_win[-1] != step:
            raise RuntimeError(
                f"Sentinel.check at step {step} on a window last observed at "
                f"{worker.s_win[-1] if worker.s_win else 'never'}: call Sentinel.observe(q_now, "
                "step, wid=...) on EVERY env step. The window is the retriever's only input and a "
                "stale or empty one silently degrades every number the run produces."
            )
        if len(worker.s_win) < 2:
            return
        gaps = np.diff(np.asarray(worker.s_win, dtype=np.int64))
        if not np.all(gaps == c.graph_stride):
            raise RuntimeError(
                f"Sentinel.check at step {step}: query-window rows are spaced {sorted(set(gaps.tolist()))} "
                f"env steps apart, not graph_stride={c.graph_stride}. Training windows are "
                f"{c.graph_win} CONSECUTIVE raw frames, so a wider spacing multiplies the seed "
                "kernel's velocity channel by that factor. Call observe() every env step."
            )

    def _worker(self, wid: int) -> _WorkerState:
        """The state for wid, constructed on first use. Cheap: no artifact is loaded here."""
        worker = self._workers.get(wid)
        if worker is None:
            worker = _WorkerState(q_win=deque(maxlen=self.cfg.graph_win),
                                  s_win=deque(maxlen=self.cfg.graph_win),
                                  a_win=deque(maxlen=self.cfg.graph_win))
            self._workers[wid] = worker
        return worker

    def _get_locator(self, wid: int) -> "GraphLocator":
        """This worker's locator, built on first use."""
        worker = self._worker(wid)
        if worker.locator is None:
            worker.locator = self.bundle.build_locator()
        return worker.locator

