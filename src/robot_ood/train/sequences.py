"""Check sequences: rows of consecutive checks on one demo, each with its window, instruction, policy chunk and label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from robot_ood.core.constants import PHASE_TOL
from robot_ood.data.geometry import edge_pad_hist, finite_diff_vel
from robot_ood.data.nodes import NodeTable
from robot_ood.train.data import DRIFT_MIN_HIST, entry_perturb, entry_sigma_range, joint_clip_bounds
from robot_ood.train.target import TargetBuilder

__all__ = ["CHECK_EVERY", "WINDOW_MODES", "WINDOW_P", "INSTR_MODES", "INSTR_P", "POLICY_NONE_P", "Check", "Sequence", "SequenceSampler", "RawWindowDataset"]

CHECK_EVERY = 8
WINDOW_MODES = ("traversal", "entry", "entry_static", "drift")
WINDOW_P = (0.48, 0.28, 0.16, 0.08)
INSTR_MODES = ("exact", "paraphrase", "none", "mixed")
INSTR_P = (0.25, 0.35, 0.25, 0.15)
# "mixed": the true task's embedding blended with another task's, the true one dominant -- the
# shape a real paraphrase takes once the deployed read is unsure between two tasks.
MIXED_W = (0.5, 0.9)
POLICY_NONE_P = 0.2
N_CHECKS = (4, 8)


class RawWindowDataset:
    """History windows sliced from a raw-frame trajectory at the deployed check cadence."""

    def __init__(self, traj_raw: np.ndarray, hist: int, check_every: int) -> None:
        self.traj = traj_raw
        starts = np.arange(0, traj_raw.shape[0], check_every, dtype=np.int64)
        self.hi = starts + 1
        self.lo = np.maximum(0, self.hi - hist)

    def __len__(self) -> int:
        return int(self.hi.shape[0])

    def __getitem__(self, index: int) -> np.ndarray:
        lo, hi = int(self.lo[index]), int(self.hi[index])
        return self.traj[lo:hi].astype(np.float64)


@dataclass(frozen=True, slots=True)
class Check:
    """One check of a batch of rows.

    Attributes:
        qh: [B, H, D] float32 joint window.
        vh: [B, H, D] float32 velocities.
        gh: [B, H] float32 gripper.
        tgt: [B] int64 the true node.
        policy: [B, chunk, 7] float32 the policy chunk just issued, zeros where absent.
        has_policy: [B] bool.
        window_mode: [B] int8 index into WINDOW_MODES.
    """

    qh: np.ndarray
    vh: np.ndarray
    gh: np.ndarray
    tgt: np.ndarray
    policy: np.ndarray
    has_policy: np.ndarray
    window_mode: np.ndarray


@dataclass(frozen=True, slots=True)
class Sequence:
    """A batch of rows, every row the same number of checks.

    Attributes:
        checks: The checks in order.
        instr: [B, instr_dim] float32 instruction embedding, zeros where absent.
        has_instr: [B] bool.
        instr_mode: [B] int8 index into INSTR_MODES.
        demo: [B] int64 owner of each row.
        task: [B] int64 true task of each row.
        instr_p: [B, n_tasks] float32 the instruction's task posterior, uniform where absent.
    """
    checks: list[Check]
    instr: np.ndarray
    has_instr: np.ndarray
    instr_mode: np.ndarray
    demo: np.ndarray
    task: np.ndarray
    instr_p: np.ndarray | None = None

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    def __len__(self) -> int:
        return int(self.demo.shape[0])


class SequenceSampler:
    """Draw check sequences over one corpus."""

    def __init__(self, nodes: NodeTable, policy_cache: Any, language: Any, corruptor: Any,
                 targets: TargetBuilder, *, hist: int, chunk: int, joint_space: bool = True,
                 seed: int = 0) -> None:
        self.nodes = nodes
        self.policy = policy_cache
        self.language = language
        self.corruptor = corruptor
        self.targets = targets
        self.hist = int(hist)
        self.chunk = int(chunk)
        self.rng = np.random.RandomState(seed)
        self.gen = np.random.default_rng(seed)
        self.raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
        self.q_raw = np.asarray(nodes.q_raw, dtype=np.float32)
        self.grip = np.asarray(nodes.grip, dtype=np.float32)
        self.owner = np.asarray(nodes.owner, dtype=np.int64)
        self.phase = np.asarray(nodes.phase, dtype=np.float64)
        self.task_of_demo = np.asarray(nodes.task_id, dtype=np.int64)[
            np.asarray(nodes.demo_ptr, dtype=np.int64)[:-1]]
        self.dim = int(nodes.dim)
        self.q_lo, self.q_hi = joint_clip_bounds(self.dim, joint_space)
        self.sigma_range = entry_sigma_range(nodes)
        self.e_task = np.asarray(language.e_task, dtype=np.float32)
        self.language = language
        self.e_aug = np.asarray(language.e_aug, dtype=np.float32)
        self.p_task = np.asarray(language.p_task, dtype=np.float32)
        self.p_aug = np.asarray(language.p_aug, dtype=np.float32)

    # ---- drawing ----
    def usable(self, owners: np.ndarray, n_checks: int) -> np.ndarray:
        need = self.hist - 1 + CHECK_EVERY * (n_checks - 1) + 1
        lengths = np.diff(self.raw_ptr)
        owners = np.asarray(owners, dtype=np.int64)
        return owners[lengths[owners] >= need]

    def draw(self, owners: np.ndarray, batch: int, n_checks: int | None = None, *,
             rank0: bool = False, instr_mode: str | None = None,
             policy_clean: bool = False) -> Sequence:
        """One batch of sequences.

        Args:
            owners: Demos to draw from.
            batch: Rows per batch.
            n_checks: Checks per row; None draws U{N_CHECKS} for the batch.
            rank0: Start every row at the demo start, the deployed operating point.
            instr_mode: Fix the instruction regime, or None to draw per row.
            policy_clean: Never corrupt or drop the policy chunk.

        Returns:
            The batch.
        """
        rng = self.rng
        n = int(n_checks) if n_checks is not None else int(rng.randint(N_CHECKS[0], N_CHECKS[1] + 1))
        pool = self.usable(owners, n)
        while n > 1 and pool.size == 0:
            n -= 1
            pool = self.usable(owners, n)
        if pool.size == 0:
            raise ValueError("SequenceSampler.draw: no demo in owners is long enough for one check")
        demo = pool[rng.randint(len(pool), size=batch)]
        lo, hi = self.raw_ptr[demo], self.raw_ptr[demo + 1]
        first = lo + self.hist - 1
        last = hi - 1 - CHECK_EVERY * (n - 1)
        start = first if rank0 else np.array([int(rng.randint(f, l + 1)) for f, l in zip(first, last)])
        frames = start[:, None] + CHECK_EVERY * np.arange(n, dtype=np.int64)[None, :]
        task = self.task_of_demo[demo]

        modes = np.full(batch, INSTR_MODES.index(instr_mode), dtype=np.int8) if instr_mode else (
            rng.choice(len(INSTR_MODES), size=batch, p=INSTR_P).astype(np.int8))
        instr = np.zeros((batch, self.e_task.shape[1]), dtype=np.float32)
        exact, para = modes == 0, modes == 1
        instr[exact] = self.e_task[task[exact]]
        which = self.language.draw_aug(task, rng.random_sample(batch))
        instr[para] = self.e_aug[which[para]]
        n_tasks = self.e_task.shape[0]
        instr_p = np.full((batch, n_tasks), 1.0 / n_tasks, dtype=np.float32)
        instr_p[exact] = self.p_task[task[exact]]
        instr_p[para] = self.p_aug[which[para]]
        mixed = modes == 3
        if mixed.any():
            m = int(mixed.sum())
            other = (task[mixed] + rng.randint(1, n_tasks, size=m)) % n_tasks
            w = rng.uniform(MIXED_W[0], MIXED_W[1], size=(m, 1)).astype(np.float32)
            v = w * self.e_task[task[mixed]] + (1.0 - w) * self.e_task[other]
            instr[mixed] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
            instr_p[mixed] = w * self.p_task[task[mixed]] + (1.0 - w) * self.p_task[other]
        checks = [self._check(demo, frames[:, k], lo, policy_clean) for k in range(n)]
        return Sequence(checks=checks, instr=instr, has_instr=modes != 2, instr_mode=modes,
                        demo=demo, task=task, instr_p=instr_p)

    def _check(self, demo: np.ndarray, frame: np.ndarray, lo: np.ndarray,
               policy_clean: bool) -> Check:
        rng = self.rng
        batch = demo.shape[0]
        tgt = self.nodes.node_at_raw_many(demo, frame)
        modes = rng.choice(len(WINDOW_MODES), size=batch, p=WINDOW_P).astype(np.int8)
        qh = np.stack([self._window(int(demo[i]), int(frame[i]), int(lo[i]), WINDOW_MODES[modes[i]])
                       for i in range(batch)])
        gh = np.repeat(self.grip[tgt][:, None], self.hist, axis=1)

        prev = frame - self.chunk
        has_policy = prev >= lo
        if not policy_clean:
            has_policy &= rng.random_sample(batch) >= POLICY_NONE_P
        policy = np.zeros((batch, self.chunk, self.policy.a_pi_raw.shape[1]), dtype=np.float32)
        rows = np.flatnonzero(has_policy)
        if rows.size:
            clean = self.policy.chunks(prev[rows], self.chunk).astype(np.float32)
            if policy_clean:
                policy[rows] = clean
            else:
                drawn = self.corruptor.draw(tgt[rows], prev[rows], self.gen)
                policy[rows] = drawn.apply(clean, self.policy, self.chunk)
        return Check(qh=qh, vh=finite_diff_vel(qh), gh=gh, tgt=tgt.astype(np.int64),
                     policy=policy, has_policy=has_policy, window_mode=modes)

    def _window(self, demo: int, frame: int, lo: int, mode: str) -> np.ndarray:
        rng, h = self.rng, self.hist
        if mode == "entry_static":
            q0 = self.q_raw[frame].astype(np.float64)
            q = entry_perturb(q0, rng, self.dim, self.sigma_range, self.q_lo, self.q_hi)
            return np.repeat(q[None, :].astype(np.float32), h, axis=0)
        w_lo = DRIFT_MIN_HIST if mode == "drift" else 1
        w = min(int(rng.randint(w_lo, h + 1)), frame - lo + 1)
        raw = self.q_raw[frame - w + 1 : frame + 1].astype(np.float64)
        if mode == "entry":
            delta = entry_perturb(raw[-1], rng, self.dim, self.sigma_range, self.q_lo, self.q_hi) - raw[-1]
            raw = np.clip(raw + delta[None, :], self.q_lo, self.q_hi)
        elif mode == "drift" and w >= DRIFT_MIN_HIST:
            n_clean = int(rng.randint(1, w - 1))
            delta = entry_perturb(raw[-1], rng, self.dim, self.sigma_range, self.q_lo, self.q_hi) - raw[-1]
            ramp = np.arange(1, w - n_clean + 1, dtype=np.float64) / (w - n_clean)
            raw = raw.copy()
            raw[n_clean:] += ramp[:, None] * delta[None, :]
            raw = np.clip(raw, self.q_lo, self.q_hi)
        return edge_pad_hist(raw.astype(np.float32), h)[0]

    # ---- memories ----
    def dense_target(self, tgt: np.ndarray, has_instr: np.ndarray, n_nodes: int) -> np.ndarray:
        """[B, V] float32 each row's target_weight scattered over the graph."""
        out = np.zeros((tgt.shape[0], n_nodes), dtype=np.float32)
        for i in range(tgt.shape[0]):
            t = self.targets(int(tgt[i]), bool(has_instr[i]))
            out[i, t.nodes] = t.weight
        return out

    def wrong_nodes(self, tgt: np.ndarray, demo: np.ndarray) -> np.ndarray:
        """[B] a node on another strand within PHASE_TOL of each true node's phase."""
        out = np.empty(tgt.shape[0], dtype=np.int64)
        for i in range(tgt.shape[0]):
            band = np.flatnonzero((np.abs(self.phase - self.phase[tgt[i]]) <= PHASE_TOL)
                                  & (self.owner != demo[i]))
            out[i] = int(band[self.rng.randint(len(band))]) if band.size else int(tgt[i])
        return out


def to_device(check: Check, dev: Any) -> dict[str, torch.Tensor]:
    """The check's window and policy tensors on device."""
    return {
        "qh": torch.as_tensor(check.qh, device=dev), "vh": torch.as_tensor(check.vh, device=dev),
        "gh": torch.as_tensor(check.gh, device=dev),
        "policy": torch.as_tensor(check.policy, device=dev),
        "has_policy": torch.as_tensor(check.has_policy.astype(np.float32), device=dev),
    }
