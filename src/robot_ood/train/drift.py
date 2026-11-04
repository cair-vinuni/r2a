"""DriftGenerator -- data-only corrupted-trajectory synthesis for the graph retriever's curriculum.

Three corruption modes (velocity, rotation, splice), all deterministic and clipped to joint limits.
Held-out separation is by demo, never by frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch

from robot_ood.data import schema

# Franka Panda datasheet limits (radians).
PANDA_JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
PANDA_JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)

# DRIFT_CALIB_* were bisected against the deployment drift quantiles; provenance and the
# recipe are in docs/method-offline.md.
DRIFT_CALIB_SIGMA_SCALE = 13.868
DRIFT_CALIB_ROTATION_BOOST = 4.0
DRIFT_CALIB_ROTATION_CAP = 170.0
DRIFT_CALIB_STEPS_MULT = 4.0


@dataclass
class DriftConfig:
    """Curriculum knobs for DriftGenerator.

    Attributes:
        sigma: Per-step velocity-noise scale range (rad, relative to |qdot_t|), sampled uniformly.
        n_steps: Corrupted-rollout length range (raw steps), sampled uniformly. The floor mirrors
            HIST_H so every rollout can always fill a full window.
        window: Query window length W, one policy action chunk.
        advance: Raw frames from the window end to the ground-truth node.
        splice_frac: Fraction of n_steps spent on the foreign demo's local flow;
            j_switch = round((1 - splice_frac) * n_steps). Left unchanged by calibration.
        rotation_deg: |theta| range (degrees), sampled once per sample.
        k_flow: Neighbour count for the splice local-velocity lookup.
        kernel_bw: Gaussian bandwidth of that lookup, reused for splice.
        flow_beta: Tube-contraction strength of that lookup.
    """

    sigma: tuple[float, float] = (0.02 * DRIFT_CALIB_SIGMA_SCALE, 0.35 * DRIFT_CALIB_SIGMA_SCALE)
    n_steps: tuple[int, int] = (schema.HIST_H - 1, int(schema.HIST_H * 8 * DRIFT_CALIB_STEPS_MULT))
    window: int = schema.HIST_H
    advance: int = schema.ADVANCE
    splice_frac: float = 0.5
    rotation_deg: tuple[float, float] = (
        5.0 * DRIFT_CALIB_ROTATION_BOOST, min(45.0 * DRIFT_CALIB_ROTATION_BOOST, DRIFT_CALIB_ROTATION_CAP),
    )
    k_flow: int = 12
    kernel_bw: float = schema.KERNEL_BW
    flow_beta: float = 0.6


class NodeTableLike(Protocol):
    """The complete set of node-table attributes and methods DriftGenerator touches.

    Attributes:
        q_raw: [N_raw, D] raw per-frame joint configs.
        qdot_raw: [N_raw, D] raw per-frame joint velocities.
        raw_ptr: [J+1] per-demo boundaries into the raw arrays.
        task_id: [V] per-node task ids.
        dim: Degrees of freedom, D.
        n_demos: Number of demos, J.
    """

    q_raw: np.ndarray
    qdot_raw: np.ndarray
    raw_ptr: np.ndarray
    task_id: np.ndarray
    dim: int
    n_demos: int

    def demo_nodes(self, owner: int) -> Sequence[int]: ...


class _DemoVelocity:
    """The velocity one demo carries, read off its own raw frames by Gaussian-kernel k-NN."""

    def __init__(self, q: np.ndarray, qdot: np.ndarray, k: int, bandwidth: float, beta: float):
        self.q = np.asarray(q, dtype=np.float64)
        self.qdot = np.asarray(qdot, dtype=np.float64)
        self.k = max(1, min(k, len(self.q)))
        self.h = bandwidth
        self.beta = beta

    def at(self, q0: np.ndarray) -> np.ndarray:
        """This demo's velocity at q0, contracted toward its own tube. Returns [D] f64."""
        q0 = np.asarray(q0, np.float64)
        d = np.linalg.norm(self.q - q0[None, :], axis=1)
        idx = np.argpartition(d, min(self.k, len(d) - 1))[: self.k]
        w = np.exp(-(d[idx] ** 2) / (2 * self.h ** 2))
        w = w / (w.sum() + 1e-12)
        q_bar = (w[:, None] * self.q[idx]).sum(0)
        qdot_bar = (w[:, None] * self.qdot[idx]).sum(0)
        return self.beta * (q_bar - q0) + qdot_bar


class DriftGenerator:
    """Synthesises corrupted rollouts from a built NodeTable."""

    def __init__(self, nodes: NodeTableLike, cfg: DriftConfig | None = None, seed: int = 0):
        """Initialize a DriftGenerator.

        Args:
            nodes: A built node table (see NodeTableLike for the exact contract).
            cfg: Curriculum knobs; None uses the calibrated DriftConfig defaults.
            seed: Determinism seed.

        Raises:
            ValueError: If nodes's arrays disagree with each other or exceed the Panda's DoF.
        """
        self.nodes = nodes
        self.cfg = cfg or DriftConfig()
        self.seed = seed
        # torch.manual_seed kept for idiom uniformity with sibling modules, even though this class's
        # own methods are pure numpy.
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        self._dim = int(nodes.dim)
        self._n_demos = int(nodes.n_demos)
        self._q_raw = np.asarray(nodes.q_raw, dtype=np.float64)
        self._qdot_raw = np.asarray(nodes.qdot_raw, dtype=np.float64)
        self._raw_ptr = np.asarray(nodes.raw_ptr, dtype=np.int64)
        self._validate()

        self._q_lo = PANDA_JOINT_LO[: self._dim]
        self._q_hi = PANDA_JOINT_HI[: self._dim]

        # Coarsening never crosses a demo boundary, so every node owned by demo j shares its task_id.
        self.demo_task = np.array(
            [int(nodes.task_id[np.asarray(nodes.demo_nodes(j))[0]]) for j in range(self._n_demos)]
        )
        # One _DemoVelocity per spliced-in demo, built on first use. It is a pure function of its
        # demo's own (q, qdot) slice, so caching cannot change a single output value.
        self._flow_cache: dict[int, _DemoVelocity] = {}

    def _validate(self) -> None:
        """Check the node-table arrays are shaped as this generator requires.

        Raises:
            ValueError: On a DoF over the Panda's 7, a raw_ptr of the wrong length, or a q_raw
                whose column count disagrees with nodes.dim.
        """
        if self._dim > len(PANDA_JOINT_LO):
            raise ValueError(
                f"DriftGenerator targets the {len(PANDA_JOINT_LO)}-DoF Panda arm; got dim={self._dim}"
            )
        if len(self._raw_ptr) != self._n_demos + 1:
            raise ValueError(
                f"raw_ptr has {len(self._raw_ptr)} entries, expected n_demos+1={self._n_demos + 1}"
            )
        if self._q_raw.shape[1] != self._dim:
            raise ValueError(f"q_raw dim {self._q_raw.shape[1]} != nodes.dim {self._dim}")

    @property
    def dim(self) -> int:
        """Degrees of freedom, D."""
        return self._dim

    @property
    def n_demos(self) -> int:
        """Number of demos, J."""
        return self._n_demos

    # ---- rollouts -------------------------------------------------------------------------------

    def rollout_velocity(self, j: int, t0: int, n_steps: int, sigma: float) -> np.ndarray:
        """q'_{t+1} = clip(q'_t + qdot_t + eps_t), eps ~ N(0, sigma*|qdot_t|).

        Args:
            j: Source demo index.
            t0: Raw-frame offset within that demo.
            n_steps: Integration steps.
            sigma: Velocity-noise scale, relative to |qdot_t|.

        Returns:
            [n_steps + 1, D] trajectory.
        """
        qdot = self._qdot_slice(j, t0, n_steps)
        # One randn(n_steps, D) call: RandomState fills C-order, matching n_steps separate randn(D)
        # calls' RNG-stream order bit for bit. tests/test_drift.py pins these seeds.
        eps = sigma * np.abs(qdot) * self.rng.randn(n_steps, self._dim)
        # eps stays a SEPARATE term: (q + qdot) + eps and q + (qdot + eps) differ in the last bits
        # (~1.8e-15 over 60 steps) and the fitted artifacts on disk used the former grouping.
        return self._integrate(self._q_at(j, t0), qdot, eps)

    def rollout_rotation(
        self, j: int, t0: int, n_steps: int, theta: float, i_ax: int, k_ax: int
    ) -> np.ndarray:
        """Rotate every qdot_t by a FIXED Givens rotation in the (i_ax, k_ax) plane.

        Args:
            j: Source demo index.
            t0: Raw-frame offset within that demo.
            n_steps: Integration steps.
            theta: Rotation angle in radians.
            i_ax: First rotation axis.
            k_ax: Second rotation axis.

        Returns:
            [n_steps + 1, D] trajectory.
        """
        qdot = self._qdot_slice(j, t0, n_steps).copy()
        c, s_ = np.cos(theta), np.sin(theta)
        vi, vk = qdot[:, i_ax].copy(), qdot[:, k_ax].copy()
        qdot[:, i_ax] = c * vi - s_ * vk
        qdot[:, k_ax] = s_ * vi + c * vk
        return self._integrate(self._q_at(j, t0), qdot)

    def rollout_splice(self, j: int, t0: int, n_steps: int, j_switch: int, other: int | None) -> np.ndarray:
        """Follow j's own clean qdot for j_switch steps, then continue along other's local flow.

        Args:
            j: Source demo index.
            t0: Raw-frame offset within that demo.
            n_steps: Integration steps.
            j_switch: Step index at which the foreign flow takes over.
            other: Foreign demo to splice in. None (no sibling demo of the same task in the pool)
                degrades to following j for the whole rollout.

        Returns:
            [n_steps + 1, D] trajectory.
        """
        qdot = self._qdot_slice(j, t0, n_steps)
        flow = None if other is None else self._flow_for(other)
        if flow is None:
            return self._integrate(self._q_at(j, t0), qdot)

        traj = np.empty((n_steps + 1, self._dim))
        traj[0] = q = self._q_at(j, t0)
        # Cannot be pre-integrated like the other two modes: after the switch, each step's
        # velocity is a function of the position the previous step reached.
        for s in range(n_steps):
            step = qdot[s] if s < j_switch else flow.at(q)
            q = np.clip(q + step, self._q_lo, self._q_hi)
            traj[s + 1] = q
        return traj

    def _integrate(self, q0: np.ndarray, *terms: np.ndarray) -> np.ndarray:
        """Integrate per-step deltas, clipping to the joint limits at EVERY step.

        Args:
            q0: [D] initial configuration.
            *terms: One or more [n_steps, D] per-step delta blocks.

        Returns:
            [n_steps + 1, D] trajectory.
        """
        n_steps = len(terms[0])
        traj = np.empty((n_steps + 1, self._dim))
        traj[0] = q = q0
        # Clipping does not commute with cumsum, so integration stays sequential with a clip at
        # every step rather than summing terms then clipping once.
        for s in range(n_steps):
            for term in terms:
                q = q + term[s]
            q = np.clip(q, self._q_lo, self._q_hi)
            traj[s + 1] = q
        return traj

    # ---- helpers --------------------------------------------------------------------------------

    def demo_len(self, j: int) -> int:
        """Number of raw frames in demo j."""
        return int(self._raw_ptr[j + 1] - self._raw_ptr[j])

    def _q_at(self, j: int, t0: int) -> np.ndarray:
        """[D] raw configuration at frame t0 of demo j, as a fresh array."""
        return self._q_raw[int(self._raw_ptr[j]) + t0].copy()

    def _qdot_slice(self, j: int, t0: int, n_steps: int) -> np.ndarray:
        """[n_steps, D] raw velocities from frame t0 of demo j -- a VIEW, never copied."""
        base = int(self._raw_ptr[j]) + t0
        return self._qdot_raw[base : base + n_steps]

    def _flow_for(self, demo: int) -> _DemoVelocity:
        """The cached velocity lookup over one demo's raw frames, built on first request.

        Args:
            demo: Demo index to build the lookup over.

        Returns:
            That demo's _DemoVelocity.
        """
        flow = self._flow_cache.get(demo)
        if flow is None:
            base, end = int(self._raw_ptr[demo]), int(self._raw_ptr[demo + 1])
            flow = _DemoVelocity(
                self._q_raw[base:end], self._qdot_raw[base:end],
                k=self.cfg.k_flow, bandwidth=self.cfg.kernel_bw, beta=self.cfg.flow_beta,
            )
            self._flow_cache[demo] = flow
        return flow

    def pick_splice_owner(self, j: int, pool: np.ndarray) -> int | None:
        """A demo != j sharing its task_id, drawn from pool -- so splicing never crosses a task.

        Args:
            j: Source demo index.
            pool: Candidate demo indices.

        Returns:
            A sibling demo index, or None when j's task has no other demo in pool (single-demo
            task -- rollout_splice then degrades to a clean replay).
        """
        task_j = self.demo_task[j]
        cand = [int(d) for d in pool if int(d) != j and self.demo_task[int(d)] == task_j]
        if not cand:
            return None
        return cand[int(self.rng.randint(len(cand)))]
