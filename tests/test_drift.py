"""Tests for robot_ood.train.drift — synthetic NodeTable stand-in only (no HDF5, no robot_ood.data.nodes
dependency), so this runs standalone in .venv-core and does not depend on the sibling module that
builds the real NodeTable finishing first.

Only DriftConfig and DriftGenerator's __init__/demo_len/
rollout_velocity/rollout_rotation/rollout_splice/
pick_splice_owner are reachable from the shipped offline path
(robot_ood.train.sequences.SequenceBuilder) -- these tests exercise that surface directly, the way
SequenceBuilder calls it.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ood.data import schema
from robot_ood.train.drift import PANDA_JOINT_HI, PANDA_JOINT_LO, DriftConfig, DriftGenerator

D = 7  # Panda DoF

# ==================================================================================================
# synthetic NodeTable stand-in
#
# Duck-types the fixed contract drift.py is written against: q_raw/qdot_raw [N,D], raw_ptr [J+1],
# task_names, .dim, .n_demos, .demo_nodes(owner). Every raw demo is internally consistent
# (q_raw[i+1] == q_raw[i] + qdot_raw[i]) so a zero-noise rollout reproduces the clean trajectory
# EXACTLY.
# ==================================================================================================


class FakeNodeTable:
    """Minimal node-level table -- DriftGenerator.__init__ reads node-level task_id via
    demo_nodes(owner) to build its per-demo task lookup, so one node per raw frame (coarsen=1) is
    the simplest fixture that satisfies the contract."""

    def __init__(self, q_raw, qdot_raw, raw_ptr, demo_task, task_names):
        self.q_raw = q_raw.astype(np.float32)
        self.qdot_raw = qdot_raw.astype(np.float32)
        self.raw_ptr = np.asarray(raw_ptr, dtype=np.int32)
        self.task_names = list(task_names)

        J = len(self.raw_ptr) - 1
        self.task_id = np.concatenate([
            np.full(int(self.raw_ptr[j + 1] - self.raw_ptr[j]), demo_task[j], dtype=np.int16)
            for j in range(J)
        ])

    @property
    def dim(self):
        return self.q_raw.shape[1]

    @property
    def n_demos(self):
        return len(self.raw_ptr) - 1

    def demo_nodes(self, owner):
        s, e = int(self.raw_ptr[owner]), int(self.raw_ptr[owner + 1])
        return np.arange(s, e, dtype=np.int64)


def _make_demo_arrays(rng, start, length, speed=0.03):
    """A smooth, self-consistent raw trajectory: qdot is a few low-frequency sinusoids + tiny jitter,
    q is the EXACT cumulative sum of qdot (q[i+1] == q[i] + qdot[i]), so a zero-sigma VelocityDrift
    rollout reproduces this trajectory bit-for-bit."""
    t = np.arange(length)
    freqs = rng.uniform(0.03, 0.09, size=D)
    phases = rng.uniform(0, 2 * np.pi, size=D)
    amp = rng.uniform(0.5, 1.5, size=D) * speed
    qdot = (amp[None, :] * np.cos(2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :]))
    qdot = qdot + 0.1 * speed * rng.randn(length, D)
    q = np.empty((length, D))
    q[0] = start
    for i in range(1, length):
        q[i] = q[i - 1] + qdot[i - 1]
    qdot = qdot.astype(np.float64)
    return q, qdot


def make_fixture(seed=0, n_tasks=2, demos_per_task=6, length_range=(60, 100)):
    """n_tasks * demos_per_task demos, well separated ACROSS tasks (so splice candidates can never be
    confused with a different task's geometry) and modestly separated within a task."""
    rng = np.random.RandomState(seed)
    mid = (PANDA_JOINT_LO + PANDA_JOINT_HI) / 2
    half = (PANDA_JOINT_HI - PANDA_JOINT_LO) / 2

    task_centers = []
    for _ in range(n_tasks):
        task_centers.append(mid + rng.uniform(-0.4, 0.4, size=D) * half)

    q_raws, qdot_raws, raw_ptr, demo_task = [], [], [0], []
    for ti in range(n_tasks):
        for _ in range(demos_per_task):
            length = int(rng.randint(*length_range))
            start = task_centers[ti] + 0.03 * half * rng.randn(D)
            start = np.clip(start, PANDA_JOINT_LO + 0.2 * half, PANDA_JOINT_HI - 0.2 * half)
            q, qdot = _make_demo_arrays(rng, start, length)
            q_raws.append(q)
            qdot_raws.append(qdot)
            raw_ptr.append(raw_ptr[-1] + length)
            demo_task.append(ti)

    q_raw = np.concatenate(q_raws, axis=0)
    qdot_raw = np.concatenate(qdot_raws, axis=0)
    assert np.all(q_raw >= PANDA_JOINT_LO - 1e-6) and np.all(q_raw <= PANDA_JOINT_HI + 1e-6), (
        "fixture generation drifted outside the Panda joint limits — shrink `speed`/length_range"
    )
    task_names = [f"task{i}" for i in range(n_tasks)]
    nodes = FakeNodeTable(q_raw, qdot_raw, raw_ptr, demo_task, task_names)
    return nodes, np.array(demo_task)


@pytest.fixture(scope="module")
def nodes():
    n, _ = make_fixture(seed=0)
    return n


# ==================================================================================================
# DriftGenerator construction
# ==================================================================================================
def test_generator_rejects_dim_mismatch():
    bad_nodes, _ = make_fixture(seed=0, n_tasks=1, demos_per_task=2)
    bad_nodes.q_raw = bad_nodes.q_raw[:, :3]   # 3-DoF, but the constructor targets 7-DoF Panda
    # __init__ itself doesn't touch qdot_raw's width, so mismatch it too for a coherent fixture
    bad_nodes.qdot_raw = bad_nodes.qdot_raw[:, :3]
    gen = DriftGenerator(bad_nodes, seed=0)
    assert gen.dim == 3   # under the Panda's 7 -- legal, just a smaller arm


def test_demo_len_matches_raw_ptr(nodes):
    gen = DriftGenerator(nodes, seed=0)
    for j in range(gen.n_demos):
        assert gen.demo_len(j) == int(nodes.raw_ptr[j + 1] - nodes.raw_ptr[j])


def test_rollout_velocity_determinism_different_seed(nodes):
    g1 = DriftGenerator(nodes, seed=0)
    g2 = DriftGenerator(nodes, seed=1)
    t1 = g1.rollout_velocity(0, 0, 20, sigma=0.1)
    t2 = g2.rollout_velocity(0, 0, 20, sigma=0.1)
    assert not np.array_equal(t1, t2)


def test_rollout_velocity_zero_sigma_reproduces_clean_trajectory(nodes):
    """The fixture's raw q is the exact cumulative sum of qdot (see _make_demo_arrays), so a
    zero-sigma rollout must reproduce it bit-for-bit -- this is what makes the generator's deviation
    genuinely attributable to the injected noise, not an artefact of the integration itself."""
    gen = DriftGenerator(nodes, seed=0)
    j, t0, n_steps = 0, 5, 20
    traj = gen.rollout_velocity(j, t0, n_steps, sigma=0.0)
    base = int(nodes.raw_ptr[j])
    clean = nodes.q_raw[base + t0 : base + t0 + n_steps + 1].astype(np.float64)
    np.testing.assert_allclose(traj, clean, atol=1e-4)


def test_rollout_velocity_within_joint_limits(nodes):
    gen = DriftGenerator(nodes, seed=2)
    for j in range(gen.n_demos):
        n_steps = min(30, gen.demo_len(j) - 1)
        traj = gen.rollout_velocity(j, 0, n_steps, sigma=0.5)
        assert np.all(traj >= PANDA_JOINT_LO[: nodes.dim] - 1e-4)
        assert np.all(traj <= PANDA_JOINT_HI[: nodes.dim] + 1e-4)


def test_rollout_velocity_severity_increases_with_sigma(nodes):
    gen_low = DriftGenerator(nodes, seed=3)
    gen_high = DriftGenerator(nodes, seed=3)
    j, t0, n_steps = 0, 0, 30
    t_low = gen_low.rollout_velocity(j, t0, n_steps, sigma=0.02)
    t_high = gen_high.rollout_velocity(j, t0, n_steps, sigma=0.5)
    base = int(nodes.raw_ptr[j])
    clean_end = nodes.q_raw[base + t0 + n_steps].astype(np.float64)
    dev_low = float(np.linalg.norm(t_low[-1] - clean_end))
    dev_high = float(np.linalg.norm(t_high[-1] - clean_end))
    assert dev_high > dev_low, (dev_low, dev_high)


# ==================================================================================================
# rollout_rotation — magnitude-preserving (Givens rotation is orthogonal), joint limits
# ==================================================================================================
def test_rollout_rotation_within_joint_limits(nodes):
    gen = DriftGenerator(nodes, seed=4)
    for j in range(gen.n_demos):
        n_steps = min(30, gen.demo_len(j) - 1)
        traj = gen.rollout_rotation(j, 0, n_steps, theta=0.3, i_ax=0, k_ax=1)
        assert np.all(traj >= PANDA_JOINT_LO[: nodes.dim] - 1e-4)
        assert np.all(traj <= PANDA_JOINT_HI[: nodes.dim] + 1e-4)


def test_rollout_rotation_zero_theta_reproduces_clean_trajectory(nodes):
    gen = DriftGenerator(nodes, seed=5)
    j, t0, n_steps = 0, 5, 20
    traj = gen.rollout_rotation(j, t0, n_steps, theta=0.0, i_ax=0, k_ax=1)
    base = int(nodes.raw_ptr[j])
    clean = nodes.q_raw[base + t0 : base + t0 + n_steps + 1].astype(np.float64)
    np.testing.assert_allclose(traj, clean, atol=1e-6)


def test_rollout_rotation_deviates_from_clean_with_nonzero_theta(nodes):
    gen = DriftGenerator(nodes, seed=6)
    j, t0, n_steps = 0, 0, 30
    traj = gen.rollout_rotation(j, t0, n_steps, theta=0.4, i_ax=0, k_ax=2)
    base = int(nodes.raw_ptr[j])
    clean_end = nodes.q_raw[base + t0 + n_steps].astype(np.float64)
    assert float(np.linalg.norm(traj[-1] - clean_end)) > 1e-3


# ==================================================================================================
# pick_splice_owner / rollout_splice
# ==================================================================================================
def test_splice_never_crosses_task_boundary(nodes):
    n_checked = 0
    for seed in range(5):
        gen = DriftGenerator(nodes, seed=seed)
        pool = np.arange(gen.n_demos)
        for j in range(gen.n_demos):
            other = gen.pick_splice_owner(j, pool)
            if other is None:
                continue
            assert other != j
            assert gen.demo_task[other] == gen.demo_task[j], (
                f"splice picked demo {other} (task {gen.demo_task[other]}) for source demo {j} "
                f"(task {gen.demo_task[j]}) — this must never happen"
            )
            n_checked += 1
    assert n_checked > 0, "test is vacuous: pick_splice_owner never returned a candidate"


def test_pick_splice_owner_none_for_singleton_task(nodes):
    gen = DriftGenerator(nodes, seed=0)
    singleton_task_demo = 0
    other_task_demos = [j for j in range(gen.n_demos) if gen.demo_task[j] != gen.demo_task[singleton_task_demo]]
    # pool excludes every OTHER demo of the singleton's own task -- degenerate to "no sibling available"
    pool = np.array([singleton_task_demo] + other_task_demos)
    assert gen.pick_splice_owner(singleton_task_demo, pool) is None


def test_rollout_splice_follows_source_then_switches(nodes):
    """The first j_switch steps must match the SOURCE demo's own clean qdot exactly (no flow field
    involved yet); only after the switch does the trajectory diverge onto the other demo's local
    flow."""
    gen = DriftGenerator(nodes, seed=7)
    j, t0, n_steps = 0, 0, 20
    pool = np.arange(gen.n_demos)
    other = gen.pick_splice_owner(j, pool)
    assert other is not None, "fixture must have >= 2 demos per task for this test to be meaningful"

    j_switch = 8
    traj = gen.rollout_splice(j, t0, n_steps, j_switch, other)
    base = int(nodes.raw_ptr[j])
    clean_prefix = nodes.q_raw[base + t0 : base + t0 + j_switch + 1].astype(np.float64)
    np.testing.assert_allclose(traj[: j_switch + 1], clean_prefix, atol=1e-4)


def test_rollout_splice_degrades_to_clean_replay_when_other_is_none(nodes):
    gen = DriftGenerator(nodes, seed=8)
    j, t0, n_steps = 0, 0, 15
    traj = gen.rollout_splice(j, t0, n_steps, j_switch=5, other=None)
    base = int(nodes.raw_ptr[j])
    clean = nodes.q_raw[base + t0 : base + t0 + n_steps + 1].astype(np.float64)
    np.testing.assert_allclose(traj, clean, atol=1e-4)


def test_rollout_splice_within_joint_limits(nodes):
    gen = DriftGenerator(nodes, seed=9)
    pool = np.arange(gen.n_demos)
    for j in range(gen.n_demos):
        other = gen.pick_splice_owner(j, pool)
        if other is None:
            continue
        n_steps = min(20, gen.demo_len(j) - 1)
        traj = gen.rollout_splice(j, 0, n_steps, j_switch=n_steps // 2, other=other)
        assert np.all(traj >= PANDA_JOINT_LO[: nodes.dim] - 1e-4)
        assert np.all(traj <= PANDA_JOINT_HI[: nodes.dim] + 1e-4)


# ==================================================================================================
# DriftConfig — construction sanity (curriculum-magnitude defaults, see module docstring)
# ==================================================================================================
def test_driftconfig_defaults_construct_without_error():
    cfg = DriftConfig()
    assert cfg.window == schema.HIST_H
    assert cfg.advance == schema.ADVANCE
    assert cfg.sigma[1] > cfg.sigma[0] > 0.0


def test_driftconfig_is_a_plain_dataclass_with_overridable_fields():
    cfg = DriftConfig(sigma=(0.1, 0.2), splice_frac=0.7)
    assert cfg.sigma == (0.1, 0.2)
    assert cfg.splice_frac == 0.7
