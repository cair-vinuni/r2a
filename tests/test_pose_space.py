"""A corpus whose dims 3:6 are coordinates differences them by subtraction, end to end."""

from __future__ import annotations

import numpy as np
import pytest

from robot_ood.core.absolute import deltas_from_targets, targets_from_deltas
from robot_ood.core.action import EUCLIDEAN, SO3, ActionScale, DemoDeltas, relative_rotation
from robot_ood.runtime.ee_track import EeTrack


def test_relative_rotation_subtracts_under_euclidean_and_composes_under_so3():
    a, b = np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, 0.5])
    assert np.allclose(relative_rotation(a, b, EUCLIDEAN), [0, 0, 2.5])
    # composing two z-rotations wraps past pi: 3.0 - 0.5 = 2.5 rad about z, not a subtraction of coordinates
    assert np.allclose(relative_rotation(a, b, SO3), [0, 0, 2.5])
    assert np.allclose(relative_rotation(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -3.0]), EUCLIDEAN), [0, 0, 6.0])
    assert not np.allclose(relative_rotation(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -3.0]), SO3), [0, 0, 6.0])
    with pytest.raises(ValueError):
        relative_rotation(a, b, "polar")


def test_joint_targets_and_steps_invert_each_other_euclidean():
    rng = np.random.default_rng(0)
    q_now = rng.uniform(-3, 3, 6)
    targets = rng.uniform(-3, 3, (50, 6))
    steps = deltas_from_targets(q_now, targets, EUCLIDEAN)
    assert np.allclose(steps[0], targets[0] - q_now) and np.allclose(steps[1:], np.diff(targets, axis=0))
    assert np.allclose(targets_from_deltas(q_now, steps, EUCLIDEAN), targets)


def test_demo_deltas_and_the_scale_carry_the_pose_space():
    rng = np.random.default_rng(1)
    joints = np.cumsum(rng.uniform(-0.05, 0.05, (40, 6)), axis=0) + 2.0    # far from the origin
    actions = np.zeros((40, 7)); actions[:-1, :6] = np.diff(joints, axis=0)
    deltas = DemoDeltas.from_demo(joints[:, :3], joints[:, 3:], actions, EUCLIDEAN)
    assert np.allclose(deltas.delta_rot, deltas.action_rot[:, 0])          # tap 0 is the step just taken
    scale = ActionScale.fit(joints[:, :3], joints[:, 3:], actions, EUCLIDEAN)
    assert scale.pose_space == EUCLIDEAN and abs(scale.rot - 1.0) < 1e-6 and abs(scale.pos - 1.0) < 1e-6
    assert ActionScale.from_dict(scale.to_dict()).pose_space == EUCLIDEAN
    assert ActionScale.from_dict({"pos": 1, "rot": 1, "r2_pos": 1, "r2_rot": 1}).pose_space == SO3


def test_the_end_effector_track_subtracts_under_euclidean():
    scale = ActionScale(pos=1.0, rot=1.0, r2_pos=1.0, r2_rot=1.0, pose_space=EUCLIDEAN)
    ref = np.array([[0.0, 0, 0, 0, 0, 3.0], [0.0, 0, 0, 0, 0, 3.1]])
    now = np.array([0.0, 0, 0, 0, 0, -3.0])
    track = EeTrack(scale).chunk(ref, now)
    assert np.allclose(track[0, 3:], [0, 0, 6.0]) and np.allclose(track[1, 3:], [0, 0, 0.1])


def test_the_scale_carries_a_corpus_calibrated_limit():
    """`limit` bounds the reference's first row in the corpus's own units: the 99th percentile
    of the recorded command magnitudes, the default ACTION_LIMIT for a scale fitted without it."""
    from robot_ood.core.action import ACTION_LIMIT, RESPONSE_TAPS, action_limit

    rng = np.random.RandomState(0)
    m = 500
    command = rng.randn(m, 3) * 0.01
    deltas = DemoDeltas(
        action_pos=np.repeat(command[:, None, :], RESPONSE_TAPS, axis=1),
        action_rot=np.zeros((m, RESPONSE_TAPS, 3)),
        delta_pos=command, delta_rot=np.zeros((m, 3)),
    )
    limit = action_limit(deltas)
    assert np.isclose(limit, np.percentile(np.abs(command), 99, axis=0).max())
    assert ActionScale.from_deltas(deltas, EUCLIDEAN).limit == limit
    plain = ActionScale(pos=1.0, rot=1.0, r2_pos=1.0, r2_rot=1.0)
    assert plain.limit == ACTION_LIMIT
    assert ActionScale.from_dict({"pos": 1, "rot": 1, "r2_pos": 1, "r2_rot": 1}).limit == ACTION_LIMIT
    fitted = ActionScale(pos=1.0, rot=1.0, r2_pos=1.0, r2_rot=1.0, limit=0.11)
    assert ActionScale.from_dict(fitted.to_dict()).limit == 0.11
