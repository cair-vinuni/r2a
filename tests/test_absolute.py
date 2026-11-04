"""Absolute targets and sequential deltas are two spellings of one chunk."""

from __future__ import annotations

import numpy as np

from robot_ood.core.absolute import (deltas_from_targets, quat_wxyz_to_rotvec, rotvec_to_quat_wxyz,
                                     targets_from_deltas)
from robot_ood.core.action import rotvec_to_matrix


def _random_rotvecs(rng, n):
    axis = rng.standard_normal((n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return axis * rng.uniform(0.0, 2.5, size=(n, 1))


def test_quaternion_round_trip_keeps_the_rotation():
    rng = np.random.default_rng(0)
    rv = _random_rotvecs(rng, 32)
    back = quat_wxyz_to_rotvec(rotvec_to_quat_wxyz(rv))
    assert np.allclose(rotvec_to_matrix(back), rotvec_to_matrix(rv), atol=1e-9)
    assert np.all(rotvec_to_quat_wxyz(rv)[:, 0] >= 0)


def test_identity_quaternion_is_the_zero_rotation():
    assert np.allclose(quat_wxyz_to_rotvec(np.array([1.0, 0, 0, 0])), 0.0)
    assert np.allclose(rotvec_to_quat_wxyz(np.zeros(3)), [1, 0, 0, 0])


def test_targets_and_deltas_invert_each_other():
    rng = np.random.default_rng(1)
    ee_now = np.concatenate([rng.standard_normal(3), _random_rotvecs(rng, 1)[0]])
    targets = np.concatenate([rng.standard_normal((30, 3)), _random_rotvecs(rng, 30)], axis=1)
    deltas = deltas_from_targets(ee_now, targets)
    back = targets_from_deltas(ee_now, deltas)
    assert np.allclose(back[:, :3], targets[:, :3], atol=1e-9)
    assert np.allclose(rotvec_to_matrix(back[:, 3:]), rotvec_to_matrix(targets[:, 3:]), atol=1e-8)


def test_the_first_delta_is_measured_from_the_live_pose():
    ee_now = np.array([0.1, 0.2, 0.3, 0, 0, 0.5])
    targets = np.array([[0.1, 0.2, 0.4, 0, 0, 0.5], [0.1, 0.3, 0.4, 0, 0, 0.5]])
    deltas = deltas_from_targets(ee_now, targets)
    assert np.allclose(deltas, [[0, 0, 0.1, 0, 0, 0], [0, 0.1, 0, 0, 0, 0]])
