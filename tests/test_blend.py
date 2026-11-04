"""Tests for robot_ood.core.action and robot_ood.runtime.ee_track — the SO(3) helpers, the tracking chunk, and the blend.

Deliberately small: each case guards one failure mode that is silent in a rollout (a rotation
delta subtracted instead of composed, a feedback row copied into the feedforward rows, an
interpolated gripper command). scipy is only imported here, as the independent SO(3) reference --
the module under test is numpy-only on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_ood.core.action import ActionScale, matrix_to_rotvec, relative_rotvec, rotvec_to_matrix
from robot_ood.runtime.ee_track import EeTrack

SCALE = ActionScale(pos=0.0122, rot=0.107, r2_pos=0.95, r2_rot=0.84)


def _random_rotvecs(n: int, seed: int) -> np.ndarray:
    return Rotation.random(n, random_state=seed).as_rotvec()


def test_relative_rotvec_matches_scipy_including_near_pi():
    a, b = _random_rotvecs(64, 0), _random_rotvecs(64, 1)
    # Near-pi pairs: the log map's ill-conditioned branch, and where LIBERO's ee_ori actually lives.
    axis = np.array([1.0, 0.0, 0.0])
    near_pi = np.stack([axis * (np.pi - 1e-7), axis * (np.pi - 1e-9), -axis * (np.pi - 1e-8)])
    a = np.concatenate([a, near_pi, np.zeros((1, 3))])
    b = np.concatenate([b, near_pi[::-1], np.zeros((1, 3))])
    expect = (Rotation.from_rotvec(a) * Rotation.from_rotvec(b).inv()).as_rotvec()
    np.testing.assert_allclose(relative_rotvec(a, b), expect, atol=1e-9)


def test_rotvec_matrix_roundtrip():
    rotvecs = np.concatenate([
        _random_rotvecs(64, 2),
        np.zeros((1, 3)),
        np.array([[1e-9, 0.0, 0.0], [0.0, np.pi - 1e-8, 0.0]]),
    ])
    np.testing.assert_allclose(matrix_to_rotvec(rotvec_to_matrix(rotvecs)), rotvecs, atol=1e-9)
    np.testing.assert_allclose(rotvec_to_matrix(rotvecs), Rotation.from_rotvec(rotvecs).as_matrix(), atol=1e-12)


def test_row_zero_is_feedback_and_later_rows_are_feedforward():
    ref = np.zeros((4, 6))
    ref[:, 0] = np.arange(4) * 0.01
    ref[:, 3] = np.pi - 0.05 * np.arange(4)
    track = EeTrack(SCALE).chunk(ref, ref[0].copy())
    # Row 0 measures against the live pose, which is exactly ref[0] here; the rest measure against
    # their own predecessor and must carry the demo's motion.
    np.testing.assert_allclose(track[0], np.zeros(6), atol=1e-12)
    assert np.abs(track[1:, 0]).min() > 1e-6
    assert np.abs(track[1:, 3]).min() > 1e-6


def test_track_offset_does_not_scale_with_chunk_length():
    offset = np.array([0.05, -0.02, 0.03, 0.0, 0.0, 0.0])
    now = np.array([0.1, 0.2, 0.3, np.pi - 0.1, 0.02, -0.03])
    tracker = EeTrack(SCALE)
    chunks = [tracker.chunk(np.tile(now + offset, (k, 1)), now) for k in (1, 8, 32)]
    for chunk in chunks:
        np.testing.assert_allclose(chunk[0], chunks[0][0], atol=1e-12)
        # A constant reference is a standing offset: only row 0 may command it, or a K-step chunk
        # would drive K times the offset.
        np.testing.assert_allclose(chunk[1:], 0.0, atol=1e-12)
    total = np.stack([c.sum(axis=0) for c in chunks])
    np.testing.assert_allclose(total, np.broadcast_to(total[0], total.shape), atol=1e-12)


def test_action_scale_fit_recovers_a_known_scale():
    rng = np.random.RandomState(4)
    actions = np.zeros((200, 7))
    actions[:, :6] = rng.uniform(-1.0, 1.0, size=(200, 6))
    ee_pos = np.zeros((201, 3))
    ee_ori = np.zeros((201, 3))
    ee_ori[0] = np.array([np.pi - 0.2, 0.05, -0.05])
    for t in range(200):
        ee_pos[t + 1] = ee_pos[t] + 0.0122 * actions[t, :3]
        step = rotvec_to_matrix(0.107 * actions[t, 3:6])
        ee_ori[t + 1] = matrix_to_rotvec(step @ rotvec_to_matrix(ee_ori[t]))
    fit = ActionScale.fit(ee_pos, ee_ori, np.concatenate([actions, actions[-1:]]))
    assert fit.pos == pytest.approx(0.0122, rel=1e-6)
    assert fit.rot == pytest.approx(0.107, rel=1e-6)
    assert fit.r2_pos > 0.999999 and fit.r2_rot > 0.999999
    fit.check()


def test_action_scale_fit_reads_a_one_step_lagged_controller_by_its_dc_gain():
    rng = np.random.RandomState(5)
    actions = np.zeros((300, 7))
    actions[:, :6] = rng.uniform(-1.0, 1.0, size=(300, 6))
    ee_pos = np.zeros((301, 3))
    ee_ori = np.zeros((301, 3))
    for t in range(300):
        prev = actions[t - 1] if t else np.zeros(7)
        ee_pos[t + 1] = ee_pos[t] + 0.05 * actions[t, :3] + 0.25 * prev[:3]
        step = rotvec_to_matrix(0.04 * actions[t, 3:6] + 0.2 * prev[3:6])
        ee_ori[t + 1] = matrix_to_rotvec(step @ rotvec_to_matrix(ee_ori[t]))
    fit = ActionScale.fit(ee_pos, ee_ori, np.concatenate([actions, actions[-1:]]))
    assert fit.pos == pytest.approx(0.30, rel=1e-6)
    assert fit.rot == pytest.approx(0.24, rel=1e-3)
    assert fit.r2_pos > 0.999999 and fit.r2_rot > 0.999
    fit.check()


def test_action_scale_json_roundtrip_and_gate(tmp_path):
    path = SCALE.to_json(tmp_path / "action_scale.json")
    assert ActionScale.from_json(path) == SCALE
    with pytest.raises(ValueError, match="rotation block"):
        ActionScale(pos=0.0122, rot=0.107, r2_pos=0.95, r2_rot=-0.026).check()
