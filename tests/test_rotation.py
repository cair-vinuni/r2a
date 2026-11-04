"""Tests for robot_ood.core.rotation, the differentiable twin of robot_ood.core.action's SO(3).

The regime that matters is theta near pi. LIBERO parks the wrist there, and since the pursuit chunk
rolls a virtual pose forward it takes the log of ABSOLUTE orientations, not just of the small
relative rotations the old feedforward construction produced. The arccos(trace) form these
functions used divides by sin(theta) and returned 2.22 rad for an angle of pi - 1e-3.

scipy is the independent reference, as in tests/test_blend.py; the modules under test do not use it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from robot_ood.core.action import matrix_to_rotvec, rotvec_to_matrix
from robot_ood.core.rotation import so3_exp, so3_log

# Every hazard in one fixture: random rotations, the exact identity, an angle far below the
# small-angle threshold, four approaches to pi, and one generic large angle.
_AXIS = np.array([1.0, 0.0, 0.0])
_HAZARDS = np.array([
    [0.0, 0.0, 0.0],
    [1e-9, 0.0, 0.0],
    [np.pi - 1e-3, 0.0, 0.0],
    [np.pi - 1e-7, 0.0, 0.0],
    [np.pi - 1e-12, 0.0, 0.0],
    [3.0, 0.5, 0.2],
])


def _fixture() -> np.ndarray:
    return np.concatenate([Rotation.random(512, random_state=0).as_rotvec(), _HAZARDS])


def test_so3_log_matches_scipy_and_the_numpy_twin_including_at_pi():
    matrices = rotvec_to_matrix(_fixture())
    got = so3_log(torch.as_tensor(matrices)).numpy()
    np.testing.assert_allclose(got, Rotation.from_matrix(matrices).as_rotvec(), atol=1e-12)
    np.testing.assert_allclose(got, matrix_to_rotvec(matrices), atol=1e-12)


def test_so3_exp_matches_scipy():
    rotvecs = _fixture()
    np.testing.assert_allclose(
        so3_exp(torch.as_tensor(rotvecs)).numpy(),
        Rotation.from_rotvec(rotvecs).as_matrix(),
        atol=1e-12,
    )


def test_so3_roundtrip_is_the_identity():
    rotvecs = _fixture()
    np.testing.assert_allclose(so3_log(so3_exp(torch.as_tensor(rotvecs))).numpy(), rotvecs, atol=1e-12)


@pytest.mark.parametrize(
    "rotvec", [np.zeros(3), np.array([1e-9, 0.0, 0.0]), np.array([np.pi - 1e-6, 0.0, 0.0]),
               np.array([0.4, -0.2, 0.1])],
)
def test_the_roundtrip_gradient_is_finite_at_the_identity_and_at_pi(rotvec):
    """The failure this guards is backward-only: a finite loss with an inf gradient poisons the
    parameters and surfaces later as a device assert inside an unrelated term."""
    x = torch.tensor(rotvec, requires_grad=True)
    so3_log(so3_exp(x)).sum().backward()
    assert torch.isfinite(x.grad).all()
    np.testing.assert_allclose(x.grad.numpy(), np.ones(3), atol=1e-6)
