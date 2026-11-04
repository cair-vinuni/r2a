"""Differentiable SO(3), for robot_ood.models.fusion.target's tracking chunk.

robot_ood.core.action holds the numpy twins. That duplication is deliberate and documented there: the
deploy path is numpy-only so it can run in the light onf-core env, and these have to carry gradient.
"""

from __future__ import annotations

from typing import Final

import torch

__all__ = ["so3_exp", "so3_log"]

# Below this angle, trig quotients are taken from their Taylor series instead.
_SMALL_ANGLE: Final[float] = 1e-3
_EPS: Final[float] = 1e-12



def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map.

    Args:
        rotvec: [..., 3] axis-angle rotation vectors, radians.

    Returns:
        [..., 3, 3] rotation matrices.
    """
    theta2 = (rotvec * rotvec).sum(-1)
    small = theta2 < _SMALL_ANGLE * _SMALL_ANGLE
    safe = torch.where(small, torch.ones_like(theta2), theta2).sqrt()
    sin_over = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, safe.sin() / safe)
    cos_over = torch.where(
        small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - safe.cos()) / (safe * safe)
    )
    zero = torch.zeros_like(theta2)
    x, y, z = rotvec[..., 0], rotvec[..., 1], rotvec[..., 2]
    k = torch.stack(
        [
            torch.stack([zero, -z, y], -1),
            torch.stack([z, zero, -x], -1),
            torch.stack([-y, x, zero], -1),
        ],
        -2,
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(k.shape)
    return eye + sin_over[..., None, None] * k + cos_over[..., None, None] * (k @ k)


def _matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Shepperd's rotation-matrix to quaternion conversion, batched and differentiable.

    Args:
        matrix: [..., 3, 3] rotation matrices.

    Returns:
        [..., 4] unit quaternions ordered [x, y, z, w].

    All four candidate constructions are evaluated and then selected, rather than masked-assigned
    as robot_ood.core.action's numpy twin does: the four are cheap, and evaluating them all keeps this
    differentiable and agnostic to the leading batch shape. Selecting on the largest of the three
    diagonal entries and the trace is what keeps the conversion conditioned -- the selected
    candidate's own component is 4x the largest squared quaternion component, hence >= 1, so the
    normalisation below can never divide by a small number.
    """
    m, trace = matrix, matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    candidates = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        parts = [None, None, None, m[..., k, j] - m[..., j, k]]
        parts[i] = 1.0 - trace + 2.0 * m[..., i, i]
        parts[j] = m[..., j, i] + m[..., i, j]
        parts[k] = m[..., k, i] + m[..., i, k]
        candidates.append(torch.stack(parts, -1))
    candidates.append(torch.stack([
        m[..., 2, 1] - m[..., 1, 2],
        m[..., 0, 2] - m[..., 2, 0],
        m[..., 1, 0] - m[..., 0, 1],
        1.0 + trace,
    ], -1))
    stacked = torch.stack(candidates, -2)
    decision = torch.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2], trace], -1)
    pick = decision.argmax(-1)
    quat = stacked.gather(-2, pick[..., None, None].expand(*pick.shape, 1, 4)).squeeze(-2)
    return quat / quat.norm(dim=-1, keepdim=True)


def so3_log(matrix: torch.Tensor) -> torch.Tensor:
    """Rodrigues logarithmic map, angle in [0, pi].

    Args:
        matrix: [..., 3, 3] rotation matrices.

    Returns:
        [..., 3] axis-angle rotation vectors.

    Routed through the quaternion rather than through arccos(trace) and the antisymmetric part.
    That older form divides by sin(theta), which vanishes at theta = pi, and no clamp repairs it:
    an angle of pi - 1e-3 came back as 2.22 rad against the numpy twin's 3.1406. LIBERO parks the
    wrist near pi, and this function is applied to ABSOLUTE orientations (robot_ood.models.fusion.target's
    integrate_pose rolls the virtual pose forward through one), so that regime is the normal one
    and not an edge case. sin(theta/2) below is 1 exactly where the old form was worst.
    """
    quat = _matrix_to_quat(matrix)
    quat = torch.where(quat[..., 3:4] < 0.0, -quat, quat)
    vec2 = (quat[..., :3] * quat[..., :3]).sum(-1)
    # theta ~ 2||v|| near the identity, so this is the same threshold as everywhere else here.
    small = vec2 < (0.5 * _SMALL_ANGLE) ** 2
    safe_vec = torch.where(small, torch.ones_like(vec2), vec2).sqrt()
    angle = torch.where(small, torch.ones_like(vec2),
                        2.0 * torch.atan2(safe_vec, quat[..., 3]))
    # 2 is the small-angle limit of theta / sin(theta/2); the series adds the next two terms, in
    # theta^2 ~ 4||v||^2 so that no square root is differentiated at zero.
    small2 = 4.0 * vec2
    scale = torch.where(
        small,
        2.0 + small2 / 12.0 + 7.0 * small2 * small2 / 2880.0,
        angle / torch.sin(0.5 * angle).clamp_min(_EPS),
    )
    return scale[..., None] * quat[..., :3]
