"""Panda kinematics: end-effector poses from joint angles, and the live-robot accessors beside them.

Both sides of every difference robot_ood.core.action takes go through pose(), so the per-scene robot
base translation (pure translation, frame-invariant) cancels exactly. pytorch_kinematics is imported
lazily: it is not in onf-core, which must stay importable. The simulator accessors at the bottom are
the only mujoco/robosuite-coupled functions in core; a real arm supplies its own equivalents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from robot_ood.core.action import matrix_to_rotvec

__all__ = [
    "matrix_to_ee_rotvec",
    "PandaKinematics",
    "ProxyKinematics",
    "kinematics_for",
    "is_joint_space",
    "ARM_JOINTS",
    "sim",
    "robot",
    "q_arm",
    "set_nullspace_home",
]

# The chain tip; panda_hand compares directly against obs/ee_ori.
END_LINK: Final[str] = "panda_hand"

# Metres from panda_hand to the grip site along the hand frame's own z.
TOOL_OFFSET: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0965)

URDF_ENV_VAR: Final[str] = "ROBOT_OOD_PANDA_URDF"

_CALVIN_URDF: Final[Path] = (
    Path(__file__).resolve().parents[3] / "stablevla/calvin/calvin_env/data/franka_panda/panda.urdf"
)


# Vendored in-repo, 10 KB of XML, so the light onf-core env (no robosuite, no simulator) can run FK
# and the test suite. Provenance and license: ../../../UPSTREAM.md.
_VENDORED_URDF: Final[Path] = Path(__file__).resolve().parents[3] / "assets/panda_arm_hand.urdf"


def _robosuite_urdf() -> Path | None:
    """robosuite's own Panda description, if robosuite is importable.

    Returns:
        The path, or None when robosuite is not installed (onf-core does not have it).
    """
    try:
        import robosuite
    except ImportError:
        return None
    path = (Path(robosuite.__file__).parent / "models/assets/bullet_data/panda_description/urdf"
            / "panda_arm_hand.urdf")
    return path if path.is_file() else None


def default_urdf() -> Path:
    """The Panda description to use when neither ROBOT_OOD_PANDA_URDF nor an explicit urdf= is given.

    Returns:
        The CALVIN copy when it is on disk, then robosuite's, then the vendored one. The CALVIN
        path is still returned when none exists, so the FileNotFoundError names where this looked
        originally.
    """
    if _CALVIN_URDF.is_file():
        return _CALVIN_URDF
    robosuite = _robosuite_urdf()
    if robosuite is not None:
        return robosuite
    return _VENDORED_URDF if _VENDORED_URDF.is_file() else _CALVIN_URDF

_PK_MISSING = (
    "pytorch_kinematics is not installed in this interpreter. It is a TRAINING-ONLY dependency "
    "(the deploy path reads the end-effector pose off the robot), so onf-core does not ship it. "
    "Run Stage 2 training under the stablevla env (see env.sh / docs/SETUP.md)."
)

_EPS = 1e-12


def matrix_to_ee_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Axis-angle from rotation matrices with angle in [0, 2*pi] for posterior averaging, avoiding wrapped [0, pi] branch cut.

    Args:
        matrix: [..., 3, 3] rotation matrices.

    Returns:
        [..., 3] float64 axis-angle vectors with angle in [0, 2*pi].
    """
    # Unwrap afterwards: negating the axis and replacing theta by 2*pi - theta is the same rotation.
    wrapped = matrix_to_rotvec(matrix)
    theta = np.linalg.norm(wrapped, axis=-1)
    axis = wrapped / np.maximum(theta, _EPS)[..., None]
    flip = axis[..., 0] < 0.0
    theta = np.where(flip, 2.0 * np.pi - theta, theta)
    axis = np.where(flip[..., None], -axis, axis)
    return axis * theta[..., None]


@dataclass(frozen=True, eq=False)
class FlangePose:
    """One batch of configurations' tool poses in the ROBOT BASE frame.

    Attributes:
        position: [N, 3] float64 grip-site position, tool offset already applied.
        rotation: [N, 3, 3] float64 hand-frame orientation.
    """

    position: np.ndarray
    rotation: np.ndarray

    def __len__(self) -> int:
        return int(self.position.shape[0])


@dataclass(eq=False)
class PandaKinematics:
    """Forward kinematics of the 7-DoF Panda the demonstrations were recorded on.

    Attributes:
        urdf: Robot description; None resolves ROBOT_OOD_PANDA_URDF, then default_urdf().
        end_link: Chain tip.
        tool_offset: Grip-site offset in the end_link frame, metres.
    """

    urdf: str | os.PathLike[str] | None = None
    end_link: str = END_LINK
    tool_offset: tuple[float, float, float] = TOOL_OFFSET
    _chain: Any = field(default=None, init=False, repr=False)

    @property
    def urdf_path(self) -> Path:
        """The robot description this instance will build its chain from.

        Returns:
            The resolved path.

        Raises:
            FileNotFoundError: Nothing is at the resolved path.
        """
        if self.urdf is not None:
            path = Path(self.urdf)
        else:
            env = os.environ.get(URDF_ENV_VAR)
            path = Path(env) if env else default_urdf()
        if not path.is_file():
            raise FileNotFoundError(
                f"panda.urdf not found at {path}. Point {URDF_ENV_VAR} at one, or pass urdf=."
            )
        return path

    @property
    def chain(self) -> Any:
        """The lazily built, cached pytorch_kinematics serial chain.

        Returns:
            The chain, tip end_link.

        Raises:
            ImportError: pytorch_kinematics is not installed. There is deliberately no fallback:
                a silently approximated end-effector pose would train the blend against the wrong
                target and report nothing.
        """
        if self._chain is None:
            try:
                import pytorch_kinematics as pk
            except ImportError as exc:
                raise ImportError(_PK_MISSING) from exc
            self._chain = pk.build_serial_chain_from_urdf(
                self.urdf_path.read_bytes(), self.end_link
            )
        return self._chain

    def chain_on(self, device: "Any") -> "Any":
        """The cached chain, moved onto device if it is not already there.

        Args:
            device: The torch device the next FK call's tensors live on.

        Returns:
            The chain, resident on device.
        """
        chain = self.chain
        if str(getattr(chain, "device", "cpu")) != str(device):
            self._chain = chain.to(device=device)
        return self._chain

    def flange(self, q: np.ndarray) -> FlangePose:
        """FK a batch of joint configurations.

        Args:
            q: [N, 7] joint positions, radians.

        Returns:
            The base-frame grip-site poses.

        Raises:
            ValueError: q is not [N, 7].
        """
        import torch

        cfg = np.asarray(q, dtype=np.float64)
        if cfg.ndim != 2 or cfg.shape[1] != 7:
            raise ValueError(f"q must be [N, 7] Panda joint positions, got shape {cfg.shape}")
        # float32 specifically: the chain's own buffers are float32 and forward_kinematics raises
        # on a float64 input rather than casting.
        matrices = (
            self.chain_on("cpu").forward_kinematics(torch.as_tensor(cfg, dtype=torch.float32))
            .get_matrix()
            .double()
            .numpy()
        )
        rotation = matrices[:, :3, :3]
        offset = np.asarray(self.tool_offset, dtype=np.float64)
        return FlangePose(
            position=matrices[:, :3, 3] + np.einsum("nij,j->ni", rotation, offset),
            rotation=rotation,
        )

    def pose(self, q: np.ndarray) -> np.ndarray:
        """Base-frame end-effector poses for a stack of joint configurations.

        Args:
            q: [N, 7] joint positions, radians.

        Returns:
            [N, 6] float64 concat(position, axis-angle orientation), the layout of
            robot_ood.data.nodes.NodeTable.ee_base and of an action command's dims 0:6.

        Raises:
            ValueError: q is not [N, 7].
        """
        flange = self.flange(q)
        return np.concatenate(
            [flange.position, matrix_to_ee_rotvec(flange.rotation)], axis=1
        )


# ==== live robot: the joint config and posture target a robosuite env exposes ====

ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]   # LIBERO / Panda 7-DoF


def sim(env):
    """The mujoco sim behind env, unwrapping one layer of wrapper."""
    return env.sim if hasattr(env, "sim") else env.env.sim


def robot(env):
    """The first robot in env, unwrapping one layer of wrapper."""
    return (env if hasattr(env, "robots") else env.env).robots[0]


class ProxyKinematics:
    """Kinematics of a proxy configuration whose q[:6] already is the base-frame pose."""

    def pose(self, q: np.ndarray) -> np.ndarray:
        """Base-frame end-effector poses read straight off the configuration.

        Args:
            q: [N, 7] proxy configurations, concat(position, axis-angle, gripper).

        Returns:
            [N, 6] float64 concat(position, axis-angle orientation).
        """
        return np.asarray(q, dtype=np.float64)[:, :6]


def kinematics_for(suite: str) -> "PandaKinematics | ProxyKinematics":
    """ProxyKinematics when configs/trees/<suite>.yaml sets ee_from_obs, else PandaKinematics."""
    from robot_ood.core.paths import suite_config

    if bool(suite_config(suite).get("ee_from_obs")):
        return ProxyKinematics()
    return PandaKinematics()



def is_joint_space(suite: str) -> bool:
    """True when the corpus is genuine joint angles, not a pose read off the observation."""
    from robot_ood.core.paths import suite_config

    return not bool(suite_config(suite).get("ee_from_obs"))


def entry_walks(suite: str) -> bool:
    """True when the corpus rows are joints the entry corrector can command: a joint-space robot, or one arm of one."""
    from robot_ood.core.paths import suite_config

    return is_joint_space(suite) or bool(suite_config(suite).get("arm"))



def q_arm(env) -> np.ndarray:
    """The arm's [7] joint positions read off the live sim, radians; onf_q_arm() overrides."""
    if hasattr(env, "onf_q_arm"):
        return np.asarray(env.onf_q_arm(), dtype=np.float64)
    s = sim(env)
    return np.array([float(s.data.qpos[s.model.get_joint_qpos_addr(n)]) for n in ARM_JOINTS])


def set_nullspace_home(env, q_home: np.ndarray) -> None:
    """Point the OSC null-space posture target at q_home (deployable posture task)."""
    if hasattr(env, "onf_q_arm"):    # no OSC null-space outside robosuite
        return
    ctrl = robot(env).controller
    ctrl = ctrl["right"] if isinstance(ctrl, dict) else ctrl
    ctrl.update_initial_joints(np.asarray(q_home, np.float64))


# ==== ALOHA-AgileX (arx5) arms: the Cartesian view of a RoboTwin joint-space lane ====

# RoboTwin's own embodiment description, cloned by evals/robotwin/setup.sh.
_ARX5_URDF: Final[Path] = (
    Path(__file__).resolve().parents[3]
    / "external/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
)
_ARX5_TIP: Final[dict[str, str]] = {"left": "fl_link6", "right": "fr_link6"}


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw to a rotation matrix (Rz Ry Rx)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


class Arx5Kinematics:
    """Tip pose and pose Jacobian of one arx5 arm from its six joint angles, batched in torch.

    A RoboTwin lane keeps its corpus in joint space (ProxyKinematics), where every demo's grasp
    lands somewhere else because the object did. This chain gives the same configurations a
    Cartesian reading, in which the demos agree on what the scene does not move: the height of a
    grasp above the table, the lift, the place. The URDF is read directly (fixed and revolute
    joints only), so neither training nor the eval client needs a kinematics package.

    Args:
        arm: left or right.
        urdf: The arx5 description; RoboTwin's own by default.
    """

    def __init__(self, arm: str, urdf: Path | None = None) -> None:
        if arm not in _ARX5_TIP:
            raise ValueError(f"arm must be one of {sorted(_ARX5_TIP)}, got {arm!r}")
        self.arm = arm
        self.urdf = Path(urdf) if urdf is not None else _ARX5_URDF
        self._chain: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] | None = None

    def _load(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
        """The joints from the root to the tip: (origin xyz, origin rotation, revolute axis or None)."""
        if self._chain is None:
            import xml.etree.ElementTree as ET

            root = ET.parse(self.urdf).getroot()
            by_child = {j.find("child").get("link"): j for j in root.iter("joint")}
            link, joints = _ARX5_TIP[self.arm], []
            while link in by_child:
                joints.append(by_child[link])
                link = by_child[link].find("parent").get("link")
            chain = []
            for j in reversed(joints):
                origin = j.find("origin")
                xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
                rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()])
                axis = None
                if j.get("type") == "revolute":
                    node = j.find("axis")
                    axis = np.array([float(v) for v in (node.get("xyz") if node is not None else "1 0 0").split()])
                    axis = axis / max(np.linalg.norm(axis), 1e-12)
                elif j.get("type") != "fixed":
                    raise ValueError(f"{self.urdf}: joint {j.get('name')} of type {j.get('type')} is not supported")
                chain.append((xyz, _rpy_matrix(rpy), axis))
            if sum(a is not None for _, _, a in chain) != 6:
                raise ValueError(f"{self.urdf}: expected six revolute joints to {_ARX5_TIP[self.arm]}")
            self._chain = chain
        return self._chain

    def pose(self, q: "torch.Tensor") -> "torch.Tensor":
        """[..., 6] tip pose, concat(position, axis-angle), for [..., >=6] joint configurations.

        Runs on q's device and dtype, batched over every leading axis, differentiable.
        """
        import torch

        from robot_ood.core.rotation import so3_exp, so3_log

        th = q[..., :6]
        lead = th.shape[:-1]
        rot = torch.eye(3, dtype=th.dtype, device=th.device).expand(*lead, 3, 3)
        pos = torch.zeros(*lead, 3, dtype=th.dtype, device=th.device)
        k = 0
        for xyz, r_origin, axis in self._load():
            pos = pos + (rot @ torch.as_tensor(xyz, dtype=th.dtype, device=th.device))
            rot = rot @ torch.as_tensor(r_origin, dtype=th.dtype, device=th.device)
            if axis is not None:
                rotvec = torch.as_tensor(axis, dtype=th.dtype, device=th.device) * th[..., k : k + 1]
                rot = rot @ so3_exp(rotvec)
                k += 1
        return torch.cat([pos, so3_log(rot)], dim=-1)

    def jacobian(self, q: "torch.Tensor", eps: float = 1e-4) -> "torch.Tensor":
        """[B, 6, 6] d(pose)/d(joints) at [B, >=6] joint configurations, by central differences.

        The rotation rows are the derivative of the axis-angle reading, the coordinates the
        corpus's pose block and track_chunk use on a joint-space lane.
        """
        import torch

        q0 = q[..., :6]
        probe = torch.eye(6, dtype=q0.dtype, device=q0.device) * eps                    # [6, 6]
        plus = self.pose(q0[:, None, :] + probe)                                          # [B, 6, 6]
        minus = self.pose(q0[:, None, :] - probe)
        return ((plus - minus) / (2.0 * eps)).transpose(-1, -2)                           # [B, pose, joint]


def cartesian_for(suite: str) -> "Arx5Kinematics | None":
    """The Cartesian chain of a joint-space lane, or None when the suite's poses already are Cartesian."""
    from robot_ood.core.paths import suite_config

    cfg = suite_config(suite)
    arm = cfg.get("arm")
    if not (bool(cfg.get("ee_from_obs")) and arm in _ARX5_TIP):
        return None
    return Arx5Kinematics(str(arm))


def lane_displacement(offset: Any, q: Any, cart: "Arx5Kinematics | None") -> Any:
    """A Cartesian displacement in a lane's own pose coordinates.

    Args:
        offset: [B, 6] torch, metres and radians (a rotation vector).
        q: [B, >=6] torch, the pose the displacement is taken at; joint angles on a joint-space lane.
        cart: The lane's Arx5Kinematics, or None on a Cartesian lane.

    Returns:
        [B, 6] the displacement to add to the lane's pose rows: offset itself on a Cartesian lane,
        the damped least-squares joint step realising it on a joint-space lane.
    """
    import torch

    if cart is None:
        return offset
    jac = cart.jacobian(q[..., :6].to(torch.float64))
    eye = torch.eye(jac.shape[-1], device=jac.device, dtype=jac.dtype)
    gain = jac.transpose(-1, -2) @ torch.linalg.inv(jac @ jac.transpose(-1, -2) + 1e-4 * eye)
    return (gain @ offset.to(jac.dtype)[..., None])[..., 0].to(offset.dtype)
