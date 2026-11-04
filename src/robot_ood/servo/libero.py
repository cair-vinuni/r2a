"""The LIBERO side of the servo: place the base in the simulator, and read the policy server's features."""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from robot_ood.core import kinematics

# One settling step after the move, arm still, gripper open: the observation the policy starts from.
HOLD_ACTION: Final[np.ndarray] = np.array([0.0] * 6 + [-1.0], np.float32)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """A robosuite xyzw quaternion as an axis-angle vector."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-9:
        return np.zeros(3)
    return (np.asarray(quat[:3]) * 2.0 * np.arccos(w)) / den


def policy_inputs(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The two 180-degree-rotated views and the proprio the LIBERO client sends the policy."""
    view = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"])).astype(np.float32)
    return view, wrist, state


class FeatureClient:
    """Reads the policy server's pooled features for a batch of observations."""

    def __init__(self, host: str, port: int, key: str = "feat_vl") -> None:
        from tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host, port)
        self.key = key

    def embed(self, observations: list[dict], task: str) -> np.ndarray:
        """[B, F] features of B raw simulator observations under one instruction."""
        views = [policy_inputs(o) for o in observations]
        payload = {"batch_images": [[v, w] for v, w, _ in views], "instructions": [task] * len(views),
                   "state": [s[None] for _, _, s in views], "episode_step": [0] * len(views),
                   "worker_id": [0] * len(views)}
        reply = self.client.infer(payload)
        if not reply.get("ok", False):
            raise RuntimeError(f"feature request failed: {reply.get('error')}")
        keys = self.key.split("+")
        return np.concatenate([np.asarray(reply["data"][k], np.float64) for k in keys], axis=1)


class BasePlacer:
    """Puts the base at a planar offset u = (dx, dy, dyaw) from its start; the Panda has no waist, so a turn turns the base."""

    def __init__(self, env: Any) -> None:
        self.env = env
        s = kinematics.sim(env)
        self.body = s.model.body_name2id(kinematics.BASE_BODY)
        self.pos0 = s.model.body_pos[self.body].copy()
        self.quat0 = s.model.body_quat[self.body].copy()
        _, R0 = kinematics.base_pose(env)
        self.R0 = R0[:2, :2].copy()

    def __call__(self, u: np.ndarray) -> dict:
        """Relocate the base to offset u and return the observation the cameras now see."""
        s = kinematics.sim(self.env)
        s.model.body_pos[self.body] = self.pos0
        s.model.body_quat[self.body] = self.quat0
        kinematics.shift_base(self.env, np.array([*(self.R0 @ np.asarray(u[:2])), float(u[2])]))
        return self.env.env._get_observations(force_update=True)


def settle(env: Any) -> dict:
    obs, *_ = env.step(HOLD_ACTION)
    return obs
