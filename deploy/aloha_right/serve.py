"""The GR00T policy server with the corrector behind get_action: python serve.py"""

from __future__ import annotations

import logging
import os
import sys

import paths

os.environ.setdefault("CUDA_VISIBLE_DEVICES", paths.GPU)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))

import numpy as np
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import PolicyWrapper
from gr00t.policy.server_client import PolicyServer

from robot_ood.runtime.aloha_deploy import ARM_DOF
from robot_ood.runtime.aloha_server import AlohaServerLoop

LANG_KEY = "annotation.human.task_description"
EMBODIMENT = "NEW_EMBODIMENT"


class CorrectedPolicy(PolicyWrapper):
    """Gr00tPolicy's wire contract, with the homing walk and the corrector between it and the node."""

    def __init__(self, policy: Gr00tPolicy, loop: AlohaServerLoop) -> None:
        super().__init__(policy, strict=False)
        self.loop = loop

    def check_observation(self, observation: dict) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict) -> None:
        self.policy.check_action(action)

    def get_modality_config(self) -> dict:
        return self.policy.get_modality_config()

    def reset(self, options: dict | None = None) -> dict:
        self.loop.reset()
        logging.info("episode reset")
        return self.policy.reset(options)

    def _get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        joints = np.asarray(observation["state"]["single_arm"], dtype=np.float64).reshape(-1, ARM_DOF)
        grip = np.asarray(observation["state"]["gripper"], dtype=np.float64).reshape(-1)
        if len(joints) != 1:
            raise ValueError(f"one arm per request, got a batch of {len(joints)}")
        task = str(observation["language"][LANG_KEY][0][0])
        info: dict = {}

        def chunk() -> np.ndarray:
            action, extra = self.policy.get_action(observation, options)
            info.update(extra)
            arm = np.asarray(action["single_arm"], dtype=np.float64).reshape(-1, ARM_DOF)
            g = np.asarray(action["gripper"], dtype=np.float64).reshape(-1, 1)
            return np.concatenate([arm, g], axis=-1)

        out = self.loop.step(joints[0], grip[0], task, chunk)
        if paths.LOG_CORRECTION:
            if self.loop.homing_leg:
                r = self.loop.homing.report(joints[0], self.loop.grip_cmd)
                logging.info("HOMING pass %d steps %d residual %.4f (radius %.4f)",
                             r["passes"], r["steps"], r["residual"], r["level_hat"])
            elif self.loop.corrector.decision:
                d = self.loop.corrector.decision
                logging.info("CORRECTOR |dq|max %.5f rad  trust %.3f shape %.3f",
                             self.loop.moved, d["trust"], d["shape"])
            else:
                logging.info("CORRECTOR pass-through")
        return {
            "single_arm": out[None, :, :ARM_DOF].astype(np.float32),
            "gripper": out[None, :, ARM_DOF:].astype(np.float32),
        }, info


def main() -> None:
    for label, path in (("checkpoint", paths.CHECKPOINT), ("artifacts", paths.ARTIFACTS)):
        if not path.exists():
            raise SystemExit(f"no {label} at {path}")
    logging.basicConfig(level=logging.INFO, force=True)
    print(f"checkpoint : {paths.CHECKPOINT}")
    print(f"artifacts  : {paths.ARTIFACTS}")
    print(f"port       : {paths.PORT}   (the node's SERVER_INFERENCE_PORT)")
    print(f"gpu        : {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"homing     : {'on' if paths.HOMING else 'off'}")
    print("\nLoading ~12 GB of weights. On a shared filesystem this takes several minutes.")

    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.resolve(EMBODIMENT),
                         model_path=str(paths.CHECKPOINT), device="cuda")
    loop = AlohaServerLoop(paths.ARTIFACTS, homing=paths.HOMING)
    server = PolicyServer(policy=CorrectedPolicy(policy, loop), host="0.0.0.0", port=paths.PORT)
    print(f"""
================================================================
 Policy server is up on port {paths.PORT}.

 Now, in another terminal with your ROS2 environment sourced:
     python aloha_infer_right.py

 If the node runs on a different machine, point it here by setting
 SERVER_INFERENCE_IP to this host's address.

 Ctrl-C here stops the policy server.
================================================================""")
    try:
        server.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
