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

from robot_ood.data.lerobot_vrh31 import POLICY_KEYS, POLICY_WIDTHS
from robot_ood.runtime.vrh31_deploy import CHUNK_DIM, lane_hand, lane_joints
from robot_ood.runtime.vrh31_server import Vrh31ServerLoop

LANG_KEY = "annotation.human.task_description"
EMBODIMENT = "NEW_EMBODIMENT"
LANGUAGE_INSTRUCTION = (
    "Grasp the large curved metal frame with both hands and lift it upward from the rack."
)


def join(block: dict, expect_rows: int | None = None) -> np.ndarray:
    """The checkpoint's per-key arrays concatenated into rows of CHUNK_DIM columns.

    Args:
        block: The state or action dict, keyed by modality.
        expect_rows: Rows the result must have, or None to take whatever arrived.

    Returns:
        [rows, 26] float64.

    Raises:
        ValueError: A key is missing, or the row counts disagree.
    """
    missing = [key for key in POLICY_KEYS if key not in block]
    if missing:
        raise ValueError(f"observation is missing {missing} (have {sorted(block)})")
    parts = [np.asarray(block[key], dtype=np.float64).reshape(-1, width)
             for key, width in zip(POLICY_KEYS, POLICY_WIDTHS)]
    rows = {len(part) for part in parts}
    if len(rows) != 1:
        raise ValueError(f"modality keys disagree on row count: {rows}")
    out = np.concatenate(parts, axis=-1)
    if expect_rows is not None and len(out) != expect_rows:
        raise ValueError(f"expected {expect_rows} row(s), got {len(out)}")
    return out


def split(rows: np.ndarray) -> dict[str, np.ndarray]:
    """[K, 26] rows back into the checkpoint's per-key action dict, each (1, K, D) float32."""
    out, start = {}, 0
    for key, width in zip(POLICY_KEYS, POLICY_WIDTHS):
        out[key] = rows[None, :, start:start + width].astype(np.float32)
        start += width
    return out


class CorrectedPolicy(PolicyWrapper):
    """Gr00tPolicy's wire contract, with the homing walk and both lanes' correctors behind it."""

    def __init__(self, policy: Gr00tPolicy, loop: Vrh31ServerLoop) -> None:
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
        state = join(observation["state"], expect_rows=1)[0]
        history = (options or {}).get("history")
        observation["language"] = {LANG_KEY: [[LANGUAGE_INSTRUCTION]]}
        task = LANGUAGE_INSTRUCTION
        info: dict = {}

        def chunk() -> np.ndarray:
            action, extra = self.policy.get_action(observation, options)
            info.update(extra)
            return join(action)

        out = self.loop.step(state, task, chunk, history=history)
        if paths.LOG_CORRECTION:
            if history is not None:
                logging.info("HISTORY executed %s measured %s", np.shape(history["executed"]),
                             np.shape(history["measured"]))
            self._log(state)
        return split(out), info

    def _log(self, state: np.ndarray) -> None:
        if self.loop.homing_leg:
            for arm, lane in self.loop.lanes.items():
                if not lane.homing_leg:
                    continue
                r = lane.homing.report(lane_joints(state, arm), lane_hand(state, arm))
                logging.info("HOMING %-5s pass %d steps %d residual %.4f (radius %.4f)",
                             arm, r["passes"], r["steps"], r["residual"], r["level_hat"])
            return
        decisions = {arm: d for arm, d in self.loop.decisions.items() if d}
        if not decisions:
            logging.info("CORRECTOR pass-through")
            return
        for arm, d in decisions.items():
            logging.info("CORRECTOR %-5s |dq|max %.5f rad  trust %.3f shape %.3f",
                         arm, self.loop.moved, d["trust"], d["shape"])


def main() -> None:
    if not paths.CHECKPOINT.exists():
        raise SystemExit(f"no checkpoint at {paths.CHECKPOINT}")
    for arm in paths.ARMS:
        if not paths.ARTIFACTS[arm].exists():
            raise SystemExit(f"no {arm} artifacts at {paths.ARTIFACTS[arm]}")
    logging.basicConfig(level=logging.INFO, force=True)
    print(f"checkpoint : {paths.CHECKPOINT}")
    for arm in paths.ARMS:
        print(f"{arm:10s} : {paths.ARTIFACTS[arm]}")
    print(f"port       : {paths.PORT}   (SERVER_INFERENCE_PORT)")
    print(f"gpu        : {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"homing     : {'on' if paths.HOMING else 'off'}")
    print(f"chunk      : {CHUNK_DIM} columns, both arms and both hands")
    print("\nLoading ~12 GB of weights. On a shared filesystem this takes several minutes.")

    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.resolve(EMBODIMENT),
                         model_path=str(paths.CHECKPOINT), device="cuda")
    loop = Vrh31ServerLoop({arm: paths.ARTIFACTS[arm] for arm in paths.ARMS}, homing=paths.HOMING)
    server = PolicyServer(policy=CorrectedPolicy(policy, loop), host="0.0.0.0", port=paths.PORT)
    print(f"""
================================================================
 Policy server is up on port {paths.PORT}.

 Point your ROS2 client at this host and port (SERVER_INFERENCE_IP,
 SERVER_INFERENCE_PORT). The server sets the task on every observation,
 so the client sends no language field.

 Ctrl-C here stops the policy server.
================================================================""")
    try:
        server.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
