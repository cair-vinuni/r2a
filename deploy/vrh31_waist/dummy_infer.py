"""Plays the robot against a running serve.py. The fake waist jumps to each commanded yaw, so the run is the
servo (arms held, one read per SETTLE) followed by the checkpoint's actions. Prints the waist command per call
and stops at the first call whose arms move."""

import time

import numpy as np

import paths
from gr00t.policy.server_client import PolicyClient

KEYS = ("left_shoulder", "left_elbow", "left_wrist", "left_hand",
        "right_shoulder", "right_elbow", "right_wrist", "right_hand")
WIDTHS = (3, 1, 3, 6, 3, 1, 3, 6)


def observation(waist: np.ndarray) -> dict:
    head = np.random.default_rng(0).integers(0, 255, size=(1, 1, *paths.IMAGE_SHAPE, 3), dtype=np.uint8)
    state = {k: np.zeros((1, 1, w), np.float32) for k, w in zip(KEYS, WIDTHS)}
    state["waist"] = waist.reshape(1, 1, 2).astype(np.float32)
    return {"video": {paths.VIDEO_KEY: head}, "state": state,
            "language": {"annotation.human.task_description": [[paths.INSTRUCTION]]}}


def main() -> None:
    client = PolicyClient(host=paths.SERVER_IP, port=paths.PORT, timeout_ms=60_000)
    client.reset()
    waist = np.zeros(2, np.float32)
    for call in range(1000):
        action, _ = client.get_action(observation(waist))
        command = np.asarray(action["waist"])[0, 0]
        moving = any(np.abs(np.asarray(v)).max() > 0 for k, v in action.items() if k != "waist")
        print(f"call {call:3d}  waist command roll {command[0]:+.3f} yaw {command[1]:+.3f}  "
              f"{'VLA' if moving else 'arms held'}")
        if moving:
            print("action ->", {k: np.asarray(v).shape for k, v in action.items()})
            break
        waist = command
        time.sleep(0.2)


if __name__ == "__main__":
    main()
