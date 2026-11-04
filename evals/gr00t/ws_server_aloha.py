#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""GR00T websocket policy server for a real ALOHA right arm — single-process, IN-PROCESS GR00T.

The sibling of ws_server.py, for a NEW_EMBODIMENT checkpoint rather than LIBERO_PANDA. Two things
differ, and both are forced by the checkpoint:

  * The observation is GR00T's NESTED format ({"video": ..., "state": ..., "language": ...}), which
    Gr00tPolicy.get_action takes directly. ws_server.py wraps the policy in Gr00tSimPolicyWrapper,
    which exists to accept the FLAT sim format instead; wrapping here would reject these keys.
  * The returned chunk is ABSOLUTE joint targets, not deltas. The checkpoint declares
    use_relative_action with single_arm REP=RELATIVE, and Gr00tPolicy._get_action passes the live
    state into StateActionProcessor.unapply_action, which adds the current joints back onto every
    row (JointActionChunk.to_absolute_chunking -- per row, not cumulatively). So what leaves this
    server is already absolute, and the caller differences it.

Protocol (the same wire contract every other policy server in this repo speaks):
  REQUEST (client -> server):
    batch_images : list (len B) of [front, wrist]   native-size uint8 HWC
    instructions : list (len B) of task strings
    state        : list (len B) of (1, 7) = [six joints, gripper], the gripper NATIVE
    episode_step : list (len B) of int   (0 => episode start)
    worker_id    : list (len B) of int
  RESPONSE (server -> client):
    {"actions": np.ndarray [B, ACTION_ROWS(=16), 7]} = ABSOLUTE [six joint targets, gripper]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

_ROBOT_OOD_ROOT = os.environ.get("ROBOT_OOD_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_HARNESS_DIR = os.path.join(_ROBOT_OOD_ROOT, "evals", "libero_plus", "harness")
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

# The checkpoint's action modality is delta_indices 0..15 over (single_arm, gripper).
ACTION_ROWS = 16
ARM_DOF = 6
LANG_KEY = "annotation.human.task_description"


def make_aloha_batch(fronts, wrists, states, instructions) -> dict:
    """GR00T's nested NEW_EMBODIMENT observation for B items, each with one frame (T=1).

    Args:
        fronts: B overhead frames, uint8 HxWx3.
        wrists: B right-wrist frames, same layout.
        states: B proprio vectors [six joints, gripper], the gripper native.
        instructions: B task strings.

    Returns:
        The observation Gr00tPolicy.get_action takes, batch first.
    """
    s = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in states])
    return {
        # (B, T, H, W, C) uint8, T == 1 -- Gr00tPolicy._validate_observation enforces five dims.
        "video": {
            "front": np.stack([np.ascontiguousarray(x, dtype=np.uint8) for x in fronts])[:, None],
            "wrist": np.stack([np.ascontiguousarray(x, dtype=np.uint8) for x in wrists])[:, None],
        },
        # (B, T, D) float32, T == 1.
        "state": {
            "single_arm": np.ascontiguousarray(s[:, :ARM_DOF].reshape(-1, 1, ARM_DOF)),
            "gripper": np.ascontiguousarray(s[:, ARM_DOF:ARM_DOF + 1].reshape(-1, 1, 1)),
        },
        # Each batch item is a list holding exactly one string.
        "language": {LANG_KEY: [[str(x)] for x in instructions]},
    }


def decode_actions(action: dict) -> np.ndarray:
    """GR00T's bare action keys as (B, ACTION_ROWS, 7) absolute [six joints, gripper].

    Gr00tPolicy._get_action returns what StateActionProcessor.unapply_action produced: the modality
    keys WITHOUT an "action." prefix (that prefix is added only by the processor's `unapply`, which
    this path does not use).

    Args:
        action: The policy's action dict.

    Returns:
        [B, ACTION_ROWS, 7] float32.

    Raises:
        KeyError: The policy returned neither bare nor prefixed keys.
    """
    def pick(name: str) -> np.ndarray:
        for key in (name, f"action.{name}"):
            if key in action:
                return np.asarray(action[key], dtype=np.float32)
        raise KeyError(f"policy returned no {name!r} (keys: {sorted(action)})")

    arm = pick("single_arm").reshape(-1, ACTION_ROWS, ARM_DOF)
    grip = pick("gripper").reshape(-1, ACTION_ROWS, 1)
    return np.concatenate([arm, grip], axis=-1).astype(np.float32)


class AlohaWSPolicy:
    """In-process GR00T policy exposing predict_action(**payload) for WebsocketPolicyServer."""

    def __init__(self, model_path: str, embodiment_tag: str = "NEW_EMBODIMENT",
                 device: str = "cuda"):
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        logging.info("Loading GR00T policy from %s (embodiment=%s)", model_path, embodiment_tag)
        # No Gr00tSimPolicyWrapper: that wrapper converts the FLAT sim observation into this nested
        # one, and we already build the nested one.
        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path,
                                  device=device)
        self.embodiment = embodiment_tag
        logging.info("GR00T ALOHA policy ready (in-process).")

    def predict_action(self, batch_images, instructions, state=None,
                       episode_step=None, worker_id=None, unnorm_key=None, **kw):
        """One batched forward for B observations; the server may have gathered them from B clients."""
        n = len(batch_images)
        fronts = [np.asarray(images[0]) for images in batch_images]
        wrists = [np.asarray(images[1]) if len(images) > 1 else np.asarray(images[0])
                  for images in batch_images]
        states = ([np.asarray(state[i]).reshape(-1) for i in range(n)] if state is not None
                  else [np.zeros(ARM_DOF + 1, np.float32)] * n)
        ret = self.policy.get_action(make_aloha_batch(fronts, wrists, states, instructions))
        action = ret[0] if isinstance(ret, tuple) else ret
        return {"actions": decode_actions(action)}


def main():
    p = argparse.ArgumentParser(description="GR00T ALOHA right-arm websocket policy server.")
    p.add_argument("--model-path", required=True, help="the finetuned NEW_EMBODIMENT checkpoint.")
    p.add_argument("--port", type=int, default=10095, help="websocket port the client connects to.")
    p.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    p.add_argument("--cuda", default="0")
    p.add_argument("--max-batch", type=int, default=16,
                   help="requests folded into one forward; use 1 for a single real robot")
    p.add_argument("--batch-wait-ms", type=float, default=3.0,
                   help="how long a batch waits for stragglers after the first request")
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    logging.basicConfig(level=logging.INFO, force=True)
    policy = AlohaWSPolicy(args.model_path, args.embodiment_tag)

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    server = WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port,
        metadata={"env": "aloha_right", "model": "gr00t_n17_aloha"},
        max_batch=args.max_batch, batch_wait_ms=args.batch_wait_ms,
    )
    logging.info("GR00T ALOHA websocket policy server running on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
