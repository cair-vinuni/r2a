#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
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

ACTION_ROWS = 16
VIDEO_KEY = "cam_head"
LANG_KEY = "annotation.human.task_description"

MODALITY_KEYS = ("left_shoulder", "left_elbow", "left_wrist", "left_hand",
                 "right_shoulder", "right_elbow", "right_wrist", "right_hand")
MODALITY_WIDTHS = (3, 1, 3, 6, 3, 1, 3, 6)
STATE_DIM = sum(MODALITY_WIDTHS)


def split_state(states) -> dict:
    """The 26-dim proprio rows as the checkpoint's per-key state dict, each (B, 1, D)."""
    rows = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in states])
    if rows.shape[1] != STATE_DIM:
        raise ValueError(f"state must be [{STATE_DIM}]-dim, got {rows.shape[1]}")
    out, start = {}, 0
    for key, width in zip(MODALITY_KEYS, MODALITY_WIDTHS):
        out[key] = np.ascontiguousarray(rows[:, start:start + width].reshape(-1, 1, width))
        start += width
    return out


def make_vrh31_batch(heads, states, instructions) -> dict:
    """GR00T's nested new_embodiment observation for B items, each with one frame (T=1).

    Args:
        heads: B head-camera frames, uint8 HxWx3.
        states: B proprio vectors, the checkpoint's action keys concatenated in order.
        instructions: B task strings.

    Returns:
        The observation Gr00tPolicy.get_action takes, batch first.
    """
    return {
        "video": {
            VIDEO_KEY: np.stack([np.ascontiguousarray(x, dtype=np.uint8) for x in heads])[:, None],
        },
        "state": split_state(states),
        "language": {LANG_KEY: [[str(x)] for x in instructions]},
    }


def decode_actions(action: dict) -> np.ndarray:
    """The policy's action dict as (B, ACTION_ROWS, 26), the modality keys concatenated in order.

    Args:
        action: The policy's action dict, bare or "action."-prefixed keys.

    Returns:
        [B, ACTION_ROWS, 26] float32 absolute joint targets.

    Raises:
        KeyError: The policy returned neither bare nor prefixed keys.
    """
    def pick(name: str, width: int) -> np.ndarray:
        for key in (name, f"action.{name}"):
            if key in action:
                return np.asarray(action[key], dtype=np.float32).reshape(-1, ACTION_ROWS, width)
        raise KeyError(f"policy returned no {name!r} (keys: {sorted(action)})")

    blocks = [pick(key, width) for key, width in zip(MODALITY_KEYS, MODALITY_WIDTHS)]
    return np.concatenate(blocks, axis=-1).astype(np.float32)


class Vrh31WSPolicy:
    """In-process GR00T policy exposing predict_action(**payload) for WebsocketPolicyServer."""

    def __init__(self, model_path: str, embodiment_tag: str = "NEW_EMBODIMENT",
                 device: str = "cuda"):
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        logging.info("Loading GR00T policy from %s (embodiment=%s)", model_path, embodiment_tag)
        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path,
                                  device=device)
        self.embodiment = embodiment_tag
        logging.info("GR00T VR_H31 policy ready (in-process).")

    def predict_action(self, batch_images, instructions, state=None,
                       episode_step=None, worker_id=None, unnorm_key=None, **kw):
        """One batched forward for B observations; the server may have gathered them from B clients."""
        n = len(batch_images)
        heads = [np.asarray(images[0]) for images in batch_images]
        states = ([np.asarray(state[i]).reshape(-1) for i in range(n)] if state is not None
                  else [np.zeros(STATE_DIM, np.float32)] * n)
        ret = self.policy.get_action(make_vrh31_batch(heads, states, instructions))
        action = ret[0] if isinstance(ret, tuple) else ret
        return {"actions": decode_actions(action)}


def main():
    p = argparse.ArgumentParser(description="GR00T VR_H31 humanoid websocket policy server.")
    p.add_argument("--model-path", required=True, help="the finetuned new_embodiment checkpoint.")
    p.add_argument("--port", type=int, default=10096, help="websocket port the client connects to.")
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
    policy = Vrh31WSPolicy(args.model_path, args.embodiment_tag)

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    server = WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port,
        metadata={"env": "vrh31_bds_pick", "model": "gr00t_n17_vrh31"},
        max_batch=args.max_batch, batch_wait_ms=args.batch_wait_ms,
    )
    logging.info("GR00T VR_H31 websocket policy server running on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
