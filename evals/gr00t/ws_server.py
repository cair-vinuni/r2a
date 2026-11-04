#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""GR00T websocket policy server — single-process, IN-PROCESS GR00T.

Drop-in replacement for StableVLA's policy server: exposes GR00T-N1.7-LIBERO behind the SAME
websocket contract the vendored LIBERO-Plus sim client (evals/libero_plus/harness/eval_libero.py)
handshakes with. Loads GR00T in-process (Gr00tPolicy + Gr00tSimPolicyWrapper) so there is no
separate back-end to orphan.

>>> Runs under the GR00T env (Isaac-GR00T's own .venv, GPU). See envs/README.md §GR00T. <<<

Imports NO ONF code — the GR00T model loading points at the Isaac-GR00T checkout; only
the transport (WebsocketPolicyServer + msgpack helpers) is resolved from $ROBOT_OOD_ROOT.

Protocol (must match stablevla_policy_server.predict_action / model2libero_interface):
  REQUEST (client -> server):
    batch_images : list (len B) of [primary, wrist]   (already 180deg-rotated, client_image_size square uint8)
    instructions : list (len B) of task strings
    state        : list (len B) of (8,) proprio = [ee_pos(3), ee_ori_axisangle(3), gripper(2)]
    episode_step : list (len B) of int   (0 => episode start)
    worker_id    : list (len B) of int
  RESPONSE (server -> client):
    {"actions": np.ndarray [B, NUM_ACTIONS_CHUNK(=8), 7]}  = unnormalized OSC delta
    [world_vec(3), rotation_delta(3), open_gripper(1)]; open_gripper RAW in [0,1] (1=open).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

_ROBOT_OOD_ROOT = os.environ.get("ROBOT_OOD_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
# The websocket transport (deployment.model_server.tools.websocket_policy_server) is vendored
# in-repo under evals/libero_plus/harness/, not an external checkout.
_HARNESS_DIR = os.path.join(_ROBOT_OOD_ROOT, "evals", "libero_plus", "harness")
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

NUM_ACTIONS_CHUNK = 8
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]
_ACT_KEYS = ["action.x", "action.y", "action.z",
             "action.roll", "action.pitch", "action.yaw", "action.gripper"]


def make_gr00t_batch(primaries, wrists, states, instructions) -> dict:
    """Flat GR00T LIBERO_PANDA observation for B items, each with one frame (T=1).

    Args:
        primaries: B agent-view frames, uint8 HxWx3, already rotated 180 degrees by the client.
        wrists: B wrist frames, same layout.
        states: B proprio vectors [pos(3), ee_ori_axisangle(3), g0, g1].
        instructions: B task strings.

    Returns:
        The observation Gr00tSimPolicyWrapper.get_action takes, batch first.
    """
    s = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in states])
    o = {"video.image": np.stack([np.ascontiguousarray(x, dtype=np.uint8) for x in primaries])[:, None],
         "video.wrist_image": np.stack([np.ascontiguousarray(x, dtype=np.uint8) for x in wrists])[:, None],
         "annotation.human.action.task_description": [str(x) for x in instructions]}
    for j, k in enumerate(STATE_KEYS):
        o[f"state.{k}"] = np.ascontiguousarray(s[:, j].reshape(-1, 1, 1))
    o["state.gripper"] = np.ascontiguousarray(s[:, 6:8].reshape(-1, 1, 2))
    return o


def decode_actions(action: dict) -> np.ndarray:
    """GR00T flat action {action.x..action.gripper}, each (B, Tchunk, 1), as (B, NUM_ACTIONS_CHUNK, 7)."""
    chunk = np.concatenate([np.asarray(action[k]) for k in _ACT_KEYS], axis=-1)
    return chunk[:, :NUM_ACTIONS_CHUNK].astype(np.float32)


class GrootWSPolicy:
    """In-process GR00T policy exposing predict_action(**payload) for WebsocketPolicyServer."""

    def __init__(self, model_path: str, embodiment_tag: str = "LIBERO_PANDA", device: str = "cuda"):
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
        logging.info("Loading GR00T policy from %s (embodiment=%s)", model_path, embodiment_tag)
        policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=device)
        self.policy = Gr00tSimPolicyWrapper(policy)
        self.embodiment = embodiment_tag
        logging.info("GR00T policy ready (in-process).")

    def predict_action(self, batch_images, instructions, state=None,
                       episode_step=None, worker_id=None, unnorm_key=None, **kw):
        """One batched forward for B observations; the server may have gathered them from B clients."""
        B = len(batch_images)
        primaries = [np.asarray(images[0]) for images in batch_images]
        wrists = [np.asarray(images[1]) if len(images) > 1 else np.asarray(images[0])
                  for images in batch_images]
        states = ([np.asarray(state[i]).reshape(-1) for i in range(B)] if state is not None
                  else [np.zeros(8, np.float32)] * B)
        ret = self.policy.get_action(make_gr00t_batch(primaries, wrists, states, instructions))
        action = ret[0] if isinstance(ret, tuple) else ret
        return {"actions": decode_actions(action)}


def main():
    p = argparse.ArgumentParser(description="GR00T websocket policy server (in-process).")
    p.add_argument("--model-path", required=True, help="GR00T ckpt dir (ckpts_n17_libero/libero_<suite>).")
    p.add_argument("--port", type=int, default=10095, help="Websocket port the sim client connects to.")
    p.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    p.add_argument("--cuda", default="0")
    p.add_argument("--max-batch", type=int, default=16,
                   help="requests folded into one forward; 16 gave 8x the items/s of 1 on an A100")
    p.add_argument("--batch-wait-ms", type=float, default=3.0,
                   help="how long a batch waits for stragglers after the first request")
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    logging.basicConfig(level=logging.INFO, force=True)
    policy = GrootWSPolicy(args.model_path, args.embodiment_tag)

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    server = WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port,
        metadata={"env": "libero_plus", "model": "gr00t_n17"},
        max_batch=args.max_batch, batch_wait_ms=args.batch_wait_ms,
    )
    logging.info("GR00T websocket policy server running on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
