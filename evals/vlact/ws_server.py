"""VLAct websocket policy server: the QwenGR00T framework behind the harness's websocket contract.

The model, its image bucketing, its unnormalization and its column order are VLAct's own, read from
the checkpoint's config.yaml and dataset_statistics.json the way
examples/Robotwin/eval_files/model2robotwin_interface.py reads them, so a chunk served here is the
chunk the vendor's RoboTwin client would have stepped.

A request carrying the episode's seed seeds the diffusion sampler from (seed, episode_step), so an
episode replays the same actions whichever client runs it and whichever batch it lands in.

Usage:
    external/VLAct/.venv/bin/python evals/vlact/ws_server.py --checkpoint <ckpt.pt> --port 14090
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

# The chunk's columns as the model emits them: both arms' six joints, then both grippers. RoboTwin
# takes each arm's gripper next to its own joints.
ARM_THEN_GRIPPER = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]

# Denoising steps of the vendor's RoboTwin client.
DDIM_STEPS = 10


def bucket_size(image: np.ndarray, buckets: list[list[int]], fallback: list[int]) -> tuple[int, int]:
    """The (width, height) VLAct resizes a frame to: the bucket closest in aspect ratio."""
    if buckets:
        height, width = image.shape[:2]
        ratio = width / float(height)
        target = min(buckets, key=lambda size: abs((size[0] / float(size[1])) - ratio))
        return int(target[0]), int(target[1])
    return int(fallback[0]), int(fallback[1])


def unnormalize(normalized: np.ndarray, stats: dict, mode: str) -> np.ndarray:
    """Joint targets in radians from the model's normalized output, as the vendor unnormalizes them.

    Args:
        normalized: [T, 14] the model's output.
        stats: The checkpoint's action statistics for the embodiment.
        mode: wrap reads the masked columns as angles; stats maps them through min/max.
    """
    mask = np.asarray(stats.get("mask", np.ones_like(stats["min"], dtype=bool)), dtype=bool)
    raw = np.asarray(normalized, dtype=np.float32)
    actions = np.clip(raw, -1, 1)
    if np.any(mask):
        if mode == "wrap":
            actions[..., mask] = (raw[..., mask] + np.pi) % (2 * np.pi) - np.pi
        else:
            low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
            actions[..., mask] = 0.5 * (actions[..., mask] + 1) * (high[mask] - low[mask]) + low[mask]
    return actions


class VLActWSPolicy:
    """VLAct's RoboTwin policy, answering the harness's (batch_images, instructions) requests."""

    def __init__(self, checkpoint: str, unnorm_key: str | None = None) -> None:
        """Build the framework from a checkpoint and read its deployment settings.

        Args:
            checkpoint: Path to the run's steps_*_pytorch_model.pt.
            unnorm_key: Embodiment key in dataset_statistics.json; the only one by default.
        """
        from accelerate import PartialState
        from starVLA.model.framework.base_framework import baseframework
        from starVLA.model.tools import read_mode_config

        # VLAct's modules log through accelerate, which refuses to log before its state exists.
        PartialState()

        path = Path(checkpoint)
        config, norm_stats = read_mode_config(path)
        key = unnorm_key if unnorm_key in norm_stats else next(iter(norm_stats))
        stats = norm_stats[key]
        self.action_stats = stats["abs"]["action"] if "abs" in stats else stats["action"]
        # The vendor reads the unnormalization mode off the checkpoint path; the training mix is
        # what actually decides it, and it is in the config the checkpoint carries.
        mix = str(((config.get("datasets") or {}).get("vla_data") or {}).get("data_mix", ""))
        self.unnorm_mode = "wrap" if "wrap" in f"{path}{mix}".lower() else "stats"

        data_config = (config.get("datasets") or {}).get("vla_data") or {}
        self.image_buckets = [[int(s[0]), int(s[1])] for s in (data_config.get("image_size_buckets") or [])]
        self.image_size = [int(s) for s in (data_config.get("image_size") or [224, 224])]
        self.chunk = int(config["framework"]["action_model"]["future_action_window_size"]) + 1

        self.model = baseframework.from_pretrained(str(path)).to(torch.bfloat16).to("cuda").eval()
        logging.info("VLAct chunk %d, unnormalization %s, buckets %s",
                     self.chunk, self.unnorm_mode, self.image_buckets)

    def _example(self, views, instruction) -> dict:
        """One request item as VLAct's RoboTwin interface builds it: three resized frames and the task.

        The state channel is left out, which is what the vendor's interface sends.
        """
        images = []
        for view in views:
            frame = np.asarray(view)
            images.append(cv.resize(frame, bucket_size(frame, self.image_buckets, self.image_size),
                                    interpolation=cv.INTER_AREA))
        return {"lang": str(instruction), "image": images}

    def predict_action(self, batch_images, instructions, state=None, episode_step=None,
                       worker_id=None, seed=None, **kw):
        """Action chunks for a batch of observations.

        Args:
            batch_images: Per item, the [head, left wrist, right wrist] uint8 frames.
            instructions: Per item, the task string.
            state: Per item, the 14-dim joint vector; the model reads the frames alone.
            episode_step: Per item, the env step the chunk is sampled at.
            seed: Per item, the episode's scene seed; with it the sampler is seeded from
                (seed, episode_step), without it it draws from the global generator.

        Returns:
            actions: float32 array [B, chunk, 14], both arms' absolute joint targets.
        """
        out = []
        for i, views in enumerate(batch_images):
            if seed is not None:
                torch.manual_seed(int(seed[i]) * (1 << 20) + int(episode_step[i]))
            with torch.inference_mode():
                response = self.model.predict_action(
                    examples=[self._example(views, instructions[i])],
                    do_sample=False, use_ddim=True, num_ddim_steps=DDIM_STEPS,
                )
            chunk = np.asarray(response["normalized_actions"][0], dtype=np.float32)
            out.append(unnormalize(chunk, self.action_stats, self.unnorm_mode)[:, ARM_THEN_GRIPPER])
        return {"actions": np.stack(out, axis=0)}


def main() -> None:
    """Serve the policy until killed."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="path to steps_*_pytorch_model.pt")
    p.add_argument("--port", type=int, default=14090)
    p.add_argument("--unnorm-key", default="new_embodiment",
                   help="embodiment key in dataset_statistics.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    policy = VLActWSPolicy(args.checkpoint, args.unnorm_key)
    # The harness's own server, not the deployment.* module of the same name VLAct ships, which
    # answers in a different envelope.
    from tools.websocket_policy_server import WebsocketPolicyServer
    server = WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port,
                                   metadata={"env": "robotwin", "model": "vlact"})
    logging.info("VLAct websocket policy server running on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
