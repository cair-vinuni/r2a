from typing import TYPE_CHECKING, Optional
import cv2 as cv
import numpy as np

# Vendored self-contained harness: tools/ is a package living alongside this file under
# evals/libero_plus/harness/.
from tools.websocket_policy_client import WebsocketClientPolicy


# StableVLA action-chunk size. Import the canonical constant from StableVLA's
# `prismatic` package when available; otherwise fall back to the LIBERO value (8).
# NOTE: `prismatic` is NOT vendored here — it resolves from the StableVLA trunk placed on
# PYTHONPATH by the launchers (scripts/run_sr.py). Kept as-is by design.
try:
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK as _STABLEVLA_NUM_ACTIONS_CHUNK
except Exception:  # prismatic not importable in this env -> use the LIBERO default
    _STABLEVLA_NUM_ACTIONS_CHUNK = 8

if TYPE_CHECKING:  # annotation only -- this module must import fine without robot_ood on the path
    from robot_ood.runtime.fuse import BlendPlan


class M1Inference:
    def __init__(
        self,
        image_size: list[int] = [224, 224],
        host="0.0.0.0",
        port=10095,
        worker_id: int = 0,
        exec_horizon: Optional[int] = None,
    ) -> None:

        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        # Stable per-client id (== shard index) so the server's AdaJEPA adapter can hold isolated
        # per-worker adaptation state under a pool of parallel clients (NUM_CLIENTS>1).
        self.worker_id = int(worker_id)
        self.image_size = image_size

        self.task_description = None
        # StableVLA executes a fixed-size action chunk (NUM_ACTIONS_CHUNK == 8 for LIBERO).
        self.action_chunk_size = _STABLEVLA_NUM_ACTIONS_CHUNK
        # Open-loop steps executed per inference before re-querying: the protocol's exec_horizon.
        # The policy still predicts the full chunk; only the first exec_horizon actions are used.
        self.exec_horizon = int(exec_horizon) if exec_horizon else self.action_chunk_size

    def reset(self, task_description: str) -> None:
        self.task_description = task_description


    def step(
        self,
        images,
        task_description: Optional[str] = None,
        state: Optional[np.ndarray] = None,
        step: int = 0,
        blend: "Optional[BlendPlan]" = None,
        **kwargs
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param images: list of (H, W, 3) uint8 views [primary, wrist]
        :param task_description: Task description text
        :param blend: Sentinel's plan for the chunk this step refills; None leaves it alone
        :return: {"raw_action": {"world_vector", "rotation_delta", "open_gripper"}}
        """

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        vla_input = {
            "batch_images": [images],
            "instructions": [self.task_description],
            # Per-episode sim step (0 at episode start). The StableVLA policy server uses it
            # for test-time cond probes (EMA reset / warm-up window); harmless otherwise.
            "episode_step": [step],
            # Stable worker id (== shard index) so the AdaJEPA adapter isolates per-worker state
            # under a parallel client pool. Ignored by non-AdaJEPA servers.
            "worker_id": [self.worker_id],
        }

        if state is not None:
            vla_input["state"] = [state]  # add batch dim

        # Track 3D-A: forward the oracle object/place 3D positions + grasp phase (batch dim) so the
        # server can build the kinesthetic displacement d for AdaLN conditioning.
        if "obj_pos" in kwargs and kwargs["obj_pos"] is not None:
            vla_input["obj_pos"] = [kwargs["obj_pos"]]
            vla_input["place_pos"] = [kwargs.get("place_pos", kwargs["obj_pos"])]
            vla_input["grasped"] = [bool(kwargs.get("grasped", False))]

        horizon = self.exec_horizon
        if step % horizon == 0:
            response = self.client.infer(vla_input)
            # NORMALIZATION SCHEME (b): the StableVLA policy server
            # (evals/stablevla/stablevla_policy_server.py) returns ALREADY-UNNORMALIZED
            # actions under "actions" (it runs StableVLA's authoritative BOUNDS_Q99
            # unnormalization internally). We therefore pass them straight through WITHOUT
            # re-normalizing here. The gripper dim is binarized sim-side in eval_libero.py.
            actions = response["data"]["actions"]  # B, chunk, D
            self.raw_actions = np.asarray(actions[0])
            # Immediately after the refill and nowhere else: this is the only pending-action buffer,
            # and a blend applied any earlier in the chunk is overwritten by the line above.
            if blend is not None:
                self.raw_actions = blend.apply(self.raw_actions)

        raw_actions = self.raw_actions[step % horizon][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return {"raw_action": raw_action}


    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image
