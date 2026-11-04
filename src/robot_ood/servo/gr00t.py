"""GR00T features of one head-camera image, for the servo on a real robot.

Any GR00T N1.7 checkpoint works: the features are read inside the VLM backbone, before the action head, so
the embodiment only decides how the image is resized and how many frames go in. A checkpoint finetuned on
VR-H3.1 demos reads the image under its own tag. The base checkpoint has none and uses the closest humanoid
tag it was trained on.
"""

from __future__ import annotations

from typing import Final

import numpy as np

LAYER: Final[int] = 12          # LLM layer whose image tokens are pooled
GRID: Final[int] = 4            # image tokens pooled onto a GRID x GRID map
BASE_TAG: Final[str] = "REAL_G1"
FINETUNED_TAG: Final[str] = "NEW_EMBODIMENT"
LANG_KEY: Final[str] = "annotation.human.task_description"


class HeadCameraFeatures:
    """embed(images, task) -> [B, F]: pooled image tokens of LAYER, then their GRID x GRID map."""

    def __init__(self, model_path: str, embodiment_tag: str | None = None, device: str = "cuda") -> None:
        import json
        import pathlib

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        if embodiment_tag is None:
            root = pathlib.Path(model_path)
            path = next(p for p in (root / "processor" / "statistics.json", root / "statistics.json") if p.exists())
            stats = json.loads(path.read_text())
            embodiment_tag = FINETUNED_TAG if FINETUNED_TAG.lower() in stats else BASE_TAG

        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=device)
        config = self.policy.modality_configs
        self.video_key = config["video"].modality_keys[0]
        self.frames = len(config["video"].delta_indices)
        stats = self.policy.processor.state_action_processor.statistics[self.policy.embodiment_tag.value]["state"]
        # The features never see the state; the statistics mean is just a valid one (orthonormal-ish rotations).
        self.state = {k: np.asarray(stats[k]["mean"], np.float32) for k in config["state"].modality_keys}
        self.language_key = self.policy.language_key
        self.feats: dict = {}
        self._tap(self.policy.model)

    def _tap(self, model) -> None:
        import torch

        image_token = model.backbone.model.config.image_token_id

        def hook(_module, _args, kwargs, out):
            mask = (kwargs["input_ids"] == image_token)[..., None]
            h = out.hidden_states[LAYER]
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            n = int(mask[0].sum()) // self.frames
            side = int(round(n ** 0.5))
            # The last frame is the current one; earlier frames repeat it.
            tokens = h[mask[..., 0]].reshape(h.shape[0], self.frames, n, h.shape[-1])[:, -1]
            maps = tokens.reshape(h.shape[0], side, side, -1).permute(0, 3, 1, 2)
            grid = torch.nn.functional.adaptive_avg_pool2d(maps.float(), GRID).permute(0, 2, 3, 1)
            self.feats["feat"] = torch.cat([pooled.float(), grid.reshape(h.shape[0], -1)], 1)

        model.backbone.model.register_forward_hook(hook, with_kwargs=True)

    def observation(self, images: list[np.ndarray], task: str) -> dict:
        video = np.stack([np.repeat(np.asarray(x, np.uint8)[None], self.frames, 0) for x in images])
        state = {k: np.tile(v, (len(images), 1, 1)) for k, v in self.state.items()}
        return {"video": {self.video_key: video}, "state": state,
                "language": {self.language_key: [[task]] * len(images)}}

    def embed(self, images: list[np.ndarray], task: str) -> np.ndarray:
        """Runs the policy's own preprocessing and its backbone, not the action head."""
        import torch
        from gr00t.policy.gr00t_policy import MessageType, _rec_to_dtype

        policy = self.policy
        steps = [policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": policy._to_vla_step_data(o)}])
                 for o in policy._unbatch_observation(self.observation(images, task))]
        inputs = _rec_to_dtype(policy.collate_fn(steps), dtype=torch.bfloat16)
        with torch.inference_mode():
            backbone_inputs, _ = policy.model.prepare_input(inputs["inputs"])
            policy.model.backbone(backbone_inputs)
        return self.feats["feat"].cpu().numpy().astype(np.float64)
