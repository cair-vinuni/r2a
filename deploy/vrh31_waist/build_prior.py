"""Fit the servo's nominal start model from the first head-camera frame of every demo episode.

Usage: python build_prior.py   (reads paths.DATASET, writes paths.PRIOR; about a minute on one GPU)
"""

import json

import numpy as np

import paths
from robot_ood.servo.gr00t import HeadCameraFeatures
from robot_ood.servo.prior import PRIOR_JSON, PRIOR_NPZ, ServoPrior

VIDEO = "observation.images.color.head"
RANK = 16
ALPHA = 0.05
FOLDS = 5


def first_frames(dataset) -> list[np.ndarray]:
    import av

    frames = []
    for video in sorted((dataset / "videos" / "chunk-000" / VIDEO).glob("episode_*.mp4")):
        with av.open(str(video)) as container:
            frames.append(next(container.decode(video=0)).to_ndarray(format="rgb24"))
    return frames


def grid_columns(model: HeadCameraFeatures) -> tuple[int, int]:
    """The GRID x GRID map: where the scene sits in the image, which is what a turn changes."""
    dim = model.policy.model.backbone.model.config.text_config.hidden_size
    return dim, dim * (1 + 16)


def main() -> None:
    frames = first_frames(paths.DATASET)
    model = HeadCameraFeatures(str(paths.CHECKPOINT))
    feats = np.concatenate([model.embed(frames[i:i + 8], paths.INSTRUCTION) for i in range(0, len(frames), 8)])
    cols = grid_columns(model)
    prior = ServoPrior.fit(feats, np.zeros(len(feats), int), [paths.INSTRUCTION], rank=RANK, cols=cols,
                           folds=np.arange(len(feats)) % FOLDS, alpha=ALPHA)
    paths.PRIOR.mkdir(parents=True, exist_ok=True)
    prior.save(paths.PRIOR / PRIOR_NPZ)
    (paths.PRIOR / PRIOR_JSON).write_text(json.dumps(
        {"checkpoint": paths.checkpoint_id(paths.CHECKPOINT), "dataset": str(paths.DATASET), "n_bank": len(feats), "rank": RANK,
         "alpha": ALPHA, "cols": cols, "tau": prior.tau, "energy_std": prior.energy_std}, indent=1))
    print(f"wrote {paths.PRIOR / PRIOR_NPZ}: {len(feats)} starts, tau {prior.tau:.3f}, "
          f"energy std {prior.energy_std:.3f}")


if __name__ == "__main__":
    main()
