from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from robot_ood.core.action import ACTION_DIM, GRIP_COL
from robot_ood.core.paths import default_paths, suite_config
from robot_ood.data.libero import write_demo_file

__all__ = ["ACTION_ARM", "ACTION_HAND", "ARMS", "ARM_JOINTS", "CAMERA_KEYS", "FINGERS",
           "HAND_JOINTS", "OPEN_THRESHOLD", "POLICY_ACTION_BLOCKS", "POLICY_ARM", "POLICY_HAND",
           "POLICY_KEYS", "POLICY_STATE_BLOCKS", "POLICY_STATE_DIM", "POLICY_WIDTHS",
           "RAW_ACTION_DIM", "STATE_ARM", "STATE_DIM", "STATE_HAND", "VIDEO_KEY",
           "convert_dataset", "convert_suite", "episode_demo", "finger_mean", "instruction_of",
           "lane_targets", "open_flag", "policy_action", "policy_state", "slug_of"]

ARMS = ("left", "right")
ARM_JOINTS = 6
HAND_JOINTS = 6
STATE_DIM = 153
RAW_ACTION_DIM = 28
POLICY_STATE_DIM = 26

FINGERS = slice(2, 6)
OPEN_THRESHOLD = 0.1

STATE_ARM = {"left": (14, 16, 17, 18, 19, 20), "right": (21, 23, 24, 25, 26, 27)}
STATE_HAND = {"left": slice(30, 36), "right": slice(36, 42)}
ACTION_ARM = {"left": (0, 2, 3, 4, 5, 6), "right": (7, 9, 10, 11, 12, 13)}
ACTION_HAND = {"left": slice(14, 20), "right": slice(20, 26)}

POLICY_KEYS = ("left_shoulder", "left_elbow", "left_wrist", "left_hand",
               "right_shoulder", "right_elbow", "right_wrist", "right_hand")
POLICY_WIDTHS = (3, 1, 3, 6, 3, 1, 3, 6)
POLICY_STATE_BLOCKS = ((14, 17), (17, 18), (18, 21), (30, 36),
                       (21, 24), (24, 25), (25, 28), (36, 42))
POLICY_ACTION_BLOCKS = ((0, 3), (3, 4), (4, 7), (14, 20),
                        (7, 10), (10, 11), (11, 14), (20, 26))
POLICY_ARM = {"left": (0, 2, 3, 4, 5, 6), "right": (13, 15, 16, 17, 18, 19)}
POLICY_HAND = {"left": slice(7, 13), "right": slice(20, 26)}

VIDEO_KEY = "observation.images.color.head"
CAMERA_KEYS = ("head_rgb",)
JPEG_QUALITY = 95


def finger_mean(hand: np.ndarray) -> np.ndarray:
    """A hand's flexion as one number: the mean of index, middle, ring and pinky, thumb excluded."""
    return np.asarray(hand, dtype=np.float64)[..., FINGERS].mean(axis=-1)


def open_flag(hand: np.ndarray) -> np.ndarray:
    """A COMMANDED hand's four-finger mean as the corpus's open flag: 1.0 open, 0.0 closed."""
    return (finger_mean(hand) < OPEN_THRESHOLD).astype(np.float64)


def policy_state(state: np.ndarray) -> np.ndarray:
    """[T, 26] the proprio row the checkpoint's state modality takes, in its own key order."""
    rows = np.asarray(state, dtype=np.float64)
    return np.concatenate([rows[:, start:stop] for start, stop in POLICY_STATE_BLOCKS], axis=1)


def policy_action(action: np.ndarray) -> np.ndarray:
    """[T, 26] a recorded action row reordered into the checkpoint's action key order."""
    rows = np.asarray(action, dtype=np.float64)
    return np.concatenate([rows[:, start:stop] for start, stop in POLICY_ACTION_BLOCKS], axis=1)


def lane_targets(chunk: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """One arm's absolute joint targets and its hand, out of a 26-dim policy chunk.

    Args:
        chunk: [K, 26] policy output, the checkpoint's action keys concatenated in order.
        arm: left or right.

    Returns:
        [K, 6] float64 joint targets and [K, 6] float64 hand joints.
    """
    rows = np.asarray(chunk, dtype=np.float64)
    return rows[:, POLICY_ARM[arm]], rows[:, POLICY_HAND[arm]]


def slug_of(instruction: str) -> str:
    """The filename stem a corpus file must carry to satisfy build.policy_cache._instruction."""
    return instruction.strip().replace(" ", "_").lower()


def instruction_of(dataset_dir: Path) -> str:
    """The single task string a LeRobot dataset's meta/tasks.jsonl records.

    Args:
        dataset_dir: The dataset root, holding meta/ and data/.

    Returns:
        The instruction, verbatim.

    Raises:
        ValueError: The dataset records no task, or more than one.
    """
    lines = [json.loads(line) for line in
             (dataset_dir / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{dataset_dir.name}: expected exactly one task, found {len(lines)}")
    return str(lines[0]["task"]).strip()


def _jpeg_stream(path: Path, n_frames: int) -> np.ndarray:
    """One video as [T] fixed-width jpeg byte strings, the layout _decode_jpegs reads back.

    Args:
        path: The .mp4 to decode.
        n_frames: Frames the episode's parquet has, which the video must match.

    Returns:
        [T] numpy S-dtype array of jpeg buffers.

    Raises:
        ValueError: The video's frame count disagrees with the parquet's.
    """
    import av
    import cv2 as cv

    buffers: list[bytes] = []
    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            ok, buf = cv.imencode(".jpg", rgb, [int(cv.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                raise ValueError(f"{path.name}: jpeg encode failed at frame {len(buffers)}")
            buffers.append(buf.tobytes())
    finally:
        container.close()
    if len(buffers) != n_frames:
        raise ValueError(f"{path.name}: decoded {len(buffers)} frames, parquet has {n_frames}")
    return np.array(buffers)


def episode_demo(parquet_path: Path, dataset_dir: Path, episode: int, arm: str) -> dict[str, np.ndarray]:
    """One arm's lane demo out of one episode.

    Args:
        parquet_path: The episode's parquet file.
        dataset_dir: The dataset root, for the video.
        episode: The episode index, for the video filename.
        arm: left or right.

    Returns:
        The datasets write_demo_file stores: joint_states [T, 7], ee_pos [T, 3], ee_ori [T, 3],
        gripper_states [T, 1] (1 open, 0 closed), policy_state [T, 26], actions [T, 7] and the
        head camera's jpeg stream.

    Raises:
        ValueError: The parquet's state or action width is not the recorded one.
    """
    import pandas as pd

    table = pd.read_parquet(parquet_path)
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(table["action"].to_numpy()).astype(np.float64)
    if state.shape[1] != STATE_DIM or action.shape[1] != RAW_ACTION_DIM:
        raise ValueError(
            f"{parquet_path.name}: expected [{STATE_DIM}] state and [{RAW_ACTION_DIM}] action, "
            f"got {state.shape[1]}/{action.shape[1]}"
        )

    joints = state[:, STATE_ARM[arm]]
    flag = open_flag(action[:, ACTION_HAND[arm]])

    actions = np.zeros((len(joints), ACTION_DIM), dtype=np.float32)
    actions[:, :ARM_JOINTS] = (action[:, ACTION_ARM[arm]] - joints).astype(np.float32)
    actions[:, GRIP_COL] = np.where(flag >= 1.0, -1.0, 1.0)

    demo = {
        "joint_states": np.concatenate([joints, flag[:, None]], axis=1).astype(np.float32),
        "ee_pos": joints[:, :3].astype(np.float32),
        "ee_ori": joints[:, 3:ARM_JOINTS].astype(np.float32),
        "gripper_states": flag[:, None].astype(np.float32),
        "policy_state": policy_state(state).astype(np.float32),
        "actions": actions,
    }
    video = dataset_dir / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{episode:06d}.mp4"
    demo[CAMERA_KEYS[0]] = _jpeg_stream(video, len(joints))
    return demo


def _episodes(dataset_dir: Path, arm: str) -> Iterator[dict[str, np.ndarray]]:
    """Every episode of one dataset, in episode-index order."""
    files = sorted((dataset_dir / "data" / "chunk-000").glob("episode_*.parquet"),
                   key=lambda p: int(p.stem[len("episode_"):]))
    for path in files:
        yield episode_demo(path, dataset_dir, int(path.stem[len("episode_"):]), arm)


def convert_dataset(dataset_dir: Path, out_dir: Path, arm: str) -> Path:
    """Write one LeRobot dataset as one lane's corpus file.

    Args:
        dataset_dir: The dataset root, holding meta/, data/ and videos/.
        out_dir: The directory the corpus file is written into.
        arm: left or right.

    Returns:
        The file written.
    """
    instruction = instruction_of(dataset_dir)
    out_path = out_dir / f"{slug_of(instruction)}_demo.hdf5"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        logging.info("%s: present", out_path.name)
        return out_path
    n = write_demo_file(out_path, instruction, _episodes(dataset_dir, arm))
    logging.info("%s: %d demos", out_path.name, n)
    return out_path


def convert_suite(suite: str, raw_root: Path, names: tuple[str, ...] = ()) -> list[Path]:
    """Write the corpus configs/trees/<suite>.yaml describes: one file per LeRobot dataset.

    Args:
        suite: The lane tree name; its config names the arm and data.hdf5.
        raw_root: Where the downloaded LeRobot datasets live, one directory each.
        names: The dataset directory names to convert. Empty converts every dataset under raw_root.

    Returns:
        The corpus files written, sorted.

    Raises:
        ValueError: The tree names no arm of this robot.
        FileNotFoundError: A named dataset is missing, or raw_root holds none at all.
    """
    cfg = suite_config(suite)
    arm = str(cfg.get("arm") or "")
    if arm not in ARMS:
        raise ValueError(f"configs/trees/{suite}.yaml must set arm to one of {ARMS}, got {arm!r}")
    hdf5_dir = default_paths().data / str((cfg.get("data") or {})["hdf5"])
    if names:
        datasets = [raw_root / name for name in names]
        missing = [d.name for d in datasets if not (d / "meta" / "tasks.jsonl").exists()]
        if missing:
            raise FileNotFoundError(f"no LeRobot dataset under {raw_root} named: {missing}")
    else:
        datasets = [d for d in sorted(raw_root.iterdir()) if (d / "meta" / "tasks.jsonl").exists()]
    if not datasets:
        raise FileNotFoundError(f"no LeRobot datasets under {raw_root}")
    return sorted(convert_dataset(d, hdf5_dir, arm) for d in datasets)


def main(argv: list[str] | None = None) -> None:
    """Convert the LeRobot datasets of one lane tree into its corpus."""
    parser = argparse.ArgumentParser(description="VR_H31 LeRobot demonstrations as one lane's corpus")
    parser.add_argument("--suite", required=True, help="the lane tree, configs/trees/<suite>.yaml")
    parser.add_argument("--raw", default=None,
                        help="where the LeRobot datasets live (default: <data>/lerobot)")
    parser.add_argument("--datasets", default="",
                        help="comma-separated dataset directory names; empty takes every dataset "
                             "under --raw")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw_root = Path(args.raw) if args.raw else default_paths().data / "lerobot"
    names = tuple(n.strip() for n in args.datasets.split(",") if n.strip())
    for path in convert_suite(args.suite, raw_root, names):
        logging.info("wrote %s", path)


if __name__ == "__main__":
    main()
