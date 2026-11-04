"""ALOHA right-arm LeRobot demonstrations as one lane's corpus: the demo files robot_ood.data.libero reads.

The lane is a joint-space proxy [six joints, open flag]; the action at t is the absolute joint
target minus the joints at t, gripper +1 closed / -1 open. Cameras ride along as jpeg streams the
way robot_ood.data.robotwin stores them. Of the 13 recorded columns only the first seven are used.

Usage:
    python -m robot_ood.data.lerobot_aloha --suite aloha_right
"""

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

__all__ = ["ARM_JOINTS", "CAMERA_KEYS", "GRIP_IDX", "OPEN_THRESHOLD", "STATE_DIM", "VIDEO_KEYS",
           "convert_dataset", "convert_suite", "episode_demo", "instruction_of", "open_flag",
           "slug_of"]

ARM_JOINTS = 6
GRIP_IDX = 6                       # the gripper column of observation.state / action
STATE_DIM = 13                     # [six joints, gripper, six eef]; only the first seven are read

# The LeRobot video keys, and the obs datasets they are stored under. Order is the order the wire
# carries them: the checkpoint's video modality is (front, wrist).
VIDEO_KEYS = ("observation.images.color.high", "observation.images.color.wrist_right")
CAMERA_KEYS = ("head_rgb", "right_wrist_rgb")

# The COMMANDED aperture at or above which the gripper counts as open (larger is wider). Read off
# the command, never the measured state: the follower stalls at a held object's width. 1.10 is the valley.
OPEN_THRESHOLD = 1.10

# The jpeg quality the streams are encoded at. 95 is visually lossless at this resolution and lands
# around 15-20 KB a frame, which is what the shipped RoboTwin corpus already costs per camera.
JPEG_QUALITY = 95


def open_flag(gripper: np.ndarray) -> np.ndarray:
    """A COMMANDED ALOHA aperture as the corpus's open flag: 1.0 open, 0.0 closed (see OPEN_THRESHOLD)."""
    return (np.asarray(gripper, dtype=np.float64) >= OPEN_THRESHOLD).astype(np.float64)


def slug_of(instruction: str) -> str:
    """The filename stem a corpus file must carry to satisfy build.policy_cache._instruction.

    Args:
        instruction: The task's language instruction, verbatim.

    Returns:
        The stem, underscored and lowercased.
    """
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
            # imencode/imdecode are an identity on the array, so the RGB frame goes in unswapped.
            rgb = frame.to_ndarray(format="rgb24")
            ok, buf = cv.imencode(".jpg", rgb, [int(cv.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                raise ValueError(f"{path.name}: jpeg encode failed at frame {len(buffers)}")
            buffers.append(buf.tobytes())
    finally:
        container.close()
    if len(buffers) != n_frames:
        raise ValueError(
            f"{path.name}: decoded {len(buffers)} frames, parquet has {n_frames}"
        )
    return np.array(buffers)


def episode_demo(parquet_path: Path, dataset_dir: Path, episode: int) -> dict[str, np.ndarray]:
    """One episode's lane demo.

    Args:
        parquet_path: The episode's parquet file.
        dataset_dir: The dataset root, for the videos.
        episode: The episode index, for the video filenames.

    Returns:
        The datasets write_demo_file stores: joint_states [T, 7], ee_pos [T, 3], ee_ori [T, 3],
        gripper_states [T, 1] (1 open, 0 closed), grip_raw [T, 1], actions [T, 7] and the two
        jpeg streams.
    """
    import pandas as pd

    table = pd.read_parquet(parquet_path)
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(table["action"].to_numpy()).astype(np.float64)
    if state.shape[1] != STATE_DIM or action.shape[1] != STATE_DIM:
        raise ValueError(
            f"{parquet_path.name}: expected {STATE_DIM}-dim state/action, got "
            f"{state.shape[1]}/{action.shape[1]}"
        )

    joints = state[:, :ARM_JOINTS]
    # Proprio stays measured; the flag comes off the commanded aperture (see OPEN_THRESHOLD).
    raw_grip = state[:, GRIP_IDX]
    flag = open_flag(action[:, GRIP_IDX])

    # The recorded action is an absolute joint target: the demo's delta at t is target minus joints,
    # as core.absolute.deltas_from_targets differences the policy's own chunks.
    actions = np.zeros((len(joints), ACTION_DIM), dtype=np.float32)
    actions[:, :ARM_JOINTS] = (action[:, :ARM_JOINTS] - joints).astype(np.float32)
    actions[:, GRIP_COL] = np.where(flag >= 1.0, -1.0, 1.0)

    demo = {
        "joint_states": np.concatenate([joints, flag[:, None]], axis=1).astype(np.float32),
        "ee_pos": joints[:, :3].astype(np.float32),
        "ee_ori": joints[:, 3:ARM_JOINTS].astype(np.float32),
        "gripper_states": flag[:, None].astype(np.float32),
        # The gripper as the policy server must see it: native, never the flag. The checkpoint's
        # gripper action config is ABSOLUTE and its range is continuous.
        "grip_raw": raw_grip[:, None].astype(np.float32),
        "actions": actions,
    }
    for video_key, camera_key in zip(VIDEO_KEYS, CAMERA_KEYS):
        path = dataset_dir / "videos" / "chunk-000" / video_key / f"episode_{episode:06d}.mp4"
        demo[camera_key] = _jpeg_stream(path, len(joints))
    return demo


def _episodes(dataset_dir: Path) -> Iterator[dict[str, np.ndarray]]:
    """Every episode of one dataset, in episode-index order."""
    files = sorted((dataset_dir / "data" / "chunk-000").glob("episode_*.parquet"),
                   key=lambda p: int(p.stem[len("episode_"):]))
    for path in files:
        episode = int(path.stem[len("episode_"):])
        yield episode_demo(path, dataset_dir, episode)


def convert_dataset(dataset_dir: Path, out_dir: Path) -> Path:
    """Write one LeRobot dataset as one corpus file.

    Args:
        dataset_dir: The dataset root, holding meta/, data/ and videos/.
        out_dir: The directory the corpus file is written into.

    Returns:
        The file written.
    """
    instruction = instruction_of(dataset_dir)
    out_path = out_dir / f"{slug_of(instruction)}_demo.hdf5"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        logging.info("%s: present", out_path.name)
        return out_path
    n = write_demo_file(out_path, instruction, _episodes(dataset_dir))
    logging.info("%s: %d demos", out_path.name, n)
    return out_path


def convert_suite(suite: str, raw_root: Path, names: tuple[str, ...] = ()) -> list[Path]:
    """Write the corpus configs/trees/<suite>.yaml describes: one file per LeRobot dataset.

    Args:
        suite: The lane tree name; its config names the arm and data.hdf5.
        raw_root: Where the downloaded LeRobot datasets live, one directory each.
        names: The dataset directory names to convert. Empty converts every dataset under raw_root,
            which is only right when that directory holds nothing else.

    Returns:
        The corpus files written, sorted -- the order task ids are assigned in.

    Raises:
        FileNotFoundError: A named dataset is missing, or raw_root holds none at all.
    """
    cfg = suite_config(suite)
    if str(cfg.get("arm") or "") != "right":
        raise ValueError(f"configs/trees/{suite}.yaml must set arm: right (this corpus is one arm)")
    hdf5_dir = default_paths().data / str((cfg.get("data") or {})["hdf5"])
    # raw_root is a shared download directory holding other corpora too; naming the datasets keeps
    # this from walking into them.
    if names:
        datasets = [raw_root / name for name in names]
        missing = [d.name for d in datasets if not (d / "meta" / "tasks.jsonl").exists()]
        if missing:
            raise FileNotFoundError(f"no LeRobot dataset under {raw_root} named: {missing}")
    else:
        datasets = [d for d in sorted(raw_root.iterdir()) if (d / "meta" / "tasks.jsonl").exists()]
    if not datasets:
        raise FileNotFoundError(f"no LeRobot datasets under {raw_root}")
    return sorted(convert_dataset(d, hdf5_dir) for d in datasets)


def main(argv: list[str] | None = None) -> None:
    """Convert the LeRobot datasets of one suite into its corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="the lane tree, configs/trees/<suite>.yaml")
    parser.add_argument("--raw", default=None,
                        help="where the LeRobot datasets live (default: <data>/lerobot)")
    parser.add_argument("--datasets", default="",
                        help="comma-separated dataset directory names; empty takes every dataset "
                             "under --raw, which is only safe when it holds nothing else")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw_root = Path(args.raw) if args.raw else default_paths().data / "lerobot"
    names = tuple(n.strip() for n in args.datasets.split(",") if n.strip())
    for path in convert_suite(args.suite, raw_root, names):
        logging.info("wrote %s", path)


if __name__ == "__main__":
    main()
