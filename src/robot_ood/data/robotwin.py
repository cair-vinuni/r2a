"""RoboTwin 2.0 demonstrations as one lane's corpus: the demo files robot_ood.data.libero reads.

A lane is one arm of RoboTwin's per-episode files as a joint-space proxy [six joints, open flag]
(the tree sets ee_from_obs and pose_space euclidean); its action at frame t is the joint step to
t + 1 with the gripper +1 closed / -1 open. The cameras and the 14-dim joint state ride along.

Usage:
    python -m robot_ood.data.robotwin --suite g1_left [--raw data/robotwin/raw]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from robot_ood.core.absolute import targets_from_deltas
from robot_ood.core.action import ACTION_DIM, EUCLIDEAN, GRIP_COL
from robot_ood.core.paths import default_paths, repo_root, suite_config
from robot_ood.data.libero import open_hdf5, write_demo_file

__all__ = ["ARMS", "ARM_JOINTS", "CAMERAS", "CAMERA_KEYS", "JOINT_DIM", "LANE_COLUMNS", "OPEN_THRESHOLD",
           "convert_task", "joint_rows", "lane_demo", "lane_step_row", "lane_targets", "open_flag",
           "raw_episodes", "task_phrasings"]

ARMS = ("left", "right")
ARM_JOINTS = 6
CAMERAS = ("head_camera", "left_camera", "right_camera")
CAMERA_KEYS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
JOINT_DIM = 14                                                  # [left 6, left grip, right 6, right grip]
LANE_COLUMNS = {"left": slice(0, 7), "right": slice(7, 14)}     # six joints then the gripper, per arm

# The gripper fraction at or above which the gripper counts as open: midway between RoboTwin's open
# command and the widest grasp its experts hold, so a gripper stopped by an object reads as closed.
OPEN_THRESHOLD = 0.8
EMBODIMENT = "aloha-agilex"
DEMO_SET = f"{EMBODIMENT}_clean_50"


def open_flag(gripper: np.ndarray) -> np.ndarray:
    """RoboTwin's gripper fraction as the corpus's open flag: 1.0 open, 0.0 closed."""
    return (np.asarray(gripper, dtype=np.float64) >= OPEN_THRESHOLD).astype(np.float64)


def lane_targets(chunk: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """One arm's absolute joint targets and gripper fractions out of a fourteen-dim joint chunk.

    Args:
        chunk: [K, 14] policy output, RoboTwin's joint_action/vector layout.
        arm: left or right.

    Returns:
        [K, 6] float64 joint targets and [K] the gripper fraction.
    """
    lane = np.asarray(chunk, dtype=np.float64)[:, LANE_COLUMNS[arm]]
    return lane[:, :ARM_JOINTS], lane[:, ARM_JOINTS]


def joint_rows(targets: dict[str, np.ndarray], grips: dict[str, np.ndarray]) -> np.ndarray:
    """env.take_action(action_type="qpos") rows from per-arm joint targets and gripper fractions."""
    return np.concatenate([np.concatenate([np.asarray(targets[arm], dtype=np.float64),
                                           np.asarray(grips[arm], dtype=np.float64)[:, None]], axis=1)
                           for arm in ARMS], axis=1)


def lane_step_row(states: Mapping[str, np.ndarray], arm: str, action: np.ndarray) -> np.ndarray:
    """One env.take_action row that moves one arm by a corpus action while the other holds its targets.

    Args:
        states: arm -> [7] its commanded joints and gripper fraction, RoboTwin's own reading.
        arm: The arm the action moves.
        action: [7] the corpus convention: six joint deltas, then the gripper command, open below zero.

    Returns:
        [14] joint targets and gripper fractions in RoboTwin's joint_action/vector layout.
    """
    action = np.asarray(action, dtype=np.float64)
    targets = {a: np.asarray(states[a], dtype=np.float64)[None, :ARM_JOINTS] for a in ARMS}
    grips = {a: np.asarray(states[a], dtype=np.float64)[None, ARM_JOINTS] for a in ARMS}
    targets[arm] = targets_from_deltas(targets[arm][0], action[None, :ARM_JOINTS], EUCLIDEAN)
    grips[arm] = np.array([1.0 if action[GRIP_COL] < 0.0 else 0.0])
    return joint_rows(targets, grips)[0]


def raw_episodes(task_dir: Path) -> list[Path]:
    """The episode files of one downloaded task, in episode order."""
    files = sorted((task_dir / DEMO_SET / "data").glob("episode*.hdf5"),
                   key=lambda p: int(p.stem[len("episode"):]))
    if not files:
        raise FileNotFoundError(f"no episode*.hdf5 under {task_dir / DEMO_SET / 'data'}")
    return files


def lane_demo(handle, arm: str) -> dict[str, np.ndarray]:
    """One lane's demo from one RoboTwin episode file.

    Args:
        handle: The open h5py file.
        arm: left or right.

    Returns:
        The datasets write_demo_file stores: joint_states [T, 7], ee_pos [T, 3], ee_ori [T, 3],
        gripper_states [T, 1] (1 open, 0 closed), actions [T, 7], the three jpeg streams and
        policy_state [T, 14].
    """
    joints = np.asarray(handle[f"joint_action/{arm}_arm"], dtype=np.float64)
    gripper = np.asarray(handle[f"joint_action/{arm}_gripper"], dtype=np.float64)
    flag = open_flag(gripper)
    actions = np.zeros((len(joints), ACTION_DIM), dtype=np.float32)
    actions[:-1, :ARM_JOINTS] = np.diff(joints, axis=0)
    actions[:, GRIP_COL] = np.where(flag >= 1.0, -1.0, 1.0)
    demo = {
        "joint_states": np.concatenate([joints, flag[:, None]], axis=1).astype(np.float32),
        "ee_pos": joints[:, :3].astype(np.float32),
        "ee_ori": joints[:, 3:ARM_JOINTS].astype(np.float32),
        "gripper_states": flag[:, None].astype(np.float32),
        "actions": actions,
        "policy_state": np.asarray(handle["joint_action/vector"], dtype=np.float32),
    }
    for camera, key in zip(CAMERAS, CAMERA_KEYS):
        demo[key] = np.asarray(handle[f"observation/{camera}/rgb"])
    return demo


def _episodes(files: list[Path], arm: str) -> Iterator[dict[str, np.ndarray]]:
    for path in files:
        with open_hdf5(path) as handle:
            yield lane_demo(handle, arm)


def task_phrasings(task_dir: Path) -> list[str]:
    """Every distinct seen phrasing the task's episodes were recorded under, in episode order.

    Args:
        task_dir: The downloaded task directory (holding <embodiment>_clean_50/).

    Returns:
        The phrasings; empty when the download carries no instruction files.
    """
    import json

    files = sorted((task_dir / DEMO_SET / "instructions").glob("episode*.json"),
                   key=lambda p: int(p.stem[len("episode"):]))
    seen: dict[str, None] = {}
    for path in files:
        for text in json.loads(path.read_text()).get("seen", []):
            seen.setdefault(str(text))
    return list(seen)


def convert_task(task_dir: Path, arm: str, out_path: Path) -> int:
    """Write one task's lane corpus.

    Args:
        task_dir: The downloaded task directory (holding <embodiment>_clean_50/).
        arm: left or right.
        out_path: The demo file to write.

    Returns:
        Demos written.
    """
    instruction = task_dir.name.replace("_", " ")   # the task, as problem_info names it
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return write_demo_file(out_path, instruction, _episodes(raw_episodes(task_dir), arm),
                           phrasings=task_phrasings(task_dir))


def convert_suite(suite: str, raw_root: Path) -> list[Path]:
    """Write the lane corpus configs/trees/<suite>.yaml describes: one file per task of its list.

    Args:
        suite: The lane tree name; its config names the arm, the task file and data.hdf5.
        raw_root: Where the downloaded tasks live, one directory each.

    Returns:
        The demo files written, in task order.
    """
    cfg = suite_config(suite)
    arm = str(cfg.get("arm") or "")
    if arm not in ARMS:
        raise ValueError(f"configs/trees/{suite}.yaml must name arm: left or right, got {arm!r}")
    hdf5_dir = default_paths().data / str((cfg.get("data") or {})["hdf5"])
    task_file = cfg.get("tasks")
    if not task_file:
        raise ValueError(f"configs/trees/{suite}.yaml must name tasks: the suite file listing its tasks")
    tasks = [line.strip() for line in (repo_root() / str(task_file)).read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    if tasks != sorted(tasks):
        raise ValueError("the task file must be alphabetical: task ids are file order in both the graph and the client")
    written = []
    for task in tasks:
        out = hdf5_dir / f"{task}_demo.hdf5"
        if out.exists():
            logging.info("%s: present", out.name)
        else:
            n = convert_task(raw_root / task, arm, out)
            logging.info("%s: %d demos", out.name, n)
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True, help="a lane tree, configs/trees/<suite>.yaml with arm:")
    p.add_argument("--raw", default=str(default_paths().data / "robotwin" / "raw"),
                   help="downloaded tasks, one directory each")
    args = p.parse_args(argv)
    for path in convert_suite(args.suite, Path(args.raw)):
        print(path)


if __name__ == "__main__":
    main()
