"""RoboTwin demonstrations as a LeRobot dataset, the way the pi0.5 finetune reads them.

RoboTwin ships fifty clean demonstrations per task (data/robotwin/raw/<task>/<setting>/): per
episode an hdf5 of the commanded joints and three jpeg camera streams, and a json of seen and
unseen instruction phrasings. openpi's Aloha pipeline trains on a LeRobot dataset with a
fourteen-dimensional `observation.state` and `action`, three `observation.images.<cam>` streams and
a `task` prompt per frame. This module writes that dataset the way RoboTwin's own conversion does
(policy/pi0/scripts/process_data.py followed by convert_aloha_data_to_lerobot_robotwin.py): the
state at a frame is its commanded joint vector, the action is the next frame's, the last frame is
dropped, and every frame of an episode carries one of the episode's `seen` phrasings.

Frames are stored at the cameras' native 320 x 240; the eval server sends the same size, and the
model pads-and-resizes either to 224 (`AlohaInputs`).

    external/openpi/.venv/bin/python evals/pi05/robotwin_data.py --suite g1 --repo-id robotwin_g1_clean

The dataset lands under $HF_LEROBOT_HOME/<repo_id>, by default pi05/lerobot/<repo_id>.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_LEROBOT_HOME", str(REPO_ROOT / "pi05" / "lerobot"))

RAW_ROOT = pathlib.Path("data/robotwin/raw")
SUITES = pathlib.Path("evals/robotwin/suites")
FPS = 50
IMAGE_HW = (240, 320)
# RoboTwin's camera streams under observation/<name>/rgb, and the openpi Aloha camera they feed.
CAMERAS = {"head_camera": "cam_high", "left_camera": "cam_left_wrist", "right_camera": "cam_right_wrist"}
MOTORS = [f"{arm}_{j}" for arm in ("left", "right")
          for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]


def suite_tasks(suite: str) -> list[str]:
    """The task names of evals/robotwin/suites/<suite>.txt, comments and blanks dropped."""
    lines = (SUITES / f"{suite}.txt").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def episode_files(task: str, setting: str, n: int) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """(hdf5, instructions json) of the first n episodes of a task, in episode order."""
    root = RAW_ROOT / task / setting
    pairs = []
    for i in range(n):
        h5, js = root / "data" / f"episode{i}.hdf5", root / "instructions" / f"episode{i}.json"
        if not h5.exists() or not js.exists():
            raise FileNotFoundError(f"{task}: episode {i} of {setting} missing under {root}")
        pairs.append((h5, js))
    return pairs


def load_episode(h5_path: pathlib.Path) -> tuple[np.ndarray, dict[str, list[bytes]]]:
    """The commanded joint vectors [T, 14] and the jpeg bytes of each camera, per frame."""
    from robot_ood.data.libero import open_hdf5

    with open_hdf5(h5_path) as f:
        ja = f["joint_action"]
        joints = np.concatenate([ja["left_arm"][()], ja["left_gripper"][()][:, None],
                                 ja["right_arm"][()], ja["right_gripper"][()][:, None]], axis=1)
        if "vector" in ja and not np.allclose(joints, ja["vector"][()]):
            raise ValueError(f"{h5_path}: joint_action/vector is not [left, l_grip, right, r_grip]")
        jpegs = {cam: list(f["observation"][cam]["rgb"][()]) for cam in CAMERAS}
    return joints.astype(np.float32), jpegs


def decode(jpeg: bytes) -> np.ndarray:
    """A jpeg frame as RGB [H, W, 3] uint8 at the native size."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img.shape[:2] != IMAGE_HW:
        raise ValueError(f"frame is {img.shape[:2]}, expected {IMAGE_HW}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_dataset(repo_id: str, root: pathlib.Path, writer_processes: int, writer_threads: int):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        shutil.rmtree(root)
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
    }
    for cam in CAMERAS.values():
        features[f"observation.images.{cam}"] = {"dtype": "image", "shape": (3, *IMAGE_HW),
                                                  "names": ["channels", "height", "width"]}
    return LeRobotDataset.create(
        repo_id=repo_id, root=root, fps=FPS, robot_type="aloha", features=features, use_videos=False,
        image_writer_processes=writer_processes, image_writer_threads=writer_threads,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--suite", default="g1", help="evals/robotwin/suites/<suite>.txt")
    p.add_argument("--setting", default="aloha-agilex_clean_50", help="demo directory under each task")
    p.add_argument("--episodes", type=int, default=50, help="episodes per task, from episode0")
    p.add_argument("--repo-id", required=True, help="dataset name under $HF_LEROBOT_HOME")
    p.add_argument("--root", default=None, help="dataset directory (default $HF_LEROBOT_HOME/<repo-id>)")
    p.add_argument("--seed", type=int, default=0, help="draw of the seen phrasing per episode")
    p.add_argument("--writer-processes", type=int, default=8)
    p.add_argument("--writer-threads", type=int, default=4)
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    root = pathlib.Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    tasks = suite_tasks(args.suite)
    plan = [(task, h5, js) for task in tasks for h5, js in episode_files(task, args.setting, args.episodes)]
    rng = np.random.default_rng(args.seed)
    dataset = create_dataset(args.repo_id, root, args.writer_processes, args.writer_threads)
    print(f"{len(plan)} episodes of {len(tasks)} tasks -> {root}", flush=True)

    frames = 0
    for k, (task, h5, js) in enumerate(plan):
        joints, jpegs = load_episode(h5)
        seen = json.loads(js.read_text())["seen"]
        prompt = str(seen[rng.integers(len(seen))])
        for t in range(len(joints) - 1):
            frame = {"observation.state": joints[t], "action": joints[t + 1], "task": prompt}
            for cam, name in CAMERAS.items():
                frame[f"observation.images.{name}"] = decode(jpegs[cam][t])
            dataset.add_frame(frame)
        dataset.save_episode()
        frames += len(joints) - 1
        print(f"[{k + 1}/{len(plan)}] {task} {h5.stem}: {len(joints) - 1} frames  ({frames} total)", flush=True)

    print(f"done: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames, "
          f"{dataset.meta.total_tasks} prompts -> {root}")


if __name__ == "__main__":
    main()
