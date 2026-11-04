import dataclasses
import json
from typing import Optional
import logging
import math
import os
import pathlib

import time
import numpy as np
import tqdm
import tyro
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model2libero_interface import M1Inference  # vendored sibling module (harness/)

from robot_ood.eval.episodes import (EPISODES_JSONL, VideoWriter, deal_episodes, shard_episodes,
                                     video_mode)
from robot_ood.core.protocol import BINARIZE, FIXED, get_protocol
from robot_ood.runtime.harness import DeployHooks


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)

import hashlib

def short_name(text, max_len=80):
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    clean = text.replace(" ", "_")[:max_len]
    return f"{clean}_{h}"

@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10
    num_steps_wait: Optional[int] = None  # settling steps; None takes the protocol's
    num_trials_per_task: int = 50  # Number of rollouts per task
    category_value: str = "Background Textures"
            #Background Textures
        #Camera Viewpoints
        #Language Instructions
        #Light Conditions
        #Objects Layout
        #Robot Initial States
        #Sensor Noise

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: Optional[int] = None  # None takes the protocol's
    protocol: str = "openvla_oft"  # the policy's rollout recipe, robot_ood.core.protocol

    pretrained_path: str = ""   # accepted from bench.yaml for provenance; the server owns the checkpoint

    #################################################################################################################
    # Sharding (sim-client pool): split tasks across N parallel clients hitting one server
    #################################################################################################################
    shard_index: int = 0   # this client's shard id in [0, num_shards)
    num_shards: int = 1     # total number of parallel sim clients sharing the server
    episodes: Optional[str] = None   # json file of [task, episode] pairs; default is the whole grid


def eval_libero(args: Args) -> None:
    protocol = get_protocol(args.protocol)
    if args.seed is None:
        args.seed = protocol.seed
    if args.num_steps_wait is None:
        args.num_steps_wait = protocol.settle_steps
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed (both numpy and python's random — benchmark id shuffling uses `random`)
    np.random.seed(args.seed)
    import random as _pyrandom
    _pyrandom.seed(args.seed)

    # LIBERO-Plus suites register every perturbed task; the bridge cuts the list to one
    # category (and the optional instance-id filter) without touching the checkout.
    import libero_plus_bridge
    task_suite = libero_plus_bridge.make_suite(args.task_suite_name, args.category_value)
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # Sharding: when running a pool of clients against one server, give each shard its own
    # output subdirectory so parallel clients don't overwrite each other's videos/logs.
    if args.num_shards > 1:
        args.video_out_path = str(pathlib.Path(args.video_out_path) / f"shard{args.shard_index}_of_{args.num_shards}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = protocol.step_budget(args.task_suite_name)
    logging.info(f"protocol={protocol.id} init={protocol.init_states} settle={args.num_steps_wait} "
                 f"max_steps={max_steps} seed={args.seed} wire={protocol.wire_image_size}px "
                 f"horizon={protocol.exec_horizon} gripper={protocol.gripper}")

    model = M1Inference(
        host=args.host,
        port=args.port,
        image_size=[protocol.wire_image_size] * 2,
        worker_id=args.shard_index,  # stable per-client id so the server can key any per-worker state
        exec_horizon=protocol.exec_horizon,
    )

    # The deploy chain $MODE asks for, one per process. The upstream-LIBERO client holds the
    # same object, so a delta between the two benchmarks is the benchmark and not the wiring.
    hooks = DeployHooks.from_env(args.task_suite_name)

    plan = (deal_episodes(json.loads(pathlib.Path(args.episodes).read_text()), args.shard_index, args.num_shards)
            if args.episodes else
            shard_episodes(num_tasks_in_suite, args.num_trials_per_task, args.shard_index, args.num_shards))
    logging.info("episodes=%d of %d", len(plan), num_tasks_in_suite * args.num_trials_per_task)
    out_dir = pathlib.Path(args.video_out_path)
    total_episodes, total_successes = 0, 0
    env, current, initial_states, task_description = None, None, [], ""
    with VideoWriter(video_mode()) as videos:
        try:
            for task_id, episode_idx in tqdm.tqdm(plan):
                if task_id != current:
                    if env is not None:
                        env.close()
                    task = task_suite.get_task(task_id)
                    initial_states = task_suite.get_task_init_states(task_id)
                    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                    current = task_id
                if episode_idx >= len(initial_states):
                    continue
                started = time.monotonic()
                done, steps, replay_images = _run_episode(env, model, hooks, task_description,
                                                          initial_states[episode_idx], args, protocol,
                                                          max_steps, keep_frames=videos.keep())
                total_episodes += 1
                total_successes += int(done)
                suffix = "success" if done else "failure"
                task_segment = short_name(task_description.replace(" ", "_"))
                videos.write(out_dir / f"rollout_{task_segment}_t{task_id}_episode{episode_idx}_{suffix}.mp4",
                             replay_images, bool(done))
                record = {"task_id": int(task_id), "shard": int(args.shard_index), "desc": task_description,
                          "episode": int(episode_idx), "success": bool(done), "steps": steps,
                          "wall_s": round(time.monotonic() - started, 1)}
                with open(out_dir / EPISODES_JSONL, "a") as f:
                    f.write(json.dumps(record) + "\n")
                logging.info("EPRESULT " + json.dumps(record))
                logging.info(f"# episodes: {total_episodes}  successes: {total_successes} "
                             f"({total_successes / total_episodes * 100:.1f}%)")
        finally:
            if env is not None:
                env.close()

    logging.info(f"Total success rate: {total_successes / total_episodes if total_episodes else float('nan')}")
    logging.info(f"Total episodes: {total_episodes}")


def _run_episode(env, model, hooks, task_description, initial_state, args: Args, protocol, max_steps: int,
                 keep_frames: bool = True):
    """Roll one episode out to success or the step budget; returns (done, steps, replay frames)."""
    model.reset(task_description=task_description)  # Reset the client connection
    hooks.reset(args.shard_index)
    obs = env.reset()
    if protocol.init_states == FIXED:
        obs = env.set_init_state(initial_state)

    t = 0
    step = 0
    done = False
    replay_images = []
    record = replay_images.append if keep_frames else (lambda _frame: None)
    while t < max_steps + args.num_steps_wait:
        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
        # and we need to wait for them to fall
        if t < args.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        # ENTRY: fires once, on the first step the policy would otherwise act on, after the
        # perturbation has fully landed. Its control steps are charged to the episode.
        if t == args.num_steps_wait:
            obs, t = hooks.correct_entry(env, obs, t, record)

        # The one proprio read of this step: concat(ee_pos, ee_ori, gripper_qpos).
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )

        # SENTINEL: plan has to run BEFORE model.step, which overwrites the whole
        # pending-action buffer.
        blend_plan = hooks.plan(
            env, state, step, shard=args.shard_index, task=str(task_description),
        )

        # IMPORTANT: rotate 180 degrees to match train preprocessing
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        record(img)

        response = model.step(images=[img, wrist_img], task_description=str(task_description),
                              step=step, blend=blend_plan, state=np.expand_dims(state, axis=0))
        raw_action = response["raw_action"]

        world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
        gripper = _binarize_gripper_open(open_gripper) if protocol.gripper == BINARIZE else open_gripper

        if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
            raise ValueError(
                f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
            )
        delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

        # SENTINEL: feed the executed (post-blend) action into the commanded-action window
        # every env step, the same cadence as observe()'s query window -- FCSR's a_hist.
        hooks.record_action(delta_action, step, args.shard_index)

        obs, reward, done, info = env.step(delta_action.tolist())
        if done:
            break
        t += 1
        step += 1
    return bool(done), step + int(done), replay_images


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        # LIBERO-Plus env_wrapper does string membership checks (e.g. "_view_" in name),
        # so the bddl path must be a str, not a pathlib.Path.
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    tyro.cli(eval_libero)
