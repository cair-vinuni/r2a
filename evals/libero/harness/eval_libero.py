"""LIBERO simulator client: one shard of one (task suite, mode) cell, on any upstream-API checkout.

Drives a registered benchmark of the `libero` package on the path -- the upstream
Lifelong-Robot-Learning/LIBERO checkout (--bench libero)
-- against the websocket policy server, applies the deploy chain through
robot_ood.runtime.harness.DeployHooks, and writes one mp4 per episode named the way
evals/libero_plus/score.py and mcnemar.py read. --protocol names the policy's own rollout recipe
(robot_ood.core.protocol): initial states, settling steps, step budget, seed, frame side, horizon.

Usage:
    python evals/libero/harness/eval_libero.py --policy-path <ckpt> --port 14090 \\
        --task-suite libero_10 --trials-per-task 50 --video-out <dir> --protocol openvla_oft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import time

import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from model2libero_interface import M1Inference  # the shared websocket policy client

from robot_ood.eval.episodes import (EPISODES_JSONL, VideoWriter, deal_episodes, shard_episodes,
                                     video_mode)
from robot_ood.core.protocol import BINARIZE, FIXED, Protocol, get_protocol, protocol_ids
from robot_ood.runtime.harness import DeployHooks

DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RESOLUTION = 256
BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The client's command line, as evals/libero/bench.yaml spells it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="policy checkpoint, for provenance")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--task-suite", required=True, help="a registered LIBERO benchmark name")
    parser.add_argument("--protocol", required=True, choices=protocol_ids(),
                        help="the policy's rollout recipe")
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--num-steps-wait", type=int, default=None,
                        help="settling steps (default: the protocol's)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--episodes", default=None,
                        help="json file of [task, episode] pairs to run; default is the whole grid")
    parser.add_argument("--video-out", required=True)
    parser.add_argument("--seed", type=int, default=None, help="default: the protocol's")
    return parser.parse_args(argv)


def base_suite(task_suite: str) -> str:
    """The unperturbed suite a perturbed variant derives from.

    Args:
        task_suite: A benchmark name such as libero_10_swap.

    Returns:
        The longest base suite name that the name starts with.

    Raises:
        ValueError: The name derives from no known suite.
    """
    for name in sorted(BASE_SUITES, key=len, reverse=True):
        if task_suite == name or task_suite.startswith(name + "_"):
            return name
    raise ValueError(f"unknown task suite {task_suite!r} (known roots: {sorted(BASE_SUITES)})")


def episode_name(task_description: str, max_len: int = 80) -> str:
    """The mp4 stem for one task, matched across methods by the paired scorer."""
    digest = hashlib.md5(task_description.encode()).hexdigest()[:8]
    return f"{task_description.replace(' ', '_')[:max_len]}_{digest}"


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert a robosuite xyzw quaternion to an axis-angle vector."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(w)) / den


def binarize_gripper(open_val: np.ndarray) -> np.ndarray:
    """The policy's continuous open signal as LIBERO's +1 (close) / -1 (open) command."""
    value = float(np.asarray(open_val, dtype=np.float32).reshape(-1)[0])
    return np.asarray([1.0 - 2.0 * (value > 0.5)], dtype=np.float32)


_LANGUAGE_RE = re.compile(r"\(:language\s+(.*?)\)", re.DOTALL)


def bddl_instruction(bddl: pathlib.Path, fallback: str) -> str:
    """The instruction a task actually states, read from its own definition.

    Args:
        bddl: The task's BDDL file.
        fallback: What to use when the file states none.

    Returns:
        The (:language ...) text, whitespace collapsed.
    """
    # The benchmark registry derives Task.language from the FILENAME, so a suite that rewords only
    # the instruction -- the Semantic dimension -- would otherwise reach the policy unperturbed.
    match = _LANGUAGE_RE.search(bddl.read_text())
    return " ".join(match.group(1).split()) if match else fallback


def make_env(task, resolution: int, seed: int):
    """Build the offscreen environment for one task.

    Args:
        task: A libero Task record.
        resolution: Camera height and width in pixels.
        seed: Environment seed; it moves object positions even under a fixed initial state.

    Returns:
        The environment and the task's language instruction.
    """
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution,
                             camera_widths=resolution)
    env.seed(seed)
    return env, bddl_instruction(bddl, task.language)


def build_suite(name: str):
    """Construct a registered LIBERO benchmark by name.

    Raises:
        ValueError: The name is not registered, or its task files were never installed.
    """
    registry = benchmark.get_benchmark_dict()
    if name.lower() not in registry:
        raise ValueError(f"{name!r} is not a registered LIBERO benchmark")
    suite = registry[name.lower()]()
    if not suite.n_tasks:
        raise ValueError(f"{name!r} has no tasks: install its bddl_files and init_files first "
                         f"(see evals/libero/setup.sh)")
    return suite


def observation_for(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The policy's two views and proprioceptive state for one step.

    Args:
        obs: The raw simulator observation.

    Returns:
        The 180-degree-rotated agent view, the rotated wrist view, and
        concat(ee_pos, ee_axis_angle, gripper_qpos).
    """
    view = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]))
    return view, wrist, state


def policy_action(response: dict, gripper_mode: str) -> np.ndarray:
    """The 7-vector the environment steps on.

    Args:
        response: The policy client's answer for this step.
        gripper_mode: The protocol's gripper reading, binarize or native.

    Returns:
        concat(world_vector, rotation_delta, gripper).

    Raises:
        ValueError: The server returned an action of the wrong shape.
    """
    raw = response["raw_action"]
    world = np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1)
    rotation = np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(raw["open_gripper"], dtype=np.float32).reshape(-1)
    if (world.size, rotation.size, gripper.size) != (3, 3, 1):
        raise ValueError(f"bad action sizes: {world.shape}, {rotation.shape}, {gripper.shape}")
    if gripper_mode == BINARIZE:
        gripper = binarize_gripper(gripper)
    return np.concatenate([world, rotation, gripper])


def run_episode(env, model: M1Inference, hooks: DeployHooks, task_description: str,
                initial_state, args: argparse.Namespace, protocol: Protocol,
                max_steps: int, keep_frames: bool = True) -> tuple[bool, int, list]:
    """Roll one episode out to success or the step budget.

    Args:
        env: The simulator environment, already constructed for this task.
        model: The websocket policy client.
        hooks: The deploy chain this arm turned on.
        task_description: The language instruction the policy is given.
        initial_state: The benchmark's initial state for this episode.
        args: The parsed command line.
        protocol: The policy's rollout recipe.
        max_steps: The policy's step budget, before the settling steps.
        keep_frames: Whether to collect the replay frames.

    Returns:
        Whether the task was solved, the policy steps taken, and the replay frames (empty when
        not kept).
    """
    model.reset(task_description=task_description)
    hooks.reset(args.shard)
    obs = env.reset()
    if protocol.init_states == FIXED:
        obs = env.set_init_state(initial_state)

    frames: list[np.ndarray] = []
    record = frames.append if keep_frames else (lambda _frame: None)
    t, step, done = 0, 0, False
    while t < max_steps + args.num_steps_wait:
        # The simulator drops objects at reset; nothing is commanded until they have settled.
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(DUMMY_ACTION)
            t += 1
            continue
        if t == args.num_steps_wait:
            obs, t = hooks.correct_entry(env, obs, t, record)

        view, wrist, state = observation_for(obs)
        blend = hooks.plan(env, state, step, shard=args.shard, task=task_description)
        record(view)

        response = model.step(images=[view, wrist], task_description=task_description, step=step,
                              blend=blend, state=np.expand_dims(state, axis=0))
        action = policy_action(response, protocol.gripper)
        hooks.record_action(action, step, args.shard)

        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1
        step += 1
    return bool(done), step + int(done), frames


def main(argv: list[str] | None = None) -> None:
    """Run this shard's slice of the task suite."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    protocol = get_protocol(args.protocol)
    if args.seed is None:
        args.seed = protocol.seed
    if args.num_steps_wait is None:
        args.num_steps_wait = protocol.settle_steps
    np.random.seed(args.seed)

    suite = build_suite(args.task_suite)
    root = base_suite(args.task_suite)
    max_steps = protocol.step_budget(root)
    out_dir = pathlib.Path(args.video_out)
    if args.num_shards > 1:
        out_dir = out_dir / f"shard{args.shard}_of_{args.num_shards}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = M1Inference(host=args.host, port=args.port, image_size=[protocol.wire_image_size] * 2,
                        worker_id=args.shard, exec_horizon=protocol.exec_horizon)
    hooks = DeployHooks.from_env(root)
    logging.info("suite=%s tasks=%d shard=%d/%d hooks=%s", args.task_suite, suite.n_tasks,
                 args.shard, args.num_shards, hooks.enabled)
    logging.info("protocol=%s init=%s settle=%d max_steps=%d seed=%d wire=%dpx horizon=%d gripper=%s",
                 protocol.id, protocol.init_states, args.num_steps_wait, max_steps, args.seed,
                 protocol.wire_image_size, protocol.exec_horizon, protocol.gripper)

    plan = (deal_episodes(json.loads(pathlib.Path(args.episodes).read_text()), args.shard, args.num_shards)
            if args.episodes else
            shard_episodes(suite.n_tasks, args.trials_per_task, args.shard, args.num_shards))
    logging.info("episodes=%d of %d, video=%s", len(plan), suite.n_tasks * args.trials_per_task, video_mode())
    episodes = successes = 0
    env, current, initial_states, task_description, stem = None, None, [], "", ""
    with VideoWriter(video_mode()) as videos:
        try:
            for task_id, episode_idx in tqdm.tqdm(plan):
                if task_id != current:
                    if env is not None:
                        env.close()
                    task = suite.get_task(task_id)
                    initial_states = suite.get_task_init_states(task_id)
                    env, task_description = make_env(task, ENV_RESOLUTION, args.seed)
                    stem, current = episode_name(task_description), task_id
                if episode_idx >= len(initial_states):
                    continue
                started = time.monotonic()
                solved, steps, frames = run_episode(env, model, hooks, task_description,
                                                    initial_states[episode_idx], args, protocol,
                                                    max_steps, keep_frames=videos.keep())
                episodes += 1
                successes += solved
                suffix = "success" if solved else "failure"
                videos.write(out_dir / f"rollout_{stem}_t{task_id}_episode{episode_idx}_{suffix}.mp4",
                             frames, solved)
                record = {"task_id": int(task_id), "shard": int(args.shard), "desc": task_description,
                          "episode": int(episode_idx), "success": solved, "steps": steps,
                          "wall_s": round(time.monotonic() - started, 1)}
                with open(out_dir / EPISODES_JSONL, "a") as f:
                    f.write(json.dumps(record) + "\n")
                logging.info("EPRESULT " + json.dumps(record))
        finally:
            if env is not None:
                env.close()
    logging.info("total %d/%d = %.1f%%", successes, episodes,
                 100.0 * successes / episodes if episodes else float("nan"))


if __name__ == "__main__":
    main()
