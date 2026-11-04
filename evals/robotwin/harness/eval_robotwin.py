"""RoboTwin 2.0 simulator client: one shard of one (task list, task config) cell.

Drives the tasks of a suite file against the websocket policy server the way RoboTwin's own
XPolicyLab evaluation (external/RoboTwin/scripts/eval_policy.py) does: an episode is one of the
cached expert seeds, described by its expert demonstration for the instruction, the policy is
queried once per chunk, every joint target of the chunk goes to env.take_action(action_type="qpos"),
and the episode ends on success or the benchmark's own per-task step limit. --protocol names the
policy's rollout recipe (robot_ood.core.protocol). Under fcsr each arm has its own deploy chain
(robot_ood.runtime.harness.DeployHooks) built from the lane tree the suite names for that arm: the
chain sees the arm as a joint-space proxy configuration and corrects the chunk's joint steps
before they become targets.

The observation is rendered only where the policy reads it -- before each query -- and per step
only while a replay video is being kept; rendering never touches the simulation.

Usage:
    python evals/robotwin/harness/eval_robotwin.py --policy-path <ckpt> --port 14090 \\
        --task-suite g1 --task-config demo_randomized --trials-per-task 20 --video-out <dir> --protocol pi05_robotwin
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import time

import numpy as np
import tqdm

import robotwin_env   # noqa: F401  (puts the checkout on the path and moves into it first)
from robotwin_env import (INSTRUCTION_KINDS, episode_instruction, episode_seeds, load_task_env,
                          task_names)
from tools.websocket_policy_client import WebsocketClientPolicy

from robot_ood.core.absolute import deltas_from_targets, targets_from_deltas
from robot_ood.core.action import ACTION_DIM, EUCLIDEAN, GRIP_COL
from robot_ood.core.paths import lane_graph_dirs, lane_suites
from robot_ood.core.protocol import BENCH_STEP_LIMIT, Protocol, get_protocol, protocol_ids
from robot_ood.data.robotwin import (ARMS, ARM_JOINTS, JOINT_DIM, joint_rows, lane_step_row, lane_targets,
                                     open_flag)
from robot_ood.eval.episodes import (EPISODES_JSONL, TRACE_PREFIX, EpisodeTrace, VideoWriter, deal_episodes,
                                     shard_episodes,
                                     video_mode)
from robot_ood.runtime.harness import DeployHooks
from robot_ood.runtime.pair import PairedRetrieval
from robot_ood.runtime.partner import PartnerPrior

VIEWS = ("head_camera", "left_camera", "right_camera")
SHADER = os.environ.get("ROBOTWIN_SHADER", "rt")   # the render shader the checkout was patched to read
# SAPIEN's render cache keeps every texture a randomised scene loaded; dropping it after each
# episode holds a client's GPU memory flat and changes nothing the policy sees.
CLEAR_CACHE_EVERY = 1
# Scenes a seed may come up unstable on one environment object before a fresh one is built.
MAX_UNSTABLE_SCENES = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The client's command line, as evals/robotwin/bench.yaml spells it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="policy checkpoint, for provenance")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14090)
    parser.add_argument("--task-suite", required=True, help="a suite file under evals/robotwin/suites")
    parser.add_argument("--task-config", required=True, help="a task config under task_config/")
    parser.add_argument("--protocol", required=True, choices=protocol_ids())
    parser.add_argument("--instruction-type", default="unseen", choices=INSTRUCTION_KINDS,
                        help="which of the benchmark's instruction pools an episode is phrased from")
    parser.add_argument("--trials-per-task", type=int, default=20)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--episodes", default=None,
                        help="json file of [task, episode] pairs to run; default is the whole grid")
    parser.add_argument("--video-out", required=True)
    parser.add_argument("--seed", type=int, default=None, help="default: the protocol's")
    return parser.parse_args(argv)


class LaneEnv:
    """One arm of the task environment as the deploy chain sees a robot: a joint-space proxy.

    The chain reads the arm through onf_q_arm(): [six joint targets, open flag], the same seven
    numbers that arm's corpus was built from, and moves it through step(): one corpus-convention
    action, joint deltas and a gripper command, while the other arm holds its targets.
    """

    def __init__(self, env, arm: str, worker: int) -> None:
        self.env, self.arm, self.worker = env, arm, worker

    def step(self, action: np.ndarray) -> tuple[None]:
        """Apply one [7] corpus action to this arm: joint deltas, then the gripper command (open below zero).

        Args:
            action: [7] the corrector's action in the corpus's own convention.

        Returns:
            A one-tuple of None: the lane has no observation of its own; the client reads the
            environment's after the walk.
        """
        states = {arm: np.asarray(getattr(self.env.robot, f"get_{arm}_arm_jointState")(), dtype=np.float64)
                  for arm in ARMS}
        self.env.take_action(lane_step_row(states, self.arm, action), action_type="qpos")
        return (None,)

    def onf_render(self) -> np.ndarray:
        """The head camera's frame, for the replay."""
        return np.asarray(self.env.get_obs()["observation"]["head_camera"]["rgb"])

    def joints(self) -> np.ndarray:
        """[7] the arm's commanded joints and gripper fraction, RoboTwin's own reading."""
        return np.asarray(getattr(self.env.robot, f"get_{self.arm}_arm_jointState")(), dtype=np.float64)

    def onf_q_arm(self) -> np.ndarray:
        """[7] the proxy configuration: six joints, then 1.0 open / 0.0 closed."""
        state = self.joints()
        return np.concatenate([state[:ARM_JOINTS], open_flag(state[ARM_JOINTS:ARM_JOINTS + 1])])

    def state(self) -> np.ndarray:
        """What DeployHooks.plan reads: the configuration, whose last entry the grip flag thresholds."""
        return self.onf_q_arm()


def load_chains(suite: str, chunk: int) -> dict[str, DeployHooks]:
    """One deploy chain per arm, each from the lane tree the suite's tree names for it; {} under base.

    Args:
        suite: The bimanual suite. ROBOT_OOD_TREE, when the driver set it, names the tree of that
            suite built against the served policy; else the suite is its own tree.
        chunk: Rows the policy executes per query.

    Returns:
        arm -> its chain, over exactly the arms the tree names.

    Raises:
        ValueError: fcsr was asked for and the tree names no lane for one of the arms.
    """
    if os.environ.get("MODE", "base") == "base":
        return {}
    tree = os.environ.get("ROBOT_OOD_TREE") or suite
    lanes = lane_suites(tree)
    if set(lanes) != set(ARMS):
        raise ValueError(f"configs/trees/{tree}.yaml must name a lane tree per arm, got {sorted(lanes)}")
    artifacts = lane_graph_dirs(tree)
    return {arm: DeployHooks.for_graph(lane, artifacts[arm], chunk) for arm, lane in lanes.items()}


def load_paired(suite: str, chains: dict[str, DeployHooks]) -> PairedRetrieval | None:
    """One retrieval over both lanes, under ROBOT_OOD_PAIR=1 with two chains; else None."""
    if len(chains) != 2 or not PAIRED:
        return None
    return PairedRetrieval.from_lanes(lane_graph_dirs(os.environ.get("ROBOT_OOD_TREE") or suite))


def load_partner(suite: str, chains: dict[str, DeployHooks], chunk: int) -> PartnerPrior | None:
    """The other arm's activity as a prior on each arm's retrieval, under ROBOT_OOD_PARTNER=1; else None.

    Args:
        suite: The bimanual suite, or the tree ROBOT_OOD_TREE names for it.
        chains: The per-arm chains; anything but two leaves the prior off.
        chunk: Rows the policy executes per query; the prior reads motion over one plan's rows.

    Returns:
        The prior, or None.
    """
    if len(chains) != 2 or os.environ.get("ROBOT_OOD_PARTNER", "0") not in ("1", "2"):
        return None
    return PartnerPrior.from_lanes(lane_graph_dirs(os.environ.get("ROBOT_OOD_TREE") or suite),
                                   plan_span(chains, chunk))


class ChunkPolicy:
    """The websocket policy client: one query per chunk, the model's joint targets back."""

    def __init__(self, host: str, port: int, worker_id: int) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.worker_id = int(worker_id)

    def chunk(self, obs: dict, instruction: str, step: int, seed: int) -> np.ndarray:
        """The model's next chunk for one observation.

        Args:
            obs: env.get_obs().
            instruction: The task's instruction for this episode.
            step: The env step so far, whose noise the chunk samples with.
            seed: The episode's scene seed; with the step it fixes the chunk's sampling noise, so
                the episode plays out the same whichever clients share the server's batches.

        Returns:
            [T, 14] absolute joint targets in RoboTwin's joint_action/vector layout.
        """
        views = [np.asarray(obs["observation"][v]["rgb"]) for v in VIEWS]
        state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        payload = {"batch_images": [views], "instructions": [instruction], "state": [state],
                   "episode_step": [step], "worker_id": [self.worker_id],
                   "seed": [int(seed)]}
        answer = self.client.infer(payload)["data"]
        chunk = np.asarray(answer["actions"], dtype=np.float64)[0]
        if chunk.ndim != 2 or chunk.shape[1] != JOINT_DIM:
            raise ValueError(f"policy chunk must be [T, {JOINT_DIM}], got {chunk.shape}")
        return chunk


# How the policy's steps are measured at a check that falls inside one query's chunk. The policy
# cache the corrector trained on differences each chunk row from the row before it (row 0 from the
# joints the policy was queried at), so a policy step is the policy's own motion. "live" differences
# the remaining rows from the joints the corrected arm actually reached, so after any correction
# the first policy step carries the whole accumulated deviation as a snap back toward the stale
# absolute targets: a step the corrector never saw in training and reads as a corrupted policy.
# "plan" measures the steps the way the cache did and executes them from the live joints.
POLICY_DELTAS = os.environ.get("ROBOT_OOD_POLICY_DELTAS", "live")
# ROBOT_OOD_PARTNER=2: before the partner prior has observed a window of the other arm, the
# policy's own plan for that arm says whether it moves (the arms share the rest pose at an
# episode's start, so nothing else tells an idle-arm demo from an active-arm one there).
PARTNER_PLAN = os.environ.get("ROBOT_OOD_PARTNER", "0") == "2"
# ROBOT_OOD_PAIR=1: one retrieval for both arms; each lane's shortlist is cut on its own tier-1
# score plus its partner lane's over the paired (same demo, same frames) segments.
PAIRED = os.environ.get("ROBOT_OOD_PAIR", "0") == "1"
# ROBOT_OOD_REQUERY=<rad>: the policy keeps its own horizon, but a plan the corrector moved by at
# least this much (any row, any joint) invalidates the rest of the chunk: the next check step
# queries the policy again from the live state instead of reading the stale rows, as a short
# horizon does on every check. 0 or unset keeps the fixed horizon.
REQUERY_TOL = float(os.environ.get("ROBOT_OOD_REQUERY", "0") or 0)
# ROBOT_OOD_HORIZON_BY_MOTION=1: the horizon counts the rows the policy moves on (some arm's mean
# joint step at least MOTION_EPS), not every row. A chunk that pauses before it acts (the model
# waits, then presses or reaches: one chunk in three on the press and handover tasks at H=50) then
# runs its pause and its first horizon rows of motion before the next query, where a fixed short
# horizon replays the pause on every query and never reaches the motion. 0 keeps the fixed horizon.
HORIZON_BY_MOTION = os.environ.get("ROBOT_OOD_HORIZON_BY_MOTION", "0") == "1"
MOTION_EPS = 0.003
# ROBOT_OOD_CONTACT_REQUERY=1: at a plan boundary, the policy is queried again from the live state
# when its chunk opens or closes a gripper within one plan span either side, so grasps and
# releases run closed loop at the plan's rate while the rest of the chunk keeps the horizon.
CONTACT_REQUERY = os.environ.get("ROBOT_OOD_CONTACT_REQUERY", "0") == "1"
# ROBOT_OOD_GRASP_SERVO=<rows>: for that many steps after an arm's gripper closes, the policy is
# queried again every SERVO_ROWS rows, so the carry after a grasp runs closed loop on the camera;
# the rest of the episode keeps the protocol's horizon. 0 or unset keeps the fixed horizon.
GRASP_SERVO = int(os.environ.get("ROBOT_OOD_GRASP_SERVO", "0") or 0)
SERVO_ROWS = 10
# ROBOT_OOD_SERVO_HOLD=1: an arm servos only while its gripper stays closed, so a release ends it.
SERVO_HOLD = os.environ.get("ROBOT_OOD_SERVO_HOLD", "0") == "1"


def corrected_rows(chunk: np.ndarray, lanes: dict[str, LaneEnv], blends: dict,
                   anchors: dict[str, np.ndarray | None] | None = None,
                   ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """env.take_action rows for a chunk, each blended arm's joint steps corrected by the chain.

    Args:
        chunk: [T, 14] the model's joint targets.
        lanes: arm -> LaneEnv, for the live joints the steps are measured from.
        blends: arm -> BlendPlan or None.
        anchors: arm -> the model's [6] joint target just before this chunk's first row, or None
            at a query's first row; read under ROBOT_OOD_POLICY_DELTAS=plan.

    Returns:
        The [T, 14] rows, per arm the [T, 7] executed steps (joint deltas, gripper command) the
        chain records, and per arm the model's own [T, 7] steps before any correction. Without
        any blend the rows are the model's own, bit for bit.
    """
    executed, policy, targets, grips, live = {}, {}, {}, {}, {}
    for arm in ARMS:
        abs_targets, grip = lane_targets(chunk, arm)
        lane = lanes.get(arm)
        deltas = np.zeros((len(chunk), ACTION_DIM), dtype=np.float32)
        deltas[:, GRIP_COL] = 1.0 - 2.0 * open_flag(grip)
        if lane is not None:
            live[arm] = lane.joints()[:ARM_JOINTS]
            origin = live[arm]
            if POLICY_DELTAS == "plan" and anchors is not None and anchors.get(arm) is not None:
                origin = np.asarray(anchors[arm], dtype=np.float64)
            deltas[:, :ARM_JOINTS] = deltas_from_targets(origin, abs_targets, EUCLIDEAN)
        policy[arm] = deltas
        targets[arm], grips[arm] = abs_targets, grip     # the chain leaves the gripper alone
    # An arm is stalled only while the whole robot is: a pause while the other arm works is the
    # model's own turn-taking, which the stall push must leave alone.
    motion = {arm: float(np.abs(policy[arm][:, :ARM_JOINTS]).mean()) for arm in ARMS}
    still = {arm: motion[arm] < MOTION_EPS for arm in ARMS}
    for arm in ARMS:
        lane, deltas = lanes.get(arm), policy[arm].copy()
        if lane is not None and blends.get(arm) is not None:
            stall = blends[arm].authority_ctx.get("stall")
            if stall is not None:
                stall["partner_still"] = all(still[other] for other in ARMS if other != arm)
                stall["partner_motion"] = max(motion[other] for other in ARMS if other != arm)
            deltas = np.asarray(blends[arm].apply(deltas), dtype=np.float32)
        if lane is not None and (blends.get(arm) is not None or POLICY_DELTAS == "plan"):
            targets[arm] = targets_from_deltas(live[arm], deltas[:, :ARM_JOINTS], EUCLIDEAN)
        executed[arm] = deltas
    if not any(b is not None for b in blends.values()):
        return chunk, executed, policy
    return joint_rows(targets, grips), executed, policy


def moving_rows(chunk: np.ndarray, env) -> np.ndarray:
    """[T] bool: the rows of a chunk on which some arm's mean joint step is at least MOTION_EPS.

    Args:
        chunk: [T, 14] the model's joint targets.
        env: The benchmark environment, for the live joints the first row's step is measured from.
    """
    moving = np.zeros(len(chunk), dtype=bool)
    for arm in ARMS:
        targets, _ = lane_targets(chunk, arm)
        joints = np.asarray(getattr(env.robot, f"get_{arm}_arm_jointState")(), dtype=np.float64)[:ARM_JOINTS]
        steps = np.abs(np.diff(np.vstack([joints[None], targets]), axis=0)).mean(axis=1)
        moving |= steps >= MOTION_EPS
    return moving


def grip_switches(chunk: np.ndarray) -> np.ndarray:
    """[T] bool: the rows of a chunk on which some arm's open flag differs from the row before.

    Args:
        chunk: [T, 14] the model's joint targets and gripper fractions.
    """
    switch = np.zeros(len(chunk), dtype=bool)
    for arm in ARMS:
        flag = open_flag(lane_targets(chunk, arm)[1])
        switch[1:] |= flag[1:] != flag[:-1]
    return switch


def plan_span(chains: dict[str, DeployHooks], chunk: int) -> int:
    """Rows one plan covers, shared by both arms so their checks fall on the same steps.

    Args:
        chains: arm -> DeployHooks, empty under base.
        chunk: Rows the policy executes per query.

    Returns:
        The row count, or the whole chunk when nothing plans.

    Raises:
        ValueError: The arms' graphs disagree on it.
    """
    rows = {arm: chain.plan_rows for arm, chain in chains.items()}
    if not rows:
        return chunk
    if len(set(rows.values())) != 1:
        raise ValueError(f"the arms' graphs disagree on the rows a plan covers: {rows}")
    return next(iter(rows.values()))


class SceneUnstable(RuntimeError):
    """A cached seed stayed unstable on this environment object; a fresh one may settle it."""


class EpisodeAborted(RuntimeError):
    """An episode raised twice; the shard stops so the cell is rerun rather than scored on a crash."""


def build_scene(env, args: dict, seed: int, episode: int) -> None:
    """setup_demo for one cached seed, rebuilding the scene while the benchmark reports it unstable.

    The stability check settles the scene for 500 physics steps. A seed the expert check accepted
    in a fresh process can come up unstable on an environment object that has run episodes
    before, and stay so; the caller then rebuilds the environment itself. The seed is kept, so
    base and fcsr still run identical episode lists.
    """
    from envs.utils.create_actor import UnStableError

    for attempt in range(MAX_UNSTABLE_SCENES):
        try:
            env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **args)
            return
        except UnStableError as exc:
            logging.info("seed %d scene unstable (%d): %s", seed, attempt + 1, exc)
            env.close_env(clear_cache=True)
    raise SceneUnstable(f"seed {seed}: {MAX_UNSTABLE_SCENES} unstable scenes on this environment")


def spare_seed(env, args: dict, task: str, task_config: str, run_seed: int, trials: int, episode: int) -> int:
    """The spare seed of one episode: the cached list's entry at trials + episode, searched if absent."""
    return episode_seeds(env, args, task, task_config, run_seed, trials + episode + 1)[trials + episode]


def step_budget(env, protocol: Protocol, task_suite: str) -> int:
    """Env steps this episode may take: the protocol's, or the benchmark's own per-task limit."""
    budget = protocol.step_budget(task_suite)
    return int(env.step_lim) if budget == BENCH_STEP_LIMIT else budget


def run_episode(env, args: dict, policy: ChunkPolicy, chains: dict[str, DeployHooks], instruction: str,
                seed: int, episode: int, protocol: Protocol, task_suite: str, keep_frames: bool,
                shard: int = 0, trace: EpisodeTrace | None = None,
                partner: PartnerPrior | None = None,
                paired: PairedRetrieval | None = None) -> tuple[bool, int, list, str]:
    """Roll one episode out to success or the step budget.

    Args:
        env: The task environment.
        args: Its setup_demo arguments.
        policy: The websocket policy client.
        chains: arm -> its deploy chain; empty under base.
        instruction: The language instruction the policy is given.
        seed: The episode's scene seed.
        episode: The episode index within the task.
        protocol: The policy's rollout recipe.
        task_suite: The suite, for the protocol's step budget.
        keep_frames: Whether to collect head-camera frames for a replay.
        shard: This client's shard id, the worker each arm's chain serves it under.
        trace: Receives every step and check of the corrected arms, or None.
        partner: The other arm's activity as a prior on each arm's retrieval, or None.
        paired: One retrieval over both lanes, or None.

    Returns:
        Whether the task was solved, the steps taken, the frames (empty when not kept), and an
        error string ("" when the rollout ran to its end).
    """
    lanes = {arm: LaneEnv(env, arm, worker=shard) for arm in chains}
    for arm, lane in lanes.items():
        chains[arm].reset(lane.worker)
    if partner is not None:
        partner.reset()
    build_scene(env, args, seed, episode)
    max_steps = step_budget(env, protocol, task_suite)
    env.step_lim = max(int(env.step_lim or 0), max_steps)   # the budget below, never the file's alone
    frames: list[np.ndarray] = []
    step, walked, solved, error, queries = 0, 0, False, "", 0
    span = plan_span(chains, protocol.exec_horizon)
    closed_at: dict[str, int] = {}
    was_closed = {arm: False for arm in ARMS}

    def servoing() -> bool:
        """Whether some arm's gripper closed within GRASP_SERVO steps; tracks the close events."""
        for arm in ARMS:
            now = open_flag(np.asarray(getattr(env.robot, f"get_{arm}_arm_jointState")(), dtype=np.float64)[ARM_JOINTS:ARM_JOINTS + 1])[0] < 0.5
            if now and not was_closed[arm]:
                closed_at[arm] = step + walked
            was_closed[arm] = now
        return any(step + walked - at < GRASP_SERVO and (was_closed[arm] or not SERVO_HOLD)
                   for arm, at in closed_at.items())

    def observe(step: int, q: dict[str, np.ndarray], planned: dict[str, np.ndarray] | None = None) -> dict:
        """Feed each arm's chain its configuration; on a check step, that chain's plan.

        planned: arm -> the policy's remaining absolute joint targets, read by the partner prior
        under ROBOT_OOD_PARTNER=2 while an episode has no observed window yet.
        """
        if partner is not None:
            partner.observe(q)
        if partner is not None and step % span == 0:
            forecast = planned if PARTNER_PLAN else None
            priors = {arm: partner.prior(arm, forecast) for arm in lanes}
        else:
            priors = {}
        if paired is not None:
            blends = paired.plan(chains, lanes, q, step, shard=shard, task=instruction)
        else:
            blends = {arm: chains[arm].plan(lane, q[arm], step, shard=lane.worker, task=instruction,
                                            seg_prior=priors.get(arm))
                      for arm, lane in lanes.items()}
        if partner is not None:
            partner.observe_phase({arm: blend.authority_ctx["query_phase"]
                                   for arm, blend in blends.items() if blend is not None})
        return blends

    try:
        env._update_render()
        env.actor_pose = True
        # Each arm's entry corrector walks it onto its corpus's start before the policy first
        # acts; those control steps are charged to the episode, as on every other benchmark.
        for arm, lane in lanes.items():
            _, walked = chains[arm].correct_entry(lane, None, walked, frames.append if keep_frames else None)
        obs = env.get_obs()
        while step + walked < max_steps and not solved:
            chunk = policy.chunk(obs, instruction, step + walked, seed)
            chunk = chunk if HORIZON_BY_MOTION else chunk[:protocol.exec_horizon]
            moving = moving_rows(chunk, env) if HORIZON_BY_MOTION else None
            switches = grip_switches(chunk) if CONTACT_REQUERY else None
            queries += 1
            rows, executed, first, moved, motion = None, None, 0, False, 0
            for k in range(len(chunk)):
                if GRASP_SERVO and servoing() and k > 0 and k % SERVO_ROWS == 0 and (not chains or k % span == 0):
                    break                    # a gripper closed recently: the carry runs closed loop
                if k % span == 0 and (moved or (moving is not None and motion >= protocol.exec_horizon)
                                      or (switches is not None and k > 0 and switches[max(0, k - span):k + span].any())):
                    break                    # the corrector moved the last plan, the horizon of motion ran out, or a grasp or release is near
                q_step = {arm: lane.state() for arm, lane in lanes.items()}
                planned = {arm: lane_targets(chunk[k:], arm)[0] for arm in lanes} if PARTNER_PLAN else None
                blends = observe(step, q_step, planned)
                if rows is None or any(b is not None for b in blends.values()):
                    anchors = {arm: (lane_targets(chunk[k - 1:k], arm)[0][0] if k > 0 else None) for arm in lanes}
                    rows, executed, policy_steps, first = *corrected_rows(chunk[k:k + span], lanes, blends, anchors), k
                    if REQUERY_TOL > 0:
                        moved = any(np.abs(executed[arm][:, :ARM_JOINTS] - policy_steps[arm][:, :ARM_JOINTS]).max() >= REQUERY_TOL
                                    for arm in lanes)
                if trace is not None:
                    for arm, blend in blends.items():
                        if blend is not None:
                            trace.check(step, arm, float(blend.authority_ctx["query_phase"]),
                                        int(blend.cands.idx[0]), float(blend.cands.score[0]),
                                        blend.decision)
                env.take_action(rows[k - first], action_type="qpos")
                for arm, lane in lanes.items():
                    chains[arm].record_action(executed[arm][k - first], step, lane.worker)
                if trace is not None:
                    trace.step(step, q_step, {arm: policy_steps[arm][k - first] for arm in lanes},
                               {arm: executed[arm][k - first] for arm in lanes})
                step += 1
                motion += int(moving[k]) if moving is not None else 0
                if keep_frames:
                    frames.append(np.asarray(env.get_obs()["observation"]["head_camera"]["rgb"]))
                if env.eval_success:
                    solved = True
                    break
                if step + walked >= max_steps:
                    break
            if solved or step + walked >= max_steps:
                break
            obs = env.get_obs()
    except Exception as exc:   # the benchmark's planner and physics can fail mid-episode
        logging.exception("episode %d seed %d aborted", episode, seed)
        error = f"{type(exc).__name__}: {exc}"
    logging.info("[QUERIES] episode=%d seed=%d queries=%d steps=%d", episode, seed, queries, step)
    return bool(solved), step + walked, frames, error


def main(argv: list[str] | None = None) -> None:
    """Run this shard's slice of the suite."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("curobo").setLevel(logging.WARNING)   # one line per plan otherwise
    args = parse_args(argv)
    protocol = get_protocol(args.protocol)
    # ROBOT_OOD_EXEC_HORIZON re-queries the policy after that many rows of its chunk. The chain
    # plans over plan_rows of a chunk and was trained on a chunk queried at the arm's current
    # state, so a horizon equal to plan_rows makes every check a fresh query, as on LIBERO.
    if os.environ.get("ROBOT_OOD_EXEC_HORIZON"):
        protocol = dataclasses.replace(protocol, exec_horizon=int(os.environ["ROBOT_OOD_EXEC_HORIZON"]))
    if args.seed is None:
        args.seed = protocol.seed
    tasks = task_names(args.task_suite)
    out_dir = pathlib.Path(args.video_out)
    if args.num_shards > 1:
        out_dir = out_dir / f"shard{args.shard}_of_{args.num_shards}"
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = ChunkPolicy(args.host, args.port, worker_id=args.shard)
    chains = load_chains(args.task_suite, protocol.exec_horizon)
    partner = load_partner(args.task_suite, chains, protocol.exec_horizon)
    paired = load_paired(args.task_suite, chains)
    plan = (deal_episodes(json.loads(pathlib.Path(args.episodes).read_text()), args.shard, args.num_shards)
            if args.episodes else
            shard_episodes(len(tasks), args.trials_per_task, args.shard, args.num_shards))
    logging.info("suite=%s config=%s tasks=%d shard=%d/%d episodes=%d chains=%s partner=%s paired=%s protocol=%s "
                 "horizon=%d instructions=%s seed=%d shader=%s video=%s", args.task_suite, args.task_config,
                 len(tasks), args.shard, args.num_shards, len(plan), sorted(chains), partner is not None, paired is not None,
                 protocol.id, protocol.exec_horizon, args.instruction_type,
                 args.seed, SHADER, video_mode())
    episodes = successes = 0
    env, env_args, current, seeds, done_here = None, {}, None, [], 0
    with VideoWriter(video_mode()) as videos:
        try:
            unbuildable: set[int] = set()
            for task_id, episode_idx in tqdm.tqdm(plan):
                if task_id in unbuildable:
                    continue
                if task_id != current:
                    if env is not None:
                        env.close_env(clear_cache=True)
                    env, env_args = load_task_env(tasks[task_id], args.task_config)
                    try:
                        seeds = episode_seeds(env, env_args, tasks[task_id], args.task_config, args.seed,
                                              args.trials_per_task)
                    except RuntimeError as exc:
                        # A perturbation can leave a task with no scene that settles at all -- the
                        # objects of put_bottles_dustbin fall over under every seed the object-pose
                        # noise of demo_objects_plus draws. That is the axis, not this shard: leave
                        # the task out of the cell, which then reports itself short, and carry on.
                        logging.error("%s: no episode runs on this axis", exc)
                        unbuildable.add(task_id)
                        continue
                    current, done_here = task_id, 0
                started = time.monotonic()
                seed = seeds[episode_idx]
                instruction = episode_instruction(env, env_args, seed, episode_idx, args.instruction_type,
                                                  args.trials_per_task)
                for attempt in range(3):
                    trace = EpisodeTrace(list(chains)) if chains else None
                    try:
                        solved, steps, frames, error = run_episode(
                            env, env_args, policy, chains, instruction, seed, episode_idx, protocol,
                            args.task_suite, keep_frames=videos.keep(), shard=args.shard, trace=trace,
                            partner=partner, paired=paired)
                    except SceneUnstable as exc:
                        env.close_env(clear_cache=True)
                        env, env_args = load_task_env(tasks[task_id], args.task_config)
                        if attempt == 0:
                            # a fresh environment object may settle what this one no longer does
                            logging.info("%s; rebuilding the environment", exc)
                        else:
                            # this process will not settle the seed: the episode's own spare seed,
                            # the one past the cell's trials at its index, is unique to it
                            seed = spare_seed(env, env_args, tasks[task_id], args.task_config, args.seed,
                                              args.trials_per_task, episode_idx)
                            logging.info("%s; spare seed %d", exc, seed)
                        error = str(exc)
                        continue
                    if not error:
                        break
                    # A crash (a GPU out of memory, a planner fault) is not a failed episode: drop
                    # the scene, try once more, and stop the shard rather than score the crash.
                    env.close_env(clear_cache=True)
                else:
                    raise EpisodeAborted(f"{tasks[task_id]} episode {episode_idx} seed {seed}: {error}")
                done_here += 1
                env.close_env(clear_cache=(done_here % CLEAR_CACHE_EVERY == 0))
                episodes += 1
                successes += solved
                suffix = "success" if solved else "failure"
                videos.write(out_dir / f"rollout_{tasks[task_id]}_t{task_id}_episode{episode_idx}_{suffix}.mp4",
                             frames, solved)
                if trace is not None:
                    trace.save(out_dir / f"{TRACE_PREFIX}{tasks[task_id]}_t{task_id}_episode{episode_idx}.npz")
                record = {"task_id": int(task_id), "shard": int(args.shard), "desc": tasks[task_id],
                          "episode": int(episode_idx), "seed": int(seed), "instruction": instruction,
                          "success": solved, "steps": int(steps), "wall_s": round(time.monotonic() - started, 1),
                          "shader": SHADER}
                with open(out_dir / EPISODES_JSONL, "a") as f:
                    f.write(json.dumps(record) + "\n")
                logging.info("EPRESULT " + json.dumps(record))
        finally:
            if env is not None:
                env.close_env(clear_cache=True)
    logging.info("total %d/%d = %.1f%%", successes, episodes,
                 100.0 * successes / episodes if episodes else float("nan"))


if __name__ == "__main__":
    main()
