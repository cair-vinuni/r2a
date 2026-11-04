"""RoboTwin 2.0 task environments and the seeds their episodes run on.

Puts the checkout on the path, builds a task's environment the way RoboTwin's own eval does (task
config, embodiment and camera files), and finds the evaluation seeds: from 2000 * (1 + seed)
upward, the first N seeds whose expert demonstration plans and succeeds -- the benchmark's own
definition of an episode. Seed lists are cached one json per (task config, task) under
evals/robotwin/seeds, so every shard and every later cell runs the same episodes without
repeating the search.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

__all__ = ["INSTRUCTION_KINDS", "ROBOTWIN_HOME", "SEEDS_DIR", "SEED_STRIDE", "episode_instruction",
           "episode_seeds", "load_task_env", "seed_file", "start_seed", "task_names"]

ROBOTWIN_HOME = Path(os.environ.get("ROBOTWIN_HOME") or Path(__file__).resolve().parents[3] / "external" / "RoboTwin")
# One seed cache per benchmark: a task config of one name does not have to describe the same scene
# in two checkouts, and a seed is only valid for the scene its expert demonstration planned in.
SEEDS_DIR = Path(os.environ.get("ROBOTWIN_SEEDS") or Path(__file__).resolve().parent.parent / "seeds")
SUITES_DIR = Path(__file__).resolve().parent.parent / "suites"

# Spaces the episodes of one seed apart in the instruction step's own seeding, so no two
# (seed, episode) pairs in a run collide on one stream.
_INSTRUCTION_STRIDE = 1_000
SEED_STRIDE = 2000
CONFIGS_DIR = "task_config"

# The phrasing pools an episode can be drawn from. seen and unseen are RoboTwin 2.0's own; r1, r2,
# r3 and language_plus are RoboTwin 2.0-Plus' pre-generated perturbations, and auto reads the pool
# out of the axis' task config the way its script/eval_policy.py does.
INSTRUCTION_KINDS = ("auto", "seen", "unseen", "r1", "r2", "r3", "language_plus")
_PLUS_POOLS = {"r1": ("r1_distraction",), "r2": ("r2_commonsense",), "r3": ("r3_reasoning",),
               "language_plus": ("r1_distraction", "r2_commonsense", "r3_reasoning")}
_LANGUAGE_MODES = {"seen": "seen", "unseen": "unseen", "r1": "r1", "r2": "r2", "r3": "r3",
                   "r1r2r3": "language_plus", "language_plus": "language_plus"}

for entry in (str(ROBOTWIN_HOME), str(ROBOTWIN_HOME / "description" / "utils")):
    if entry not in sys.path:
        sys.path.insert(0, entry)
os.chdir(ROBOTWIN_HOME)   # the checkout resolves its assets and configs against its own directory


def task_names(task_suite: str) -> list[str]:
    """The tasks of one suite file, in order."""
    path = SUITES_DIR / f"{task_suite}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def start_seed(seed: int) -> int:
    """The first candidate seed of a run: 2000 * (1 + seed), the benchmark's convention."""
    return SEED_STRIDE * (1 + int(seed))


def _yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_task_env(task_name: str, task_config: str) -> tuple[Any, dict]:
    """A task's environment and the setup_demo keyword arguments its config implies.

    Args:
        task_name: The task module under envs/.
        task_config: The task config name under task_config/, without .yml.

    Returns:
        The environment instance and the arguments, in evaluation mode: no viewer, no video log.
    """
    module = importlib.import_module(f"envs.{task_name}")
    env = getattr(module, task_name)()
    configs = ROBOTWIN_HOME / CONFIGS_DIR
    args = _yaml(configs / f"{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    embodiment_types = _yaml(configs / "_embodiment_config.yml")
    camera_types = _yaml(configs / "_camera_config.yml")

    def embodiment_file(kind: str) -> str:
        path = embodiment_types[kind]["file_path"]
        if path is None:
            raise ValueError(f"embodiment {kind!r} has no files")
        return path

    head = args["camera"]["head_camera_type"]
    args["head_camera_h"], args["head_camera_w"] = camera_types[head]["h"], camera_types[head]["w"]
    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        args["left_robot_file"] = args["right_robot_file"] = embodiment_file(embodiment[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"], args["right_robot_file"] = embodiment_file(embodiment[0]), embodiment_file(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment lists one arm kind or two kinds and a distance")
    args["left_embodiment_config"] = _yaml(Path(args["left_robot_file"]) / "config.yml")
    args["right_embodiment_config"] = _yaml(Path(args["right_robot_file"]) / "config.yml")
    args["embodiment_name"] = "+".join(embodiment) if len(embodiment) > 1 else embodiment[0]
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False
    return env, args


def episode_instruction(env: Any, args: dict, seed: int, episode: int, kind: str, count: int) -> str:
    """The instruction RoboTwin's own evaluation gives a policy for one episode.

    The benchmark plays the expert demonstration of the seed first and describes what it did
    (which objects, which arm) through description/utils; one of the task's seen or unseen
    phrasings is then drawn.

    The whole step is made a function of the episode's own seed and index. Both halves of it draw
    from the global generators: the expert rollout that produces the scene description, and the
    generator that writes the phrasings and picks one. Neither can be handed a generator of its
    own from here, so the global ones are seeded from the episode instead. Without it the phrasing
    an episode is given is not the same on a second run, and the policy reads the instruction --
    two runs of one seed were answering different prompts on seven episodes in eight.

    Args:
        env: The task environment.
        args: Its setup_demo arguments.
        seed: The episode's scene seed.
        episode: The episode index within the task.
        kind: The instruction pool to draw from, one of INSTRUCTION_KINDS; auto reads it out of
            the task config's own language.mode.
        count: How many phrasings the benchmark generates before one is drawn (its trials per task).

    Returns:
        The instruction; the task name when the benchmark can describe nothing.
    """
    import random

    import numpy as np
    from generate_episode_instructions import generate_episode_descriptions

    kind = _instruction_kind(kind, args)

    stream = (int(seed) * _INSTRUCTION_STRIDE + int(episode)) % 2**32
    np.random.seed(stream)
    random.seed(stream)

    info: dict = {}
    try:
        env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **args)
        info = (env.play_once() or {}).get("info", {})
    except Exception as exc:   # an expert that fails still leaves the scene description
        logging.info("seed %d expert run for the instruction failed: %s", seed, exc)
    finally:
        env.close_env()
    try:
        options = (_plus_options(args["task_name"], kind) if kind in _PLUS_POOLS
                   else generate_episode_descriptions(args["task_name"], [info], count)[0].get(kind) or [])
    except Exception as exc:
        logging.info("seed %d no generated instruction: %s", seed, exc)
        options = []
    if not options:
        return str(args["task_name"])
    return str(np.random.default_rng([int(seed), int(episode)]).choice(options))



def _instruction_kind(kind: str, args: dict) -> str:
    """The pool an episode is phrased from, resolving auto against the task config.

    RoboTwin 2.0-Plus names the perturbation in the config's language.mode and its own evaluation
    maps that to a pool; every axis but Language leaves it at the default, the seen pool.

    Args:
        kind: The requested pool, or auto.
        args: The task config, as load_task_env returned it.

    Returns:
        A pool name; seen when the config names none.
    """
    if kind != "auto":
        return kind
    language = args.get("language")
    mode = language.get("mode", "seen") if isinstance(language, dict) else "seen"
    return _LANGUAGE_MODES.get(str(mode), "seen")


def _plus_options(task_name: str, kind: str) -> list[str]:
    """One task's pre-generated RoboTwin 2.0-Plus phrasings for a perturbation pool.

    Args:
        task_name: The task.
        kind: A key of _PLUS_POOLS.

    Returns:
        Every phrasing of that pool, the sub-pools concatenated in the benchmark's own order.
    """
    from generate_episode_instructions import load_plus_instructions

    data = load_plus_instructions(task_name) or {}
    return [str(text) for key in _PLUS_POOLS[kind] for text in data.get(key, [])]


def seed_file(task_config: str, task_name: str, seed: int) -> Path:
    """Where one task's episode seeds are cached."""
    return SEEDS_DIR / task_config / f"{task_name}_s{seed}.json"


# Candidates a task may reject in a row before the search is declared broken rather than unlucky.
MAX_REJECTED_RUN = 200
CLEAR_CACHE_EVERY = 5      # candidate scenes between SAPIEN render-cache clears


def _expert_succeeds(env: Any, args: dict, candidate: int, episode: int) -> bool:
    """Whether the expert demonstration of one seed plans and succeeds, as the benchmark requires."""
    try:
        env.setup_demo(now_ep_num=episode, seed=candidate, is_test=True, **args)
        env.play_once()
        ok = bool(env.plan_success and env.check_success())
    except Exception as exc:   # an unstable scene or a failed plan is a rejected seed, not an error
        logging.info("seed %d rejected: %s: %s", candidate, type(exc).__name__, exc)
        ok = False
    env.close_env(clear_cache=candidate % CLEAR_CACHE_EVERY == 0)   # the render cache grows per scene
    return ok


def episode_seeds(env: Any, args: dict, task_name: str, task_config: str, seed: int, n: int) -> list[int]:
    """The first n seeds from start_seed(seed) whose expert demonstration succeeds.

    Args:
        env: The task environment, used for the search when the cache holds no list.
        args: Its setup_demo arguments.
        task_name: The task, for the cache file.
        task_config: The task config, for the cache file.
        seed: The run seed.
        n: Episodes wanted.

    Returns:
        n seeds, ascending. A cached list at least n long is returned as is; a shorter one is
        extended. The search runs under a file lock, so parallel shards search each task once.
    """
    path = seed_file(task_config, task_name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = json.loads(path.read_text()) if path.exists() else {"start": start_seed(seed), "seeds": [], "tried": 0}
        if len(doc["seeds"]) < n:
            candidate = doc["start"] + doc["tried"]
            t0 = time.monotonic()
            rejected_run = 0
            while len(doc["seeds"]) < n:
                if _expert_succeeds(env, args, candidate, len(doc["seeds"])):
                    doc["seeds"].append(candidate)
                    rejected_run = 0
                else:
                    rejected_run += 1
                    if rejected_run > MAX_REJECTED_RUN:
                        raise RuntimeError(f"{task_config}/{task_name}: {rejected_run} seeds rejected in a "
                                           f"row from {candidate - rejected_run + 1}; the environment is broken")
                candidate += 1
                doc["tried"] = candidate - doc["start"]
            logging.info("%s/%s: %d seeds from %d candidates in %.0f s", task_config, task_name,
                         len(doc["seeds"]), doc["tried"], time.monotonic() - t0)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc) + "\n")
            os.replace(tmp, path)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return list(doc["seeds"][:n])
