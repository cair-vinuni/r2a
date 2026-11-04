"""Find every task's evaluation seeds ahead of a cell, several tasks at a time.

The seed search is expert rollouts: planning on the CPU, one scene on a GPU per worker. Run once
per (task config, seed) it fills evals/robotwin/seeds so the sim clients of every later cell start
their episodes at once instead of each waiting on a search.

Usage:
    python evals/robotwin/harness/find_seeds.py --task-config demo_randomized --trials 100 --workers 32 --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os

_GPUS: list[str] = []


def _init(gpus: list[str], counter) -> None:
    """Pin this worker to one of the GPUs, round robin over the pool."""
    with counter.get_lock():
        k, counter.value = counter.value, counter.value + 1
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[k % len(gpus)]


def _one(job: tuple[str, str, int, int]) -> tuple[str, int]:
    task, config, seed, n = job
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {task} %(message)s")
    logging.getLogger("curobo").setLevel(logging.WARNING)
    import robotwin_env   # noqa: F401  (after the GPU pin: the checkout imports the simulator)
    from robotwin_env import episode_seeds, load_task_env
    env, args = load_task_env(task, config)
    seeds = episode_seeds(env, args, task, config, seed, n)
    env.close_env(clear_cache=True)
    return task, len(seeds)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-suite", default="all")
    p.add_argument("--task-config", required=True)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--gpus", default="", help="comma-separated CUDA ids the workers render on")
    args = p.parse_args()
    from robotwin_env import seed_file, task_names
    tasks = task_names(args.task_suite)
    pending = [t for t in tasks
               if not seed_file(args.task_config, t, args.seed).exists()
               or len(json.loads(seed_file(args.task_config, t, args.seed).read_text())["seeds"]) < args.trials]
    print(f"{len(pending)} of {len(tasks)} tasks need seeds", flush=True)
    jobs = [(t, args.task_config, args.seed, args.trials) for t in pending]
    ctx = mp.get_context("spawn")
    gpus = [g for g in args.gpus.split(",") if g]
    with ctx.Pool(min(args.workers, max(1, len(jobs))), initializer=_init,
                  initargs=(gpus, ctx.Value("i", 0))) as pool:
        for task, n in pool.imap_unordered(_one, jobs):
            print(f"{task}: {n} seeds", flush=True)


if __name__ == "__main__":
    main()
