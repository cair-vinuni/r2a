"""Per-episode plumbing shared by the sim clients and the driver.

A cell's episodes are dealt in contiguous blocks to its clients (shard_episodes); each client
appends one json line per episode to episodes.jsonl, the record the scorer counts and the one
pending_episodes reads back to find the episodes a dead client never reached, and one EpisodeTrace
npz per corrected episode. Replay videos are off unless $ROBOT_OOD_VIDEO asks for them.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["EPISODES_JSONL", "SUMMARY_JSON", "TRACE_PREFIX", "VIDEO_MODES", "EpisodeTrace", "VideoWriter",
           "deal_episodes", "format_table", "pending_episodes", "per_task", "read_episodes",
           "shard_episodes", "video_mode", "write_summary"]

EPISODES_JSONL = "episodes.jsonl"
SUMMARY_JSON = "summary.json"


def deal_episodes(pairs: Sequence[tuple[int, int]], shard: int, num_shards: int) -> list[tuple[int, int]]:
    """One contiguous block of an episode list, so a shard builds as few scenes as it can.

    Args:
        pairs: The (task, episode) pairs to divide, in task-major order.
        shard: This client's index.
        num_shards: Clients sharing the cell.

    Returns:
        A contiguous slice, the shards' counts differing by at most one.

    Raises:
        ValueError: shard is not in range(num_shards).
    """
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} is not in range({num_shards})")
    # A client rebuilds the environment whenever the task changes, and where the axis perturbs the
    # scene that costs more than the episode; striding hands out a new task on every pick.
    base, extra = divmod(len(pairs), num_shards)
    start = shard * base + min(shard, extra)
    return [tuple(p) for p in pairs[start:start + base + (1 if shard < extra else 0)]]


def shard_episodes(n_tasks: int, trials: int, shard: int, num_shards: int) -> list[tuple[int, int]]:
    """This client's block of the full task-major grid.

    Args:
        n_tasks: Tasks in the suite.
        trials: Episodes per task.
        shard: This client's index.
        num_shards: Clients sharing the cell.

    Returns:
        Task-major pairs, contiguous, the shards' counts differing by at most one.
    """
    return deal_episodes([(t, e) for t in range(n_tasks) for e in range(trials)], shard, num_shards)


def pending_episodes(result_dir: Path, n_tasks: int, trials: int) -> list[tuple[int, int]]:
    """The grid's episodes that the cell's result dir holds no record of, in task-major order.

    Args:
        result_dir: The cell's result directory.
        n_tasks: Tasks in the suite.
        trials: Episodes per task.

    Returns:
        The missing (task, episode) pairs.
    """
    done = {(int(r["task_id"]), int(r["episode"])) for r in read_episodes(result_dir)
            if "task_id" in r and "episode" in r}
    return [(t, e) for t in range(n_tasks) for e in range(trials) if (t, e) not in done]


VIDEO_MODES = ("none", "failures", "all")


def video_mode() -> str:
    """Which replays a client keeps: $ROBOT_OOD_VIDEO, one of none (default), failures, all.

    Raises:
        ValueError: The variable holds anything else.
    """
    mode = os.environ.get("ROBOT_OOD_VIDEO", "none")
    if mode not in VIDEO_MODES:
        raise ValueError(f"ROBOT_OOD_VIDEO={mode!r}; expected one of {VIDEO_MODES}")
    return mode


class VideoWriter:
    """Encodes replay videos on one background thread, so the client starts its next episode at once.

    Args:
        mode: none writes nothing (and keep() says not to collect frames), failures writes the
            unsolved episodes, all writes every episode.
        max_pending: Videos queued ahead of the encoder before write() waits for the oldest.
    """

    def __init__(self, mode: str = "all", max_pending: int = 8) -> None:
        if mode not in VIDEO_MODES:
            raise ValueError(f"video mode {mode!r}; expected one of {VIDEO_MODES}")
        self.mode = mode
        self._max_pending = max(1, int(max_pending))
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video")
        self._pending: list[Future] = []

    def keep(self) -> bool:
        """Whether a client should collect frames at all."""
        return self.mode != "none"

    def write(self, path: Path, frames: Sequence[Any], solved: bool, fps: int = 10) -> bool:
        """Queue one mp4 when the mode keeps this episode.

        Args:
            path: Where it lands.
            frames: HxWx3 uint8 frames, kept referenced until encoded.
            solved: The episode's outcome.
            fps: Playback rate.

        Returns:
            Whether the video was queued.
        """
        if self.mode == "none" or (self.mode == "failures" and solved):
            return False
        import imageio
        while len(self._pending) >= self._max_pending:
            self._pending.pop(0).result()
        self._pending = [f for f in self._pending if not f.done()] + [
            self._pool.submit(imageio.mimwrite, str(path), [np.asarray(f) for f in frames], fps=fps)]
        return True

    def close(self) -> None:
        """Wait for every queued video; re-raises the first encoding error."""
        try:
            for fut in self._pending:
                fut.result()
        finally:
            self._pending.clear()
            self._pool.shutdown(wait=True)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


TRACE_PREFIX = "trace_"
_STEP_KEYS = ("q", "a_policy", "a_exec")
_CHECK_KEYS = ("check_step", "phase", "cand_idx", "cand_score", "trust", "shape", "authority", "z_chk")


class EpisodeTrace:
    """One episode's corrected arms, step by step: configuration, the policy's step, the executed step.

    Args:
        arms: The arm names traced, in a fixed order.
    """

    def __init__(self, arms: Sequence[str]) -> None:
        self.arms = tuple(str(a) for a in arms)
        self.steps: list[int] = []
        self._rows: dict[str, list] = {f"{k}_{a}": [] for a in self.arms for k in _STEP_KEYS + _CHECK_KEYS}

    def step(self, step: int, q: Mapping[str, Any], a_policy: Mapping[str, Any], a_exec: Mapping[str, Any]) -> None:
        """Record one environment step.

        Args:
            step: The harness step counter.
            q: arm -> the configuration read before the step.
            a_policy: arm -> the policy's own step for this row.
            a_exec: arm -> the step executed after the chain's correction.
        """
        self.steps.append(int(step))
        for arm in self.arms:
            for key, src in zip(_STEP_KEYS, (q, a_policy, a_exec)):
                self._rows[f"{key}_{arm}"].append(np.asarray(src[arm], dtype=np.float32))

    def check(self, step: int, arm: str, phase: float, cand_idx: int, cand_score: float,
              decision: Mapping[str, Any]) -> None:
        """Record one check's answer for one arm: its phase estimate, top candidate and decision.

        Args:
            step: The harness step counter.
            arm: The arm checked.
            phase: The phase estimate the plan was built on.
            cand_idx: The top candidate's segment id.
            cand_score: Its retrieval score.
            decision: The authority's own numbers for this check, as trust, the mean shape, the
                authority and the [N_CHK_FEAT] check features (z_chk) the trust head read; a
                missing number lands as nan, missing features as a nan row.
        """
        from robot_ood.models.fusion.corrector import N_CHK_FEAT

        z_chk = np.asarray(decision.get("z_chk", np.full(N_CHK_FEAT, np.nan)), dtype=np.float32)
        values = (step, phase, cand_idx, cand_score, decision.get("trust", float("nan")),
                  decision.get("shape", float("nan")), decision.get("authority", float("nan")), z_chk)
        for key, value in zip(_CHECK_KEYS, values):
            self._rows[f"{key}_{arm}"].append(value)

    def save(self, path: Path) -> Path:
        """Write the trace as one compressed npz, steps as int32 and everything else float32.

        Args:
            path: Where it lands.

        Returns:
            The path written.
        """
        arrays = {"arms": np.asarray(self.arms), "steps": np.asarray(self.steps, dtype=np.int32)}
        for key, rows in self._rows.items():
            arrays[key] = np.asarray(rows, dtype=np.float32)
        np.savez_compressed(path, **arrays)
        return Path(path)


def read_episodes(result_dir: Path) -> list[dict[str, Any]]:
    """Every episode record under a cell's result directory, in file then line order.

    Args:
        result_dir: The cell's result directory; shards write episodes.jsonl in subdirectories.

    Returns:
        The parsed records; a line that is not valid json is skipped.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(Path(result_dir).glob(f"**/{EPISODES_JSONL}")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def per_task(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fold episode records into one row per task, ordered by task id.

    Args:
        records: EPRESULT records.

    Returns:
        Rows of task_id, desc, successes, n and rate (percent, or None when n is 0).
    """
    rows: dict[int, dict[str, Any]] = {}
    for rec in records:
        task_id = int(rec["task_id"])
        row = rows.setdefault(task_id, {"task_id": task_id, "desc": str(rec.get("desc", "")),
                                        "successes": 0, "n": 0})
        row["n"] += 1
        row["successes"] += int(bool(rec.get("success")))
    for row in rows.values():
        row["rate"] = round(100.0 * row["successes"] / row["n"], 1) if row["n"] else None
    return [rows[k] for k in sorted(rows)]


def write_summary(result_dir: Path, log_dir: Path, **fields: Any) -> Path:
    """Write <result_dir>/summary.json from the cell's episode records.

    Args:
        result_dir: The cell's result directory, holding the shards' episodes.jsonl files.
        log_dir: The cell's log directory, recorded for reference.
        **fields: Identity and outcome fields recorded verbatim (policy, mode, protocol, ...).

    Returns:
        The path written.
    """
    rows = per_task(read_episodes(result_dir))
    n = sum(r["n"] for r in rows)
    successes = sum(r["successes"] for r in rows)
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **fields,
        "successes": successes, "n": n,
        "rate": round(100.0 * successes / n, 1) if n else None,
        "per_task": rows,
        "result_dir": str(result_dir), "log_dir": str(log_dir),
    }
    out = Path(result_dir) / SUMMARY_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return out


def format_table(rows: Sequence[Mapping[str, Any]], width: int = 60) -> str:
    """The per-task table as printed after a cell's RESULT line.

    Args:
        rows: per_task() rows.
        width: Characters of the task description kept.

    Returns:
        One line per task, then a total line; empty when there are no rows.
    """
    if not rows:
        return ""
    lines = [f"  task {r['task_id']:>2}  {r['successes']:>3}/{r['n']:<3} {r['rate']:>5.1f}%  "
             f"{r['desc'][:width]}" for r in rows]
    n = sum(r["n"] for r in rows)
    successes = sum(r["successes"] for r in rows)
    lines.append(f"  total    {successes:>3}/{n:<3} {100.0 * successes / n:>5.1f}%")
    return "\n".join(lines)
