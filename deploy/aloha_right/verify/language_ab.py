"""Replay one episode under both instructions; the corrector must react: python -u verify/language_ab.py"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_corrector import ARM, CHUNK, GRAPH_DIR, episode  # noqa: E402

ROOT = os.environ.get("ALOHA_DATA_ROOT", os.path.join(os.environ.get("ROBOT_OOD_DATA", "data"), "lerobot"))
DRAWER = os.path.join(ROOT, "aloha_placing_drawer_lerobot")
KITCHEN = os.path.join(ROOT, "aloha_placing_kitchen_lerobot")


def instruction(dataset: str) -> str:
    """The task string this dataset was recorded with, straight from its own meta.

    Args:
        dataset: LeRobot dataset root.

    Returns:
        The first task string.

    Raises:
        FileNotFoundError: No readable task metadata.
    """
    jsonl = os.path.join(dataset, "meta", "tasks.jsonl")
    if os.path.isfile(jsonl):
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    return json.loads(line)["task"]
    plain = os.path.join(dataset, "meta", "tasks.json")
    if os.path.isfile(plain):
        with open(plain, encoding="utf-8") as fh:
            obj = json.load(fh)
        if isinstance(obj, dict):
            return next(iter(obj.values())) if obj else ""
        if isinstance(obj, list) and obj:
            return obj[0]["task"] if isinstance(obj[0], dict) else str(obj[0])
    raise FileNotFoundError(f"no task metadata under {dataset}/meta")


def replay(state: np.ndarray, action: np.ndarray, task: str) -> np.ndarray:
    """Replay one episode through a fresh corrector under a fixed instruction.

    The observation stream is a pure function of the recorded episode, so two calls differing only
    in `task` differ only by the language channel.

    Args:
        state: [N, 7] measured.
        action: [N, 7] commanded absolute targets; doubles as the oracle policy chunk.
        task: Instruction handed to correct().

    Returns:
        [n_chunks, CHUNK, 7] corrected chunks.
    """
    from robot_ood.runtime.aloha_deploy import AlohaCorrector

    c = AlohaCorrector(GRAPH_DIR)
    c.reset()
    grip_cmd = float(state[0, ARM])
    c.observe(state[0, :ARM], grip_cmd)

    out = []
    n = len(state)
    t = 0
    while t + CHUNK + 1 < n:
        chunk = action[t:t + CHUNK].copy()
        joints = state[t, :ARM]
        corrected = c.correct(chunk, joints, grip_cmd, task=task)
        out.append(np.asarray(corrected, dtype=np.float64).copy())

        rows = c.action_rows(corrected, joints)
        for k in range(CHUNK):
            c.observe(state[min(t + k, n - 1), :ARM], grip_cmd, rows[k])
            grip_cmd = float(corrected[k, ARM])
        t += CHUNK
    return np.stack(out)


def main() -> int:
    need = os.path.join(GRAPH_DIR, "g_fcsr.npz")
    if not os.path.isfile(need):
        print(f"SKIP: {need} absent -- correct() raises rather than degrading, so neither arm of "
              f"this A/B can run. Nothing to compare until FCSR finishes training.")
        return 2

    right = instruction(DRAWER)
    wrong = instruction(KITCHEN)
    print(f"drawer instruction : {right}")
    print(f"kitchen instruction: {wrong}")
    if right == wrong:
        print("\nFAIL: both datasets report the same task string -- nothing to A/B")
        return 1
    pre = os.path.commonprefix([right, wrong])
    print(f"shared prefix      : {len(pre)} chars -- {pre!r}")

    state, action = episode(0)
    print(f"episode            : {len(state)} frames\n")

    a = replay(state, action, right)          # correct instruction
    b = replay(state, action, wrong)          # deliberately wrong instruction
    print(f"chunks replayed    : {len(a)}")

    if a.shape != b.shape:
        print(f"\nFAIL: shape drift {a.shape} vs {b.shape}")
        return 1

    per_chunk = np.abs(a[:, :, :ARM] - b[:, :, :ARM]).reshape(len(a), -1).max(axis=1)
    differing = int((per_chunk > 0.0).sum())
    print(f"chunks that differ : {differing}/{len(a)}  ({differing / len(a):.1%})")
    if differing:
        d = per_chunk[per_chunk > 0.0]
        print(f"|A-B| on those     : p50 {np.percentile(d,50):.6f}  max {d.max():.6f} rad")

    grip_a = int((a[:, :, ARM] != b[:, :, ARM]).any(axis=1).sum())
    print(f"gripper col differs: {grip_a} chunk(s)  (MUST be 0 -- it is passed through)")

    oracle = np.stack([action[t:t + CHUNK] for t in range(0, len(a) * CHUNK, CHUNK)])
    ea = np.abs(a[:, :, :ARM] - oracle[:, :, :ARM])
    eb = np.abs(b[:, :, :ARM] - oracle[:, :, :ARM])
    da, db = float(ea.mean()), float(eb.mean())
    ma, mb = float(ea.max()), float(eb.max())
    print(f"\n|correction| away from the recorded demo:")
    print(f"  correct instruction  mean {da:.6f}  max {ma:.6f} rad")
    print(f"  wrong   instruction  mean {db:.6f}  max {mb:.6f} rad")
    if differing:
        print(f"  -> the wrong instruction perturbs "
              f"{'MORE' if db > da else 'LESS'} ({db / max(da, 1e-12):.2f}x)")

    if grip_a:
        print("\nFAIL: the gripper column moved -- it must be passed through verbatim")
        return 1
    if differing == 0:
        print("\nRESULT: the instruction changed NOTHING.")
        print("The language channel is inert on this code path. CAVEATS.md #3/#4 and the README's")
        print("'send the instruction verbatim' rule overstate its role -- record that honestly")
        print("rather than shipping a rule that does nothing.")
        return 1
    print("\nRESULT: the instruction demonstrably changes the corrector's output.")
    print("The language channel is live, so the README rule is real and a wrong or empty")
    print("instruction does move the arm. PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
