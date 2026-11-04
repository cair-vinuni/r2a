from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_corrector import CHUNK, DATASET, episode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))

CONTRAST_INSTRUCTION = (
    "Use the left hand to open the drawer, pick up the red square block and place it inside."
)


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
    """Replay one episode through fresh correctors under a fixed instruction.

    The observation stream is a pure function of the recorded episode, so two calls differing only
    in `task` differ only by the language channel.

    Args:
        state: [N, 26] measured.
        action: [N, 26] commanded absolute targets; doubles as the oracle policy chunk.
        task: Instruction handed to correct().

    Returns:
        [n_chunks, CHUNK, 26] corrected chunks.
    """
    from robot_ood.runtime.vrh31_server import Vrh31ServerLoop

    loop = Vrh31ServerLoop({a: paths.ARTIFACTS[a] for a in paths.ARMS}, homing=False)
    out = []
    n = len(state)
    t = 0
    while t + CHUNK + 1 < n:
        chunk = action[t:t + CHUNK].copy()
        corrected = loop.step(state[t], task, lambda: chunk)
        out.append(np.asarray(corrected, dtype=np.float64).copy())
        t += CHUNK
    return np.stack(out)


def main() -> int:
    from robot_ood.data.lerobot_vrh31 import POLICY_ARM

    missing = [str(paths.ARTIFACTS[a] / "g_fcsr.npz") for a in paths.ARMS
               if not (paths.ARTIFACTS[a] / "g_fcsr.npz").is_file()]
    if missing:
        print(f"SKIP: {missing} absent -- correct() raises rather than degrading, so neither arm "
              f"of this A/B can run. Nothing to compare until FCSR finishes training.")
        return 2

    right = instruction(DATASET)
    wrong = CONTRAST_INSTRUCTION
    print(f"recorded instruction : {right}")
    print(f"contrast instruction : {wrong}")
    print("NOTE: this corpus holds ONE task, so the contrast string is synthetic rather than a")
    print("      second recorded task. A null result here means the language channel cannot")
    print("      discriminate on a single-task graph, which is expected, not a defect.")
    if right == wrong:
        print("\nFAIL: the contrast string equals the recorded one -- nothing to A/B")
        return 1

    state, action = episode(0)
    print(f"episode              : {len(state)} frames\n")

    a = replay(state, action, right)
    b = replay(state, action, wrong)
    print(f"chunks replayed      : {len(a)}")

    if a.shape != b.shape:
        print(f"\nFAIL: shape drift {a.shape} vs {b.shape}")
        return 1

    owned = np.concatenate([np.asarray(POLICY_ARM[arm]) for arm in paths.ARMS])
    untouched = np.setdiff1d(np.arange(state.shape[1]), owned)

    per_chunk = np.abs(a[:, :, owned] - b[:, :, owned]).reshape(len(a), -1).max(axis=1)
    differing = int((per_chunk > 0.0).sum())
    print(f"chunks that differ   : {differing}/{len(a)}  ({differing / len(a):.1%})")
    if differing:
        d = per_chunk[per_chunk > 0.0]
        print(f"|A-B| on those       : p50 {np.percentile(d,50):.6f}  max {d.max():.6f} rad")

    hand_diff = int((a[:, :, untouched] != b[:, :, untouched]).any(axis=(1, 2)).sum())
    print(f"hand columns differ  : {hand_diff} chunk(s)  (MUST be 0 -- they are passed through)")

    oracle = np.stack([action[t:t + CHUNK] for t in range(0, len(a) * CHUNK, CHUNK)])
    ea = np.abs(a[:, :, owned] - oracle[:, :, owned])
    eb = np.abs(b[:, :, owned] - oracle[:, :, owned])
    da, db = float(ea.mean()), float(eb.mean())
    ma, mb = float(ea.max()), float(eb.max())
    print(f"\n|correction| away from the recorded demo:")
    print(f"  recorded instruction  mean {da:.6f}  max {ma:.6f} rad")
    print(f"  contrast instruction  mean {db:.6f}  max {mb:.6f} rad")
    if differing:
        print(f"  -> the contrast instruction perturbs "
              f"{'MORE' if db > da else 'LESS'} ({db / max(da, 1e-12):.2f}x)")

    if hand_diff:
        print("\nFAIL: a hand column moved -- those must be passed through verbatim")
        return 1
    if differing == 0:
        print("\nRESULT: the instruction changed NOTHING on this single-task graph.")
        print("The language channel is inert here. Record that honestly rather than claiming the")
        print("prompt steers the corrector on this deployment.")
        return 0
    print("\nRESULT: the instruction demonstrably changes the corrector's output.")
    print("The language channel is live, so a wrong or empty instruction does move the arms. PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
