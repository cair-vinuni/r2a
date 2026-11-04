"""Open-loop replay of the server loop on a recorded episode: python -u verify/replay_corrector.py"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))

GRAPH_DIR = str(paths.ARTIFACTS)
DATASET = os.environ.get(
    "ALOHA_DATASET",
    os.path.join(os.environ.get("ROBOT_OOD_DATA", "data"), "lerobot", "aloha_placing_drawer_lerobot"),
)
INSTRUCTION = (
    "Use the right gripper to open the top drawer, pick up the red square block, "
    "place it inside the top drawer, and then close the drawer."
)
CHUNK = 16
ARM = 6


def episode(idx: int = 0):
    """One episode's measured state and commanded action.

    Args:
        idx: Which parquet file, in sorted order.

    Returns:
        (state [N, 7] measured, action [N, 7] commanded absolute joint targets + native gripper).
    """
    files = sorted(glob.glob(os.path.join(DATASET, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {DATASET}")
    df = pd.read_parquet(files[idx], columns=["observation.state", "action"])
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)
    return state[:, :7], action[:, :7]


def main() -> int:
    from robot_ood.runtime.aloha_server import AlohaServerLoop

    need = os.path.join(GRAPH_DIR, "g_fcsr.npz")
    if not os.path.isfile(need):
        print(f"SKIP: {need} not present -- correct() RAISES without it (it does not pass "
              f"through), so there is nothing to measure until FCSR finishes training")
        return 2

    state, action = episode(0)
    n = len(state)
    print(f"episode: {n} frames, state {state.shape}, action {action.shape}")

    loop = AlohaServerLoop(GRAPH_DIR, homing=False)
    c = loop.corrector
    print(f"plan_rows {c.plan_rows}  policy_chunk {c.policy_chunk}  graph_win {c.graph_win}")

    n_chunks = 0
    n_corrected = 0
    n_passthrough = 0
    arm_deltas: list[float] = []
    policy_steps: list[float] = []
    grip_violations = 0
    shape_violations = 0

    t = 0
    while t + CHUNK + 1 < n:
        chunk = action[t:t + CHUNK].copy()
        corrected = loop.step(state[t, :ARM], state[t, ARM], INSTRUCTION, lambda: chunk)
        n_chunks += 1

        if corrected.shape != chunk.shape:
            shape_violations += 1
        if not np.array_equal(corrected[:, ARM], chunk[:, ARM]):
            grip_violations += 1

        if loop.moved > 0.0:
            n_corrected += 1
            arm_deltas.append(loop.moved)
        else:
            n_passthrough += 1
        policy_steps.append(float(np.abs(np.diff(chunk[:, :ARM], axis=0)).max()))
        t += CHUNK

    print()
    print(f"chunks           {n_chunks}")
    print(f"  corrected      {n_corrected}  ({n_corrected / max(n_chunks,1):.1%})")
    print(f"  pass-through   {n_passthrough}  ({n_passthrough / max(n_chunks,1):.1%})")
    if arm_deltas:
        a = np.array(arm_deltas)
        print(f"correction |dq|  p50 {np.percentile(a,50):.5f}  p90 {np.percentile(a,90):.5f}  "
              f"max {a.max():.5f} rad")
    p = np.array(policy_steps)
    print(f"policy per-step  p50 {np.percentile(p,50):.5f}  max {p.max():.5f} rad")
    print()
    print(f"gripper column altered in {grip_violations} chunk(s)  (MUST be 0)")
    print(f"shape changed in        {shape_violations} chunk(s)  (MUST be 0)")

    bad = grip_violations or shape_violations
    if bad:
        print("\nFAIL: the corrector violated its own contract")
        return 1
    if n_corrected == 0:
        print("\nFAIL: nothing was ever corrected -- retrieval never fired")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
