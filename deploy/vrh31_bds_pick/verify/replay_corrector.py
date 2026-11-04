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

DATASET = os.environ.get(
    "VRH31_DATASET",
    "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/triquang/vrh31_bds_pick_100eps",
)
INSTRUCTION = (
    "Grasp the large curved metal frame with both hands and lift it upward from the rack."
)
CHUNK = 16


def episode(idx: int = 0):
    """One episode's measured state and commanded action, both in the checkpoint's column order.

    Args:
        idx: Which parquet file, in sorted order.

    Returns:
        (state [N, 26] measured, action [N, 26] commanded absolute joint targets).

    Raises:
        FileNotFoundError: No parquet under DATASET.
    """
    from robot_ood.data.lerobot_vrh31 import policy_action, policy_state

    files = sorted(glob.glob(os.path.join(DATASET, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {DATASET}")
    df = pd.read_parquet(files[idx], columns=["observation.state", "action"])
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)
    return policy_state(state), policy_action(action)


def main() -> int:
    from robot_ood.data.lerobot_vrh31 import POLICY_ARM, POLICY_HAND
    from robot_ood.runtime.vrh31_server import Vrh31ServerLoop

    missing = [str(paths.ARTIFACTS[a] / "g_fcsr.npz") for a in paths.ARMS
               if not (paths.ARTIFACTS[a] / "g_fcsr.npz").is_file()]
    if missing:
        print(f"SKIP: {missing} not present -- correct() RAISES without it (it does not pass "
              f"through), so there is nothing to measure until FCSR finishes training")
        return 2

    state, action = episode(0)
    n = len(state)
    print(f"episode: {n} frames, state {state.shape}, action {action.shape}")

    loop = Vrh31ServerLoop({a: paths.ARTIFACTS[a] for a in paths.ARMS}, homing=False)
    for arm in paths.ARMS:
        c = loop.lanes[arm].corrector
        print(f"{arm:5s} plan_rows {c.plan_rows}  policy_chunk {c.policy_chunk}  "
              f"graph_win {c.graph_win}")

    owned = np.concatenate([np.asarray(POLICY_ARM[a]) for a in paths.ARMS])
    untouched = np.setdiff1d(np.arange(state.shape[1]), owned)

    n_chunks = 0
    n_corrected = 0
    n_passthrough = 0
    arm_deltas: list[float] = []
    policy_steps: list[float] = []
    hand_violations = 0
    shape_violations = 0

    t = 0
    while t + CHUNK + 1 < n:
        chunk = action[t:t + CHUNK].copy()
        corrected = loop.step(state[t], INSTRUCTION, lambda: chunk)
        n_chunks += 1

        if corrected.shape != chunk.shape:
            shape_violations += 1
        elif not np.array_equal(corrected[:, untouched], chunk[:, untouched]):
            hand_violations += 1

        if loop.moved > 0.0:
            n_corrected += 1
            arm_deltas.append(loop.moved)
        else:
            n_passthrough += 1
        policy_steps.append(float(np.abs(np.diff(chunk[:, owned], axis=0)).max()))
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
    print(f"hand and dropped columns altered in {hand_violations} chunk(s)  (MUST be 0)")
    print(f"shape changed in                   {shape_violations} chunk(s)  (MUST be 0)")

    if hand_violations or shape_violations:
        print("\nFAIL: the corrector violated its own contract")
        return 1
    if n_corrected == 0:
        print("\nFAIL: nothing was ever corrected -- retrieval never fired")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
