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

ARM = 6
DATASET = os.environ.get(
    "VRH31_DATASET",
    "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/triquang/vrh31_bds_pick_100eps",
)
DISPLACE_RAD = 0.35
N_WALKS = 64


def starts() -> np.ndarray:
    """[n_episodes, 26] measured configuration at t=0, in the checkpoint's column order."""
    from robot_ood.data.lerobot_vrh31 import policy_state

    files = sorted(glob.glob(os.path.join(DATASET, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {DATASET}")
    out = []
    for f in files:
        df = pd.read_parquet(f, columns=["observation.state"])
        row = np.asarray(df["observation.state"].iloc[0], dtype=np.float64)[None, :]
        out.append(policy_state(row)[0])
    return np.stack(out)


def walk(homing, joints: np.ndarray, hand: np.ndarray) -> tuple[dict, np.ndarray]:
    """Run one walk, the arm landing on each chunk's last row; returns the report and every row sent."""
    state = np.asarray(joints, dtype=np.float64).copy()
    rows: list[np.ndarray] = []
    homing.reset()
    while (leg := homing.chunk(state, hand)) is not None:
        rows.append(leg)
        state = leg[-1].copy()
    return homing.report(state, hand), (np.concatenate(rows) if rows else np.zeros((0, ARM)))


def run_arm(arm: str, q0: np.ndarray) -> int:
    from robot_ood.runtime.vrh31_deploy import lane_hand, lane_joints
    from robot_ood.runtime.vrh31_entry import Vrh31Homing

    print(f"\n======== {arm} ========")
    try:
        h = Vrh31Homing(paths.ARTIFACTS[arm])
    except FileNotFoundError as err:
        print(f"SKIP: {err}")
        return 0

    print(f"step_cap         {h.step_cap:.6f} rad ({np.degrees(h.step_cap):.2f} deg)")
    print(f"refine steps     {h.cfg.score_refine_steps}")
    print(f"entry gripper    {h.entry_grip():.4f}")

    joints0, hands0 = lane_joints(q0, arm), lane_hand(q0, arm)
    rng = np.random.default_rng(0)
    displaced = joints0 + rng.normal(0.0, DISPLACE_RAD, size=joints0.shape)
    print(f"\ngenuine starts   {len(q0)} episodes")
    print(f"walking          {N_WALKS} genuine and {N_WALKS} displaced")

    fail = 0
    worst_leg = 0.0
    closer = 0

    for name, population in (("genuine", joints0[:N_WALKS]), ("displaced", displaced[:N_WALKS])):
        steps, passes, missed, residual, level_hat = [], [], 0, [], []
        for i, start in enumerate(population):
            report, rows = walk(h, start, hands0[i])
            steps.append(report["steps"])
            passes.append(report["passes"])
            residual.append(report["residual"])
            level_hat.append(report["level_hat"])
            missed += not report["arrived"]
            if rows.size:
                legs = np.linalg.norm(np.diff(np.vstack([start, rows]), axis=0), axis=1)
                worst_leg = max(worst_leg, float(legs.max()))
            if name == "displaced" and report["err1"] < report["err0"]:
                closer += 1
        s, p = np.asarray(steps), np.asarray(passes)
        print(f"\n{name}: steps p50 {np.percentile(s, 50):.0f}  max {s.max()}  "
              f"passes max {p.max()}  residual p50 {np.percentile(residual, 50):.4f}  "
              f"level_hat p50 {np.percentile(level_hat, 50):.4f}")
        print(f"  ended outside level_hat {missed}/{N_WALKS}")
        if missed:
            print("  FAIL: a walk ended outside the stopping radius")
            fail += 1

    print(f"\nlongest leg {worst_leg:.6f} rad (cap {h.step_cap:.6f})")
    if worst_leg > h.step_cap * (1 + 1e-9):
        print("  FAIL: a leg exceeded step_cap -- the arm would jump")
        fail += 1

    print(f"\ndisplaced starts ending closer to the barycentre: {closer}/{N_WALKS}")
    if closer < N_WALKS * 0.9:
        print("  FAIL: the walk does not move displaced starts toward the entry configuration")
        fail += 1

    lg = np.array([h.level(joints0[i], hands0[i]) for i in range(N_WALKS)])
    ld = np.array([h.level(displaced[i], hands0[i]) for i in range(N_WALKS)])
    print(f"\nlevel(): genuine p50 {np.percentile(lg, 50):.4f}   "
          f"displaced p50 {np.percentile(ld, 50):.4f}")
    if not np.isfinite(lg).all() or not np.isfinite(ld).all():
        print("  FAIL: level() returned a non-finite value")
        fail += 1
    elif np.percentile(ld, 50) <= np.percentile(lg, 50):
        print("  FAIL: the level head does not rank displaced starts worse")
        fail += 1
    return fail


def main() -> int:
    q0 = starts()
    fail = sum(run_arm(arm, q0) for arm in paths.ARMS)
    print("\nPASS" if fail == 0 else f"\nFAIL ({fail} check(s))")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
