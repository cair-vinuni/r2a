"""The star servo on the ten real-world VRH3-Pick displacement cases, in a synthetic energy bowl.

No simulator and no policy server: the robot offset is placed on a stub and the feature is a linear map
of the displacement left after the servo's move. The search and the motion are the deployed code.

Two feature models:
    decoupled  x, y and yaw act on independent feature directions, so a waist turn can only undo yaw
    view       a head camera looks at a scene at depth DEPTH: a lateral shift dy moves the image like a yaw
               of dy / DEPTH, so a turn undoes that part and only PARALLAX of the shift is left. Forward and
               back shifts change the scale of the view, which no turn undoes.

Usage: PYTHONPATH=src python3 scripts/servo_cases.py --motion waist --out outputs/servo/cases/waist.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from robot_ood.servo.motion import UNIT
from robot_ood.servo.search import StarServo

TASK = "task"
NOISE = 0.01
DEPTH = 0.6        # m, assumed head-camera distance to the work surface
PARALLAX = 0.3     # assumed fraction of a lateral shift a turn cannot hide
DEG = np.pi / 180
# Displacement D of the robot (x forward, y left, yaw counter-clockwise); the servo must find u = -D.
CASES = {
    "lech_phai_2cm": (0.0, -0.02, 0.0), "lech_phai_8cm": (0.0, -0.08, 0.0), "lech_phai_12cm": (0.0, -0.12, 0.0),
    "lech_trai_2cm": (0.0, 0.02, 0.0), "lech_trai_4cm": (0.0, 0.04, 0.0),
    "lui_xuong_10cm": (-0.10, 0.0, 0.0), "tien_len_4cm": (0.04, 0.0, 0.0),
    "xoay_phai_15do": (0.0, 0.0, -15 * DEG), "xoay_phai_30do": (0.0, 0.0, -30 * DEG),
    "xoay_trai_30do": (0.0, 0.0, 30 * DEG),
}


class StubPrior:
    tasks = [TASK]
    tau = 0.15
    energy_std = 0.05

    def residual(self, features: np.ndarray, task: int) -> np.ndarray:
        return features


class BowlClient:
    """Feature of the view after the robot was displaced by `shift` and the servo then moved it by u."""

    def __init__(self, shift: np.ndarray, model: str, seed: int) -> None:
        self.shift, self.model, self.rng = shift, model, np.random.default_rng(seed)
        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(8, 3)))
        self.map = q * np.array([1.0, 0.8, 0.5])

    def embed(self, observations: list[dict], task: str) -> np.ndarray:
        left = np.stack([(self.shift + o["u"]) / UNIT for o in observations])
        if self.model == "view":
            x, y, yaw = left.T
            left = np.stack([x, y + DEPTH * yaw * UNIT[2] / UNIT[1], PARALLAX * y], 1)
        return left @ self.map.T + self.rng.normal(scale=NOISE, size=(len(left), 8))


def stub_placer(env) -> callable:
    return lambda u: {"agentview_image": np.zeros((2, 2, 3), np.uint8), "u": np.asarray(u, float).copy()}


def run(motion: str, model: str, shift: np.ndarray, seed: int) -> dict:
    servo = StarServo(StubPrior(), BowlClient(shift, model, seed), motion, placer=stub_placer, settle=lambda env: {})
    servo.run(None, {}, 0, None, task=TASK)
    return servo.last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", default="waist", choices=("waist", "walk"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    results = []
    for model in ("decoupled", "view"):
        for case, shift in CASES.items():
            runs = [run(a.motion, model, np.array(shift), seed) for seed in range(a.seeds)]
            mean = lambda key: float(np.mean([r[key] for r in runs]))  # noqa: E731
            final = np.mean([r["u"] for r in runs], 0)
            row = {"motion": a.motion, "model": model, "case": case, "shift": list(shift),
                   "stop": sorted({r["stop"] for r in runs}),
                   "certified": sum(r["stop"] == "certified" for r in runs),
                   "robot_seconds": round(mean("robot_seconds"), 1), "reads": round(mean("reads"), 1),
                   "move_cm": np.round(final[:2] * 100, 1).tolist(), "turn_deg": round(float(final[2]) / DEG, 1),
                   "energy0": round(mean("energy0"), 3), "energy": round(mean("energy"), 3)}
            results.append(row)
            print(f"{a.motion:5s} {model:9s} {case:15s} {row['certified']}/{a.seeds} {','.join(row['stop']):22s} "
                  f"t={row['robot_seconds']:5.1f}s move={row['move_cm']}cm turn={row['turn_deg']:+6.1f}deg "
                  f"E {row['energy0']:6.2f}->{row['energy']:.3f}", flush=True)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
