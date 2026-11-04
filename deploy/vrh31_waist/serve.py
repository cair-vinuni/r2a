"""GPU machine: GR00T's PolicyServer for the checkpoint, with the waist servo run inside get_action at t=0.

The robot's client sends the usual observation plus state["waist"] = [waist_roll, waist_yaw] (B, T, 2), and
every action it gets back carries action["waist"] = [waist_roll, waist_yaw] (B, H, 2), joint positions in rad
for the waist_joint_trajectory_controller.

Each run starts with the servo. The arms hold the pose of the first observation, the waist goes to the
servo's next target, and once state["waist"] has sat on that target for SETTLE seconds the servo reads the
cam_head frame there. When the search ends, the waist goes to the best pose and settles. From the next call on,
get_action serves the checkpoint's arm actions with the waist held at that pose. Roll stays where the run
found it.

A run starts when the server starts, when the client calls reset, or on the first get_action after IDLE
seconds without one.
"""

import logging
import time

import numpy as np

import paths
from gr00t.policy.server_client import PolicyServer
from robot_ood.servo.gr00t import HeadCameraFeatures
from robot_ood.servo.limits import WAIST_RANGE, WAIST_RATE
from robot_ood.servo.prior import PRIOR_NPZ, ServoPrior
from robot_ood.servo.session import ServoSession

WAIST_KEY = "waist"
SETTLE = 1.5
TOL = 0.02
SLACK = 3.0
IDLE = 10.0


class WaistServer:
    def __init__(self) -> None:
        self.features = HeadCameraFeatures(str(paths.CHECKPOINT))
        self.policy = self.features.policy
        self.horizon = len(self.policy.get_modality_config()["action"].delta_indices)
        self.session = ServoSession(ServoPrior.load(paths.PRIOR / PRIOR_NPZ), self.features.embed)
        self.last_call = None
        self.reset()

    def reset(self, options: dict | None = None) -> dict:
        logging.info("SERVO armed: the next get_action starts a run")
        self.phase = "start"
        return {}

    def get_action(self, observation: dict, options: dict | None = None):
        now = time.monotonic()
        if self.last_call is not None and now - self.last_call > IDLE and self.phase != "start":
            logging.info("SERVO no get_action for %.0f s, new run", now - self.last_call)
            self.phase = "start"
        self.last_call = now

        state = dict(observation["state"])
        if WAIST_KEY not in state:
            raise ValueError(f"observation['state'] has no '{WAIST_KEY}' ([waist_roll, waist_yaw], shape (B, T, 2))")
        roll, yaw = (float(x) for x in np.asarray(state.pop(WAIST_KEY), np.float32)[0, -1])

        if self.phase == "start":
            self.roll = roll
            self.hold = {k: np.repeat(np.asarray(v, np.float32)[:, -1:], self.horizon, axis=1) for k, v in state.items()}
            logging.info("SERVO start: waist roll %+.3f yaw %+.3f, arms hold", roll, yaw)
            self.goto(self.session.reset(yaw)["waist_yaw"], yaw, "search")

        if self.phase == "run":
            action, info = self.policy.get_action({**observation, "state": state}, options)
            self.report(roll, yaw)
            return {**action, WAIST_KEY: self.waist(len(next(iter(action.values()))))}, info

        if self.settled(yaw, now):
            if self.phase == "search":
                image = np.asarray(observation["video"][paths.VIDEO_KEY])[0, -1]
                if image.shape[:2] != paths.IMAGE_SHAPE:
                    raise ValueError(f"{paths.VIDEO_KEY} is {image.shape[:2]}, the demos' is {paths.IMAGE_SHAPE}")
                reply = self.log(self.session.step(image))
                self.goto(reply["waist_yaw"], yaw, "final" if reply["done"] else "search")
            else:
                logging.info("SERVO waist settled at %+.3f, serving the checkpoint's actions", self.target)
                self.phase = "run"
        self.report(roll, yaw)
        return {**self.hold, WAIST_KEY: self.waist(len(next(iter(self.hold.values()))))}, {}

    def goto(self, target: float, yaw: float, phase: str) -> None:
        self.target = float(np.clip(target, -WAIST_RANGE, WAIST_RANGE))
        self.phase, self.since = phase, None
        self.deadline = time.monotonic() + abs(self.target - yaw) / WAIST_RATE + SLACK
        logging.info("SERVO waist yaw -> %+.3f (%s)", self.target, phase)

    def settled(self, yaw: float, now: float) -> bool:
        if abs(yaw - self.target) > TOL:
            self.since = None
            if now > self.deadline:
                raise RuntimeError(f"waist yaw {yaw:+.3f} did not reach {self.target:+.3f}: check the waist "
                                   f"controller and that state['{WAIST_KEY}'] is [roll, yaw]")
            return False
        self.since = self.since or now
        return now - self.since >= SETTLE

    def report(self, roll: float, yaw: float) -> None:
        logging.info("WAIST %-6s state roll %+.3f yaw %+.3f  command roll %+.3f yaw %+.3f",
                     self.phase, roll, yaw, self.roll, self.target)

    def waist(self, batch: int) -> np.ndarray:
        return np.tile(np.float32([self.roll, self.target]), (batch, self.horizon, 1))

    def log(self, r: dict) -> dict:
        logging.info("SERVO read %d  E %.2f (tau %.2f)  best %+.2f", r["reads"], r["energy"], r["tau"], r["best"])
        if r["done"]:
            logging.info("SERVO done  E %.2f -> %.2f  waist yaw %+.3f  %s", r["energy0"], r["best_energy"],
                         r["waist_yaw"], r["stop"])
        return r


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    server = WaistServer()
    policy_server = PolicyServer(policy=server.policy, host="0.0.0.0", port=paths.PORT)
    policy_server.register_endpoint("get_action", server.get_action)
    policy_server.register_endpoint("reset", server.reset)
    logging.info("waist servo + policy ready on port %d (checkpoint %s)", paths.PORT, paths.CHECKPOINT.name)
    policy_server.run()


if __name__ == "__main__":
    main()
