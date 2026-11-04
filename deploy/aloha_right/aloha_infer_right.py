"""ALOHA right-arm inference node for GR00T N1.7.

Targets a GR00T N1.7 checkpoint finetuned on our ALOHA right-arm data (withheld for review), whose modality config is:

    video : front (overhead), wrist (right wrist)   (B, T, H, W, C) uint8, T=1
    state : single_arm (6), gripper (1)             (B, T, D) float32, T=1
    action: single_arm (6), gripper (1)             returned as (B, 16, D)

The policy server is started with `python serve.py` in this folder.
"""

import os
import sys
import time
import traceback
from collections import deque
from threading import Lock, Thread

import cv2
import numpy as np
import rclpy
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState

from eval.server_client import PolicyClient

import paths

os.environ["SERVER_INFERENCE_IP"] = paths.SERVER_IP
os.environ["SERVER_INFERENCE_PORT"] = str(paths.PORT)

JOINT_STATE_SHAPE = 7  # 6 arm joints + 1 gripper
ARM_DOF = 6

# Use the exact wording the checkpoint was trained on — it has not seen paraphrases.
DRAWER_INSTRUCTION = (
    "Use the right gripper to open the top drawer, pick up the red square block, place it inside the top drawer, and then close the drawer."
)
KITCHEN_INSTRUCTION = (
    "Use the right gripper to pick up the banana and place it into the pot. Then pick up the lid with the right gripper and place it on top of the pot to close it."
)
LANGUAGE_INSTRUCTION = KITCHEN_INSTRUCTION


class InferencePublisher(Node):
    def __init__(self):
        super().__init__("aloha_right_arm_inference")

        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Publishers — right arm only
        self.right_arm_publisher = self.create_publisher(
            JointGroupCommand, "/follower_right/commands/joint_group", 10
        )
        self.right_hand_publisher = self.create_publisher(
            JointSingleCommand, "/follower_right/commands/joint_single", 10
        )

        # Subscribers — two cameras + right arm state
        self.create_subscription(
            CompressedImage,
            "/camera_high/camera/color/image_rect_raw/compressed",
            self.front_cam_callback,
            qos_profile,
        )
        self.create_subscription(
            CompressedImage,
            "/camera_wrist_right/camera/color/image_rect_raw/compressed",
            self.wrist_right_cam_callback,
            qos_profile,
        )
        self.create_subscription(
            JointState, "/follower_right/joint_states", self.right_joint_state_callback, 10
        )

        # State — every attribute the guard in call_inference() reads must exist up front,
        # otherwise the first pass raises AttributeError inside a thread that has no handler
        # and the loop dies silently.
        self.front_cam = None
        self.wrist_right_cam = None
        self.right_arm = None
        self.velocity_right_arm = None
        self.effort_right_arm = None
        self.prev_right_arm = None

        self.index_cam_front = 0
        self.index_cam_wrist_right = 0
        self.joint_states_times = deque(maxlen=50)
        self.front_cam_times = deque(maxlen=50)
        self.wrist_right_cam_times = deque(maxlen=50)

        self.lock = Lock()
        self.executed_length = 16
        assert self.executed_length <= 50
        self.smooth_step = 40
        self.end_action = None

        self.policy_client = PolicyClient(
            host=os.environ["SERVER_INFERENCE_IP"],
            port=int(os.environ["SERVER_INFERENCE_PORT"]),
            timeout_ms=120_000,
        )
        self.policy_client.reset()

        self.get_logger().info(
            f"Publisher and subscribers initialized. instruction={LANGUAGE_INSTRUCTION!r}"
        )

        self.main_thread = Thread(target=self.thread_inference, daemon=True)
        self.main_thread.start()

    # ------------------------------------------------------------------ callbacks
    def front_cam_callback(self, msg: CompressedImage):
        self.front_cam_times.append(time.time())
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            front_cam = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.index_cam_front += 1
            self.front_cam = cv2.cvtColor(front_cam, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode front image: {e}")

    def wrist_right_cam_callback(self, msg: CompressedImage):
        self.wrist_right_cam_times.append(time.time())
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            wrist_right_cam = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.index_cam_wrist_right += 1
            self.wrist_right_cam = cv2.cvtColor(wrist_right_cam, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode wrist image: {e}")

    def right_joint_state_callback(self, msg: JointState):
        self.joint_states_times.append(time.time())
        try:
            self.right_arm = [msg.position[:JOINT_STATE_SHAPE]]
            self.velocity_right_arm = msg.velocity[:JOINT_STATE_SHAPE]
            self.effort_right_arm = msg.effort[:JOINT_STATE_SHAPE]
            self.prev_right_arm = np.array(self.right_arm)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse joint states: {e}")

    # ------------------------------------------------------------------ inference
    def thread_inference(self):
        time.sleep(4)
        self.end_action = True
        while rclpy.ok():
            if self.end_action is True:
                self.call_inference()

    def build_observation(self):
        """Shapes and dtypes are asserted server-side in Gr00tPolicy._validate_observation."""
        joints = np.asarray(self.right_arm[0][:ARM_DOF], dtype=np.float32).reshape(1, 1, ARM_DOF)
        gripper = np.asarray(self.right_arm[0][ARM_DOF:JOINT_STATE_SHAPE], dtype=np.float32).reshape(1, 1, 1)
        return {
            "video": {
                # (B, T, H, W, C) uint8 — five dims, T == 1
                "front": self.front_cam[None][None],
                "wrist": self.wrist_right_cam[None][None],
            },
            "state": {
                # (B, T, D) float32 — three dims, T == 1
                "single_arm": joints,
                "gripper": gripper,
            },
            # nested under "language", each batch item a list holding one string
            "language": {"annotation.human.task_description": [[LANGUAGE_INSTRUCTION]]},
        }

    def call_inference(self):
        self.end_action = False
        if self.front_cam is None or self.wrist_right_cam is None or self.right_arm is None:
            self.get_logger().info("Waiting for full sensor data...")
            return

        obs = self.build_observation()

        try:
            start_time = time.time()
            action, _info = self.policy_client.get_action(obs, options=None)
            self.get_logger().info(f"INFERENCE_INTERVAL {time.time() - start_time}")

            # The policy returns the bare modality keys, not "action.right_arm".
            # Join them back into the 7-dim vector the publisher below slices.
            arm_raw = np.asarray(action["single_arm"], dtype=np.float32)
            grip_raw = np.asarray(action["gripper"], dtype=np.float32)
            self.get_logger().info(
                f"RAW ACTION keys={list(action)} "
                f"single_arm={arm_raw.shape} gripper={grip_raw.shape}"
            )

            # reshape thay vi squeeze(0): dung cho ca (1,16,6) lan (16,6)
            arm = arm_raw.reshape(-1, ARM_DOF)   # (16, 6)
            grip = grip_raw.reshape(-1, 1)       # (16, 1)
            chunk = np.concatenate([arm, grip], axis=-1)   # (16, 7)

            if self.smooth_step != 1:
                right_arm_traj = self.smooth_action(
                    np.concatenate((self.prev_right_arm, chunk), axis=0)
                )
            else:
                right_arm_traj = chunk

            execute_start_time = time.time()
            for i in range(self.executed_length * self.smooth_step):
                step_start = time.time()

                right_arm = right_arm_traj[i][: JOINT_STATE_SHAPE - 1]
                right_hand = right_arm_traj[i][JOINT_STATE_SHAPE - 1]

                if np.isnan(right_arm).any() or np.isnan(right_hand):
                    self.get_logger().error("NaN in predicted action — skipping publish")
                    break

                msg_arm = JointGroupCommand()
                msg_arm.cmd = right_arm.tolist()
                msg_arm.name = "arm"

                msg_hand = JointSingleCommand()
                msg_hand.cmd = float(right_hand)
                msg_hand.name = "gripper"

                self.right_arm_publisher.publish(msg_arm)
                self.right_hand_publisher.publish(msg_hand)

                time.sleep(max(0, 1 / 200 - (time.time() - step_start)))

            self.end_action = True
            self.get_logger().info(f"EXECUTION_INTERVAL {time.time() - execute_start_time}")

        except Exception as e:
            self.get_logger().error(f"Failed to get or publish action: {e}")
            print(traceback.format_exc())
            self.end_action = True  # otherwise one failure stops the loop forever

    # ------------------------------------------------------------------ helpers
    def smooth_action(self, action):
        result = []
        for i in range(action.shape[0] - 1):
            interp = np.linspace(action[i], action[i + 1], self.smooth_step, endpoint=False)
            result.append(interp)
        return np.concatenate(result, axis=0)

    def destroy(self):
        self.main_thread.join(timeout=2.0)
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    infer_node = InferencePublisher()
    try:
        rclpy.spin(infer_node)
    except KeyboardInterrupt:
        pass
    finally:
        infer_node.destroy()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
