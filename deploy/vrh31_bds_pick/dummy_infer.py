import sys
import threading
import time

import numpy as np

import paths

if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))

from gr00t.policy.policy import BasePolicy
from gr00t.policy.server_client import PolicyClient, PolicyServer

KEYS = ("left_shoulder", "left_elbow", "left_wrist", "left_hand",
        "right_shoulder", "right_elbow", "right_wrist", "right_hand")
WIDTHS = (3, 1, 3, 6, 3, 1, 3, 6)
STATE_DIM = 26
CHUNK_ROWS = 16


def join(block):
    return np.concatenate([np.asarray(block[k], np.float64).reshape(-1, w)
                           for k, w in zip(KEYS, WIDTHS)], axis=-1)


def split(rows):
    out, start = {}, 0
    for k, w in zip(KEYS, WIDTHS):
        out[k] = rows[None, :, start:start + w].astype(np.float32)
        start += w
    return out


class DummyPolicy(BasePolicy):
    def __init__(self):
        super().__init__(strict=False)

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass

    def get_modality_config(self):
        return {}

    def reset(self, options=None):
        return {}

    def _get_action(self, observation, options=None):
        state = join(observation["state"])[0]                                   # [26]
        executed = np.asarray(options["history"]["executed"]).reshape(-1, STATE_DIM)  # [H, 26] rows the robot actually ran
        measured = np.asarray(options["history"]["measured"]).reshape(-1, STATE_DIM)  # [H, 26] joints measured at those rows
        print(f"state{state.shape} executed{executed.shape} measured{measured.shape}")
        chunk = np.repeat(state[None, :], CHUNK_ROWS, axis=0)                   # [16, 26]
        return split(chunk), {"base_shift": np.zeros(3, np.float32)}            # [3] dx dy dyaw


def selftest(port):
    threading.Thread(target=lambda: PolicyServer(DummyPolicy(), port=port).run(), daemon=True).start()
    client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=10_000)
    while not client.ping():
        time.sleep(0.1)
    state = np.zeros(STATE_DIM, np.float32)
    executed, measured = [], []
    for request in range(2):
        obs = {"video": {"cam_head": np.zeros((1, 1, 600, 960, 3), np.uint8)},
               "state": split(state[None])}
        history = {"executed": np.asarray(executed, np.float32).reshape(-1, STATE_DIM),
                   "measured": np.asarray(measured, np.float32).reshape(-1, STATE_DIM)}
        action, info = client.get_action(obs, options={"history": history})
        chunk = join(action)
        print("request", request, "->", chunk.shape, info["base_shift"])
        for row in chunk:
            executed.append(row)
            measured.append(row + 0.01)
        state = measured[-1]
    client.kill_server()


if __name__ == "__main__":
    selftest(int(sys.argv[1]) if len(sys.argv) > 1 else paths.PORT)
