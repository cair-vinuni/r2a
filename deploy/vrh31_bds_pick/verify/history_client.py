import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dummy_infer import STATE_DIM, join, split
import paths

from gr00t.policy.server_client import PolicyClient


def main():
    client = PolicyClient(host="127.0.0.1", port=paths.PORT, timeout_ms=300_000)
    while not client.ping():
        time.sleep(1.0)
    client.reset()
    rng = np.random.default_rng(0)
    state = np.zeros(STATE_DIM, np.float32)
    executed, measured = [], []
    for request in range(int(os.environ.get("REQUESTS") or 3)):
        obs = {"video": {"cam_head": rng.integers(0, 255, size=(1, 1, 600, 960, 3), dtype=np.uint8)},
               "state": split(state[None])}
        history = {"executed": np.asarray(executed, np.float32).reshape(-1, STATE_DIM),
                   "measured": np.asarray(measured, np.float32).reshape(-1, STATE_DIM)}
        t0 = time.time()
        action, info = client.get_action(obs, options={"history": history})
        chunk = join(action)
        print(f"request {request}: history {len(executed)} rows -> chunk {chunk.shape} "
              f"in {time.time() - t0:.1f}s", flush=True)
        for row in chunk:
            executed.append(row)
            measured.append(row + rng.normal(scale=0.005, size=STATE_DIM))
        state = np.asarray(measured[-1], np.float32)


if __name__ == "__main__":
    main()
