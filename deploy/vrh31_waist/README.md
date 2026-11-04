# VR-H3.1 waist servo + GR00T N1.7

One server for the whole run. At the start of each run it turns the robot's waist until the head camera
matches the demo view, then serves the checkpoint's actions as a plain GR00T `PolicyServer`.

## GPU machine

```bash
source <repo>/external/Isaac-GR00T/.venv/bin/activate
pip install -U huggingface_hub
hf sync hf://buckets/twanghcmut/robot-ood-iclr/vrh31_waist/prior <repo>/outputs/vrh31_waist/artifacts/prior
export VRH31_CKPT=<gr00t-n17-vrh31-pick-100eps checkpoint directory>
cd <repo>/deploy/vrh31_waist
python check.py        # must print "All checks passed"
python serve.py        # loads ~12 GB, stays running, Ctrl-C stops it
```

`check.py` fails if `VRH31_CKPT` is not the `gr00t-n17-vrh31-pick-100eps` checkpoint the prior was built
from. Settings live in `paths.py`, each also settable through its env var:

| name | set to |
|---|---|
| `CHECKPOINT` (`VRH31_CKPT`) | the GR00T checkpoint directory (`config.json` inside) |
| `ARTIFACTS` (`VRH31_ARTIFACTS`) | the directory `prior/` is synced into |
| `PORT` (`SERVER_INFERENCE_PORT`) | port the server listens on |
| `GPU` (`VRH31_GPU`) | GPU index |

## Robot machine

The VLA client sends the usual observation with one more state key, the waist joint positions in rad with shape (B, T, 2):

```python
"waist": np.concatenate([waist_roll, waist_yaw], axis=-1)[None].astype("float32")
```

Every action it gets back has one more key, `action["waist"]` of shape (B, H, 2): `[waist_roll, waist_yaw]`
in rad for each step. Send it to `/waist_joint_trajectory_controller/follow_joint_trajectory` with
`joint_names: [waist_roll_joint, waist_yaw_joint]`, the same way the arm actions go to their controllers.

A run goes like this:

1. While the servo runs, the arm actions repeat the arm pose of the first observation, so the arms hold
   still, and `waist` carries the yaw the servo wants to look from.
2. Once `state["waist"]` has sat within 0.02 rad of that yaw for 1.5 s, the servo reads the `cam_head` frame
   of that call and picks the next yaw. It needs 1 to 8 reads.
3. The waist then goes to the best yaw and settles for 1.5 s. From the next call on, the arm actions come
   from the checkpoint and `waist` stays at that yaw for the rest of the run.

Roll stays where the run found it. A new run starts when the server starts, when the client calls `reset`,
or on the first `get_action` after 10 s without one. If the waist does not reach a target within its travel
time at 0.5 rad/s plus 3 s, `get_action` raises an error naming the target.

Every `get_action` logs one `WAIST` line with the phase, the waist position the client sent, and the
waist command sent back. The server log also shows one `SERVO read N` line per frame, then `SERVO done` and `serving the checkpoint's
actions`. `certified` means the view is back inside the demos' range, `already nominal` that the waist did
not need to move.

`python dummy_infer.py` plays the robot against a running `serve.py` with a waist that jumps to each command,
and prints the waist command of every call until the arms start moving.
