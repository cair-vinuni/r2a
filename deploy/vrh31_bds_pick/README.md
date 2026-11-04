# VR_H31 bimanual pick with GR00T N1.7 and the corrector

The deployment runs on two machines. The GPU machine runs `serve.py`, which puts GR00T's
`Gr00tPolicy` behind GR00T's own `PolicyServer` with the corrector in between. The robot machine
runs your own ROS2 client, which sends observations to the server and applies the joint targets that
come back.

## GPU machine

```bash
source <repo>/external/Isaac-GR00T/.venv/bin/activate
pip install -U huggingface_hub
hf sync hf://buckets/twanghcmut/robot-ood-iclr/artifact_vrh31 <repo>/outputs/vrh31_artifacts
export VRH31_GRAPH_DIR_LEFT=<repo>/outputs/vrh31_artifacts/left
export VRH31_GRAPH_DIR_RIGHT=<repo>/outputs/vrh31_artifacts/right
cd <repo>/deploy/vrh31_bds_pick
python check.py        # must print "All checks passed"
python serve.py        # loads ~12 GB, stays running, Ctrl-C stops it
```

Set each path either through the env var in parentheses, or by editing its default in `paths.py`,
for example `CHECKPOINT = _path("VRH31_CKPT", Path("/home/vrh31/checkpoints/vrh31_bds_pick_local"))`.

| name | set to |
|---|---|
| `ARTIFACTS["left"]` (`VRH31_GRAPH_DIR_LEFT`) | the left lane's artifacts directory |
| `ARTIFACTS["right"]` (`VRH31_GRAPH_DIR_RIGHT`) | the right lane's artifacts directory |
| `CHECKPOINT` (`VRH31_CKPT`) | the GR00T checkpoint directory (`config.json` inside) |
| `PORT` (`SERVER_INFERENCE_PORT`) | port the server listens on |
| `GPU` (`VRH31_GPU`) | GPU index |

The task prompt is `LANGUAGE_INSTRUCTION` at the top of `serve.py`, and the corpus holds a single
task.

## Robot machine

Point your ROS2 client at the server. It reads `SERVER_INFERENCE_IP` and `SERVER_INFERENCE_PORT`,
with defaults in `paths.py`.

The server reads actual execution history from each `get_action` request rather than assuming
the last chunk ran intact.

Maintain two [H, 26] arrays on the robot: `executed` (blended targets) and `measured` (joint
states). Pass both as `options={"history": {...}}` on every `get_action`, as `dummy_infer.py` does.
Leave the key out to fall back to unrolling the server's own chunk.

```bash
python dummy_infer.py 5599              # client + dummy policy in one process, two requests
python verify/history_client.py         # plays the robot against a running serve.py
```

## Verify (needs the LeRobot dataset)

```bash
python -u verify/homing_openloop.py     # both lanes
python -u verify/replay_corrector.py    # contract: hands and the other arm untouched
python -u verify/language_ab.py         # reports whether language is live on one task
```
