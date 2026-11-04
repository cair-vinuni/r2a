# ALOHA right arm with GR00T N1.7 and the corrector

The deployment runs on two machines. The GPU machine runs `serve.py`, which puts GR00T's
`Gr00tPolicy` behind GR00T's own `PolicyServer` with the corrector sitting in between. The robot
machine runs `aloha_infer_right.py`, the plain ROS2 inference node. Isaac-GR00T is imported, never
modified.

For the first few requests the server returns chunks that walk the arm to the demonstrated start
pose. After that, every policy chunk is corrected before it goes back to the node.

## GPU machine

```bash
source <repo>/external/Isaac-GR00T/.venv/bin/activate
pip install -U huggingface_hub
hf sync hf://buckets/twanghcmut/robot-ood-artifact/aloha_right/artifacts <repo>/outputs/aloha_right/artifacts
cd <repo>/deploy/aloha_right
python check.py        # must print "All checks passed"
python serve.py        # loads ~12 GB, stays running, Ctrl-C stops it
```

`paths.py` holds the settings. Each one can also be set through the env var shown in parentheses.

| name | set to |
|---|---|
| `ARTIFACTS` (`ROBOT_OOD_GRAPH_DIR`) | the synced artifacts directory |
| `CHECKPOINT` (`ALOHA_CKPT`) | the GR00T checkpoint directory (`config.json` inside) |
| `PORT` (`SERVER_INFERENCE_PORT`) | port the server listens on |
| `GPU` (`ALOHA_GPU`) | GPU index |

The server log shows one line per chunk. You see `HOMING ...` while the arm walks in, then either
`CORRECTOR |dq|max ...` or `CORRECTOR pass-through`. Setting `ALOHA_HOMING=0` skips the walk, which
does not avoid obstacles.

## Robot machine

```bash
export SERVER_INFERENCE_IP=<GPU machine address>   # default in paths.py
python aloha_infer_right.py
```

The prompt is `LANGUAGE_INSTRUCTION` at the top of `aloha_infer_right.py`. It goes to both the
policy and the corrector, which matches it to one of the two tasks the graph was built on, drawer
and kitchen. Both wordings are embedded in the artifacts, so no language model loads. If you need a
new wording, tell us and we will re-embed it and update the artifacts.

## Verify (needs the LeRobot datasets)

```bash
python -u verify/homing_openloop.py     # PASS
python -u verify/replay_corrector.py    # PASS
python -u verify/language_ab.py         # PASS
```
