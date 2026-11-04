# RoboTwin 2.0-Plus

The robustness benchmark of [Robot-Robustness/RoboTwin2.0-Plus](https://github.com/Robot-Robustness/RoboTwin2.0-Plus),
run as `--bench robotwin_plus`: the fifty RoboTwin 2.0 tasks on the Aloha-AgileX embodiment, each
axis one structured perturbation of the scene, the robot, the camera, the language or the sensor.
The taxonomy is LIBERO-Plus' -- both come from the same paper -- so the axes carry the LIBERO-Plus
names. The policy is RoboTwin's own pi0.5 cotrain checkpoint ([`evals/pi05/README.md`](../pi05/README.md)).

```bash
python -m robot_ood.cli bench setup robotwin        # the 2.0 platform: env, assets, Vulkan manifest
bash evals/robotwin_plus/setup.sh                   # the Plus checkout beside it
python -m robot_ood.cli eval --bench robotwin_plus --suite plus50 --axis Original \
    --mode base --policy pi05_robotwin --gpus 0,1,2,3
```

`--suite plus50` is the fifty tasks as one cell, which is what one row of the benchmark's table is;
`--suite g1 .. g5` are the same five groups of ten the 2.0 bench uses, and `plus_smoke` is three
tasks. `trials_per_task` is 1 here, against the 2.0 bench's 20; the group cells in
[`docs/eval_robotwin_plus.md`](../../docs/eval_robotwin_plus.md) run ten trials per task.

## The axes

| Axis | Task config | What it perturbs |
|---|---|---|
| Original | `demo_clean` | nothing: the clean scene |
| Camera | `demo_camera` | head-camera distance (C1) and orientation (C3); spherical position (C2) ships disabled |
| Robot | `demo_robot_state` | initial joint angles, Gaussian and clipped, plus gripper extremes |
| Language | `demo_language_plus` | the instruction: distraction (R1), rewording (R2), reasoning chain (R3) |
| Light | `demo_light` | diffuse tint (L1), direction (L2), specular (L3), shadows (L4) |
| Background | `demo_background_plus` | wall and floor colour (B1+), table material (B2) |
| Noise | `demo_vision_noise` | one of motion, gaussian, zoom blur, fog, glass blur per episode (N1-N5) |
| Layout | `demo_objects_plus` | three to fifteen distractors (O1+) and target pose noise (O2) |

Every perturbation is the checkout's own, applied inside the environment: the config the axis names
is the only difference between two cells.

## What one cell runs

`evals/robotwin/harness/eval_robotwin.py`, the 2.0 bench's client, unchanged but for the
instruction pool -- the Plus checkout is the 2.0 platform, so the rollout, the seeds, the protocol
`pi05_robotwin` and the observation are the ones [`evals/robotwin/README.md`](../robotwin/README.md)
describes. Two differences:

- the phrasing pool is `--instruction-type auto`, which reads the axis' own `language.mode` the way
  the checkout's `script/eval_policy.py` derives it: the benchmark's `seen` pool everywhere but the
  Language axis, and there the R1+R2+R3 variants of `description/task_instruction_plus/`;
- seeds are cached under `evals/robotwin_plus/seeds/` (`ROBOTWIN_SEEDS`), not the 2.0 bench's. A
  task config of one name does not describe the same scene in both checkouts, and a seed is only
  valid for the scene its expert demonstration planned in.

## A task the Layout axis cannot build

`put_bottles_dustbin` has no scene that settles under `demo_objects_plus`: the object-pose noise
(2 cm, +/- 15 degrees) topples its three bottles on every seed -- six hundred candidates from 2000,
none accepted. The client leaves such a task out and says so, and the cell reports itself one task
short rather than losing a shard; the Layout column is 49 tasks.

## Install

`setup.sh` stands on `evals/robotwin/setup.sh`: same conda environment, and the fifteen gigabytes
of assets and the curobo checkout are linked from `external/RoboTwin` rather than fetched twice.
It applies the same shader-from-env patch, so `ROBOTWIN_SHADER` selects the rasteriser for a quick
look and the default stays the benchmark's ray tracer.

## Pinned

RoboTwin2.0-Plus `8f07761`, on the RoboTwin 2.0 environment of [`evals/robotwin/README.md`](../robotwin/README.md).
