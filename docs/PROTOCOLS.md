# Rollout protocols

A policy is scored under the rollout its authors published. The harness replays that recipe; the
policy descriptor (`evals/policies/<id>.yaml`, key `protocol`) names it, and `fcsr` on top of a
policy inherits it. `run.json` and `summary.json` record the protocol of every cell. Values are
defined once, in `src/robot_ood/core/protocol.py`, copied from the vendor file each column cites.

| | `openvla_oft` | `nvidia` | `openpi` | `pi05_robotwin` |
|---|---|---|---|---|
| policies | `stablevla` | `gr00t` | `pi05` | `pi05_robotwin` |
| benchmark | LIBERO | LIBERO | LIBERO | RoboTwin 2.0 |
| source | `stablevla/experiments/robot/libero/run_libero_eval.py` | `external/Isaac-GR00T/gr00t/eval/rollout_policy.py` | `external/openpi/examples/libero/main.py` | `external/RoboTwin/scripts/eval_policy.py` (XPolicyLab `policy/Pi_05`) |
| initial state | the suite's 50 stored states, cycled per episode | `env.reset()` as is | the suite's 50 stored states, cycled per episode | the first `trials_per_task` seeds from 2000 whose expert demo succeeds |
| settling steps | 10 | 0 | 10 | 0 |
| step budget | spatial 220, object 280, goal 300, libero_10 520 | 504, every suite | spatial 220, object 280, goal 300, libero_10 520 | the benchmark's own per-task limit (`_eval_step_limit.yml`) |
| seed | 7 | 7 | 7 | 0 |
| frames on the wire | 224 px | 256 px | 256 px, resized to 224 by openpi's own `resize_with_pad` in the server | as rendered, 320 x 240, three cameras; openpi's Aloha transform pads and resizes to 224 |
| actions executed per query | 8 | 8 | 5 | 50, the whole chunk |
| gripper command | open value thresholded at 0.5 to -1 / +1 | open value thresholded at 0.5 to -1 / +1 | the policy's value as is | the gripper fraction as is |

Shared by the three LIBERO protocols: 256 px offscreen render, both views rotated 180 degrees,
proprio `[eef_pos(3), axis_angle(3), gripper_qpos(2)]`, the instruction read from the task's BDDL
file, success when the environment reports done. Under `pi05_robotwin` the state is the fourteen
commanded joints (six per arm and a gripper fraction), the instruction is one of the benchmark's
own phrasings of the expert episode, and every action of a chunk is an absolute joint target the
benchmark time-parameterises and simulates ([`evals/robotwin/README.md`](../evals/robotwin/README.md)).

Two things the parallel harness does that the vendor scripts do not, neither of which changes
what a policy sees: a cell's episodes are dealt round robin over its sim clients, each with its
own environment (under `nvidia`, which random state `env.reset()` draws for an episode therefore
depends on the client count; the distribution does not); and the GR00T and pi0.5 servers answer
several clients from one batched forward. Batched and single-item actions differ by exactly the
sampling noise two single-item calls differ by (measured in-process, 8 items).

Under `fcsr`, the belief's behaviour channel scores the first K rows of each policy chunk
against K-row reference segments (K = 8, `GR_SEG_K`); pi0.5's 10-row chunk is judged on its
first 8, and only the protocol's 5 are executed.

## Calibration against the vendor stack

Each protocol was checked once by running the vendor's own code on the same machine
(`docs/REFERENCE.md`). GR00T N1.7 on `libero_10`, 500 episodes: NVIDIA's `rollout_policy.py`
455/500, this harness under `nvidia` 440/500.
