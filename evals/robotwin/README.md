# RoboTwin 2.0

The bimanual tabletop benchmark of [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin),
run as `--bench robotwin`: fifty tasks on the Aloha-AgileX embodiment, evaluated in the two
leaderboard settings, `Clean` (task config `demo_clean`, Easy) and `Randomized` (`demo_randomized`,
Hard: random backgrounds, clutter, lighting and table height). The policy is RoboTwin's own pi0.5
cotrain checkpoint (`--policy pi05_robotwin`), or pi0.5 finetuned here on the group's clean
demonstrations alone (`--policy pi05_robotwin_clean`, out of distribution on Randomized); both in
[`evals/pi05/README.md`](../pi05/README.md).

```bash
python -m robot_ood.cli bench setup robotwin      # checkout, conda env, assets, Vulkan manifest
bash evals/pi05/setup.sh                          # openpi and the RoboTwin pi0.5 checkpoint
python -m robot_ood.cli eval --bench robotwin --suite g1 --axis Randomized --mode base --policy pi05_robotwin --gpus 0,1,2,3
```

The fifty tasks are five alphabetical groups of ten, `g1` .. `g5` (`suites/g*.txt`); a cell is one
group in one setting, 200 episodes at the default 20 trials per task (the leaderboard's 100 is
`--trials 100`). `suites/all.txt` is the full list.

## What one cell runs

`harness/eval_robotwin.py` replays the rollout of RoboTwin's XPolicyLab evaluation
(`external/RoboTwin/scripts/eval_policy.py` with `XPolicyLab/policy/Pi_05`), protocol
`pi05_robotwin` in `src/robot_ood/core/protocol.py`:

- an episode is one scene seed; the seeds of a task are the first `trials_per_task` from
  `2000 * (1 + seed)` whose expert demonstration plans and succeeds. `harness/find_seeds.py`
  finds them for every task at once and caches them under `seeds/<task_config>/`; a client whose
  task has no cached list searches it itself, under a file lock, so parallel shards search each
  task once;
- the instruction is the benchmark's own: the expert demonstration of the seed is replayed and
  described (`description/utils`), and one of the task's `unseen` phrasings is drawn;
- the policy sees the head and both wrist cameras as rendered (320 x 240) and the fourteen
  commanded joints (six per arm plus a gripper fraction); it answers fifty joint targets, all
  executed through `env.take_action(action_type="qpos")` before the next query;
- the episode ends on success or the benchmark's own per-task step limit
  (`task_config/_eval_step_limit.yml`);
- episodes are dealt round robin over the cell's sim clients (`shard_episodes`), so every shard
  runs the same episodes whichever server it talks to.

The observation is rendered only where the policy reads it, before each query, and per step only
while a replay video is kept (`--video failures|all`); rendering never touches the simulation.
A scene the stability check rejects is rebuilt, then its environment object, then the episode
runs on its spare seed (the cached list's entry past the cell's trials at its index); every
record names the seed it ran. Records go to `episodes.jsonl` per shard and `summary.json` per cell.

## The corrector on a bimanual robot

`fcsr` runs one corrector per arm. Each arm is a lane: a tree of its own
(`configs/trees/g1_left.yaml`, `g1_right.yaml`) whose corpus is a joint-space proxy configuration
-- the arm's six commanded joints and open flag, read off RoboTwin's own recordings -- and whose
actions are the joint steps between frames. Six joints stand where a pose would, so the tree
declares `pose_space: euclidean` and dims 3:6 add rather than compose. The group's bimanual tree
(`g1.yaml`) names its two lanes; the eval CLI gates both and hands them to the client, which turns
the policy's chunk of joint targets into each arm's steps, lets that arm's chain correct them, and
integrates the result back into targets (`robot_ood.core.absolute`). Without a correction the rows
are the policy's own, bit for bit. A lane's rows are joints, so the entry corrector walks each arm
onto its corpus's start before the policy first acts, the other arm holding its targets; its
control steps are charged to the episode.

```bash
python -m robot_ood.data.robotwin --suite g1_left      # the lane corpus from data/robotwin/raw
python -m robot_ood.data.robotwin --suite g1_right
python -m robot_ood.cli build all --suite g1_left --device cuda:0   # graph, calibration, policy cache
python -m robot_ood.cli train locator --suite g1_left --device cuda:0   # then the same for g1_right
python -m robot_ood.cli train fcsr --suite g1_left --device cuda:0
python -m robot_ood.cli train entry --suite g1_left
python -m robot_ood.cli eval --bench robotwin --suite g1 --axis Randomized --mode fcsr --policy pi05_robotwin --gpus 0,1,2,3
```

The raw demos are the fifty `aloha-agilex_clean_50.zip` archives of the RoboTwin2.0 dataset,
one per task, unpacked under `data/robotwin/raw/<task>/`.
Each archive's `instructions/episodeN.json` carries the seen phrasing pool of that episode; the
converter stores every distinct seen phrasing of the task in the demo file, and `build language`
fits the task head on that bank, so the head reads the natural phrasings the benchmark hands the
policy rather than the task ids.

## Throughput

Every action is a joint target: RoboTwin time-parameterises the step (TOPP, CPU) and simulates
it, so a client is CPU-bound and one GPU serves seven of them plus the policy server. Each client
still holds a curobo planner for the expert demonstration that describes the episode; start CUDA
MPS once per boot so those GPU kernels overlap instead of time-slicing:

```bash
nvidia-cuda-mps-control -d
```

Processes started before the daemon keep time-slicing. Give a training job its own GPU: seven
clients beside it slow its epochs several times over.

## Rendering

SAPIEN renders through Vulkan. The benchmark's own shader is the ray tracer (`rt`, 32 samples per
pixel, OIDN denoiser), which is what the RoboTwin datasets and the checkpoint were made with;
`setup.sh` applies `patches/shader-from-env.patch` so `ROBOTWIN_SHADER=default` selects the
rasteriser for a quick look, and every episode record names the shader. On a machine without a
Vulkan driver, `setup.sh` writes `$ROBOT_OOD_NVGL/nvidia_icd.json` for the NVIDIA libraries `env.sh`
already extracts for EGL; `env.sh` then exports `VK_DRIVER_FILES`. The A100 exposes
`VK_KHR_ray_tracing_pipeline`, so the ray tracer runs there. Scene construction draws from one
seeded numpy stream: never skip a camera or an actor to save work, the objects move.

## Pinned

RoboTwin `6e75b8b` (2025-10-22, the `envs/` + `task_config/` layout), SAPIEN 3.0.0b1, mplib 0.2.1
(with RoboTwin's own two-line planner edit), curobo `ebb7170`, warp-lang 1.8.1, pytorch3d stable,
torch 2.4.1. Assets (15 GB) come from the `TianxingChen/RoboTwin2.0` dataset.
