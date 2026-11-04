# pi0.5 (openpi)

**Checkout:** `external/openpi` @ `215abfb`, its own uv environment at `external/openpi/.venv`
**Weights:** `pi05/openpi-assets/checkpoints/pi05_libero` (relocate with `$PI05_HOME`), the
published `gs://openpi-assets/checkpoints/pi05_libero`
**Code:** `evals/pi05/ws_server.py` (policy server), `evals/policies/pi05.yaml` (descriptor),
protocol `openpi` ([docs/PROTOCOLS.md](../../docs/PROTOCOLS.md))

## Install

```bash
evals/pi05/setup.sh
```

Clones openpi at the pinned commit, runs `uv sync` (Python 3.11, JAX CUDA wheels), downloads the
checkpoint with openpi's own downloader, and checks the imports. Idempotent.

## Run

```bash
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy pi05 --gpus 0,1,2,3
```

Two servers per GPU, three clients each (`evals/policies/pi05.yaml`); a server samples the
requests of its clients in one padded batch (`--max-batch 8`).

`--suite object | spatial | goal | long`; `--bench libero_plus --axis <axis>` runs the perturbed
benchmark with the same checkpoint.

## Layout

| | |
|---|---|
| server interpreter | `external/openpi/.venv/bin/python` |
| server | `evals/pi05/ws_server.py --checkpoint <dir> --port <port> --max-batch 8` |
| what the server does | `create_trained_policy(get_config("pi05_libero"), <dir>)`, frames through `resize_with_pad(224)` as in `examples/libero/main.py`; one request goes through `Policy.infer` unchanged, a gathered batch through the same transforms and `sample_actions` at once |
| results | `results/libero_<task_suite>/Original/pi05_base_<suite>_Original/` |

## Finetuning on RoboTwin

`--policy pi05_robotwin_clean` serves pi0.5 finetuned from its base weights
(`gs://openpi-assets/checkpoints/pi05_base`) on a RoboTwin group's fifty clean demonstrations per
task, under the config RoboTwin's leaderboard finetunes with (`policy/pi0`'s `*_aloha_robotwin_full`:
full finetune, global batch 32, 30k steps, openpi's default cosine schedule, EMA 0.99), stated for
pi0.5 in `evals/pi05/robotwin_config.py` (`finetune_config`). Trained on Clean and evaluated on
Randomized it is an out-of-distribution policy, unlike the cotrain checkpoint.

```bash
P=external/openpi/.venv/bin/python
$P evals/pi05/robotwin_data.py --suite g1 --repo-id robotwin_g1_clean      # LeRobot dataset, pi05/lerobot/
$P evals/pi05/train_robotwin.py norm-stats --repo-id robotwin_g1_clean     # pi05/assets/
CUDA_VISIBLE_DEVICES=0,1,2,3 $P evals/pi05/train_robotwin.py train --repo-id robotwin_g1_clean --exp-name run0
python -m robot_ood.cli eval --bench robotwin --suite g1 --axis Randomized --mode base --policy pi05_robotwin_clean --gpus 0,1,2,3
```

`scripts/research/pi05_clean_g1.sh` chains the four. The dataset is written the way RoboTwin's own
conversion writes it (state = the frame's commanded joints, action = the next frame's, one `seen`
phrasing per episode), at the cameras' native 320 x 240. Training splits the batch over every
visible GPU (`--fsdp-devices 1`); the checkpoint lands in `pi05/checkpoints/<repo-id>/<exp-name>/<step>`
with its norm stats under `assets/<repo-id>`, which is what the server reads; the suite's tree yaml
names it (`eval: pi05_robotwin_clean_ckpt`). Only the last checkpoint is kept (~50 GB with the
optimizer state; `params/` alone serves).

