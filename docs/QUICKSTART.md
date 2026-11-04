# Quickstart

Two paths to a scored cell: fetch a trained tree from the bucket and eval immediately, or
build one yourself. Install first; every shell starts with `source ./env.sh`.

## Install

| component | command | detail |
|---|---|---|
| repo | `git clone <this-repo> robot-ood && cd robot-ood` | |
| policy StableVLA | trunk + `./install_stablevla.sh` + weights | [`evals/stablevla/README.md`](../evals/stablevla/README.md) |
| bench LIBERO-Plus | `python -m robot_ood.cli bench setup libero_plus` | [`evals/libero_plus/README.md`](../evals/libero_plus/README.md) |
| policy GR00T (optional) | checkout + uv env + gated weights | [`evals/gr00t/README.md`](../evals/gr00t/README.md) |

```bash
python -m robot_ood.cli eval --suite long --mode base --check-env
```

## Fetch and eval immediately

No demonstrations needed; each fetch installs `outputs/<suite>/` and re-runs the deploy gate on
arrival.

StableVLA, `long`:

```bash
python -m robot_ood.cli artifacts fetch long --suite long
python -m robot_ood.cli eval --suite long --mode fcsr --cell --axis Robot_Initial_States --gpus 0   # smoke
python -m robot_ood.cli eval --suite long --mode fcsr --rung R4 --gpus 0                            # full battery
python evals/libero_plus/score.py --suites long --methods base:stablevla_base fcsr:stablevla_fcsr
```

GR00T, `long`:

```bash
python -m robot_ood.cli artifacts fetch long_gr00t --suite long_gr00t
python -m robot_ood.cli eval --suite long_gr00t --mode fcsr --rung R1 --gpus 0
```

## Build, train, eval step by step

Demonstrations first (13.7 GB):

```bash
python -c "
from robot_ood.core.paths import default_paths
from libero.libero.utils.download_utils import download_from_huggingface
dest = default_paths().data / 'libero_datasets'
download_from_huggingface('libero_10', str(dest), check_overwrite=False)"
```

Then one GPU, about a GPU-day; each step prints what it wrote and where:

```bash
python -m robot_ood.cli build --suite long --device cuda:0      # graph, segments, calib, lang, policy cache
python -m robot_ood.cli train --suite long --device cuda:0      # locator, entry, fcsr; ends in the deploy gate
python -m robot_ood.cli eval  --suite long --mode fcsr --cell --axis Robot_Initial_States --gpus 0
```

Same commands with `--suite long_gr00t` build and train the GR00T tree against GR00T's own
cached actions. `build --dry-run` shows what is stale and why; [SETUP.md](SETUP.md) has the
full spine, [operations.md](operations.md) the artifact contract.
