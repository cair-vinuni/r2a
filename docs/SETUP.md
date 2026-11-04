# Setup

Fast path (fetch the trained tree, no demos, no training): [QUICKSTART.md](QUICKSTART.md).
Per-component installs: [StableVLA](../evals/stablevla/README.md),
[GR00T](../evals/gr00t/README.md), [LIBERO-Plus](../evals/libero_plus/README.md).

Everything is written inside the clone; `rm -rf robot-ood/` removes all of it.
`long` = `libero_10`; swap for `object` / `spatial` / `goal`, nothing else changes.
Every shell needs `source ./env.sh` first.
~28 GB downloads, ~11 GB packages, ~2.5 h to a `base` cell, plus the artifact build.

## Clone

```bash
git clone <this-repo> robot-ood && cd robot-ood
```

Then the StableVLA trunk: [`evals/stablevla/README.md`](../evals/stablevla/README.md).

## Environments

```bash
./install_stablevla.sh     # idempotent; re-run after any failure
source ./env.sh            # every shell from here on; bare `python` = this repo's interpreter
```

## Benchmark

```bash
python -m robot_ood.cli bench list
python -m robot_ood.cli bench setup libero_plus     # idempotent, ends in a preflight
python -m robot_ood.cli bench setup libero          # upstream LIBERO; --bench libero selects it
python -m robot_ood.cli bench setup robotwin        # RoboTwin 2.0; --bench robotwin selects it (evals/robotwin/README.md)
```

## Weights

StableVLA: [`evals/stablevla/README.md`](../evals/stablevla/README.md).
GR00T: [`evals/gr00t/README.md`](../evals/gr00t/README.md).
pi0.5: `evals/pi05/setup.sh` ([`evals/pi05/README.md`](../evals/pi05/README.md)).
RoboTwin's pi0.5 checkpoint comes with `evals/pi05/setup.sh`.

## Install check

Run before the 13.7 GB download and before any GPU time. `base` needs no demos and no artifacts.

```bash
python -m robot_ood.cli eval --suite long --mode base --check-env
python -m robot_ood.cli eval --suite long --axis Robot_Initial_States --mode base --cell
```

## Demonstrations

```bash
python -c "
from robot_ood.core.paths import default_paths
from libero.libero.utils.download_utils import download_from_huggingface
dest = default_paths().data / 'libero_datasets'
download_from_huggingface('libero_10', str(dest), check_overwrite=False)"
```

Pass the bench name (`libero_10`, `libero_object`, `libero_spatial`, `libero_goal`) — asking for
`long` downloads nothing. Check — fewer than 10 files means an interrupted download; re-run:

```bash
python -c "from robot_ood.core.paths import default_paths as d; p=d().hdf5('long'); \
           import glob; print(p, len(glob.glob(f'{p}/*.hdf5')), 'files')"
# -> .../data/libero_datasets/libero_10 10 files
```

## Artifacts

Every arm but `base` reads `outputs/<suite>/artifacts/`. Fetch the trained tree (gated on
arrival, prints the variables to export):

```bash
python -m robot_ood.cli artifacts fetch long --suite long
```

Or build it — the run ends in the deploy gate, so reaching the end is the verification:

```bash
python -m robot_ood.cli build --suite long --device cuda:0        # graph, segments, calib, lang, cache
python -m robot_ood.cli train --suite long --device cuda:0        # locator, entry, fcsr, deploy gate
python -m robot_ood.cli build --suite long --dry-run              # what is stale, and why
python -m robot_ood.cli build --suite long --profile smoke        # 7 min, untrained heads, NOT the method
```

`calibrate` exits 2 on `long`: expected, the artifact is still written. Long steps detach; follow
`outputs/<suite>/pipeline_logs/`. Re-check any tree:

```bash
python -m robot_ood.cli artifacts check outputs/long/artifacts --stage deploy
```

## Run

`base` = frozen policy, no artifacts; `fcsr` = the method, needs the tree.

```bash
python -m robot_ood.cli eval --suite long --axis Robot_Initial_States --mode fcsr --cell   # smoke, ~12 episodes
python -m robot_ood.cli eval --suite long --mode fcsr --rung R4 --gpus 0 --axis Robot_Initial_States   # measured, ~1 h
python evals/libero_plus/score.py --suites long --methods base:stablevla_base fcsr:stablevla_fcsr
```

What matters in the output: `EXIT=0` and the `mode=... graph_dir=... artifact_digest=...` line.
A rerun goes to its own tree: add `--results-root results_rerun` (rerunning a cell in place is
refused).

## Tests

```bash
.local/conda/envs/onf-core/bin/python -m pytest tests/ -q
```

Each skip names the section that produces its missing input.

## NVML preload

This box's kernel nvidia module (570.172.08) and userspace NVML (580.173) mismatch:
`nvmlInit_v2` hard-asserts, killing `nvidia-smi` and torch's allocator when it grows its pool.
`robot_ood.pipeline.interpreter.nvml_preload_path` resolves a matching library to `LD_PRELOAD`
into every spawned child; `ROBOT_OOD_NVML_PRELOAD` overrides it. A box without the known-good
file silently gets no preload.

## Upstream LIBERO

`--bench libero` runs the benchmark as published, each policy under its authors' rollout
([`docs/PROTOCOLS.md`](PROTOCOLS.md)); install, weights, commands and the output layout are in
[`evals/libero/README.md`](../evals/libero/README.md).

## GPU rendering

The container carries `libcuda` but no `libEGL_nvidia`, so MuJoCo renders on the CPU by default.

```bash
scripts/setup_nvidia_gl.sh            # once per driver version; extracts the driver's EGL libraries
source env.sh                          # picks them up by driver version
```

`robot_ood.eval.egl_devices` maps each cell's CUDA device to its EGL device, so `--gpus 0,1,2,3`
renders on every GPU; `run.json` records the `MUJOCO_EGL_DEVICE_ID` a cell used.

### Placing policy servers

A cell gets GPUs from the schedule (a lone cell takes every idle GPU, cells of a rung take one
each, `--per-gpu N` stacks N cells on a GPU) and its policy's descriptor turns them into servers
and clients (`evals/policies/<id>.yaml`, `placement`): `servers_per_gpu` server processes on
each GPU, `clients_per_server` sim clients per server. Episodes are dealt round robin over the
clients, so any client count balances. Override per run:

```bash
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy gr00t \
    --gpus 0,1,2,3 --servers-per-cell 4 --num-clients 24
```

The GR00T and pi0.5 servers fold the requests of their clients into one forward
(`--max-batch` in the descriptor); StableVLA's pipeline is one item at a time, so its lever is
servers per GPU ([evals/libero/README.md](../evals/libero/README.md), Throughput).
