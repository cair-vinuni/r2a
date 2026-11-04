# Configuration

**Code:** `src/robot_ood/core/paths.py`, `src/robot_ood/core/config.py`,
`src/robot_ood/eval/policy.py`, `src/robot_ood/eval/bench.py`

Three independent axes describe a run: a **policy** (the frozen VLA), a **tree** (one demo corpus
turned into one artifact directory), and a **bench** (the measurement). Each owns one config file;
nothing is declared twice.

## Ownership

| config | owns |
|---|---|
| `configs/robot_ood.yaml` | filesystem roots, global training defaults, the axis list |
| `configs/trees/<tree>.yaml` | one artifact tree: corpus paths, policy id, artifact directory, checkpoints, per-tree train overrides |
| `evals/policies/<policy>.yaml` | one frozen policy: conda env, server launch argv, cache client, checkpoint key |
| `evals/<bench>/bench.yaml` | one benchmark: client entry point, axes, task-suite names, results prefix |
| `configs/sr_ladder.yaml` | the measurement plan: rungs, cells, expected episode counts, gates |
| `<tree>/run_config.json` | machine-written snapshot of one training run; never a config source |

The tree file's stem is the suite name: `--suite long` reads `configs/trees/long.yaml`. A new
tree — a new corpus, another policy, a reseed — is one new file there
(`long_seed2.yaml` pointing at `long_seed2/artifacts`), no code change.

Three optional keys describe a corpus that is not LIBERO's: `ee_from_obs: true` makes the
configuration a proxy (the end pose read off the observation, posed without a kinematic chain);
`arm` and `tasks` name the arm and the task list a converter writes the corpus from; `lanes` on a
bimanual tree names one lane tree per arm, each corrected by its own chain
([`evals/robotwin/README.md`](../evals/robotwin/README.md)). The entry corrector walks any tree
whose rows are joints: a joint-space robot, or one arm of one (a tree naming `arm`); a proxy
without an arm deploys without it.

A tree file:

```yaml
policy: stablevla                  # which evals/policies/<id>.yaml built and runs this tree
task_suite: libero_10              # benchmark name the eval client expects
data: {hdf5: libero_hdf5/libero_10, fwm: fwm/long}
artifacts: long/artifacts          # under roots.outputs; what $GR_GRAPH_DIR pins
eval:
  stablevla_ckpt: stablevla/hf_weights/long
  light_filter: evals/libero_plus/suite_filters/libero_10_Light_Conditions.txt
train:                             # optional; same shape as defaults.train
  locator: {epochs: 40}
```

A policy file:

```yaml
id: stablevla
conda_env: stablevla               # resolves the interpreter for server and cache client
ckpt_key: stablevla_ckpt           # which trees/<t>.yaml eval key names the checkpoint
server:
  script: evals/stablevla/stablevla_policy_server.py
  args: ["--ckpt_path", "{checkpoint}", "--port", "{port}", "--cuda", "0", "--use_bf16"]
placement: {servers_per_gpu: 4, clients_per_server: 2}   # eval and cache-policy sizing per GPU
pythonpath: [stablevla, evals/libero_plus/harness]
```

`evals/policies/gr00t.yaml` adds `python: external/Isaac-GR00T/.venv/bin/python` — a policy whose
interpreter is not a conda env names it directly; `resolve_policy_python` prefers it over
`conda_env`. Install: [`evals/gr00t/README.md`](../evals/gr00t/README.md).

## Precedence

```
CLI flag  >  environment variable  >  yaml  >  code default
```

Environment variables are for one-off experiments; configuring an installation means editing the
yaml files.

| variable | overrides |
|---|---|
| `ROBOT_OOD_CONFIG` | which root yaml is read (its sibling `trees/` is read with it) |
| `ROBOT_OOD_DATA`, `ROBOT_OOD_OUTPUTS`, `ROBOT_OOD_RESULTS` | the three filesystem roots |
| `GR_GRAPH_DIR` | the artifact directory every loader reads |
| `ROBOT_OOD_CONSTANTS` | a C0 calibration file, ahead of the copy embedded in `g_fcsr.npz` |
| `ROBOT_OOD_POLICY` | the default policy descriptor |
| `ROBOT_OOD_PY` | the interpreter `cache-policy` runs children under |
| `MODE` | the arm: `base` or `fcsr` (set per cell by the eval driver) |

Calibrated constants resolve in order: an explicit `set_constants` install, the
`$ROBOT_OOD_CONSTANTS` file, the copy embedded in `g_fcsr.npz`, then code defaults with a warning
(`robot_ood.core.config.calibrated_constants`).

## run_config.json

`train locator` writes `run_config.json` beside the checkpoint: git commit, command, policy id,
resolved hyperparameters, paths, the constants in effect, and the selected best epoch. `eval`
reads only its `meta.policy` to pick the server; everything else is a receipt. Edit yaml, never
this file.
