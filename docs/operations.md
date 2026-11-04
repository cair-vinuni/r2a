# Operations

**Reads:** LIBERO demo HDF5, the policy checkpoints, the benchmark checkouts
**Writes:** `outputs/<suite>/artifacts/`, `results/`, `logs/`
**Code:** `src/robot_ood/cli/`, `scripts/run_sr.py`, `evals/libero_plus/score.py`

Five command groups cover the whole lifecycle:

| group | meaning | output |
|---|---|---|
| `build` | transform or measure the demo corpus; no model optimization | `g_nodes.npz`, `g_edges.npz`, `g_segments.npz`, `action_scale.json`, `g_lang.npz` |
| `cache-policy` | run the configured policy over the corpus | `a_pi_raw.npz` |
| `train` | optimize the learned components | `g_head.npz`, `g_score.npz`, `g_invdyn.npz`, `g_fcsr.npz` |
| `eval` | gate the artifact tree, launch the policy, run a benchmark | result directory + `run.json` provenance |
| `score` | aggregate results already on disk | tables and deltas |

## Layout

```
robot-ood/
├── stablevla/hf_weights/                  policy weights
├── external/LIBERO-plus/                  benchmark fork
├── data/                                  demo HDF5, fwm fields
├── outputs/<suite>/artifacts/             every artifact the method loads
├── logs/                                  eval logs
└── results/                               per-episode records + run.json
```

`ROBOT_OOD_DATA`, `ROBOT_OOD_OUTPUTS`, `ROBOT_OOD_RESULTS` override the three roots
([configuration](configuration.md)). [SETUP.md](SETUP.md) installs environments, benchmark,
weights and demonstrations.

## Get an artifact tree

Fetch the trained tree (no GPU, no demos needed) — the download is gated on arrival:

```bash
python -m robot_ood.cli artifacts fetch long --suite long
```

Or build it yourself (one GPU, about a GPU-day):

```bash
python -m robot_ood.cli build --suite long --device cuda:0     # graph, segments, calib, lang, cache
python -m robot_ood.cli train --suite long --device cuda:0     # locator, entry, fcsr, deploy gate
python -m robot_ood.cli build --suite long --dry-run           # what is stale, and why
```

Bare `build` and `train` run every stale step of their group in order; `robot_ood.pipeline.dag`
declares the steps, `robot_ood.pipeline.manifest` hashes each step's inputs and re-runs only what
changed. Long steps detach; follow them in `outputs/<suite>/pipeline_logs/`.

Single steps, in build order:

```bash
python -m robot_ood.cli build graph --suite long               # g_nodes, g_edges; seeds outputs/long/artifacts
python -m robot_ood.cli build segments --suite long            # g_segments.npz
python -m robot_ood.cli build calibrate --suite long           # action_scale.json
python -m robot_ood.cli build language --suite long            # g_lang.npz
python -m robot_ood.cli cache-policy --suite long --port 14190 # a_pi_raw.npz (policy env + GPU)
python -m robot_ood.cli train locator --suite long             # g_head.npz
python -m robot_ood.cli train entry --suite long               # g_score.npz + g_invdyn.npz
python -m robot_ood.cli train fcsr --suite long                # g_fcsr.npz
```

`cache-policy` starts the policy server named by the tree's policy descriptor
(`evals/policies/<id>.yaml`, [configuration](configuration.md)), caches the corpus against it, and
tears it down. `--policy` overrides the descriptor; `ROBOT_OOD_PY` overrides the interpreter.

Two measurements ride inside checkpoints instead of being steps: `train locator` embeds its
phase-skew replay report in `g_head.npz`, and `train fcsr` embeds its C0 calibration in
`g_fcsr.npz`. Each locator run also snapshots `run_config.json` beside the checkpoint.

## Check

```bash
python -m robot_ood.cli artifacts check outputs/long/artifacts --stage deploy
python -m pytest tests/ -q
```

`--stage` takes `build | locator | entry | fcsr | deploy`; `deploy` is the union of the other
four ([artifacts](artifacts.md)). `tests/test_parity.py` replays recorded retrievals against the
artifacts on disk and asserts bit-exact floats.

## Eval

```bash
python -m robot_ood.cli eval --suite long --mode fcsr --check-env               # preflight
python -m robot_ood.cli eval --suite long --mode fcsr --rung R1 --gpus 0 --dry-run
python -m robot_ood.cli eval --suite long --mode fcsr --rung R4 --gpus 0,1 --axis Objects_Layout
```

`--mode` names the arm (`base` = frozen policy, `fcsr` = the method); `--rung` names the scope;
the two are independent. `eval` resolves the artifact tree from `--artifacts`, else
`$GR_GRAPH_DIR`, else the suite's `configs/trees/<suite>.yaml`; an `fcsr` run must pass the
deploy gate before anything launches. The tree's `run_config.json` names the policy descriptor to
launch.

Rungs live in `configs/sr_ladder.yaml`:

| rung | scope | cells | episodes |
|---|---|---|---|
| `R1` | smoke | 4 | 47 |
| `L4` | `long` × four reported axes | 4 | 1507 |
| `L3` | `long` × the other three axes | 3 | 1011 |
| `G3` | `object`, `spatial`, `goal` × seven axes | 21 | 7508 |
| `R4` | four suites × seven axes | 28 | 10026 |

`scripts/run_sr.py` is the backend and takes cells directly:

```bash
scripts/run_sr.py --rung R4 --gpus 0,1 --mode fcsr --cells long:Robot_Initial_States
```

`--resume` skips complete cells and refuses partial ones; `--rerun-incomplete` purges and reruns
partial cells; `--score` reports without launching; `--results-root` sends a rerun to its own tree
(rerunning a cell in place is refused).
Exit 2: clean run, gate not met. Exit 1: operational failure.

Result and log names share one grammar:

```
results/<prefix>_<task_suite>/<axis>/<policy>_<mode>_<suite>_<axis>[_variant]/   records + run.json
logs/<utc>_<policy>_<mode>_<suite>_<axis>/                                       server.log + client_shard*.log
```

Every result directory carries `run.json`: policy, mode, suite, axis, expected n, graph dir,
`graph_hash`, `artifact_digest`, `run_config_sha`, git commit, and the full child environment.

## Score

```bash
python evals/libero_plus/score.py --suites long --methods base:stablevla_base fcsr:stablevla_fcsr
```

Success rate = successes / records over the `episodes.jsonl` each client appends, recursive per
result directory (a directory older than those records is counted by its `*_success.mp4` replays). `--methods` takes
`name:tag_prefix` pairs; a prefix that matches several directories sums them, and the report
prints each method's `artifact_digest` set — a column mixing trees is flagged `MIXED TREES`.
`--bench <id>` reads another benchmark's tree.

## Publish

```bash
export HF_TOKEN=<write token>
python -m robot_ood.cli artifacts push long --suite long       # gates first, uploads on pass
```

The bucket subdirectory is the suite name; `artifacts fetch <subdir> --suite <s>` is the inverse.
