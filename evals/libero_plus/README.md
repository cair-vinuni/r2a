# LIBERO-Plus

**Checkout:** `external/LIBERO-plus` (pinned fork; relocate with `$LIBERO_HOME`)
**Code:** `evals/libero_plus/bench.yaml` (registration), `evals/libero_plus/harness/eval_libero.py`
(rollout client), `evals/libero_plus/harness/libero_plus_bridge.py` (suite filter),
`evals/libero_plus/score.py` (scorer), `evals/libero_plus/setup.sh` (what `bench setup` runs)

The default benchmark: 2518 perturbed variants of the four LIBERO task suites, grouped into seven
perturbation axes. The checkout is never modified — `libero_plus_bridge.py` builds the upstream
suite and filters its tasks by axis from outside, using the fork's
`task_classification.json`.

## Install

```bash
python -m robot_ood.cli bench setup libero_plus     # ~10 min, 6.4 GB; idempotent
```

Check:

```bash
python -m robot_ood.cli eval --suite long --mode base --check-env
```

## Axes

| `--axis` | label |
|---|---|
| `Background_Textures` | Background |
| `Camera_Viewpoints` | Camera |
| `Language_Instructions` | Language |
| `Light_Conditions` | Light |
| `Objects_Layout` | Obj-Layout |
| `Robot_Initial_States` | Robot-Init |
| `Sensor_Noise` | Sensor |

Suites map `long/object/spatial/goal` → `libero_10/libero_object/libero_spatial/libero_goal`;
episode counts per cell live in `configs/sr_ladder.yaml`.

## Score

```bash
python evals/libero_plus/score.py --suites long --methods base:stablevla_base fcsr:stablevla_fcsr
```
