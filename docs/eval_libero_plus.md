# Eval — LIBERO-Plus, 1 trial

```bash
source ./env.sh
python -m robot_ood.cli bench setup libero_plus
```

Axes: `Robot_Initial_States` `Objects_Layout` `Camera_Viewpoints` `Sensor_Noise`
`Background_Textures` `Light_Conditions` `Language_Instructions`.
Suites: `long` `object` `spatial` `goal`. Trees: `<suite>` (stablevla), `<suite>_gr00t`, `<suite>_pi05`.

## Preflight

```bash
python -m robot_ood.cli eval --bench libero_plus --suite long --mode base --check-env
```

## base, one cell

```bash
python -m robot_ood.cli eval --bench libero_plus --suite long --axis Robot_Initial_States \
    --mode base --policy stablevla --trials 1 --rung R4 --gpus 0,1,2,3

python -m robot_ood.cli eval --bench libero_plus --suite long --axis Robot_Initial_States \
    --mode base --policy gr00t --trials 1 --rung R4 --gpus 0,1,2,3

python -m robot_ood.cli eval --bench libero_plus --suite long --axis Robot_Initial_States \
    --mode base --policy pi05 --trials 1 --rung R4 --gpus 0,1,2,3
```

## fcsr, one cell

```bash
python -m robot_ood.cli eval --bench libero_plus --suite long --axis Robot_Initial_States \
    --mode fcsr --rung R4 --gpus 0,1,2,3 --artifacts outputs/long/artifacts

python -m robot_ood.cli eval --bench libero_plus --suite goal --axis Language_Instructions \
    --mode fcsr --rung R4 --gpus 0,1,2,3 --artifacts outputs/goal_gr00t/artifacts

python -m robot_ood.cli eval --bench libero_plus --suite long --axis Camera_Viewpoints \
    --mode fcsr --rung R4 --gpus 0,1,2,3 --artifacts outputs/long_pi05/artifacts
```

## Seven axes, one tree

```bash
for axis in Robot_Initial_States Objects_Layout Camera_Viewpoints Sensor_Noise \
            Background_Textures Light_Conditions Language_Instructions; do
  python -m robot_ood.cli eval --bench libero_plus --suite long --axis "$axis" \
      --mode fcsr --rung R4 --gpus 0,1,2,3 --artifacts outputs/long/artifacts --resume
done
```

## Full grid

```bash
python -m robot_ood.cli eval --bench libero_plus --suite long --mode base --policy stablevla \
    --rung R4 --gpus 0,1,2,3 --resume
```

| rung | scope |
|---|---|
| `R1` | 4-cell smoke |
| `L4` | `long` x 4 reported axes |
| `L3` | `long` x the other 3 axes |
| `G3` | `object` `spatial` `goal` x 7 axes |
| `R4` | 4 suites x 7 axes |

## Score

```bash
python evals/libero_plus/score.py --suites long --methods base:stablevla_base fcsr:stablevla_fcsr
python evals/libero_plus/score.py --suites goal --methods base:gr00t_base fcsr:gr00t_fcsr
python evals/libero_plus/score.py --suites long --methods base:pi05_base fcsr:pi05_fcsr
```

## Smoke

```bash
python -m robot_ood.cli eval --bench libero_plus --suite long --axis Robot_Initial_States \
    --mode fcsr --cell --gpus 0 --artifacts outputs/long/artifacts
```
