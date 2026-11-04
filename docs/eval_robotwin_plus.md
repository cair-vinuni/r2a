# Eval: RoboTwin 2.0-Plus

```bash
source ./env.sh
python -m robot_ood.cli bench setup robotwin
bash evals/robotwin_plus/setup.sh
nvidia-cuda-mps-control -d
```

A group cell is ten tasks at ten trials per task, one hundred episodes. The `plus50` cell is the
fifty tasks at one trial each, the benchmark's own protocol.

## Deployed rollout settings

The corrector on RoboTwin runs with these settings, exported once in the shell that launches the
`fcsr` cells. The base cells run without them.

```bash
export ROBOT_OOD_EXEC_HORIZON=50   # execute the whole fifty-row chunk before the next query
export ROBOT_OOD_PAIR=1            # one retrieval over both arms' lanes
export ROBOT_OOD_STALL_PUSH=3      # push along the retrieved demo after three still checks
export ROBOT_OOD_STALL_DWELL=1     # the still time that counts as a stall follows the demos
export ROBOT_OOD_GRASP_SERVO=80    # requery every ten rows for eighty steps after a gripper closes
```

Axes: `Original` `Camera` `Robot` `Language` `Light` `Background` `Noise` `Layout`.
Suites: `plus50` (50 tasks, pi0.5), `g1`…`g5` (10 tasks, pi0.5), `g1_vlact`…`g5_vlact` (10 tasks, VLAct).

## Preflight

```bash
python -m robot_ood.cli eval --bench robotwin_plus --suite plus50 --axis Original \
    --mode base --policy pi05_robotwin --check-env
```

## pi0.5 base, the fifty tasks

```bash
python -m robot_ood.cli eval --bench robotwin_plus --suite plus50 --axis Original \
    --mode base --policy pi05_robotwin --trials 1 --gpus 0,1,2,3

for axis in Original Camera Robot Language Light Background Noise Layout; do
  python -m robot_ood.cli eval --bench robotwin_plus --suite plus50 --axis "$axis" \
      --mode base --policy pi05_robotwin --trials 1 --gpus 0,1,2,3 --resume
done
```

## pi0.5 fcsr, the five groups

```bash
for g in g1 g2 g3 g4 g5; do
  for axis in Original Camera Robot Language Light Background Noise Layout; do
    python -m robot_ood.cli eval --bench robotwin_plus --suite "$g" --axis "$axis" \
        --mode fcsr --policy pi05_robotwin --trials 10 --gpus 0,1,2,3 --resume \
        --results-root results_pi05_robotwin_fcsr
  done
done
```

## VLAct base

```bash
for g in g1 g2 g3 g4 g5; do
  for axis in Original Camera Robot Language Light Background Noise Layout; do
    python -m robot_ood.cli eval --bench robotwin_plus --suite "$g" --axis "$axis" \
        --mode base --policy vlact --trials 10 --gpus 0,1,2,3 --resume \
        --results-root results_vlact_base
  done
done
```

## VLAct fcsr

```bash
for g in g1 g2 g3 g4 g5; do
  for axis in Original Camera Robot Language Light Background Noise Layout; do
    python -m robot_ood.cli eval --bench robotwin_plus --suite "${g}_vlact" --axis "$axis" \
        --mode fcsr --policy vlact --trials 10 --gpus 0,1,2,3 --resume \
        --results-root results_vlact_fcsr
  done
done
```

## Driver form

```bash
python scripts/run_sr.py --bench robotwin_plus \
    --cells g5:Original,g5:Camera,g5:Robot,g5:Language,g5:Light,g5:Background,g5:Noise,g5:Layout \
    --policy pi05_robotwin --mode fcsr --gpus 0,1,2,3 --trials 10 --resume \
    --results-root results_pi05_robotwin_fcsr
```

## RoboTwin 2.0

```bash
python -m robot_ood.cli eval --bench robotwin --suite g5 --axis Randomized \
    --mode fcsr --policy pi05_robotwin --trials 50 --gpus 0,1,2,3
python -m robot_ood.cli eval --bench robotwin --suite g5 --axis Clean \
    --mode base --policy pi05_robotwin --trials 50 --gpus 0,1,2,3
```

## Score

```bash
python evals/libero_plus/score.py --bench robotwin_plus --suites g1 g2 g3 g4 g5 \
    --methods base:pi05_robotwin_base fcsr:pi05_robotwin_fcsr
python evals/libero_plus/score.py --bench robotwin_plus --suites g1 g2 g3 g4 g5 \
    --methods base:vlact_base fcsr:vlact_fcsr
python evals/libero_plus/score.py --bench robotwin_plus --suites plus50 --methods base:pi05_robotwin_base
```

## Smoke

```bash
python -m robot_ood.cli eval --bench robotwin_plus --suite plus_smoke --axis Original \
    --mode base --policy pi05_robotwin --trials 1 --gpus 0
```
