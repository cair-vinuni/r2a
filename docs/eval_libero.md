# Eval — LIBERO, 50 trials

```bash
source ./env.sh
python -m robot_ood.cli bench setup libero
```

500 episodes per cell (10 tasks x 50 trials). Suites: `long` `object` `spatial` `goal`.

## Preflight

```bash
python -m robot_ood.cli eval --bench libero --suite long --mode base --check-env
```

## base

```bash
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy stablevla \
    --trials 50 --gpus 0,1,2,3 --servers-per-cell 8 --num-clients 10

python -m robot_ood.cli eval --bench libero --suite long --mode base --policy gr00t \
    --trials 50 --gpus 0,1,2,3 --servers-per-cell 8 --num-clients 10

python -m robot_ood.cli eval --bench libero --suite long --mode base --policy pi05 \
    --trials 50 --gpus 0,1,2,3 --servers-per-cell 4 --num-clients 8
```

## fcsr

```bash
python -m robot_ood.cli eval --bench libero --suite long --mode fcsr --policy stablevla \
    --trials 50 --gpus 0,1,2,3 --artifacts outputs/long/artifacts

python -m robot_ood.cli eval --bench libero --suite long --mode fcsr --policy gr00t \
    --trials 50 --gpus 0,1,2,3 --artifacts outputs/long_gr00t/artifacts

python -m robot_ood.cli eval --bench libero --suite long --mode fcsr --policy pi05 \
    --trials 50 --gpus 0,1,2,3 --artifacts outputs/long_pi05/artifacts
```

## Four suites, one policy

```bash
for suite in long object spatial goal; do
  python -m robot_ood.cli eval --bench libero --suite "$suite" --mode base --policy gr00t \
      --trials 50 --gpus 0,1,2,3 --resume
done
```

## Score

```bash
python evals/libero_plus/score.py --bench libero --suites long object spatial goal \
    --methods base:stablevla_base fcsr:stablevla_fcsr
python evals/libero_plus/score.py --bench libero --suites long --methods base:gr00t_base fcsr:gr00t_fcsr
python evals/libero_plus/score.py --bench libero --suites long --methods base:pi05_base fcsr:pi05_fcsr
```

## Smoke

```bash
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy gr00t \
    --trials 1 --variant smoke --gpus 0
```
