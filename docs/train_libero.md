# Train — LIBERO

```bash
source ./env.sh
```

## Suites

| policy | suites |
|---|---|
| stablevla | `long` `object` `spatial` `goal` |
| gr00t | `long_gr00t` `object_gr00t` `spatial_gr00t` `goal_gr00t` |
| pi05 | `long_pi05` `object_pi05` `spatial_pi05` `goal_pi05` |

## Demonstrations

```bash
python -c "
from robot_ood.core.paths import default_paths
from libero.libero.utils.download_utils import download_from_huggingface
dest = default_paths().data / 'libero_datasets'
for name in ('libero_10', 'libero_goal', 'libero_object', 'libero_spatial'):
    download_from_huggingface(name, str(dest), check_overwrite=False)"
```

## Full build + train, one suite

```bash
python -m robot_ood.cli build --suite long --device cuda:0
python -m robot_ood.cli train --suite long --device cuda:0
python -m robot_ood.cli artifacts check outputs/long/artifacts --stage deploy
```

## gr00t / pi05 trees: cache the policy first

```bash
python -m robot_ood.cli cache-policy --suite long_gr00t --port 14200 --gpu 0
python -m robot_ood.cli build --suite long_gr00t --device cuda:0
python -m robot_ood.cli train --suite long_gr00t --device cuda:0

python -m robot_ood.cli cache-policy --suite long_pi05 --port 14300 --gpu 0
python -m robot_ood.cli build --suite long_pi05 --device cuda:0
python -m robot_ood.cli train --suite long_pi05 --device cuda:0
```

## Single steps

```bash
python -m robot_ood.cli build graph     --suite long
python -m robot_ood.cli build segments  --suite long
python -m robot_ood.cli build calibrate --suite long
python -m robot_ood.cli build language  --suite long --device cuda:0
python -m robot_ood.cli cache-policy    --suite long --port 14190 --gpu 0
python -m robot_ood.cli train locator   --suite long --device cuda:0 --epochs 30
python -m robot_ood.cli train entry     --suite long --device cuda:0 --epochs 6000
python -m robot_ood.cli train fcsr      --suite long --device cuda:0 --epochs 30
```

## Retrain the corrector only

```bash
python -m robot_ood.cli train fcsr --suite long --device cuda:0 --epochs 30 --force
python -m robot_ood.cli artifacts check outputs/long/artifacts --stage deploy
```

## All twelve trees, eight GPUs

```bash
i=0
for suite in long object spatial goal \
             long_gr00t object_gr00t spatial_gr00t goal_gr00t \
             long_pi05 object_pi05 spatial_pi05 goal_pi05; do
  setsid nohup python -m robot_ood.cli train fcsr --suite "$suite" --device "cuda:$((i % 8))" \
      --epochs 30 --force >> "logs/${suite}_fcsr.log" 2>&1 < /dev/null & disown
  i=$((i + 1))
done
```

```bash
for suite in long object spatial goal \
             long_gr00t object_gr00t spatial_gr00t goal_gr00t \
             long_pi05 object_pi05 spatial_pi05 goal_pi05; do
  python -m robot_ood.cli artifacts check "outputs/$suite/artifacts" --stage deploy
done
```

## Status

```bash
python -m robot_ood.cli build --suite long --dry-run
python -m robot_ood.cli train --suite long --dry-run
```
