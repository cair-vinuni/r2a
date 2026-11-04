# Train — RoboTwin 2.0

```bash
source ./env.sh
```

## Suites

| policy | lane trees | bimanual cell |
|---|---|---|
| pi05_robotwin | `g1_left` `g1_right` … `g5_left` `g5_right` | `g1` … `g5` |
| vlact | `g1_left_vlact` `g1_right_vlact` … `g5_left_vlact` `g5_right_vlact` | `g1_vlact` … `g5_vlact` |

## Corpus

The fifty clean demonstrations of each task, `dataset/<task>/aloha-agilex_clean_50.zip` in the
`TianxingChen/RoboTwin2.0` dataset, unpacked under `data/robotwin/raw/<task>/`:

```bash
python - <<'EOF'
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download
raw = Path("data/robotwin/raw")
for line in Path("evals/robotwin/suites/all.txt").read_text().splitlines():
    task = line.strip()
    if not task or task.startswith("#") or (raw / task).exists():
        continue
    z = hf_hub_download("TianxingChen/RoboTwin2.0", f"dataset/{task}/aloha-agilex_clean_50.zip",
                        repo_type="dataset", local_dir="data/robotwin/zips")
    zipfile.ZipFile(z).extractall(raw / task)
EOF
```

One lane corpus per arm:

```bash
for g in g1 g2 g3 g4 g5; do
  python -m robot_ood.data.robotwin --suite "${g}_left"
  python -m robot_ood.data.robotwin --suite "${g}_right"
done
```

## pi0.5 lanes

```bash
python -m robot_ood.cli build --suite g5_left  --device cuda:0
python -m robot_ood.cli train --suite g5_left  --device cuda:0
python -m robot_ood.cli build --suite g5_right --device cuda:1
python -m robot_ood.cli train --suite g5_right --device cuda:1
python -m robot_ood.cli artifacts check outputs/g5_left/artifacts  --stage deploy
python -m robot_ood.cli artifacts check outputs/g5_right/artifacts --stage deploy
```

## VLAct lanes

```bash
python -m robot_ood.cli cache-policy --suite g5_left_vlact  --port 14490 --gpu 0
python -m robot_ood.cli build --suite g5_left_vlact  --device cuda:0
python -m robot_ood.cli train --suite g5_left_vlact  --device cuda:0
python -m robot_ood.cli cache-policy --suite g5_right_vlact --port 14491 --gpu 1
python -m robot_ood.cli build --suite g5_right_vlact --device cuda:1
python -m robot_ood.cli train --suite g5_right_vlact --device cuda:1
```

## Single steps

```bash
python -m robot_ood.cli build graph     --suite g5_left
python -m robot_ood.cli build segments  --suite g5_left
python -m robot_ood.cli build calibrate --suite g5_left
python -m robot_ood.cli build language  --suite g5_left --device cuda:0
python -m robot_ood.cli cache-policy    --suite g5_left --port 14190 --gpu 0
python -m robot_ood.cli train locator   --suite g5_left --device cuda:0 --epochs 30
python -m robot_ood.cli train entry     --suite g5_left --device cuda:0 --epochs 6000
python -m robot_ood.cli train fcsr      --suite g5_left --device cuda:0 --epochs 30
```

## Locator over four GPUs

```bash
torchrun --nproc_per_node=4 scripts/research/train_locator_mp.py --suite g5_left --epochs 30
torchrun --nproc_per_node=4 scripts/research/train_fcsr_mp.py    --suite g5_left --epochs 30
```

## Retrain the corrector only

```bash
i=0
for g in g1 g2 g3 g4 g5; do
  for arm in left right; do
    setsid nohup python -m robot_ood.cli train fcsr --suite "${g}_${arm}" \
        --device "cuda:$((i % 8))" --epochs 30 --force \
        >> "logs/${g}_${arm}_fcsr.log" 2>&1 < /dev/null & disown
    i=$((i + 1))
  done
done
```

```bash
i=0
for g in g1 g2 g3 g4 g5; do
  for arm in left right; do
    setsid nohup python -m robot_ood.cli train fcsr --suite "${g}_${arm}_vlact" \
        --device "cuda:$((i % 8))" --epochs 30 --force \
        >> "logs/${g}_${arm}_vlact_fcsr.log" 2>&1 < /dev/null & disown
    i=$((i + 1))
  done
done
```

```bash
for g in g1 g2 g3 g4 g5; do
  for arm in left right; do
    python -m robot_ood.cli artifacts check "outputs/${g}_${arm}/artifacts" --stage deploy
    python -m robot_ood.cli artifacts check "outputs/${g}_${arm}_vlact/artifacts" --stage deploy
  done
done
```
