# Train — ALOHA right arm (GR00T multitask)

```bash
source ./env.sh
```

Suite `aloha_right`, policy `gr00t_aloha`, checkpoint `gr00t/hf_weights/aloha_right_multitask_local`.

## Corpus

```bash
python -m robot_ood.data.lerobot_aloha --suite aloha_right
```

## Build

```bash
python -m robot_ood.cli build graph     --suite aloha_right --device cuda:0
python -m robot_ood.cli build segments  --suite aloha_right --force
python -m robot_ood.cli build calibrate --suite aloha_right --device cuda:0 --min-r2 0.70 --force
python -m robot_ood.cli build language  --suite aloha_right --device cuda:0 --force
python -m robot_ood.cli cache-policy    --suite aloha_right --port 14190 --gpu 0
```

## Train

```bash
python -m robot_ood.cli train locator --suite aloha_right --device cuda:0 --epochs 30 --force
python -m robot_ood.cli train entry   --suite aloha_right --device cuda:0 --epochs 6000 --force
python -m robot_ood.cli train fcsr    --suite aloha_right --device cuda:0 --epochs 30 --force
python -m robot_ood.cli artifacts check outputs/aloha_right/artifacts --stage deploy
```

## Retrain the corrector only

```bash
python -m robot_ood.cli train fcsr --suite aloha_right --device cuda:0 --epochs 30 --force
python -m robot_ood.cli artifacts check outputs/aloha_right/artifacts --stage deploy
```

## Deploy

```bash
cd deploy/aloha_right
python check.py
python serve.py
```

`deploy/aloha_right/README.md` holds the robot-machine side.
