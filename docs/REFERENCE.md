# Reference numbers

Base policies under their own protocol ([PROTOCOLS.md](PROTOCOLS.md)), on this repository's
harness, on one machine (4x A100 80GB, 32 cores, driver 610.43.02, MuJoCo rendering on the
GPU). Each row is one `summary.json`. Published numbers are the authors'. Sampling noise at 500
episodes is about +-1.4 points (one standard deviation), and `nvidia` draws a random initial
state per episode.

The method's own numbers are not here: [results.md](results.md) holds every `fcsr` cell.

## LIBERO (`--bench libero`)

| policy | suite | protocol | episodes | this harness | minutes | vendor stack, same machine | published |
|---|---|---|---|---|---|---|---|
| `gr00t` | `libero_10` | `nvidia` | 500 | 440/500 = 88.0% | 7.8 | 455/500 = 91.0% (`rollout_policy.py`) | 94.35% (200 episodes) |
| `pi05` | `libero_10` | `openpi` | 500 | 467/500 = 93.4% | 8.6 | | 92.4% |
| `stablevla` | `libero_10` | `openvla_oft` | 500 | 468/500 = 93.6% | 7.0 | | 95.0% |

Minutes are the whole cell: server start, every episode, scoring. The placement each policy's
descriptor declares was used (`evals/policies/<id>.yaml`); nothing else was passed.

```bash
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy gr00t     --gpus 0,1,2,3
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy pi05      --gpus 0,1,2,3
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy stablevla --gpus 0,1,2,3
```
