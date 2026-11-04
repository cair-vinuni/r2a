# LIBERO

The upstream benchmark, [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
as published: four suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`), ten
tasks each, fifty stored initial states per task. `--suite long` is `libero_10`; `object`,
`spatial`, `goal` select the others. Every policy runs under its authors' rollout
([docs/PROTOCOLS.md](../../docs/PROTOCOLS.md)).

## Install

```bash
./install_stablevla.sh                            # simulator environment
source ./env.sh
python -m robot_ood.cli bench setup libero        # external/LIBERO at the pinned commit
scripts/setup_nvidia_gl.sh                        # MuJoCo on the GPU (optional, 20x faster)
```

One policy, its checkout and weights:

| policy | install | weights land in |
|---|---|---|
| `gr00t` | [evals/gr00t/README.md](../gr00t/README.md) | `gr00t/hf_weights/` |
| `pi05` | `evals/pi05/setup.sh` ([evals/pi05/README.md](../pi05/README.md)) | `pi05/` |
| `stablevla` | [evals/stablevla/README.md](../stablevla/README.md) | `stablevla/hf_weights/` |

## Run

```bash
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy gr00t \
    --gpus 0,1,2,3 --servers-per-cell 8 --num-clients 10
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy pi05 \
    --gpus 0,1,2,3 --servers-per-cell 4 --num-clients 8
python -m robot_ood.cli eval --bench libero --suite long --mode base --policy stablevla \
    --gpus 0,1,2,3 --servers-per-cell 8 --num-clients 10
```

500 episodes each (10 tasks x 50 initial states). `--trials 1 --variant smoke` runs 10 episodes
first. `--mode fcsr` runs the method over the same policy and protocol; it reads the suite's
trained tree ([docs/SETUP.md](../../docs/SETUP.md), Artifacts).

## Output

The driver prints one `RESULT` line, then one line per task and the two directories:

```
RESULT gr00t_base_long_Original: 448/500 = 89.6%  [PASS]
  task  0   47/50   94.0%  put both the alarm clock and the book on the desk
  ...
  total    448/500  89.6%
  results results/libero_libero_10/Original/gr00t_base_long_Original
  logs    logs/20260903_120102_gr00t_base_long_Original
```

| path | holds |
|---|---|
| `results/libero_<task_suite>/Original/<policy>_<mode>_<suite>_Original[_<variant>]/` | one cell |
| `.../summary.json` | policy, protocol, placement, git commit, successes / n / rate, episodes per minute, per-task rows |
| `.../run.json` | the launch recipe: full child environment, EGL device, artifact digest, status |
| `.../shard<i>_of_<n>/episodes.jsonl` | one record per episode (task_id, episode, success, steps, wall_s); the score is the count of `success: true` |
| `.../shard<i>_of_<n>/rollout_<task>_t<id>_episode<k>_{success,failure}.mp4` | replay videos, only with `--video failures` or `--video all` (default: none; older result dirs without records are scored by their mp4s) |
| `logs/<utc>_<tag>/server*.log`, `client_shard<i>.log` | the processes; each client logs one EPRESULT line per episode |

A finished cell is never overwritten: rerun with another `--variant` or `--results-root`.
Reference numbers: [docs/REFERENCE.md](../../docs/REFERENCE.md).

## Throughput

One cell takes every GPU it is given; the policy descriptor turns them into servers and clients
(`placement`), episodes are dealt round robin over the clients, and the GR00T and pi0.5 servers
answer their clients from one batched forward. `--servers-per-cell` and `--num-clients` override
the descriptor. On 4x A100 / 32 cores, `libero_10`, 500 episodes, whole cell including server
start:

| policy | servers x clients | episodes/min | steady state |
|---|---|---|---|
| `gr00t` | 8 x 32 | 64 | 104 |
| `pi05` | 4 x 32 | 58 | 83 |
| `stablevla` | 16 x 32 | 71 | 121 |
| `stablevla`, fcsr | 16 x 32 | 65 | 107 |

Steady state is the clients' aggregate simulator rate divided by the mean episode length, i.e.
the rate a long benchmark runs at once the servers are up. The simulator itself tops out near
1200 environment steps/s on this box (24 clients), so a client spends most of its time waiting
on the policy; more cores mean more clients per server. pi0.5 compiles each batch size once per
machine (`pi05/jax_cache`), so its first launch is a few minutes slower.

## Layout

| | |
|---|---|
| checkout | `external/LIBERO` @ `8f1084e` |
| config the package reads | `external/LIBERO/libero/config.yaml`, written by `evals/libero/setup.sh` |
| client | `evals/libero/harness/eval_libero.py` |
| protocols | `src/robot_ood/core/protocol.py` |
