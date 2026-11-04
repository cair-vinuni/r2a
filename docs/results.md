# Results — `fcsr` against the frozen policy

StableVLA frozen, two suites: `long` (LIBERO task suite `libero_10`) and `goal` (`libero_goal`).
`base` is the frozen policy alone; `fcsr` is the shipping arm — the entry corrector at t = 0, then
the locator, segment retrieval and the corrector at every check — reading `outputs/<suite>/artifacts/`.

A result directory's success rate is `#success / #records` over the `episodes.jsonl` its clients
wrote (`evals/libero_plus/score.py`).

## LIBERO-Plus, suite `long`

| axis | base | `fcsr` | Δ |
|---|---|---|---|
| Robot_Initial_States | 170/393 = 43.3% | **391/393 = 99.5%** | +56.2 |
| Sensor_Noise | 343/449 = 76.4% | **407/449 = 90.6%** | +14.3 |
| Language_Instructions | 303/383 = 79.1% | **324/383 = 84.6%** | +5.5 |
| Background_Textures | 243/289 = 84.1% | **265/289 = 91.7%** | +7.6 |
| Light_Conditions | 246/273 = 90.1% | **267/273 = 97.8%** | +7.7 |
| Objects_Layout | 253/312 = 81.1% | **277/312 = 88.8%** | +7.7 |
| Camera_Viewpoints | 243/419 = 58.0% | **313/419 = 74.7%** | +16.7 |
| **TOTAL** | **1801/2518 = 71.5%** | **2244/2518 = 89.1%** | **+17.6** |

## LIBERO-Plus, suite `goal`

| axis | base | `fcsr` | Δ |
|---|---|---|---|
| Robot_Initial_States | 210/409 = 51.3% | **406/409 = 99.3%** | +47.9 |
| Sensor_Noise | 317/379 = 83.6% | **345/379 = 91.0%** | +7.4 |
| Language_Instructions | 233/410 = 56.8% | **309/410 = 75.4%** | +18.5 |
| Background_Textures | 242/281 = 86.1% | **277/281 = 98.6%** | +12.5 |
| Light_Conditions | 277/278 = 99.6% | 269/278 = 96.8% | −2.9 |
| Objects_Layout | 276/425 = 64.9% | 273/425 = 64.2% | −0.7 |
| Camera_Viewpoints | 328/408 = 80.4% | **349/408 = 85.5%** | +5.1 |
| **TOTAL** | **1883/2590 = 72.7%** | **2228/2590 = 86.0%** | **+13.3** |

Two goal axes are negative. Light_Conditions starts at 99.6% — one episode from the ceiling, so
any correction can only lose there. Objects_Layout is flat: half its episodes move the target
object, and a demonstrated continuation of a task whose object has moved is the wrong continuation
no matter how well it is retrieved.

## Clean LIBERO

No perturbation, the suite as it ships. Repeated runs measure the harness's own spread.

| suite | base | `fcsr` |
|---|---|---|
| long | 464/500, 468/500 | 477/500, 478/500, 480/500 (mean 95.7%) |
| goal | 482/500 = 96.4% | 488/500 = 97.6% |

## Reproducing the tables

Both columns of both suites sit in this checkout's `results/`, one directory per cell:

```bash
python evals/libero_plus/score.py --suites long --methods base:stablevla_base fcsr:stablevla_fcsr
python evals/libero_plus/score.py --suites goal --methods base:stablevla_base fcsr:stablevla_fcsr
```

One suite per call: each suite has its own artifact tree, and scoring both at once reports the two
digests as `MIXED TREES`. Per suite the `fcsr` column is one digest — `2c652ff219f6` for long,
`a4693a483019` for goal. The repeat runs of clean long are held out of the default root, under
`results/repeats/`, so every scored cell has exactly one directory.

Re-measuring a cell:

```bash
scripts/research/eval_fcsr.sh long fcsr 0,1,2,3
```
