# Environments

Install commands are in [`../docs/SETUP.md`](../docs/SETUP.md) and
[`../evals/gr00t/README.md`](../evals/gr00t/README.md). This file is why there are four of them
and not one.

| env | file | holds |
|---|---|---|
| `onf-core` | `envs/onf-core.yml` | the `robot_ood` package: graph, locator, corrector, driver. CPU-only torch is enough. |
| `stablevla` | `envs/stablevla.yml` | StableVLA policy server **and** the LIBERO-Plus sim client. torch 2.2.0+cu121, an OpenVLA-OFT transformers fork, LIBERO/robosuite/mujoco. |
| gr00t | Isaac-GR00T's own `.venv` | the GR00T-N1.7 server. Reached over the websocket transport in `evals/gr00t/ws_server.py`. |
| openpi | openpi's own `.venv` (`evals/pi05/setup.sh`) | the pi0.5 server, `evals/pi05/ws_server.py`: JAX 0.5.3 CUDA, Python 3.11. |

They cannot be merged. `onf-core` is light and pins nothing heavy; the others pin conflicting torch
builds, transformers forks and a JAX stack. Keeping them apart is what lets the `robot_ood` package's tests
run on a CPU box with no policy stack installed at all.

The sim client runs in the **same** env as the StableVLA policy — LIBERO-Plus is the benchmark run
*with* that policy, not a fourth runtime. `envs/stablevla.yml` is a portable rebuild frozen from a
working env; recreate with `conda env create -f envs/stablevla.yml` if the upstream README's
`flash-attn` step fights you.

## Headless rendering

LIBERO / robosuite render offscreen. The GL backend must be set **before** the sim is imported — the
eval drivers do this themselves, so this only matters if you invoke the harness by hand:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

Software OSMesa. Use `egl` instead if a GPU EGL context is available.

## How the eval reaches GR00T

The eval driver reads `evals/policies/gr00t.yaml` and starts `evals/gr00t/ws_server.py` under the
checkout's `.venv` python (`resolve_policy_python`); the sim-client pool runs under the `stablevla`
conda env. The client speaks the same websocket protocol to both servers, so the policy is
swappable without touching the method.

The `websockets` / `msgpack_numpy` / `image_tools` transport is **vendored** at
`evals/libero_plus/harness/tools/` and runs client-side, so the GR00T env stays untouched apart
from one `websockets h5py` install. Artifacts are policy-specific: a gr00t tree
(`configs/trees/long_gr00t.yaml`) builds and trains against GR00T's own cached actions.

Do **not** add GR00T dependencies to `onf-core` or `stablevla`.
