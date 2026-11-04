# Upstream provenance

Everything under `src/robot_ood/` and `evals/` is original code for this project, except the two
vendored files below. The policy and benchmark trees are fetched into `external/` and never committed.

## Vendored in-repo (tracked in git)

| Path | Role | Source | Commit | License |
|---|---|---|---|---|
| `evals/libero_plus/harness/` | LIBERO-Plus eval client: `eval_libero.py`, `model2libero_interface.py`, and the websocket / msgpack transport under `tools/` | [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA) | `ec8c70f6e155e2377bbd4d787004c14179c00c7c` | no LICENSE upstream; resolve before redistribution |
| `evals/stablevla/stablevla_policy_server.py` | StableVLA websocket policy server | [DAGroup-PKU/HumanNet](https://github.com/DAGroup-PKU/HumanNet) | `68eb86d555fc4d3b09003319cff7644cd4c8b571` | see `stablevla/LICENSE` |
| `assets/panda_arm_hand.urdf` | Franka Panda kinematic description, read by `robot_ood.core.kinematics` for forward kinematics. 10 KB of XML, no meshes | `franka_description` inside [robosuite](https://github.com/ARISE-Initiative/robosuite) 1.4.1 | robosuite `1.4.1` | Apache 2.0 (Franka Emika GmbH); robosuite itself MIT |

The harness imports resolve inside the vendored tree: `harness/tools/` holds the transport, and
`harness/deployment/model_server/tools/` re-exports it under the donor's own import path. Its
`model2libero_interface.py` is the websocket policy client BOTH benchmarks drive, so the
upstream-LIBERO client has that directory on its path too. Running the evals needs neither
`external/VLA-JEPA` nor a clone of the donor repo. Vendoring the URDF lets
the light `robot-ood-core` env run FK and the test suite with no robosuite and no simulator.

## Fetched, not committed

`docs/SETUP.md` §1 and `python -m robot_ood.cli bench setup <id>` populate these; `.gitignore` excludes them all.
Trained artifacts are not third-party and live outside git too: `python -m robot_ood.cli artifacts fetch`
installs a tree from the project's Hugging Face bucket.

| Path | Role | Source | Commit | License |
|---|---|---|---|---|
| `external/LIBERO-plus` | LIBERO-Plus benchmark: the `libero` package, perturbation suites, bddl | [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus), a fork of [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `4976dc30028e805ff8094b55501d532c48fec182` (main) | see tree `LICENSE` |
| `external/LIBERO` | the upstream LIBERO benchmark, run as published by `--bench libero` | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e3132a39270c3a13ebe37270a43ece2a01` (master) | see tree `LICENSE` |
| `external/Isaac-GR00T` | GR00T N1.7 model code; its `.venv` runs `evals/gr00t/ws_server.py` | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | tag `n1.7-release` (`23ace64`) | see tree `LICENSE` |
| `external/openpi` | pi0.5 model code; its `.venv` runs `evals/pi05/ws_server.py`; `pi05/` holds the `pi05_libero` checkpoint and RoboTwin's `cotrain_ckpt/Pi_05` checkpoint from the [RoboTwin2.0 dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) (trained by [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), served under `evals/pi05/robotwin_config.py`) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `215abfb217dbac7d5f1273282331b9b1866c0479` (main) | see tree `LICENSE` |
| `external/VLAct` | VLAct model code; its `.venv` runs `evals/vlact/ws_server.py` for `--policy vlact` on RoboTwin | [starVLA/VLAct](https://github.com/starVLA/VLAct) | `621b01b` | see tree `LICENSE` |
| `external/RoboTwin2.0-Plus` | RoboTwin 2.0-Plus robustness benchmark: the eight perturbation task configs over the RoboTwin 2.0 tasks; `evals/robotwin_plus/setup.sh` links RoboTwin's assets and environment | [Robot-Robustness/RoboTwin2.0-Plus](https://github.com/Robot-Robustness/RoboTwin2.0-Plus) | `8f07761` | see tree `LICENSE` |
| `external/RoboTwin` | RoboTwin 2.0 benchmark: the `envs/` task simulator (SAPIEN), `task_config/`, and its assets from the [RoboTwin2.0 dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0). `evals/robotwin/setup.sh` applies `evals/robotwin/patches/shader-from-env.patch` (the render shader from `$ROBOTWIN_SHADER`) | [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | `6e75b8b80ee32d2f495eb756cf5687e4e16f0eea` (main, 2025-10-22) | see tree `LICENSE` |
| `stablevla/` | StableVLA trunk (model code) | [DAGroup-PKU/HumanNet](https://github.com/DAGroup-PKU/HumanNet) `src/model/StableVLA` | `68eb86d555fc4d3b09003319cff7644cd4c8b571` | see tree `LICENSE` |

`evals/stablevla/stablevla_policy_server.py` and every `stablevla_ckpt` path in
`configs/trees/<suite>.yaml` read the trunk at the repo root as `stablevla/`. Weights download
separately under `stablevla/hf_weights/` ([`evals/stablevla/README.md`](evals/stablevla/README.md)).
The trunk is a runtime dependency of the evals, not of the `robot_ood` package.

## Data

`data/` holds demonstrations, `outputs/` the built artifacts and `results/` the rollouts. All three
are gitignored directories at the repo root; `configs/robot_ood.yaml` names the roots and each
`configs/trees/<suite>.yaml` names its demonstration directory under `data/`.
