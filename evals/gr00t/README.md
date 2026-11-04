# GR00T-N1.7

**Checkout:** `external/Isaac-GR00T` @ tag `n1.7-release` (23ace64)
**Weights:** `gr00t/hf_weights/<task_suite>` (relocate with `$GR00T_HOME`)
**Code:** `evals/gr00t/ws_server.py` (policy server), `evals/policies/gr00t.yaml` (descriptor)

## Install

The backbone processor streams from the gated repo
[nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B): accept its license
there with your Hugging Face account first, and be logged in (`hf auth login`) — the server dies
with `403 gated repo` otherwise.

```bash
git clone --branch n1.7-release --depth 1 https://github.com/NVIDIA/Isaac-GR00T external/Isaac-GR00T
(cd external/Isaac-GR00T && uv sync && uv pip install websockets h5py)
external/Isaac-GR00T/.venv/bin/hf download nvidia/GR00T-N1.7-LIBERO --local-dir gr00t/hf_weights \
    --exclude "*/global_step*/*" "*/rng_state*" "*.pth" "*optim*" "*/latest" "*/zero_to_fp32.py" "*/scheduler.pt"
```

Check:

```bash
external/Isaac-GR00T/.venv/bin/python -c "import gr00t, websockets, h5py"
```

The checkout is never modified; `evals/policies/gr00t.yaml` points every launcher at its `.venv`.

## Use

`configs/trees/long_gr00t.yaml` is the GR00T tree over the `libero_10` corpus
([operations](../../docs/operations.md)):

```bash
python -m robot_ood.cli cache-policy --suite long_gr00t --port 14200   # a_pi_raw.npz under GR00T
python -m robot_ood.cli build --suite long_gr00t --device cuda:0
python -m robot_ood.cli train --suite long_gr00t --device cuda:0
python -m robot_ood.cli eval --suite long_gr00t --mode fcsr --rung R1 --gpus 0
```

A base-arm run needs no tree:

```bash
scripts/run_sr.py --cells long:Robot_Initial_States --policy gr00t --mode base --gpus 0
```

