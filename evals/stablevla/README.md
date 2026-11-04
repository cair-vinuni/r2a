# StableVLA

**Trunk:** `stablevla/` — sparse copy of
[DAGroup-PKU/HumanNet](https://github.com/DAGroup-PKU/HumanNet) @ `68eb86d`, path
`src/model/StableVLA` (relocate with `$STABLEVLA_HOME`)
**Weights:** `stablevla/hf_weights/<suite>` + `stablevla/hf_weights/pretrained_models/`
**Code:** `evals/stablevla/stablevla_policy_server.py` (policy server), `evals/policies/stablevla.yaml`
(descriptor)

## Install

Trunk:

```bash
git clone --filter=blob:none --no-checkout https://github.com/DAGroup-PKU/HumanNet.git /tmp/hn
git -C /tmp/hn sparse-checkout init --cone
git -C /tmp/hn sparse-checkout set src/model/StableVLA
git -C /tmp/hn checkout 68eb86d555fc4d3b09003319cff7644cd4c8b571
mkdir -p stablevla external && cp -a /tmp/hn/src/model/StableVLA/. stablevla/ && rm -rf /tmp/hn
```

Toolchain — micromamba plus the `stablevla` (policy + simulator) and `onf-core` (`robot_ood` +
pytest) conda envs, everything inside the clone:

```bash
./install_stablevla.sh     # idempotent; re-run after any failure
source ./env.sh            # every shell from here on; puts the stablevla env on PATH
```

Weights (~6.9 GB):

```bash
hf download DAGroup-PKU/StableVLA --local-dir stablevla/hf_weights \
    --include "pretrained_models/*" --include "long/*"
hf download Qwen/Qwen2.5-0.5B
```

Repeat `--include` per pattern (two after one flag half-populates `hf_weights/` silently).
Swap `long/*` per suite.

Check:

```bash
python -m robot_ood.cli eval --suite long --mode base --check-env
```

## Use

The default policy: every command in [operations](../../docs/operations.md) runs against it
unless `--policy` says otherwise; `configs/trees/{long,object,spatial,goal}.yaml` name its
checkpoints.
