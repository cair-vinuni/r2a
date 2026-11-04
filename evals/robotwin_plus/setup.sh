#!/usr/bin/env bash
# evals/robotwin_plus/setup.sh — provision external/RoboTwin2.0-Plus so `--bench robotwin_plus`
# runs the robustness benchmark.
#
# Idempotent, and it stands on evals/robotwin/setup.sh: the Plus checkout is the 2.0 platform with
# the perturbation configs added, so it takes the same conda environment, the same fifteen
# gigabytes of assets and the same Vulkan manifest. Run the 2.0 setup first; this script clones
# the Plus checkout at its pinned commit, links the assets and the curobo checkout beside it, and
# applies the recorded patches: shader from the environment so no run can silently
# rasterise, and the paper's noise severity.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ./env.sh

ROOT=external/RoboTwin2.0-Plus
BASE=external/RoboTwin
COMMIT=8f077616e5c3c1cb29b88db2323212d51243438a
ENV=robotwin
PY="$MAMBA_ROOT_PREFIX/envs/$ENV/bin/python"
PATCHES="evals/robotwin_plus/patches/shader-from-env.patch
         evals/robotwin_plus/patches/noise-severity.patch"
step() { echo; echo "=== $* ==="; }

step "1. the 2.0 platform this one extends"
[ -x "$PY" ] && [ -d "$BASE/assets/objects" ] || { echo "run evals/robotwin/setup.sh first"; exit 1; }

step "2. checkout $ROOT @ ${COMMIT:0:7}"
if [ ! -d "$ROOT/.git" ]; then
    git clone https://github.com/Robot-Robustness/RoboTwin2.0-Plus.git "$ROOT"
fi
git -C "$ROOT" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$ROOT" checkout --quiet "$COMMIT"
git -C "$ROOT" rev-parse --short HEAD

step "3. assets and curobo, shared with $BASE"
for name in embodiments objects background_texture; do
    ln -sfn "../../RoboTwin/assets/$name" "$ROOT/assets/$name"
done
ln -sfn ../../RoboTwin/envs/curobo "$ROOT/envs/curobo"
ls -l "$ROOT/assets" | grep -c '^l'

step "4. shader from the environment, severity from the paper"
for patch in $PATCHES; do
    if git -C "$ROOT" apply --reverse --check "$PWD/$patch" 2>/dev/null; then
        echo "already applied: $patch"
    else
        git -C "$ROOT" apply "$PWD/$patch" && echo "applied: $patch"
    fi
done

step "5. import check"
ROBOTWIN_HOME="$PWD/$ROOT" PYTHONPATH="$PWD/$ROOT:${PYTHONPATH:-}" "$PY" - <<'PYEOF'
import os
os.chdir(os.environ["ROBOTWIN_HOME"])
from envs import beat_block_hammer          # noqa: F401
import yaml
for name in ("demo_clean", "demo_camera", "demo_robot_state", "demo_language_plus",
             "demo_light", "demo_background_plus", "demo_vision_noise", "demo_objects_plus"):
    yaml.safe_load(open(f"task_config/{name}.yml"))
    print("ok", name)
PYEOF

echo
echo "ready: python -m robot_ood.cli eval --bench robotwin_plus --suite plus50 --axis Original --mode base --policy pi05_robotwin --gpus 0,1,2,3"
