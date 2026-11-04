#!/usr/bin/env bash
# evals/libero_plus/setup.sh — provision external/LIBERO-plus so `--bench libero_plus` can run.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ./env.sh

PLUS=external/LIBERO-plus
COMMIT=4976dc30028e805ff8094b55501d532c48fec182
DATASET=Sylvest/LIBERO-plus
PY=$(python -c 'from robot_ood.pipeline.interpreter import resolve_env_python; print(resolve_env_python("stablevla") or "python3")')
step() { echo; echo "=== $* ==="; }

step "1. checkout $PLUS @ ${COMMIT:0:7}"
if [ ! -d "$PLUS/.git" ]; then
    git clone https://github.com/sylvestf/LIBERO-plus.git "$PLUS"
fi
git -C "$PLUS" checkout --quiet "$COMMIT"
git -C "$PLUS" rev-parse --short HEAD

step "2. assets from $DATASET"
if [ ! -d "$PLUS/libero/libero/assets" ]; then
    STAGE=${TMPDIR:-/tmp}/lp
    "$PY" - "$DATASET" "$STAGE" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], "assets.zip", repo_type="dataset", local_dir=sys.argv[2]))
PYEOF
    unzip -q "$STAGE/assets.zip" -d "$STAGE/x"
    mv "$(find "$STAGE/x" -type d -name assets -print -quit)" "$PLUS/libero/libero/assets"
    rm -rf "$STAGE"
fi

step "3. libero/config.yaml (LIBERO_CONFIG_PATH points at its directory)"
R=$PWD/$PLUS/libero/libero
printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' \
    "$R" "$R" "$R" "$R" "$R" > "$PLUS/libero/config.yaml"

step "4. editable install"
"$PY" -m pip install -q --no-deps -e "./$PLUS"

step "5. self-test: episodes per suite"
python -m robot_ood.cli eval --suite long --mode base --bench libero_plus --check-env
