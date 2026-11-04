#!/usr/bin/env bash
# evals/libero/setup.sh — provision external/LIBERO so `--bench libero` runs the upstream benchmark.
#
# Idempotent. Clones Lifelong-Robot-Learning/LIBERO at a pinned commit; the bddl task files and the
# fifty init states per task ship in that tree, so nothing is downloaded beyond the clone. The
# package asks on first import where to keep datasets and writes ~/.libero/config.yaml; that file
# is written here instead, into the checkout, and LIBERO_CONFIG_PATH points the package at it, so
# no prompt, nothing under $HOME, and the answer is the checkout's own directories.
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT=external/LIBERO
COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
step() { echo; echo "=== $* ==="; }

step "1. checkout $ROOT @ ${COMMIT:0:7}"
if [ ! -d "$ROOT/.git" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$ROOT"
fi
git -C "$ROOT" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$ROOT" checkout --quiet "$COMMIT"
git -C "$ROOT" rev-parse --short HEAD

step "2. config.yaml the package would otherwise prompt for"
PKG="$(cd "$ROOT/libero/libero" && pwd)"
CFG="$ROOT/libero/config.yaml"
printf 'assets: %s/assets\nbenchmark_root: %s\nbddl_files: %s/bddl_files\ndatasets: %s/datasets\ninit_states: %s/init_files\n' \
    "$PKG" "$PKG" "$PKG" "$(cd "$ROOT/libero" && pwd)" "$PKG" > "$CFG"
cat "$CFG"

step "3. task files"
for suite in libero_spatial libero_object libero_goal libero_10; do
    n_bddl=$(ls "$PKG/bddl_files/$suite"/*.bddl | wc -l)
    n_init=$(ls "$PKG/init_files/$suite"/*.pruned_init | wc -l)
    echo "$suite: $n_bddl bddl, $n_init init"
    [ "$n_bddl" = 10 ] && [ "$n_init" = 10 ] || { echo "expected 10 of each" >&2; exit 1; }
done

echo
echo "ready: python -m robot_ood.cli eval --bench libero --suite long --axis Original --mode base --policy gr00t --gpus 0"
