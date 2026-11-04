#!/usr/bin/env bash
# Every axis of one suite, base or fcsr, against one artifact tree.
#
#   scripts/research/eval_fcsr.sh <suite> <mode> [gpus] [results-root]
#
# Cells already complete under the results root are skipped (--rerun-incomplete), so a killed run
# resumes by re-invoking the same line.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ./env.sh

suite=${1:?suite (long|goal|object|spatial)}
mode=${2:-fcsr}
gpus=${3:-0,1,2,3}
results=${4:-results/$suite}
artifacts=$PWD/outputs/$suite/artifacts

run () {  # run <bench> <axis> [extra flags...]
    local bench=$1 axis=$2; shift 2
    echo "[$suite/$mode] $(date -u +%FT%TZ) $axis"
    python -m robot_ood.cli eval --bench "$bench" --suite "$suite" --mode "$mode" \
        --policy stablevla --gpus "$gpus" --axis "$axis" "$@" \
        --artifacts "$artifacts" --results-root "$results" --rerun-incomplete
}

for axis in Robot_Initial_States Objects_Layout Camera_Viewpoints Sensor_Noise \
            Background_Textures Light_Conditions Language_Instructions; do
    run libero_plus "$axis" --rung R4
done
run libero Original
echo "[$suite/$mode] $(date -u +%FT%TZ) all done"
