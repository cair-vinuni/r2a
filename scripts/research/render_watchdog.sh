#!/usr/bin/env bash
# Kills a sim client whose renderer has stopped returning.
#
# SAPIEN's take_picture does not always come back: one client sat inside update_picture for over
# three hours with its GPU idle while the other twenty-seven finished. The runner waits on every
# client, so one such client holds a whole cell. A client that has written nothing for the idle
# window is killed; the runner then closes the cell and --rerun-incomplete picks the shard up.
#
#   scripts/research/render_watchdog.sh [idle_seconds]
set -uo pipefail
IDLE=${1:-1500}
while true; do
    for pid in $(pgrep -f "[e]val_robotwin.py --policy-path"); do
        out=$(tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | grep -A1 -- '--video-out' | tail -1)
        shard=$(tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | grep -A1 -- '^--shard$' | tail -1)
        n=$(tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | grep -A1 -- '^--num-shards$' | tail -1)
        d="$out/shard${shard}_of_${n}"
        [ -d "$d" ] || continue
        last=$(find "$d" -newermt "-${IDLE} seconds" | head -1)
        started=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -n "${started:-}" ] || continue
        if [ -z "$last" ] && [ "$started" -gt "$IDLE" ]; then
            echo "$(date -u '+%F %T') killing $pid: $d idle > ${IDLE}s"
            kill -9 "$pid"
        fi
    done
    sleep 120
done
