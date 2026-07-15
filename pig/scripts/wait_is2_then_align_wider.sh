#!/bin/bash
#
# Wait for the in-flight pig.cache_icesat2 wider-AOI job (PID supplied
# as $1) to exit cleanly, then launch the wider IS2-era re-align under
# nohup. Detached from any controlling terminal via the setsid wrapper
# the caller uses — SIGHUP-immune on ssh drop.
#
# Usage:
#   setsid bash wait_is2_then_align_wider.sh <is2_pid> \
#       </dev/null >/dev/null 2>&1 &
#
# Logs land at:
#   pig/logs/align_is2era_wider_chain.log  — this watcher's narration
#   pig/logs/align_is2era_wider.log         — the align job itself

set -u
IS2_PID="${1:?need IS2 cache PID as arg 1}"

REPO=/wd2/projects/stereo_melt
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOG_DIR="$REPO/pig/logs"
CHAIN_LOG="$LOG_DIR/align_is2era_wider_chain.log"
IS2_LOG="$LOG_DIR/cache_icesat2_wider.log"
ALIGN_LOG="$LOG_DIR/align_is2era_wider.log"

mkdir -p "$LOG_DIR"

# pyproj/proj.db env (matches the in-flight cache jobs)
export PROJ_DATA="/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj"
export PROJ_LIB="$PROJ_DATA"

echo "[$(date)] watcher started, waiting for IS2 cache PID=$IS2_PID" >> "$CHAIN_LOG"

while kill -0 "$IS2_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] IS2 cache PID=$IS2_PID exited" >> "$CHAIN_LOG"

# Sanity-check that the IS2 cache reached its end-of-run summary line.
if grep -q "=== IS2 cache summary:" "$IS2_LOG"; then
    summary=$(grep "=== IS2 cache summary:" "$IS2_LOG" | tail -1)
    echo "[$(date)] IS2 cache summary line found: $summary" >> "$CHAIN_LOG"
else
    echo "[$(date)] IS2 cache log has no summary line — aborting align launch" >> "$CHAIN_LOG"
    exit 1
fi

cd "$REPO" || { echo "[$(date)] cd $REPO failed" >> "$CHAIN_LOG"; exit 1; }

nohup "$PY" -u -m pig.align_strips \
    --asp-suffix _is2atmlvis \
    --control is2 \
    --use-atm --use-lvis \
    --start 2018-10-14 --end 2024-01-10 \
    --parallel 4 \
    > "$ALIGN_LOG" 2>&1 </dev/null &

ALIGN_PID=$!
echo "[$(date)] launched pig.align_strips PID=$ALIGN_PID, log $ALIGN_LOG" >> "$CHAIN_LOG"
