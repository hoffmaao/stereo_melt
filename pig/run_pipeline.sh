#!/usr/bin/env bash
# PIG end-to-end pipeline chain. Disconnect-safe: launch with nohup.
#
# Usage:
#   nohup bash pig/run_pipeline.sh > pig/logs/pipeline.log 2>&1 < /dev/null &
#
# Stages (each writes its own log under pig/logs/):
#   1. wait for pig.fetch_strips (PID 25239) to finish
#   2. pig.cache_icesat2  (IS2 control points for align_strips)
#   3. pig.align_strips --parallel 8
#   4. pig.build_stack
#   5. pig.tilt_fit
#   6. pig.run_melt + pig.run_pseudospectral in parallel
#
# Aborts on first failure (set -e). Each stage's full log is in
# pig/logs/<stage>.log; this top-level log carries the chain summary
# and timestamps.

set -euo pipefail

cd /wd2/projects/stereo_melt

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
export PYTHONUNBUFFERED=1
LOG_DIR=pig/logs
mkdir -p "$LOG_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo; echo "[$(stamp)] === $* ==="; }

# --- Stage 1: wait for fetch_strips ---------------------------------------
banner "Stage 1/6: wait for pig.fetch_strips"
while pgrep -f "pig.fetch_strips" > /dev/null 2>&1; do
    n_done=$(grep -c "Saved to" "$LOG_DIR/fetch_strips.log" 2>/dev/null || echo 0)
    echo "[$(stamp)] fetch_strips alive; $n_done strips saved so far"
    sleep 120
done
n_done=$(grep -c "Saved to" "$LOG_DIR/fetch_strips.log" 2>/dev/null || echo 0)
echo "[$(stamp)] fetch_strips exited; $n_done strips saved"

# --- Stage 2: cache_icesat2 -----------------------------------------------
banner "Stage 2/6: pig.cache_icesat2"
$PY -u -m pig.cache_icesat2 >> "$LOG_DIR/cache_icesat2.log" 2>&1
echo "[$(stamp)] cache_icesat2 OK"

# --- Stage 3: align_strips ------------------------------------------------
banner "Stage 3/6: pig.align_strips --parallel 8 --control is2"
$PY -u -m pig.align_strips --parallel 8 --control is2 \
    >> "$LOG_DIR/align_strips.log" 2>&1
echo "[$(stamp)] align_strips OK"

# --- Stage 4: build_stack -------------------------------------------------
banner "Stage 4/6: pig.build_stack"
$PY -u -m pig.build_stack >> "$LOG_DIR/build_stack.log" 2>&1
echo "[$(stamp)] build_stack OK"

# --- Stage 5: tilt_fit ----------------------------------------------------
banner "Stage 5/6: pig.tilt_fit"
$PY -u -m pig.tilt_fit >> "$LOG_DIR/tilt_fit.log" 2>&1
echo "[$(stamp)] tilt_fit OK"

# --- Stage 6: solvers in parallel -----------------------------------------
banner "Stage 6/6: pig.run_melt + pig.run_pseudospectral (parallel)"
$PY -u -m pig.run_melt          >> "$LOG_DIR/run_melt.log"          2>&1 &
PID_MELT=$!
$PY -u -m pig.run_pseudospectral >> "$LOG_DIR/run_pseudospectral.log" 2>&1 &
PID_PS=$!
echo "[$(stamp)] run_melt PID=$PID_MELT  run_pseudospectral PID=$PID_PS"

set +e
wait $PID_MELT;  RC_MELT=$?
wait $PID_PS;    RC_PS=$?
set -e
echo "[$(stamp)] run_melt rc=$RC_MELT  run_pseudospectral rc=$RC_PS"

if [ "$RC_MELT" -ne 0 ] || [ "$RC_PS" -ne 0 ]; then
    echo "[$(stamp)] !! one or both solvers failed; check stage logs"
    exit 1
fi

banner "Pipeline complete"
echo "[$(stamp)] Outputs:"
ls -la pig/results/ 2>&1 || true
