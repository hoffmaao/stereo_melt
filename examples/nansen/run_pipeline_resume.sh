#!/usr/bin/env bash
# Nansen tilt_fit-forward re-run. Disconnect-safe: launch with nohup.
#
# Picks up post-2026-04-26 fixes (tilt z_ref pre-subtraction, IRLS,
# pyTMD CRSError, linear-inverse FFT ringing). Does NOT re-run align
# or build_stack — those are unchanged since the original Apr 26 run.
#
# Usage:
#   nohup bash nansen/run_pipeline_resume.sh > nansen/logs/pipeline.log 2>&1 < /dev/null &
#
# Stages (each writes its own log under nansen/logs/):
#   1. nansen.tilt_fit
#   2. nansen.run_melt + nansen.run_pseudospectral + nansen.run_stationary in parallel

set -euo pipefail

cd /wd2/projects/stereo_melt/examples

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
export PYTHONUNBUFFERED=1
LOG_DIR=nansen/logs
mkdir -p "$LOG_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo; echo "[$(stamp)] === $* ==="; }

# --- Stage 1: tilt_fit ----------------------------------------------------
banner "Stage 1/2: nansen.tilt_fit"
$PY -u -m nansen.tilt_fit >> "$LOG_DIR/tilt_fit.log" 2>&1
echo "[$(stamp)] tilt_fit OK"

# --- Stage 2: solvers in parallel -----------------------------------------
banner "Stage 2/2: nansen.run_melt + run_pseudospectral + run_stationary (parallel)"
$PY -u -m nansen.run_melt           >> "$LOG_DIR/run_melt.log"           2>&1 &
PID_MELT=$!
$PY -u -m nansen.run_pseudospectral >> "$LOG_DIR/run_pseudospectral.log" 2>&1 &
PID_PS=$!
$PY -u -m nansen.run_stationary     >> "$LOG_DIR/run_stationary.log"     2>&1 &
PID_ST=$!
echo "[$(stamp)] run_melt PID=$PID_MELT  run_pseudospectral PID=$PID_PS  run_stationary PID=$PID_ST"

set +e
wait $PID_MELT;  RC_MELT=$?
wait $PID_PS;    RC_PS=$?
wait $PID_ST;    RC_ST=$?
set -e
echo "[$(stamp)] run_melt rc=$RC_MELT  run_pseudospectral rc=$RC_PS  run_stationary rc=$RC_ST"

if [ "$RC_MELT" -ne 0 ] || [ "$RC_PS" -ne 0 ] || [ "$RC_ST" -ne 0 ]; then
    echo "[$(stamp)] !! one or more solvers failed; check stage logs"
    exit 1
fi

banner "Nansen re-run complete"
echo "[$(stamp)] Outputs:"
ls -la nansen/results/ 2>&1 || true
