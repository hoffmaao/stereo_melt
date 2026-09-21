#!/usr/bin/env bash
# Nansen tilt_fit-forward re-run. Disconnect-safe: launch with nohup.
#
# Picks up post-2026-04-26 fixes (tilt z_ref pre-subtraction, IRLS,
# pyTMD CRSError, linear-inverse FFT ringing). Does NOT re-run align
# or build_stack — those are unchanged since the original Apr 26 run.
#
# Usage:
#   nohup bash examples/nansen/run_pipeline_resume.sh > examples/nansen/logs/pipeline.log 2>&1 < /dev/null &
#
# Stages (each writes its own log under nansen/logs/):
#   1. nansen.tilt_fit
#   2. nansen.run_melt

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

# --- Stage 2: solver --------------------------------------------------
banner "Stage 2/2: nansen.run_melt"
$PY -u -m nansen.run_melt >> "$LOG_DIR/run_melt.log" 2>&1
echo "[$(stamp)] run_melt OK"

banner "Nansen re-run complete"
echo "[$(stamp)] Outputs:"
ls -la nansen/results/ 2>&1 || true
