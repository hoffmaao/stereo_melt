#!/bin/bash
# P3 re-tilt chain (2026-07-09) — IS2-era production stack, first tilt_fit with
# the PIG-parity P3 offset-only demote (end_p50>3m -> alpha_z-only) + explicit
# shelf-excluded obs domain. Then the production path solver for the A/B vs the
# pre-P3 baseline (median +1.40 / CLIP -6.25 / robust -9.76, 164 ep), then the
# P4 find_bad_epochs re-screen (PIG framework: confirm convergence post-tilt).
#
# Launch (disconnect-safe):
#   nohup setsid bash beardmore_shelf/scripts/p3_retilt_chain.sh \
#     > beardmore_shelf/logs/p3_retilt_chain.log 2>&1 < /dev/null &
#
# Sentinels (beardmore_shelf/logs/): P3_TILT_DONE, P3_CHAIN_DONE,
#   P3_TILT_FAILED / P3_MELT_FAILED / P3_RESCREEN_FAILED on error.
# Pre-P3 baseline preserved as *.prep3bak alongside the originals.
set -u
cd /wd2/projects/stereo_melt/examples
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
L=beardmore_shelf/logs
# IS2-era production window = config defaults; do NOT set BEARDMORE_SHELF_SOURCES/START.
export STEREO_MELT_BACKEND=numpy
export OMP_NUM_THREADS=14
rm -f $L/P3_TILT_DONE $L/P3_CHAIN_DONE $L/P3_TILT_FAILED $L/P3_MELT_FAILED $L/P3_RESCREEN_FAILED

echo "[p3chain] stage 1/3 tilt_fit start $(date)"
$PY -u -m beardmore_shelf.tilt_fit > $L/tilt_fit_125m_p3.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[p3chain] tilt_fit FAILED rc=$rc $(date)"; touch $L/P3_TILT_FAILED; exit 1
fi
touch $L/P3_TILT_DONE
echo "[p3chain] stage 2/3 run_melt_path start $(date)"
$PY -u -m beardmore_shelf.run_melt_path > $L/run_melt_path_125m_p3.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[p3chain] run_melt_path FAILED rc=$rc $(date)"; touch $L/P3_MELT_FAILED; exit 1
fi
echo "[p3chain] stage 3/3 find_bad_epochs re-screen start $(date)"
$PY -u -m beardmore_shelf.scripts.find_bad_epochs > $L/find_bad_epochs_125m_p3.log 2>&1 \
  || touch $L/P3_RESCREEN_FAILED
touch $L/P3_CHAIN_DONE
echo "[p3chain] done $(date)"
