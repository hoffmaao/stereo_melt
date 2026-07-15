#!/bin/bash
# Shean-complete tilt A/B, run 2: trans-only regression. Re-tilt the P3
# BASE stack (164 trans epochs, no nocorr) with the full domain + dh/dt
# smoothness under the throwaway tag `smoothab` (raw stack symlinked), then
# path solve + two-frame compare vs P3 production. This is the check that
# the shelf-inclusive system does NOT reintroduce the 2026-06-29
# manufactured-front-accretion artifact on the trans-only product.
# Sentinels: SMOOTHTILT_BASE_DONE / SMOOTHTILT_BASE_FAILED_*.
set -u
cd /wd2/projects/stereo_melt
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOG=beardmore_shelf/logs
P=beardmore_shelf/processed
W=2019-01-01_2024-01-10
say() { echo "=== $* $(date '+%F %T')"; }

ln -sf "beardmore_shelf_stack_$W.nc" "$P/beardmore_shelf_stack_smoothab_$W.nc"

say "TILT(full+smooth, base stack) start"
BEARDMORE_SHELF_TILT_DOMAIN=full BEARDMORE_SHELF_TILT_DHDT_SMOOTH=1.0 \
STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
  "$PY" -u -m beardmore_shelf.tilt_fit --tag smoothab > "$LOG/tilt_fit_smoothab.log" 2>&1
rc=$?; say "TILT rc=$rc"
if [ $rc -ne 0 ]; then echo SMOOTHTILT_BASE_FAILED_TILT; exit 1; fi

say "SCREEN start (report-only)"
"$PY" -u -m beardmore_shelf.scripts.find_bad_epochs --tag smoothab > "$LOG/find_bad_epochs_smoothab.log" 2>&1
say "SCREEN rc=$?"

say "PATH_SOLVE start"
STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
  "$PY" -u -m beardmore_shelf.run_melt_path --tag smoothab > "$LOG/run_melt_path_smoothab.log" 2>&1
rc=$?; say "PATH rc=$rc"
if [ $rc -ne 0 ]; then echo SMOOTHTILT_BASE_FAILED_PATH; exit 1; fi

say "COMPARE start"
BEARDMORE_SHELF_COMPARE_TAG_B=smoothab \
  "$PY" -u -m beardmore_shelf.compare_nocorr > "$LOG/compare_smoothab.log" 2>&1
rc=$?; say "COMPARE rc=$rc"
if [ $rc -ne 0 ]; then echo SMOOTHTILT_BASE_FAILED_COMPARE; exit 1; fi
echo SMOOTHTILT_BASE_DONE
