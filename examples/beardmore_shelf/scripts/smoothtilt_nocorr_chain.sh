#!/bin/bash
# Shean-complete tilt A/B, run 1: re-tilt the nocorr stack with the full
# (shelf-inclusive) observation domain + dh/dt smoothness 1.0 (vendor
# ndinterp.py parity, 2026-07-11), then screen + path solve + two-frame
# compare vs the P3 baseline. Backs up the failed static-domain artifacts
# first. Sentinels: SMOOTHTILT_NOCORR_DONE / SMOOTHTILT_NOCORR_FAILED_*.
set -u
cd /wd2/projects/stereo_melt/examples
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOG=beardmore_shelf/logs
P=beardmore_shelf/processed
W=2019-01-01_2024-01-10
# Fused-sources mode: config.STRIP_SOURCES must include ASP_nocorr so the
# per-layer Ez resolution finds the nocorr sidecars (without it the 99
# nocorr layers resolve 'missing'/is2 — the 07-11 first-launch mistake).
export BEARDMORE_SHELF_SOURCES=nocorr
say() { echo "=== $* $(date '+%F %T')"; }

for f in "$P/beardmore_shelf_tilt_params_nocorr_$W.nc" \
         "$P/beardmore_shelf_stack_nocorr_tilt_corrected_$W.nc" \
         "$P/beardmore_shelf_melt_path_125m_nocorr_$W.nc"; do
  if [ -f "$f" ] && [ ! -f "$f.presmooth_bak" ]; then cp -p "$f" "$f.presmooth_bak"; fi
done
if [ -f beardmore_shelf/figures/tilt_fit_qc.png ]; then
  cp -p beardmore_shelf/figures/tilt_fit_qc.png beardmore_shelf/figures/tilt_fit_qc_nocorr_static.png
fi

say "TILT(full+smooth) start"
BEARDMORE_SHELF_TILT_DOMAIN=full BEARDMORE_SHELF_TILT_DHDT_SMOOTH=1.0 \
STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
  "$PY" -u -m beardmore_shelf.tilt_fit --tag nocorr > "$LOG/tilt_fit_nocorr_smooth.log" 2>&1
rc=$?; say "TILT rc=$rc"
if [ $rc -ne 0 ]; then echo SMOOTHTILT_NOCORR_FAILED_TILT; exit 1; fi

say "SCREEN start (report-only)"
"$PY" -u -m beardmore_shelf.scripts.find_bad_epochs --tag nocorr > "$LOG/find_bad_epochs_nocorr_smooth.log" 2>&1
say "SCREEN rc=$?"

say "PATH_SOLVE start"
STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
  "$PY" -u -m beardmore_shelf.run_melt_path --tag nocorr > "$LOG/run_melt_path_nocorr_smooth.log" 2>&1
rc=$?; say "PATH rc=$rc"
if [ $rc -ne 0 ]; then echo SMOOTHTILT_NOCORR_FAILED_PATH; exit 1; fi

say "COMPARE start"
"$PY" -u -m beardmore_shelf.compare_nocorr > "$LOG/compare_nocorr_smooth.log" 2>&1
rc=$?; say "COMPARE rc=$rc"
if [ $rc -ne 0 ]; then echo SMOOTHTILT_NOCORR_FAILED_COMPARE; exit 1; fi
echo SMOOTHTILT_NOCORR_DONE
