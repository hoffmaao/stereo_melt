#!/bin/bash
# Shean-consistency "nocorr" chain (project_shean_nocorr_framework_2026_07_10):
# wait for the GLAS stage-B chain to release the machine, then ingest the
# ~101 fully-floating strips (a-priori geolocation + class-mean -2.34 m),
# build the fused ASP+ASP_nocorr stack, tilt-fit (nocorr epochs Ez=1.0,
# full plane -- no end_p50 so the P3 gate passes them), re-screen, run the
# production path solver, and A/B vs the P3 baseline in BOTH frames
# (path-integrated Lagrangian + Eulerian).
set -u
cd /wd2/projects/stereo_melt
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOGS=beardmore_shelf/logs
GATE_LOG=$LOGS/glas_chain_stageB.log

echo "=== nocorr chain start $(date '+%F %T') -- waiting for GLAS stage B ==="
until grep -q "GLAS_CHAIN_STAGE_B_DONE" "$GATE_LOG" 2>/dev/null; do sleep 300; done
echo "=== GLAS chain done -- proceeding $(date '+%F %T') ==="

run () {  # run <NAME> <log> <cmd...>
  local name=$1 log=$2; shift 2
  echo "=== $name start $(date '+%F %T')  (log: $log)"
  "$@" > "$log" 2>&1
  local rc=$?
  echo "=== $name rc=$rc $(date '+%F %T')"
  if [ $rc -ne 0 ]; then
    echo "NOCORR_CHAIN_FAILED_$name"
    exit 1
  fi
}

run INGEST $LOGS/ingest_nocorr.log \
  $PY -u -m beardmore_shelf.ingest_nocorr

run BUILD $LOGS/build_stack_nocorr.log \
  env BEARDMORE_SHELF_SOURCES=nocorr \
  $PY -u -m beardmore_shelf.build_stack --tag nocorr

# tilt_fit_qc.png carries no tag -- preserve the P3 run's figure first.
if [ -f beardmore_shelf/figures/tilt_fit_qc.png ]; then
  cp -n beardmore_shelf/figures/tilt_fit_qc.png \
        beardmore_shelf/figures/tilt_fit_qc_p3.png
fi

run TILT $LOGS/tilt_fit_nocorr.log \
  env BEARDMORE_SHELF_SOURCES=nocorr STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
  $PY -u -m beardmore_shelf.tilt_fit --tag nocorr

run SCREEN $LOGS/find_bad_epochs_nocorr.log \
  env BEARDMORE_SHELF_SOURCES=nocorr \
  $PY -u -m beardmore_shelf.scripts.find_bad_epochs --tag nocorr

run PATH_SOLVE $LOGS/run_melt_path_nocorr.log \
  $PY -u -m beardmore_shelf.run_melt_path --tag nocorr

run COMPARE $LOGS/compare_nocorr.log \
  env BEARDMORE_SHELF_SOURCES=nocorr \
  $PY -u -m beardmore_shelf.compare_nocorr

echo "NOCORR_CHAIN_DONE $(date '+%F %T')"
