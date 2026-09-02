#!/bin/bash
# P2 full-record chain (project_fullrecord_program_2026_07_11): fuse all four
# roots (ASP is2 + ASP_ctempoatm + ASP_glas + ASP_nocorr) over 2009-01-01 →
# 2024-01-10, build the 125 m stack, run the Shean-complete tilt (full
# shelf-inclusive domain + dh/dt Laplacian smoothness 1.0, IRLS capped at 5 —
# both 07-11 runs plateaued by iter 4), report-only bad-epoch screen, then the
# production path solve. LAUNCH ONLY after the nocorr re-verdict passes.
# Waits internally for the gap align+ingest chain sentinel.
# Sentinels: FULLREC_CHAIN_DONE / FULLREC_CHAIN_FAILED_<stage>.
set -u
cd /wd2/projects/stereo_melt/examples
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOGS=beardmore_shelf/logs
GATE_LOG=$LOGS/gap_align_ingest_chain.log

export BEARDMORE_SHELF_SOURCES=full
export BEARDMORE_SHELF_START=2009-01-01
export BEARDMORE_SHELF_END=2024-01-10

echo "=== fullrec chain start $(date '+%F %T') -- waiting for gap align+ingest chain ==="
until grep -q "GAP_CHAIN_INGEST_DONE" "$GATE_LOG" 2>/dev/null; do
  if grep -q "GAP_CHAIN_INGEST_FAILED" "$GATE_LOG" 2>/dev/null; then
    echo "FULLREC_CHAIN_FAILED_GAP_GATE (gap chain failed; not building)"
    exit 1
  fi
  sleep 120
done
echo "=== gap chain done -- proceeding $(date '+%F %T') ==="

run () {  # run <NAME> <log> <cmd...>
  local name=$1 log=$2; shift 2
  echo "=== $name start $(date '+%F %T')  (log: $log)"
  "$@" > "$log" 2>&1
  local rc=$?
  echo "=== $name rc=$rc $(date '+%F %T')"
  if [ $rc -ne 0 ]; then
    echo "FULLREC_CHAIN_FAILED_$name"
    exit 1
  fi
}

run BUILD $LOGS/build_stack_fullrec.log \
  $PY -u -m beardmore_shelf.build_stack --tag fullrec

# Preserve untagged figures the stages overwrite (names carry no tag).
for f in stack_coverage.png; do
  if [ -f "beardmore_shelf/figures/$f" ]; then
    cp -p "beardmore_shelf/figures/$f" "beardmore_shelf/figures/${f%.png}_pre_fullrec.png" 2>/dev/null
  fi
done

run TILT $LOGS/tilt_fit_fullrec.log \
  env BEARDMORE_SHELF_TILT_DOMAIN=full BEARDMORE_SHELF_TILT_DHDT_SMOOTH=1.0 \
      BEARDMORE_SHELF_TILT_IRLS_MAX=5 \
      STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
  $PY -u -m beardmore_shelf.tilt_fit --tag fullrec

run SCREEN $LOGS/find_bad_epochs_fullrec.log \
  $PY -u -m beardmore_shelf.scripts.find_bad_epochs --tag fullrec

run PATH_SOLVE $LOGS/run_melt_path_fullrec.log \
  env STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
  $PY -u -m beardmore_shelf.run_melt_path --tag fullrec

echo "FULLREC_CHAIN_DONE $(date '+%F %T')"
