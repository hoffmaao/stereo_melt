#!/bin/bash
# Gap-era (2010-02-07 → 2012-11-17) completion chain, 2026-07-11 evening
# (project_fullrecord_program_2026_07_11 P1 tail):
#   1. CryoTEMPO-controlled align of the 49 fetched gap strips into
#      ASP_ctempoatm (16 usable + 5 sparse caches; empty-cache strips are
#      upfront-skipped and become nocorr candidates).
#   2. Refresh + apply the ctempoatm QC screen (now including gap strips).
#   3. Re-run the GLAS-era nocorr ingest — the 11:28 run raced the gap fetch,
#      so 2010-02→2010-10-13 gap strips were missing from its census
#      (idempotent: already-ingested strips are skipped).
#   4. The big pre-IS2 (2010-10-13 → 2019-01-01) nocorr ingest @ −0.80 m —
#      floating + no-control + align-failed + readmit-underdetermined.
# Sentinels: GAP_CHAIN_INGEST_DONE / GAP_CHAIN_INGEST_FAILED_<stage>.
set -u
cd /wd2/projects/stereo_melt/examples
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOGS=beardmore_shelf/logs

run () {  # run <NAME> <log> <cmd...>
  local name=$1 log=$2; shift 2
  echo "=== $name start $(date '+%F %T')  (log: $log)"
  "$@" > "$log" 2>&1
  local rc=$?
  echo "=== $name rc=$rc $(date '+%F %T')"
  if [ $rc -ne 0 ]; then
    echo "GAP_CHAIN_INGEST_FAILED_$name"
    exit 1
  fi
}

run ALIGN_GAP $LOGS/align_gap_era.log \
  $PY -u -m beardmore_shelf.align_strips --control cs2 --cs2-source cryotempo \
      --asp-suffix _ctempoatm --start 2010-02-07 --end 2012-11-17 --parallel 8

run QC_REPORT $LOGS/qc_ctempoatm_gap_report.log \
  $PY -u -m beardmore_shelf.scripts.qc_glas_align --root ASP_ctempoatm

run QC_APPLY $LOGS/qc_ctempoatm_gap_apply.log \
  $PY -u -m beardmore_shelf.scripts.qc_glas_align --root ASP_ctempoatm --apply

# GLAS-era re-census (gap strips 2010-02-07..2010-10-13 were fetched after
# the 11:28 era ingest ran; CS2 ops begin 2010-07 so most have no control).
run INGEST_GLASERA $LOGS/ingest_nocorr_glasera2.log \
  env BEARDMORE_SHELF_START=2009-01-01 BEARDMORE_SHELF_END=2010-10-13 \
  $PY -u -m beardmore_shelf.ingest_nocorr --z-offset -2.25 \
      --include-align-failed --include-no-control --readmit-underdetermined

# Pre-IS2 era: the ~337-strip class (211 floating + 95 no-control +
# 8 align-failed + readmits) plus whatever the gap align could not anchor.
run INGEST_PREIS2 $LOGS/ingest_nocorr_preis2.log \
  env BEARDMORE_SHELF_START=2010-10-13 BEARDMORE_SHELF_END=2019-01-01 \
  $PY -u -m beardmore_shelf.ingest_nocorr --z-offset -0.80 \
      --include-align-failed --include-no-control --readmit-underdetermined

echo "GAP_CHAIN_INGEST_DONE $(date '+%F %T')"
