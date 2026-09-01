#!/bin/bash
# GLAS chain stage B (2026-07-10): wait for the REAL fetch python (pid arg;
# stage A accidentally waited on the dead launcher wrapper) → incremental
# cache_glas over the full 555-strip set → full GLAS-era align IF the pilot
# was approved (sentinel beardmore_shelf/data/ASP_glas/PILOT_APPROVED),
# else stop after caching.
set -u
cd /wd2/projects/stereo_melt
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
FETCH_PID=${1:?usage: glas_chain_stageB.sh <fetch-python-pid>}
SENTINEL=beardmore_shelf/data/ASP_glas/PILOT_APPROVED

echo "=== stage B waiting for fetch python (pid $FETCH_PID) $(date '+%F %T')"
while kill -0 "$FETCH_PID" 2>/dev/null; do sleep 120; done
echo "=== fetch done $(date '+%F %T'); log tail:"
tail -4 beardmore_shelf/logs/fetch_glas_era.log

# Don't overlap stage A's cache/pilot if they're still going.
while pgrep -f "beardmore_shelf.cache_glas" >/dev/null; do sleep 60; done
while pgrep -f "align_strips --control glas --asp-suffix _glas" >/dev/null; do sleep 60; done

echo
echo "=== cache_glas over the full strip set (incremental) $(date '+%F %T')"
BEARDMORE_SHELF_FETCH_START=2009-01-01 BEARDMORE_SHELF_FETCH_END=2010-10-12 \
  $PY -u -m beardmore_shelf.cache_glas --workers 6
echo "=== cache_glas rc=$? $(date '+%F %T')"

if [ -f "$SENTINEL" ]; then
  echo
  echo "=== PILOT_APPROVED found — launching full GLAS-era align $(date '+%F %T')"
  $PY -u -m beardmore_shelf.align_strips --control glas --asp-suffix _glas \
    --start 2009-01-01 --end 2010-10-12 --parallel 12 \
    > beardmore_shelf/logs/align_glas_full.log 2>&1
  echo "=== full align rc=$? $(date '+%F %T')  (log: align_glas_full.log)"
else
  echo "=== no $SENTINEL — stopping before the full align (pilot not approved yet)"
fi
echo "GLAS_CHAIN_STAGE_B_DONE"
