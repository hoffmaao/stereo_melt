#!/bin/bash
# GLAS-era back-extension chain, stage A (2026-07-10):
#   wait for the 2009→2010-10 strip fetch → cache_glas (GLAH12 prefetch +
#   per-strip control CSVs) → pilot align of 3 strips with GLAS control.
# STOPS after the pilot — inspect pc_align convergence + residuals before
# launching the full ~555-strip align:
#   nohup setsid python -u -m beardmore_shelf.align_strips \
#     --control glas --asp-suffix _glas --start 2009-01-01 --end 2010-10-12 \
#     --parallel 12 > beardmore_shelf/logs/align_glas_full.log 2>&1 &
set -u
cd /wd2/projects/stereo_melt/examples
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
FETCH_PID=${1:-17226}

echo "=== waiting for fetch_strips (pid $FETCH_PID) $(date '+%F %T')"
while kill -0 "$FETCH_PID" 2>/dev/null; do sleep 60; done
echo "=== fetch done $(date '+%F %T'); log tail:"
tail -4 beardmore_shelf/logs/fetch_glas_era.log

echo
echo "=== cache_glas: GLAH12 prefetch + per-strip CSVs $(date '+%F %T')"
BEARDMORE_SHELF_FETCH_START=2009-01-01 BEARDMORE_SHELF_FETCH_END=2010-10-12 \
  $PY -u -m beardmore_shelf.cache_glas --workers 6
echo "=== cache_glas rc=$? $(date '+%F %T')"

echo
echo "=== pilot align: 3 strips, GLAS-only control $(date '+%F %T')"
$PY -u -m beardmore_shelf.align_strips --control glas --asp-suffix _glas \
  --start 2009-01-01 --end 2010-10-12 --limit 3
echo "=== pilot align rc=$? $(date '+%F %T')"
echo
echo "=== pilot outputs:"
ls -la beardmore_shelf/data/ASP_glas/asp_aligned/ 2>/dev/null | head -20
echo "GLAS_CHAIN_STAGE_A_DONE"
