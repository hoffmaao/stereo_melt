#!/usr/bin/env bash
# Re-run pc_align on the 29 PIG strips that previously failed with
# "no transformed cloud", now that their IS2 caches have been
# rebuilt with the Shean filter (median 159k photons/strip, min 101).
#
# Usage:
#   nohup bash pig/scripts/realign_pc_align_failures.sh \
#     > pig/logs/realign_pc_align.log 2>&1 < /dev/null &
#
# Inputs:  pig/scripts/pc_align_failed_strips.txt  (one dem-id per line)
# Output:  /wd2/projects/stereo_melt/data/REMA/strips/ASP/asp_aligned/
#          <dem_id>-trans_reference-DEM.tif

set -uo pipefail
cd /wd2/projects/stereo_melt

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
export PYTHONUNBUFFERED=1
LOG_DIR=pig/logs
mkdir -p "$LOG_DIR"

LIST=pig/scripts/pc_align_failed_strips.txt
N=$(wc -l < "$LIST")

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] === PIG re-align for $N pc_align-failed strips (Shean caches) ==="

ok=0; fail=0; i=0
while IFS= read -r dem_id; do
    [ -z "$dem_id" ] && continue
    i=$((i+1))
    echo
    echo "[$(stamp)] [$i/$N] $dem_id"
    if $PY -u -m pig.align_strips --dem-id "$dem_id"; then
        ok=$((ok+1))
        echo "[$(stamp)] [$i/$N] OK   (running ok=$ok fail=$fail)"
    else
        fail=$((fail+1))
        echo "[$(stamp)] [$i/$N] FAIL (running ok=$ok fail=$fail)"
    fi
done < "$LIST"

echo
echo "[$(stamp)] === PIG re-align done: $ok ok, $fail fail (of $N) ==="
