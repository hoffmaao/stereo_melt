#!/usr/bin/env bash
# Re-cache IS2 control with the Shean filter (|v|·|Δt| ≤ 10 m, ±365 d
# outer cap) for the 15 Nansen strips that hit
# "pc_align: no transformed cloud" in align_strips_allison.log.
# The original _allison cache used the older sample-based filter
# (smoothed |v| < cap within a fixed time window), which was too
# sparse on slow-flow grounded ice and starved pc_align of control.
#
# Usage:
#   nohup bash nansen/scripts/recache_pc_align_failures.sh \
#     > nansen/logs/recache_pc_align.log 2>&1 < /dev/null &
#
# Inputs:  nansen/scripts/pc_align_failed_strips.txt  (one dem-id per line)
# Output:  nansen/data/ASP/icesat2_data/
#          icesat2_filtered_<dem_id>.csv  (overwritten)

set -uo pipefail
cd /wd2/projects/stereo_melt

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
export PYTHONUNBUFFERED=1
LOG_DIR=nansen/logs
mkdir -p "$LOG_DIR"

LIST=nansen/scripts/pc_align_failed_strips.txt
N=$(wc -l < "$LIST")

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] === Nansen Shean re-cache for $N pc_align-failed strips ==="

ok=0; fail=0; i=0
while IFS= read -r dem_id; do
    [ -z "$dem_id" ] && continue
    i=$((i+1))
    echo
    echo "[$(stamp)] [$i/$N] $dem_id"
    if $PY -u -m nansen.cache_icesat2 --overwrite --dem-id "$dem_id"; then
        ok=$((ok+1))
        echo "[$(stamp)] [$i/$N] OK   (running ok=$ok fail=$fail)"
    else
        fail=$((fail+1))
        echo "[$(stamp)] [$i/$N] FAIL (running ok=$ok fail=$fail)"
    fi
done < "$LIST"

echo
echo "[$(stamp)] === Nansen Shean re-cache done: $ok ok, $fail fail (of $N) ==="
