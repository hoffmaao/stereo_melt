#!/bin/bash
# Auto-launch densetie-from-raw once the CryoTEMPO budget re-cache (PID 29022)
# finishes, so the Stage-2 datum CSVs are built from the freshly-recached
# CryoTEMPO control. Armed 2026-06-18 per user "auto-launch tonight".
# Run detached so it survives logout:
#   setsid bash pig/autolaunch_fromraw.sh < /dev/null >> pig/logs/autolaunch_fromraw.log 2>&1 &
set -u
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
cd /wd2/projects/stereo_melt || exit 1
CACHE_PID=29022
CACHE_LOG=pig/logs/cache_cryotempo_budget.log

echo "[$(date)] === auto-launch armed; gate = budget re-cache PID $CACHE_PID ==="

# Block until the budget re-cache exits (no-op if it's already gone). Guard
# against a reused PID by confirming it's still the cache_cryotempo process.
if pgrep -af cache_cryotempo | grep -q "^${CACHE_PID} "; then
  echo "[$(date)] $CACHE_PID is cache_cryotempo; waiting for it to finish..."
  tail --pid="$CACHE_PID" -f /dev/null
else
  echo "[$(date)] $CACHE_PID is not running as cache_cryotempo (already done?); proceeding."
fi

echo "[$(date)] budget re-cache done. Recovery tally (baseline = 376 ok / 198 sparse / 325 empty of 899):"
grep -aE 'CryoTEMPO cache summary' "$CACHE_LOG" | tail -n 8
grep -aoE 'ok=[0-9]+ skip=[0-9]+ sparse=[0-9]+ empty=[0-9]+ fail=[0-9]+' "$CACHE_LOG" | tail -n 1

echo "[$(date)] launching densetie-from-raw (--parallel 8)..."
$PY -u -m pig.align_strips_densetie_fromraw --parallel 8
echo "[$(date)] === densetie-from-raw finished (exit $?) ==="
