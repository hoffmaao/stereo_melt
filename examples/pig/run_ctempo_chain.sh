#!/bin/bash
# Disconnect-safe CryoTEMPO chain for PIG (created 2026-06-09).
#
# Stage 1 (already running, PID below): pig.cache_cryotempo pre-IS2 prefetch +
#   per-strip caching → cryotempo_data/cryotempo_filtered_*.h5
# Stage 2 (this script): wait for stage 1 to EXIT, verify it COMPLETED cleanly
#   (summary line + >0 h5), then launch the pre-IS2 re-align with CryoTEMPO as
#   the CS2 control base into ASP_ctempoatmlvis/ (A/B vs raw-CS2 _cs2atmlvis).
#
# Unattended on a shared box → --parallel 1 (the 734 GB OOM was higher
# parallelism). nohup'd so it survives session close. Re-align does NOT fire
# if the cache died partway (no summary) or cached nothing.
set -u
cd /wd2/projects/stereo_melt/examples
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
CACHE_PID=22154
CACHE_LOG=pig/logs/cryotempo_cache_preIS2.log
ALIGN_LOG=pig/logs/align_ctempoatmlvis_preIS2.log
CHAIN_LOG=pig/logs/ctempo_chain.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$CHAIN_LOG"; }

log "chain start; waiting on cache PID $CACHE_PID"

# 1. Block until the cache process exits (12h runaway guard).
timeout 43200 tail --pid=$CACHE_PID -f /dev/null 2>/dev/null

if kill -0 $CACHE_PID 2>/dev/null; then
  log "GUARD FIRED: cache PID $CACHE_PID still alive after 12h; NOT launching re-align"
  exit 1
fi

# 2. Verify the cache completed cleanly.
if ! grep -q "CryoTEMPO cache summary" "$CACHE_LOG"; then
  log "cache process exited but NO summary line → partial/failed cache; NOT launching re-align"
  tail -6 "$CACHE_LOG" >> "$CHAIN_LOG"
  exit 1
fi
NCACHE=$(ls pig/data/ASP/cryotempo_data/*.h5 2>/dev/null | wc -l)
if [ "$NCACHE" -eq 0 ]; then
  log "summary present but 0 cryotempo h5 cached; NOT launching re-align"
  exit 1
fi
log "cache completed cleanly; $NCACHE cryotempo h5 cached"
[ "$NCACHE" -lt 100 ] && log "WARN: only $NCACHE cached (expected ~159) — launching anyway, review on return"

# 3. Guard against a double launch.
if pgrep -f "pig.align_strips.*_ctempoatmlvis" >/dev/null 2>&1; then
  log "a _ctempoatmlvis align is already running; NOT launching another"
  exit 0
fi

# 4. Launch the pre-IS2 re-align (mirrors the _cs2atmlvis window).
log "launching pre-IS2 re-align (_ctempoatmlvis, --parallel 1)"
nohup $PY -u -m pig.align_strips --control cs2 --cs2-source cryotempo \
  --use-atm --use-lvis --asp-suffix _ctempoatmlvis \
  --start 2010-01-01 --end 2018-10-13 --parallel 1 \
  > "$ALIGN_LOG" 2>&1 &
log "re-align launched PID $!; log $ALIGN_LOG"
