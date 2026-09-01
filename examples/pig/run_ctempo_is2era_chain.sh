#!/bin/bash
# Disconnect-safe IS2-era CryoTEMPO chain for PIG (created 2026-06-12).
#
# Goal: a uniform-control 2010-2024 record — EVERY strip aligned against
# CryoTEMPO LI (+ ATM/LVIS airborne where present), no IS2 — so there is no
# control discontinuity at Oct 2018 and common-mode retracker bias cancels
# in DEM differences. User-approved trade: per-strip accuracy drops from
# IS2-class (±0.07 m) to ctempo-class (~0.8 m MAD), and sparse/empty-cache
# strips skip. IS2-era cache tally: 210 ok / 127 sparse / 99 empty of 437.
#
# Stage 0: wait for pig.cache_cryotempo IS2-era (PID below) — already at
#   [437/437] when this chain was written, so this should pass immediately.
# Stage 1: IS2-era re-align into the SAME ASP_ctempoatmlvis/ root holding
#   the finished pre-IS2 leg (163 aligned, 2010-01-01 → 2018-10-13). Flags
#   mirror the pre-IS2 leg exactly; --parallel 3 is safe post QC-plot OOM
#   fix (2026-06-10) — the pre-IS2 relaunch ran parallel-3 clean for 24 h.
# Stages 2-4: uniform fused 250 m stack from the single root via the
#   --is2-asp flag (added 2026-06-12), tagged ctempouniform:
#   pig_stack_250m_ctempouniform_* → *_tilt_corrected_* →
#   pig_melt_250m_ctempouniform_*.
# A/B partners already on disk: pig_melt_250m_*.nc (baseline) and
# pig_melt_250m_ctempo_*.nc (hybrid: IS2-era is2cs2 + pre-IS2 ctempo).
set -u
cd /wd2/projects/stereo_melt
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
CACHE_PID=30269
CACHE_LOG=pig/logs/cache_cryotempo_is2era.log
CHAIN_LOG=pig/logs/ctempo_is2era_chain.log
export PROJ_DATA=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj
export PROJ_LIB=$PROJ_DATA

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$CHAIN_LOG"; }

run_stage() {
  local name=$1; shift
  log "stage $name: $*"
  "$@" >> "pig/logs/${name}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "stage $name FAILED rc=$rc; aborting chain (see pig/logs/${name}.log)"
    exit 1
  fi
  log "stage $name done"
}

log "chain start; waiting on IS2-era ctempo cache PID $CACHE_PID"

# Stage 0: block until the cache exits (6 h runaway guard).
timeout 21600 tail --pid=$CACHE_PID -f /dev/null 2>/dev/null
if kill -0 $CACHE_PID 2>/dev/null; then
  log "GUARD FIRED: cache PID $CACHE_PID still alive after 6 h; NOT launching re-align"
  exit 1
fi
if ! grep -q "CryoTEMPO cache summary" "$CACHE_LOG"; then
  log "cache exited with NO summary line → partial/failed cache; NOT launching re-align"
  tail -6 "$CACHE_LOG" >> "$CHAIN_LOG"
  exit 1
fi
NCACHE=$(ls pig/data/ASP/cryotempo_data/*.h5 2>/dev/null | wc -l)
log "cache completed cleanly; $NCACHE cryotempo h5 total (143 pre-IS2 + IS2-era)"
if [ "$NCACHE" -lt 250 ]; then
  log "GUARD: only $NCACHE h5 cached (<250 floor incl. 143 pre-IS2); NOT proceeding"
  exit 1
fi

# Stage 1: IS2-era re-align (same root + flags as the pre-IS2 leg).
if pgrep -f "pig.align_strips.*_ctempoatmlvis" >/dev/null 2>&1; then
  log "a _ctempoatmlvis align is already running; NOT launching another"
  exit 0
fi
run_stage align_ctempoatmlvis_IS2era \
  $PY -u -m pig.align_strips --control cs2 --cs2-source cryotempo \
  --use-atm --use-lvis --asp-suffix _ctempoatmlvis \
  --start 2018-10-13 --end 2024-01-10 --parallel 3

# Post-align guards before the ~17 h downstream stages.
NOK=$(ls pig/data/ASP_ctempoatmlvis/asp_aligned/*-trans_reference-DEM.tif 2>/dev/null | wc -l)
log "align done; total aligned DEMs in ASP_ctempoatmlvis: $NOK (163 pre-IS2 + new IS2-era)"
if [ "$NOK" -lt 240 ]; then
  log "GUARD: only $NOK aligned DEMs (<240 floor = 163 pre-IS2 + 80 new); NOT proceeding"
  exit 1
fi
FREE_GB=$(df --output=avail -BG /wd2 | tail -1 | tr -dc 0-9)
if [ "$FREE_GB" -lt 150 ]; then
  log "GUARD: only ${FREE_GB}G free on /wd2 (<150G floor); NOT proceeding"
  exit 1
fi
if pgrep -f "pig\.(build_stack|tilt_fit|run_melt).*ctempouniform" >/dev/null 2>&1; then
  log "a ctempouniform downstream stage is already running; NOT launching another"
  exit 0
fi

# Stages 2-4: uniform-control fused pipeline.
export STEREO_MELT_BACKEND=numpy
export OMP_NUM_THREADS=14
run_stage build_stack_250m_ctempouniform \
  $PY -u -m pig.build_stack --res 250 --tag ctempouniform \
  --is2-asp ctempoatmlvis --pre-is2-asp ctempoatmlvis
run_stage tilt_fit_250m_ctempouniform \
  $PY -u -m pig.tilt_fit --res 250 --tag ctempouniform \
  --is2-asp ctempoatmlvis --pre-is2-asp ctempoatmlvis
run_stage run_melt_250m_ctempouniform \
  $PY -u -m pig.run_melt --res 250 --tag ctempouniform

log "chain COMPLETE: pig/results/pig_melt_250m_ctempouniform_*.nc (uniform ctempo control; A/B vs pig_melt_250m_*.nc baseline + pig_melt_250m_ctempo_*.nc hybrid)"
