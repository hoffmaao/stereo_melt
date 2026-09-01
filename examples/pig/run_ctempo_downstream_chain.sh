#!/bin/bash
# Disconnect-safe downstream chain for the PIG CryoTEMPO A/B (2026-06-10).
#
# Stage 0 (already running): pig.align_strips pre-IS2 re-align into
#   ASP_ctempoatmlvis/ (driver PID below, --parallel 3, ~1.5 d ETA).
# This script: wait for the align driver to EXIT, verify it produced a sane
#   number of aligned strips, then run the fused-stack pipeline with the
#   pre-IS2 source swapped to CryoTEMPO:
#     build_stack --res 250 --tag ctempo --pre-is2-asp ctempoatmlvis
#     tilt_fit    --res 250 --tag ctempo --pre-is2-asp ctempoatmlvis
#     run_melt    --res 250 --tag ctempo
# All outputs carry the _ctempo tag (pig_stack_250m_ctempo_*,
# pig_melt_250m_ctempo_*), so the cs2atmlvis baseline artifacts from the
# 06-02→06-07 run stay untouched for the A/B.
set -u
cd /wd2/projects/stereo_melt/examples
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ALIGN_PID=3933
ALIGN_LOG=pig/logs/align_ctempoatmlvis_preIS2.log
CHAIN_LOG=pig/logs/ctempo_downstream_chain.log
export PROJ_DATA=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj
export PROJ_LIB=$PROJ_DATA
export STEREO_MELT_BACKEND=numpy
export OMP_NUM_THREADS=14

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

log "downstream chain start; waiting on align driver PID $ALIGN_PID"

# 1. Block until the align driver exits (4-day runaway guard).
timeout 345600 tail --pid=$ALIGN_PID -f /dev/null 2>/dev/null
if kill -0 $ALIGN_PID 2>/dev/null; then
  log "GUARD FIRED: align PID $ALIGN_PID still alive after 4 d; NOT proceeding"
  exit 1
fi

# 2. Verify the align batch looks complete.
NOK=$(ls pig/data/ASP_ctempoatmlvis/asp_aligned/*-trans_reference-DEM.tif 2>/dev/null | wc -l)
LAST=$(grep -oE "\[[0-9]+/303\]" "$ALIGN_LOG" | tail -1)
log "align driver exited; last marker ${LAST:-none}; aligned DEMs: $NOK"
if [ "$NOK" -lt 80 ]; then
  log "GUARD: only $NOK aligned strips (<80 floor) — looks wrong; NOT proceeding"
  exit 1
fi
FREE_GB=$(df --output=avail -BG /wd2 | tail -1 | tr -dc 0-9)
if [ "$FREE_GB" -lt 150 ]; then
  log "GUARD: only ${FREE_GB}G free on /wd2 (<150G floor); NOT proceeding"
  exit 1
fi

# 3. Guard against a double launch.
if pgrep -f "pig\.(build_stack|tilt_fit|run_melt).*ctempo" >/dev/null 2>&1; then
  log "a ctempo downstream stage is already running; NOT launching another"
  exit 0
fi

# 4. Per-strip align A/B table (informational; never blocks the chain).
$PY -u -m pig.scripts.compare_ctempo_align_stats \
  > pig/logs/compare_ctempo_align_stats.log 2>&1 \
  || log "WARN: compare_ctempo_align_stats failed (non-fatal)"

# 5. The pipeline.
run_stage build_stack_250m_ctempo \
  $PY -u -m pig.build_stack --res 250 --tag ctempo --pre-is2-asp ctempoatmlvis
run_stage tilt_fit_250m_ctempo \
  $PY -u -m pig.tilt_fit --res 250 --tag ctempo --pre-is2-asp ctempoatmlvis
run_stage run_melt_250m_ctempo \
  $PY -u -m pig.run_melt --res 250 --tag ctempo

log "downstream chain COMPLETE: pig/results/pig_melt_250m_ctempo_*.nc (A/B vs pig_melt_250m_*.nc baseline)"
