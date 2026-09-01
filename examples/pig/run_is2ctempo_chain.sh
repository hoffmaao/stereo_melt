#!/bin/bash
# Disconnect-safe IS2-era IS2+CryoTEMPO chain for PIG (created 2026-06-12).
#
# Goal: the production-candidate IS2-era control mix — IS2 + CryoTEMPO +
# ATM/LVIS — i.e. the existing is2cs2atmlvis baseline with the raw ESA L2
# CS2 component swapped for CryoTEMPO (mirroring the swap already adopted
# pre-IS2). User directive 2026-06-12: KEEP IS2 in the control and KEEP the
# ASP_is2cs2atmlvis baseline root on disk (no deletions there).
# Flags mirror the is2cs2 baseline exactly (no balancing) so the per-strip
# A/B isolates the CS2-source swap. ctempo has ~10x fewer, gated points vs
# raw CS2, so the CS2-overweight concern is weaker here; an
# --inv-variance-balance variant stays a separate lever if needed.
#
# Both control caches are complete (IS2 wider cache; ctempo IS2-era cache
# 210 ok / 437, finished 2026-06-12), so the align starts immediately.
# Runs CONCURRENTLY with the ctempouniform chain (3+3 pc_align workers,
# 32 cores, 1 TB RAM, 2.1 TB free after the 919 GB point-cloud sweep).
# Alignments write SLIM: asp.py now deletes the ~1.3 GB *-trans_reference.tif
# intermediate after point2dem (keep_point_cloud=False default), so this
# root should land at ~60-100 GB, not 1.4 TB.
#
# Downstream fuses the production-candidate record:
#   IS2 era  <- ASP_is2ctempoatmlvis  (this align)
#   pre-IS2  <- ASP_ctempoatmlvis     (done 2026-06-11)
# tagged is2ctempo -> pig_melt_250m_is2ctempo_*.nc. A/B partners:
# pig_melt_250m_*.nc (baseline), _ctempo_* (hybrid), _ctempouniform_* (no-IS2).
set -u
cd /wd2/projects/stereo_melt/examples
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
CHAIN_LOG=pig/logs/is2ctempo_chain.log
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

log "chain start (IS2+ctempo IS2-era align, concurrent with ctempouniform chain)"

# Stage 1: IS2-era align, baseline flags with cs2-source swapped to cryotempo.
if pgrep -f "pig.align_strips.*_is2ctempoatmlvis" >/dev/null 2>&1; then
  log "an _is2ctempoatmlvis align is already running; NOT launching another"
  exit 0
fi
run_stage align_is2ctempoatmlvis_IS2era \
  $PY -u -m pig.align_strips --control is2+cs2 --cs2-source cryotempo \
  --use-atm --use-lvis --asp-suffix _is2ctempoatmlvis \
  --start 2018-10-13 --end 2024-01-10 --parallel 3

# Post-align guards before the ~17 h downstream stages.
NOK=$(ls pig/data/ASP_is2ctempoatmlvis/asp_aligned/*-trans_reference-DEM.tif 2>/dev/null | wc -l)
log "align done; aligned DEMs in ASP_is2ctempoatmlvis: $NOK (is2cs2 baseline got 375/422)"
if [ "$NOK" -lt 300 ]; then
  log "GUARD: only $NOK aligned DEMs (<300 floor); NOT proceeding"
  exit 1
fi
FREE_GB=$(df --output=avail -BG /wd2 | tail -1 | tr -dc 0-9)
if [ "$FREE_GB" -lt 150 ]; then
  log "GUARD: only ${FREE_GB}G free on /wd2 (<150G floor); NOT proceeding"
  exit 1
fi

# Housekeeping: sweep stale point-cloud intermediates from the two ctempo
# roots (>60 min old; in-flight strips untouched). The ctempouniform chain's
# align predates the asp.py keep_point_cloud patch, so it still writes
# clouds. NEVER sweep ASP_is2cs2atmlvis (user directive: keep IS2 data;
# any cleanup there is a separate, user-approved decision).
NSWEEP=$(find pig/data/ASP_ctempoatmlvis pig/data/ASP_is2ctempoatmlvis \
  -name "*-trans_reference.tif" -mmin +60 2>/dev/null | wc -l)
find pig/data/ASP_ctempoatmlvis pig/data/ASP_is2ctempoatmlvis \
  -name "*-trans_reference.tif" -mmin +60 -delete 2>/dev/null
log "swept $NSWEEP stale point-cloud intermediates from ctempo roots"

if pgrep -f "pig\.(build_stack|tilt_fit|run_melt).*tag is2ctempo" >/dev/null 2>&1; then
  log "an is2ctempo downstream stage is already running; NOT launching another"
  exit 0
fi

# Stages 2-4: production-candidate fused pipeline.
export STEREO_MELT_BACKEND=numpy
export OMP_NUM_THREADS=14
run_stage build_stack_250m_is2ctempo \
  $PY -u -m pig.build_stack --res 250 --tag is2ctempo \
  --is2-asp is2ctempoatmlvis --pre-is2-asp ctempoatmlvis
run_stage tilt_fit_250m_is2ctempo \
  $PY -u -m pig.tilt_fit --res 250 --tag is2ctempo \
  --is2-asp is2ctempoatmlvis --pre-is2-asp ctempoatmlvis
run_stage run_melt_250m_is2ctempo \
  $PY -u -m pig.run_melt --res 250 --tag is2ctempo

log "chain COMPLETE: pig/results/pig_melt_250m_is2ctempo_*.nc (IS2+ctempo era mix; A/B vs baseline + _ctempo_* + _ctempouniform_*)"
