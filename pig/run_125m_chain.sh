#!/usr/bin/env bash
# 125 m resolution-pilot chain (2026-06-29).
#
# Re-grids the production is2ctempo aligned DEMs onto a 125 m analysis grid
# (2x finer than the 250 m production baseline) and re-runs the full solver
# stack, to test whether finer resolution sharpens melt-map features
# (basal channels, calving-front band) and how the median / IQR / gated flux
# move. pc_align coregistration is UNCHANGED (resolution-independent, already
# Shean-grade) -- this is reprojection + tilt + solvers only.
#
# Apples-to-apples with the 2026-06-22 250 m baseline (Eul -4.32 / Lagr -10.24
# m/yr; robust flux 4.5 / 8.1 Gt/yr): identical sources, window, BAD_STRIPS,
# ase-quarterly v05 velocity -- differs ONLY in --res (125 vs 250).
#
# DISCONNECT-IMMUNE: launched under setsid (own session, no controlling
# terminal -> no SIGHUP on logout). Stage 1 is SELF-HEALING -- it gates on the
# actual stack file (opens it, asserts dem_id + time dim), waits for the
# build_stack already running at launch (PID 15107), and re-runs build itself
# if that output never lands. Stages 2-3 abort the chain on failure.
set -u
cd /wd2/projects/stereo_melt || exit 2
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOGDIR=pig/logs
STAMP=2026_06_29
CHAIN=$LOGDIR/run_125m_chain_${STAMP}.chain.log
RAW=pig/processed/pig_stack_125m_is2ctempo_2010-01-01_2024-01-10.nc
TILT=pig/processed/pig_stack_125m_is2ctempo_tilt_corrected_2010-01-01_2024-01-10.nc
MELT=pig/results/pig_melt_125m_is2ctempo_2010-01-01_2024-01-10.nc
BUILD_PAT="pig.build_stack --res 125 --tag is2ctempo"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CHAIN"; }

echo $$ > "$LOGDIR/.run_125m_chain.pid"
say "=== 125 m chain START (pid $$, setsid-detached) ==="

# ---- Stage 1/3: ensure the 125 m raw stack exists (self-healing) -----------
# Success criterion = the file opens and carries dem_id + a full time dim,
# independent of which process built it. Robust to the launch-time build_stack
# dying: pgrep-waits for it, then re-runs build in-chain if still missing.
build_ok() {
  STEREO_MELT_BACKEND=numpy $PY - "$RAW" <<'PYEOF' >/dev/null 2>&1
import sys, xarray as xr
ds = xr.open_dataset(sys.argv[1])
assert "dem_id" in ds.coords, "no dem_id"
assert ds.sizes.get("time", 0) > 400, "short time dim"
PYEOF
}

say "Stage 1/3 ensure 125 m raw stack (+dem_id)"
if build_ok; then
  say "  build output already present & valid -> $RAW"
else
  if pgrep -f "$BUILD_PAT" >/dev/null; then
    say "  build_stack already running; waiting (poll 60 s, cap 6 h)..."
    for i in $(seq 1 360); do
      pgrep -f "$BUILD_PAT" >/dev/null || { say "  build_stack process exited after ${i} min"; break; }
      sleep 60
      [ $((i % 10)) -eq 0 ] && say "  ...still waiting on build_stack (${i} min elapsed)"
    done
  fi
  if build_ok; then
    say "  build_stack completed (waited) -> $RAW"
  else
    say "  no valid build output; running build_stack IN-CHAIN"
    if ! STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
          $PY -u -m pig.build_stack --res 125 --tag is2ctempo \
          > "$LOGDIR/build_stack_125m_is2ctempo_inchain_${STAMP}.log" 2>&1; then
      say "Stage 1 in-chain build_stack FAILED -- aborting"; exit 1
    fi
    build_ok || { say "Stage 1 build output STILL invalid -- aborting"; exit 1; }
    say "  in-chain build_stack DONE -> $RAW"
  fi
fi
say "Stage 1 DONE"

# ---- Stage 2/3: tilt_fit (applies 20 BAD_STRIPS via dem_id) -----------------
# Wider-AOI tilt_fit runs on CPU (numpy, no CUDA_VISIBLE_DEVICES); the 11 GB GPU
# can't hold the wider LSQ -- and at 125 m it is ~4x the 250 m system. CPU has
# 990 GB free, so RAM is not the limit; expect ~2-4 days for the 8-iter IRLS.
say "Stage 2/3 tilt_fit --res 125 --tag is2ctempo (CPU; BAD_STRIPS=20 via dem_id)"
if ! STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
      $PY -u -m pig.tilt_fit --res 125 --tag is2ctempo \
      > "$LOGDIR/tilt_fit_125m_is2ctempo_${STAMP}.log" 2>&1; then
  say "Stage 2 tilt_fit FAILED -- aborting (see tilt_fit_125m_is2ctempo_${STAMP}.log)"; exit 1
fi
[ -s "$TILT" ] || { say "Stage 2 produced no tilt-corrected stack ($TILT) -- aborting"; exit 1; }
say "Stage 2 tilt_fit DONE -> $TILT"

# ---- Stage 3/3: run_melt (ase-quarterly v05, baseline-matched) -------------
say "Stage 3/3 run_melt --res 125 --tag is2ctempo (PIG_VELOCITY=ase-quarterly)"
if ! PIG_VELOCITY=ase-quarterly STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
      $PY -u -m pig.run_melt --res 125 --tag is2ctempo \
      > "$LOGDIR/run_melt_125m_is2ctempo_${STAMP}.log" 2>&1; then
  say "Stage 3 run_melt FAILED -- aborting (see run_melt_125m_is2ctempo_${STAMP}.log)"; exit 1
fi
[ -s "$MELT" ] || { say "Stage 3 produced no melt product ($MELT) -- aborting"; exit 1; }
say "Stage 3 run_melt DONE -> $MELT"

say "=== 125 m chain COMPLETE ==="
# Headline 125 m numbers for quick eyeballing, with the 250 m baseline alongside.
say "--- 125 m medians + flux (this run) ---"
grep -E "floating-only|Integrated basal flux|Eulerian   area|Lagrangian area|melt_rate: median|median=" \
  "$LOGDIR/run_melt_125m_is2ctempo_${STAMP}.log" | tee -a "$CHAIN"
say "--- 250 m baseline for reference: Eul -4.32 / Lagr -10.24 m/yr; robust flux 4.5 / 8.1 Gt/yr (1666/1670 km2) ---"
say "Compare figure: pig/figures/melt_comparison_125m_is2ctempo.png  vs  melt_comparison_250m_is2ctempo.png"
