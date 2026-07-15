#!/usr/bin/env bash
# Dense-tie flux-gap verdict chain (2026-06-25).
#
# Produces a clean, apples-to-apples densetie melt run to compare against the
# 2026-06-22 baseline is2ctempo run (Eul -4.32 / Lagr -10.24 m/yr; robust flux
# 4.5 / 8.1 Gt/yr vs Shean 82-93). Differs from baseline ONLY in the alignment
# root: dense-tie (ASP_densetie, Stage1 strip->static-masked 32m REMA) instead
# of the baseline CryoTEMPO aligns.
#
# Why a rebuild: the raw densetie stack on disk (2026-06-19) predates the dem_id
# keystone, so per-DEM BAD_STRIPS can't apply to it (that's why the 06-20 tilt
# fell back to date-keyed BAD_EPOCHS and over-dropped 536->277). The re-align is
# already done, so build_stack here is just reprojection (+dem_id) -> tilt
# (applies 20 BAD_STRIPS via dem_id) -> melt.
#
# Disconnect-safe: launched under nohup/setsid; each stage tees to its own log
# and aborts the chain on failure.
set -u
cd /wd2/projects/stereo_melt || exit 2
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOGDIR=pig/logs
STAMP=2026_06_25
CHAIN=$LOGDIR/densetie_verdict_${STAMP}.chain.log

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$CHAIN"; }

say "=== dense-tie flux-gap verdict chain START (pid $$) ==="

# ---- Stage 1/3: build raw densetie stack WITH dem_id -----------------------
say "Stage 1/3 build_stack --tag densetie (densetie source, +dem_id)"
if ! STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
      $PY -u -m pig.build_stack --res 250 --tag densetie \
        --is2-asp densetie --pre-is2-asp densetie \
      > "$LOGDIR/densetie_build_${STAMP}.log" 2>&1; then
  say "Stage 1 build_stack FAILED -- aborting (see densetie_build_${STAMP}.log)"; exit 1
fi
# Confirm the rebuilt raw stack now carries dem_id before proceeding.
if ! STEREO_MELT_BACKEND=numpy $PY - <<'PYEOF' >> "$CHAIN" 2>&1
import xarray as xr
ds = xr.open_dataset("pig/processed/pig_stack_250m_densetie_2010-01-01_2024-01-10.nc")
assert "dem_id" in ds.coords, "rebuilt raw densetie stack STILL has no dem_id"
print(f"  rebuilt raw densetie stack OK: time={ds.sizes['time']}, dem_id present")
PYEOF
then
  say "Stage 1 post-check FAILED (no dem_id) -- aborting"; exit 1
fi
say "Stage 1 build_stack DONE"

# ---- Stage 2/3: tilt_fit (applies 20 BAD_STRIPS via dem_id) -----------------
# Wider-AOI tilt_fit runs on CPU (numpy, no CUDA_VISIBLE_DEVICES); the 11 GB GPU
# can't hold the wider LSQ. See feedback_tilt_fit_cpu_default.
say "Stage 2/3 tilt_fit --tag densetie (CPU; BAD_STRIPS=20 via dem_id)"
if ! STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
      $PY -u -m pig.tilt_fit --res 250 --tag densetie \
        --is2-asp densetie --pre-is2-asp densetie \
      > "$LOGDIR/densetie_tilt_${STAMP}.log" 2>&1; then
  say "Stage 2 tilt_fit FAILED -- aborting (see densetie_tilt_${STAMP}.log)"; exit 1
fi
say "Stage 2 tilt_fit DONE"

# ---- Stage 3/3: run_melt (ase-quarterly v05, baseline-matched) -------------
say "Stage 3/3 run_melt --tag densetie (PIG_VELOCITY=ase-quarterly)"
if ! PIG_VELOCITY=ase-quarterly STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
      $PY -u -m pig.run_melt --res 250 --tag densetie \
      > "$LOGDIR/densetie_melt_${STAMP}.log" 2>&1; then
  say "Stage 3 run_melt FAILED -- aborting (see densetie_melt_${STAMP}.log)"; exit 1
fi
say "Stage 3 run_melt DONE"

say "=== dense-tie verdict chain COMPLETE ==="
# Surface the headline comparison lines for quick eyeballing.
say "--- densetie medians + flux ---"
grep -E "floating-only|Integrated basal flux|Eulerian   area|Lagrangian area|melt_rate: median" \
  "$LOGDIR/densetie_melt_${STAMP}.log" | tee -a "$CHAIN"
