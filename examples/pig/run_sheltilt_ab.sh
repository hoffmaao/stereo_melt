#!/bin/bash
# A/B test: Shean-faithful shelf-EXCLUDED tilt domain (obs_domain = static
# control) vs the baseline DEM-interior obs_domain (full ice + shelf).
#
# Only the tilt-fit observation domain changes; same raw stack (symlinked under
# the is2ctempo_sheltilt tag), same 513-epoch population (20 BAD_STRIPS dropped
# at load), same velocity (ase-quarterly v05), same CPU/numpy backend. Baseline
# files (is2ctempo) are left untouched for the comparison.
#
# Baseline to beat:  Eul med -4.32 / Lagr med -10.24 ; flux robust 4.5/8.1 Gt/yr
# Hypothesis: removing the shelf from the tilt LSQ kills the joint-LSQ nullspace
# that aliases the along-flow melt gradient into a per-epoch tilt -> front
# accretion vanishes, integrated flux moves toward Shean's 82-93 Gt/yr.
set -euo pipefail
cd /wd2/projects/stereo_melt
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python

export STEREO_MELT_BACKEND=numpy
export OMP_NUM_THREADS=14
export PIG_VELOCITY=ase-quarterly
unset CUDA_VISIBLE_DEVICES || true

TS=2026_06_27
TILT_LOG=pig/logs/tilt_fit_sheltilt_${TS}.log
MELT_LOG=pig/logs/run_melt_sheltilt_${TS}.log

echo "=== [$(date)] Stage 1/2: tilt_fit (shelf-excluded, obs_domain=control) -> $TILT_LOG ==="
$PY -u -m pig.tilt_fit --res 250 --tag is2ctempo_sheltilt > "$TILT_LOG" 2>&1
echo "=== [$(date)] tilt_fit DONE. Stage 2/2: run_melt -> $MELT_LOG ==="
$PY -u -m pig.run_melt --res 250 --tag is2ctempo_sheltilt > "$MELT_LOG" 2>&1
echo "=== [$(date)] run_melt DONE. A/B complete. ==="
grep -E "floating-only|Integrated|Eulerian|Lagrangian" "$MELT_LOG" | tail -8
