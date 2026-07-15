#!/bin/bash
# Beardmore wider-AOI tilt_fit + run_melt on CPU. Same rationale as
# scripts/nansen_tilt_run_cpu.sh — wider-AOI LSQ exceeds 1080 Ti's 11 GB.
#
# Note: this run uses whatever wider-AOI stack is on disk now. fetch_strips
# (PID 15116 at launch) is still pulling the +480 wider-AOI strips, and the
# stack on disk was built before they all landed. Plan to re-run align_strips
# -> build_stack -> tilt_fit -> run_melt once fetch_strips completes.
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOGDIR=/wd2/projects/stereo_melt/beardmore/logs
cd /wd2/projects/stereo_melt
unset CUDA_VISIBLE_DEVICES
export STEREO_MELT_BACKEND=numpy
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
export OPENBLAS_NUM_THREADS=14

echo "=== Beardmore tilt_fit (CPU, $OMP_NUM_THREADS threads) === $(date)"
$PY -u -m beardmore.tilt_fit > "$LOGDIR/wider_aoi_tilt_fit.log" 2>&1
RC=$?
echo "   Beardmore tilt_fit exit=$RC $(date)"

if [ $RC -eq 0 ]; then
  echo "=== Beardmore run_melt (CPU) === $(date)"
  $PY -u -m beardmore.run_melt > "$LOGDIR/wider_aoi_run_melt.log" 2>&1
  RC2=$?
  echo "   Beardmore run_melt exit=$RC2 $(date)"
fi

echo "=== Beardmore wider-AOI CPU chain done === $(date)"
