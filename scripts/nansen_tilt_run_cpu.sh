#!/bin/bash
# Nansen wider-AOI tilt_fit + run_melt on CPU.
# Both 1080 Ti GPUs OOM on the wider-AOI LSQ (Nansen 22M unknowns at 25 m,
# 11 GB GPU can't hold A_dev + A_obs_dev + A_reg_w + the LSMR adjoint copy).
# Until GPU memory is fixed (or a no-adjoint-cache LinearOperator wrapper
# lands), wider-AOI tilt_fit runs on CPU. 1 TB system RAM is plenty.
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOGDIR=/wd2/projects/stereo_melt/examples/nansen/logs
cd /wd2/projects/stereo_melt
unset CUDA_VISIBLE_DEVICES
export STEREO_MELT_BACKEND=numpy
# Cap BLAS threads so parallel basin runs don't oversubscribe 32 cores.
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
export OPENBLAS_NUM_THREADS=14

echo "=== Nansen tilt_fit (CPU, $OMP_NUM_THREADS threads) === $(date)"
$PY -u -m nansen.tilt_fit > "$LOGDIR/wider_aoi_tilt_fit.log" 2>&1
RC=$?
echo "   Nansen tilt_fit exit=$RC $(date)"

if [ $RC -eq 0 ]; then
  echo "=== Nansen run_melt (CPU) === $(date)"
  $PY -u -m nansen.run_melt > "$LOGDIR/wider_aoi_run_melt.log" 2>&1
  RC2=$?
  echo "   Nansen run_melt exit=$RC2 $(date)"
fi

echo "=== Nansen wider-AOI CPU chain done === $(date)"
