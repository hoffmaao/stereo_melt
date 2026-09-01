#!/bin/bash
# Beardmore wider-AOI rebuild chain on CPU.
#
# Re-runs build_stack -> tilt_fit -> run_melt now that the May 8-9 realign
# chain finished and the asp_aligned dirs hold the complete set of
# Beardmore-AOI strips (276 unique vs the 72 the previous wider-AOI build
# saw on May 7). The earlier 47-epoch stack covered only ~26% of the
# available aligned DEMs and left a 38 km x 77 km hole at the inland
# (Queen Alexandra Range) edge of the AOI. Backup of the pre-rebuild
# artifacts is in beardmore/processed/_pre_rebuild_20260511/.
#
# Runs on CPU per feedback_tilt_fit_cpu_default: the 11 GB 1080 Ti can't
# hold the wider-AOI tilt LSQ.

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOGDIR=/wd2/projects/stereo_melt/examples/beardmore/logs
cd /wd2/projects/stereo_melt

unset CUDA_VISIBLE_DEVICES
export STEREO_MELT_BACKEND=numpy
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
export OPENBLAS_NUM_THREADS=14

CHAINLOG="$LOGDIR/wider_aoi_rebuild_chain.log"
exec > >(tee -a "$CHAINLOG") 2>&1

echo "=== Beardmore wider-AOI rebuild chain start === $(date)"
df -h /wd2 | awk 'NR==1 || /wd/'

echo
echo "=== stage 1/3: build_stack === $(date)"
$PY -u -m beardmore.build_stack > "$LOGDIR/wider_aoi_build_stack.log" 2>&1
RC1=$?
echo "   build_stack exit=$RC1 $(date)"
if [ $RC1 -ne 0 ]; then
  echo "!! build_stack failed; abort chain"
  exit $RC1
fi

echo
echo "=== stage 2/3: tilt_fit (CPU, $OMP_NUM_THREADS threads) === $(date)"
$PY -u -m beardmore.tilt_fit > "$LOGDIR/wider_aoi_tilt_fit.log" 2>&1
RC2=$?
echo "   tilt_fit exit=$RC2 $(date)"
if [ $RC2 -ne 0 ]; then
  echo "!! tilt_fit failed; abort chain"
  exit $RC2
fi

echo
echo "=== stage 3/3: run_melt (CPU) === $(date)"
$PY -u -m beardmore.run_melt > "$LOGDIR/wider_aoi_run_melt.log" 2>&1
RC3=$?
echo "   run_melt exit=$RC3 $(date)"

echo
echo "=== Beardmore wider-AOI rebuild chain done === $(date)"
df -h /wd2 | awk 'NR==1 || /wd/'
