#!/bin/bash
# Canonical wider-AOI tilt_fit + run_melt chain for Nansen and Beardmore.
#
# History: an earlier version of this script ran on GPU (cupy / 1080 Ti).
# Both basins' wider-AOI LSQs exceed 11 GB GPU memory, so until the GPU
# memory situation improves (or a no-adjoint-cache LinearOperator wrapper
# lands in stereo_melt.coregister.tilt), wider-AOI tilt_fit is CPU-only.
# Per-basin scripts (nansen_tilt_run_cpu.sh, beardmore_tilt_run_cpu.sh)
# carry the actual env + module invocations; this file just launches both
# in parallel so they share the 32 cores fairly (14 BLAS threads each).
#
# Logs:
#   nansen/logs/wider_aoi_chain_cpu.log
#   beardmore/logs/wider_aoi_chain_cpu.log
#   plus per-stage logs already written by each chain.
SCRIPT_DIR=/wd2/projects/stereo_melt/scripts

bash "$SCRIPT_DIR/nansen_tilt_run_cpu.sh" \
    > /wd2/projects/stereo_melt/nansen/logs/wider_aoi_chain_cpu.log 2>&1 &
NANSEN_PID=$!

bash "$SCRIPT_DIR/beardmore_tilt_run_cpu.sh" \
    > /wd2/projects/stereo_melt/beardmore/logs/wider_aoi_chain_cpu.log 2>&1 &
BEARDMORE_PID=$!

echo "Nansen chain PID=$NANSEN_PID"
echo "Beardmore chain PID=$BEARDMORE_PID"
echo "$NANSEN_PID $BEARDMORE_PID" > /tmp/wider_aoi_cpu_chain.pids

wait $NANSEN_PID
echo "=== Nansen chain exited $(date) ==="
wait $BEARDMORE_PID
echo "=== Beardmore chain exited $(date) ==="
echo "=== wider-AOI CPU chain DONE === $(date)"
