#!/bin/bash
# Quick status check for the Nansen wider-AOI rebuild chain
LOGDIR=/wd2/projects/stereo_melt/nansen/logs
PID=$(cat /tmp/nansen_chain.pid 2>/dev/null)

echo "=== chain process (pid $PID) ==="
ps -p "$PID" -o pid,etime,stat,pcpu,cmd 2>&1 | head -3
echo
echo "=== stages reached ==="
grep -E "^=== Stage|^   .*done" "$LOGDIR/wider_aoi_chain.log" 2>/dev/null
echo
echo "=== build_stack tail (last 5) ==="
tail -5 "$LOGDIR/wider_aoi_build_stack.log" 2>/dev/null
echo
echo "=== tilt_fit tail (last 8) ==="
tail -8 "$LOGDIR/wider_aoi_tilt_fit.log" 2>/dev/null
echo
echo "=== run_melt tail (last 5) ==="
tail -5 "$LOGDIR/wider_aoi_run_melt.log" 2>/dev/null
echo
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>&1 | head -6
