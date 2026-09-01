#!/usr/bin/env bash
# Watcher: wait for pig.tilt_fit (PID 27776) to finish cleanly, then chain pig.run_melt.
# Disconnect-safe: launched under setsid nohup; run_melt itself relaunched the same way.
set -u

TILT_PID=27776
TILT_LOG=/wd2/projects/stereo_melt/examples/pig/logs/tilt_fit_250m_is2cs2atmlvis_2026_06_03.log
OUT_NC=/wd2/projects/stereo_melt/examples/pig/processed/pig_stack_250m_tilt_corrected_2010-01-01_2024-01-10.nc
REF_EPOCH=1780538019            # "now" baseline; a fresh tilt_corrected write must beat this
WATCH_LOG=/wd2/projects/stereo_melt/examples/pig/logs/chain_run_melt_2026_06_03.watch.log
RUN_LOG=/wd2/projects/stereo_melt/examples/pig/logs/run_melt_250m_is2cs2atmlvis_2026_06_03.log
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] watcher up, waiting on tilt_fit PID $TILT_PID" >> "$WATCH_LOG"

# 1) Wait for tilt_fit to exit.
while kill -0 "$TILT_PID" 2>/dev/null; do
    sleep 120
done
echo "[$(stamp)] tilt_fit PID $TILT_PID has exited; verifying success" >> "$WATCH_LOG"

# Let the FS settle (final netCDF flush / figure write).
sleep 20

# 2) Verify clean completion: QC-figure success line AND a freshly-rewritten output.
OK_LOG=0
grep -q "tilt_fit_qc" "$TILT_LOG" && OK_LOG=1

OK_NC=0
if [ -f "$OUT_NC" ]; then
    NC_MTIME=$(stat -c '%Y' "$OUT_NC")
    if [ "$NC_MTIME" -gt "$REF_EPOCH" ]; then OK_NC=1; fi
fi

if [ "$OK_LOG" -ne 1 ] || [ "$OK_NC" -ne 1 ]; then
    echo "[$(stamp)] FAILED gate: OK_LOG=$OK_LOG OK_NC=$OK_NC -> NOT launching run_melt" >> "$WATCH_LOG"
    echo "[$(stamp)]   inspect tail of $TILT_LOG" >> "$WATCH_LOG"
    exit 1
fi

echo "[$(stamp)] tilt_fit clean (OK_LOG=$OK_LOG OK_NC=$OK_NC); launching run_melt -> $RUN_LOG" >> "$WATCH_LOG"

# 3) Launch run_melt, same proven config as the prior 250 m wider run (numpy/CPU backend).
cd /wd2/projects/stereo_melt/examples
PROJ_DATA=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj \
PROJ_LIB=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj \
STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \
setsid nohup "$PY" -u -m pig.run_melt --res 250 </dev/null > "$RUN_LOG" 2>&1 &
RM_PID=$!
echo "[$(stamp)] run_melt launched, PID $RM_PID" >> "$WATCH_LOG"
