#!/usr/bin/env bash
# Chains the rest of the CS2 acquisition for Beardmore:
#   stage 1 (already running): dry-run prefetch — HDR pull + per-strip match counts
#   stage 2: real prefetch     — download the matched .nc files
#   stage 3: per-strip cache   — read, filter, write cs2_filtered_*.h5 caches
#
# Each stage's log is separate. The chain aborts if a stage fails.
#
# Usage:
#   ./beardmore/scripts/cs2_chain.sh <waiting_pid>
#
# <waiting_pid> is the dry-run prefetch PID we wait on before starting
# stage 2. Pass 0 to skip waiting (e.g., if dry-run already finished).

set -u
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt
LOGDIR="$ROOT/beardmore/logs"
mkdir -p "$LOGDIR"

CHAIN_LOG="$LOGDIR/cs2_chain.log"
exec >>"$CHAIN_LOG" 2>&1

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

WAIT_PID="${1:-0}"
echo "[$(stamp)] cs2_chain start, waiting on pid=$WAIT_PID"

# Stage 1: wait for dry-run to finish.
if [[ "$WAIT_PID" != "0" ]]; then
    if [[ -d "/proc/$WAIT_PID" ]]; then
        # tail --pid waits for a non-child PID and exits when that PID does.
        tail --pid="$WAIT_PID" -f /dev/null
    fi
    # Sanity: did the dry-run reach a "prefetch summary"? If not, abort.
    if ! grep -q "prefetch summary:" "$LOGDIR/cs2_prefetch_dryrun.log"; then
        echo "[$(stamp)] dry-run did NOT print 'prefetch summary:' — aborting chain"
        exit 11
    fi
    echo "[$(stamp)] stage 1 (dry-run) finished — last 12 lines of summary:"
    grep -A 8 "prefetch summary:" "$LOGDIR/cs2_prefetch_dryrun.log" | tail -12
fi

# Stage 2: real prefetch (no --prefetch-dry-run). HDRs are now cached so
# this is bandwidth-bound on the matched .nc set.
echo
echo "[$(stamp)] stage 2 start: real .nc prefetch"
STAGE2_LOG="$LOGDIR/cs2_prefetch_real.log"
"$PY" -u -m beardmore.cache_cryosat2 \
    --prefetch-only --workers 4 --heartbeat-every 500 \
    >"$STAGE2_LOG" 2>&1
RC=$?
echo "[$(stamp)] stage 2 exit code: $RC"
if [[ $RC -ne 0 ]]; then
    echo "[$(stamp)] real prefetch failed — aborting before per-strip cache"
    exit 22
fi
grep -A 8 "prefetch summary:" "$STAGE2_LOG" | tail -12

# Stage 3: per-strip CS2 cache build (cache_one for each in-window strip).
# --no-prefetch keeps stage 3 from re-walking ESA when it already has data.
echo
echo "[$(stamp)] stage 3 start: per-strip CS2 HDF5 caches"
STAGE3_LOG="$LOGDIR/cs2_cache_strips.log"
"$PY" -u -m beardmore.cache_cryosat2 --no-prefetch \
    >"$STAGE3_LOG" 2>&1
RC=$?
echo "[$(stamp)] stage 3 exit code: $RC"
tail -3 "$STAGE3_LOG"

echo
echo "[$(stamp)] cs2_chain done"
