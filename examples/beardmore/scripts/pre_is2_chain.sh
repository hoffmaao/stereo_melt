#!/usr/bin/env bash
# Pre-IS2 era acquisition + alignment chain for Beardmore.
#
#   stage A: cs2 granule prefetch (real .nc download for the new 2013-2018 window)
#   stage B: per-strip CS2 HDF5 caches for the new strips
#   stage C: ASP pc_align with --control cs2 over the new strips
#
# Each stage logs separately. Chain aborts on stage failure. cache_cryosat2
# and align_strips are both idempotent (skip already-completed work), so this
# can be re-launched safely if something interrupts.
#
# Usage:
#   ./examples/beardmore/scripts/pre_is2_chain.sh [parallel]
#
#   parallel = number of pc_align workers in stage C (default 8)

set -u
# SSH-disconnect immunity: ignore terminal hangups and stray SIGTERMs from a
# session leader teardown. The python children inherit these dispositions.
trap '' HUP TERM
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt/examples
LOGDIR="$ROOT/beardmore/logs"
mkdir -p "$LOGDIR"

CHAIN_LOG="$LOGDIR/pre_is2_chain.log"
exec >>"$CHAIN_LOG" 2>&1

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

PARALLEL="${1:-8}"
echo "[$(stamp)] pre_is2_chain start (parallel=$PARALLEL)"

cd "$ROOT"
export PYTHONPATH="$ROOT"

# Stage A: real CS2 prefetch for the extended 2013-2018 window. Existing
# 2018-10 onward .nc files are skipped on incremental cache check.
echo
echo "[$(stamp)] stage A start: CS2 granule prefetch (2013-01 .. 2018-12)"
STAGEA_LOG="$LOGDIR/pre_is2_cs2_prefetch.log"
"$PY" -u -m beardmore.cache_cryosat2 \
    --prefetch-only --workers 4 --heartbeat-every 500 \
    >"$STAGEA_LOG" 2>&1
RC=$?
echo "[$(stamp)] stage A exit code: $RC"
if [[ $RC -ne 0 ]]; then
    echo "[$(stamp)] CS2 prefetch failed — aborting chain"
    exit 1
fi
grep -A 8 "prefetch summary:" "$STAGEA_LOG" | tail -12

# Stage B: per-strip CS2 cache (h5) for any 2013-2018 strip that doesn't
# already have one. --no-prefetch skips ESA walk; idempotent on existing
# 2019-2023 caches.
echo
echo "[$(stamp)] stage B start: per-strip CS2 HDF5 caches"
STAGEB_LOG="$LOGDIR/pre_is2_cs2_cache_strips.log"
"$PY" -u -m beardmore.cache_cryosat2 --no-prefetch \
    >"$STAGEB_LOG" 2>&1
RC=$?
echo "[$(stamp)] stage B exit code: $RC"
if [[ $RC -ne 0 ]]; then
    echo "[$(stamp)] per-strip CS2 cache failed — aborting before align"
    exit 2
fi
tail -3 "$STAGEB_LOG"

# Stage C: ASP pc_align, cs2-only control, parallel batch over new strips.
# --asp-suffix _cs2 directs output to <STRIPS_DIR>/ASP_cs2/asp_aligned/, which
# already holds the 2019-2023 cs2-aligned outputs; align_strips' resumability
# means only the pre-IS2 strips actually run.
echo
echo "[$(stamp)] stage C start: align_strips --control cs2 --parallel $PARALLEL"
STAGEC_LOG="$LOGDIR/pre_is2_align_strips_cs2.log"
"$PY" -u -m beardmore.align_strips \
    --control cs2 --asp-suffix _cs2 --parallel "$PARALLEL" \
    >"$STAGEC_LOG" 2>&1
RC=$?
echo "[$(stamp)] stage C exit code: $RC"
tail -5 "$STAGEC_LOG"

echo
echo "[$(stamp)] pre_is2_chain done"
