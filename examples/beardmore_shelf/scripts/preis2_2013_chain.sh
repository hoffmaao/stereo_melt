#!/usr/bin/env bash
# Beardmore_Shelf pre-IS2 2013 orientation leg: fetch + coregister the
# WorldView strips around the sector's single IceBridge ATM flight
# (2013-11-18; the only airborne lidar this AOI ever got — LVIS is zero).
#
# Window = the ±90 d ATM pairing band 2013-08-20 .. 2014-02-17: 47 strips in
# the s2s041 index over the stack extent, 40 already in the shared pool,
# 7 to download. Control = CryoTEMPO (CS2 flies 2010→, HDR index on disk
# covers 2013) + ATM, no IS2 (didn't exist). Alignment lands in
# data/ASP_ctempoatm/ — orientation ONLY; folding the era into a stack /
# extending config.START_TIME is a separate decision.
#
# Stages (each logs separately; aborts on first non-zero exit):
#   1 fetch_strips     -> data/REMA/strips/ (missing 2013-era strips + quality files)
#   2 cache_cryotempo  -> data/ASP/cryotempo_data/  (2013-window strips)
#   3 cache_atm        -> data/ASP/atm_data/        (ILATM2 2013-11-18)
#   4 align_strips     -> data/ASP_ctempoatm/asp_aligned/   (~1 h at 8-way)
#
# Usage (disconnect-safe):
#   nohup setsid bash beardmore_shelf/scripts/preis2_2013_chain.sh \
#     > beardmore_shelf/logs/preis2_2013_chain.log 2>&1 < /dev/null &

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt/examples
LOGDIR="$ROOT/beardmore_shelf/logs"
mkdir -p "$LOGDIR"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PROJ_DATA=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj

# The 2013 era window (±90 d around the 2013-11-18 ATM flight).
export BEARDMORE_SHELF_FETCH_START=2013-08-20
export BEARDMORE_SHELF_FETCH_END=2014-02-17

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

run_stage() {
    local name="$1"; local log="$LOGDIR/$2"; shift 2
    echo
    echo "[$(stamp)] stage $name start: $*"
    "$@" >"$log" 2>&1
    local rc=$?
    echo "[$(stamp)] stage $name exit code: $rc  (log: $log)"
    if [[ $rc -ne 0 ]]; then
        echo "[$(stamp)] $name failed -- aborting"
        exit 1
    fi
}

echo "[$(stamp)] === Beardmore_Shelf pre-IS2 2013 orientation leg start ==="
echo "[$(stamp)] window $BEARDMORE_SHELF_FETCH_START .. $BEARDMORE_SHELF_FETCH_END"

run_stage "1/4 fetch_strips"    fetch_strips_2013.log    "$PY" -u -m beardmore_shelf.fetch_strips
run_stage "2/4 cache_cryotempo" cache_cryotempo_2013.log "$PY" -u -m beardmore_shelf.cache_cryotempo
run_stage "3/4 cache_atm"       cache_atm_2013.log       "$PY" -u -m beardmore_shelf.cache_atm
run_stage "4/4 align_strips"    align_strips_2013.log \
    "$PY" -u -m beardmore_shelf.align_strips \
        --control cs2 --cs2-source cryotempo --use-atm \
        --start "$BEARDMORE_SHELF_FETCH_START" --end "$BEARDMORE_SHELF_FETCH_END" \
        --asp-suffix _ctempoatm --parallel 8

echo
echo "[$(stamp)] === Beardmore_Shelf pre-IS2 2013 orientation leg done ==="
echo "Aligned 2013-era strips in beardmore_shelf/data/ASP_ctempoatm/asp_aligned/."
echo "Next decisions (separate): fold into an era-fused STRIP_SOURCES stack"
echo "(extend config.START_TIME pre-2019) and/or pull the 2012-11 tasking"
echo "burst (279 more strips, ctempo-only control)."
