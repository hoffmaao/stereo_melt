#!/usr/bin/env bash
# Beardmore_Shelf pre-IS2 FULL cohort (user asks 2026-07-05: "add strips with
# the CryoTEMPO data where we can" + "what about CryoTEMPO before IS2,
# 2013-2018"): fetch + coregister every s2s041 strip from the Nov-2012
# tasking burst through the IS2 handover — 702 index hits over the stack
# extent (2012-11-18 .. 2019-01-01: 346 in the ±1 yr ATM band incl. the
# 279-strip Nov-2012 burst, + 356 in the 2015-2018 gap era; 363 already on
# disk, ~339 to download). Closing the gap makes the record CONTINUOUS
# 2012→2024 — the Lagrangian 1.5-2.5 yr pair band needs the temporal
# neighbors. Control = CryoTEMPO (the only altimeter of the era) + the
# 2013-11-18 ATM flight where the ±365 d / 10 m displacement budget admits
# it + BedMachine rock GCPs (fixed 2026-07-05 — the earlier 30-strip 2013
# leg was rock-free and was wiped for a uniform redo into the same root).
#
# Strips whose footprints never reach grounded ice, or whose CryoTEMPO+ATM
# caches are empty, are skipped/refused — grounded-control criterion
# (floating surfaces move with tide/IBE/melt between pass and strip; not
# usable as absolute control). "Where we can" = where grounded control exists.
#
# --parallel 4 (not 8): coexists with the running is2ctempo-era 8-way align.
#
# Stages (abort on first non-zero exit):
#   1 fetch_strips     -> data/REMA/strips/            (~265 downloads, ~30-60 GB)
#   2 cache_cryotempo  -> data/ASP/cryotempo_data/
#   3 cache_atm        -> data/ASP/atm_data/           (--time-window 365)
#   4 align_strips     -> data/ASP_ctempoatm/asp_aligned/
#
# Usage (disconnect-safe):
#   nohup setsid bash beardmore_shelf/scripts/preis2_full_chain.sh \
#     > beardmore_shelf/logs/preis2_full_chain.log 2>&1 < /dev/null &

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

# Full pre-IS2 era: Nov-2012 burst through the IS2 handover.
export BEARDMORE_SHELF_FETCH_START=2012-11-18
export BEARDMORE_SHELF_FETCH_END=2019-01-01

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

echo "[$(stamp)] === Beardmore_Shelf pre-IS2 FULL cohort (fetch -> ctempo+atm+rock align) start ==="
echo "[$(stamp)] window $BEARDMORE_SHELF_FETCH_START .. $BEARDMORE_SHELF_FETCH_END"

run_stage "1/4 fetch_strips"    fetch_strips_preis2full.log    "$PY" -u -m beardmore_shelf.fetch_strips
run_stage "2/4 cache_cryotempo" cache_cryotempo_preis2full.log "$PY" -u -m beardmore_shelf.cache_cryotempo
run_stage "3/4 cache_atm"       cache_atm_preis2full.log \
    "$PY" -u -m beardmore_shelf.cache_atm --time-window 365
run_stage "4/4 align_strips"    align_strips_preis2full.log \
    "$PY" -u -m beardmore_shelf.align_strips \
        --control cs2 --cs2-source cryotempo --use-atm \
        --start "$BEARDMORE_SHELF_FETCH_START" --end "$BEARDMORE_SHELF_FETCH_END" \
        --asp-suffix _ctempoatm --parallel 4

echo
echo "[$(stamp)] === Beardmore_Shelf pre-IS2 FULL cohort done ==="
echo "Aligned pre-IS2 strips in beardmore_shelf/data/ASP_ctempoatm/asp_aligned/."
echo "Next decision: era-fused STRIP_SOURCES stack + pre-2019 START_TIME"
echo "(mixed-era Ez: loosen the CS2-era prior per feedback_ez_per_gcp_source)."
