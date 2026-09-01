#!/usr/bin/env bash
# Beardmore_Shelf: rebuild the stack at config.RES from the existing
# ASP-aligned DEMs (fetch/IS2/align are resolution-independent), then the
# corrections + production path melt. Written for the 2026-07-05 default-grid
# change 25 m -> 125 m; reusable whenever config.RES or the strip curation
# changes.
#
# Assumes config.BAD_STRIPS is already curated (per-dem_id rejection is
# resolution-independent); stage 3 find_bad_epochs is a REGRESSION check on
# the new grid — review its log, and if it flags NEW strips beyond the pasted
# set, paste and re-run this script (it will then be a fast re-tilt).
#
# Stages (each logs separately; aborts on first non-zero exit):
#   1 build_stack      -> processed/beardmore_shelf_stack_<window>.nc  [config.RES]
#   2 tilt_fit         -> processed/beardmore_shelf_stack_tilt_corrected_*.nc
#   3 find_bad_epochs  -> log only (curation regression)
#   4 run_melt_path    -> processed/beardmore_shelf_melt_path_*.nc     [PRODUCT]
#
# Melt-rate sign (Shean): negative = melt, positive = accretion.
#
# Usage (disconnect-safe):
#   nohup setsid bash beardmore_shelf/scripts/restack_solve.sh \
#     > beardmore_shelf/logs/restack_solve.log 2>&1 < /dev/null &
#
# Tunables (env): BEARDMORE_SHELF_VELOCITY (advection field, default "measures").

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt
LOGDIR="$ROOT/beardmore_shelf/logs"
mkdir -p "$LOGDIR"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PROJ_DATA=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj
export BEARDMORE_SHELF_VELOCITY="${BEARDMORE_SHELF_VELOCITY:-measures}"

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

echo "[$(stamp)] === Beardmore_Shelf restack (config.RES) -> path melt start ==="
echo "[$(stamp)] BEARDMORE_SHELF_VELOCITY=$BEARDMORE_SHELF_VELOCITY"

run_stage "1/4 build_stack"     build_stack_125m.log     "$PY" -u -m beardmore_shelf.build_stack
run_stage "2/4 tilt_fit"        tilt_fit_125m.log        "$PY" -u -m beardmore_shelf.tilt_fit
run_stage "3/4 find_bad_epochs" find_bad_epochs_125m.log "$PY" -u -m beardmore_shelf.scripts.find_bad_epochs
run_stage "4/4 run_melt_path"   run_melt_path_125m.log   "$PY" -u -m beardmore_shelf.run_melt_path

echo
echo "[$(stamp)] === Beardmore_Shelf restack -> path melt done ==="
echo "Review beardmore_shelf/logs/find_bad_epochs_125m.log (curation regression)."
echo "Path melt map + figure in beardmore_shelf/figures/ ;"
echo "array in beardmore_shelf/processed/beardmore_shelf_melt_path_*.nc."
echo "Sanity-check magnitudes vs the Ross-sector literature (Davison 2023"
echo "Ross_West) -- negative = melt in Shean convention."
