#!/usr/bin/env bash
# Beardmore_Shelf Stage B -- melt rate map. Re-runs tilt_fit with the curated
# config.BAD_STRIPS dropped, then the PRODUCTION Lagrangian path solver. Run
# AFTER pasting the find_bad_epochs output (from full_chain.sh) into
# beardmore_shelf/config.py. (Safe to run with BAD_STRIPS still empty for a
# first, uncurated look.)
#
# Solver verdict (2026-07-02, see feedback_keep_lagrangian_path_solver): the
# production melt product is the Lagrangian PATH solver
# (run_melt_path -> lagrangian_melt_rate output=path pair_median pairs=all
# 1.5-2.5yr, PIG config). The old combined run_melt bundles an out-of-date Lagrangian config
# (seed_stride=2, mean aggregator) plus the retired linear-inverse call, so it
# is NOT in this chain. Use `(cd examples && python -m beardmore_shelf.run_melt)`
# by hand only for an Eulerian cross-check.
#
# Stages (each logs separately; aborts on first non-zero exit):
#   1 tilt_fit       -> processed/beardmore_shelf_stack_tilt_corrected_*.nc (curated)
#   2 run_melt_path  -> processed/beardmore_shelf_melt_path_*.nc            [PRODUCT]
#
# Melt-rate sign (Shean): negative = melt, positive = accretion.
#
# Usage (disconnect-safe):
#   nohup setsid bash examples/beardmore_shelf/scripts/run_solvers.sh \
#     > examples/beardmore_shelf/logs/run_solvers.log 2>&1 < /dev/null &
#
# Tunables (env): BEARDMORE_SHELF_VELOCITY (advection field, default "measures").
# Velocity is the static MEaSUREs phase map -- flag the mean-velocity
# flux-inflation caveat (velocity-representation-first-order) in write-ups.

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

echo "[$(stamp)] === Beardmore_Shelf Stage B (curated tilt -> path melt) start ==="
echo "[$(stamp)] BEARDMORE_SHELF_VELOCITY=$BEARDMORE_SHELF_VELOCITY"

run_stage "1/2 tilt_fit(curated)"  tilt_fit_curated.log  "$PY" -u -m beardmore_shelf.tilt_fit
run_stage "2/2 run_melt_path"      run_melt_path.log     "$PY" -u -m beardmore_shelf.run_melt_path

echo
echo "[$(stamp)] === Beardmore_Shelf Stage B done ==="
echo "Path melt map + figure in beardmore_shelf/figures/melt_path_*.png ;"
echo "array in beardmore_shelf/processed/beardmore_shelf_melt_path_*.nc."
echo "Sanity-check magnitudes vs the Ross-sector literature (e.g. Davison 2023"
echo "Ross_West) -- negative = melt in Shean convention."
