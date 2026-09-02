#!/usr/bin/env bash
# Venable Stage B -- melt rate maps. Re-runs tilt_fit with the curated
# config.BAD_STRIPS dropped, then the solvers. Run AFTER pasting the
# find_bad_epochs output (from full_chain.sh) into venable/config.py.
# (Safe to run with BAD_STRIPS still empty for a first, uncurated look.)
#
# Stages (each logs separately; aborts on first non-zero exit):
#   1 tilt_fit           -> processed/venable_stack_tilt_corrected_*.nc (curated)
#   2 run_melt           -> Eulerian melt map      (Shean Eq. 10)        [PRODUCT]
#   3 run_pseudospectral -> Lagrangian melt map    (FFT pseudospectral)  [PRODUCT]
#   4 run_stationary     -> linear-inverse          (diagnostic, NOT a melt product)
#
# Melt-rate sign (Shean): negative = melt, positive = accretion.
# Outputs: arrays in venable/processed/*.nc, maps/figures in venable/figures/.
#
# Usage (disconnect-safe):
#   nohup setsid bash examples/venable/scripts/run_solvers.sh \
#     > examples/venable/logs/run_solvers.log 2>&1 < /dev/null &
#
# Tunables (env): VENABLE_VELOCITY (advection field, default "measures").

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt/examples
LOGDIR="$ROOT/venable/logs"
mkdir -p "$LOGDIR"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PROJ_DATA=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj
export VENABLE_VELOCITY="${VENABLE_VELOCITY:-measures}"

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

echo "[$(stamp)] === Venable Stage B (solvers -> melt maps) start ==="
echo "[$(stamp)] VENABLE_VELOCITY=$VENABLE_VELOCITY"

run_stage "1/4 tilt_fit(curated)"  tilt_fit_curated.log    "$PY" -u -m venable.tilt_fit
run_stage "2/4 run_melt"           run_melt.log            "$PY" -u -m venable.run_melt
run_stage "3/4 run_pseudospectral" run_pseudospectral.log  "$PY" -u -m venable.run_pseudospectral
run_stage "4/4 run_stationary"     run_stationary.log      "$PY" -u -m venable.run_stationary

echo
echo "[$(stamp)] === Venable Stage B done ==="
echo "Melt maps + figures in venable/figures/ ; arrays in venable/processed/."
echo "Eulerian (run_melt) + Lagrangian (run_pseudospectral) are the melt products;"
echo "run_stationary is diagnostic. Sanity-check magnitudes vs the Bellingshausen"
echo "literature (e.g. Davison 2023) -- negative = melt in Shean convention."
