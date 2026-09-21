#!/usr/bin/env bash
# Nansen re-run with 2 km asymmetric grounded-ice erosion on the static-control
# polygon (rock outcrops preserved). Diagnoses whether the +0.066 m/yr static-
# control bias closes when BedMachine grounding-zone slivers are pushed out of
# the LSQ anchor set. Same Ez=0.1 prior as previous rerun.
#
# Usage:
#   nohup bash examples/nansen/scripts/rerun_erode2km_chain.sh \
#     > examples/nansen/logs/chain_erode2km.log 2>&1 < /dev/null &

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt/examples
LOGDIR="$ROOT/nansen/logs"
mkdir -p "$LOGDIR"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

run_stage() {
    local name="$1"; local module="$2"; local log="$LOGDIR/$3"
    echo
    echo "[$(stamp)] stage $name start: $module"
    "$PY" -u -m "$module" >"$log" 2>&1
    local rc=$?
    echo "[$(stamp)] stage $name exit code: $rc  (log: $log)"
    if [[ $rc -ne 0 ]]; then
        echo "[$(stamp)] $name failed -- aborting chain"
        exit 1
    fi
}

echo "[$(stamp)] === Nansen erode-2km re-run start ==="

run_stage "1/3 tilt_fit"            nansen.tilt_fit            tilt_fit_erode2km.log
run_stage "2/3 diagnose_dhdt"       nansen.diagnose_dhdt       diagnose_dhdt_erode2km.log
run_stage "3/3 run_melt"            nansen.run_melt            run_melt_erode2km.log

echo
echo "[$(stamp)] === Nansen erode-2km re-run done ==="
