#!/usr/bin/env bash
# Nansen Ez=0.1 re-run chain: tilt_fit -> diagnose_dhdt -> run_melt -> compare_five_methods.
# Tests whether tightening the IS2 Ez prior 0.3 -> 0.1 m closes the +0.07 m/yr static-control
# bias diagnosed on 2026-05-05 (project_nansen_dhdt_bias.md).
#
# Usage:
#   nohup bash nansen/scripts/rerun_ez_0p1_chain.sh \
#     > nansen/logs/chain_ez_0p1.log 2>&1 < /dev/null &
set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt
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

echo "[$(stamp)] === Nansen Ez=0.1 re-run start ==="

run_stage "1/4 tilt_fit"            nansen.tilt_fit            tilt_fit_ez_0p1.log
run_stage "2/4 diagnose_dhdt"       nansen.diagnose_dhdt       diagnose_dhdt_ez_0p1.log
run_stage "3/4 run_melt"            nansen.run_melt            run_melt_ez_0p1.log
run_stage "4/4 compare_five_methods" nansen.compare_five_methods compare_five_methods_ez_0p1.log

echo
echo "[$(stamp)] === Nansen Ez=0.1 re-run done ==="
