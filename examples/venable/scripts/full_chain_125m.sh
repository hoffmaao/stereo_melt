#!/usr/bin/env bash
# Venable 125 m re-run -- build_stack onward only (fetch/IS2/align already done
# by the 25 m Stage A; aligned DEMs are resolution-independent). Produces the
# 125 m tilt-corrected stack + bad-strip candidates, then STOPS at the QC
# checkpoint (paste BAD_STRIPS into config.py before any solver run).
#
# Stages:
#   1 build_stack --res 125  -> processed/venable_stack_125m_<window>.nc
#   2 tilt_fit    --res 125  -> processed/venable_stack_125m_tilt_corrected_*.nc
#   3 find_bad_epochs --res 125 -> stdout: BAD_STRIPS candidates
#
# Usage (disconnect-safe):
#   nohup setsid bash examples/venable/scripts/full_chain_125m.sh \
#     > examples/venable/logs/full_chain_125m.log 2>&1 < /dev/null &

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

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

run_stage() {
    local name="$1"; local log="$LOGDIR/$2"; shift 2
    echo
    echo "[$(stamp)] stage $name start: $*"
    "$@" >"$log" 2>&1
    local rc=$?
    echo "[$(stamp)] stage $name exit code: $rc  (log: $log)"
    if [[ $rc -ne 0 ]]; then
        echo "[$(stamp)] $name failed -- aborting chain"
        exit 1
    fi
}

echo "[$(stamp)] === Venable 125 m re-run (build_stack -> QC) start ==="

run_stage "1/3 build_stack"     build_stack_125m.log     "$PY" -u -m venable.build_stack --res 125
run_stage "2/3 tilt_fit"        tilt_fit_125m.log        "$PY" -u -m venable.tilt_fit --res 125
run_stage "3/3 find_bad_epochs" find_bad_epochs_125m.log "$PY" -u -m venable.scripts.find_bad_epochs --res 125

echo
echo "[$(stamp)] === Venable 125 m re-run done (QC checkpoint) ==="
echo "NEXT: review venable/logs/find_bad_epochs_125m.log, paste BAD_STRIPS into"
echo "venable/config.py, then run solvers at --res 125. Production melt = the"
echo "standard Lagrangian path solver (A/B verdict 2026-07-02: kept; parcel-LSQ"
echo "remains opt-in via PIG_PARCEL_LSQ-style env, diagnostic only)."
