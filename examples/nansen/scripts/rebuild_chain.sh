#!/usr/bin/env bash
# Nansen rebuild chain: regenerate the stack and all solver outputs after
# the 2026-05-01 re-align (15 strips refreshed) and Shean-gate quarantine
# (3 silent-failure strips moved to bad_align/).
#
#   stage 1: build_stack          -> processed/nansen_stack_<window>.nc
#   stage 2: tilt_fit             -> processed/nansen_stack_tilt_corrected_*.nc
#                                    processed/nansen_tilt_params_*.nc
#   stage 3: run_melt             -> results/nansen_melt_<window>.nc
#
# Each stage logs separately. Chain aborts on first non-zero exit.
#
# Usage:
#   nohup bash examples/nansen/scripts/rebuild_chain.sh \
#     > examples/nansen/logs/rebuild_chain.log 2>&1 < /dev/null &

set -u
# SSH-disconnect immunity.
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

echo "[$(stamp)] === Nansen rebuild chain start ==="

run_stage "1/3 build_stack"        nansen.build_stack        rebuild_build_stack.log
run_stage "2/3 tilt_fit"           nansen.tilt_fit           rebuild_tilt_fit.log
run_stage "3/3 run_melt"           nansen.run_melt           rebuild_run_melt.log

echo
echo "[$(stamp)] === Nansen rebuild chain done ==="
