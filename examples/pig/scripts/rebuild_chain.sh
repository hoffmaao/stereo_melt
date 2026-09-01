#!/usr/bin/env bash
# PIG rebuild chain: stack + tilt + 3 solvers, post-align.
#
#   stage 1: build_stack          -> processed/pig_stack_<window>.nc
#   stage 2: tilt_fit             -> processed/pig_stack_tilt_corrected_*.nc
#                                    processed/pig_tilt_params_*.nc
#   stage 3: run_melt             -> results/pig_melt_<window>.nc
#   stage 4: run_stationary       -> results/pig_stationary_<window>.nc
#   stage 5: run_pseudospectral   -> results/pig_pseudospectral_<window>.nc
#
# Each stage logs separately. Chain aborts on first non-zero exit.
#
# Usage:
#   nohup bash pig/scripts/rebuild_chain.sh \
#     > pig/logs/rebuild_chain.log 2>&1 < /dev/null &

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt/examples
LOGDIR="$ROOT/pig/logs"
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

echo "[$(stamp)] === PIG rebuild chain start ==="

run_stage "1/5 build_stack"        pig.build_stack        rebuild_build_stack.log
run_stage "2/5 tilt_fit"           pig.tilt_fit           rebuild_tilt_fit.log
run_stage "3/5 run_melt"           pig.run_melt           rebuild_run_melt.log
run_stage "4/5 run_stationary"     pig.run_stationary     rebuild_run_stationary.log
run_stage "5/5 run_pseudospectral" pig.run_pseudospectral rebuild_run_pseudospectral.log

echo
echo "[$(stamp)] === PIG rebuild chain done ==="
