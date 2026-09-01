#!/usr/bin/env bash
# Dotson + Crosson tilt-and-compare chain. Scaffolded May 4 mirroring the
# Nansen / PIG framework: Shean-order tilt_fit (tide -> IBE -> MDT ->
# geoid -> tilt LSQ) then 5-method melt comparison + Davison 2023 overlay.
#
#   stage 1: tilt_fit               -> processed/dotson_crosson_stack_tilt_corrected_*.nc
#                                      processed/dotson_crosson_tilt_params_*.nc
#   stage 2: compare_five_methods   -> results/dotson_crosson_five_methods_*.nc
#                                      figures/five_methods_comparison.png
#
# Prereq: dotson_crosson_stack_<window>.nc must already exist in processed/.
# Run dotson_crosson.align_strips and dotson_crosson.build_stack first.
#
# Usage:
#   nohup bash dotson_crosson/scripts/rebuild_compare_chain.sh \
#     > dotson_crosson/logs/rebuild_compare_chain.log 2>&1 < /dev/null &

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt
LOGDIR="$ROOT/dotson_crosson/logs"
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

echo "[$(stamp)] === Dotson + Crosson rebuild-compare chain start ==="

run_stage "1/2 tilt_fit"             dotson_crosson.tilt_fit              rebuild_tilt_fit_v1.log
run_stage "2/2 compare_five_methods" dotson_crosson.compare_five_methods  compare_five_methods_v1.log

echo
echo "[$(stamp)] === Dotson + Crosson rebuild-compare chain done ==="
