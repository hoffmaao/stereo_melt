#!/usr/bin/env bash
# PIG re-run after killing the stale May 3 rebuild_chain (which produced a
# pig_stack_tilt_corrected_*.nc with v2 order: tilt LSQ before geoid and
# MDT entirely skipped). The current pig/tilt_fit.py is fixed to apply
# tide -> IBE -> MDT -> geoid -> tilt LSQ (Shean order, matches Nansen).
#
#   stage 1: tilt_fit               -> processed/pig_stack_tilt_corrected_*.nc
#                                      processed/pig_tilt_params_*.nc
#   stage 2: compare_five_methods   -> results/pig_five_methods_*.nc
#                                      figures/five_methods_comparison.png
#
# Skips build_stack -- the raw stack pig_stack_2019-01-01_2024-01-10.nc
# is fine; only the tilt-corrected derivative was stale.
#
# Usage:
#   nohup bash pig/scripts/rebuild_compare_chain.sh \
#     > pig/logs/rebuild_compare_chain.log 2>&1 < /dev/null &

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

echo "[$(stamp)] === PIG rebuild-compare chain start ==="

run_stage "1/2 tilt_fit"             pig.tilt_fit              rebuild_tilt_fit_v2.log
run_stage "2/2 compare_five_methods" pig.compare_five_methods  compare_five_methods_v1.log

echo
echo "[$(stamp)] === PIG rebuild-compare chain done ==="
