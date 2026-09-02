#!/usr/bin/env bash
# Beardmore re-run after the May 4 Shean-order tilt_fit refactor.
# The May 3 stack on disk was produced by the old v2 order
# (tide+IBE -> tilt LSQ -> geoid). The current beardmore/tilt_fit.py is
# fixed to apply tide -> IBE -> geoid -> tilt LSQ (Shean order; MDT skipped
# since Beardmore is at ~-84°S, south of DTU22's -79°S coverage limit).
#
#   stage 1: tilt_fit               -> processed/beardmore_stack_tilt_corrected_*.nc
#                                      processed/beardmore_tilt_params_*.nc
#   stage 2: compare_five_methods   -> results/beardmore_five_methods_*.nc
#                                      figures/five_methods_comparison.png
#
# Skips build_stack -- the raw stack beardmore_stack_2013-01-01_2023-03-01.nc
# is fine; only the tilt-corrected derivative was stale.
#
# Usage:
#   nohup bash examples/beardmore/scripts/rebuild_compare_chain.sh \
#     > examples/beardmore/logs/rebuild_compare_chain.log 2>&1 < /dev/null &

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt/examples
LOGDIR="$ROOT/beardmore/logs"
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

echo "[$(stamp)] === Beardmore rebuild-compare chain start ==="

run_stage "1/2 tilt_fit"             beardmore.tilt_fit              rebuild_tilt_fit_v3.log
run_stage "2/2 compare_five_methods" beardmore.compare_five_methods  compare_five_methods_v6.log

echo
echo "[$(stamp)] === Beardmore rebuild-compare chain done ==="
