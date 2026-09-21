#!/usr/bin/env bash
# McMurdo full pipeline chain: fetch -> align -> stack -> tilt -> find_bad_epochs.
# Solvers are intentionally NOT chained -- they need post-find_bad_epochs
# curation (populate config.BAD_EPOCHS, re-run tilt_fit) before melt-rate
# results are trustworthy.
#
#   stage 1: fetch_strips         -> data/REMA/strips/<dem_id>.tif (~40-70 GB)
#   stage 2: cache_icesat2        -> mcmurdo/data/ASP/icesat2_data/*.csv
#   stage 3: align_strips         -> mcmurdo/data/ASP/asp_aligned/* (~750 GB)
#   stage 4: build_stack          -> processed/mcmurdo_stack_<window>.nc
#   stage 5: tilt_fit             -> processed/mcmurdo_stack_tilt_corrected_*.nc
#                                    processed/mcmurdo_tilt_params_*.nc
#   stage 6: find_bad_epochs      -> stdout (review before re-run + solvers)
#
# Each stage logs separately. Chain aborts on first non-zero exit.
#
# Usage:
#   nohup bash examples/mcmurdo/scripts/full_chain.sh \
#     > examples/mcmurdo/logs/full_chain.log 2>&1 < /dev/null &

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt/examples
LOGDIR="$ROOT/mcmurdo/logs"
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

echo "[$(stamp)] === McMurdo full pipeline chain start ==="

run_stage "1/6 fetch_strips"     mcmurdo.fetch_strips           fetch_strips.log
run_stage "2/6 cache_icesat2"    mcmurdo.cache_icesat2          cache_icesat2.log
run_stage "3/6 align_strips"     mcmurdo.align_strips           align_strips.log
run_stage "4/6 build_stack"      mcmurdo.build_stack            build_stack.log
run_stage "5/6 tilt_fit"         mcmurdo.tilt_fit               tilt_fit.log
run_stage "6/6 find_bad_epochs"  mcmurdo.scripts.find_bad_epochs find_bad_epochs.log

echo
echo "[$(stamp)] === McMurdo chain done ==="
echo "Next: review mcmurdo/logs/find_bad_epochs.log, populate"
echo "mcmurdo.config.BAD_EPOCHS, re-run tilt_fit, then run the solvers"
echo "(run_melt, run_melt_path)."
