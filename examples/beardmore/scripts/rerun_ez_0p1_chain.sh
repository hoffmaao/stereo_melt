#!/usr/bin/env bash
# Beardmore mixed-GCP (CS2 + IS2) re-run with Ez=0.1 prior + standardized 6-panel set.
# Stages:
#   1. tilt_fit  -> processed/beardmore_stack_tilt_corrected_*.nc + beardmore_tilt_params_*.nc
#                   (per-epoch Ez: 0.1 for IS2-era epochs, 2.0 for pre-IS2 CS2-era)
#   2. run_melt  -> results/beardmore_melt_*.nc + figures/melt_comparison.png
#                   (standardized 6-panel set: Eulerian, Lagrangian, dh/dt FFT,
#                    dh/dt DCT, D-masked CG DCT λ=1e-1 L=H_ref/2, Davison)
#   3. compare_five_methods -> results/beardmore_five_methods_*.nc +
#                              figures/five_methods_comparison.png  (same 6-panel set)
#
# Usage:
#   nohup bash beardmore/scripts/rerun_ez_0p1_chain.sh \
#     > beardmore/logs/chain_ez_0p1.log 2>&1 < /dev/null &

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt
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

echo "[$(stamp)] === Beardmore mixed-GCP Ez=0.1 re-run start ==="

run_stage "1/3 tilt_fit"             beardmore.tilt_fit             tilt_fit_ez_0p1.log
run_stage "2/3 run_melt"             beardmore.run_melt             run_melt_ez_0p1.log
run_stage "3/3 compare_five_methods" beardmore.compare_five_methods compare_five_methods_ez_0p1.log

echo
echo "[$(stamp)] === Beardmore mixed-GCP Ez=0.1 re-run done ==="
