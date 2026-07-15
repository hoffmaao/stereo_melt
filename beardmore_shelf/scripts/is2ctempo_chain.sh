#!/usr/bin/env bash
# Beardmore_Shelf IS2+CryoTEMPO alignment leg (A/B against the IS2-only
# production chain). Pulls CryoTEMPO Land Ice control, re-runs pc_align with
# the fused is2+cryotempo control into data/ASP_is2ctempo/, then stack ->
# tilt -> QC-regression -> path melt at the config.RES default (125 m),
# all tagged `is2ctempo` so nothing touches the IS2-only products.
#
# Mirrors the PIG production recipe (project_ctempo_ab_verdict: CryoTEMPO
# LMC supersedes raw SARIn L2 as the CS2 control base; unc<3 gate inside
# the library cache).
#
# Coverage caveat: the TDP_LI prefetch is indexed by SARIn-mode HDRs; parts
# of the AOI in SAR-mode territory will have thin ctempo control (strips
# fall back to IS2-only pc_align with a logged warning). Read the stage-1
# per-strip summary + stage-2 tally before judging the A/B.
#
# Stages (each logs separately; aborts on first non-zero exit):
#   1 cache_cryotempo  -> data/ASP/cryotempo_data/cryotempo_filtered_*.h5
#   2 align_strips     -> data/ASP_is2ctempo/asp_aligned/*.tif   (~16 h)
#   3 build_stack      -> processed/beardmore_shelf_stack_is2ctempo_*.nc
#   4 tilt_fit         -> processed/beardmore_shelf_stack_is2ctempo_tilt_corrected_*.nc
#   5 find_bad_epochs  -> log only (curation regression on the new alignment)
#   6 run_melt_path    -> processed/beardmore_shelf_melt_path_125m_is2ctempo_*.nc [A/B leg]
#
# Usage (disconnect-safe):
#   nohup setsid bash beardmore_shelf/scripts/is2ctempo_chain.sh \
#     > beardmore_shelf/logs/is2ctempo_chain.log 2>&1 < /dev/null &

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt
LOGDIR="$ROOT/beardmore_shelf/logs"
mkdir -p "$LOGDIR"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PROJ_DATA=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj
export BEARDMORE_SHELF_VELOCITY="${BEARDMORE_SHELF_VELOCITY:-measures}"

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

run_stage() {
    local name="$1"; local log="$LOGDIR/$2"; shift 2
    echo
    echo "[$(stamp)] stage $name start: $*"
    "$@" >"$log" 2>&1
    local rc=$?
    echo "[$(stamp)] stage $name exit code: $rc  (log: $log)"
    if [[ $rc -ne 0 ]]; then
        echo "[$(stamp)] $name failed -- aborting"
        exit 1
    fi
}

echo "[$(stamp)] === Beardmore_Shelf is2ctempo leg (cache -> align -> melt) start ==="
echo "[$(stamp)] BEARDMORE_SHELF_VELOCITY=$BEARDMORE_SHELF_VELOCITY"

run_stage "1/6 cache_cryotempo" cache_cryotempo.log \
    "$PY" -u -m beardmore_shelf.cache_cryotempo
run_stage "2/6 align_strips"    align_strips_is2ctempo.log \
    "$PY" -u -m beardmore_shelf.align_strips \
        --control is2+cs2 --cs2-source cryotempo \
        --asp-suffix _is2ctempo --parallel 8
run_stage "3/6 build_stack"     build_stack_is2ctempo.log \
    "$PY" -u -m beardmore_shelf.build_stack --asp-suffix _is2ctempo --tag is2ctempo
run_stage "4/6 tilt_fit"        tilt_fit_is2ctempo.log \
    "$PY" -u -m beardmore_shelf.tilt_fit --tag is2ctempo
run_stage "5/6 find_bad_epochs" find_bad_epochs_is2ctempo.log \
    "$PY" -u -m beardmore_shelf.scripts.find_bad_epochs --tag is2ctempo
run_stage "6/6 run_melt_path"   run_melt_path_is2ctempo.log \
    "$PY" -u -m beardmore_shelf.run_melt_path --tag is2ctempo

echo
echo "[$(stamp)] === Beardmore_Shelf is2ctempo leg done ==="
echo "A/B against the IS2-only leg: compare"
echo "  processed/beardmore_shelf_melt_path_125m_2019-01-01_2024-01-10.nc      (IS2-only)"
echo "  processed/beardmore_shelf_melt_path_125m_is2ctempo_2019-01-01_2024-01-10.nc"
echo "on median / IQR / GL-2km flux; review find_bad_epochs_is2ctempo.log for"
echo "curation regressions. Negative = melt (Shean convention)."
