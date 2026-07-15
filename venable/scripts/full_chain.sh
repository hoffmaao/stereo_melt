#!/usr/bin/env bash
# Venable Stage A -- raw data to QC. Produces the tilt-corrected stack and the
# bad-strip candidates, then STOPS: melt rates need post-QC curation (paste the
# find_bad_epochs output into config.BAD_STRIPS, then run run_solvers.sh).
#
# Prerequisites (run once, before this chain):
#   python -m venable.cache_climate          # ERA5 MSL pressure (post-coreg IBE)
#   python scripts/compute_aoi.py venable     # finalize the optimal v15 AOI
#                                             # (strip-aware; run after a seed fetch)
#
# Stages (each logs separately; the chain aborts on the first non-zero exit):
#   1 fetch_strips     -> data/REMA/strips/<dem_id>.tif   (delta vs current AOI)
#   2 cache_icesat2    -> venable/data/ASP/icesat2_data/* (IS2 ATL06 control)
#   3 align_strips     -> venable/data/ASP/asp_aligned/*  (ASP pc_align; long pole)
#   4 build_stack      -> processed/venable_stack_<window>.nc  (25 m EPSG:3031)
#   5 tilt_fit         -> processed/venable_stack_tilt_corrected_*.nc (first pass)
#   6 find_bad_epochs  -> stdout: BAD_STRIPS candidates (review before solvers)
#
# Usage (disconnect-safe):
#   nohup setsid bash venable/scripts/full_chain.sh \
#     > venable/logs/full_chain.log 2>&1 < /dev/null &
#
# Tunables (env): VENABLE_ALIGN_PARALLEL (align workers, default 8).

set -u
trap '' HUP TERM

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt
LOGDIR="$ROOT/venable/logs"
ASP_ROOT="$ROOT/venable/data/ASP"
PARALLEL="${VENABLE_ALIGN_PARALLEL:-8}"
mkdir -p "$LOGDIR"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PROJ_DATA=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj

stamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -d ' '; }

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

echo "[$(stamp)] === Venable Stage A (data -> QC) start ==="
echo "[$(stamp)] /wd2 free: $(free_gb /wd2)  ASP_ROOT=$ASP_ROOT"

run_stage "1/6 fetch_strips"    fetch_strips.log    "$PY" -u -m venable.fetch_strips
run_stage "2/6 cache_icesat2"   cache_icesat2.log   "$PY" -u -m venable.cache_icesat2

# Disk preflight before the long, space-hungry align. align_strips self-guards at
# a 100 GB floor and aborts early if breached; if the projected need is tight,
# point $ASP_ROOT at /wd1 via symlink before launching this chain.
echo "[$(stamp)] pre-align free at $ROOT: $(free_gb "$ROOT")  (align workers: $PARALLEL)"
run_stage "3/6 align_strips"    align_strips.log    "$PY" -u -m venable.align_strips --parallel "$PARALLEL"

run_stage "4/6 build_stack"     build_stack.log     "$PY" -u -m venable.build_stack
run_stage "5/6 tilt_fit"        tilt_fit.log        "$PY" -u -m venable.tilt_fit
run_stage "6/6 find_bad_epochs" find_bad_epochs.log "$PY" -u -m venable.scripts.find_bad_epochs

echo
echo "[$(stamp)] === Venable Stage A done ==="
echo "NEXT (manual QC checkpoint):"
echo "  1. Review venable/logs/find_bad_epochs.log"
echo "  2. Paste the printed BAD_STRIPS tuple into venable/config.py"
echo "     (leave BAD_EPOCHS empty -- strip-level rejection is Shean-faithful)"
echo "  3. Launch the solvers:"
echo "       nohup setsid bash venable/scripts/run_solvers.sh \\"
echo "         > venable/logs/run_solvers.log 2>&1 < /dev/null &"
