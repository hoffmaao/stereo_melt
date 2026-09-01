#!/usr/bin/env bash
# Combined 2012-2024 stack lever — GATE 1 only (build + coverage diagnostic).
# project_session_resume_2026_07_07 lever (1): fold the pre-IS2 CryoTEMPO/ATM
# epochs into the stack so the per-epoch tilt LSQ is better conditioned where
# the RED accretion stripe lives.
#
# This wrapper:
#   1. WAITS for the pre-IS2 `align_strips --asp-suffix _ctempoatm` job to finish.
#   2. Builds the fused 2012-2024 stack (BEARDMORE_SHELF_SOURCES=combined,
#      BEARDMORE_SHELF_START=2012-11-18 -> beardmore_shelf_stack_2012-11-18_*.nc,
#      a NEW file, does not clobber the IS2-era baseline).
#   3. Runs the epoch-count diagnostic (GATE 1): does the RED_accr band's
#      baseline 32.8% under-det (<5 epochs) shrink?
#
# It STOPS before tilt_fit on purpose. The combined re-tilt is a multi-hour
# CPU LSQ (~455 epochs); it is gated on the epoch-count evidence + a human look
# (the coverage-conditioning hypothesis is weaker than first framed: the RED
# band already has ZERO <3-epoch cells at baseline).
set -uo pipefail

ROOT=/wd2/projects/stereo_melt
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ALIGN_LOG="$ROOT/beardmore_shelf/logs/align_strips_preis2full.log"
CTEMPO_DIR="$ROOT/beardmore_shelf/data/ASP_ctempoatm/asp_aligned"
cd "$ROOT"

# Combined mode must be set BEFORE python imports beardmore_shelf.config.
export BEARDMORE_SHELF_SOURCES=combined
export BEARDMORE_SHELF_START=2012-11-18
# BEARDMORE_SHELF_END stays the config default (2024-01-10).

echo "[combined_chain] $(date) waiting for pre-IS2 ctempoatm align to finish..."
while pgrep -f "beardmore_shelf.align_strips" >/dev/null 2>&1; do
  last=$(grep -aoE "\[[0-9]+/[0-9]+\]" "$ALIGN_LOG" 2>/dev/null | tail -1)
  echo "[combined_chain] $(date +%H:%M) align still running, progress ${last:-?}"
  sleep 180
done
echo "[combined_chain] $(date) align done (no align_strips python running)."

NDEM=$(ls -1 "$CTEMPO_DIR"/*-trans_reference-DEM.tif 2>/dev/null | wc -l)
echo "[combined_chain] ctempoatm aligned DEMs on disk: $NDEM"
if [ "$NDEM" -lt 300 ]; then
  echo "[combined_chain] ABORT: only $NDEM aligned DEMs (<300 expected); align may have died. Not building."
  exit 1
fi

echo "[combined_chain] ===================================================="
echo "[combined_chain] $(date) BUILD combined stack (SOURCES=combined START=2012-11-18)"
echo "[combined_chain] ===================================================="
$PY -m beardmore_shelf.build_stack
rc=$?
if [ $rc -ne 0 ]; then echo "[combined_chain] build_stack FAILED rc=$rc"; exit $rc; fi

echo "[combined_chain] ===================================================="
echo "[combined_chain] $(date) GATE 1: epoch-count diagnostic on combined stack"
echo "[combined_chain] ===================================================="
$PY -m beardmore_shelf.diag_epoch_count --res 125
rc=$?
if [ $rc -ne 0 ]; then echo "[combined_chain] diag_epoch_count FAILED rc=$rc"; exit $rc; fi

echo "[combined_chain] ===================================================="
echo "[combined_chain] $(date) DONE (gate 1). Baseline to beat:"
echo "[combined_chain]   whole shelf: thin(<3)=5.6%  under-det(<5)=10.2%"
echo "[combined_chain]   RED_accr band: under-det(<5)=32.8% (median 5)"
echo "[combined_chain] NEXT (human/gated): if RED under-det shrinks materially,"
echo "[combined_chain]   launch combined tilt_fit on CPU:"
echo "[combined_chain]   BEARDMORE_SHELF_SOURCES=combined BEARDMORE_SHELF_START=2012-11-18 \\"
echo "[combined_chain]   STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=14 \$PY -m beardmore_shelf.tilt_fit"
echo "[combined_chain] ===================================================="
