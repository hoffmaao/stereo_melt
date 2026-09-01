#!/usr/bin/env bash
# Balanced full re-align of the PIG CryoTEMPO timeseries (committed 2026-06-18).
#
# WHY: production aligns never enabled inverse-variance balancing, so control
# enters pc_align at raw row counts. Measured cryotempo:is2 ≈ 13:1 in the
# IS2-era ICP cost (is2-only strips ~23.4k pts vs is2+cryotempo ~324k → cryotempo
# ~300k ≈ 13× is2). pc_align weights by sum-of-sq residuals = row count, so
# the noisier cryotempo (MAD 0.82 m) drives alignment over the 8×-precise IS2.
# Densetie does NOT escape it: its Stage-2 Δz = median(control-aligned) over the
# pooled CSV (asp.py:582), ~93% cryotempo → median is cryotempo's.
#
# FIX: --inv-variance-balance subsamples per source ∝ 1/σ² (ICP_SIGMA_PER_SOURCE_M):
# cryotempo σ=0.82 → ~74 rows, is2 σ=0.1 → 5000 rows (equal ICP contribution).
# Writes balanced combined_reference_*.csv that a later densetie pass inherits
# for free (no densetie code change) — its Stage-2 median flips to IS2-dominated.
#
# Writes NEW roots so the current unbalanced product stays intact for A/B.
# Launch AFTER pig.cache_cryotempo (unc<3, pid 5560) finishes — reads the fresh
# data/ASP/cryotempo_data with the recovered strips folded in.
#
# Usage:  setsid bash pig/run_balanced_realign.sh [PARALLEL] &   # default 8
set -u
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
cd /wd2/projects/stereo_melt || exit 1
LOGDIR=pig/logs
P=${1:-8}   # parallel ASP workers

echo "=== balanced re-align START $(date) | parallel=$P ==="

# IS2 era first (is2+cryotempo+atm+lvis) → ASP_is2ctempoatmlvisbal.
# This is where the 13:1 overweight bites (IS2 coexists), so the highest-value
# balanced product lands first.
$PY -u -m pig.align_strips --control is2cs2 --cs2-source cryotempo --use-atm --use-lvis \
    --inv-variance-balance --asp-suffix _is2ctempoatmlvisbal \
    --start 2018-10-13 --end 2024-01-10 --parallel "$P" \
    > "$LOGDIR/align_is2ctempoatmlvisbal_IS2era.log" 2>&1
echo "=== IS2 era done $(date) (exit $?) → $LOGDIR/align_is2ctempoatmlvisbal_IS2era.log ==="

# pre-IS2 era (cryotempo+atm+lvis, no IS2) → ASP_ctempoatmlvisbal.
# Here cryotempo is often the sole control, so balancing leaves its count alone
# (most-precise source present); the win is folding in the recovered unc<3 strips.
$PY -u -m pig.align_strips --control cs2 --cs2-source cryotempo --use-atm --use-lvis \
    --inv-variance-balance --asp-suffix _ctempoatmlvisbal \
    --start 2010-01-01 --end 2018-10-13 --parallel "$P" \
    > "$LOGDIR/align_ctempoatmlvisbal_preIS2.log" 2>&1
echo "=== pre-IS2 era done $(date) (exit $?) → $LOGDIR/align_ctempoatmlvisbal_preIS2.log ==="

echo "=== balanced re-align COMPLETE $(date) ==="
echo "Next: build_stack --is2-asp is2ctempoatmlvisbal --pre-is2-asp ctempoatmlvisbal"
echo "      → tilt_fit → run_melt  (and densetie --is2-asp/--pre-is2-asp at the *bal roots)"
