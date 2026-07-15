#!/bin/bash
# Overnight Shean-matching sweep (2026-06-16).
#
# Regenerates PIG melt with the FUSED velocity representation
# (Kalman temporal smoother + EOF low-rank + harmonic gap-fill + 1 km spatial
# Gaussian on velocity only), at 250 m, Lagrangian H*div(u) -- the Shean-matching
# config but keeping our 250 m resolution (no 512 m / 3.5 km blur).
#
# Writes pig_melt_250m_is2ctempo_fusedvel_*.nc BESIDE the raw-velocity baseline
# pig_melt_250m_is2ctempo_*.nc (MELT_OUT_SUFFIX decouples output from stack), so
# the morning A/B (fused vs raw vs Davison) is a one-liner.
#
# Disconnect-safe: launched via setsid (PPID=1), python -u, per-run logs.
set -u
cd /wd2/projects/stereo_melt || exit 1
export PROJ_DATA=/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj
export PIG_VELOCITY=fused
export MELT_OUT_SUFFIX=_fusedvel
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
L=pig/logs

echo "[driver] START $(date)  PIG_VELOCITY=$PIG_VELOCITY  out=$MELT_OUT_SUFFIX"

run() {  # $1=label ; $2.. = extra run_melt args
  local lbl="$1"; shift
  echo "[driver] === $lbl === start $(date)"
  $PY -u -m pig.run_melt --res 250 --tag is2ctempo "$@" \
      > "$L/melt_fusedvel_${lbl}.log" 2>&1
  local rc=$?
  echo "[driver] === $lbl === exit=$rc $(date)"
  echo "[driver] $lbl medians:"
  grep -h "median=" "$L/melt_fusedvel_${lbl}.log" 2>/dev/null | tail -4
}

# Quick window first (fewest epochs) to confirm the fused path end-to-end,
# then the longer / full-record runs.
run W4   --start 2020-02-09 --end 2024-01-10
run W3   --start 2017-09-01 --end 2020-02-08
run full
run W2   --start 2013-11-01 --end 2017-08-31
run W1   --start 2011-02-04 --end 2013-10-31

echo "[driver] ALL DONE $(date)"
