#!/usr/bin/env bash
# Densetie-vs-is2ctempo A/B melt runner (scratch; 2026-06-19).
# Runs the 3 user-specified 2015-anchored windows for one alignment tag.
#   usage: _ab_run.sh <tag> [wait]
#     <tag>  : is2ctempo | densetie   (selects ASP alignment + tilt-corrected stack)
#     wait   : if "wait", block until that tag's tilt-corrected stack is complete
#              (used for densetie, whose build+tilt is still running)
# Velocity: ase-quarterly v05 for ALL windows (2015 start => full coverage),
# held constant across both arms so the only A/B variable is the alignment.
set -uo pipefail
TAG="${1:?usage: _ab_run.sh <tag> [wait]}"
WAIT="${2:-no}"
cd /wd2/projects/stereo_melt/examples
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
TILT="/wd2/projects/stereo_melt/examples/pig/processed/pig_stack_250m_${TAG}_tilt_corrected_2010-01-01_2024-01-10.nc"

if [ "$WAIT" = "wait" ]; then
  echo "[$TAG] waiting for tilt-corrected stack: $TILT ($(date))"
  WAITED=0; MAXWAIT=$((18*3600))  # build+tilt is ~12h+; 6h was too short and aborted the 06-19 densetie arm
  while true; do
    if [ -f "$TILT" ] && ! pgrep -f "pig.tilt_fit --res 250 --tag ${TAG}" >/dev/null; then
      s1=$(stat -c%s "$TILT"); sleep 20; s2=$(stat -c%s "$TILT")
      [ "$s1" = "$s2" ] && break
    fi
    sleep 60; WAITED=$((WAITED+80))
    if [ $WAITED -ge $MAXWAIT ]; then
      echo "[$TAG] TIMEOUT (${MAXWAIT}s) waiting for tilt stack — build/tilt likely failed. Aborting."
      exit 1
    fi
  done
  echo "[$TAG] tilt-corrected stack ready ($(date))"
fi

export STEREO_MELT_BACKEND=numpy OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PIG_VELOCITY=ase-quarterly

run() {  # <start> <end> <label>
  echo "===== [$TAG] $3  $1 -> $2   START $(date) ====="
  "$PY" -u -m pig.run_melt --res 250 --tag "$TAG" --start "$1" --end "$2"
  echo "===== [$TAG] $3  rc=$?   END $(date) ====="
}

run 2015-01-01 2023-12-31 whole_2015-2023
run 2015-01-01 2020-02-08 before_B49
run 2020-02-09 2023-12-31 after_B49
echo "[$TAG] ALL A/B RUNS DONE $(date)"
