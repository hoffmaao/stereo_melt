#!/bin/bash
# L-curve sweep over the prior strength gamma for the PIG dual fluidity
# inversion, following the mismip_time-dependent-da procedure: each gamma is
# an INDEPENDENT COLD-START run (theta0 = 0, i.e. uniform A0) writing its own
# summary.json; no warm-starting between sweep points, so the curve carries no
# path dependence. scripts/lcurve_eta_knee.py then locates the knee.
#
# The box has 32 cores and each run is serial (~6 GB, one core), so the points
# run CONCURRENTLY at the full iteration budget rather than being shortened.
# Each gets its own firedrake/PyOP2 cache dir: the JIT disk cache races when
# several processes compile the same kernels at once.
#
# Usage:  examples/pig/scripts/run_eta_lcurve_sweep.sh [MAX_ITERS] [GAMMA ...]
set -u
REPO=/wd2/projects/stereo_melt/examples
cd "$REPO" || exit 1

MAX_ITERS="${1:-150}"; shift || true
GAMMAS=("$@")
if [ ${#GAMMAS[@]} -eq 0 ]; then
    GAMMAS=(1e-2 1e-1 1e0 1e1 1e2 1e3)
fi
CACHE_ROOT="${TMPDIR:-/tmp}/stereo_melt_lcurve_cache"
mkdir -p "$CACHE_ROOT" "$REPO/pig/logs"

echo "L-curve sweep: gammas ${GAMMAS[*]}  max_iters $MAX_ITERS"
echo "domain: $(basename $REPO/pig/processed/pig_eta_inv_inputs.npz) (current exporter output)"
for g in "${GAMMAS[@]}"; do
    tag="lcurve_g${g}"
    out="$REPO/pig/processed/pig_eta_field_250m_dual_${tag}.npz"
    if [ -f "${out%.npz}_summary.json" ]; then
        echo "SKIP  gamma=$g (summary exists)"
        continue
    fi
    cache="$CACHE_ROOT/$tag"
    mkdir -p "$cache"
    XDG_CACHE_HOME="$cache" PYOP2_CACHE_DIR="$cache/pyop2" \
    nohup "$REPO/elmer_synth/scripts/icepack_python.sh" -u \
        "$REPO/pig/scripts/infer_eta_icepack2_pig.py" \
        --gamma "$g" --max-iters "$MAX_ITERS" --out-tag "$tag" \
        > "$REPO/pig/logs/pig_eta_lcurve_g${g}.log" 2>&1 &
    echo "START gamma=$g  pid $!  -> pig/logs/pig_eta_lcurve_g${g}.log"
    sleep 30      # stagger: let each populate its cache before the next starts
done
wait
echo "LCURVE_SWEEP_DONE"
