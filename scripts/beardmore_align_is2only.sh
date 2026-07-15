#!/bin/bash
# Re-align every post-Oct 2018 Beardmore-AOI strip with IS2-only control
# (no CryoSat-2). Output dir is data/REMA/strips/ASP_is2only/asp_aligned/
# so it doesn't conflict with the existing is2cs2 (mixed control) and cs2
# (CS2-only) trees.
#
# Comparing the three populations in fig4 tells us how much CS2 actually
# helped post-IS2: if IS2-only ≈ IS2+CS2 in terms of post-align residual
# and z_off scatter, then CS2 was redundant once IS2 was available.
#
# Parallel=4 is safe alongside the wider-AOI build_stack/tilt_fit chain
# (load was 1.68 at launch). Tune down if contention shows up.

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
LOGDIR=/wd2/projects/stereo_melt/beardmore/logs
cd /wd2/projects/stereo_melt

# CPU-only (no GPU needed for ASP).
unset CUDA_VISIBLE_DEVICES
export STEREO_MELT_BACKEND=numpy

CHAINLOG="$LOGDIR/align_is2only_chain.log"
exec > >(tee -a "$CHAINLOG") 2>&1

echo "=== Beardmore IS2-only alignment start === $(date)"
echo "Window: 2018-10-15 .. 2023-03-01"
echo "Control: IS2 only (no CS2)"
echo "Output:  data/REMA/strips/ASP_is2only/asp_aligned/"
echo

$PY -u -m beardmore.align_strips \
    --control is2 \
    --asp-suffix _is2only \
    --start 2018-10-15 \
    --end 2023-03-01 \
    --parallel 4 \
    > "$LOGDIR/align_is2only.log" 2>&1
RC=$?
echo
echo "=== align_strips exit=$RC === $(date)"
echo "ASP_is2only/asp_aligned count: $(ls /wd2/projects/stereo_melt/data/REMA/strips/ASP_is2only/asp_aligned/*-trans_reference-DEM.tif 2>/dev/null | wc -l)"
