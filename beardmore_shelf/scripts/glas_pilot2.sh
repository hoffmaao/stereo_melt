#!/bin/bash
set -u
cd /wd2/projects/stereo_melt
PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
echo "=== pilot 2a: 3 earliest cached strips $(date '+%F %T')"
$PY -u -m beardmore_shelf.align_strips --control glas --asp-suffix _glas \
  --start 2009-01-01 --end 2010-10-12 --limit 3
echo "=== pilot 2b: latest cached strip (SETSM_s2s041_W1W1_20100206_102001000B555F00_102001000BEB0800_2m_lsf_seg3) $(date '+%F %T')"
$PY -u -m beardmore_shelf.align_strips --control glas --asp-suffix _glas \
  --start 2009-01-01 --end 2010-10-12 --dem-id SETSM_s2s041_W1W1_20100206_102001000B555F00_102001000BEB0800_2m_lsf_seg3
echo "=== pilot outputs:"
ls beardmore_shelf/data/ASP_glas/asp_aligned/ | head -30
echo "GLAS_PILOT2_DONE"
