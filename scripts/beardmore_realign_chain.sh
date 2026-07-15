#!/bin/bash
# Beardmore wider-AOI realign chain (2026-05-08).
#
# Chain: cache_cryosat2 + cache_icesat2 (parallel) -> align_strips (cs2 era) ->
# align_strips (is2cs2 era). Each stage is era-split so pre-Oct 2018 strips
# land in ASP_cs2/ (Ez=2.0 m prior) and post-Oct 2018 land in ASP_is2cs2/
# (Ez=0.1 m prior); see feedback_ez_per_gcp_source.md.
#
# Strip selection uses BEARDMORE_STACK_AOI_SHP (wider analysis-grid AOI);
# per-point IS2/CS2 queries use the strip's own bbox (lib default).
#
# Designed to run alongside the wider-AOI tilt_fits (PIDs 5023/5024) without
# resource contention: --parallel 12 keeps ~12 CPU workers under 32 cores
# while the 2 tilt_fits use ~7 effective cores combined.

PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
ROOT=/wd2/projects/stereo_melt
LOGS=$ROOT/beardmore/logs
mkdir -p "$LOGS"

cd "$ROOT" || exit 1

echo "[$(date)] === Beardmore realign chain start ==="
df -BG /wd2 | tail -1 | awk '{printf("[disk] %s used / %s total / %s free\n", $3, $2, $4)}'
ps -p 5023,5024 -o pid,etime,stat,pcpu,rss,cmd 2>/dev/null | tail -n +1
echo

echo "[$(date)] Stage 1: caches (cs2 + is2 in parallel)"
$PY -u -m beardmore.cache_cryosat2 > "$LOGS/realign_cache_cs2.log" 2>&1 &
PID_CS2=$!
$PY -u -m beardmore.cache_icesat2 > "$LOGS/realign_cache_is2.log" 2>&1 &
PID_IS2=$!
echo "[$(date)] cache PIDs: cs2=$PID_CS2 is2=$PID_IS2"
wait $PID_CS2
RC_CS2=$?
echo "[$(date)] cs2 cache exit=$RC_CS2"
wait $PID_IS2
RC_IS2=$?
echo "[$(date)] is2 cache exit=$RC_IS2"
echo

echo "[$(date)] Stage 2a: align pre-Oct 2018 (--control cs2 --asp-suffix _cs2)"
$PY -u -m beardmore.align_strips \
    --start 2013-01-01 --end 2018-10-15 \
    --asp-suffix _cs2 --control cs2 --parallel 12 \
    > "$LOGS/realign_align_cs2.log" 2>&1
RC_ALIGN_CS2=$?
echo "[$(date)] cs2-era align exit=$RC_ALIGN_CS2"
echo

echo "[$(date)] Stage 2b: align post-Oct 2018 (--control is2+cs2 --asp-suffix _is2cs2)"
$PY -u -m beardmore.align_strips \
    --start 2018-10-15 --end 2023-03-01 \
    --asp-suffix _is2cs2 --control is2+cs2 --parallel 12 \
    > "$LOGS/realign_align_is2cs2.log" 2>&1
RC_ALIGN_IS2CS2=$?
echo "[$(date)] is2cs2-era align exit=$RC_ALIGN_IS2CS2"
echo

echo "[$(date)] === Beardmore realign chain done ==="
df -BG /wd2 | tail -1 | awk '{printf("[disk] %s used / %s total / %s free\n", $3, $2, $4)}'
ls /wd2/projects/stereo_melt/data/REMA/strips/ASP_cs2/asp_aligned/*-trans_reference-DEM.tif 2>/dev/null | wc -l \
    | awk '{printf("[count] ASP_cs2 aligned: %d\n", $1)}'
ls /wd2/projects/stereo_melt/data/REMA/strips/ASP_is2cs2/asp_aligned/*-trans_reference-DEM.tif 2>/dev/null | wc -l \
    | awk '{printf("[count] ASP_is2cs2 aligned: %d\n", $1)}'
