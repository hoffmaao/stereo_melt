"""Pine Island Glacier study configuration.

All paths and scalar parameters specific to the Pine Island ice-shelf
basal-melt-rate study. Mirrors :mod:`beardmore.config` /
:mod:`nansen.config`. The library carries no hardcoded paths; this file
is the single point where PIG-specific context lives.

Pine Island Glacier (Amundsen Sea Embayment, West Antarctica, ~-75°S /
-101°E) drains ~10% of the West Antarctic Ice Sheet. The remaining
floating tongue (post the 2017-2020 calving sequence) is ~6120 km^2.
The AOI shapefile is extracted from the MEaSUREs ``IceShelf_Antarctica_v02``
layer (feature NAME == ``"Pine_Island"``).

Latitude (-75°S) is north of the DTU22 -79°S coverage limit, so geoid +
MDT corrections both apply (unlike Beardmore which is geoid-only).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Workspace paths
# ---------------------------------------------------------------------------
MAIN_DIR = Path("/wd2/projects/stereo_melt")
BASIN_DIR = Path(__file__).resolve().parent

# Study outputs (gitignored)
FIGURES_DIR = BASIN_DIR / "figures"
PROCESSED_DIR = BASIN_DIR / "processed"
RESULTS_DIR = BASIN_DIR / "results"
CLIMATE_CACHE_DIR = BASIN_DIR / "data" / "climate"

# ---------------------------------------------------------------------------
# Shared data cache (same as Beardmore/Nansen — pan-Antarctic rasters here).
# ---------------------------------------------------------------------------
DATA_DIR = MAIN_DIR / "data"

REMA_DIR = DATA_DIR / "REMA"
STRIPS_DIR = REMA_DIR / "strips"
# Per-basin ASP roots. The wider-AOI re-alignment (2026-05) split the
# record into two complementary, non-overlapping eras, each aligned with
# the full set of altimetric GCPs available in that era:
# - ASP_IS2CS2ATMLVIS_ROOT: IS2 era (2018-10-25 → 2023-12-25, 375 strips),
#   IS2 + CS2 control + ATM + LVIS airborne. Align done 2026-05-30.
# - ASP_CS2ATMLVIS_ROOT: pre-IS2 era (2010-12-09 → 2018-04-14, 159 strips),
#   CS2 control + ATM + LVIS airborne (no IS2 before Oct 2018). Done 2026-05-25.
# Per-strip <dem_id>.sources.json sidecars record the actual GCPs used, so
# tilt_fit's per-epoch Ez resolves from the sidecar, not the variant label.
# The legacy narrow-AOI ASP_ROOT (pig/data/ASP, ~87 strips) and the deleted
# ASP_is2atmlvis root are superseded and intentionally excluded.
ASP_ROOT = BASIN_DIR / "data" / "ASP"            # legacy; kept only for ensure_output_dirs
STRIP_ALIGNED_DIR = ASP_ROOT / "asp_aligned"
ASP_IS2CS2ATMLVIS_ROOT = BASIN_DIR / "data" / "ASP_is2cs2atmlvis"
ASP_CS2ATMLVIS_ROOT = BASIN_DIR / "data" / "ASP_cs2atmlvis"
# CryoTEMPO control base = production baseline (supersedes raw CS2 --
# project_ctempo_ab_verdict). 376 is2ctempoatmlvis + 258 ctempoatmlvis aligned
# DEMs on disk (verified 2026-06-20).
ASP_IS2CTEMPOATMLVIS_ROOT = BASIN_DIR / "data" / "ASP_is2ctempoatmlvis"
ASP_CTEMPOATMLVIS_ROOT = BASIN_DIR / "data" / "ASP_ctempoatmlvis"
# Shean-style "nocorr" root (2026-09-06 test, see pig.ingest_nocorr): strips no
# control polygon touches, so the align chain never attempted them. Ingested at
# a-priori geolocation plus ONE per-era class-mean vertical bias (Shean
# stack_nocorr_adjust.py) with a sources=["nocorr"] sidecar (Ez 1.0 m tier).
# Opt-in via PIG_SOURCES=nocorr, which appends the root LAST to STRIP_SOURCES so
# any granule that also has a real alignment keeps the aligned version.
ASP_NOCORR_ROOT = BASIN_DIR / "data" / "ASP_nocorr"
ASP_NOCORR_ALIGNED_DIR = ASP_NOCORR_ROOT / "asp_aligned"
# Production STRIP_SOURCES = the CryoTEMPO baseline alignment. Drives build_stack
# AND the per-DEM screen's aggregate_basin_quality(STRIP_SOURCES), so both read the
# same alignment. The is2cs2/cs2 roots above are the older raw-CS2 A/B baseline
# (still used by rescue_bad_aligns + prototype_two_stage); the densetie roots are
# the parallel coreg-fix A/B. Was stale (pointed at is2cs2/cs2); fixed 2026-06-20.
STRIP_SOURCES: list[tuple["Path", str]] = [
    (ASP_IS2CTEMPOATMLVIS_ROOT / "asp_aligned", "is2ctempoatmlvis"),  # IS2 era 2018-10 → 2023-12
    (ASP_CTEMPOATMLVIS_ROOT / "asp_aligned", "ctempoatmlvis"),        # pre-IS2 2010-12 → 2018-04
]
if os.environ.get("PIG_SOURCES", "").lower() == "nocorr":
    STRIP_SOURCES.append((ASP_NOCORR_ALIGNED_DIR, "nocorr"))
# Dense-tie (Option-B two-stage) re-alignment root — step 5 of
# literature/plan_alignment.md. Every production-aligned strip in STRIP_SOURCES
# re-aligned to the static-masked REMA reference (REMA_STATIC_REFERENCE_TIF) via
# align_strip_two_stage, written here so the is2cs2atmlvis/cs2atmlvis baselines
# stay intact for the A/B. Strips with no static overlap fall back to a symlink
# of their baseline DEM (coverage parity). Built by pig.align_strips_densetie.
ASP_DENSETIE_ROOT = BASIN_DIR / "data" / "ASP_densetie"
# Dense-tie-FROM-RAW root. Unlike ASP_DENSETIE_ROOT (which re-ties only the
# already-aligned STRIP_SOURCES strips), align_strips_densetie_fromraw discovers
# every RAW strip in window∩AOI∩grounded — including the ~150 pre-IS2 strips that
# one-stage altimetry pc_align failed to converge on (sparse-altimetry divergence).
# Stage 1 aligns raw→static-masked REMA (geometry, altimetry-free); Stage 2 ties
# the datum with a plain robust median of (control−aligned) over ALL altimetry
# (Shean coregistration — no CryoTEMPO discarded; IS2 dominates by count where
# present, CryoTEMPO sets it where alone). IS2-vs-CryoTEMPO precision weighting
# lives in the tilt-fit LSQ (per-source Ez). Fixes the ~9.4× coreg ramp + recovers
# the dropped pre-IS2 strips; the pc_align 13:1 overweight is moot here (Stage 1
# uses REMA, not altimetry).
ASP_DENSETIE_FROMRAW_ROOT = BASIN_DIR / "data" / "ASP_densetie_fromraw"
MOSAIC_DIR = REMA_DIR / "mosaic"
# Dense static reference for the Option-B two-stage coregistration
# (literature/plan_alignment.md): the PIG REMA 32 m mosaic tiles merged
# over the AOI and masked to the *static* surface (BedMachine rock +
# grounded, 2 km eroded — the SAME mask the tilt LSQ uses via
# build_static_area_polygon_mask). Built by pig.build_rema_static_reference;
# consumed by stereo_melt.coregister.asp.align_strip_two_stage as
# `reference_dem`. PGC REMA only — never Shean's uploaded grids.
REMA_STATIC_REFERENCE_TIF = MOSAIC_DIR / "pig_rema_static_reference_32m.tif"

SHAPE_DIR = DATA_DIR / "shapefiles"
# Canonical wider stack/tilt AOI: rectangular AOI computed by the v15
# algorithm (greedy expansion subject to TS-area target ≥ 5000 km²,
# asymmetric upstream + perpendicular bias from shelf-only flow
# direction). See scripts/compute_aoi.py — bounds (-1717300, -363500)
# → (-1536850, -142100), 39,952 km². Used for strip discovery,
# alignment, control caches, build_stack, and tilt-fit alike. The old
# narrow pig.shp shelf polygon was archived 2026-05-16.
PIG_AOI_SHP = SHAPE_DIR / "pig_stack_extent.shp"
# Back-compat alias; new code should reference PIG_AOI_SHP directly.
PIG_STACK_AOI_SHP = PIG_AOI_SHP
# True ice-shelf boundary (MEaSUREs IceShelf_Antarctica_v02) for clipping
# melt-rate viz / interpretation plots to the shelf proper. Use with
# ICESHELF_FEATURE_NAME to extract the Pine Island feature.
ICESHELF_FEATURE_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ICESHELF_FEATURE_NAME = "Pine_Island"
GROUNDING_LINE_SHP = SHAPE_DIR / "GroundingLine_Antarctica_v02.shp"
ICE_SHELF_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ROCK_POLYGONS_SHP = SHAPE_DIR / "rock_bedmachine.shp"
LOW_VELOCITY_POLYGONS_SHP = SHAPE_DIR / "low_velocity_polygons.shp"
MOSAIC_INDEX_SHP = SHAPE_DIR / "REMA_Mosaic_Index_v2_32m.shp"
STRIP_INDEX_SHP = SHAPE_DIR / "REMA_Strip_Index_s2s041.parquet"

VEL_DIR = DATA_DIR / "ITsLIVE"
# PIG is in fast-flowing ice with strong feature-tracking signal, so
# ITS_LIVE annual mosaics work well here (unlike Beardmore's slow
# grounding zone). RGI19A = Antarctic Peninsula + WAIS region.
ITS_LIVE_2019 = VEL_DIR / "ITS_LIVE_velocity_120m_RGI19A_2019_v02.nc"
ITS_LIVE_2020 = VEL_DIR / "ITS_LIVE_velocity_120m_RGI19A_2020_v02.nc"
ITS_LIVE_2021 = VEL_DIR / "ITS_LIVE_velocity_120m_RGI19A_2021_v02.nc"

MEASURES_VEL_DIR = DATA_DIR / "NSIDC-0754" / "1996.01.01"
MEASURES_PHASE_NC = MEASURES_VEL_DIR / "antarctic_ice_vel_phase_map_v01.nc"

# Quarterly ASE velocity mosaics (vx/vy/vv, 250 m). Coords are EPSG:3031 even
# though the GeoTIFFs are tagged LOCAL_CS, so .interp aligns by coordinate value
# correctly. Joughin v05.0 "May 2024 update": 2015 Q1 -> 2024 Q1, 37 contiguous
# quarters (no gaps), copied 2026-06-16 from knut's crevasses archive
# (THW_crevasse_prediction_23jul2024/joughin_vel_may2024/quarterly). Gives true
# sub-annual cadence for the time-varying Lagrangian advection
# (PIG_VELOCITY=ase-quarterly), now spanning the full PIG window including the
# post-B-49 (2020-02-09) calving era. Supersedes the older v02.1 2015-2021Q1 set
# still in data/ASE_vel_quarterly/ (kept for comparison).
# Filenames: ASE_vel_mosaic_Quarterly_<ddMonyy>_<ddMonyy>_v{x,y,v}_v05.0.tif
ASE_QUARTERLY_VEL_DIR = DATA_DIR / "ASE_vel_quarterly_v05"

BM_DIR = DATA_DIR / "bedmachine"
BEDMACHINE_NC = BM_DIR / "BedMachineAntarctica-v3.nc"

# Greene et al. 2022 time-evolving Antarctic ice masks (annual, 240 m,
# 1997.75-2021.2; Zenodo 5903643) — the observed-coastline input to the
# window-minimum shelf extent (pig.build_min_extent_mask).
GREENE_DIR = DATA_DIR / "Greene2022"
GREENE_ICEMASK_MAT = GREENE_DIR / "icemask_composite.mat"

MDT_DIR = DATA_DIR / "MDT"
DTU22_MDT_XYZ = MDT_DIR / "dtuuh22mdt.xyz"

RACMO_DIR = DATA_DIR / "RACMO"
RACMO_SMB_NC = RACMO_DIR / "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202512.nc"
RACMO_MASK_NC = RACMO_DIR / "ANT11_masks.nc"

# pyTMD model. PIG sits on the Amundsen Sea shelf; CATS2008 is the
# standard pan-Antarctic choice and covers the full domain.
TIDE_MODEL = "CATS2008"
TIDE_MODEL_DIR = MAIN_DIR

# ---------------------------------------------------------------------------
# Study window
# ---------------------------------------------------------------------------
SHELF = "PineIsland"
# Start extended back to 2010 so pre-IS2 ATM/LVIS-aligned strips
# (ASP_is2atmlvis/asp_aligned/) can be folded in alongside the IS2/CS2 era.
# End is the s2s041 strip-index ceiling (acqdate1 max = 2024-01-09); extend
# once a newer REMA index release lands.
START_TIME = "2010-01-01"
END_TIME = "2024-01-10"

# Target grid
RES = 25  # meters

# Epochs dropped before the per-epoch tilt LSQ. Regenerated 2026-06-20 on the
# Shean-faithful coregistration-quality screen (stereo_melt.coregister.tilt_qc),
# replacing the pre-2026-06-20 frac_static<0.10 gate. On PIG's ~4%-static wide
# stack that gate degenerated to a flat |static-resid|>~2 m drop, flagging
# cleanly-coregistered SHELF strips for noisy thin rock-margin pixels and
# discarding ~96 post-2018 shelf DEMs (69 with pc_align end_p50<3 m — tied to
# sub-metre). Shean 2019 rejects DEMs by COREGISTRATION QUALITY instead; thin
# coverage is handled per-pixel (kinematics min_count) and noise by the tilt-LSQ
# IRLS + the end_p50>3 m offset-only demote (PIG_OFFSET_ONLY_END_P50_M), not by
# deleting shelf epochs.
#
# Two data-derived gates (see pig.scripts.find_bad_epochs /
# assemble_bad_strips_pretilt --res 250 --tag is2ctempo), applied PER-DEM and
# expressed as BAD_STRIPS below:
#   (a) pc_align end_p50 > 10 m — genuine alignment divergence. Clean populations
#       sit at median 0.28 m (IS2 era) / 0.63 m (CS2 era); 10 m is far above both
#       and above the 3 m demote band. 16 strips.
#   (b) |raw static-median deviation| > 50 m — catastrophic garbage DEMs (bright
#       bands / blunders over grounded ice) that pc_align tied on a good patch but
#       that would poison the per-pixel temporal-median reference. 8 strips.
# Union = 20 strips. Re-run after tilt_fit to confirm convergence.
#
# Per-DATE rejection is RETIRED (2026-06-21). It was an artifact of REMA strip
# filenames carrying no time-of-day: many strips on one acquisition day normalized
# to a single date, so a date-keyed drop discarded clean same-day siblings (e.g.
# 2018-12-31 had 6 strips, 5 clean, all killed for one +123 m blunder). Shean 2019
# rejected individual DEMs by coregistration quality (tocut / tocut_lowcount
# basename lists + mincount) — never by date. See BAD_STRIPS below.
BAD_EPOCHS: tuple[str, ...] = ()

# Per-STRIP drop list (SETSM dem_id) -- the Shean-faithful rejection unit, now the
# SOLE pre-tilt DEM-rejection mechanism (BAD_EPOCHS retired to () above).
# load_basin_stack keys these on the stack's per-slice `dem_id` coord (written by
# build_stack since 2026-06-20); the is2ctempo 250 m stack carries it. Keying on
# dem_id drops only the offending strip, never clean same-day siblings, and a
# future clean strip landing on a previously-bad day is never caught by a stale
# date entry. Regenerate with:
#   python -m pig.scripts.assemble_bad_strips_pretilt --res 250 --tag is2ctempo
#
# Migration off date-keying (2026-06-21) was verified safe by that script's audit:
# vs the 13 retired BAD_EPOCHS dates it spared 31 clean same-day slices, re-admitted
# 6 dates that were artifacts of the stale is2cs2 alignment (now end_p50 ≤ 0.6 m
# under ctempo) and 3 already filtered upstream by the count/source cut (Shean
# tocut_lowcount analog), with ZERO genuinely-bad strips re-admitted.
# --- Tilt-plane priors for fit_tilt_stack (2026-09-02, control-residual calibration).
# Measured on the 513-strip canon stack from the signed residual of each ALIGNED
# DEM against the control pc_align was fed (robust across-strip scale, estimator
# variance removed): tau_x = 1.31e-6, tau_y = 3.02e-6 m/m. The library defaults
# (Shean PIG: Ex 2e-6, Ey = Ex/3 = 6.67e-7) have the anisotropy BACKWARDS on real
# strips -- residual tilt is ~2.3x larger across-track than along -- and Ey was
# 4.5x too tight. Truth-free, from independent altimetry (no plane<->trend
# degeneracy); see results/pig_strip_residual_planes_250m_is2ctempo_sheltilt.csv.
# Override per run with PIG_TILT_EX / PIG_TILT_EY; PIG_TILT_EX=2e-6
# PIG_TILT_EY=6.6667e-7 restores the canon PRIORS, but NOT the canon PRODUCT --
# the 15-strip control-QC drop in BAD_STRIPS below is unconditional and
# deliberate (they are genuine alignment failures), and has no escape hatch. The
# canon numbers 84.6/89.6/90.4/93.3 belong to the products already on disk under
# tag is2ctempo_sheltilt and are a historical record, not something this config
# regenerates. The corrected state is is2ctempo_sheltilt_qcey. The two agree to
# <= 0.2 Gt/yr on every field because the strip drop (-0.6..-0.9) and the
# recalibrated priors (+0.6..+0.7) very nearly cancel.
TILT_EX: float = float(os.environ.get("PIG_TILT_EX", "1.31e-6"))
TILT_EY: float = float(os.environ.get("PIG_TILT_EY", "3.02e-6"))

BAD_STRIPS: tuple[str, ...] = (
    # --- 2026-09-02: control-residual plane QC (alignment_quality.residual_planes_for_strips):
    # signed aligned-DEM minus pc_align-control, robust plane fit; residual scatter sd > 5 m
    # = alignment did not converge over a substantial part of the footprint. pc_align's own
    # end_p50 MISSES most of these (partially broken strips pass a median screen; p84 does not),
    # and the tilt fit did not absorb them (tilt_dz ~0.05 m; IRLS kept 18-87 % of their
    # pixels), so their 1-207 m errors sat in the corrected stack. See
    # results/pig_control_residual_alignment_failures.csv; the canon 84.6/89.6/90.4/93.3
    # was built WITH these 15 in (tag is2ctempo_sheltilt); the re-run tags are
    # is2ctempo_sheltilt_qc (this drop only) and is2ctempo_sheltilt_qcey (+ TILT_EX/EY).
    "SETSM_s2s041_WV01_20200321_10200100941A8000_10200100950A7300_2m_lsf_seg3",  # sd=175m offset=-111m n=3496 [is2ctempoatmlvis]
    "SETSM_s2s041_WV02_20210126_10300100B141D400_10300100B405E700_2m_lsf_seg3",  # sd=115m offset=+207m n=1061 [is2ctempoatmlvis]
    "SETSM_s2s041_WV03_20151030_10400100135FB800_1040010012AF0200_2m_lsf_seg2",  # sd=78.9m offset=+61.8m n=1294 [ctempoatmlvis]
    "SETSM_s2s041_WV01_20151005_1020010046B3BC00_1020010044DAF500_2m_lsf_seg2",  # sd=68.3m offset=+87m n=2579 [ctempoatmlvis]
    "SETSM_s2s041_WV01_20181231_102001007FA19D00_102001007F7A2400_2m_lsf_seg2",  # sd=19.9m offset=+1.57m n=1810 [is2ctempoatmlvis]
    "SETSM_s2s041_W2W2_20111222_103001000F578C00_1030010010AD2800_2m_lsf_seg7",  # sd=13.7m offset=-0.463m n=2745 [ctempoatmlvis]
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg5",  # sd=13.4m offset=+1.02m n=2104 [is2ctempoatmlvis]
    "SETSM_s2s041_WV02_20191103_103001009A0BE100_103001009DA09500_2m_lsf_seg2",  # sd=10.5m offset=+3.48m n=841 [is2ctempoatmlvis]
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg3",  # sd=9.86m offset=+0.479m n=4711 [is2ctempoatmlvis]
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg16",  # sd=9.05m offset=+0.787m n=2725 [is2ctempoatmlvis]
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg18",  # sd=8.98m offset=+2.73m n=4748 [is2ctempoatmlvis]
    "SETSM_s2s041_WV02_20200315_10300100A4071C00_10300100A4AD0500_2m_lsf_seg4",  # sd=6.21m offset=-0.7m n=806 [is2ctempoatmlvis]
    "SETSM_s2s041_WV02_20230126_10300100E13EAF00_10300100E157EA00_2m_seg3",  # sd=6.12m offset=-0.148m n=3572 [is2ctempoatmlvis]
    "SETSM_s2s041_WV02_20191103_103001009A0BE100_103001009DA09500_2m_lsf_seg4",  # sd=6.04m offset=+0.193m n=851 [is2ctempoatmlvis]
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg17",  # sd=5.98m offset=+1.95m n=5193 [is2ctempoatmlvis]
    "SETSM_s2s041_W1W1_20201103_102001009E050D00_10200100A0606400_2m_lsf_seg3",  # raw_dev=+582m
    "SETSM_s2s041_W1W2_20201215_10200100A3CB1100_10300100B1A01E00_2m_lsf_seg1",  # end_p50=94.2m
    "SETSM_s2s041_W2W2_20111231_1030010010118200_1030010010BA0D00_2m_lsf_seg1",  # raw_dev=-77m
    "SETSM_s2s041_WV01_20181231_102001007FA19D00_102001007F7A2400_2m_lsf_seg1",  # raw_dev=+123m
    "SETSM_s2s041_WV01_20200321_10200100941A8000_10200100950A7300_2m_lsf_seg1",  # end_p50=16.9m
    "SETSM_s2s041_WV01_20200321_10200100941A8000_10200100950A7300_2m_lsf_seg2",  # end_p50=10.8m, raw_dev=-63m
    "SETSM_s2s041_WV01_20200403_10200100959E0700_1020010092256300_2m_lsf_seg10",  # end_p50=31.0m, raw_dev=-66m
    "SETSM_s2s041_WV01_20200403_10200100959E0700_1020010092256300_2m_lsf_seg11",  # end_p50=32.1m
    "SETSM_s2s041_WV01_20200403_10200100959E0700_1020010092256300_2m_lsf_seg7",  # end_p50=24.9m, raw_dev=+52m
    "SETSM_s2s041_WV01_20200403_10200100959E0700_1020010092256300_2m_lsf_seg8",  # end_p50=37.1m, raw_dev=+53m
    "SETSM_s2s041_WV01_20200403_10200100959E0700_1020010092256300_2m_lsf_seg9",  # end_p50=42.6m
    "SETSM_s2s041_WV01_20201026_102001009FAFCE00_10200100A0E88A00_2m_lsf_seg5",  # end_p50=25.2m
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg11",  # end_p50=14.8m
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg12",  # end_p50=12.3m
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg14",  # end_p50=12.9m
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg15",  # end_p50=14.7m
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg4",  # end_p50=11.6m
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg7",  # end_p50=13.2m
    "SETSM_s2s041_WV02_20200326_10300100A3CCAE00_10300100A2364400_2m_lsf_seg8",  # end_p50=17.6m
    "SETSM_s2s041_WV03_20230119_1040010082067300_104001008233E700_2m_seg1",  # raw_dev=-117m
)

# IS2 control filter parameters. Default to Shean 2019 (TC 13:2633)
# Sect. 2.2.1 rule: cap advection error per photon at 10 m
# (|v|·|Δt| ≤ budget) within a ±1 yr outer time cap. Adapts the
# effective time window to local velocity — slow grounding zones get
# wide windows, fast trunks get tight ones. More physical than the
# Chartrand fixed-window rule for fast-flow basins like PIG.
#   IS2_TIME_WINDOW_DAYS: outer cap on |t_photon − t_DEM|. Shean uses 365.
#   IS2_DISPLACEMENT_BUDGET_M: 10 m advection budget (Shean default).
#   IS2_MAX_SPEED_MYR: outer velocity cap; 1e9 ≈ disabled when the
#     displacement budget is doing the real filtering.
IS2_TIME_WINDOW_DAYS = 365
IS2_DISPLACEMENT_BUDGET_M = 10.0
IS2_MAX_SPEED_MYR = 1.0e9

# CryoTEMPO control linking — adopt IS2's Shean displacement-budget rule so fast
# grounded ice is kept with a velocity-tightened temporal window instead of the
# old hard slow<10 m/yr cut, which zeroed PIG's fast grounded zone and was the
# dominant cause of dropped pre-IS2 strips (diagnosed 2026-06-18). Tuned 2×
# COMPRESSED vs IS2 — CryoTEMPO POCA is noisier than ATL06 on fast/crevassed
# ice, so it gets less temporal slack: half the advection budget and half the
# outer cap. "To start" values; tune against align convergence.
CRYOTEMPO_DISPLACEMENT_BUDGET_M = IS2_DISPLACEMENT_BUDGET_M / 2.0   # 5.0 m
CRYOTEMPO_TIME_WINDOW_DAYS = IS2_TIME_WINDOW_DAYS // 2              # 182 d

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------


def ensure_output_dirs():
    """Create output directories if missing."""
    for d in (FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, CLIMATE_CACHE_DIR,
              ASP_ROOT, STRIP_ALIGNED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Cache file for the one-shot ERA5 surface-pressure bulk pull. The
# stack window is encoded so that a different START/END produces a
# different cache file (no stale-cache risk).
ERA5_CACHE_NC = CLIMATE_CACHE_DIR / f"era5_pressure_{START_TIME}_{END_TIME}.nc"
