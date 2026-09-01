"""Beardmore shelf-sector study configuration.

All paths and scalar parameters specific to the Ross Ice Shelf sector at the
Beardmore Glacier outlet. Mirrors :mod:`venable.config` (the freshest
IS2-era driver template).

The target is the shelf immediately downstream of the Beardmore Glacier
grounding zone (~-84degS / 171degE, Transantarctic Mountains front). The
shelf-outline input is the hand-drawn ``beardmore_gz`` polygon (446 km^2,
archived from the original beardmore basin 2026-05-16, copied here as
``beardmore_shelf.shp``). The stack AOI is the strip-aware v15 rectangle
produced by ``scripts/compute_aoi.py beardmore_shelf`` — raw strips for
this sector are already in the shared pool from the beardmore runs, so the
v15 sizing is strip-aware from day one (no provisional-bbox stage).

Latitude (~-84degS) is poleward of the DTU22 -79degS coverage limit, so the
correction chain is geoid-only (matches beardmore; NOT the geoid+MDT chain
used by PIG / Nansen / Dotson-Crosson / McMurdo / Venable).
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
# Shared data cache (same as the other basins — pan-Antarctic rasters).
# ---------------------------------------------------------------------------
DATA_DIR = MAIN_DIR / "data"

REMA_DIR = DATA_DIR / "REMA"
STRIPS_DIR = REMA_DIR / "strips"
# Per-basin ASP root. Single-source IS2-era basin so one variant label.
ASP_ROOT = BASIN_DIR / "data" / "ASP"
STRIP_ALIGNED_DIR = ASP_ROOT / "asp_aligned"
# Pre-IS2 (2012-2019) CryoTEMPO+ATM-controlled align root, produced by the
# `align_strips --asp-suffix _ctempoatm` pre-IS2 chain. Folded into the stack
# only in combined mode (see below) for the coverage-conditioning lever
# (project_session_resume_2026_07_07): pre-IS2 epochs push the RED stripe
# band's 3-4-epoch cells over the tilt-LSQ conditioning threshold.
ASP_CTEMPOATM_ROOT = BASIN_DIR / "data" / "ASP_ctempoatm"
ASP_CTEMPOATM_ALIGNED_DIR = ASP_CTEMPOATM_ROOT / "asp_aligned"
# GLAS-era (ICESat-1 GLAH12-controlled) align root: the 2009 → 2010-10-11
# archive back-extension (project_glas_era_scoping_2026_07_10 — the only
# laser control that reaches those strips). Produced by
# `align_strips --control glas --asp-suffix _glas` after
# `cache_glas`; NOT in STRIP_SOURCES yet — stack fusion is a separate,
# later decision once alignment QC passes.
ASP_GLAS_ROOT = BASIN_DIR / "data" / "ASP_glas"
ASP_GLAS_ALIGNED_DIR = ASP_GLAS_ROOT / "asp_aligned"
# Shean-style "nocorr" root: the ~101 fully-floating in-window strips that
# intersect ONLY floating ice (no grounded/rock control -> align_strips
# upfront-skips them). Following Shean 2019 (stack_proc.sh order,
# stack_nocorr_adjust.py) they are ingested at a-priori geolocation plus ONE
# class-mean vertical bias by `ingest_nocorr.py`, then datum-corrected by
# the joint tilt LSQ with Ez=1.0 (his exact ndinterp.py value for DEMs
# without ICP; see project_shean_nocorr_framework_2026_07_10).
ASP_NOCORR_ROOT = BASIN_DIR / "data" / "ASP_nocorr"
ASP_NOCORR_ALIGNED_DIR = ASP_NOCORR_ROOT / "asp_aligned"
# Class-mean raw-WV vertical bias (m, applied additively at ingest). Raw
# strips sit ABOVE the aligned control frame here (Shean PIG: -3.1 m);
# measured as the median initial-geodiff (control minus DEM = -2.34 m,
# IQR -4.6..-0.3) across the 173 coregistered IS2-era strips, so ingest
# ADDS -2.34 m to strip elevations.
NOCORR_Z_OFFSET_M = -2.34
# Fused strip sources (build_stack precedence order; first occurrence of a
# granule stem wins). Default = IS2-era only. `BEARDMORE_SHELF_SOURCES=combined`
# appends the pre-IS2 ctempoatm root so build_stack fuses both eras into one
# 2012-2024 stack. Pair with the widened window (BEARDMORE_SHELF_START), else
# build_stack's [START,END) filter drops every pre-IS2 strip and combined mode
# is a no-op. `BEARDMORE_SHELF_SOURCES=nocorr` appends the Shean-style
# nocorr root (same 2019+ window, no START change needed) for the
# 265-epoch A/B stack (--tag nocorr artifacts).
_sources_mode = os.environ.get("BEARDMORE_SHELF_SOURCES", "").lower()
if _sources_mode == "combined":
    STRIP_SOURCES: list[tuple["Path", str]] = [
        (STRIP_ALIGNED_DIR, "is2cs2"),
        (ASP_CTEMPOATM_ALIGNED_DIR, "ctempoatm"),
    ]
elif _sources_mode == "nocorr":
    STRIP_SOURCES: list[tuple["Path", str]] = [
        (STRIP_ALIGNED_DIR, "is2cs2"),
        (ASP_NOCORR_ALIGNED_DIR, "nocorr"),
    ]
elif _sources_mode == "full":
    # Full-record 2009->2024 fusion (project_fullrecord_program_2026_07_11):
    # every align root + the nocorr root. Precedence = alignment quality
    # (IS2-era ICP > pre-IS2 CryoTEMPO/ATM > GLAS-era > nocorr ingest), so a
    # granule stem present in two roots keeps its best-controlled version.
    # Pair with BEARDMORE_SHELF_START=2009-01-01, else the [START,END) filter
    # drops every pre-2019 strip.
    STRIP_SOURCES: list[tuple["Path", str]] = [
        (STRIP_ALIGNED_DIR, "is2cs2"),
        (ASP_CTEMPOATM_ALIGNED_DIR, "ctempoatm"),
        (ASP_GLAS_ALIGNED_DIR, "glas"),
        (ASP_NOCORR_ALIGNED_DIR, "nocorr"),
    ]
else:
    STRIP_SOURCES: list[tuple["Path", str]] = [(STRIP_ALIGNED_DIR, "is2cs2")]
MOSAIC_DIR = REMA_DIR / "mosaic"

SHAPE_DIR = DATA_DIR / "shapefiles"
# Strip-aware v15 stack/align/fetch AOI rectangle, written by
# `scripts/compute_aoi.py beardmore_shelf`; used for strip discovery,
# alignment, control caches, build_stack, and tilt-fit alike.
BEARDMORE_SHELF_AOI_SHP = SHAPE_DIR / "beardmore_shelf_stack_extent.shp"
# Back-compat alias; new code should reference BEARDMORE_SHELF_AOI_SHP directly.
BEARDMORE_SHELF_STACK_AOI_SHP = BEARDMORE_SHELF_AOI_SHP
# Narrow shelf polygon — input to scripts/compute_aoi.py (v15 optimal-AOI sizer),
# which expands it into BEARDMORE_SHELF_AOI_SHP (the wider stack_extent rectangle).
SHELF_INPUT_SHP = SHAPE_DIR / "beardmore_shelf.shp"
# The Beardmore outlet is NOT a named feature in MEaSUREs
# IceShelf_Antarctica_v02 — it sits on the Ross_West / Ross_East seam, so
# clipping viz to either feature would cut the AOI in half. No true-shelf
# feature for viz; use the AOI rectangle for plot bounds (matches beardmore).
ICESHELF_FEATURE_SHP = None
ICESHELF_FEATURE_NAME = None
GROUNDING_LINE_SHP = SHAPE_DIR / "GroundingLine_Antarctica_v02.shp"
ICE_SHELF_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
# BedMachine mask==1 (ice-free ground) polygonized over the AOI + 25 km
# margin (417 features, 981 km2; scripts in session scratchpad 2026-07-05).
# The shared rock_polygons.shp (19 hand-drawn features) intersects this
# AOI NOWHERE — every alignment before 2026-07-05 ran with zero rock GCPs.
ROCK_POLYGONS_SHP = SHAPE_DIR / "beardmore_shelf_rock_bedmachine.shp"
LOW_VELOCITY_POLYGONS_SHP = SHAPE_DIR / "low_velocity_polygons.shp"
MOSAIC_INDEX_SHP = SHAPE_DIR / "REMA_Mosaic_Index_v2_32m.shp"
STRIP_INDEX_SHP = SHAPE_DIR / "REMA_Strip_Index_s2s041.parquet"

VEL_DIR = DATA_DIR / "ITsLIVE"
# Ross sector. ITS_LIVE RGI19A annual tiles (2019-2021) are on disk, but
# feature tracking has large gaps over the slow Beardmore grounding zone
# (see feedback_measures_over_itslive) — mirror the beardmore basin and use
# the MEaSUREs phase map (NSIDC-0754) as the time-mean velocity.

MEASURES_VEL_DIR = DATA_DIR / "NSIDC-0754" / "1996.01.01"
MEASURES_PHASE_NC = MEASURES_VEL_DIR / "antarctic_ice_vel_phase_map_v01.nc"

BM_DIR = DATA_DIR / "bedmachine"
BEDMACHINE_NC = BM_DIR / "BedMachineAntarctica-v3.nc"

# DTU22 MDT stops at -79degS; this sector (~-84degS) gets NO MDT correction.
# Paths kept only for cross-basin config parity — tilt_fit here is geoid-only.
MDT_DIR = DATA_DIR / "MDT"
DTU22_MDT_XYZ = MDT_DIR / "dtuuh22mdt.xyz"

RACMO_DIR = DATA_DIR / "RACMO"
RACMO_SMB_NC = RACMO_DIR / "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202512.nc"
RACMO_MASK_NC = RACMO_DIR / "ANT11_masks.nc"

# pyTMD tide model. CATS2008 is circum-Antarctic and the Ross Sea is its
# native, best-validated domain (matches beardmore / mcmurdo).
TIDE_MODEL = "CATS2008"
TIDE_MODEL_DIR = MAIN_DIR

# ---------------------------------------------------------------------------
# Study window
# ---------------------------------------------------------------------------
SHELF = "Beardmore_Shelf"
# IS2 era. End is the s2s041 strip-index ceiling (acqdate1 max = 2024-01-09).
# Env-overridable for the combined 2012-2024 stack lever: BEARDMORE_SHELF_START
# (e.g. 2012-11-18, the pre-IS2 align --start) widens the window that every
# downstream stage reads from config — build_stack's [START,END) filter and
# output filename, tilt_fit's load window + filenames, load_stack_res,
# ERA5_CACHE_NC, and the diagnostics. Defaults reproduce the IS2-era
# production window byte-for-byte. Mirrors fetch_strips' BEARDMORE_SHELF_FETCH_*.
START_TIME = os.environ.get("BEARDMORE_SHELF_START", "2019-01-01")
END_TIME = os.environ.get("BEARDMORE_SHELF_END", "2024-01-10")

# Target grid. 125 m is the production default (user call 2026-07-05): the
# 25 m grid made the tilt LSQ a ~21 h / 46M-unknown solve and would have put
# the path solver over 23M cells; 125 m matches the venable production
# config. The first-run 25 m products were renamed to the `_25m` variant
# (suffix convention) and remain loadable via `--res 25`.
RES = 125  # meters

# Empty until the first tilt_fit + find_bad_epochs pass identifies
# epochs the LSQ couldn't lock.
BAD_EPOCHS: tuple[str, ...] = ()

# Per-DEM bad-strip rejection (Shean-faithful, keyed on the stack's dem_id
# coord). Populated by beardmore_shelf.scripts.find_bad_epochs after the first
# tilt_fit; preferred over date-keyed BAD_EPOCHS (spares clean same-day
# siblings). See project_strip_level_dropping_2026_06_20.
# Two tiers (both keyed on dem_id, dropped identically at load):
#   tilt-space  — find_bad_epochs, 2026-07-04 Stage A (4x zero static-control
#                 coverage / NaN weight, 2x sparse+low-weight |resid| 0.6-0.9m)
#   melt-space  — diag_stripe_strips, 2026-07-07 (per-epoch thickness-bias LSQ
#                 over the vertical stripe bands; see project_melt_space_strip_qc)
BAD_STRIPS: tuple[str, ...] = (
    # --- tilt-space tier (find_bad_epochs 2026-07-04) ---
    "SETSM_s2s041_WV01_20201112_10200100A1762B00_10200100A1936B00_2m_lsf_seg1",
    "SETSM_s2s041_WV01_20211119_10200100BA01BE00_10200100BA211600_2m_lsf_seg2",
    "SETSM_s2s041_WV01_20220206_10200100BEA46700_10200100BFA51200_2m_lsf_seg1",
    "SETSM_s2s041_WV01_20231018_10200100E084F700_10200100E3A13400_2m_seg1",
    "SETSM_s2s041_WV02_20210215_10300100B3677200_10300100B571A800_2m_lsf_seg1",
    "SETSM_s2s041_WV02_20230220_10300100E38C8A00_10300100E394CB00_2m_seg1",
    # --- melt-space tier (diag_stripe_strips 2026-07-07): HELD, do NOT drop ---
    # The RED near-vertical accretion stripe is carried by the two strips below,
    # but per user 2026-07-07 we HOLD on removing data: with the large pre-IS2
    # CryoTEMPO epochs added to the tilt fit, the joint fit + per-pixel median
    # should CORRECT (not just drop) these scenes. Revisit after the
    # ASP_ctempoatm align lands + a combined-stack re-tilt, by re-running
    # `python -m beardmore_shelf.diag_stripe_strips` on the combined stack.
    # Candidates (kept for reference; re-enable only if still flagged then):
    #   "SETSM_s2s041_WV03_20230110_104001007DCB1900_104001007E3C0000_2m_seg1"      # +19.4 m ice-eq, RED stripe
    #   "SETSM_s2s041_WV02_20211125_10300100C95CB700_10300100C95D3A00_2m_lsf_seg1"  # +14.2 m ice-eq, 2nd RED strip
)

# IS2 control filter parameters (Shean 2019 TC 13:2633 Sect. 2.2.1).
IS2_TIME_WINDOW_DAYS = 365
IS2_DISPLACEMENT_BUDGET_M = 10.0

# CryoTEMPO Land Ice (TDP_LI) control gates — the CS2 control base
# (supersedes raw SARIn L2; see reference_cryotempo_li / ctempo A/B verdict).
# Halved vs the IS2 criteria (coarser footprint, LMC retracker noise);
# mirrors pig/config.py.
CRYOTEMPO_TIME_WINDOW_DAYS = IS2_TIME_WINDOW_DAYS // 2              # 182 d
CRYOTEMPO_DISPLACEMENT_BUDGET_M = IS2_DISPLACEMENT_BUDGET_M / 2.0   # 5.0 m
IS2_MAX_SPEED_MYR = 1.0e9

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------


def ensure_output_dirs():
    """Create output directories if missing."""
    for d in (FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, CLIMATE_CACHE_DIR,
              ASP_ROOT, STRIP_ALIGNED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Cache file for the one-shot ERA5 surface-pressure bulk pull.
ERA5_CACHE_NC = CLIMATE_CACHE_DIR / f"era5_pressure_{START_TIME}_{END_TIME}.nc"
