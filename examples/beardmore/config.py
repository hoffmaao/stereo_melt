"""Beardmore Glacier study configuration.

All paths and scalar parameters specific to the Beardmore grounding-zone
melt-rate study. The `stereo_melt` library itself carries no hardcoded
paths; this file is the single point where study context lives.
"""

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
# Shared data cache — pan-Antarctic rasters + this basin's raw inputs.
# For now everything lives under the workspace-level data/ directory; split
# into a dedicated /wd2/shared/antarctic/ cache later if a second basin comes
# online.
# ---------------------------------------------------------------------------
DATA_DIR = MAIN_DIR / "data"

REMA_DIR = DATA_DIR / "REMA"
STRIPS_DIR = REMA_DIR / "strips"
# Fused-stack precedence. build_stack walks this list in order and, for each
# source-granule basename, takes the first occurrence — preferring the
# is2cs2 two-control alignment (IS2 era only) and falling back to cs2 single-
# control alignment (which is also the only option pre-IS2). The is2-only
# variant (ASP/asp_aligned) is intentionally omitted: every is2-only strip
# is also present under is2cs2 or cs2 with stronger control.
STRIP_SOURCES: list[tuple["Path", str]] = [
    (STRIPS_DIR / "ASP_is2cs2" / "asp_aligned", "is2cs2"),
    (STRIPS_DIR / "ASP_cs2" / "asp_aligned", "cs2"),
]
# Backward-compat for downstream tools (tilt_fit, run_melt, etc.) that still
# consume a single aligned-DEM directory. Points at the highest-precedence
# source.
STRIP_ALIGNED_DIR = STRIP_SOURCES[0][0]
MOSAIC_DIR = REMA_DIR / "mosaic"

SHAPE_DIR = DATA_DIR / "shapefiles"
# Canonical wider stack/tilt AOI: hand-drawn narrow grounding-zone AOI
# extended ~30 km northward (less-negative y, upstream/inland into the
# Queen Alexandra Range — 80% grounded ice + 12% rock outcrop per
# BedMachine v3) so the joint LSQ in fit_tilt_stack has stable surfaces
# to anchor per-pixel intercepts. Beardmore flows toward grid-south
# (more-negative y) into the Ross Ice Shelf, so grounded ice sits on
# the grid-NORTH side. Used for strip discovery, alignment, control
# caches, build_stack, and tilt-fit alike. The old narrow
# beardmore_gz.shp was archived 2026-05-16.
BEARDMORE_AOI_SHP = SHAPE_DIR / "beardmore_stack_extent.shp"
# Back-compat alias; new code should reference BEARDMORE_AOI_SHP directly.
BEARDMORE_STACK_AOI_SHP = BEARDMORE_AOI_SHP
# Beardmore Glacier is not in MEaSUREs IceShelf_Antarctica_v02 (it's a
# grounded outlet glacier discharging into Ross Ice Shelf, not its own
# named shelf). No true-shelf polygon for viz clipping; use the wider
# AOI rectangle for plot bounds.
ICESHELF_FEATURE_SHP = None
ICESHELF_FEATURE_NAME = None
GROUNDING_LINE_SHP = SHAPE_DIR / "GroundingLine_Antarctica_v02.shp"
ICE_SHELF_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ROCK_POLYGONS_SHP = SHAPE_DIR / "rock_bedmachine.shp"
LOW_VELOCITY_POLYGONS_SHP = SHAPE_DIR / "low_velocity_polygons.shp"
MOSAIC_INDEX_SHP = SHAPE_DIR / "REMA_Mosaic_Index_v2_32m.shp"
STRIP_INDEX_SHP = SHAPE_DIR / "REMA_Strip_Index_s2s041.parquet"

VEL_DIR = DATA_DIR / "ITsLIVE"
# ITS_LIVE annual v02 mosaics, 120 m resolution — one file per year.
# ITS_LIVE has large coverage gaps over slow-flow grounding zones (feature-
# tracking does poorly on featureless snow); for Beardmore the annual tile
# has ~0% finite data. We fall back to MEaSUREs (NSIDC-0754) which has
# ~97% coverage here at 450 m resolution.
ITS_LIVE_2019 = VEL_DIR / "ITS_LIVE_velocity_120m_RGI19A_2019_v02.nc"
ITS_LIVE_2020 = VEL_DIR / "ITS_LIVE_velocity_120m_RGI19A_2020_v02.nc"
ITS_LIVE_2021 = VEL_DIR / "ITS_LIVE_velocity_120m_RGI19A_2021_v02.nc"

MEASURES_VEL_DIR = DATA_DIR / "NSIDC-0754" / "1996.01.01"
MEASURES_PHASE_NC = MEASURES_VEL_DIR / "antarctic_ice_vel_phase_map_v01.nc"

# Candidate replacement for the phase map: NSIDC-0525 (Scheuchl, Mouginot,
# & Rignot 2012, *Central Antarctica Ice Velocity*) -- a different InSAR +
# speckle-tracking processing chain over central Antarctica at 900 m. We
# keep the 2009 mosaic alone (1997 is intentionally not extracted) for use
# as the climatological surface velocity in the melt inversion. The file
# has no x/y coord axes; reconstruct them from the metadata
# `xmin = -1422100 m`, `ymax = +1443700 m`, `spacing = 900 m` (EPSG:3031).
NSIDC_0525_DIR = BASIN_DIR / "data" / "NSIDC-0525"
NSIDC_0525_2009_NC = NSIDC_0525_DIR / "Central_Antarctica_ice_velocity_2009.nc"

BM_DIR = DATA_DIR / "bedmachine"
BEDMACHINE_NC = BM_DIR / "BedMachineAntarctica-v3.nc"

MDT_DIR = DATA_DIR / "MDT"
DTU22_MDT_XYZ = MDT_DIR / "dtuuh22mdt.xyz"

RACMO_DIR = DATA_DIR / "RACMO"
RACMO_SMB_NC = RACMO_DIR / "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202512.nc"
RACMO_MASK_NC = RACMO_DIR / "ANT11_masks.nc"

# pyTMD model selection. Antarctic shelves use CATS2008. Greenland /
# Norwegian / Arctic Canada drivers should swap to Arc5km2018 or
# TPXO9-atlas-v5; everything in stereo_melt.pipeline reads these two
# values (no other code path is allowed to hardcode CATS2008).
#
# TIDE_MODEL_DIR is the *parent* directory containing the model
# subdirectory -- pyTMD's pathfinder appends the model name itself.
# CATS2008 grid files therefore live at TIDE_MODEL_DIR/CATS2008/...
TIDE_MODEL = "CATS2008"
TIDE_MODEL_DIR = MAIN_DIR

# ---------------------------------------------------------------------------
# Study window
# ---------------------------------------------------------------------------
SHELF = "Beardmore"
# Window pushed back to 2013 to cover the pre-IS2 era. CS2 SARIn POCA is the
# only altimetric control available before Oct 2018; cs2-only pc_align is
# known to leave a per-strip vertical offset relative to is2-anchored strips
# (see literature/inverse_method_explainer.md §8 and the rock-vs-ice bias
# diagnosis), to be calibrated at the era boundary using the 2019-2023
# overlap where both controls exist. Velocity falls back to MEaSUREs phase
# map (NSIDC-0754, time-mean) for the pre-2019 era since ITS_LIVE annual
# mosaics on disk only cover 2019-2021.
START_TIME = "2013-01-01"
END_TIME = "2023-03-01"

# Target grid
RES = 25  # meters

# Epochs whose per-epoch tilt residual blows up the LSQ fit
# (|αx|·R/2 + |αy|·R/2 + |αz| > ~150 m at the domain corner). The
# tilt_fit per-epoch LSQ tries to absorb these as a single planar tilt
# and fails — see beardmore/figures/diagnose_artifacts.png. Dropping
# them outright is more honest than continuing to fit through them.
#
# A second batch (catastrophic vertical bias, identified by MAD-based
# screen of per-epoch median bias on rock pixels vs the median-of-epochs
# reference) was added 2026-04-27. These strips made it through ASP
# pc_align but the post-coreg vertical residual on rock is hundreds of
# meters, so they cannot be reconciled by any planar tilt fit.
BAD_EPOCHS = (
    "2019-01-30",
    "2019-10-13",
    "2019-12-03",  # rock bias  -441 m
    "2020-02-24",  # rock bias +2033 m
    "2020-10-23",
    "2020-12-30",  # rock bias  -910 m
    "2021-11-07",
    "2021-11-21",  # rock bias   -33 m
    "2021-11-25",
    "2022-12-08",
    "2023-01-10",  # zero rock overlap (cannot anchor coregistration)
    "2023-02-16",  # tilt-fit residuals up to +908 m on the floating tongue
                    # (99th pct |h_anom| 31.5 m vs max 908 m); poisons v2
                    # infill at the (90000, -692000) hole — diagnosed 2026-04-28
    # 2026-05-03: added after applying nansen-style framework. Both epochs
    # have weight_mean ≈ 0 in the IRLS — Tukey biweight rejected the bulk
    # of obs, indicating tilt-fit cannot lock these strips.
    "2017-11-19",  # weight_mean=0.000, weight_frac_kept=0.000 (failed fit)
    "2015-11-04",  # weight_mean=0.081, weight_frac_kept=0.156
    # 2026-05-05: added from beardmore.scripts.find_bad_epochs.
    # Initial Nansen-rule pass caught these two (sparse static-control):
    "2015-11-12",  # resid=-9.00 m, frac=0.001, w=nan, [sparse+nan-w]
    "2016-11-09",  # resid=-1.92 m, frac=0.058, w=0.373, [sparse]
    # Expanded filter (added IRLS-failure + low-weight clauses to the
    # shared tilt_qc.suggest_bad_epochs) caught five more on Beardmore.
    # These have high static-control coverage but the IRLS step itself
    # failed (weight_mean=NaN) or rejected the bulk of obs (low w):
    "2017-11-29",  # resid=+92.02 m, frac=0.407, w=nan, [nan-w]
    "2018-12-18",  # resid=+47.32 m, frac=0.321, w=nan, [nan-w]
    "2022-01-19",  # resid=-32.21 m, frac=0.248, w=nan, [nan-w]
    "2015-10-30",  # resid=-12.48 m, frac=0.395, w=0.347, [low-w]
    "2015-11-02",  # resid=+5.33 m, frac=0.119, w=0.152, [low-w]
)

# Per-DEM bad-strip rejection (Shean-faithful successor to BAD_EPOCHS, keyed on
# the stack's dem_id coord, so it spares clean same-day siblings that date-keyed
# BAD_EPOCHS over-drops). Empty until the dem_id stack is rebuilt and
# beardmore.scripts.assemble_bad_strips_pretilt is run; then paste its list here
# and set BAD_EPOCHS = (). See project_strip_level_dropping_2026_06_20. Until
# then BAD_EPOCHS above stays active and BAD_STRIPS is a no-op on the date-only
# stack (load_basin_stack warns + skips it).
BAD_STRIPS: tuple[str, ...] = ()

# IS2 control filter parameters — Shean 2019 (TC 13:2633) Sect. 2.2.1 rule.
# Cap per-photon advection error |v|·|Δt| at 10 m within a ±1 yr outer
# time cap. Adapts the effective time window to local velocity.
# Pre-IS2 strips (2013-2018, before ATL06 launch) will hit "0 ATL03 granules"
# regardless and need CS2 fallback (already cached separately for Beardmore).
IS2_TIME_WINDOW_DAYS = 365
IS2_DISPLACEMENT_BUDGET_M = 10.0
IS2_MAX_SPEED_MYR = 1.0e9


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------


def ensure_output_dirs():
    """Create output directories if missing. Called at start of driver scripts."""
    for d in (FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, CLIMATE_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Cache file for the one-shot ERA5 surface-pressure bulk pull. The
# stack window is encoded so that a different START/END produces a
# different cache file (no stale-cache risk). The file is written by
# stereo_melt.pipeline.bulk_fetch_era5_pressure_window once per basin
# window; per-strip IBE is then a cheap interpolation off the cube,
# never another CDS request.
ERA5_CACHE_NC = CLIMATE_CACHE_DIR / f"era5_pressure_{START_TIME}_{END_TIME}.nc"
