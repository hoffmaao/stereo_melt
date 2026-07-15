"""Nansen Ice Shelf study configuration.

All paths and scalar parameters specific to the Nansen Ice Shelf basal
melt-rate study. Mirrors :mod:`beardmore.config`. The library carries no
hardcoded paths; this file is the single point where Nansen-specific
context lives.

Nansen Ice Shelf is in East Antarctica (Terra Nova Bay region), feeds
into the Drygalski Ice Tongue's neighbourhood, and is ~1942 km^2 in
extent. The AOI shapefile was extracted from the MEaSUREs
``IceShelf_Antarctica_v02`` layer.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Workspace paths
# ---------------------------------------------------------------------------
MAIN_DIR = Path("/wd2/projects/stereo_melt")
BASIN_DIR = MAIN_DIR / "nansen"

# Study outputs (gitignored)
FIGURES_DIR = BASIN_DIR / "figures"
PROCESSED_DIR = BASIN_DIR / "processed"
RESULTS_DIR = BASIN_DIR / "results"
CLIMATE_CACHE_DIR = BASIN_DIR / "data" / "climate"

# ---------------------------------------------------------------------------
# Shared data cache (same as Beardmore — pan-Antarctic rasters live here).
# ---------------------------------------------------------------------------
DATA_DIR = MAIN_DIR / "data"

REMA_DIR = DATA_DIR / "REMA"
STRIPS_DIR = REMA_DIR / "strips"
MOSAIC_DIR = REMA_DIR / "mosaic"

# Per-basin ASP working tree. The shared STRIPS_DIR holds raw REMA
# downloads (basin-agnostic), but every ASP output -- aligned DEMs,
# pc_align logs, geodiff residuals, IS2 / CS2 caches, combined-reference
# CSVs -- is basin-local so two basins never collide on the same files.
ASP_ROOT = BASIN_DIR / "data" / "ASP"
STRIP_ALIGNED_DIR = ASP_ROOT / "asp_aligned"

# Aligned-strip sources for pc_align quality aggregation
# (aggregate_basin_quality -> find_bad_epochs / assemble_bad_strips_pretilt).
# One (dir, variant) tuple; the variant label is cosmetic metadata on the
# parsed per-strip end-error CSVs. Mirrors dotson_crosson / mcmurdo.
STRIP_SOURCES: list[tuple["Path", str]] = [(STRIP_ALIGNED_DIR, "is2cs2")]

SHAPE_DIR = DATA_DIR / "shapefiles"
# Canonical wider stack/tilt AOI: rectangular AOI derived from the v15
# algorithm (greedy expansion subject to TS-area target ≥ 5000 km²,
# asymmetric upstream + perpendicular bias from shelf-only flow
# direction). Built so the tilt-fit LSQ has interior leverage beyond
# the narrow shelf. Used for strip discovery, alignment, control
# caches, build_stack, and tilt-fit alike. The old narrow nansen.shp
# shelf polygon was archived 2026-05-16.
NANSEN_AOI_SHP = SHAPE_DIR / "nansen_stack_extent.shp"
# Back-compat alias; new code should reference NANSEN_AOI_SHP directly.
NANSEN_STACK_AOI_SHP = NANSEN_AOI_SHP
# True ice-shelf boundary (MEaSUREs IceShelf_Antarctica_v02) for clipping
# melt-rate viz / interpretation plots to the shelf proper. Use with
# ICESHELF_FEATURE_NAME to extract the Nansen feature.
ICESHELF_FEATURE_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ICESHELF_FEATURE_NAME = "Nansen"
GROUNDING_LINE_SHP = SHAPE_DIR / "GroundingLine_Antarctica_v02.shp"
ICE_SHELF_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ROCK_POLYGONS_SHP = SHAPE_DIR / "rock_bedmachine.shp"
LOW_VELOCITY_POLYGONS_SHP = SHAPE_DIR / "low_velocity_polygons.shp"
MOSAIC_INDEX_SHP = SHAPE_DIR / "REMA_Mosaic_Index_v2_32m.shp"
STRIP_INDEX_SHP = SHAPE_DIR / "REMA_Strip_Index_s2s041.parquet"

VEL_DIR = DATA_DIR / "ITsLIVE"
# Annual ITS_LIVE mosaics — Nansen sits in RGI18 (East Antarctica). Files
# are not yet on disk; MEaSUREs phase map (NSIDC-0754) is the time-mean
# fallback and works pan-Antarctic.

MEASURES_VEL_DIR = DATA_DIR / "NSIDC-0754" / "1996.01.01"
MEASURES_PHASE_NC = MEASURES_VEL_DIR / "antarctic_ice_vel_phase_map_v01.nc"

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
SHELF = "Nansen"
# Same ICESat-2-era start as Beardmore so ASP coregistration always has
# altimetric control. End date matches the s2s041 strip index release.
START_TIME = "2019-01-01"
END_TIME = "2023-03-01"

# Target grid
RES = 25  # meters

# Epochs whose per-epoch tilt residual blows up the joint LSQ fit and
# bleeds into the shared per-pixel intercept/dh/dt. Even with per-epoch
# tilt parameters, Shean's joint LSQ couples epochs through the
# per-pixel time-mean and trend, so one wild epoch warps everyone's
# corrected elevation. Started with 2021-01-14 (corner-residual ~2.3 km);
# 2021-01-01 (~5.5 km) and 2020-01-03 (~4.9 km) are even worse and
# likely need to be added — see beardmore artifact diagnostic flow.
BAD_EPOCHS = (
    "2021-01-14",
    # 2026-05-03: added after diagnose_dhdt + find_bad_epochs revealed the
    # Eulerian / Lagrangian sign flip. Selection rule: post-tilt-fit
    # |median residual| over static-control > 0.5 m AND frac_static < 0.10
    # (epoch's static-control population can't anchor αz, so its residual
    # blows out the basin-mean OLS dh/dt). See project_nansen_dhdt_bias.
    "2021-01-01",  # resid=-200.0 m, frac=0.017, w=0.22
    "2020-01-03",  # resid=-71.3  m, frac=0.023, w=0.26
    "2020-02-10",  # resid=+41.4  m, frac=0.006, w=0.07
    "2022-09-25",  # resid=-1.1   m, frac=0.021, w=0.19
    # 2026-05-05: extended filter (now in stereo_melt.coregister.tilt_qc)
    # added a marginal candidate. Residual floor on Nansen is 0.053 m so
    # this is small but consistent with the same selection rule applied
    # across all basins.
    "2022-12-30",  # resid=+0.78 m, frac=0.001, w=0.477, [sparse]
)

# Per-DEM bad-strip rejection (Shean-faithful successor to BAD_EPOCHS, keyed on
# the stack's dem_id coord, so it spares clean same-day siblings that date-keyed
# BAD_EPOCHS over-drops). Empty until the dem_id stack is rebuilt and
# nansen.scripts.assemble_bad_strips_pretilt is run; then paste its list here
# and set BAD_EPOCHS = (). See project_strip_level_dropping_2026_06_20. Until
# then BAD_EPOCHS above stays active and BAD_STRIPS is a no-op on the date-only
# stack (load_basin_stack warns + skips it).
BAD_STRIPS: tuple[str, ...] = ()

# IS2 control filter parameters — Shean 2019 (TC 13:2633) Sect. 2.2.1 rule.
# Cap per-photon advection error |v|·|Δt| at 10 m within a ±1 yr outer
# time cap. Adapts the effective time window to local velocity.
IS2_TIME_WINDOW_DAYS = 365
IS2_DISPLACEMENT_BUDGET_M = 10.0
IS2_MAX_SPEED_MYR = 1.0e9

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------


def ensure_output_dirs():
    """Create output directories if missing."""
    for d in (FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, CLIMATE_CACHE_DIR, ASP_ROOT, STRIP_ALIGNED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Cache file for the one-shot ERA5 surface-pressure bulk pull. The
# stack window is encoded so that a different START/END produces a
# different cache file (no stale-cache risk). The file is written by
# stereo_melt.pipeline.bulk_fetch_era5_pressure_window once per basin
# window; per-strip IBE is then a cheap interpolation off the cube,
# never another CDS request.
ERA5_CACHE_NC = CLIMATE_CACHE_DIR / f"era5_pressure_{START_TIME}_{END_TIME}.nc"
