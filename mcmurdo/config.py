"""McMurdo Ice Shelf study configuration.

All paths and scalar parameters specific to the McMurdo Ice Shelf
basal-melt-rate study. Mirrors :mod:`pig.config`.

McMurdo Ice Shelf (Ross Sea sector, ~-78°S / 166°E) is the small wedge
of the Ross Ice Shelf adjacent to Ross Island. AOI delivered as a
user-supplied GeoJSON (single MultiPolygon, EPSG:3031, ~2963 km^2,
bounds ``x ∈ [257767, 351307] m``, ``y ∈ [-1325683, -1251980] m``).

Latitude (-78°S) is north of the DTU22 -79°S coverage limit, so geoid +
MDT corrections both apply (matches PIG / Nansen / Dotson-Crosson; not
geoid-only like Beardmore).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Workspace paths
# ---------------------------------------------------------------------------
MAIN_DIR = Path("/wd2/projects/stereo_melt")
BASIN_DIR = MAIN_DIR / "mcmurdo"

# Study outputs (gitignored)
FIGURES_DIR = BASIN_DIR / "figures"
PROCESSED_DIR = BASIN_DIR / "processed"
RESULTS_DIR = BASIN_DIR / "results"
CLIMATE_CACHE_DIR = BASIN_DIR / "data" / "climate"

# ---------------------------------------------------------------------------
# Shared data cache (same as Beardmore/Nansen/PIG — pan-Antarctic rasters).
# ---------------------------------------------------------------------------
DATA_DIR = MAIN_DIR / "data"

REMA_DIR = DATA_DIR / "REMA"
STRIPS_DIR = REMA_DIR / "strips"
# Per-basin ASP root. Single-source IS2-era basin so one variant label.
ASP_ROOT = BASIN_DIR / "data" / "ASP"
STRIP_ALIGNED_DIR = ASP_ROOT / "asp_aligned"
STRIP_SOURCES: list[tuple["Path", str]] = [(STRIP_ALIGNED_DIR, "is2cs2")]
MOSAIC_DIR = REMA_DIR / "mosaic"

SHAPE_DIR = DATA_DIR / "shapefiles"
# Canonical wider stack/tilt AOI: rectangular AOI computed by the v15
# algorithm (greedy expansion subject to TS-area target, asymmetric
# upstream + perp bias from shelf-only flow direction). See
# scripts/compute_aoi.py. Used for strip discovery, alignment, control
# caches, build_stack, and tilt-fit alike. The old narrow
# mcmurdo.geojson user-supplied AOI was archived 2026-05-16.
MCMURDO_AOI_SHP = SHAPE_DIR / "mcmurdo_stack_extent.shp"
# Back-compat alias; new code should reference MCMURDO_AOI_SHP directly.
MCMURDO_STACK_AOI_SHP = MCMURDO_AOI_SHP
# McMurdo Ice Shelf is the southwestern lobe of the Ross Ice Shelf in
# MEaSUREs IceShelf_Antarctica_v02; not split out as its own feature.
# No true-shelf polygon for viz clipping; use the wider AOI rectangle.
ICESHELF_FEATURE_SHP = None
ICESHELF_FEATURE_NAME = None
GROUNDING_LINE_SHP = SHAPE_DIR / "GroundingLine_Antarctica_v02.shp"
ICE_SHELF_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ROCK_POLYGONS_SHP = SHAPE_DIR / "rock_bedmachine.shp"
LOW_VELOCITY_POLYGONS_SHP = SHAPE_DIR / "low_velocity_polygons.shp"
MOSAIC_INDEX_SHP = SHAPE_DIR / "REMA_Mosaic_Index_v2_32m.shp"
STRIP_INDEX_SHP = SHAPE_DIR / "REMA_Strip_Index_s2s041.parquet"

VEL_DIR = DATA_DIR / "ITsLIVE"
# McMurdo is in the Ross Sea sector. ITS_LIVE annual mosaics for this
# region are not on disk yet; mirror Nansen and use MEaSUREs phase map
# (NSIDC-0754) as the time-mean velocity fallback. Add ITS_LIVE_*
# entries here once an RGI19B (or Ross-subset) tile is downloaded.

MEASURES_VEL_DIR = DATA_DIR / "NSIDC-0754" / "1996.01.01"
MEASURES_PHASE_NC = MEASURES_VEL_DIR / "antarctic_ice_vel_phase_map_v01.nc"

BM_DIR = DATA_DIR / "bedmachine"
BEDMACHINE_NC = BM_DIR / "BedMachineAntarctica-v3.nc"

MDT_DIR = DATA_DIR / "MDT"
DTU22_MDT_XYZ = MDT_DIR / "dtuuh22mdt.xyz"

RACMO_DIR = DATA_DIR / "RACMO"
RACMO_SMB_NC = RACMO_DIR / "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202512.nc"
RACMO_MASK_NC = RACMO_DIR / "ANT11_masks.nc"

# pyTMD model. CATS2008 is calibrated for the Ross Sea — McMurdo's
# obvious choice (matches Nansen which also sits on the Ross periphery).
TIDE_MODEL = "CATS2008"
TIDE_MODEL_DIR = MAIN_DIR

# ---------------------------------------------------------------------------
# Study window
# ---------------------------------------------------------------------------
SHELF = "McMurdo"
# IS2 era. End is the s2s041 strip-index ceiling (acqdate1 max = 2024-01-09).
START_TIME = "2019-01-01"
END_TIME = "2024-01-10"

# Target grid
RES = 25  # meters

# Empty until the first tilt_fit + find_bad_epochs pass identifies
# epochs the LSQ couldn't lock. Mirrors pig.config.BAD_EPOCHS pattern.
BAD_EPOCHS: tuple[str, ...] = ()

# Per-DEM bad-strip rejection (Shean-faithful, keyed on the stack's dem_id
# coord). Populated by mcmurdo.scripts.find_bad_epochs after the first tilt_fit;
# preferred over date-keyed BAD_EPOCHS (spares clean same-day siblings).
# See project_strip_level_dropping_2026_06_20.
BAD_STRIPS: tuple[str, ...] = ()

# IS2 control filter parameters (Shean 2019 TC 13:2633 Sect. 2.2.1).
IS2_TIME_WINDOW_DAYS = 365
IS2_DISPLACEMENT_BUDGET_M = 10.0
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
