"""Dotson + Crosson combined ice-shelf study configuration.

All paths and scalar parameters specific to the combined Dotson + Crosson
ice-shelf basal-melt-rate study. Mirrors :mod:`pig.config` (same Amundsen
Sea sector, same ITS_LIVE RGI19A tiles, same s2s041 strip-index ceiling).

Dotson and Crosson are adjacent ice shelves in the Amundsen Sea Embayment,
West Antarctica, fed by Smith / Pope / Kohler glaciers. Combined floating
extent ~9000 km^2; the AOI is the union of MEaSUREs
``IceShelf_Antarctica_v02`` features ``"Dotson"`` and ``"Crosson"``.

Latitude is well north of the DTU22 -79°S limit, so geoid + MDT
corrections both apply.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Workspace paths
# ---------------------------------------------------------------------------
MAIN_DIR = Path("/wd2/projects/stereo_melt")
BASIN_DIR = Path(__file__).resolve().parent

FIGURES_DIR = BASIN_DIR / "figures"
PROCESSED_DIR = BASIN_DIR / "processed"
RESULTS_DIR = BASIN_DIR / "results"
CLIMATE_CACHE_DIR = BASIN_DIR / "data" / "climate"

# ---------------------------------------------------------------------------
# Shared data cache (pan-Antarctic rasters, same as Beardmore/Nansen/PIG).
# ---------------------------------------------------------------------------
DATA_DIR = MAIN_DIR / "data"

REMA_DIR = DATA_DIR / "REMA"
STRIPS_DIR = REMA_DIR / "strips"
ASP_ROOT = BASIN_DIR / "data" / "ASP"
STRIP_ALIGNED_DIR = ASP_ROOT / "asp_aligned"
STRIP_SOURCES: list[tuple["Path", str]] = [(STRIP_ALIGNED_DIR, "is2cs2")]
MOSAIC_DIR = REMA_DIR / "mosaic"

SHAPE_DIR = DATA_DIR / "shapefiles"
# Canonical wider stack/tilt AOI: 30 km buffered union of Dotson +
# Crosson shelves, captures the coupled grounding zones plus upstream
# slow-grounded control. Used for strip discovery, alignment, control
# caches, build_stack, and tilt-fit alike. The old narrow
# dotson_crosson.shp union polygon was archived 2026-05-16.
DC_AOI_SHP = SHAPE_DIR / "dotson_crosson_stack.shp"
# Back-compat alias; new code should reference DC_AOI_SHP directly.
DC_STACK_AOI_SHP = DC_AOI_SHP
# True ice-shelf boundaries (MEaSUREs IceShelf_Antarctica_v02) for
# clipping melt-rate viz / interpretation plots to the shelves proper.
# Use with ICESHELF_FEATURE_NAMES to extract both Dotson and Crosson.
ICESHELF_FEATURE_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ICESHELF_FEATURE_NAMES = ("Dotson", "Crosson")
GROUNDING_LINE_SHP = SHAPE_DIR / "GroundingLine_Antarctica_v02.shp"
ICE_SHELF_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ROCK_POLYGONS_SHP = SHAPE_DIR / "rock_bedmachine.shp"
LOW_VELOCITY_POLYGONS_SHP = SHAPE_DIR / "low_velocity_polygons.shp"
MOSAIC_INDEX_SHP = SHAPE_DIR / "REMA_Mosaic_Index_v2_32m.shp"
STRIP_INDEX_SHP = SHAPE_DIR / "REMA_Strip_Index_s2s041.parquet"

VEL_DIR = DATA_DIR / "ITsLIVE"
# Same Amundsen Sea sector as PIG -- ITS_LIVE RGI19A annual mosaics work
# well over fast-flowing ice. Kohler/Smith/Pope drainage feeds the shelves.
ITS_LIVE_2019 = VEL_DIR / "ITS_LIVE_velocity_120m_RGI19A_2019_v02.nc"
ITS_LIVE_2020 = VEL_DIR / "ITS_LIVE_velocity_120m_RGI19A_2020_v02.nc"
ITS_LIVE_2021 = VEL_DIR / "ITS_LIVE_velocity_120m_RGI19A_2021_v02.nc"

MEASURES_VEL_DIR = DATA_DIR / "NSIDC-0754" / "1996.01.01"
MEASURES_PHASE_NC = MEASURES_VEL_DIR / "antarctic_ice_vel_phase_map_v01.nc"

BM_DIR = DATA_DIR / "bedmachine"
BEDMACHINE_NC = BM_DIR / "BedMachineAntarctica-v3.nc"

MDT_DIR = DATA_DIR / "MDT"
DTU22_MDT_XYZ = MDT_DIR / "dtuuh22mdt.xyz"

RACMO_DIR = DATA_DIR / "RACMO"
RACMO_SMB_NC = RACMO_DIR / "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202512.nc"
RACMO_MASK_NC = RACMO_DIR / "ANT11_masks.nc"

# CATS2008 pan-Antarctic tide model.
TIDE_MODEL = "CATS2008"
TIDE_MODEL_DIR = MAIN_DIR

# ---------------------------------------------------------------------------
# Study window
# ---------------------------------------------------------------------------
SHELF = "DotsonCrosson"
START_TIME = "2019-01-01"
END_TIME = "2024-01-10"  # s2s041 ceiling; same cap as PIG.

RES = 25  # meters

BAD_EPOCHS: tuple[str, ...] = ()

# Per-DEM bad-strip rejection (Shean-faithful, keyed on the stack's dem_id
# coord). Populated by dotson_crosson.scripts.find_bad_epochs after the first
# tilt_fit; preferred over date-keyed BAD_EPOCHS (spares clean same-day
# siblings). See project_strip_level_dropping_2026_06_20.
BAD_STRIPS: tuple[str, ...] = ()

IS2_TIME_WINDOW_DAYS = 365
IS2_DISPLACEMENT_BUDGET_M = 10.0
IS2_MAX_SPEED_MYR = 1.0e9

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------


def ensure_output_dirs():
    for d in (
        FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, CLIMATE_CACHE_DIR,
        ASP_ROOT, STRIP_ALIGNED_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


ERA5_CACHE_NC = CLIMATE_CACHE_DIR / f"era5_pressure_{START_TIME}_{END_TIME}.nc"
