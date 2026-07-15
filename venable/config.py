"""Venable Ice Shelf study configuration.

All paths and scalar parameters specific to the Venable Ice Shelf
basal-melt-rate study. Mirrors :mod:`mcmurdo.config`.

Venable Ice Shelf (Bellingshausen Sea sector, ~-73.1degS / -87.2degW) is a
small (~3155 km^2) shelf on the Eltanin Bay coast of Ellsworth Land,
West Antarctica. The AOI is a provisional rectangle = the MEaSUREs
'Venable' feature bbox + 30 km margin (~19,200 km^2, EPSG:3031). Replace
with the strip-aware v15 rectangle (scripts/compute_aoi.py) after a first
fetch if tighter static-control capture is wanted.

Latitude (-73degS) is north of the DTU22 -79degS coverage limit, so geoid +
MDT corrections both apply (matches PIG / Nansen / Dotson-Crosson / McMurdo;
not geoid-only like Beardmore).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Workspace paths
# ---------------------------------------------------------------------------
MAIN_DIR = Path("/wd2/projects/stereo_melt")
BASIN_DIR = MAIN_DIR / "venable"

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
STRIP_SOURCES: list[tuple["Path", str]] = [(STRIP_ALIGNED_DIR, "is2cs2")]
MOSAIC_DIR = REMA_DIR / "mosaic"

SHAPE_DIR = DATA_DIR / "shapefiles"
# Provisional rectangular stack/align/fetch AOI (MEaSUREs Venable bbox + 30 km).
# Mirrors the box format the v15 algorithm emits; used for strip discovery,
# alignment, control caches, build_stack, and tilt-fit alike.
VENABLE_AOI_SHP = SHAPE_DIR / "venable_stack_extent.shp"
# Back-compat alias; new code should reference VENABLE_AOI_SHP directly.
VENABLE_STACK_AOI_SHP = VENABLE_AOI_SHP
# Narrow shelf polygon — input to scripts/compute_aoi.py (v15 optimal-AOI sizer),
# which expands it into VENABLE_AOI_SHP (the wider stack_extent rectangle).
SHELF_INPUT_SHP = SHAPE_DIR / "venable.shp"
# Venable IS a named feature in MEaSUREs IceShelf_Antarctica_v02 (NAME field),
# so it can clip viz to the true shelf (unlike McMurdo, an unnamed Ross lobe).
ICESHELF_FEATURE_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ICESHELF_FEATURE_NAME = "Venable"
GROUNDING_LINE_SHP = SHAPE_DIR / "GroundingLine_Antarctica_v02.shp"
ICE_SHELF_SHP = SHAPE_DIR / "IceShelf_Antarctica_v02.shp"
ROCK_POLYGONS_SHP = SHAPE_DIR / "rock_bedmachine.shp"
LOW_VELOCITY_POLYGONS_SHP = SHAPE_DIR / "low_velocity_polygons.shp"
MOSAIC_INDEX_SHP = SHAPE_DIR / "REMA_Mosaic_Index_v2_32m.shp"
STRIP_INDEX_SHP = SHAPE_DIR / "REMA_Strip_Index_s2s041.parquet"

VEL_DIR = DATA_DIR / "ITsLIVE"
# Venable is in the Bellingshausen Sea sector. No ITS_LIVE annual mosaic tile
# for this region is on disk, and the PIG/Thwaites ASE quarterly mosaics do
# NOT extend here; mirror Nansen/McMurdo and use the MEaSUREs phase map
# (NSIDC-0754) as the time-mean velocity. Add ITS_LIVE_* entries here once an
# Antarctic-Peninsula / Bellingshausen tile is downloaded.

MEASURES_VEL_DIR = DATA_DIR / "NSIDC-0754" / "1996.01.01"
MEASURES_PHASE_NC = MEASURES_VEL_DIR / "antarctic_ice_vel_phase_map_v01.nc"

BM_DIR = DATA_DIR / "bedmachine"
BEDMACHINE_NC = BM_DIR / "BedMachineAntarctica-v3.nc"

MDT_DIR = DATA_DIR / "MDT"
DTU22_MDT_XYZ = MDT_DIR / "dtuuh22mdt.xyz"

RACMO_DIR = DATA_DIR / "RACMO"
RACMO_SMB_NC = RACMO_DIR / "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202512.nc"
RACMO_MASK_NC = RACMO_DIR / "ANT11_masks.nc"

# pyTMD tide model. CATS2008 is circum-Antarctic and covers the
# Bellingshausen/Amundsen sector (matches Dotson-Crosson's choice).
TIDE_MODEL = "CATS2008"
TIDE_MODEL_DIR = MAIN_DIR

# ---------------------------------------------------------------------------
# Study window
# ---------------------------------------------------------------------------
SHELF = "Venable"
# IS2 era. End is the s2s041 strip-index ceiling (acqdate1 max = 2024-01-09).
START_TIME = "2019-01-01"
END_TIME = "2024-01-10"

# Target grid
RES = 25  # meters

# Empty until the first tilt_fit + find_bad_epochs pass identifies
# epochs the LSQ couldn't lock.
BAD_EPOCHS: tuple[str, ...] = ()

# Per-DEM bad-strip rejection (Shean-faithful, keyed on the stack's dem_id
# coord). Populated by venable.scripts.find_bad_epochs after the first
# tilt_fit; preferred over date-keyed BAD_EPOCHS (spares clean same-day
# siblings). See project_strip_level_dropping_2026_06_20.
# 34 strips flagged by find_bad_epochs on the 125 m stack (2026-07-03).
BAD_STRIPS: tuple[str, ...] = (
    "SETSM_s2s041_W1W1_20200828_102001009D38A400_10200100A0BE2000_2m_lsf_seg2",
    "SETSM_s2s041_W1W2_20210915_10200100B707B700_10300100C5A83300_2m_lsf_seg2",
    "SETSM_s2s041_W2W2_20231130_10300100F1299200_10300100F1AE5200_2m_seg1",
    "SETSM_s2s041_W2W3_20231008_10300100EE568F00_104001008B8B5200_2m_seg1",
    "SETSM_s2s041_W2W3_20231009_10300100EE568F00_104001008A2B2700_2m_seg1",
    "SETSM_s2s041_WV01_20191125_102001008D2CDF00_1020010090C5B800_2m_lsf_seg1",
    "SETSM_s2s041_WV01_20191214_102001008B650D00_102001008E9A5D00_2m_lsf_seg1",
    "SETSM_s2s041_WV01_20191226_10200100911F8B00_102001009057DE00_2m_lsf_seg3",
    "SETSM_s2s041_WV01_20210227_10200100A27A3300_10200100A324E600_2m_lsf_seg1",
    "SETSM_s2s041_WV01_20211113_10200100BA368B00_10200100BBC69C00_2m_lsf_seg4",
    "SETSM_s2s041_WV01_20211113_10200100BA368B00_10200100BBC69C00_2m_lsf_seg5",
    "SETSM_s2s041_WV01_20211113_10200100BA368B00_10200100BBC69C00_2m_lsf_seg6",
    "SETSM_s2s041_WV01_20211113_10200100BA368B00_10200100BBC69C00_2m_lsf_seg7",
    "SETSM_s2s041_WV01_20211113_10200100BA368B00_10200100BBC69C00_2m_lsf_seg8",
    "SETSM_s2s041_WV01_20211118_10200100BA49D800_10200100BAD14F00_2m_lsf_seg1",
    "SETSM_s2s041_WV01_20220114_10200100BD710A00_10200100BE349000_2m_lsf_seg1",
    "SETSM_s2s041_WV01_20230119_10200100D12ACC00_10200100D1B5A500_2m_seg2",
    "SETSM_s2s041_WV02_20201001_10300100AD82AE00_10300100AF6BA000_2m_lsf_seg2",
    "SETSM_s2s041_WV02_20201007_10300100B0627C00_10300100B15F0E00_2m_lsf_seg1",
    "SETSM_s2s041_WV02_20210317_10300100BB1D6400_10300100BB88FD00_2m_lsf_seg1",
    "SETSM_s2s041_WV02_20211201_10300100C957FD00_10300100C95DEF00_2m_lsf_seg3",
    "SETSM_s2s041_WV02_20211226_10300100CA44A900_10300100CB64AE00_2m_lsf_seg1",
    "SETSM_s2s041_WV02_20220105_10300100CB21A100_10300100CCABDF00_2m_lsf_seg1",
    "SETSM_s2s041_WV02_20220105_10300100CB683400_10300100CB6A0E00_2m_lsf_seg1",
    "SETSM_s2s041_WV03_20190910_10400100518D8B00_1040010051210A00_2m_lsf_seg1",
    "SETSM_s2s041_WV03_20201105_10400100614E6000_10400100616B4100_2m_lsf_seg1",
    "SETSM_s2s041_WV03_20201105_10400100614E6000_10400100616B4100_2m_lsf_seg2",
    "SETSM_s2s041_WV03_20220103_10400100710DDF00_10400100724A7300_2m_lsf_seg1",
    "SETSM_s2s041_WV03_20221027_104001007B48CD00_104001007D88FA00_2m_lsf_seg1",
    "SETSM_s2s041_WV03_20221027_104001007B48CD00_104001007D88FA00_2m_lsf_seg3",
    "SETSM_s2s041_WV03_20221027_104001007B48CD00_104001007D88FA00_2m_lsf_seg4",
    "SETSM_s2s041_WV03_20221106_104001007D818C00_104001007E9DEB00_2m_lsf_seg2",
    "SETSM_s2s041_WV03_20221126_104001007E077E00_104001007F6E9C00_2m_seg1",
    "SETSM_s2s041_WV03_20230327_10400100836C4100_1040010084528200_2m_seg1",
    # --- melt-space stripe QC additions (diag_stripe_strips, 2026-07-04) ---
    # +41..+52 m ice-eq (~+4.5 m surface) bias; most-anomalous pairs shelf-wide
    "SETSM_s2s041_WV02_20190923_1030010098167300_103001009B61D800_2m_lsf_seg1",
    # +31.8/+22.3 m bias in stripe bands A/B and +31.1 shelf-wide as later
    # member; painted the rows-648-659 accretion stripe (3.1k-cell pairs);
    # sibling seg2 of this 2021-09-15 acquisition already curated out
    "SETSM_s2s041_W1W2_20210915_10200100B6203400_10300100C5A83300_2m_lsf_seg1",
)

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
