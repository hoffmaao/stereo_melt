"""Pre-cache CryoSat-2 L2 SARIn POCA control points for the PIG stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_cs2`. See that module
for scope locks, server-layout notes, and the granule-then-perstrip
pipeline. Run:

    python -m pig.cache_cryosat2                       # all in-window strips
    python -m pig.cache_cryosat2 --overwrite            # refresh every cache entry
    python -m pig.cache_cryosat2 --dem-id <id>          # one strip
    python -m pig.cache_cryosat2 --limit 3              # smoke test first 3
    python -m pig.cache_cryosat2 --time-window 60       # ±N day temporal match

Per-strip CS2 filtered CSVs land under the per-basin ASP root
(``pig/data/ASP/cs2_data/``); the upstream NSIDC granule + HDR store is
shared across all basins under ``<MAIN_DIR>/data/CS2/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Point pyproj at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import fiona
from shapely.geometry import shape

from stereo_melt.coregister import cache_cs2

from pig import config


def _load_aoi():
    # Mirror cache_atm / cache_lvis / cache_icesat2 / align_strips /
    # fetch_strips: select strips by the wider stack extent when available
    # so the upstream band that build_stack reprojects has CS2 control.
    aoi_path = config.PIG_AOI_SHP
    with fiona.open(aoi_path) as src:
        return shape(next(iter(src))["geometry"])


def main() -> None:
    config.ensure_output_dirs()

    cache_dir = config.ASP_ROOT / "cs2_data"
    granule_dir = config.MAIN_DIR / "data" / "CS2" / "granules"
    hdr_dir = config.MAIN_DIR / "data" / "CS2" / "hdrs"

    aoi = _load_aoi()
    rc = cache_cs2.run_cache_cs2_main(
        aoi_polygon=aoi,
        strips_dir=Path(config.STRIPS_DIR),
        cache_dir=cache_dir,
        granule_dir=granule_dir,
        hdr_dir=hdr_dir,
        bedmachine_path=config.BEDMACHINE_NC,
        velocity_path=config.MEASURES_PHASE_NC,
        start_time=config.START_TIME,
        end_time=config.END_TIME,
        fetch_window_env=("PIG_FETCH_START", "PIG_FETCH_END"),
        description="Cache CS2 SARIn POCA control points for the Pine Island stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
