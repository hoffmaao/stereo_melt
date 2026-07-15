"""Pre-cache CryoSat-2 L2 SARIn POCA control points for the Dotson + Crosson stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_cs2`. Per-strip
filtered HDF5 caches land under the per-basin ASP root
(:data:`dotson_crosson.config.ASP_ROOT` / ``cs2_data/``); the upstream
granule and HDR caches are shared across all basins under
``<MAIN_DIR>/data/CS2/``.

Run:

    python -m dotson_crosson.cache_cryosat2                  # all in-window strips
    python -m dotson_crosson.cache_cryosat2 --overwrite       # refresh every cache entry
    python -m dotson_crosson.cache_cryosat2 --dem-id <id>     # one strip
    python -m dotson_crosson.cache_cryosat2 --limit 3         # smoke test first 3
    python -m dotson_crosson.cache_cryosat2 --time-window 60  # ±N day temporal match
    python -m dotson_crosson.cache_cryosat2 --prefetch-only   # warm shared granule store
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# PROJ_DATA fix for this conda env's broken base proj.db.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import fiona
from shapely.geometry import shape

from stereo_melt.coregister import cache_cs2

from dotson_crosson import config


def _load_aoi():
    with fiona.open(config.DC_AOI_SHP) as src:
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
        fetch_window_env=("DC_FETCH_START", "DC_FETCH_END"),
        description="Cache CS2 SARIn POCA control points for the Dotson+Crosson stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
