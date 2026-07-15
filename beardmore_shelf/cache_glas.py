"""Pre-cache ICESat-1 GLAS (GLAH12) GCPs for the Beardmore-shelf stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_glas`, mirroring
``cache_atm``. Per-strip filtered CSVs land under the canonical per-basin
ASP root (``beardmore_shelf/data/ASP/glas_data/``); the upstream NSIDC
granule store is shared across all basins under
``<MAIN_DIR>/data/GLAS/granules/``.

GLAS is the only laser control that reaches the earliest REMA strips over
this sector (2009-01-22 → 2010-10-11; CryoTEMPO starts 2010-07, ATM here
starts 2013 — see project_glas_era_scoping_2026_07_10). Point the window
at that era when caching outside the config window:

    BEARDMORE_SHELF_FETCH_START=2009-01-01 BEARDMORE_SHELF_FETCH_END=2010-10-12 \
        python -m beardmore_shelf.cache_glas

Other flags mirror cache_atm (--overwrite / --dem-id / --limit /
--time-window [default ±365 d, the Shean ±1 yr control cap] /
--prefetch-only / --no-prefetch).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import fiona
from shapely.geometry import shape

from stereo_melt.coregister import cache_glas

from beardmore_shelf import config


def _load_aoi():
    with fiona.open(config.BEARDMORE_SHELF_AOI_SHP) as src:
        return shape(next(iter(src))["geometry"])


def main() -> None:
    config.ensure_output_dirs()

    cache_dir = config.ASP_ROOT / "glas_data"
    granule_dir = config.MAIN_DIR / "data" / "GLAS" / "granules"

    aoi = _load_aoi()
    rc = cache_glas.run_cache_glas_main(
        aoi_polygon=aoi,
        strips_dir=Path(config.STRIPS_DIR),
        cache_dir=cache_dir,
        granule_dir=granule_dir,
        bedmachine_path=config.BEDMACHINE_NC,
        velocity_path=config.MEASURES_PHASE_NC,
        start_time=config.START_TIME,
        end_time=config.END_TIME,
        fetch_window_env=("BEARDMORE_SHELF_FETCH_START", "BEARDMORE_SHELF_FETCH_END"),
        description="Cache ICESat-1 GLAS (GLAH12) GCPs for the Beardmore-shelf stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
