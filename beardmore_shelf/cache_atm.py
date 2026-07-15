"""Pre-cache IceBridge ATM (ILATM2) GCPs for the Beardmore-shelf stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_airborne` for
``source='atm'``, mirroring ``pig.cache_atm``. Per-strip filtered CSVs land
under the canonical per-basin ASP root
(``beardmore_shelf/data/ASP/atm_data/``); the upstream NSIDC granule store
is shared across all basins under ``<MAIN_DIR>/data/ATM/granules/``.

This sector's entire IceBridge ATM history is ONE flight — 2013-11-18
(6 ILATM2 granules; CMR query 2026-07-05, LVIS has zero) — so this cache
only matters for the pre-IS2 2013-era orientation leg. Set the window when
caching outside the config window:

    BEARDMORE_SHELF_FETCH_START=2013-08-20 BEARDMORE_SHELF_FETCH_END=2014-02-17 \
        python -m beardmore_shelf.cache_atm

Other flags mirror pig.cache_atm (--overwrite / --dem-id / --limit /
--time-window / --prefetch-only).
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

from stereo_melt.coregister import cache_airborne

from beardmore_shelf import config


def _load_aoi():
    with fiona.open(config.BEARDMORE_SHELF_AOI_SHP) as src:
        return shape(next(iter(src))["geometry"])


def main() -> None:
    config.ensure_output_dirs()

    cache_dir = config.ASP_ROOT / "atm_data"
    granule_dir = config.MAIN_DIR / "data" / "ATM" / "granules"

    aoi = _load_aoi()
    rc = cache_airborne.run_cache_airborne_main(
        source="atm",
        aoi_polygon=aoi,
        strips_dir=Path(config.STRIPS_DIR),
        cache_dir=cache_dir,
        granule_dir=granule_dir,
        bedmachine_path=config.BEDMACHINE_NC,
        velocity_path=config.MEASURES_PHASE_NC,
        start_time=config.START_TIME,
        end_time=config.END_TIME,
        fetch_window_env=("BEARDMORE_SHELF_FETCH_START", "BEARDMORE_SHELF_FETCH_END"),
        description="Cache IceBridge ATM (ILATM2) GCPs for the Beardmore-shelf stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
