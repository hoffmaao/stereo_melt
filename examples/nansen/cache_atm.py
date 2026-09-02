"""Pre-cache IceBridge ATM (ILATM2) GCPs for the Nansen Ice Shelf stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_airborne` for
``source='atm'``. Per-strip filtered CSVs land under the per-basin ASP
root (``nansen/data/ASP/atm_data/``); the upstream NSIDC granule store is
shared across all basins under ``<MAIN_DIR>/data/ATM/granules/``.

Run:

    python -m nansen.cache_atm                        # all in-window strips
    python -m nansen.cache_atm --prefetch-only        # warm shared granule store
    python -m nansen.cache_atm --prefetch-only --prefetch-dry-run  # CMR query only

Use ``NANSEN_FETCH_START`` / ``NANSEN_FETCH_END`` env vars to override the
strip + prefetch window without editing :mod:`nansen.config`.
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

from nansen import config


def _load_aoi():
    with fiona.open(config.NANSEN_STACK_AOI_SHP) as src:
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
        fetch_window_env=("NANSEN_FETCH_START", "NANSEN_FETCH_END"),
        description="Cache IceBridge ATM (ILATM2) GCPs for the Nansen stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
