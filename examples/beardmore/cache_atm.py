"""Pre-cache IceBridge ATM (ILATM2) GCPs for the Beardmore stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_airborne` for
``source='atm'``. Per-strip filtered CSVs land at
``data/REMA/strips/ASP/atm_data/`` (Beardmore still uses the legacy
shared ASP layout); the upstream NSIDC granule store is shared across
all basins under ``<MAIN_DIR>/data/ATM/granules/``.

Beardmore sits at ~−84°S in the TAM. IceBridge Antarctic campaigns
focused primarily on West Antarctica + the Peninsula; coverage at
Beardmore is expected to be sparse to nonexistent. This wrapper exists
so the per-strip filter still runs (and writes empty/missing sidecars
gracefully) when the upstream airborne signal is thin.

Run:

    python -m beardmore.cache_atm                  # all in-window strips
    python -m beardmore.cache_atm --prefetch-only  # warm shared granule store
    python -m beardmore.cache_atm --time-window 180  # widen temporal half-window
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

from beardmore import config


def _load_aoi():
    with fiona.open(config.BEARDMORE_AOI_SHP) as src:
        return shape(next(iter(src))["geometry"])


def main() -> None:
    config.ensure_output_dirs()

    cache_dir = Path(config.STRIPS_DIR) / "ASP" / "atm_data"
    granule_dir = config.MAIN_DIR / "data" / "ATM" / "granules"
    cache_dir.mkdir(parents=True, exist_ok=True)

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
        fetch_window_env=("BEARDMORE_FETCH_START", "BEARDMORE_FETCH_END"),
        description="Cache IceBridge ATM (ILATM2) GCPs for the Beardmore stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
