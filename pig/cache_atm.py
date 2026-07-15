"""Pre-cache IceBridge ATM (ILATM2) GCPs for the Pine Island Glacier stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_airborne` for
``source='atm'``. Per-strip filtered CSVs land under the per-basin ASP
root (``pig/data/ASP/atm_data/``); the upstream NSIDC granule store is
shared across all basins under ``<MAIN_DIR>/data/ATM/granules/``.

IceBridge airborne campaigns ran October-November (austral spring) from
2009 through 2019; PIG was a primary target. The per-strip ±90 day
half-window is set wide so that PIG strips outside the campaign months
still find the closest temporal-neighbour ATM flight.

Run:

    python -m pig.cache_atm                        # all in-window strips
    python -m pig.cache_atm --overwrite            # refresh every cache entry
    python -m pig.cache_atm --dem-id <id>          # one strip
    python -m pig.cache_atm --limit 3              # smoke test first 3
    python -m pig.cache_atm --time-window 180      # widen temporal half-window
    python -m pig.cache_atm --prefetch-only        # warm shared granule store
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

from pig import config


def _load_aoi():
    # Mirror align_strips / fetch_strips / plot_fig4_alignment_diagnostics:
    # cache ATM control across the wider stack extent when available so
    # the upstream band that build_stack reprojects has airborne control.
    aoi_path = config.PIG_AOI_SHP
    with fiona.open(aoi_path) as src:
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
        fetch_window_env=("PIG_FETCH_START", "PIG_FETCH_END"),
        description="Cache IceBridge ATM (ILATM2) GCPs for the Pine Island stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
