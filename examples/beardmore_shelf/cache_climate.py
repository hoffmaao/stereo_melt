"""Populate Beardmore_Shelf's ERA5 surface-pressure cache.

Run once per basin window. The pull is explicit (rather than auto-
triggered inside ``align_strips``) so CDS contact and queue waits are
visible to the operator. Per-strip IBE then reads from the cube on disk
and never touches CDS.

Run:

    python -m beardmore_shelf.cache_climate                # default hourly cadence
    python -m beardmore_shelf.cache_climate --cadence 3    # 3-hourly
    python -m beardmore_shelf.cache_climate --overwrite    # refresh existing cube
"""

from __future__ import annotations

import argparse
import os
import sys

# Point pyproj at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from stereo_melt.pipeline import populate_climate_cache

from beardmore_shelf import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cadence", type=int, default=1,
        help="Hourly stride for ERA5 time axis (default 1 = hourly).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Refresh cube even if ERA5_CACHE_NC already exists.",
    )
    args = parser.parse_args()

    config.ensure_output_dirs()
    populate_climate_cache(config, cadence_hours=args.cadence, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
