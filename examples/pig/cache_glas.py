"""Pre-cache ICESat-1 GLAS (GLAH12) GCPs for the PineIsland stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_glas`, mirroring
``pig.cache_atm``. Per-strip filtered CSVs land under
``pig/data/ASP/glas_data/``; the NSIDC granule store is shared across
basins at ``<MAIN_DIR>/data/GLAS/granules/``.

NOTE: PIG's REMA archive starts 2010-12-09 — after the GLAS reachability
cutoff (last campaign ends 2009-10-11, +1 yr control cap → 2010-10-11) —
so this cache is expected to find nothing under the current s2s041
index. The wrapper exists for cross-basin uniformity and future index
releases. Window override:

    PIG_FETCH_START=2009-01-01 PIG_FETCH_END=2010-10-12 \
        python -m pig.cache_glas
"""

from __future__ import annotations

import sys
from pathlib import Path

import stereo_melt.envsetup  # noqa: F401

import fiona
from shapely.geometry import shape

from stereo_melt.coregister import cache_glas

from pig import config


def main() -> None:
    config.ensure_output_dirs()
    with fiona.open(config.PIG_AOI_SHP) as src:
        aoi = shape(next(iter(src))["geometry"])
    rc = cache_glas.run_cache_glas_main(
        aoi_polygon=aoi,
        strips_dir=Path(config.STRIPS_DIR),
        cache_dir=config.ASP_ROOT / "glas_data",
        granule_dir=config.MAIN_DIR / "data" / "GLAS" / "granules",
        bedmachine_path=config.BEDMACHINE_NC,
        velocity_path=config.MEASURES_PHASE_NC,
        start_time=config.START_TIME,
        end_time=config.END_TIME,
        fetch_window_env=("PIG_FETCH_START", "PIG_FETCH_END"),
        description="Cache ICESat-1 GLAS (GLAH12) GCPs for the PineIsland stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
