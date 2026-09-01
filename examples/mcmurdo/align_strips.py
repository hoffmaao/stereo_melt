"""Batch ASP coregistration of downloaded REMA strips for McMurdo.

Thin wrapper over :func:`stereo_melt.coregister.align_driver.run_align_strips_main`
(the shared batch driver — see that module for the full flag reference).
Study-specific pieces live in ``mcmurdo.config``.

Run:

    python -m mcmurdo.align_strips                 # all unaligned strips
    python -m mcmurdo.align_strips --limit 1       # one strip (smoke test)
    python -m mcmurdo.align_strips --dem-id <id>   # specific strip by id
"""

from __future__ import annotations

import sys

import stereo_melt.envsetup  # noqa: F401  (PROJ fix before pyproj imports)
from stereo_melt.coregister.align_driver import run_align_strips_main

from mcmurdo import config


def main() -> None:
    rc = run_align_strips_main(
        basin="mcmurdo",
        aoi_shp=config.MCMURDO_AOI_SHP,
        grounding_line_shp=config.GROUNDING_LINE_SHP,
        strips_dir=config.STRIPS_DIR,
        canonical_asp_root=config.ASP_ROOT,
        mosaic_dir=config.MOSAIC_DIR,
        rock_shapefile=config.ROCK_POLYGONS_SHP,
        tide_model=config.TIDE_MODEL,
        start_time=config.START_TIME,
        end_time=config.END_TIME,
        ensure_output_dirs=config.ensure_output_dirs,
        description=__doc__.splitlines()[0],
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
