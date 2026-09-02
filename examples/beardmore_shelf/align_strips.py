"""Batch ASP coregistration of downloaded REMA strips for Beardmore_Shelf.

Thin wrapper over :func:`stereo_melt.coregister.align_driver.run_align_strips_main`
(the shared batch driver — see that module for the full flag reference:
--control is2|cs2|is2+cs2|glas|none, --cs2-source, --use-atm/--use-lvis/
--use-glas, --cap-per-source/--inv-variance-balance, --asp-suffix,
--start/--end era splits, --parallel). Study-specific pieces live in
``beardmore_shelf.config``.

Run:

    python -m beardmore_shelf.align_strips                 # all unaligned strips, IS2 control
    python -m beardmore_shelf.align_strips --limit 1       # one strip (smoke test)
    python -m beardmore_shelf.align_strips --control glas \
        --asp-suffix _glas --parallel 12   # ICESat-1 back-extension era
        # (2009 → 2010-10: no IS2/CS2 exists; run beardmore_shelf.cache_glas
        #  first — strips without a cached GLAS CSV are SKIPPED up front)
"""

from __future__ import annotations

import sys

import stereo_melt.envsetup  # noqa: F401  (PROJ fix before pyproj imports)
from stereo_melt.coregister.align_driver import run_align_strips_main

from beardmore_shelf import config


def main() -> None:
    rc = run_align_strips_main(
        basin="beardmore_shelf",
        aoi_shp=config.BEARDMORE_SHELF_AOI_SHP,
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
