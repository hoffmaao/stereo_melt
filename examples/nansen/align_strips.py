"""Batch ASP coregistration of downloaded REMA strips for Nansen.

Thin wrapper over :func:`stereo_melt.coregister.align_driver.run_align_strips_main`
(the shared batch driver — see that module for the full flag reference,
including the ``--start/--end`` era splits and CS2/ATM/LVIS channels this
basin's pre-IS2 runs use). Study-specific pieces live in ``nansen.config``.

Run:

    python -m nansen.align_strips                 # all unaligned strips
    python -m nansen.align_strips --limit 1       # smoke test
    python -m nansen.align_strips --start 2010-01-01 --end 2018-10-15 \
        --control cs2 --use-atm --asp-suffix _cs2 --parallel 4   # pre-IS2 era
"""

from __future__ import annotations

import sys

import stereo_melt.envsetup  # noqa: F401  (PROJ fix before pyproj imports)
from stereo_melt.coregister.align_driver import run_align_strips_main

from nansen import config


def main() -> None:
    rc = run_align_strips_main(
        basin="nansen",
        aoi_shp=config.NANSEN_AOI_SHP,
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
