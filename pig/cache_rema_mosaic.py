"""Stage REMA V2 32 m mosaic tiles over the PIG AOI -> data/REMA/mosaic/.

The REMA mosaic is the dense, time-invariant **reference surface** for the
Option B two-stage coregistration (see ``literature/plan_alignment.md``): each
strip is pc_align'd to it over static (rock + slow-ground) surfaces for the
tilt/rotation constraint sparse altimetry can't give. Pulled from the PGC
open-data S3 bucket (unsigned) via the existing ``stereo_melt.io.rema`` helpers.

This is **our own independent data** (PGC REMA) -- explicitly NOT the Shean
elevation grids in ``data/Shean2019/`` (those stay validation-only).

Run:
    python -m pig.cache_rema_mosaic
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import geopandas as gpd

from stereo_melt.io.rema import download_tiles, list_rema_v2_tiles

from pig import config


def main(prefix: str = "rema/mosaics/v2.0/32m") -> None:
    aoi = gpd.read_file(config.PIG_AOI_SHP)
    if aoi.crs is None or str(aoi.crs) != "EPSG:3031":
        aoi = aoi.to_crs("EPSG:3031")
    roi = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union

    tiles = list_rema_v2_tiles(roi, config.MOSAIC_INDEX_SHP, prefix=prefix)
    if not tiles:
        raise SystemExit("No REMA mosaic tiles intersect the AOI — check PIG_AOI_SHP / index.")
    config.MOSAIC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Staging {len(tiles)} REMA {prefix.split('/')[-1]} mosaic tiles -> {config.MOSAIC_DIR}")
    download_tiles(tiles, str(config.MOSAIC_DIR))

    got = sorted(config.MOSAIC_DIR.glob("*_dem.tif"))
    total_mb = sum(g.stat().st_size for g in got) / 1e6
    print(f"done: {len(got)} mosaic tiles on disk ({total_mb:.0f} MB total)")
    for g in got:
        print(f"   {g.name}  {g.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
