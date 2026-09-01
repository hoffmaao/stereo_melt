"""Build the dense **static reference DEM** for two-stage coregistration.

Stage of the Option-B alignment rethink (``literature/plan_alignment.md``).
Merges the PIG REMA V2 32 m mosaic tiles over the AOI into one raster, then
masks it to the *static* surface — BedMachine rock + grounded ice, 2 km
eroded from the non-grounded boundary — the SAME mask the tilt LSQ uses
(:func:`stereo_melt.coregister.tilt.build_static_area_polygon_mask`). Pixels
off the static surface (floating shelf, ocean, lake, eroded grounding zone)
are set to nodata so that, when ``pc_align`` builds a point cloud from this
reference, only static surfaces contribute correspondences.

Why static-only matters: a multi-year REMA mean tied to the floating shelf
would erase the very dh/dt we measure. Over rock + stable grounded ice the
REMA mean *is* (to first order) the epoch elevation, so the dense
surface-to-surface tie there constrains each strip's tilt/rotation without
contaminating the dynamic signal. The altimetry control (IS2/CS2/ATM/LVIS)
remains the absolute datum + time anchor in Stage 2.

The reference is PGC REMA — our own independent data — explicitly NOT the
Shean elevation grids in ``data/Shean2019/`` (validation-only).

Run:
    python -m pig.build_rema_static_reference
"""

from __future__ import annotations

import os
import sys

# Point pyproj/GDAL at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.merge import merge

from stereo_melt.coregister.tilt import build_static_area_polygon_mask
from stereo_melt.io.rema import list_rema_v2_tiles

from pig import config


def _local_tiles_for_aoi():
    """Return ``(roi_polygon, [local tile paths])`` for the PIG AOI.

    The mosaic dir is a *shared* pan-Antarctic cache, so we re-intersect
    the AOI with the mosaic index and keep only the tiles that both
    overlap the AOI and are already staged on disk (by
    ``pig.cache_rema_mosaic``).
    """
    aoi = gpd.read_file(config.PIG_AOI_SHP)
    if aoi.crs is None or str(aoi.crs) != "EPSG:3031":
        aoi = aoi.to_crs("EPSG:3031")
    roi = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union

    s3_uris = list_rema_v2_tiles(roi, config.MOSAIC_INDEX_SHP)
    local = []
    missing = []
    for uri in s3_uris:
        name = os.path.basename(uri)  # e.g. 27_15_32m_v2.0_dem.tif
        p = config.MOSAIC_DIR / name
        (local if p.exists() else missing).append(str(p) if p.exists() else name)
    if missing:
        print(f"  ⚠ {len(missing)} AOI tile(s) not staged (run pig.cache_rema_mosaic): "
              f"{', '.join(os.path.basename(m) for m in missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    return roi, local


def main(nodata: float = -9999.0, buffer_m: float = 2000.0) -> None:
    roi, tiles = _local_tiles_for_aoi()
    if not tiles:
        raise SystemExit(
            "No staged REMA tiles intersect the PIG AOI — run "
            "`python -m pig.cache_rema_mosaic` first."
        )
    print(f"Merging {len(tiles)} REMA 32 m tiles over the PIG AOI "
          f"(cropped to AOI bbox + {buffer_m:.0f} m)...")

    # Crop the merge to the AOI bbox + buffer so the reference is only as
    # large as the analysis grid — keeps the masked raster tractable.
    minx, miny, maxx, maxy = roi.bounds
    bounds = (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)

    srcs = [rasterio.open(t) for t in tiles]
    try:
        src_nodata = srcs[0].nodata if srcs[0].nodata is not None else nodata
        mosaic, transform = merge(srcs, bounds=bounds, nodata=src_nodata)
        ref_crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()

    dem = mosaic[0].astype("float32")  # (ny, nx)
    dem[dem == src_nodata] = np.nan
    ny, nx = dem.shape
    res = abs(transform.a)
    # Pixel-center coordinates from the merge transform (x ascending,
    # y descending for a north-up raster: transform.e < 0).
    xs = transform.c + transform.a * (np.arange(nx) + 0.5)
    ys = transform.f + transform.e * (np.arange(ny) + 0.5)
    print(f"  merged grid: {nx} x {ny} px @ {res:.0f} m  "
          f"x[{xs.min():.0f}, {xs.max():.0f}]  y[{ys.min():.0f}, {ys.max():.0f}]")

    # Static mask on the merged grid — identical definition to the tilt LSQ
    # (build_static_area_polygon_mask reads only the x/y coords of its first
    # argument, so a coords-only DataArray suffices).
    grid = xr.DataArray(
        np.zeros((ny, nx), dtype="float32"),
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
    )
    print("  building static mask (BedMachine rock + grounded, 2 km erode)...")
    static = build_static_area_polygon_mask(grid, config.BEDMACHINE_NC).values.astype(bool)
    frac = 100.0 * static.sum() / static.size
    print(f"  static footprint: {int(static.sum()):,} / {static.size:,} px ({frac:.2f}% of grid)")

    out = dem.copy()
    out[~static] = nodata
    out[~np.isfinite(out)] = nodata
    n_valid = int(np.count_nonzero(out != nodata))
    frac_valid = 100.0 * n_valid / out.size
    print(f"  static ∩ REMA-valid pixels: {n_valid:,} ({frac_valid:.2f}% of grid)")
    if n_valid == 0:
        raise SystemExit(
            "Static reference is empty — the AOI's REMA mosaic has no valid "
            "pixels over the BedMachine static surface. Check tile coverage / mask."
        )

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": "float32",
        "crs": ref_crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    config.MOSAIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.REMA_STATIC_REFERENCE_TIF
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"✅ wrote static reference: {out_path}  ({size_mb:.0f} MB)")
    print(f"   valid-pixel elevation range: "
          f"{np.nanmin(np.where(out != nodata, out, np.nan)):.1f} .. "
          f"{np.nanmax(np.where(out != nodata, out, np.nan)):.1f} m")


if __name__ == "__main__":
    main()
