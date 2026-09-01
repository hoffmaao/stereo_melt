# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""REMA ingestion.

Lists and downloads REMA V2 mosaic tiles from the PGC open-data S3
bucket, searches the strip index via :mod:`pdemtools`, downloads strips
through :func:`pdemtools.load.from_search`, and renders PyGMT plots of
strip footprints against control-surface polygons.
"""

import os

import boto3
import geopandas as gpd
import pandas as pd
import pdemtools as pdt
import pygmt
from botocore import UNSIGNED
from botocore.config import Config

from .masks import create_low_velocity_rock_polygons


def plot_strip_intersections_with_polygons_pygmt_v2(
    strips_df,
    lv_gdf,
    rock_gdf,
    output_path="/wd2/projects/stereo_melt/plots/intersections_plot.png",
):
    r"""Render REMA strip footprints against control-surface polygons with PyGMT.

    Parameters
    ----------
    strips_df, lv_gdf, rock_gdf : geopandas.GeoDataFrame
        Strip footprints, low-velocity polygons, and rock-exposure
        polygons, each in any projected CRS (reprojected to EPSG:3031
        internally).
    output_path : str
        Destination PNG path.
    """
    strips_df = strips_df.to_crs("EPSG:3031")
    lv_gdf = lv_gdf.to_crs("EPSG:3031")
    rock_gdf = rock_gdf.to_crs("EPSG:3031")

    all_bounds = gpd.GeoDataFrame(pd.concat([strips_df, lv_gdf, rock_gdf])).total_bounds
    region = [all_bounds[0], all_bounds[2], all_bounds[1], all_bounds[3]]
    print(f"Plotting region: {region}")

    projection = "x1:550000"

    fig = pygmt.Figure()
    fig.basemap(
        region=region, projection=projection, frame=["a", "+tREMA Strips and Intersections"]
    )

    for polygon in lv_gdf.geometry:
        if polygon.is_valid:
            fig.plot(data=polygon.wkt, pen="1p,blue", fill="blue", transparency=70)

    for polygon in rock_gdf.geometry:
        if polygon.is_valid:
            fig.plot(data=polygon.wkt, pen="1p,brown", fill="brown", transparency=70)

    for polygon in strips_df.geometry:
        if polygon.is_valid:
            fig.plot(data=polygon.wkt, pen="2p,red", fill="none")

    fig.savefig(output_path)
    print(f"✅ Intersection plot saved to {output_path}")


def plot_strip_intersections_with_polygons_pygmt(
    strips_df,
    lv_gdf,
    rock_gdf,
    output_path="/wd2/projects/stereo_melt/plots/intersections_plot.png",
):
    r"""Render REMA strip footprints with PyGMT using the whole-GDF plot path.

    Same semantics as :func:`plot_strip_intersections_with_polygons_pygmt_v2`,
    but passes each GeoDataFrame directly to :meth:`pygmt.Figure.plot`
    rather than iterating geometries.
    """
    strips_df = strips_df.to_crs("EPSG:3031")
    lv_gdf = lv_gdf.to_crs("EPSG:3031")
    rock_gdf = rock_gdf.to_crs("EPSG:3031")

    fig = pygmt.Figure()

    bounds = strips_df.total_bounds
    region = [bounds[0], bounds[2], bounds[1], bounds[3]]
    print(f"Region: {region}")

    print(f"Strips bounds: {strips_df.total_bounds}")
    print(f"Low-velocity bounds: {lv_gdf.total_bounds}")
    print(f"Rock bounds: {rock_gdf.total_bounds}")

    projection = "x1:1500000"

    fig.basemap(
        region=region, projection=projection, frame=["a", "+tREMA Strips and Intersections"]
    )
    fig.plot(
        data=lv_gdf, projection=projection, pen="1p,blue", close=True, fill="blue", transparency=70
    )
    fig.plot(
        data=rock_gdf,
        projection=projection,
        pen="1p,brown",
        close=True,
        fill="brown",
        transparency=70,
    )
    fig.plot(data=strips_df, pen="2p,red", close=True)

    fig.savefig(output_path)
    print(f"✅ Intersection plot saved to {output_path}")


def list_rema_v2_tiles(
    roi_polygon,
    mosaic_index_fpath,
    s3_bucket="pgc-opendata-dems",
    prefix="rema/mosaics/v2.0/32m",
):
    r"""Return S3 paths of REMA V2 mosaic tiles intersecting ``roi_polygon``.

    Parameters
    ----------
    roi_polygon : shapely.geometry.Polygon
        Region of interest.
    mosaic_index_fpath : str or pathlib.Path
        Path to the REMA mosaic index shapefile (e.g.
        ``REMA_Mosaic_Index_v2_32m.shp``). Supplied by the caller so
        the library does not hardcode any study-specific paths.
    s3_bucket : str
        PGC open-data bucket name.
    prefix : str
        Mosaic tile path prefix.

    Returns
    -------
    list of str
        S3 URIs of the overlapping tiles.
    """
    print("Loading tile index shapefile...")
    tiles_gdf = gpd.read_file(mosaic_index_fpath)

    if tiles_gdf.crs != "EPSG:3031":
        print("Reprojecting tile index to EPSG:3031...")
        tiles_gdf = tiles_gdf.to_crs("EPSG:3031")

    print("Performing spatial intersection to find overlapping tiles...")
    roi_gdf = gpd.GeoDataFrame({"geometry": [roi_polygon]}, crs=tiles_gdf.crs)
    overlapping_tiles = gpd.overlay(tiles_gdf, roi_gdf, how="intersection")

    if overlapping_tiles.empty:
        print("No tiles found for the given region of interest.")
        return []

    tile_paths = []
    for idx, row in overlapping_tiles.iterrows():
        tile_name = row["tile"]
        tile_path = f"s3://{s3_bucket}/{prefix}/{tile_name}/{tile_name}_32m_v2.0_dem.tif"
        tile_paths.append(tile_path)

    print(f"Found {len(tile_paths)} tiles overlapping with the ROI.")
    return tile_paths


def download_tiles(tile_paths, output_dir):
    r"""Download REMA V2 tiles to ``output_dir``, skipping files already present.

    Parameters
    ----------
    tile_paths : list of str
        S3 URIs from :func:`list_rema_v2_tiles`.
    output_dir : str
        Local destination directory; created if missing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))

    print(f"Starting download of {len(tile_paths)} REMA tiles...")

    for idx, tile_path in enumerate(tile_paths):
        s3_parts = tile_path.replace("s3://", "").split("/", 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        local_file = os.path.join(output_dir, os.path.basename(key))

        if os.path.exists(local_file):
            print(f"[{idx + 1}/{len(tile_paths)}] File already exists: {local_file}")
            continue

        try:
            print(f"[{idx + 1}/{len(tile_paths)}] Downloading {tile_path}...")
            s3.Bucket(bucket).download_file(key, local_file)
            print(f"Saved to: {local_file}")
        except Exception as e:
            print(f"Error downloading {tile_path}: {e}")

    print("Download process completed.")


def merge_tiles(tile_paths, output_path):
    r"""Merge REMA V2 tiles into a single GeoTIFF via :func:`rasterio.merge.merge`.

    Parameters
    ----------
    tile_paths : list of str
        Local paths of the input tiles.
    output_path : str
        Destination GeoTIFF path.
    """
    import rasterio
    from rasterio.merge import merge

    print("Merging tiles...")
    rasters = [rasterio.open(tile) for tile in tile_paths]
    mosaic, transform = merge(rasters)

    out_meta = rasters[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
        }
    )

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Merged mosaic saved to {output_path}")


def list_rema_strips_v2(
    polygon,
    strip_index_fpath,
    date_range=None,
    velocity_netcdf_path=None,
    bedmachine_path=None,
    output_path=None,
    visualize_path=None,
):
    r"""Search REMA strips overlapping ``polygon`` and filter by control polygons.

    Delegates strip search to :func:`pdemtools.search` and filters the
    hits to those intersecting the control-surface polygons built by
    :func:`~stereo_melt.io.masks.create_low_velocity_rock_polygons`.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Region of interest.
    strip_index_fpath : str or pathlib.Path
        Path to the REMA strip index parquet file (e.g.
        ``REMA_Strip_Index_s2s041.parquet``). Supplied by the caller.
    date_range : tuple of str, optional
        ``(start_date, end_date)``; omit for the full catalog.
    velocity_netcdf_path, bedmachine_path : str, optional
        Inputs for control-polygon construction.
    output_path : str, optional
        If given, filtered strips are written here as GeoJSON.
    visualize_path : str, optional
        If given, a PyGMT plot is rendered.

    Returns
    -------
    geopandas.GeoDataFrame or None
        Filtered strip footprints, or ``None`` if no strips were found.
    """
    print("🔍 Searching for REMA strips using pDEMtools...")

    # pdt.search expects a single Polygon (it unpacks .bounds internally
    # via shapely.geometry.box). MultiPolygon AOIs trip the unpacker, so
    # collapse them: unary_union returns a single Polygon when all parts
    # are connected; otherwise we fall back to convex_hull which always
    # returns a single Polygon. The strip filter is bbox-grade in
    # pdt.search anyway -- the fine AOI mask is applied downstream.
    if polygon.geom_type == "MultiPolygon":
        from shapely.ops import unary_union
        merged = unary_union(polygon)
        polygon = merged if merged.geom_type == "Polygon" else merged.convex_hull

    try:
        if date_range:
            start_date, end_date = date_range
            strips_df = pdt.search(
                index_fpath=strip_index_fpath,
                bounds=polygon,
                dates=(start_date, end_date),
                accuracy=2,
            )
        else:
            strips_df = pdt.search(
                index_fpath=strip_index_fpath,
                bounds=polygon,
                accuracy=2,
            )

        if strips_df.empty:
            print("❌ No REMA strips found.")
            return None

        print(f"✅ Found {len(strips_df)} REMA strips. Verifying geometry and CRS...")

        strips_df = strips_df[~strips_df.geometry.is_empty]
        strips_df = strips_df[strips_df.geometry.is_valid]

        if str(strips_df.crs) != "EPSG:4326":
            print("🔄 Reprojecting strips_df to EPSG:4326...")
            strips_df = strips_df.to_crs("EPSG:4326")

        print(f"✅ Remaining strips after geometry validation: {len(strips_df)}")

        low_velocity_polygons, rock_polygons = create_low_velocity_rock_polygons(
            velocity_netcdf_path,
            bedmachine_path,
            polygon,
            output_path,
            velocity_threshold=10,
        )

        lv_gdf = gpd.GeoDataFrame(geometry=low_velocity_polygons, crs="EPSG:3031").to_crs(
            "EPSG:4326"
        )
        rock_gdf = gpd.GeoDataFrame(geometry=rock_polygons, crs="EPSG:3031").to_crs("EPSG:4326")

        valid_strips = strips_df[
            strips_df.geometry.apply(
                lambda g: lv_gdf.intersects(g).any() or rock_gdf.intersects(g).any()
            )
        ]

        print(f"✅ Filtered down to {len(valid_strips)} REMA strips after polygon intersection.")

        if visualize_path:
            print("🔄 Visualizing strips and overlaps...")
            plot_strip_intersections_with_polygons_pygmt(
                strips_df=strips_df,
                rock_gdf=rock_gdf,
                lv_gdf=lv_gdf,
                output_path="./plots/intersections_plot.png",
            )
        if output_path:
            valid_strips.to_file(output_path, driver="GeoJSON")
            print(f"✅ Filtered strips saved to {output_path}")

        return valid_strips

    except Exception as e:
        print(f"[ERROR] Failed to list REMA strips: {e}")
        return None


def list_rema_strips_old(
    polygon,
    strip_index_fpath,
    date_range=None,
    velocity_netcdf_path=None,
    bedmachine_path=None,
    output_path=None,
    visualize_path=None,
):
    r"""Legacy REMA strip lister preserved for comparison.

    Superseded by :func:`list_rema_strips_v2`. Retained so prior
    notebooks that depend on this signature keep working.
    """
    print("🔍 Searching for REMA strips using pDEMtools...")

    try:
        if date_range:
            start_date, end_date = date_range
            strips_df = pdt.search(
                index_fpath=strip_index_fpath,
                bounds=polygon,
                dates=(start_date, end_date),
                accuracy=2,
            )
        else:
            strips_df = pdt.search(
                index_fpath=strip_index_fpath,
                bounds=polygon,
                dates=(start_date, end_date),
                accuracy=2,
            )

        if strips_df.empty:
            print("❌ No REMA strips found.")
            return None

        print(f"✅ Found {len(strips_df)} REMA strips. Generating polygons...")

        print("🔄 Checking CRS of strips_df...")
        print(f"CRS of strips_df: {strips_df.crs}")

        low_velocity_polygons, rock_polygons = create_low_velocity_rock_polygons(
            velocity_netcdf_path,
            bedmachine_path,
            polygon,
            output_path,
            velocity_threshold=10,
        )

        lv_gdf = gpd.GeoDataFrame(geometry=low_velocity_polygons, crs="EPSG:3031")
        rock_gdf = gpd.GeoDataFrame(geometry=rock_polygons, crs="EPSG:3031")

        print("🔄 Visualizing strips and overlaps...")
        plot_strip_intersections_with_polygons_pygmt(
            strips_df=strips_df,
            rock_gdf=rock_gdf,
            lv_gdf=lv_gdf,
        )

        valid_strips = strips_df[
            strips_df.geometry.apply(
                lambda g: lv_gdf.intersects(g).any() or rock_gdf.intersects(g).any()
            )
        ]

        print(f"✅ Filtered down to {len(valid_strips)} REMA strips after polygon intersection.")
        return valid_strips

    except Exception as e:
        print(f"[ERROR] Failed to list REMA strips: {e}")
        return None


def plot_aoi_strip_coverage(
    polygon,
    strip_index_fpath,
    output_path,
    date_range=None,
    title=None,
):
    r"""Plot the AOI polygon, bounding box, and intersecting strip footprints.

    Visual sanity check before fetching: confirms the polygon used for the
    REMA query is what the user expects, and shows the temporal density of
    available strips.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or geopandas.GeoDataFrame
        Region of interest. If a GeoDataFrame, its CRS must be EPSG:3031.
    strip_index_fpath : str or pathlib.Path
        Path to the REMA strip index parquet file.
    output_path : str or pathlib.Path
        File to write the PNG to.
    date_range : tuple of str, optional
        ``(start_date, end_date)`` window filter applied to ``acqdate1``
        before plotting. Strips outside the window are excluded.
    title : str, optional
        Suptitle. Defaults to ``"AOI strip coverage"``.

    Returns
    -------
    int
        Number of strips plotted (after intersection + date filter).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.geometry import shape as _shape

    idx = gpd.read_parquet(strip_index_fpath)

    # AOI in plotting coords is EPSG:3031 (Antarctic Polar Stereographic);
    # the s2s041 strip index is stored in lon/lat (OGC:CRS84). Carry both
    # so we can reproject for the intersection test but still draw in the
    # native EPSG:3031 frame.
    if isinstance(polygon, gpd.GeoDataFrame):
        aoi_3031 = polygon.to_crs("EPSG:3031")
    else:
        aoi_3031 = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:3031")
    aoi_geom_3031 = aoi_3031.geometry.union_all()
    aoi_geoms_3031 = list(aoi_3031.geometry)

    aoi_for_index = aoi_3031.to_crs(idx.crs).geometry.union_all()
    sub = idx[idx.intersects(aoi_for_index)].to_crs("EPSG:3031").copy()
    if "acqdate1" in sub.columns:
        sub["acqdate1"] = pd.to_datetime(sub["acqdate1"])
        if date_range is not None:
            tz = sub["acqdate1"].dt.tz
            t0 = pd.Timestamp(date_range[0]).tz_localize(tz) if tz is not None else pd.Timestamp(date_range[0])
            t1 = pd.Timestamp(date_range[1]).tz_localize(tz) if tz is not None else pd.Timestamp(date_range[1])
            sub = sub[(sub["acqdate1"] >= t0) & (sub["acqdate1"] < t1)]

    bx_min, by_min, bx_max, by_max = aoi_geom_3031.bounds

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    if len(sub):
        # Color by year-since-min-epoch to keep matplotlib's normalization in
        # a friendly float range. Tick labels are explicit YYYY-MM-DD strings.
        dates = list(sub["acqdate1"])
        epoch_min = min(dates)
        epoch_max = max(dates)
        def _yrs_since(t):
            return (t - epoch_min).total_seconds() / (86400.0 * 365.25)

        years_lo = 0.0
        years_hi = _yrs_since(epoch_max)
        norm = plt.Normalize(vmin=years_lo, vmax=years_hi)
        cmap = plt.get_cmap("viridis")
        geoms = list(sub.geometry)
        for geom, dt in zip(geoms, dates):
            color = cmap(norm(_yrs_since(dt)))
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for p in polys:
                xs, ys = p.exterior.xy
                ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.6)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label("acquisition date")
        epoch_mid = epoch_min + (epoch_max - epoch_min) / 2
        cbar.set_ticks([years_lo, years_hi / 2, years_hi])
        cbar.set_ticklabels([
            str(epoch_min.date()),
            str(epoch_mid.date()),
            str(epoch_max.date()),
        ])

    for geom in aoi_geoms_3031:
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, color="orange", alpha=0.25, edgecolor="black", linewidth=2)
        elif geom.geom_type == "MultiPolygon":
            for p in geom.geoms:
                xs, ys = p.exterior.xy
                ax.fill(xs, ys, color="orange", alpha=0.25, edgecolor="black", linewidth=2)

    ax.add_patch(plt.Rectangle(
        (bx_min, by_min), bx_max - bx_min, by_max - by_min,
        fill=False, edgecolor="red", linewidth=2, linestyle="--", label="bbox",
    ))

    pad = 0.1 * max(bx_max - bx_min, by_max - by_min)
    ax.set_xlim(bx_min - pad, bx_max + pad)
    ax.set_ylim(by_min - pad, by_max + pad)
    ax.set_xlabel("x (EPSG:3031, m)")
    ax.set_ylabel("y (EPSG:3031, m)")
    ax.set_aspect("equal")

    n = len(sub)
    if "acqdate1" in sub.columns and n:
        per_year = sub["acqdate1"].dt.year.value_counts().sort_index()
        year_str = ", ".join(f"{int(y)}: {int(c)}" for y, c in per_year.items())
        subtitle = f"{n} strips intersect AOI ({year_str})"
    else:
        subtitle = f"{n} strips intersect AOI"
    ax.set_title(
        f"{title or 'AOI strip coverage'}\n"
        f"bbox: x=[{bx_min:.0f}, {bx_max:.0f}]  y=[{by_min:.0f}, {by_max:.0f}]\n"
        f"{subtitle}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return n


def list_rema_strips(polygon, strip_index_fpath, date_range=None):
    r"""Return the unfiltered set of REMA strips overlapping ``polygon``.

    Same as :func:`list_rema_strips_v2` but without the low-velocity /
    rock control-surface filter.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Region of interest.
    strip_index_fpath : str or pathlib.Path
        Path to the REMA strip index parquet file.
    date_range : tuple of str, optional
        ``(start_date, end_date)`` window filter.
    """
    print("🔍 Searching for REMA strips using pDEMtools...")

    # pdt.search expects a single Polygon (it unpacks .bounds internally
    # via shapely.geometry.box). MultiPolygon AOIs trip the unpacker, so
    # collapse them: unary_union returns a single Polygon when all parts
    # are connected; otherwise we fall back to convex_hull which always
    # returns a single Polygon. The strip filter is bbox-grade in
    # pdt.search anyway -- the fine AOI mask is applied downstream.
    if polygon.geom_type == "MultiPolygon":
        from shapely.ops import unary_union
        merged = unary_union(polygon)
        polygon = merged if merged.geom_type == "Polygon" else merged.convex_hull

    try:
        if date_range:
            start_date, end_date = date_range
            strips_df = pdt.search(
                index_fpath=strip_index_fpath,
                bounds=polygon,
                dates=(start_date, end_date),
                accuracy=2,
            )
        else:
            strips_df = pdt.search(
                index_fpath=strip_index_fpath,
                bounds=polygon,
                accuracy=2,
            )

        if strips_df.empty:
            print("❌ No REMA strips found.")
            return None

        print(f"✅ Found {len(strips_df)} REMA strips. Verifying geometry and CRS...")

        strips_df = strips_df[~strips_df.geometry.is_empty]
        strips_df = strips_df[strips_df.geometry.is_valid]

        return strips_df

    except Exception as e:
        print(f"[ERROR] Failed to list REMA strips: {e}")
        return None


def download_rema_strips(gdf, output_dir):
    r"""Download REMA strips in ``gdf`` via :func:`pdemtools.load.from_search`.

    Existing output files are skipped. Each strip is written as a
    ZSTD-compressed GeoTIFF.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Search results from :func:`pdemtools.search`.
    output_dir : str
        Destination directory; created if missing.
    """
    if gdf is None or gdf.empty:
        print("No REMA strips to download.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting download of {len(gdf)} REMA strips...")

    for idx, row in gdf.iterrows():
        dem_id = row.get("dem_id", f"unknown_id_{idx}")
        filename = f"{dem_id}.tif"
        local_path = os.path.join(output_dir, filename)

        if os.path.exists(local_path):
            print(f"[{idx + 1}/{len(gdf)}] File already exists: {local_path}")
            continue

        try:
            print(f"[{idx + 1}/{len(gdf)}] Downloading DEM ID: {dem_id}")
            dem = pdt.load.from_search(row, bitmask=True)
            dem.compute()
            dem.rio.to_raster(local_path, compress="ZSTD", predictor=3, zlevel=1)
            print(f"Saved to: {local_path}")
        except Exception as e:
            print(f"[{idx + 1}/{len(gdf)}] Failed to download DEM ID {dem_id}: {e}")

    print("Download process complete.")


# ---------------------------------------------------------------------------
# Quality companion files (matchtag, bitmask)
# ---------------------------------------------------------------------------

REMA_S3_PUBLIC_BASE = (
    "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/rema/strips/s2s041/2m"
)
"""Public S3 endpoint for REMA s2s041 2-m strips. Each strip's companion
files live at ``{REMA_S3_PUBLIC_BASE}/{geocell}/{dem_id}_<kind>.tif`` where
``<kind>`` is ``dem``, ``matchtag``, ``bitmask``, etc."""


# Per-suffix mapping of PGC asset kinds -> file extensions. Raster
# companion products use ``.tif`` (matchtag, bitmask, browse); per-strip
# textual metadata uses ``.txt`` (mdf, readme); the STAC item is bare
# ``.json`` at the strip prefix without a ``_kind`` suffix.
_QUALITY_EXT = {
    "matchtag": "tif",
    "bitmask": "tif",
    "browse": "tif",
    "mdf": "txt",
    "readme": "txt",
}


def _quality_url(dem_id: str, geocell: str, kind: str) -> str:
    if kind == "stac":
        return f"{REMA_S3_PUBLIC_BASE}/{geocell}/{dem_id}.json"
    ext = _QUALITY_EXT.get(kind, "tif")
    return f"{REMA_S3_PUBLIC_BASE}/{geocell}/{dem_id}_{kind}.{ext}"


def _quality_local(output_dir: str, dem_id: str, kind: str) -> str:
    if kind == "stac":
        return os.path.join(output_dir, f"{dem_id}.json")
    ext = _QUALITY_EXT.get(kind, "tif")
    return os.path.join(output_dir, f"{dem_id}_{kind}.{ext}")


def download_rema_quality_files(
    gdf,
    output_dir,
    kinds=("matchtag", "bitmask", "mdf", "readme", "stac"),
):
    r"""Download per-strip quality + metadata companion files from PGC's
    open-data S3.

    Companion files are fetched as separate objects from
    ``s3://pgc-opendata-dems/rema/strips/s2s041/2m/<geocell>/``. They're
    relatively small (5–20 MB raster, kB-scale text) and provide:

    - ``matchtag`` --- per-pixel stereo-match flag (binary)
    - ``bitmask`` --- per-pixel quality flag (edge / water / cloud)
    - ``browse`` --- 10-m hillshade thumbnail
    - ``mdf`` --- machine-readable metadata file (acquisition times,
      sun angles, sensor IDs, geometry --- the PGC "meta-data file")
    - ``readme`` --- license + product summary
    - ``stac`` --- STAC item JSON pointing at all the above

    Existing output files are skipped.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Search results from :func:`pdemtools.search`. Must include
        ``dem_id`` and ``geocell`` columns (both present in the s2s041
        index).
    output_dir : str
        Destination directory; the same directory the DEM strips
        themselves live in.
    kinds : tuple of str
        Companion-file suffixes. Default pulls quality (matchtag,
        bitmask) plus all metadata (mdf, readme, stac).
    """
    if gdf is None or gdf.empty:
        print("No strips to fetch quality files for.")
        return
    if "geocell" not in gdf.columns:
        raise KeyError("gdf must include a 'geocell' column (s2s041 index has it).")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    import urllib.request

    print(f"Fetching {kinds} for {len(gdf)} strips from PGC open-data S3...")
    n_ok, n_skip, n_fail = 0, 0, 0
    for idx, row in gdf.iterrows():
        dem_id = row.get("dem_id")
        geocell = row.get("geocell")
        if not dem_id or not geocell:
            continue
        for kind in kinds:
            local = _quality_local(output_dir, dem_id, kind)
            if os.path.exists(local):
                n_skip += 1
                continue
            url = _quality_url(dem_id, geocell, kind)
            try:
                urllib.request.urlretrieve(url, local)
                n_ok += 1
                print(f"  [{idx + 1}/{len(gdf)}] {kind}  ok  {os.path.basename(local)}")
            except Exception as exc:
                n_fail += 1
                print(f"  [{idx + 1}/{len(gdf)}] {kind}  FAIL  {exc}")

    print(f"Quality-file download done: ok={n_ok}, skipped={n_skip}, failed={n_fail}")


def apply_pgc_quality_mask(
    strip_path,
    matchtag_keep=(1,),
    bitmask_keep=(0, 2),
    return_mask=False,
):
    r"""Apply PGC matchtag + bitmask quality filters to a REMA strip.

    Mirrors the filter in ``altimetryFit.pgc_2m_dem``: keep pixels whose
    matchtag is in ``matchtag_keep`` AND whose bitmask is in
    ``bitmask_keep``. PGC's conventions:

    - ``matchtag``: binary, 1 = stereo matched, 0 = no match.
    - ``bitmask``: bit-packed flag --- 0 = clean, 1 = edge, 2 = water,
      4 = cloud, 8 = no data. The default ``{0, 2}`` keeps land/ice and
      water-flagged pixels (floating ice shelves often get the water
      flag from underlying-terrain logic and we want them kept) while
      excluding edges (1) and clouds (4).

    Looks for ``<strip_path stem>_matchtag.tif`` and
    ``<strip_path stem>_bitmask.tif`` next to the input strip. If
    either is missing, returns the input unchanged with a warning.

    Parameters
    ----------
    strip_path : str or Path
        Path to the DEM strip GeoTIFF.
    matchtag_keep, bitmask_keep : tuple of int
        Per-flag whitelists. See above.
    return_mask : bool
        If True, also return the boolean keep-mask.

    Returns
    -------
    numpy.ndarray or (ndarray, ndarray)
        DEM array with rejected pixels set to NaN. If
        ``return_mask=True``, also returns the (H, W) boolean keep-mask.
    """
    import numpy as _np
    import rasterio as _rio

    strip_path = str(strip_path)
    base = strip_path[:-4] if strip_path.endswith(".tif") else strip_path
    matchtag_path = f"{base}_matchtag.tif"
    bitmask_path = f"{base}_bitmask.tif"

    with _rio.open(strip_path) as src:
        dem = src.read(1).astype("float32")
        nodata = src.nodata

    if not (os.path.exists(matchtag_path) and os.path.exists(bitmask_path)):
        print(
            f"⚠️ Missing quality files for {os.path.basename(strip_path)}; "
            "returning unmasked DEM. Run download_rema_quality_files first."
        )
        if nodata is not None:
            dem[dem == nodata] = _np.nan
        return (dem, _np.ones_like(dem, dtype=bool)) if return_mask else dem

    with _rio.open(matchtag_path) as src:
        matchtag = src.read(1)
    with _rio.open(bitmask_path) as src:
        bitmask = src.read(1)

    if matchtag.shape != dem.shape or bitmask.shape != dem.shape:
        raise ValueError(
            f"Shape mismatch in quality files for {os.path.basename(strip_path)}: "
            f"dem={dem.shape}, matchtag={matchtag.shape}, bitmask={bitmask.shape}"
        )

    keep = _np.isin(matchtag, list(matchtag_keep)) & _np.isin(bitmask, list(bitmask_keep))
    dem[~keep] = _np.nan
    if nodata is not None:
        dem[dem == nodata] = _np.nan
    return (dem, keep) if return_mask else dem
