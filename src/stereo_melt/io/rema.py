"""REMA ingestion: S3 mosaic tile listing/downloads, strip search via pdemtools,
strip download via pdt.load.from_search, and pygmt intersection plots."""

import os

import boto3
import geopandas as gpd
import pandas as pd
import pdemtools as pdt
import pygmt
from botocore import UNSIGNED
from botocore.config import Config

from .. import config
from .masks import create_low_velocity_rock_polygons


def plot_strip_intersections_with_polygons_pygmt_v2(
    strips_df, lv_gdf, rock_gdf, output_path="/wd2/projects/stereo_melt/plots/intersections_plot.png"
):
    """Plot REMA strips and their intersections with low-velocity and rock polygons using PyGMT."""
    # Ensure all data is in EPSG:3031 (Antarctic Polar Stereographic)
    strips_df = strips_df.to_crs("EPSG:3031")
    lv_gdf = lv_gdf.to_crs("EPSG:3031")
    rock_gdf = rock_gdf.to_crs("EPSG:3031")

    # Define the region (total bounds of all data)
    all_bounds = gpd.GeoDataFrame(pd.concat([strips_df, lv_gdf, rock_gdf])).total_bounds
    region = [all_bounds[0], all_bounds[2], all_bounds[1], all_bounds[3]]  # [xmin, xmax, ymin, ymax]
    print(f"Plotting region: {region}")

    # Projection for Antarctic Polar Stereographic
    projection = "x1:550000"

    # Create a PyGMT figure
    fig = pygmt.Figure()

    # Plot base map
    fig.basemap(region=region, projection=projection, frame=["a", "+tREMA Strips and Intersections"])

    # Plot low-velocity polygons
    for polygon in lv_gdf.geometry:
        if polygon.is_valid:
            wkt = polygon.wkt
            fig.plot(data=wkt, pen="1p,blue", fill="blue", transparency=70)

    # Plot rock polygons
    for polygon in rock_gdf.geometry:
        if polygon.is_valid:
            wkt = polygon.wkt
            fig.plot(data=wkt, pen="1p,brown", fill="brown", transparency=70)

    # Plot REMA strips
    for polygon in strips_df.geometry:
        if polygon.is_valid:
            wkt = polygon.wkt
            fig.plot(data=wkt, pen="2p,red", fill="none")

    # Save the figure
    fig.savefig(output_path)
    print(f"✅ Intersection plot saved to {output_path}")


def plot_strip_intersections_with_polygons_pygmt(
    strips_df, lv_gdf, rock_gdf, output_path="/wd2/projects/stereo_melt/plots/intersections_plot.png"
):
    """Plot REMA strips and their intersections with low-velocity and rock polygons using PyGMT."""
    # Ensure all data is in EPSG:3031 (Antarctic Polar Stereographic)
    strips_df = strips_df.to_crs("EPSG:3031")
    lv_gdf = lv_gdf.to_crs("EPSG:3031")
    rock_gdf = rock_gdf.to_crs("EPSG:3031")

    # Create a PyGMT figure
    fig = pygmt.Figure()

    # Define the region and projection (Antarctic Polar Stereographic)
    bounds = strips_df.total_bounds
    region = [bounds[0], bounds[2], bounds[1], bounds[3]]  # [xmin, xmax, ymin, ymax]
    print(f"Region: {region}")

    print(f"Strips bounds: {strips_df.total_bounds}")
    print(f"Low-velocity bounds: {lv_gdf.total_bounds}")
    print(f"Rock bounds: {rock_gdf.total_bounds}")

    projection = "x1:1500000"  # Stereographic projection centered on the South Pole

    # Plot base map
    fig.basemap(region=region, projection=projection, frame=["a", "+tREMA Strips and Intersections"])

    # Plot low-velocity polygons
    fig.plot(data=lv_gdf, projection=projection, pen="1p,blue", close=True, fill="blue", transparency=70)

    # Plot rock polygons
    fig.plot(data=rock_gdf, projection=projection, pen="1p,brown", close=True, fill="brown", transparency=70)

    # Plot REMA strips
    fig.plot(data=strips_df, pen="2p,red", close=True)

    # Save the figure
    fig.savefig(output_path)
    print(f"✅ Intersection plot saved to {output_path}")


def list_rema_v2_tiles(roi_polygon, s3_bucket="pgc-opendata-dems", prefix="rema/mosaics/v2.0/32m"):
    """
    List REMA V2 tiles overlapping with a given polygon using a tile index shapefile.

    Parameters:
    - roi_polygon: Shapely Polygon defining the region of interest.
    - s3_bucket: S3 bucket name (default: "pgc-opendata-dems").
    - prefix: Path prefix for REMA mosaic tiles (default: "rema/mosaic/v2.0/32m").

    Returns:
    - tile_paths: List of S3 paths to the REMA tiles.
    """
    print("Loading tile index shapefile...")
    tiles_gdf = gpd.read_file(config.MOSAIC_INDEX_SHP)

    # Ensure the ROI and tile index are in the same CRS
    if tiles_gdf.crs != "EPSG:3031":  # REMA tiles are in EPSG:3031
        print("Reprojecting tile index to EPSG:3031...")
        tiles_gdf = tiles_gdf.to_crs("EPSG:3031")

    # Find overlapping tiles
    print("Performing spatial intersection to find overlapping tiles...")
    roi_gdf = gpd.GeoDataFrame({"geometry": [roi_polygon]}, crs=tiles_gdf.crs)
    overlapping_tiles = gpd.overlay(tiles_gdf, roi_gdf, how="intersection")

    if overlapping_tiles.empty:
        print("No tiles found for the given region of interest.")
        return []

    # Extract tile names and construct S3 paths
    tile_paths = []
    for idx, row in overlapping_tiles.iterrows():
        tile_name = row["tile"]  # Adjust 'tile' if column name differs
        tile_path = f"s3://{s3_bucket}/{prefix}/{tile_name}/{tile_name}_32m_v2.0_dem.tif"
        tile_paths.append(tile_path)

    print(f"Found {len(tile_paths)} tiles overlapping with the ROI.")
    return tile_paths


def download_tiles(tile_paths, output_dir):
    """
    Download REMA V2 tiles from AWS to a local directory, skipping already downloaded files.

    Parameters:
    - tile_paths: List of S3 paths to the tiles.
    - output_dir: Directory to save the tiles.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize S3 resource with unsigned configuration for public access
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))

    print(f"Starting download of {len(tile_paths)} REMA tiles...")

    # Iterate through the list of tile paths
    for idx, tile_path in enumerate(tile_paths):
        # Parse S3 bucket and key
        s3_parts = tile_path.replace("s3://", "").split("/", 1)
        bucket = s3_parts[0]
        key = s3_parts[1]
        local_file = os.path.join(output_dir, os.path.basename(key))

        # Check if the file already exists locally
        if os.path.exists(local_file):
            print(f"[{idx + 1}/{len(tile_paths)}] File already exists: {local_file}")
            continue  # Skip to the next file

        # Attempt to download the file
        try:
            print(f"[{idx + 1}/{len(tile_paths)}] Downloading {tile_path}...")
            s3.Bucket(bucket).download_file(key, local_file)
            print(f"Saved to: {local_file}")
        except Exception as e:
            print(f"Error downloading {tile_path}: {e}")

    print("Download process completed.")


def merge_tiles(tile_paths, output_path):
    """
    Merge REMA V2 tiles into a single GeoTIFF.

    Parameters:
    - tile_paths: List of local paths to the tiles.
    - output_path: Path to the merged GeoTIFF.
    """
    import rasterio
    from rasterio.merge import merge

    print("Merging tiles...")
    rasters = [rasterio.open(tile) for tile in tile_paths]
    mosaic, transform = merge(rasters)

    out_meta = rasters[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Merged mosaic saved to {output_path}")


def list_rema_strips_v2(
    polygon,
    date_range=None,
    velocity_netcdf_path=None,
    bedmachine_path=None,
    output_path=None,
    visualize_path=None
):
    """List REMA strips overlapping with a polygon using low-velocity and rock exposure filters."""
    print("🔍 Searching for REMA strips using pDEMtools...")

    try:
        # Step 1: Query REMA Strips
        if date_range:
            start_date, end_date = date_range
            strips_df = pdt.search(
                index_fpath=config.STRIP_INDEX_SHP,
                bounds=polygon,
                dates=(start_date, end_date),
                accuracy=2
            )
        else:
            strips_df = pdt.search(
                index_fpath=config.STRIP_INDEX_SHP,
                bounds=polygon,
                accuracy=2
            )

        if strips_df.empty:
            print("❌ No REMA strips found.")
            return None

        print(f"✅ Found {len(strips_df)} REMA strips. Verifying geometry and CRS...")

        # Step 2: Geometry Validation
        strips_df = strips_df[~strips_df.geometry.is_empty]
        strips_df = strips_df[strips_df.geometry.is_valid]

        # Reproject to match ROI CRS
        if str(strips_df.crs) != "EPSG:4326":
            print("🔄 Reprojecting strips_df to EPSG:4326...")
            strips_df = strips_df.to_crs("EPSG:4326")

        print(f"✅ Remaining strips after geometry validation: {len(strips_df)}")

        # Step 3: Generate Low-Velocity and Rock Polygons
        low_velocity_polygons, rock_polygons = create_low_velocity_rock_polygons(
            velocity_netcdf_path,
            bedmachine_path,
            polygon,
            output_path,
            velocity_threshold=10
        )

        # Step 4: Create GeoDataFrame for Polygons
        lv_gdf = gpd.GeoDataFrame(geometry=low_velocity_polygons, crs="EPSG:3031").to_crs("EPSG:4326")
        rock_gdf = gpd.GeoDataFrame(geometry=rock_polygons, crs="EPSG:3031").to_crs("EPSG:4326")

        # Step 5: Filter Strips by Intersection with Polygons
        valid_strips = strips_df[
            strips_df.geometry.apply(
                lambda g: lv_gdf.intersects(g).any() or rock_gdf.intersects(g).any()
            )
        ]

        print(f"✅ Filtered down to {len(valid_strips)} REMA strips after polygon intersection.")

        # Step 6: Visualization
        if visualize_path:
            print("🔄 Visualizing strips and overlaps...")
            plot_strip_intersections_with_polygons_pygmt(
                strips_df=strips_df,
                rock_gdf=rock_gdf,
                lv_gdf=lv_gdf,
                output_path="./plots/intersections_plot.png"
            )
        # Step 7: Save Filtered Strips (if output_path is provided)
        if output_path:
            valid_strips.to_file(output_path, driver="GeoJSON")
            print(f"✅ Filtered strips saved to {output_path}")

        return valid_strips

    except Exception as e:
        print(f"[ERROR] Failed to list REMA strips: {e}")
        return None


def list_rema_strips_old(polygon, date_range=None, velocity_netcdf_path=None, bedmachine_path=None, output_path=None, visualize_path=None):
    """List REMA strips overlapping with a polygon using low-velocity and rock exposure filters (legacy)."""
    print("🔍 Searching for REMA strips using pDEMtools...")

    try:
        # Step 1: Query REMA Strips
        if date_range:
            start_date, end_date = date_range
            strips_df = pdt.search(
                index_fpath=config.STRIP_INDEX_SHP,
                bounds=polygon,
                dates=(start_date, end_date),
                accuracy=2,
            )
        else:
            strips_df = pdt.search(
                index_fpath=config.STRIP_INDEX_SHP,
                bounds=polygon,
                dates=(start_date, end_date),
                accuracy=2
            )

        if strips_df.empty:
            print("❌ No REMA strips found.")
            return None

        print(f"✅ Found {len(strips_df)} REMA strips. Generating polygons...")

        # Check CRS of strips_df
        print("🔄 Checking CRS of strips_df...")
        print(f"CRS of strips_df: {strips_df.crs}")

        # Step 2: Generate Low-Velocity and Rock Polygons
        low_velocity_polygons, rock_polygons = create_low_velocity_rock_polygons(
            velocity_netcdf_path,
            bedmachine_path,
            polygon,
            output_path,
            velocity_threshold=10
        )

        # Step 3: Create GeoDataFrame for Polygons
        lv_gdf = gpd.GeoDataFrame(geometry=low_velocity_polygons, crs="EPSG:3031")
        rock_gdf = gpd.GeoDataFrame(geometry=rock_polygons, crs="EPSG:3031")

        # Step 6: Visualization
        print("🔄 Visualizing strips and overlaps...")
        plot_strip_intersections_with_polygons_pygmt(
            strips_df=strips_df,
            rock_gdf=rock_gdf,
            lv_gdf=lv_gdf,
        )

        # Step 4: Filter Strips by Intersection with Polygons
        valid_strips = strips_df[
            strips_df.geometry.apply(lambda g: lv_gdf.intersects(g).any() or rock_gdf.intersects(g).any())
        ]

        print(f"✅ Filtered down to {len(valid_strips)} REMA strips after polygon intersection.")
        return valid_strips

    except Exception as e:
        print(f"[ERROR] Failed to list REMA strips: {e}")
        return None


def list_rema_strips(polygon, date_range=None):
    """List REMA strips overlapping with a polygon (no control-surface filter)."""
    print("🔍 Searching for REMA strips using pDEMtools...")

    try:
        # Step 1: Query REMA Strips
        if date_range:
            start_date, end_date = date_range
            strips_df = pdt.search(
                index_fpath=config.STRIP_INDEX_SHP,
                bounds=polygon,
                dates=(start_date, end_date),
                accuracy=2
            )
        else:
            strips_df = pdt.search(
                index_fpath=config.STRIP_INDEX_SHP,
                bounds=polygon,
                accuracy=2
            )

        if strips_df.empty:
            print("❌ No REMA strips found.")
            return None

        print(f"✅ Found {len(strips_df)} REMA strips. Verifying geometry and CRS...")

        # Step 2: Geometry Validation
        strips_df = strips_df[~strips_df.geometry.is_empty]
        strips_df = strips_df[strips_df.geometry.is_valid]

        return strips_df

    except Exception as e:
        print(f"[ERROR] Failed to list REMA strips: {e}")
        return None


def download_rema_strips(gdf, output_dir):
    """
    Download REMA strips using a GeoDataFrame from pDEMtools, skipping already downloaded files.

    Parameters:
    - gdf: GeoDataFrame containing search results for REMA strips.
    - output_dir: Directory to save the downloaded REMA strips.
    """
    # Validate input GeoDataFrame
    if gdf is None or gdf.empty:
        print("No REMA strips to download.")
        return

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting download of {len(gdf)} REMA strips...")

    # Iterate through each row in the GeoDataFrame
    for idx, row in gdf.iterrows():
        dem_id = row.get("dem_id", f"unknown_id_{idx}")
        filename = f"{dem_id}.tif"
        local_path = os.path.join(output_dir, filename)

        # Check if the file already exists
        if os.path.exists(local_path):
            print(f"[{idx + 1}/{len(gdf)}] File already exists: {local_path}")
            continue

        try:
            # Download and save the DEM
            print(f"[{idx + 1}/{len(gdf)}] Downloading DEM ID: {dem_id}")
            dem = pdt.load.from_search(row, bitmask=True)
            dem.compute()  # Force download
            dem.rio.to_raster(local_path, compress='ZSTD', predictor=3, zlevel=1)
            print(f"Saved to: {local_path}")
        except Exception as e:
            print(f"[{idx + 1}/{len(gdf)}] Failed to download DEM ID {dem_id}: {e}")

    print("Download process complete.")
