"""Build coregistration reference datasets: IS2 ATL06 over slow-velocity grounded ice
(dynamic control surfaces) and rock elevation points from REMA mosaics (static control
surfaces). These are fed to ASP pc_align via `combine into CSV` in asp.py."""

import os
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import sliderule
from pyproj import Transformer
from rasterio.mask import mask
from sliderule import icesat2


def extract_rock_elevations_from_mosaics(
    strip_boundary,
    rock_shapefile,
    mosaic_dir,
    output_csv
):
    """
    Extract rock elevation points from REMA mosaic DEM tiles using rock polygons.

    Parameters:
    - strip_boundary (Polygon): Shapely Polygon representing the strip boundary.
    - rock_shapefile (str): Path to the shapefile containing exposed rock polygons.
    - mosaic_dir (str): Directory containing downloaded REMA mosaic tiles.
    - output_csv (str): Path to save extracted rock elevation points.
    """
    if not os.path.exists(mosaic_dir):
        raise FileNotFoundError(f"❌ Mosaic directory not found: {mosaic_dir}")

    if not os.path.exists(rock_shapefile):
        raise FileNotFoundError(f"❌ Rock shapefile not found: {rock_shapefile}")

    print("🔍 Loading rock shapefile...")
    rock_gdf = gpd.read_file(rock_shapefile)
    if rock_gdf.empty:
        raise ValueError("❌ Rock shapefile is empty.")

    # Ensure CRS matches DEM tiles (EPSG:3031)
    if rock_gdf.crs != "EPSG:3031":
        print("🌍 Reprojecting rock shapefile to EPSG:3031...")
        rock_gdf = rock_gdf.to_crs("EPSG:3031")

    # Convert strip boundary to GeoDataFrame
    strip_boundary_gdf = gpd.GeoDataFrame(geometry=[strip_boundary], crs="EPSG:3031")

    # Intersection of rock polygons with strip boundary
    print("🔄 Calculating intersection of rock polygons with strip boundary...")
    intersected_rock_gdf = gpd.overlay(rock_gdf, strip_boundary_gdf, how="intersection")

    if intersected_rock_gdf.empty:
        print("❌ No overlap between strip boundary and rock polygons. Skipping extraction.")
        return

    # Save intersection for debugging (optional)
    intersection_output_path = os.path.join(os.path.dirname(output_csv), "rock_strip_intersection.shp")
    intersected_rock_gdf.to_file(intersection_output_path)
    print(f"✅ Intersection shapefile saved to {intersection_output_path}")

    # Initialize list to store rock elevation points
    elevation_points = []

    # Iterate through mosaic tiles
    print("🔄 Processing DEM mosaic tiles for rock elevation extraction...")
    for tile_file in os.listdir(mosaic_dir):
        if tile_file.endswith(".tif"):
            tile_path = os.path.join(mosaic_dir, tile_file)
            print(f"📊 Processing tile: {tile_file}")

            try:
                with rasterio.open(tile_path) as src:
                    # Mask the DEM using the intersected rock polygons
                    out_image, out_transform = mask(src, intersected_rock_gdf.geometry, crop=True)
                    out_image = out_image[0]  # Extract the single DEM band

                    # Get coordinates of valid (non-NaN) elevation points
                    valid_mask = ~np.isnan(out_image)
                    rows, cols = np.where(valid_mask)
                    xs, ys = rasterio.transform.xy(out_transform, rows, cols)
                    elevations = out_image[valid_mask]

                    for x, y, z in zip(xs, ys, elevations):
                        elevation_points.append({
                            "x": x,
                            "y": y,
                            "elevation": z,
                            "tile": tile_file
                        })
            except Exception as e:
                print(f"❌ Failed to process tile {tile_file}: {e}")

    # Save elevation points to CSV
    if elevation_points:
        elevation_df = pd.DataFrame(elevation_points)
        elevation_df.to_csv(output_csv, index=False)
        print(f"✅ Rock elevation points saved to {output_csv}")
    else:
        print("❌ No valid rock elevation points found across all tiles.")


def extract_rock_elevations(dem_path, rock_shapefile, output_csv):
    """
    Extract elevations and coordinates from a DEM for rock regions defined in a shapefile.

    Parameters:
    - dem_path (str): Path to the DEM mosaic.
    - rock_shapefile (str): Path to the rock shapefile.
    - output_csv (str): Path to save extracted rock elevations and coordinates.
    """
    print("🔍 Extracting rock elevations from DEM...")
    rock_gdf = gpd.read_file(rock_shapefile)

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform

        elevation_data = []
        for _, row in rock_gdf.iterrows():
            # Create a mask for the current rock geometry
            mask_ = rasterio.features.geometry_mask(
                [row.geometry],
                transform=transform,
                out_shape=src.shape,
                invert=True
            )

            # Extract DEM values within the mask
            rock_values = dem[mask_]
            rows, cols = np.where(mask_)
            valid_mask = ~np.isnan(rock_values)
            rock_values = rock_values[valid_mask]
            rows, cols = rows[valid_mask], cols[valid_mask]

            # Compute x, y coordinates for valid points
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")

            # Append data to the elevation_data list
            elevation_data.extend([
                {"easting": x, "northing": y, "h_mean": z}
                for x, y, z in zip(xs, ys, rock_values)
            ])

    # Save the extracted data to a CSV
    if elevation_data:
        elevation_df = pd.DataFrame(elevation_data)
        elevation_df.to_csv(output_csv, index=False)
        print(f"✅ Rock elevations and coordinates saved to {output_csv}")
    else:
        print("❌ No valid rock elevation data found.")


def download_icesat2_data_v2(
    strip_boundary,
    slow_velocity_shapefile,
    dem_center_date,
    output_dir,
    time_window=5
):
    """
    Download IceSat-2 elevation data in slow-velocity regions within a time window.

    Parameters:
    - strip_boundary (Polygon): Shapely Polygon representing the strip boundary.
    - slow_velocity_shapefile (str): Path to the shapefile defining slow-velocity regions.
    - dem_center_date (str): Center date of the DEM strip in 'YYYY-MM-DD' format.
    - output_dir (str): Directory to save the downloaded IceSat-2 data.
    - time_window (int): Number of days before/after DEM center date to include.

    Returns:
    - str: Path to the saved CSV file containing IceSat-2 data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the slow velocity shapefile
    print("🔍 Loading slow-velocity shapefile...")
    roi_gdf = gpd.read_file(slow_velocity_shapefile)
    if roi_gdf.empty:
        raise ValueError("❌ Slow-velocity shapefile is empty.")

    # Ensure CRS compatibility
    if roi_gdf.crs != "EPSG:4326":
        print("🌍 Reprojecting shapefile to EPSG:4326...")
        roi_gdf = roi_gdf.to_crs("EPSG:4326")
    strip_boundary_4326 = gpd.GeoDataFrame(geometry=[strip_boundary], crs="EPSG:3031").to_crs("EPSG:4326")

    # Union of strip boundary and slow velocity regions
    intersection_gdf = gpd.overlay(
        roi_gdf,
        strip_boundary_4326,
        how="intersection"
    )
    intersection_gdf.to_file('./low_velocity_strip.shp')
    # Convert intersection to Sliderule-compatible region using sliderule.toregion
    print("🔄 Generating region polygon for Sliderule query...")
    combined_region = intersection_gdf.geometry.unary_union
    if combined_region.is_empty:
        print("❌ Combined region is empty after intersection. Skipping download.")
        return None

    # Define temporal filter
    dem_date = datetime.strptime(dem_center_date, "%Y-%m-%d")
    start_date = (dem_date - timedelta(days=time_window)).strftime("%Y-%m-%d")
    end_date = (dem_date + timedelta(days=time_window)).strftime("%Y-%m-%d")

    print(f"📅 Downloading IceSat-2 data from {start_date} to {end_date}...")

    sliderule.earthdata.set_max_resources(1000)
    region = sliderule.toregion(intersection_gdf)

    # Query IceSat-2 data
    parms = {
        "poly": region["poly"],
        "time": [start_date, end_date],
        "srt": icesat2.SRT_LAND,
        "cnf": icesat2.CNF_SURFACE_HIGH,
        "ats": 7.0,
        "cnt": 10,
        "len": 40.0,
        "res": 20.0,
    }

    try:
        print("🔄 Querying Sliderule for IceSat-2 data...")
        results = icesat2.atl06p(parms)

        if results.empty:
            print("❌ No IceSat-2 data found for the given region and time window.")
            return None
        # Project data to EPSG:3031
        print("🌍 Projecting IceSat-2 data to EPSG:3031...")
        print(results.columns)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        results["easting"], results["northing"] = transformer.transform(
            results.geometry.x, results.geometry.y
        )
        results = results[["easting", "northing", "h_mean"]]

        # Save results locally
        output_file = os.path.join(output_dir, f"icesat2_{dem_center_date}.csv")
        results.to_csv(output_file, index=False)
        print(f"✅ IceSat-2 data saved to {output_file}")

        return output_file

    except Exception as e:
        print(f"❌ Failed to download IceSat-2 data: {e}")
        return None


def download_icesat2_data(
    strip_boundary,
    slow_velocity_shapefile,
    grounded_shapefile,
    dem_center_date,
    output_dir,
    time_window=5
):
    """
    Download and filter IceSat-2 elevation data in slow-velocity grounded regions
    within a time window.

    Parameters:
    - strip_boundary (Polygon): Shapely Polygon representing the strip boundary.
    - slow_velocity_shapefile (str): Path to the shapefile defining slow-velocity regions.
    - grounded_shapefile (str): Path to the grounded-ice shapefile.
    - dem_center_date (str): Center date of the DEM strip in 'YYYY-MM-DD' format.
    - output_dir (str): Directory to save the filtered IceSat-2 data.
    - time_window (int): Number of days before/after DEM center date to include.

    Returns:
    - str: Path to the saved CSV file containing filtered IceSat-2 data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the slow velocity shapefile
    print("🔍 Loading slow-velocity shapefile...")
    roi_gdf = gpd.read_file(slow_velocity_shapefile)
    grounded_gdf = gpd.read_file(grounded_shapefile)
    if roi_gdf.empty:
        raise ValueError("❌ Slow-velocity shapefile is empty.")

    # Ensure CRS compatibility
    if roi_gdf.crs != "EPSG:4326":
        print("🌍 Reprojecting slow-velocity shapefile to EPSG:4326...")
        roi_gdf = roi_gdf.to_crs("EPSG:4326")
    if grounded_gdf.crs != "EPSG:4326":
        print("🌍 Reprojecting grounded-ice shapefile to EPSG:4326...")
        grounded_gdf = grounded_gdf.to_crs("EPSG:4326")

    strip_boundary_4326 = gpd.GeoDataFrame(geometry=[strip_boundary], crs="EPSG:3031").to_crs("EPSG:4326")

    # Intersection of strip boundary and slow-velocity region
    intersection_roi_strip_gdf = gpd.overlay(roi_gdf, strip_boundary_4326, how="intersection")
    intersection_gdf = gpd.overlay(grounded_gdf, intersection_roi_strip_gdf, how="intersection")

    if intersection_gdf.empty:
        print("❌ Combined region is empty after intersection. Skipping download.")
        return None

    print("✅ Intersection region created. Generating region polygon for query...")

    # Define temporal filter
    dem_date = datetime.strptime(dem_center_date, "%Y-%m-%d")
    start_date = (dem_date - timedelta(days=time_window)).strftime("%Y-%m-%d")
    end_date = (dem_date + timedelta(days=time_window)).strftime("%Y-%m-%d")

    print(f"📅 Querying IceSat-2 data from {start_date} to {end_date}...")

    # Query IceSat-2 data
    try:
        parms = {
            "poly": sliderule.toregion(intersection_gdf)["poly"],
            "time": [start_date, end_date],
            "srt": icesat2.SRT_LAND,
            "cnf": icesat2.CNF_SURFACE_HIGH,
            "ats": 7.0,
            "cnt": 10,
            "len": 40.0,
            "res": 20.0,
        }
        print("🔄 Querying Sliderule for IceSat-2 data...")
        results = icesat2.atl06p(parms)
        results = results.to_crs(epsg="4326+4979")

        if results.empty:
            print("❌ No IceSat-2 data found in the given region and time window.")
            return None

        # Project to EPSG:3031
        print("🌍 Projecting IceSat-2 data to EPSG:3031...")
        results["easting"], results["northing"] = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True).transform(
            results.geometry.x, results.geometry.y
        )

        # Filter results to ensure they fall in the intersection region
        results_gdf = gpd.GeoDataFrame(
            results,
            geometry=gpd.points_from_xy(results["easting"], results["northing"]),
            crs="EPSG:3031",
        )
        filtered_results = gpd.sjoin(results_gdf, intersection_gdf.to_crs("EPSG:3031"), how="inner")

        if filtered_results.empty:
            print("❌ Filtered IceSat-2 data is empty after spatial intersection.")
            return None

        # Save filtered data
        output_file = os.path.join(output_dir, f"icesat2_filtered_{dem_center_date}.csv")
        filtered_results[["easting", "northing", "h_mean"]].to_csv(output_file, index=False)
        print(f"✅ Filtered IceSat-2 data saved to {output_file}")

        return output_file

    except Exception as e:
        print(f"❌ Failed to download IceSat-2 data: {e}")
        return None
