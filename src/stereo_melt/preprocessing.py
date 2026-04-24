import scipy.io
import rasterio
from rasterio.mask import mask
import h5py
import geopandas as gpd
import netCDF4
import xarray as xr
import pandas as pd
from pyproj import Proj, transform
import numpy as np
from shapely.geometry import box, mapping, shape, Polygon
from botocore import UNSIGNED
from botocore.config import Config
from scipy.ndimage import median_filter, gaussian_filter, label
from skimage.measure import find_contours
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.stats import median_abs_deviation
import os
import boto3
from botocore.config import Config
from botocore import UNSIGNED
from . import config
from datetime import datetime
from sliderule import raster
import sliderule
import pdemtools as pdt
from rasterio.transform import from_origin
from rasterio.features import shapes
import pygmt
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import cdsapi
from pyproj import Transformer
import pyTMD
from pyTMD.compute import tide_elevations


def plot_strip_intersections_with_polygons_pygmt_v2(
    strips_df, lv_gdf, rock_gdf, output_path="/wd2/projects/stereo_melt/plots/intersections_plot.png"
):
    """
    Plot REMA strips and their intersections with low-velocity and rock polygons using PyGMT.

    Parameters:
    - strips_df: GeoDataFrame containing REMA strips.
    - lv_gdf: GeoDataFrame containing low-velocity polygons.
    - rock_gdf: GeoDataFrame containing rock polygons.
    - output_path: Path to save the generated plot.
    """
    import pygmt

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
    """
    Plot REMA strips and their intersections with low-velocity and rock polygons using PyGMT.

    Parameters:
    - strips_df: GeoDataFrame containing REMA strips.
    - lv_gdf: GeoDataFrame containing low-velocity polygons.
    - rock_gdf: GeoDataFrame containing rock polygons.
    - output_path: Path to save the generated plot.
    """

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

    projection = "x1:1500000" # Stereographic projection centered on the South Pole


    # Plot base map
    fig.basemap(region=region, projection=projection, frame=["a", "+tREMA Strips and Intersections"])

    # Plot low-velocity polygons
    #for polygon in lv_gdf.geometry:
    fig.plot(data=lv_gdf, projection=projection, pen="1p,blue", close=True, fill="blue", transparency=70)

    # Plot rock polygons
    
    fig.plot(data=rock_gdf, projection=projection, pen="1p,brown", close=True, fill="brown", transparency=70)


    # Plot REMA strips
    #for polygon in strips_df.geometry:
    fig.plot(data=strips_df, pen="2p,red", close=True)
    print("we made it here")

    # Save the figure
    fig.savefig(output_path)
    print(f"✅ Intersection plot saved to {output_path}")



def list_rema_v2_tiles(roi_polygon, s3_bucket="pgc-opendata-dems", prefix="rema/mosaics/v2.0/32m"):
    """
    List REMA V2 tiles overlapping with a given polygon using a tile index shapefile.

    Parameters:
    - roi_polygon: Shapely Polygon defining the region of interest.
    - tile_index_shapefile: Path to the REMA tile index shapefile.
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

def validate_polygon(polygon):
    """Validate if the given object is a valid Shapely Polygon."""
    if polygon is None:
        raise ValueError("❌ ROI polygon is None. Please ensure a valid polygon is passed.")
    if not isinstance(polygon, Polygon):
        raise TypeError(f"❌ ROI polygon is not a valid Shapely Polygon, got {type(polygon)} instead.")
    if polygon.is_empty:
        raise ValueError("❌ ROI polygon is empty. Please provide a valid polygon with geometry.")
    return True

def extend_bounds(bounds, extension=200000):
    """
    Extend bounding box by a given distance in meters.

    Parameters:
    - bounds (tuple): (min_x, min_y, max_x, max_y).
    - extension (float): Distance to extend in meters.

    Returns:
    - extended_bounds (tuple): Extended bounding box.
    """
    min_x, min_y, max_x, max_y = bounds
    return (min_x - extension, min_y - extension, max_x + extension, max_y + extension)


def subset_dataset(dataset, roi_polygon, extension=200000, spacing=10000, gap=5000):
    """
    Subset an xarray dataset based on an ROI polygon with optional extension.

    Parameters:
    - dataset (xarray.Dataset): The dataset to be subset.
    - roi_polygon (Shapely Polygon): The region of interest polygon.
    - extension (int): Extension in meters to add to the ROI bounds.

    Returns:
    - subset (xarray.Dataset): Subsetted xarray dataset.
    - extent_polygon (Shapely Polygon): Polygon representing the extent of the subset.
    """
    print("🔄 Subsetting dataset based on ROI polygon and extension...")

    if not roi_polygon or not isinstance(roi_polygon, Polygon) or roi_polygon.is_empty:
        raise ValueError("❌ Invalid ROI polygon provided. Ensure it's a valid Shapely Polygon.")

    # Calculate bounds with extension
    bounds = roi_polygon.bounds
    min_x, min_y, max_x, max_y = (
        bounds[0] - extension,
        bounds[1] - extension,
        bounds[2] + extension,
        bounds[3] + extension
    )

    print(f"🔹 Subset bounds: ({min_x}, {min_y}, {max_x}, {max_y})")

    # Subset the dataset
    subset = dataset.sel(
        x=slice(min_x, max_x),
        y=slice(max_y, min_y)  # y is typically inverted
    )

    # Adjust bounds with gap
    adjusted_min_y = min_y + gap
    adjusted_max_y = max_y - gap

    # Create alternating points for the comb polygon
    points = []
    for x in range(int(min_x), int(max_x) + 1):
        points.append((x, adjusted_min_y if len(points) % 2 == 0 else adjusted_max_y))

    # Close the polygon by connecting back to the starting point
    points.append((min_x, adjusted_min_y))

    # Create a single polygon
    extent_polygon = Polygon(points)








    # Ensure the polygon is filled
    #extent_polygon = Polygon([
    #    (min_x, min_y),
    #    (min_x, max_y),
    #    (max_x, max_y),
    #    (max_x, min_y),
    #    (min_x, min_y)
    #]).buffer(0)  # `buffer(0)` ensures valid geometry

    print("✅ Subsetting complete. Subset dimensions:", subset.dims)

    return subset, extent_polygon




def save_mask_to_raster(mask, x_coords, y_coords, output_path, crs="EPSG:3031", dtype="float32"):
    """
    Save a binary mask as a GeoTIFF file with a specified data type.
    
    Parameters:
    - mask (ndarray): 2D array mask.
    - x_coords (ndarray): X-coordinates of the mask grid.
    - y_coords (ndarray): Y-coordinates of the mask grid.
    - output_path (str): Path to save the output raster.
    - crs (str): Coordinate reference system (default: EPSG:3031).
    - dtype (str): Data type for the saved raster (default: float32).
    """
    print("🔄 Saving mask to raster...")

    if mask is None or not mask.any():
        raise ValueError("❌ Mask contains no valid data to save.")

    # Ensure mask is cast to the chosen dtype
    mask = mask.astype(dtype)
    print(f"✅ Final mask dtype: {mask.dtype}")
    print(f"🔹 Mask shape: {mask.shape}")
    print(f"🔹 X coords range: {x_coords.min()} to {x_coords.max()}")
    print(f"🔹 Y coords range: {y_coords.min()} to {y_coords.max()}")

    transform = from_origin(
        x_coords.min(), y_coords.max(),
        np.abs(x_coords[1] - x_coords[0]),
        np.abs(y_coords[1] - y_coords[0])
    )

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan if dtype == "float32" else -9999
    ) as dst:
        dst.write(mask, 1)
    
    print(f"✅ Mask saved successfully to {output_path}")



def raster_to_polygons(raster_path, value=1):
    """
    Convert a raster mask to polygons using Rasterio.
    """
    print("🔄 Converting raster to polygons...")

    polygons = []
    with rasterio.open(raster_path) as src:
        mask = src.read(1)
        transform = src.transform

        print(f"✅ Original Raster dtype: {mask.dtype}")
        print(f"✅ Raster shape: {mask.shape}")
        print(f"✅ Unique values in mask: {np.unique(mask)}")

        # Cast mask to uint8 for shapes extraction
        binary_mask = np.where(np.isnan(mask), 0, mask).astype(np.uint8)
        binary_mask[binary_mask != value] = 0
        binary_mask[binary_mask == value] = 1

        print(f"✅ Binary mask dtype after casting: {binary_mask.dtype}")
        print(f"✅ Unique values in binary mask: {np.unique(binary_mask)}")

        for geom, val in shapes(binary_mask, mask=binary_mask == value, transform=transform):
            if val == 1:
                polygons.append(shape(geom))

    print(f"✅ Extracted {len(polygons)} polygons from raster.")
    return polygons


def save_polygons_to_shapefile(polygons, output_path, crs="EPSG:3031"):
    """
    Save a list of polygons to a shapefile using Fiona.
    """
    print("🔄 Saving polygons to shapefile...")

    if not polygons:
        print("❌ No valid polygons to save.")
        return

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf.to_file(output_path)
    print(f"✅ Shapefile saved successfully to {output_path}")


def create_low_velocity_rock_polygons(
    velocity_netcdf_path,
    bedmachine_path,
    roi_polygon,
    output_dir,
    velocity_threshold=10
):
    """
    Create polygons for low-velocity and rock regions using Rasterio and Fiona.
    """
    os.makedirs(output_dir, exist_ok=True)
    lv_shapefile = os.path.join(output_dir, 'low_velocity_polygons.shp')
    rock_shapefile = os.path.join(output_dir, 'rock_polygons.shp')

    try:
        # Subset Velocity Dataset
        print("🔍 Subsetting velocity dataset...")
        ds_vel = xr.open_dataset(velocity_netcdf_path)
        ds_vel_subset, extent_polygon = subset_dataset(ds_vel, roi_polygon)

        speed = np.sqrt(ds_vel_subset['VX'].values**2 + ds_vel_subset['VY'].values**2)
        x = ds_vel_subset['x'].values
        y = ds_vel_subset['y'].values

        print(f"🔹 Speed array dtype: {speed.dtype}")
        print(f"🔹 Speed array min/max: {np.nanmin(speed)}, {np.nanmax(speed)}")

        low_velocity_mask = (np.nan_to_num(speed) < velocity_threshold).astype(np.uint8)
        print(f"✅ Low-velocity mask created. Unique values: {np.unique(low_velocity_mask)}")
        
        save_mask_to_raster(low_velocity_mask, x, y, os.path.join(output_dir, 'low_velocity_mask.tif'))
        low_velocity_polygons = raster_to_polygons(os.path.join(output_dir, 'low_velocity_mask.tif'))
        save_polygons_to_shapefile(low_velocity_polygons, lv_shapefile)

        # Subset BedMachine Dataset
        print("🔍 Subsetting BedMachine dataset...")
        ds_bm = xr.open_dataset(bedmachine_path)
        ds_bm_subset, extent_polygon = subset_dataset(ds_bm, roi_polygon)

        mask = ds_bm_subset['mask'].values
        bm_x = ds_bm_subset['x'].values
        bm_y = ds_bm_subset['y'].values

        print(f"🔹 BedMachine mask dtype: {mask.dtype}")
        print(f"🔹 BedMachine mask unique values: {np.unique(mask)}")

        rock_mask = (mask == 1).astype(np.uint8)
        print(f"✅ Rock mask created. Unique values: {np.unique(rock_mask)}")

        save_mask_to_raster(rock_mask, bm_x, bm_y, os.path.join(output_dir, 'rock_mask.tif'))
        rock_polygons = raster_to_polygons(os.path.join(output_dir, 'rock_mask.tif'))
        save_polygons_to_shapefile(rock_polygons, rock_shapefile)

        return low_velocity_polygons, rock_polygons, extent_polygon

    except Exception as e:
        print(f"[ERROR] Failed to create low-velocity and rock polygons: {e}")
        return [], []


def list_rema_strips_v2(
    polygon,
    date_range=None,
    velocity_netcdf_path=None,
    bedmachine_path=None,
    output_path=None,
    visualize_path=None
):
    """
    List REMA strips overlapping with a polygon using low-velocity and rock exposure filters.

    Parameters:
    - polygon: Shapely Polygon defining the region of interest.
    - date_range: Optional tuple of (start_date, end_date) in 'YYYY-MM-DD' format.
    - velocity_netcdf_path: Path to NetCDF file containing velocity data.
    - bedmachine_path: Path to NetCDF file containing BedMachine data.
    - output_path: Optional path to save the filtered GeoDataFrame.
    - visualize_path: Optional path to save the visualization of strips and overlaps.

    Returns:
    - strips_df (GeoDataFrame): Filtered REMA strips intersecting low-velocity or rock regions.
    """
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
    """
    List REMA strips overlapping with a polygon using low-velocity and rock exposure filters.

    Parameters:
    - polygon: Shapely Polygon defining the region of interest.
    - date_range: Optional tuple of (start_date, end_date) in 'YYYY-MM-DD' format.
    - velocity_netcdf_path: Path to NetCDF file containing velocity data.
    - bedmachine_path: Path to NetCDF file containing BedMachine data.

    Returns:
    - strips_df (GeoDataFrame): Filtered REMA strips intersecting low-velocity or rock regions.
    """
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
                #min_aoi_frac=0.1
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
    """
    List REMA strips overlapping with a polygon using low-velocity and rock exposure filters.

    Parameters:
    - polygon: Shapely Polygon defining the region of interest.
    - date_range: Optional tuple of (start_date, end_date) in 'YYYY-MM-DD' format.

    Returns:
    - strips_df (GeoDataFrame): Filtered REMA strips intersecting low-velocity or rock regions, with full geometries restored.
    """


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
    - bounds: Optional bounds for cropping DEMs (default: None). Bounds should be in the format (minx, miny, maxx, maxy).
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





def load_mat_file(file_path):
    """
    Load a .mat file and return its content.
    """
    data = scipy.io.loadmat(file_path)
    return data

def load_geotiff(file_path):
    """
    Load a GeoTIFF file and return its metadata and data.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        data[data < -1e3] = np.nan  # Apply the NaN mask
        metadata = src.meta
    return data, metadata

def load_bedmachine(file_path):
    """
    Load BedMachine data from an HDF5 file.

    Parameters:
    - file_path: Path to the HDF5 file.

    Returns:
    - dict: A dictionary containing relevant datasets.
    """
    with h5py.File(file_path, 'r') as f:
        bedmachine_data = {
            "x": f["x"][:],
            "y": f["y"][:],
            "geoid": f["geoid"][:],
            "bed": f["bed"][:],
            "thickness": f["thickness"][:],
            "surface": f["surface"][:],
            "source": f["source"][:],
            "errbed": f["errbed"][:],
            "mask": f["mask"][:],
            "firn": f["firn"][:],
        }
    return bedmachine_data


def load_gravsoft_xyz(file_path):
    """
    Load GRAVSOFT grid data in ASCII XYZ format.

    Parameters:
    - file_path (str): Path to the ASCII XYZ file.

    Returns:
    - dict: A dictionary with 'lon', 'lat', and 'mdt' arrays.
    """
    try:
        # Read the ASCII file into a DataFrame
        data = pd.read_csv(
            file_path, 
            delim_whitespace=True, 
            header=None, 
            names=["lon", "lat", "mdt", "err"]
        )
        # Extract unique latitude and longitude
        lons = np.unique(data["lon"])
        lats = np.unique(data["lat"])
        
        # Reshape MDT values to a 2D grid
        mdt = data["mdt"].values.reshape(len(lats), len(lons))

        return {
            "lon": lons,
            "lat": lats,
            "mdt": mdt,
        }
    except Exception as e:
        raise ValueError(f"Error loading GRAVSOFT grid file: {e}")



def load_dtu10_mdt(file_path):
    """
    Load DTU10 Mean Dynamic Topography (MDT) data.

    Parameters:
    - file_path: Path to the DTU10 MDT NetCDF file.
    """
    ds = load_gravsoft_xyz(file_path)
    return ds


def load_grounding_zone(shapefile_path):
    """
    Reads grounding zone positions from a shapefile.

    Parameters:
    - shapefile_path (str): Path to the grounding zone shapefile.

    Returns:
    - dict: A dictionary with keys:
        'coordinates' - A list of tuples (x, y) or (lat, lon) for each geometry.
        'properties'  - A list of dictionaries containing feature attributes.
    """
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Ensure the CRS is consistent
    gdf = gdf.to_crs("EPSG:3031")  # Antarctic Polar Stereographic (EPSG:3031)

    # Extract coordinates and properties
    features = {
        "coordinates": [],
        "properties": []
    }
    
    for _, row in gdf.iterrows():
        geometry = row.geometry
        if geometry.is_empty:
            continue
        
        # Handle different geometry types
        if geometry.geom_type == "Polygon":
            coords = list(geometry.exterior.coords)
        elif geometry.geom_type == "MultiPolygon":
            coords = [list(poly.exterior.coords) for poly in geometry.geoms]
        else:
            raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")

        # Append coordinates and properties
        features["coordinates"].append(coords)
        features["properties"].append(row.to_dict())

    return features



def measuresann_interp(variable, data_dir, xi, yi, method="linear", inpaint_nans=False):
    """
    Interpolate MEaSUREs dataset for the given variable.

    Parameters:
    - variable (str): 'velocity', 'speed', 'error', or 'count'.
    - xi, yi (array-like): Coordinates in polar stereographic meters.
    - data_dir (str): Directory containing MEaSUREs .nc files.
    - method (str): Interpolation method ('linear', 'nearest'). Default is 'linear'.
    - inpaint_nans (bool): Whether to fill missing values using griddata. Default is False.

    Returns:
    - (vx, vy, t) for 'velocity' or 'error', or (v, t) for 'speed' or 'count'.
    """
    # Supported variables and corresponding NetCDF variable names
    variable_mapping = {
        "velocity": ("vx", "vy"),
        "error": ("vx_error", "vy_error"),
    }

    # Check if the variable is valid
    if variable.lower() not in variable_mapping:
        raise ValueError(f"Invalid variable '{variable}'. Choose from {list(variable_mapping.keys())}.")

    # List all NetCDF files in the data directory
    file_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")])
    if not file_list:
        raise FileNotFoundError(f"No NetCDF files found in {data_dir}.")

    # Determine the variable names to read
    vars_to_read = variable_mapping[variable.lower()]
    is_velocity = len(vars_to_read) == 2  # True for 'velocity' and 'error'

    # Initialize storage for interpolated values
    values_x, values_y = [], []
    t_values = np.arange(2018, 2021)

    for file in file_list:
        # Open the NetCDF file
        with xr.open_dataset(file) as ds:
            # Get spatial and temporal dimensions
            x = ds["x"].values
            y = ds["y"].values
            
            # Read the variables
            data_x = ds[vars_to_read[0]].values
            if is_velocity:
                data_y = ds[vars_to_read[1]].values

            # Interpolate using RegularGridInterpolator
            interpolator_x = RegularGridInterpolator((y, x), data_x, method=method, bounds_error=False, fill_value=np.nan)
            interpolated_x = interpolator_x(np.column_stack((yi.ravel(), xi.ravel()))).reshape(xi.shape)

            if is_velocity:
                interpolator_y = RegularGridInterpolator((y, x), data_y, method=method, bounds_error=False, fill_value=np.nan)
                interpolated_y = interpolator_y(np.column_stack((yi.ravel(), xi.ravel()))).reshape(xi.shape)

            # Handle missing values if requested
            if inpaint_nans:
                interpolated_x = _inpaint_nans(interpolated_x)
                if is_velocity:
                    interpolated_y = _inpaint_nans(interpolated_y)

            # Store interpolated values
            values_x.append(interpolated_x)
            if is_velocity:
                values_y.append(interpolated_y)

    # Combine results into numpy arrays
    values_x = np.stack(values_x, axis=-1)
    if is_velocity:
        values_y = np.stack(values_y, axis=-1)

    # Return results based on the variable type
    t_values = np.array(t_values).flatten()
    if variable.lower() == "speed":
        return np.sqrt(values_x**2 + values_y**2), t_values
    elif variable.lower() in ["velocity", "error"]:
        return values_x, values_y, t_values
    elif variable.lower() == "count":
        return values_x, t_values


def _inpaint_nans(data):
    """
    Fill NaN values in a 2D or 3D array using griddata interpolation.

    Parameters:
    - data (ndarray): Input array with NaN values.

    Returns:
    - ndarray: Array with NaNs filled.
    """
    coords = np.array(np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"))
    valid_mask = ~np.isnan(data)

    # Interpolate for each time step if 3D
    if data.ndim == 3:
        for i in range(data.shape[2]):
            time_slice = data[:, :, i]
            data[:, :, i] = griddata(
                coords[:, valid_mask[:, :, i]].T,
                time_slice[valid_mask[:, :, i]],
                coords.reshape(2, -1).T,
                method="linear",
            ).reshape(data.shape[:2])
    else:
        data = griddata(
            coords[:, valid_mask].T,
            data[valid_mask],
            coords.reshape(2, -1).T,
            method="linear",
        ).reshape(data.shape)

    return data

# TODO add ability to include other velocity products.

def create_velocity_stack(x, y, vel_dir, MYrs,std_thr=2.5, QAnn=None, QYrs=None):
    """
    Create a velocity stack using MEaSUREs annual velocities and optionally additional velocity fields (QAnn).

    Parameters:
    - x, y (1D arrays): Original grid coordinates.
    - measuresann_interp_func (function): Function to interpolate MEaSUREs annual velocities.
    - measures_interp_func (function): Function to interpolate another velocity dataset.
    - MYrs (1D array): Years corresponding to MEaSUREs annual velocities.
    - QAnn (dict, optional): Dictionary containing additional velocity fields (vx, vy). Defaults to None.
    - QYrs (1D array, optional): Years corresponding to QAnn velocity fields. Defaults to None.

    Returns:
    - VelAnn (dict): Dictionary containing stacked and filtered velocity data.
    """
    # Define new grid and map info
    XB = [min(x) - 10000, max(x) + 10000]  # Extend bounds by 10 km
    YB = [min(y) - 10000, max(y) + 10000]
    dx = abs(x[1] - x[0])  # Grid resolution
    x_new = np.arange(XB[0], XB[1] + dx, dx)
    y_new = np.arange(YB[0], YB[1] + dx, dx)

    # Create grid
    xg, yg = np.meshgrid(x_new, y_new)

    # Interpolate MEaSUREs annual velocities
    MeasAnn_vx, MeasAnn_vy, MeasAnn_t = measuresann_interp("velocity", vel_dir, xg, yg, method="linear", inpaint_nans=True)

    # Initialize VelAnn dictionary
    VelAnn = {
        "x": x_new,
        "y": y_new,
        "vx": [],
        "vy": [],
        "Years": [],
    }

    # Match MEaSUREs data to MYrs
    Mapidx = np.argmin(np.abs(MeasAnn_t - np.array(MYrs)[:, None]), axis=1)
    print(MeasAnn_vx)
    print(Mapidx)
    MeasAnn_vx = MeasAnn_vx[:, :, Mapidx]
    MeasAnn_vy = MeasAnn_vy[:, :, Mapidx]
    VelAnn["Years"].extend(MYrs)

    # Stack additional QAnn velocity data if provided
    if QAnn and QYrs:
        VelAnn["vx"] = np.concatenate((MeasAnn_vx, QAnn["vx"]), axis=-1)
        VelAnn["vy"] = np.concatenate((MeasAnn_vy, QAnn["vy"]), axis=-1)
        VelAnn["Years"].extend(QYrs)
    else:
        VelAnn["vx"] = MeasAnn_vx
        VelAnn["vy"] = MeasAnn_vy

    # Calculate statistics
    VelAnn["vx_mean"] = np.nanmean(VelAnn["vx"], axis=-1)
    VelAnn["vy_mean"] = np.nanmean(VelAnn["vy"], axis=-1)
    VelAnn["vx_med"] = np.nanmedian(VelAnn["vx"], axis=-1)
    VelAnn["vy_med"] = np.nanmedian(VelAnn["vy"], axis=-1)
    VelAnn["vx_std"] = np.nanstd(VelAnn["vx"], axis=-1)
    VelAnn["vy_std"] = np.nanstd(VelAnn["vy"], axis=-1)

    # Filter and smooth velocities
    for ii in range(VelAnn["vx"].shape[-1]):
        # Process vx
        vx_tmp = VelAnn["vx"][:, :, ii]
        mask_vx = (vx_tmp > VelAnn["vx_mean"] + std_thr * VelAnn["vx_std"]) | \
                  (vx_tmp < VelAnn["vx_mean"] - std_thr * VelAnn["vx_std"])
        vx_tmp[mask_vx | np.isnan(vx_tmp)] = VelAnn["vx_mean"][mask_vx | np.isnan(vx_tmp)]
        VelAnn["vx"][:, :, ii] = gaussian_filter(vx_tmp, sigma=30)

        # Process vy
        vy_tmp = VelAnn["vy"][:, :, ii]
        mask_vy = (vy_tmp > VelAnn["vy_mean"] + std_thr * VelAnn["vy_std"]) | \
                  (vy_tmp < VelAnn["vy_mean"] - std_thr * VelAnn["vy_std"])
        vy_tmp[mask_vy | np.isnan(vy_tmp)] = VelAnn["vy_mean"][mask_vy | np.isnan(vy_tmp)]
        VelAnn["vy"][:, :, ii] = gaussian_filter(vy_tmp, sigma=30)

    # Calculate velocity magnitude
    VelAnn["vMag"] = np.sqrt(VelAnn["vx"]**2 + VelAnn["vy"]**2)
    VelAnn["vMag_mean"] = np.nanmean(VelAnn["vMag"], axis=-1)
    VelAnn["vMag_med"] = np.nanmedian(VelAnn["vMag"], axis=-1)

    # Gradient and divergence calculations
    dx = np.diff(x_new)[0]
    dy = np.diff(y_new)[0]
    VelAnn["grad_vx"] = np.gradient(VelAnn["vx"], dx, axis=0)
    VelAnn["grad_vy"] = np.gradient(VelAnn["vy"], dy, axis=1)
    VelAnn["vdiv_stack"] = VelAnn["grad_vx"] + VelAnn["grad_vy"]

    return VelAnn



def compute_gradients_and_masks(VelAnn, Corrections):
    """
    Compute velocity gradients and create masks.

    Parameters:
    - VelAnn: Stacked velocity data.
    - Corrections: Dictionary containing surface height and mask data.

    Returns:
    - VelAnn: Updated velocity data with gradients and velocity magnitude.
    """
    dx = np.diff(VelAnn['x'])[0]
    dy = np.diff(VelAnn['y'])[0]

    # Calculate velocity gradients
    VelAnn['grad_vx'], _ = np.gradient(VelAnn['vx'], dx, axis=0)
    _, VelAnn['grad_vy'] = np.gradient(VelAnn['vy'], dy, axis=1)
    VelAnn['vdiv_stack'] = VelAnn['grad_vx'] + VelAnn['grad_vy']

    # Velocity magnitude
    VelAnn['vMag'] = np.sqrt(VelAnn['vx']**2 + VelAnn['vy']**2)
    VelAnn['vMag_mean'] = np.nanmean(VelAnn['vMag'], axis=-1)
    VelAnn['vMag_med'] = np.nanmedian(VelAnn['vMag'], axis=-1)

    # Create masks
    Corrections['rockPoly'] = create_polyshape(Corrections['mask'] == 1, Corrections['x'], Corrections['y'])
    return VelAnn

def create_polyshape(mask, x, y):
    """
    Generate polygons for a mask.

    Parameters:
    - mask: Binary mask indicating regions of interest.
    - x, y: Coordinates corresponding to the mask.

    Returns:
    - List of polygons representing the mask.
    """
    from shapely.geometry import Polygon

    polygons = []
    for region in mask:
        coords = np.column_stack(np.nonzero(region))
        if coords.shape[0] > 2:  # Only include valid polygons
            polygons.append(Polygon(coords))
    return polygons


# === Custom Loading Functions ===

def load_mat_file(file_path):
    """Load a .mat file and return its content."""
    return scipy.io.loadmat(file_path)


def load_geotiff(file_path):
    """Load a GeoTIFF file and return its metadata and data."""
    with rasterio.open(file_path) as src:
        data = src.read(1)
        data[data < -1e3] = np.nan  # Apply the NaN mask
        metadata = src.meta
    return data, metadata


def load_bedmachine(file_path):
    """Load BedMachine data from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        bedmachine_data = {
            "x": f["x"][:],
            "y": f["y"][:],
            "geoid": f["geoid"][:],
            "bed": f["bed"][:],
            "thickness": f["thickness"][:],
            "surface": f["surface"][:],
            "source": f["source"][:],
            "errbed": f["errbed"][:],
            "mask": f["mask"][:],
            "firn": f["firn"][:],
        }
    return bedmachine_data


def load_gravsoft_xyz(file_path):
    """Load GRAVSOFT grid data in ASCII XYZ format."""
    try:
        data = pd.read_csv(
            file_path, 
            delim_whitespace=True, 
            header=None, 
            names=["lon", "lat", "mdt", "err"]
        )
        lons = np.unique(data["lon"])
        lats = np.unique(data["lat"])
        mdt = data["mdt"].values.reshape(len(lats), len(lons))

        return {"lon": lons, "lat": lats, "mdt": mdt}
    except Exception as e:
        raise ValueError(f"Error loading GRAVSOFT grid file: {e}")

def apply_ibe_correction(dem_data, xs, ys, center_time, region):
    """
    Applies the Inverse Barometer Effect (IBE) correction to the DEM.

    Parameters:
    - dem_data (ndarray): The DEM elevation data.
    - xs (ndarray): X coordinates of DEM pixels.
    - ys (ndarray): Y coordinates of DEM pixels.
    - center_time (str): Center timestamp of the DEM in 'YYYY-MM-DDTHH:MM' format.
    - region (list): [North, West, South, East] lat/lon coordinates.

    Returns:
    - dem_data_corrected (ndarray): DEM after IBE correction.
    """

    print(f"🔄 Fetching ERA5 surface pressure for IBE correction...")
    try:
        era5_ds = fetch_era5_surface_pressure(center_time, region)
        
        # Convert pressure to hPa
        pressure = era5_ds.surface_pressure.values / 100.0

        # Create interpolator for pressure data
        pressure_interp = RegularGridInterpolator(
            (era5_ds.latitude.values, era5_ds.longitude.values), 
            pressure, 
            bounds_error=False, 
            fill_value=1013  # Default mean sea level pressure
        )

        # Interpolate pressure onto DEM grid
        pressure_values = pressure_interp((ys, xs))

        # Compute IBE correction (1 cm per hPa)
        ibe_correction = -1.0 * (pressure_values - 1013.0)

        # Apply correction to DEM
        dem_data_corrected = dem_data - ibe_correction.reshape(dem_data.shape)

        print(f"✅ IBE correction applied. Max adjustment: {np.nanmax(ibe_correction):.2f} cm")
        return dem_data_corrected

    except Exception as e:
        print(f"❌ Failed to apply IBE correction: {e}")
        return dem_data


def load_dtu10_mdt(file_path):
    """Load DTU10 Mean Dynamic Topography (MDT) data."""
    return load_gravsoft_xyz(file_path)


# === Fetch Surface Pressure from ERA5 ===

def fetch_era5_surface_pressure(center_time, dem_path):
    """
    Queries the CDS API for ERA5 surface pressure at a specific time and region.

    Parameters:
        center_time (datetime): Center timestamp of the DEM.
        dem_path (str): Path to the DEM file to extract lat/lon bounds.

    Returns:
        xarray.Dataset: ERA5 surface pressure dataset reprojected to EPSG:3031.
    """

    print(f"🛰 Fetching ERA5 surface pressure for {center_time} using DEM bounds from {dem_path}...")

    # Ensure center_time is a datetime object
    if not isinstance(center_time, datetime):
        raise ValueError(f"❌ Invalid center_time format: {center_time}. Expected a datetime object.")

    # Step 1: Extract DEM bounds and convert to lat/lon
    print("📍 Extracting lat/lon bounds from DEM...")
    with rasterio.open(dem_path) as src:
        bounds = src.bounds  # (left, bottom, right, top)
        transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)

        # Convert DEM bounds to lat/lon
        west, south = transformer.transform(bounds.left, bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)

        # Ensure lat/lon format is correct for CDS API
        era5_region = [north, west, south, east]  # [N, W, S, E]
    
    print(f"   🌍 ERA5 Query Region: {era5_region}")

    # Step 2: Download ERA5 Surface Pressure Data
    print("📡 Connecting to ERA5 CDS API...")
    c = cdsapi.Client()

    try:
        result = c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'variable': 'surface_pressure',
                'product_type': 'reanalysis',
                'year': str(center_time.year),
                'month': f"{center_time.month:02d}",  # Ensure zero-padding
                'day': f"{center_time.day:02d}",
                'time': f"{center_time.hour:02d}:00",  # Ensure HH:MM format
                'area': era5_region,  # [North, West, South, East]
                'format': 'netcdf',
            }
        )

        # Load NetCDF data into memory
        print("📥 Downloading ERA5 data...")
        data = io.BytesIO(result.download(target=None))
        era5_ds = xr.open_dataset(data)

    except Exception as e:
        raise RuntimeError(f"❌ Failed to download ERA5 data: {e}")

    # Step 3: Reproject ERA5 Data to EPSG:3031
    print("🔄 Reprojecting ERA5 data to EPSG:3031...")
    try:
        era5_lons, era5_lats = np.meshgrid(era5_ds.longitude.values, era5_ds.latitude.values)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

        # Transform lat/lon grid to EPSG:3031
        era5_xs, era5_ys = transformer.transform(era5_lons, era5_lats)

        # Add transformed coordinates to the dataset
        era5_ds = era5_ds.assign_coords({"x": (("latitude", "longitude"), era5_xs),
                                         "y": (("latitude", "longitude"), era5_ys)})
        
        print("✅ ERA5 data successfully reprojected to EPSG:3031!")

    except Exception as e:
        print(f"❌ Failed to reproject ERA5 data: {e}")

    return era5_ds

    
def apply_tidal_correction(xs, ys, center_time, model="CATS2008"):
    """Apply tidal correction to the DEM strip using pyTMD."""
    print("🌊 Applying Tidal Correction with pyTMD...")

    # Convert polar stereographic (EPSG:3031) to geographic (EPSG:4326)
    proj_polar = Proj("epsg:3031")  
    proj_geo = Proj(proj="latlong", datum="WGS84")
    lon, lat = transform(proj_polar, proj_geo, xs, ys)

    # Convert time to seconds since 2000-01-01T00:00:00
    tide_dates = np.datetime64(center_time)
    delta_time = (tide_dates - np.datetime64("2000-01-01T00:00:00")) / np.timedelta64(1, "s")

    # Compute tidal elevations
    tide_h = tide_elevations(
        lon, lat, delta_time,
        MODEL=model, EPOCH=(2000, 1, 1, 0, 0, 0),
        EPSG=3031, TYPE="grid"
    )

    print(f"   📊 Tidal correction applied: min={np.nanmin(tide_h):.2f}, max={np.nanmax(tide_h):.2f}")
    
    return tide_h


def correct_strip(dem_path, output_path, bm_file, mdt_file, center_time, grounded_shapefile, model="CATS2008"):
    """
    Apply geophysical corrections (firn, geoid, tides, and inverse barometer effect) directly to a DEM strip.
    """

    print(f"\n🔄 Starting corrections for: {os.path.basename(dem_path)}")

    # Step 1: Load DEM
    print(f"📂 Loading DEM: {dem_path}")
    dem_data, profile = load_geotiff(dem_path)
    transform = profile["transform"]
    nodata_value = profile["nodata"]

    # Step 2: Generate coordinate grid
    print("📍 Extracting DEM coordinates...")
    rows, cols = np.meshgrid(
        np.arange(dem_data.shape[0]), 
        np.arange(dem_data.shape[1]), 
        indexing="ij"
    )
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()
    dem_data_flat = dem_data.flatten()

    # Step 3: Mask out invalid pixels
    valid_mask = dem_data_flat != nodata_value
    if valid_mask.sum() == 0:
        print(f"❌ No valid data in {dem_path}. Skipping correction.")
        return None

    # Step 4: Load correction datasets
    print("📥 Loading correction datasets...")
    bedmachine_data = load_bedmachine(bm_file)
    mdt_data = load_dtu10_mdt(mdt_file)

    # Step 5: Apply Firn Air Content (FAC) Correction
    print("🧊 Applying Firn Air Content (FAC) correction...")
    try:
        firn_interp = RegularGridInterpolator(
            (bedmachine_data["y"], bedmachine_data["x"]), bedmachine_data["firn"], bounds_error=False, fill_value=0
        )
        firn_correction = firn_interp((ys, xs))
        print(f"   📊 Firn correction applied: min={np.nanmin(firn_correction):.2f}, max={np.nanmax(firn_correction):.2f}")
        dem_data_flat[valid_mask] -= firn_correction[valid_mask]
    except Exception as e:
        print(f"   ❌ Error applying firn correction: {e}")

    # Step 6: Apply Geoid Height Correction
    print("🌍 Applying Geoid Height correction...")
    try:
        geoid_interp = RegularGridInterpolator(
            (bedmachine_data["y"], bedmachine_data["x"]), bedmachine_data["geoid"], bounds_error=False, fill_value=0
        )
        geoid_correction = geoid_interp((ys, xs))
        print(f"   📊 Geoid correction applied: min={np.nanmin(geoid_correction):.2f}, max={np.nanmax(geoid_correction):.2f}")
        dem_data_flat[valid_mask] += geoid_correction[valid_mask]
    except Exception as e:
        print(f"   ❌ Error applying geoid correction: {e}")

    # Step 7: Apply Mean Dynamic Topography (MDT) Correction
    print("🌊 Applying Mean Dynamic Topography (MDT) correction...")
    try:
        mdt_interp = RegularGridInterpolator(
            (mdt_data["lat"], mdt_data["lon"]), mdt_data["mdt"], bounds_error=False, fill_value=0
        )
        mdt_correction = mdt_interp((ys, xs))
        print(f"   📊 MDT correction applied: min={np.nanmin(mdt_correction):.2f}, max={np.nanmax(mdt_correction):.2f}")
        dem_data_flat[valid_mask] += mdt_correction[valid_mask]
    except Exception as e:
        print(f"   ❌ Error applying MDT correction: {e}")

    # Step 8: Apply Tidal Correction
    try:
        tide_correction = apply_tidal_correction(xs, ys, center_time, model)
        dem_data_flat[valid_mask] += tide_correction[valid_mask]
    except Exception as e:
        print(f"   ❌ Failed to apply tidal correction: {e}")

    # Step 9: Fetch and Apply ERA5 Surface Pressure Correction (IBE)
    print("🌎 Fetching ERA5 surface pressure data for IBE correction...")
    try:
        era5_ds = fetch_era5_surface_pressure(center_time, dem_path)
        
        # Convert pressure to hPa
        pressure_values = era5_ds.surface_pressure.values / 100.0
        
        # Interpolate pressure onto DEM grid
        pressure_interp = RegularGridInterpolator(
            (era5_ds.latitude.values, era5_ds.longitude.values), 
            pressure_values, 
            bounds_error=False, 
            fill_value=1013  # Mean sea level pressure
        )
        interpolated_pressure = pressure_interp((ys, xs))
        
        # Compute IBE correction
        ibe_correction = -1.0 * (interpolated_pressure - 1013.0)  # 1 cm per hPa
        print(f"   📊 IBE correction applied: min={np.nanmin(ibe_correction):.2f}, max={np.nanmax(ibe_correction):.2f}")
        
        # Apply IBE correction to DEM
        dem_data_flat[valid_mask] -= ibe_correction[valid_mask]

    except Exception as e:
        print(f"   ❌ Failed to fetch or apply IBE correction: {e}")

    # Step 10: Reshape corrected data and save
    corrected_dem = dem_data_flat.reshape(dem_data.shape)

    print(f"💾 Saving corrected DEM to {output_path}...")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(corrected_dem, 1)

    print(f"✅ Corrections successfully applied and saved to {output_path}\n")
    return output_path

def correct_strips(strip_df, bm_file, mdt_file, strips_dir, output_dir, grounded_shapefile, model="CATS2008"):
    """
    Apply corrections to multiple DEM strips.

    Parameters:
    - strip_df (GeoDataFrame): DataFrame containing strip metadata and file names.
    - bm_file (str): Path to BedMachine dataset (for firn and geoid corrections).
    - mdt_file (str): Path to Mean Dynamic Topography dataset.
    - strips_dir (str): Directory containing aligned DEM strips.
    - output_dir (str): Directory to save corrected DEM strips.
    - grounded_shapefile (str): Path to the grounding line shapefile.
    - model (str): Tidal model to use (default: "CATS2008").
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🔄 Starting correction for {len(strip_df)} strips...")

    for idx, row in strip_df.iterrows():
        file_path = os.path.join(strips_dir, row['dem_id'] + '-trans_reference-DEM.tif')
        corrected_output = os.path.join(output_dir, row['dem_id'] + '_corrected.tif')
        time_format = '%Y-%m-%d %H:%M:%S%z'

        try:
            # Convert numerical timestamps to datetime if needed
            if isinstance(row['pdt_time1'], (int, float)): 
                time1 = pd.to_datetime(row['pdt_time1'], unit='s')  # Convert from Unix timestamp
            else:
                time1 = pd.Timestamp(row['pdt_time1'])

            if isinstance(row['pdt_time2'], (int, float)): 
                time2 = pd.to_datetime(row['pdt_time2'], unit='s')
            else:
                time2 = pd.Timestamp(row['pdt_time2'])

            # Compute center time and format it correctly
            center_time = (time1 + (time2 - time1) / 2).strftime('%Y-%m-%dT%H:%M')
            print(f"   🕒 Center Time: {center_time}")

        except Exception as e:
            print(f"   ❌ Error computing center time: {e}")
            continue  # Skip this DEM if the timestamp is invalid

        try:
            correct_strip(
                dem_path=file_path,
                output_path=corrected_output,
                bm_file=bm_file,
                mdt_file=mdt_file,
                center_time=center_time,
                grounded_shapefile=grounded_shapefile,
                model=model
            )
            print(f"✅ Successfully processed and saved: {corrected_output}")

        except Exception as e:
            print(f"❌ [ERROR] Failed to process {file_path}: {e}")

    print("✅ All corrections completed.")