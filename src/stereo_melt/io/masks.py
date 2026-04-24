"""Polygon / mask helpers: AOI validation, control-surface (low-velocity + rock) polygons,
grounding-zone loading, and generic mask⇄polygon conversions."""

import os

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import Polygon, shape


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
    """Convert a raster mask to polygons using Rasterio."""
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
    """Save a list of polygons to a shapefile using Fiona."""
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
    """Create polygons for low-velocity and rock regions using Rasterio and Fiona."""
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


def create_polyshape(mask, x, y):
    """
    Generate polygons for a mask.

    Parameters:
    - mask: Binary mask indicating regions of interest.
    - x, y: Coordinates corresponding to the mask.

    Returns:
    - List of polygons representing the mask.
    """
    polygons = []
    for region in mask:
        coords = np.column_stack(np.nonzero(region))
        if coords.shape[0] > 2:  # Only include valid polygons
            polygons.append(Polygon(coords))
    return polygons
