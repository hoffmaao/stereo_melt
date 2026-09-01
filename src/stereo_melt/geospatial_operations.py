import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from pyproj import Proj, transform, Transformer
from shapely.geometry import box
from shapely.ops import transform as opstransform


def query_ice_shelf(shapefile_path, ice_shelf_name):
    """
    Query an ice shelf by name from a shapefile.

    Parameters:
        shapefile_path (str): Path to the ice shelf shapefile.
        ice_shelf_name (str): Name of the ice shelf to query.

    Returns:
        shapely.geometry.Polygon: The polygon of the queried ice shelf.
    """
    # Load the shapefile
    print(ice_shelf_name)
    gdf = gpd.read_file(shapefile_path)

    # Check available columns
    print(f"Available columns: {gdf.columns}")

    # Query the ice shelf by name (adjust column name as needed)
    ice_shelf = gdf[gdf["NAME"].str.contains(ice_shelf_name, case=False)]

    if ice_shelf.empty:
        raise ValueError(f"No ice shelf found with name containing: {ice_shelf_name}")

    # Combine polygons if there are multiple matching entries
    ice_shelf_polygon = ice_shelf.unary_union

    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    ice_shelf_polygon_ll = opstransform(transformer.transform, ice_shelf_polygon)
    return ice_shelf_polygon, ice_shelf_polygon_ll


def query_bounds(shapefile):
    gdf = gpd.read_file(shapefile)
    gdf = gdf.unary_union

    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    gdf_ll = opstransform(transformer.transform, gdf)
    return gdf, gdf_ll


def reproject_geometry(geometry, source_crs, target_crs):
    """
    Reproject a Shapely geometry from source CRS to target CRS.

    Parameters:
    - geometry (shapely.geometry): Input geometry (e.g., Polygon).
    - source_crs (str): Source CRS (e.g., "EPSG:3031").
    - target_crs (str): Target CRS (e.g., "EPSG:4326").

    Returns:
    - reprojected_geometry (shapely.geometry): Reprojected geometry.
    """
    # Initialize the transformer
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Reproject the geometry
    reprojected_geometry = transform(transformer.transform, geometry)
    return reprojected_geometry


def get_expanded_bounds(ice_shelf_polygon, buffer_km=5.0):
    """
    Get an expanded bounding box around an ice shelf polygon.

    Parameters:
        ice_shelf_polygon (shapely.geometry.Polygon): Polygon of the ice shelf.
        buffer_km (float): Buffer distance in kilometers to expand the bounding box.

    Returns:
        shapely.geometry.Polygon: Expanded bounding box as a shapely box object.
    """
    # Get the bounds of the polygon
    minx, miny, maxx, maxy = ice_shelf_polygon.bounds

    # Expand the bounds by the buffer (convert km to meters)
    buffer_m = buffer_km * 1000
    expanded_bounds = box(minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)

    return expanded_bounds


def project_to_latlon(x, y):
    """
    Convert Polar Stereographic (EPSG:3031) X, Y coordinates to Latitude and Longitude.

    Parameters:
    - x, y: Arrays of X and Y coordinates in meters.

    Returns:
    - lat, lon: Arrays of Latitude and Longitude.
    """
    epsg3031 = Proj(proj="stere", lat_0=-90, lon_0=0, k=1, x_0=0, y_0=0, datum="WGS84")
    wgs84 = Proj(proj="latlong", datum="WGS84")
    lon, lat = transform(epsg3031, wgs84, x, y)
    lon[lon < 0] += 360  # Ensure longitudes are 0-360
    return lat, lon


def project_to_epsg3031(lat, lon):
    """
    Reproject latitude and longitude to EPSG:3031 (Polar Stereographic).

    Parameters:
    - lat (array-like): Array of latitudes.
    - lon (array-like): Array of longitudes.

    Returns:
    - x, y: Reprojected coordinates in meters.
    """
    # Define the WGS84 and EPSG:3031 projections
    wgs84 = Proj(proj="latlong", datum="WGS84")
    epsg3031 = Proj(proj="stere", lat_0=-90, lon_0=0, k=1, x_0=0, y_0=0, datum="WGS84")

    # Transform the coordinates
    x, y = transform(wgs84, epsg3031, lon, lat)
    return x, y


def cleanup_grounding_lines(grounding_lines, min_area=0.01, valid_classes=None):
    """
    Clean and simplify grounding line geometries.

    Parameters:
    - grounding_lines (GeoDataFrame): Input GeoDataFrame containing grounding line data.
    - min_area (float): Minimum polygon area to keep. Smaller areas are removed.
    - valid_classes (list): Optional list of valid classification labels to retain.

    Returns:
    - GeoDataFrame: Cleaned GeoDataFrame with valid geometries.
    """
    # Ensure geometries are valid
    grounding_lines = grounding_lines[grounding_lines.geometry.notnull()]
    grounding_lines["geometry"] = grounding_lines.geometry.buffer(0)  # Fix invalid geometries

    # Filter by valid classes if provided
    if valid_classes:
        if "class" in grounding_lines.columns:
            grounding_lines = grounding_lines[grounding_lines["class"].isin(valid_classes)]
        else:
            print("Warning: 'class' column not found in data.")

    # Remove small polygons
    grounding_lines = grounding_lines[
        grounding_lines.geometry.apply(lambda geom: geom.area >= min_area)
    ]

    # Dissolve multi-polygons into single polygons where possible
    grounding_lines["geometry"] = grounding_lines.geometry.apply(
        lambda geom: (
            geom
            if isinstance(geom, Polygon)
            else max(geom.geoms, key=lambda p: p.area) if isinstance(geom, MultiPolygon) else geom
        )
    )

    return grounding_lines


def define_study_area(x_min, x_max, y_min, y_max):
    """
    Create bounding box and polygon for the study area.
    """
    polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    return polygon


def create_grid(x_min, x_max, y_min, y_max, resolution):
    """
    Create a grid based on the study area boundaries and resolution.
    """
    x = np.arange(x_min, x_max + resolution, resolution)
    y = np.arange(y_max, y_min - resolution, -resolution)  # Reversed to match MATLAB order
    return x, y


def cleanup_grounding_lines(grounding_lines, min_area=0.01, valid_classes=None):
    """
    Clean and simplify grounding line geometries.

    Parameters:
    - grounding_lines (GeoDataFrame): Input GeoDataFrame containing grounding line data.
    - min_area (float): Minimum polygon area to keep. Smaller areas are removed.
    - valid_classes (list): Optional list of valid classification labels to retain.

    Returns:
    - GeoDataFrame: Cleaned GeoDataFrame with valid geometries.
    """
    # Ensure geometries are valid
    grounding_lines = grounding_lines[grounding_lines.geometry.notnull()]
    grounding_lines["geometry"] = grounding_lines.geometry.buffer(0)  # Fix invalid geometries

    # Filter by valid classes if provided
    if valid_classes:
        if "class" in grounding_lines.columns:
            grounding_lines = grounding_lines[grounding_lines["class"].isin(valid_classes)]
        else:
            print("Warning: 'class' column not found in data.")

    # Remove small polygons
    grounding_lines = grounding_lines[
        grounding_lines.geometry.apply(lambda geom: geom.area >= min_area)
    ]

    # Dissolve multi-polygons into single polygons where possible
    grounding_lines["geometry"] = grounding_lines.geometry.apply(
        lambda geom: (
            geom
            if isinstance(geom, Polygon)
            else max(geom.geoms, key=lambda p: p.area) if isinstance(geom, MultiPolygon) else geom
        )
    )

    return grounding_lines
