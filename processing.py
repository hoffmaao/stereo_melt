import os
import numpy as np
import rioxarray as rioxr
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from pyproj import Proj, transform, Transformer
import geopandas as gpd
import pandas as pd
from sliderule import sliderule, icesat2
from datetime import datetime, timedelta
from shapely.geometry import Polygon
import config
import rasterio
from rasterio.mask import mask
import rasterio.features
import subprocess
from shutil import which
import asp_binder_utils as asp_utils
from distutils.spawn import find_executable
import glob
import pyTMD



model_list = sorted(pyTMD.io.model.ocean_elevation())

def load_strip(file_path, center_time):
    """
    Load a downloaded DEM strip and return its data along with the boundary polygon.

    Parameters:
    - file_path (str): Path to the DEM strip file.
    - center_time (datetime): Center time for the strip.

    Returns:
    - dem_data (dict): Dictionary containing x, y, z, and strip_date.
    - boundary_polygon (Polygon): Shapely Polygon representing the strip boundary.
    """
    dem = rioxr.open_rasterio(file_path, masked=True)
    dem_data = {
        "x": dem.x.values,
        "y": dem.y.values,
        "z": dem.values[0],  # Assuming single-band DEM
        "strip_date": np.datetime64(dem.attrs.get("acquisition_date", center_time))
    }

    # Generate boundary polygon
    min_x, max_x = dem_data["x"].min(), dem_data["x"].max()
    min_y, max_y = dem_data["y"].min(), dem_data["y"].max()
    boundary_polygon = Polygon([
        (min_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
        (min_x, min_y)  # Close the polygon
    ])

    print(f"Strip date: {dem_data['strip_date']}")
    return dem_data, boundary_polygon

def subset_strip(strip, x_min, x_max, y_min, y_max):
    """Subset the DEM strip based on spatial bounds."""
    mask_x = (strip["x"] >= x_min) & (strip["x"] <= x_max)
    mask_y = (strip["y"] >= y_min) & (strip["y"] <= y_max)

    strip["x"] = strip["x"][mask_x]
    strip["y"] = strip["y"][mask_y]
    strip["z"] = strip["z"][np.ix_(mask_y, mask_x)]

    strip["z"][strip["z"] == -9999] = np.nan  # Set nodata values to NaN

    return strip

def filter_strip(strip, corrections, smooth_kern=5000, std_res=1000, std_kern=5000):
    """Filter the DEM strip to remove outliers and smooth data."""
    strip_res = np.abs(strip["x"][1] - strip["x"][0])
    
    # Original data copy
    strip["z_orig"] = strip["z"].copy()

    X_grid, Y_grid=np.meshgrid(corrections.x,corrections.y)
    X_strip,Y_strip=np.meshgrid(strip["x"], strip["y"])


    # Mosaic interpolation
    z_mos = griddata(
        (X_grid.ravel(), Y_grid.ravel()),
        corrections.z.ravel(),
        (X_strip, Y_strip),
        method="linear",
    )

    if np.nanmean(z_mos) > 0:  # Apply filter if strip is not mostly ocean
        max_z_mos = np.nanmax(z_mos)
        min_z_mos = np.nanmin(z_mos)
        z_resid = np.abs(strip["z_orig"] - z_mos)
        z_resid_mean = np.nanmean(z_resid)

        strip["z"][strip["z"] > (max_z_mos + z_resid_mean)] = np.nan
        strip["z"][strip["z"] < (min_z_mos - z_resid_mean)] = np.nan

    # Remove points below sea level threshold
    strip["z"][strip["z"] < -40] = np.nan

    print("we made it here 1")

    # Smooth and standard deviation filtering
    z_smooth = gaussian_filter(strip["z"], smooth_kern / strip_res)
    z_smooth_resid = np.abs(strip["z"] - z_smooth)

    print("we made it here 2")

    z_std = gaussian_filter(strip["z_orig"], std_kern / std_res)
    print("we made it here 2.5")
    z_std[z_std < 0] = 0
    z_std_interp = griddata(
        (X_strip.ravel(), Y_strip.ravel()), z_std.ravel(), (X_strip, Y_strip), method="linear"
    )

    print("we made it here 3")


    if np.nanmean(strip["z"]) > 0 and np.nanmean(z_std_interp) > 5:
        strip["z"][z_smooth_resid > np.nanmax(z_std_interp)] = np.nan

    return strip


def apply_tidal_correction(strip, tide_dates, model = "CATS2008"):
    """Apply tidal correction to the DEM strip using pyTMD and CATS2008 model."""
    # Specify tidal model

    # Convert polar stereographic to latitude and longitude
    proj_polar = Proj("epsg:3031")  # EPSG:3031 for Antarctic polar stereographic
    proj_geo = Proj(proj="latlong", datum="WGS84")
    lon, lat = transform(proj_polar, proj_geo, strip["x"], strip["y"])

    # Convert tide_dates to datetime array
    delta_time = (tide_dates - np.datetime64("2000-01-01T00:00:00")) / np.timedelta64(1, "s")
    # Predict tides using pyTMD
    tide_h = tide_elevations(
        lon, lat, delta_time,
        MODEL=model, EPOCH=(2000, 1, 0, 0, 0, 0),
        EPSG=3031, TYPE="grid"
    )

    # Apply correction
    strip["z"] = strip["z"] - tide_h

    return strip


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



def process_strip(file_path, corrections, x_min, x_max, y_min, y_max, 
                  mosaic_dir, low_velocity_shapefile, rock_shapefile, grounded_shapefile, output_dir, model="CATS2008",center_time=None):
    """
    Process a single DEM strip with alignment, tidal corrections, and filtering.

    Parameters:
    - file_path: Path to the original DEM strip.
    - corrections: Dictionary of correction grids (x, y, z).
    - x_min, x_max, y_min, y_max: Spatial bounds for subsetting.
    - mosaic_dir: Directory containing DEM mosaic tiles.
    - low_velocity_shapefile: Path to low-velocity shapefile.
    - rock_shapefile: Path to rock shapefile.
    - output_dir: Directory to save processed outputs.
    - model: Tidal model to use (default: CATS2008).
    """
    # Step 1: Load Strip Metadata
    strip, strip_boundary = load_strip(file_path, center_time=center_time)
    center_time = pd.Timestamp(strip["strip_date"])
    dem_center_date = center_time.strftime('%Y-%m-%d')



    # Step 2: Download IceSat-2 Data
    print("🔄 Downloading IceSat-2 elevation data for current strip...")
    icesat2_file = download_icesat2_data(
        strip_boundary=strip_boundary, 
        slow_velocity_shapefile=low_velocity_shapefile,
        grounded_shapefile=grounded_shapefile,
        dem_center_date=dem_center_date,
        output_dir=os.path.join(output_dir, "icesat2_data"),
        time_window=5
    )
    if not icesat2_file:
        raise ValueError("❌ IceSat-2 data download failed. Skipping strip.")

    # Step 3: Extract Rock Elevations
    print("🔄 Extracting rock elevation points from DEM mosaics for current strip...")
    rock_elevation_csv = os.path.join(output_dir, f"rock_elevations_{os.path.basename(file_path).replace('.tif', '')}.csv")
    extract_rock_elevations_from_mosaics(
        strip_boundary=strip_boundary,
        rock_shapefile=rock_shapefile,
        mosaic_dir=mosaic_dir,
        output_csv=rock_elevation_csv
    )
    if not os.path.exists(rock_elevation_csv):
        print("⚠️ Rock elevation extraction failed, proceeding with IceSat-2 data only.")
        rock_elevation_csv = None  # ✅ Allow missing rock data
    # Step 4: Align DEM with ASP
    print("🔄 Aligning DEM strip using ASP with IceSat-2 and rock elevation data...")
    aligned_dem_path = os.path.join(output_dir, f"aligned_{os.path.basename(file_path)}")
    print("🔄 Aligning DEM strip using ASP with IceSat-2 and rock elevation data...")
    aligned_dem_path = align_strip_with_asp(
        file_path=file_path,
        rock_csv=rock_elevation_csv,
        icesat2_csv=icesat2_file,
        output_dir=output_dir,
    )
    if not os.path.exists(aligned_dem_path):
        raise ValueError("❌ ASP alignment failed. Skipping strip.")

    # Step 5: Load the Aligned DEM
    print("🔄 Loading aligned DEM strip for tidal corrections and processing...")
    strip_aligned, strip_aligned_boundary = load_strip(aligned_dem_path, center_time=center_time)
    # Step 6: Subset and Filter Aligned DEM
    #strip_aligned = subset_strip(strip_aligned, x_min, x_max, y_min, y_max)
    #strip_aligned = filter_strip(strip_aligned, corrections)

    # Step 7: Apply Tidal Corrections
    #strip_aligned = apply_tidal_correction(strip_aligned, tide_dates=strip_aligned["strip_date"], model=model)

    print(f"✅ Processing complete for strip {aligned_dem_path}")
    return strip_aligned



def process_strips(strip_df, corrections, x_min, x_max, y_min, y_max, strips_dir, output_dir,
                   mosaic_dir, low_velocity_shapefile, rock_shapefile, grounded_shapefile, model="CATS2008"):
    """
    Process multiple DEM strips from a GeoDataFrame.

    Parameters:
    - strip_df: GeoDataFrame containing strip metadata and file names.
    - corrections: Corrections object for geoid, MDT, and firn data.
    - x_min, x_max, y_min, y_max: Spatial bounds for subsetting.
    - strips_dir: Directory where DEM strips are stored.
    - output_dir: Directory to save processed strips.
    - mosaic_dir: Directory containing DEM mosaic DEM tiles.
    - low_velocity_shapefile: Path to low-velocity shapefile.
    - rock_shapefile: Path to rock shapefile.
    - model: Tidal model to use (default: CATS2008).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(strip_df)


    for idx, row in strip_df.iterrows():
        file_path = os.path.join(strips_dir, row['dem_id'] + '.tif')
        center_time = pd.Timestamp((row['pdt_time1'].value + row['pdt_time2'].value) / 2)
        output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.tif', '.npz'))
        
        print(f"\n[INFO] Processing strip {file_path} ({idx + 1}/{len(strip_df)})")
        
        if os.path.exists(output_file):
            print(f"[SKIPPED] {output_file} already exists. Skipping...")
            continue

        try:
            processed_strip = process_strip(
                file_path=file_path,
                corrections=corrections,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                mosaic_dir=mosaic_dir,
                low_velocity_shapefile=low_velocity_shapefile,
                rock_shapefile=rock_shapefile,
                grounded_shapefile=grounded_shapefile,
                output_dir=output_dir,
                model=model,
                center_time=center_time
            )
            
            np.savez_compressed(output_file, **processed_strip)
            print(f"[SUCCESS] Processed strip saved to {output_file}")
        
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")




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
            mask = rasterio.features.geometry_mask(
                [row.geometry],
                transform=transform,
                out_shape=src.shape,
                invert=True
            )
            
            # Extract DEM values within the mask
            rock_values = dem[mask]
            rows, cols = np.where(mask)  # Get row and column indices of the masked values
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
    # Make ATL06 Request
    gdf = icesat2.atl06p(parms)


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
    Download and filter IceSat-2 elevation data in slow-velocity regions within a time window.

    Parameters:
    - strip_boundary (Polygon): Shapely Polygon representing the strip boundary.
    - slow_velocity_shapefile (str): Path to the shapefile defining slow-velocity regions.
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
    intersection_region = intersection_gdf.geometry.unary_union

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
        results=results.to_crs(epsg="4326+4979")

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





def prepare_asp_inputs(dem_path, rock_csv, icesat2_csv, output_dir):
    """
    Prepare DEM and reference datasets for ASP pc_align.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🔄 Preparing ASP inputs...")
    inputs = {
        "dem": dem_path,
        "rock_ref": rock_csv,
        "icesat2_ref": icesat2_csv,
        "output_dir": output_dir
    }
    return inputs


def align_strip_with_asp(
    file_path,
    rock_csv=None,
    icesat2_csv=None,
    output_dir=None,
    max_displacement=100,
    alignment_method="point-to-plane",
    verbose=True,
    tr = 2
):
    """
    Align a DEM strip using ASP pc_align with combined rock and IceSat-2 elevation data.

    Parameters:
    - file_path (str): Path to the DEM strip in EPSG:3031.
    - rock_csv (str): Path to the CSV file with rock elevation data in EPSG:3031 (optional).
    - icesat2_csv (str): Path to the CSV file with IceSat-2 elevation data in EPSG:3031 (optional).
    - output_dir (str): Directory to save alignment results.
    - max_displacement (float): Maximum allowed displacement for pc_align (default: 100).
    - alignment_method (str): ASP alignment method (default: "point-to-plane").
    - verbose (bool): If True, print detailed ASP command output.

    Returns:
    - aligned_dem_path (str): Path to the aligned DEM strip.
    """

    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Combine rock and IceSat-2 data into a single CSV file
    combined_csv_path = output_dir+"reference_files/combined_reference.csv"
    if rock_csv and os.path.exists(rock_csv) and icesat2_csv and os.path.exists(icesat2_csv):
        print("🔄 Combining rock and IceSat-2 elevation data into a single CSV...")
        try:
            rock_df = pd.read_csv(rock_csv)
            rock_df=rock_df[rock_df['elevation']!=-9999]
            rock_df=rock_df.drop('tile',axis=1)
            icesat2_df = pd.read_csv(icesat2_csv)
            print(rock_df)
            print(icesat2_df)
            combined_df = pd.concat([icesat2_df,rock_df.rename(columns={'elevation':'h_mean'})], ignore_index=True)
            combined_df.to_csv(combined_csv_path, index=False)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to combine CSV files: {e}")
    elif rock_csv and os.path.exists(rock_csv):
        print("🔄 Using only rock elevation data...")
        rock_df=rock_df[rock_df['elevation']!=-9999]
        rock_df.to_csv(combined_csv_path, index=False)
    elif icesat2_csv and os.path.exists(icesat2_csv):
        print("🔄 Using only IceSat-2 elevation data...")
        combined_csv_path = icesat2_csv
    else:
        raise RuntimeError("❌ No valid reference data available for alignment.")

    pc_align = find_executable('pc_align')
    ref_alitmetry = combined_csv_path
    src_dem = file_path
    alignment_dir =output_dir+"asp_aligned/"+file_name_no_ext
    csv_proj4 = '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs' 
    altimetry_datum = 'WGS84'
    csv_format = '1:easting,2:northing,3:height_above_datum'
    alignment_call = f"{pc_align} --highest-accuracy --csv-format '{csv_format}' --csv-srs '{csv_proj4}' --save-inv-transformed-reference-points --alignment-method {alignment_method}  --max-displacement {max_displacement} {src_dem} {ref_alitmetry} -o {alignment_dir}"




    # Run the ASP command
    print(f"🚀 Running ASP pc_align: ")
    asp_utils.run_bash_command(alignment_call,verbose=verbose)
    if os.path.exists(output_dir):
        print(f"✅ Aligned DEM saved at {output_dir}")
    else:
        raise RuntimeError("❌ ASP alignment failed. Check logs for errors.")

    print("we made it here")


    point2dem = find_executable('point2dem')
    csv_format = '1:easting,2:northing,3:height_above_datum'
    tsrs = 'EPSG:3031'
    #p2dem_args = '--errorimage'
    nodata_value = -9999.0
    pointcloud = glob.glob(alignment_dir+'*-trans_reference.tif')[0]
    print ("Gridding pointcloud {} at {} m/px".format(pointcloud,tr))
    point2dem_call  = f"{point2dem} --tr {tr} --t_srs '{tsrs}' --nodata-value {nodata_value} {pointcloud}"
    asp_utils.run_bash_command(point2dem_call,verbose=verbose)
    aligned_dem = glob.glob(alignment_dir+'*-DEM.tif')[0]
    print("DEM saved at {}".format(aligned_dem))

    #Initial DEM and altimetry difference
    geodiff = find_executable('geodiff')
    initial_output_prefix = output_dir+'initial/'+file_name_no_ext+"-initial"
    print(f"Computing elevation difference before alignment between {ref_alitmetry} and {src_dem}\n\n")
    geodiff_call = f"{geodiff} {ref_alitmetry} {src_dem} --csv-format {csv_format} --csv-srs '{csv_proj4}' -o {initial_output_prefix}"
    asp_utils.run_bash_command(geodiff_call,verbose=verbose)
    initial_elevation_difference_fn = glob.glob(initial_output_prefix+'-diff.csv')[0]
    print("\n\nInitial elevation difference saved at {}".format(initial_elevation_difference_fn))

    #Final DEM and altimetry difference
    geodiff = find_executable('geodiff')
    final_output_prefix = output_dir+'final/'+file_name_no_ext+"-final"
    print(f"Computing elevation difference before alignment between {ref_alitmetry} and {aligned_dem}\n\n")
    geodiff_call = f"{geodiff} {ref_alitmetry} {aligned_dem} --csv-format {csv_format} --csv-srs '{csv_proj4}' -o {final_output_prefix}"
    asp_utils.run_bash_command(geodiff_call,verbose=verbose)
    final_elevation_difference_fn = glob.glob(final_output_prefix+'-diff.csv')[0]
    print("\n\nFinal elevation difference saved at {}".format(initial_elevation_difference_fn))

    asp_utils.plot_alignment_maps_altimetry(combined_df,src_dem,initial_elevation_difference_fn,final_elevation_difference_fn,plot_crs=tsrs,output_dir=output_dir)


    return aligned_dem




def correct_strip(dem_path, corrections, output_path, tidal_model="CATS2008"):
    """
    Apply geophysical corrections (firn, geoid, surface pressure, tides) to a single DEM.

    Parameters:
    - dem_path (str): Path to the input aligned DEM.
    - corrections (dict): Dictionary of correction grids for firn, geoid, surface pressure, and tides.
    - output_path (str): Path to save the corrected DEM.
    - tidal_model (str): Tidal model to use (default: "CATS2008").

    Returns:
    - corrected_path (str): Path to the corrected DEM.
    """
    print(f"🔄 Applying corrections to {os.path.basename(dem_path)}...")

    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        profile = src.profile
        transform = src.transform

        # Get coordinates of pixels
        rows, cols = np.meshgrid(np.arange(dem_data.shape[0]), np.arange(dem_data.shape[1]), indexing="ij")
        xs, ys = rasterio.transform.xy(transform, rows, cols)

        # Flatten arrays
        xs = np.array(xs).flatten()
        ys = np.array(ys).flatten()
        dem_data_flat = dem_data.flatten()

        # Mask invalid data
        valid_mask = dem_data_flat != src.nodata

        if valid_mask.sum() == 0:
            print(f"❌ No valid data in {dem_path}. Skipping correction.")
            return None

        # Apply Firn Air Content (FAC) Correction
        if "firn" in corrections:
            firn_interp = RegularGridInterpolator(
                (corrections["firn"]["y"], corrections["firn"]["x"]),
                corrections["firn"]["z"],
                bounds_error=False,
                fill_value=0
            )
            firn_correction = firn_interp((ys, xs))
            dem_data_flat[valid_mask] -= firn_correction[valid_mask]  # Subtract firn correction

        # Apply Geoid Correction
        if "geoid" in corrections:
            geoid_interp = RegularGridInterpolator(
                (corrections["geoid"]["y"], corrections["geoid"]["x"]),
                corrections["geoid"]["z"],
                bounds_error=False,
                fill_value=0
            )
            geoid_correction = geoid_interp((ys, xs))
            dem_data_flat[valid_mask] += geoid_correction[valid_mask]  # Add geoid height

        # Apply Surface Pressure Correction (Optional)
        if "pressure" in corrections:
            pressure_interp = RegularGridInterpolator(
                (corrections["pressure"]["y"], corrections["pressure"]["x"]),
                corrections["pressure"]["z"],
                bounds_error=False,
                fill_value=0
            )
            pressure_correction = pressure_interp((ys, xs))
            dem_data_flat[valid_mask] -= pressure_correction[valid_mask]  # Subtract pressure effect

        # Apply Tidal Correction
        if tidal_model:
            dem_date = os.path.basename(dem_path).split("_")[-2]  # Assuming timestamp in filename
            tide_correction = apply_tidal_correction(xs, ys, dem_date, model=tidal_model)
            dem_data_flat[valid_mask] += tide_correction[valid_mask]  # Add tidal correction

        # Reshape back to original DEM shape
        corrected_dem = dem_data_flat.reshape(dem_data.shape)

        # Save corrected DEM
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(corrected_dem, 1)

    print(f"✅ Corrections applied and saved to {output_path}")
    return output_path


def fetch_era5_surface_pressure(center_time, region):
    """
    Queries the CDS API for surface pressure data at a specific time and region.

    Parameters:
        center_time (str): Center date-time in 'YYYY-MM-DDTHH:MM' format.
        region (list): [North, West, South, East] lat/lon coordinates.

    Returns:
        xarray.Dataset: The ERA5 surface pressure dataset.
    """
    c = cdsapi.Client()

    # Extract date and time
    date, time = center_time.split("T")

    # Request data from CDS API
    result = c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': 'surface_pressure',
            'product_type': 'reanalysis',
            'year': date[:4],
            'month': date[5:7],
            'day': date[8:10],
            'time': time,  # Ensure time is in 'HH:MM' format
            'area': region,  # [North, West, South, East]
            'format': 'netcdf',
        }
    )

    # Stream data into memory instead of downloading
    data = io.BytesIO(result.download(target=None))

    # Open dataset directly from memory
    ds = xr.open_dataset(data)
    
    return ds