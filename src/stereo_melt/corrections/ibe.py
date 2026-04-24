"""Inverse Barometer Effect (IBE) correction using ERA5 surface pressure from the CDS API."""

import io
from datetime import datetime

import cdsapi
import numpy as np
import rasterio
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator


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

    print("🔄 Fetching ERA5 surface pressure for IBE correction...")
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
