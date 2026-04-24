"""Per-strip correction orchestrators.

Staging module for correct_strip / correct_strips. These combine firn, geoid,
MDT, tidal, and IBE corrections into a single corrected GeoTIFF. Will likely be
refactored further once the new Shean-style mass-budget stages come in.
"""

import os

import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import RegularGridInterpolator

from .corrections.ibe import fetch_era5_surface_pressure
from .corrections.tides import apply_tidal_correction
from .io.bedmachine import load_bedmachine
from .io.common import load_geotiff
from .io.mdt import load_dtu10_mdt


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
