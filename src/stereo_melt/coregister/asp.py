"""ASP pc_align primary coregistration workflow.

align_strip / align_strips orchestrate, per DEM:
  1. Build IS2 + rock control-surface CSV via coregister.reference
  2. Run ASP pc_align --highest-accuracy --alignment-method point-to-plane
  3. Rasterize the transformed point cloud with point2dem
  4. Compute geodiff residuals before/after and plot QC.
"""

import glob
import os

import pandas as pd
from distutils.spawn import find_executable

from .. import asp_binder_utils as asp_utils
from ..io.strip import load_strip
from .reference import download_icesat2_data, extract_rock_elevations_from_mosaics


def prepare_asp_inputs(dem_path, rock_csv, icesat2_csv, output_dir):
    """Prepare DEM and reference datasets for ASP pc_align."""
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
    tr=2
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
    combined_csv_path = output_dir + "reference_files/combined_reference.csv"
    if rock_csv and os.path.exists(rock_csv) and icesat2_csv and os.path.exists(icesat2_csv):
        print("🔄 Combining rock and IceSat-2 elevation data into a single CSV...")
        try:
            rock_df = pd.read_csv(rock_csv)
            rock_df = rock_df[rock_df['elevation'] != -9999]
            rock_df = rock_df.drop('tile', axis=1)
            icesat2_df = pd.read_csv(icesat2_csv)
            combined_df = pd.concat([icesat2_df, rock_df.rename(columns={'elevation': 'h_mean'})], ignore_index=True)
            combined_df.to_csv(combined_csv_path, index=False)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to combine CSV files: {e}")
    elif rock_csv and os.path.exists(rock_csv):
        print("🔄 Using only rock elevation data...")
        rock_df = pd.read_csv(rock_csv)
        rock_df = rock_df[rock_df['elevation'] != -9999]
        rock_df.to_csv(combined_csv_path, index=False)
    elif icesat2_csv and os.path.exists(icesat2_csv):
        print("🔄 Using only IceSat-2 elevation data...")
        combined_csv_path = icesat2_csv
    else:
        raise RuntimeError("❌ No valid reference data available for alignment.")

    pc_align = find_executable('pc_align')
    ref_alitmetry = combined_csv_path
    src_dem = file_path
    alignment_dir = output_dir + "asp_aligned/" + file_name_no_ext
    csv_proj4 = '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'
    csv_format = '1:easting,2:northing,3:height_above_datum'
    alignment_call = f"{pc_align} --highest-accuracy --csv-format '{csv_format}' --csv-srs '{csv_proj4}' --save-inv-transformed-reference-points --alignment-method {alignment_method}  --max-displacement {max_displacement} {src_dem} {ref_alitmetry} -o {alignment_dir}"

    # Run the ASP command
    print("🚀 Running ASP pc_align: ")
    asp_utils.run_bash_command(alignment_call, verbose=verbose)
    if os.path.exists(output_dir):
        print(f"✅ Aligned DEM saved at {output_dir}")
    else:
        raise RuntimeError("❌ ASP alignment failed. Check logs for errors.")

    point2dem = find_executable('point2dem')
    tsrs = 'EPSG:3031'
    nodata_value = -9999.0
    pointcloud = glob.glob(alignment_dir + '*-trans_reference.tif')[0]
    print(f"Gridding pointcloud {pointcloud} at {tr} m/px")
    point2dem_call = f"{point2dem} --tr {tr} --t_srs '{tsrs}' --nodata-value {nodata_value} {pointcloud}"
    asp_utils.run_bash_command(point2dem_call, verbose=verbose)
    aligned_dem = glob.glob(alignment_dir + '*-DEM.tif')[0]
    print(f"DEM saved at {aligned_dem}")

    # Initial DEM and altimetry difference
    geodiff = find_executable('geodiff')
    initial_output_prefix = output_dir + 'initial/' + file_name_no_ext + "-initial"
    print(f"Computing elevation difference before alignment between {ref_alitmetry} and {src_dem}\n\n")
    geodiff_call = f"{geodiff} {ref_alitmetry} {src_dem} --csv-format {csv_format} --csv-srs '{csv_proj4}' -o {initial_output_prefix}"
    asp_utils.run_bash_command(geodiff_call, verbose=verbose)
    initial_elevation_difference_fn = glob.glob(initial_output_prefix + '-diff.csv')[0]
    print(f"\n\nInitial elevation difference saved at {initial_elevation_difference_fn}")

    # Final DEM and altimetry difference
    final_output_prefix = output_dir + 'final/' + file_name_no_ext + "-final"
    print(f"Computing elevation difference after alignment between {ref_alitmetry} and {aligned_dem}\n\n")
    geodiff_call = f"{geodiff} {ref_alitmetry} {aligned_dem} --csv-format {csv_format} --csv-srs '{csv_proj4}' -o {final_output_prefix}"
    asp_utils.run_bash_command(geodiff_call, verbose=verbose)
    final_elevation_difference_fn = glob.glob(final_output_prefix + '-diff.csv')[0]
    print(f"\n\nFinal elevation difference saved at {final_elevation_difference_fn}")

    asp_utils.plot_alignment_maps_altimetry(
        combined_df, src_dem, initial_elevation_difference_fn, final_elevation_difference_fn,
        plot_crs=tsrs, output_dir=output_dir
    )

    return aligned_dem


def align_strip(
    file_path, corrections, x_min, x_max, y_min, y_max,
    mosaic_dir, low_velocity_shapefile, rock_shapefile, grounded_shapefile,
    output_dir, model="CATS2008", center_time=None
):
    """
    Coregister a single DEM strip: build IS2+rock reference, run pc_align, return the aligned strip.

    Parameters:
    - file_path: Path to the original DEM strip.
    - corrections: Corrections field (unused in current ASP alignment, reserved for downstream filtering).
    - x_min, x_max, y_min, y_max: Spatial bounds (reserved for downstream subsetting).
    - mosaic_dir: Directory containing DEM mosaic tiles.
    - low_velocity_shapefile: Path to low-velocity shapefile.
    - rock_shapefile: Path to rock shapefile.
    - grounded_shapefile: Path to grounded-ice shapefile.
    - output_dir: Directory to save alignment outputs.
    - model: Tidal model to use (reserved; corrections are applied downstream in pipeline.correct_strip).
    - center_time: Center datetime for the strip.
    """
    # Step 1: Load strip and get boundary
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
        rock_elevation_csv = None

    # Step 4: Align DEM with ASP
    print("🔄 Aligning DEM strip using ASP with IceSat-2 and rock elevation data...")
    aligned_dem_path = align_strip_with_asp(
        file_path=file_path,
        rock_csv=rock_elevation_csv,
        icesat2_csv=icesat2_file,
        output_dir=output_dir,
    )
    if not os.path.exists(aligned_dem_path):
        raise ValueError("❌ ASP alignment failed. Skipping strip.")

    # Step 5: Load the Aligned DEM for return
    print("🔄 Loading aligned DEM strip...")
    strip_aligned, strip_aligned_boundary = load_strip(aligned_dem_path, center_time=center_time)

    print(f"✅ Coregistration complete for strip {aligned_dem_path}")
    return strip_aligned


def align_strips(
    strip_df, corrections, x_min, x_max, y_min, y_max, strips_dir, output_dir,
    mosaic_dir, low_velocity_shapefile, rock_shapefile, grounded_shapefile, model="CATS2008"
):
    """
    Coregister multiple DEM strips from a GeoDataFrame.

    Parameters:
    - strip_df: GeoDataFrame from pdt.search containing strip metadata and IDs.
    - corrections: Corrections object (reserved for downstream filtering).
    - x_min, x_max, y_min, y_max: Spatial bounds (reserved for downstream).
    - strips_dir: Directory where DEM strips are stored.
    - output_dir: Directory to save ASP outputs.
    - mosaic_dir: Directory containing DEM mosaic tiles.
    - low_velocity_shapefile, rock_shapefile, grounded_shapefile: Control-surface inputs.
    - model: Tidal model (reserved; downstream).
    """
    import numpy as np

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, row in strip_df.iterrows():
        file_path = os.path.join(strips_dir, row['dem_id'] + '.tif')
        center_time = pd.Timestamp((row['pdt_time1'].value + row['pdt_time2'].value) / 2)
        output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.tif', '.npz'))

        print(f"\n[INFO] Processing strip {file_path} ({idx + 1}/{len(strip_df)})")

        if os.path.exists(output_file):
            print(f"[SKIPPED] {output_file} already exists. Skipping...")
            continue

        try:
            processed_strip = align_strip(
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
