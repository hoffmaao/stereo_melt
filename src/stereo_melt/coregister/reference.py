# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Coregistration reference datasets.

Builds the control surfaces that feed ASP ``pc_align``:

- ICESat-2 ATL06 elevations over slow-velocity grounded ice (dynamic
  control) via the sliderule service.
- Rock-outcrop elevations extracted from REMA mosaic DEMs using a rock
  polygon shapefile (static control).

The resulting CSVs are combined by
:func:`stereo_melt.coregister.asp.align_strip_with_asp`.
"""

import os
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import sliderule
import xarray as xr
from pyproj import Transformer
from rasterio.mask import mask
from scipy.ndimage import gaussian_filter
from sliderule import icesat2


def _filter_is2_by_sampled_fields(
    is2_gdf,
    velocity_path,
    bedmachine_path,
    max_speed_myr=10.0,
    vel_smooth_sigma_m=2000.0,
    grounded_mask_value=2,
    dem_center_date=None,
    displacement_budget_m=None,
):
    r"""Filter IS2 control points by sampled velocity + BedMachine mask.

    Two filtering modes:

    - **Velocity-threshold (legacy)**: keep where
      ``smoothed |v| < max_speed_myr`` AND ``mask == grounded_mask_value``.
      Set ``displacement_budget_m=None`` (default) to use this.
    - **Shean 2019 displacement budget** (preferred for fast-flow basins):
      keep where ``smoothed |v| × |t_photon − t_DEM| ≤ displacement_budget_m``
      AND ``mask == grounded_mask_value``. The advection error per
      photon is bounded directly rather than indirectly via a fixed
      velocity gate, so slow-grounded points get a wide effective time
      window (limited by the outer ``time_window`` in the caller) and
      fast-flowing points get a tight one. Activated by passing both
      ``dem_center_date`` and ``displacement_budget_m``. Matches Shean
      et al. 2019 (TC 13:2633), Sect. 2.2.1: "We removed points with
      ... maximum expected displacement ... threshold of 10 m."

    Operates on a GeoDataFrame in EPSG:3031 with ``easting``/``northing``
    columns. Crops the velocity and mask rasters to a small envelope
    around the IS2 points to keep memory bounded; smooths speed with a
    Gaussian at ``vel_smooth_sigma_m`` before sampling so single-pixel
    velocity-product speckle does not push points across the threshold.

    Parameters
    ----------
    is2_gdf : geopandas.GeoDataFrame, EPSG:3031
        Sliderule output with ``easting`` and ``northing`` columns. The
        index is the per-photon UTC timestamp (or a ``time`` column).
    velocity_path : str
        NetCDF containing ``VX``/``VY`` on an EPSG:3031 grid (MEaSUREs
        NSIDC-0754 phase map convention).
    bedmachine_path : str
        BedMachine NetCDF with categorical ``mask`` variable.
    max_speed_myr, vel_smooth_sigma_m, grounded_mask_value : tuning.
    dem_center_date : str, optional
        Strip date ``YYYY-MM-DD``. Required for the displacement-budget
        filter; ignored otherwise.
    displacement_budget_m : float, optional
        Per-photon advection budget in meters. When supplied (and
        ``dem_center_date`` is also given), activates the
        displacement-budget filter and the ``max_speed_myr`` gate is
        treated as a soft outer cap.

    Returns
    -------
    geopandas.GeoDataFrame
        Subset of ``is2_gdf`` passing the active filter.
    """
    if is2_gdf.empty:
        return is2_gdf

    east = is2_gdf["easting"].to_numpy(dtype=np.float64)
    north = is2_gdf["northing"].to_numpy(dtype=np.float64)
    pad = max(5000.0, vel_smooth_sigma_m * 4)
    bbox = (
        float(east.min()) - pad,
        float(east.max()) + pad,
        float(north.min()) - pad,
        float(north.max()) + pad,
    )

    def _crop(ds, var):
        # MEaSUREs/BedMachine: x ascends, y descends. Both must be
        # supported defensively in case of upstream re-encoding.
        if ds["y"].values[0] > ds["y"].values[-1]:
            sub = ds[var].sel(
                x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2])
            )
        else:
            sub = ds[var].sel(
                x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3])
            )
        return sub.load()

    def _sample(arr, x_coords, y_coords, east, north):
        ix = np.searchsorted(x_coords, east).clip(0, len(x_coords) - 1)
        if y_coords[0] > y_coords[-1]:
            iy = (len(y_coords) - 1) - np.searchsorted(
                y_coords[::-1], north
            ).clip(0, len(y_coords) - 1)
        else:
            iy = np.searchsorted(y_coords, north).clip(0, len(y_coords) - 1)
        return arr[iy, ix]

    print("🔍 Sampling smoothed velocity at IS2 points...")
    vel_ds = xr.open_dataset(velocity_path)
    vx_sub = _crop(vel_ds, "VX")
    vy_sub = _crop(vel_ds, "VY")
    vx_arr = np.where(np.isfinite(vx_sub.values), vx_sub.values, 0.0)
    vy_arr = np.where(np.isfinite(vy_sub.values), vy_sub.values, 0.0)
    speed = np.hypot(vx_arr, vy_arr)
    dx = abs(float(vx_sub["x"].values[1] - vx_sub["x"].values[0]))
    dy = abs(float(vx_sub["y"].values[1] - vx_sub["y"].values[0]))
    sigma_pix = (vel_smooth_sigma_m / dy, vel_smooth_sigma_m / dx)
    speed_smooth = gaussian_filter(speed, sigma=sigma_pix, mode="nearest")
    point_speed = _sample(
        speed_smooth, vx_sub["x"].values, vx_sub["y"].values, east, north
    )

    print("🔍 Sampling BedMachine mask at IS2 points...")
    bm_ds = xr.open_dataset(bedmachine_path)
    bm_sub = _crop(bm_ds, "mask")
    point_mask = _sample(
        bm_sub.values, bm_sub["x"].values, bm_sub["y"].values, east, north
    )

    grounded = point_mask == grounded_mask_value
    n_grounded = int(grounded.sum())
    n_total = len(is2_gdf)
    print(f"  IS2 over grounded ice: {n_grounded}/{n_total} photons")
    use_shean = displacement_budget_m is not None and dem_center_date is not None
    if use_shean:
        # Per-photon advection error |v| × |Δt| ≤ budget.
        # Pull the timestamp: sliderule's atl06p indexes by UTC time, but
        # downstream code may have already done reset_index() — handle both.
        if "time" in is2_gdf.columns:
            t_photon = pd.to_datetime(is2_gdf["time"], utc=True).to_numpy()
        else:
            t_photon = pd.to_datetime(is2_gdf.index, utc=True).to_numpy()
        t_dem = np.datetime64(pd.Timestamp(dem_center_date, tz="UTC").tz_convert(None))
        dt_yr = np.abs((t_photon.astype("datetime64[ns]") - t_dem)
                       .astype("timedelta64[s]").astype(np.float64)) / (86400.0 * 365.25)
        adv = point_speed * dt_yr  # meters
        keep = grounded & (adv <= displacement_budget_m) & (point_speed < max_speed_myr)
        n_in = len(is2_gdf)
        n_kept = int(keep.sum())
        median_dt = float(np.median(dt_yr[keep])) * 365.25 if n_kept else float("nan")
        median_v = float(np.median(point_speed[keep])) if n_kept else float("nan")
        print(
            f"  IS2 Shean filter: {n_kept}/{n_in} points retained "
            f"(|v|·|Δt| ≤ {displacement_budget_m:g} m ∩ grounded; "
            f"kept median Δt={median_dt:.0f} d, |v|={median_v:.1f} m/yr)"
        )
    else:
        keep = (point_speed < max_speed_myr) & grounded
        n_in = len(is2_gdf)
        n_kept = int(keep.sum())
        print(
            f"  IS2 sample-based filter: {n_kept}/{n_in} points retained "
            f"(smoothed |v| < {max_speed_myr} m/yr ∩ grounded)"
        )
    return is2_gdf.loc[keep].copy()


def extract_rock_elevations_from_mosaics(strip_boundary, rock_shapefile, mosaic_dir, output_csv):
    r"""Extract rock-outcrop elevations from REMA mosaic tiles.

    Rock polygons are intersected with the strip footprint, and every
    DEM pixel under the intersection is emitted as an ``(x, y, z,
    tile)`` row in ``output_csv``.

    Parameters
    ----------
    strip_boundary : shapely.geometry.Polygon
        DEM strip footprint in EPSG:3031.
    rock_shapefile : str
        Path to the exposed-rock polygon shapefile.
    mosaic_dir : str
        Directory of REMA mosaic GeoTIFFs.
    output_csv : str
        Destination CSV path.
    """
    if not os.path.exists(mosaic_dir):
        raise FileNotFoundError(f"❌ Mosaic directory not found: {mosaic_dir}")

    if not os.path.exists(rock_shapefile):
        raise FileNotFoundError(f"❌ Rock shapefile not found: {rock_shapefile}")

    print("🔍 Loading rock shapefile...")
    rock_gdf = gpd.read_file(rock_shapefile)
    if rock_gdf.empty:
        raise ValueError("❌ Rock shapefile is empty.")

    if rock_gdf.crs != "EPSG:3031":
        print("🌍 Reprojecting rock shapefile to EPSG:3031...")
        rock_gdf = rock_gdf.to_crs("EPSG:3031")

    strip_boundary_gdf = gpd.GeoDataFrame(geometry=[strip_boundary], crs="EPSG:3031")

    print("🔄 Calculating intersection of rock polygons with strip boundary...")
    intersected_rock_gdf = gpd.overlay(rock_gdf, strip_boundary_gdf, how="intersection")

    if intersected_rock_gdf.empty:
        print("❌ No overlap between strip boundary and rock polygons. Skipping extraction.")
        return

    intersection_output_path = os.path.join(
        os.path.dirname(output_csv), "rock_strip_intersection.shp"
    )
    intersected_rock_gdf.to_file(intersection_output_path)
    print(f"✅ Intersection shapefile saved to {intersection_output_path}")

    elevation_points = []

    print("🔄 Processing DEM mosaic tiles for rock elevation extraction...")
    for tile_file in os.listdir(mosaic_dir):
        if tile_file.endswith(".tif"):
            tile_path = os.path.join(mosaic_dir, tile_file)
            print(f"📊 Processing tile: {tile_file}")

            try:
                with rasterio.open(tile_path) as src:
                    out_image, out_transform = mask(src, intersected_rock_gdf.geometry, crop=True)
                    out_image = out_image[0]

                    valid_mask = ~np.isnan(out_image)
                    rows, cols = np.where(valid_mask)
                    xs, ys = rasterio.transform.xy(out_transform, rows, cols)
                    elevations = out_image[valid_mask]

                    for x, y, z in zip(xs, ys, elevations):
                        elevation_points.append({"x": x, "y": y, "elevation": z, "tile": tile_file})
            except Exception as e:
                print(f"❌ Failed to process tile {tile_file}: {e}")

    if elevation_points:
        elevation_df = pd.DataFrame(elevation_points)
        elevation_df.to_csv(output_csv, index=False)
        print(f"✅ Rock elevation points saved to {output_csv}")
    else:
        print("❌ No valid rock elevation points found across all tiles.")


def extract_rock_elevations(dem_path, rock_shapefile, output_csv):
    r"""Extract rock elevations from a single DEM using a rock shapefile.

    Parameters
    ----------
    dem_path : str
        Path to a DEM mosaic covering the rock polygons.
    rock_shapefile : str
        Path to the rock polygon shapefile.
    output_csv : str
        Destination CSV path with ``easting``, ``northing``, ``h_mean``
        columns.
    """
    print("🔍 Extracting rock elevations from DEM...")
    rock_gdf = gpd.read_file(rock_shapefile)

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform

        elevation_data = []
        for _, row in rock_gdf.iterrows():
            mask_ = rasterio.features.geometry_mask(
                [row.geometry], transform=transform, out_shape=src.shape, invert=True
            )

            rock_values = dem[mask_]
            rows, cols = np.where(mask_)
            valid_mask = ~np.isnan(rock_values)
            rock_values = rock_values[valid_mask]
            rows, cols = rows[valid_mask], cols[valid_mask]

            xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")

            elevation_data.extend(
                [{"easting": x, "northing": y, "h_mean": z} for x, y, z in zip(xs, ys, rock_values)]
            )

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
    time_window=5,
):
    r"""Download ICESat-2 ATL06 in slow-velocity regions within a time window.

    Queries sliderule's ``icesat2.atl06p`` over the intersection of the
    strip footprint and the slow-velocity polygon, reprojects to
    EPSG:3031, and writes the result to a CSV.

    Parameters
    ----------
    strip_boundary : shapely.geometry.Polygon
        Strip footprint in EPSG:3031.
    slow_velocity_shapefile : str
        Path to the slow-velocity polygon shapefile.
    dem_center_date : str
        Strip center date, ``YYYY-MM-DD``.
    output_dir : str
        Destination directory for the CSV.
    time_window : int
        Half-width of the temporal filter in days.

    Returns
    -------
    str or None
        Path to the CSV, or ``None`` if no data was returned.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("🔍 Loading slow-velocity shapefile...")
    roi_gdf = gpd.read_file(slow_velocity_shapefile)
    if roi_gdf.empty:
        raise ValueError("❌ Slow-velocity shapefile is empty.")

    if roi_gdf.crs != "EPSG:4326":
        print("🌍 Reprojecting shapefile to EPSG:4326...")
        roi_gdf = roi_gdf.to_crs("EPSG:4326")
    strip_boundary_4326 = gpd.GeoDataFrame(geometry=[strip_boundary], crs="EPSG:3031").to_crs(
        "EPSG:4326"
    )

    intersection_gdf = gpd.overlay(roi_gdf, strip_boundary_4326, how="intersection")
    intersection_gdf.to_file("./low_velocity_strip.shp")

    print("🔄 Generating region polygon for Sliderule query...")
    combined_region = intersection_gdf.geometry.unary_union
    if combined_region.is_empty:
        print("❌ Combined region is empty after intersection. Skipping download.")
        return None

    dem_date = datetime.strptime(dem_center_date, "%Y-%m-%d")
    start_date = (dem_date - timedelta(days=time_window)).strftime("%Y-%m-%d")
    end_date = (dem_date + timedelta(days=time_window)).strftime("%Y-%m-%d")

    print(f"📅 Downloading IceSat-2 data from {start_date} to {end_date}...")

    sliderule.earthdata.set_max_resources(1000)
    region = sliderule.toregion(intersection_gdf)

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

        print("🌍 Projecting IceSat-2 data to EPSG:3031...")
        print(results.columns)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        results["easting"], results["northing"] = transformer.transform(
            results.geometry.x, results.geometry.y
        )
        results = results[["easting", "northing", "h_mean"]]

        output_file = os.path.join(output_dir, f"icesat2_{dem_center_date}.csv")
        results.to_csv(output_file, index=False)
        print(f"✅ IceSat-2 data saved to {output_file}")

        return output_file

    except Exception as e:
        print(f"❌ Failed to download IceSat-2 data: {e}")
        return None


def download_icesat2_data(
    strip_boundary,
    grounded_shapefile,
    dem_center_date,
    output_dir,
    time_window=5,
    velocity_path=None,
    bedmachine_path=None,
    max_speed_myr=10.0,
    vel_smooth_sigma_m=2000.0,
    grounded_mask_value=2,
    slow_velocity_shapefile=None,
    dem_id=None,
    displacement_budget_m=None,
):
    r"""Download and filter ICESat-2 ATL06 control points for a strip.

    Two filtering modes:

    - **Sample-based** (default when ``velocity_path`` and
      ``bedmachine_path`` are supplied): Sliderule queries IS2 over
      ``grounded_shapefile ∩ strip``; returned points are filtered by
      sampling a Gaussian-smoothed MEaSUREs speed and the BedMachine v3
      mask at each point and keeping only those with smoothed
      ``|v| < max_speed_myr`` and ``mask == grounded_mask_value``. No
      slow-velocity shapefile required; works pan-Antarctic.
    - **Legacy polygon-based** (when ``slow_velocity_shapefile`` is
      supplied without raster paths): query and filter both go through
      ``slow_velocity ∩ grounded ∩ strip`` with a fallback to
      ``grounded ∩ strip`` if the slow-velocity polygon doesn't cover.

    Parameters
    ----------
    strip_boundary : shapely.geometry.Polygon
        Strip footprint in EPSG:3031.
    grounded_shapefile : str
        Antarctic-wide grounded-ice / grounding-line polygon shapefile.
        Used to define the Sliderule query region (broad — intersected
        with the strip footprint).
    dem_center_date : str
        Strip center date, ``YYYY-MM-DD``.
    output_dir : str
        Destination directory for the CSV.
    time_window : int
        Half-width of the temporal filter in days.
    velocity_path, bedmachine_path : str, optional
        NetCDF paths for sample-based filtering. If both are supplied,
        the per-point sampled filter is used instead of any polygon
        intersection beyond the query region.
    max_speed_myr, vel_smooth_sigma_m, grounded_mask_value : float / int
        Tuning for the sample-based filter (passed through to
        :func:`_filter_is2_by_sampled_fields`).
    slow_velocity_shapefile : str, optional
        Legacy slow-velocity polygon shapefile. Only consulted if both
        raster paths are absent; preserved for backward compatibility.

    Returns
    -------
    str or None
        Path to the CSV, or ``None`` if nothing passed the filters.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    use_sample_filter = (
        velocity_path is not None and bedmachine_path is not None
    )

    grounded_gdf = gpd.read_file(grounded_shapefile)
    if grounded_gdf.crs != "EPSG:4326":
        print("🌍 Reprojecting grounded-ice shapefile to EPSG:4326...")
        grounded_gdf = grounded_gdf.to_crs("EPSG:4326")

    strip_boundary_4326 = gpd.GeoDataFrame(
        geometry=[strip_boundary], crs="EPSG:3031"
    ).to_crs("EPSG:4326")

    if use_sample_filter:
        # Wide query region: just grounded ∩ strip. Per-point filter
        # narrows to slow-flowing grounded ice after Sliderule returns.
        intersection_gdf = gpd.overlay(
            grounded_gdf, strip_boundary_4326, how="intersection"
        )
        if intersection_gdf.empty:
            print(
                "❌ Strip does not intersect grounded ice; "
                "no IS2 control region. Skipping."
            )
            return None
        if displacement_budget_m is not None:
            print(
                f"✅ Query region: grounded ∩ strip. Shean filter will keep "
                f"points with |v|·|Δt| ≤ {displacement_budget_m:g} m ∩ grounded "
                f"(within ±{time_window}d outer cap)."
            )
        else:
            print(
                "✅ Query region: grounded ∩ strip. Sample-based filter will "
                f"keep points with smoothed |v| < {max_speed_myr} m/yr ∩ grounded."
            )
    else:
        if slow_velocity_shapefile is None:
            raise ValueError(
                "Either (velocity_path + bedmachine_path) for sample-based "
                "filtering, or slow_velocity_shapefile for legacy polygon "
                "filtering, must be supplied."
            )
        print("🔍 Loading slow-velocity shapefile (legacy mode)...")
        roi_gdf = gpd.read_file(slow_velocity_shapefile)
        if roi_gdf.empty:
            raise ValueError("❌ Slow-velocity shapefile is empty.")
        if roi_gdf.crs != "EPSG:4326":
            print("🌍 Reprojecting slow-velocity shapefile to EPSG:4326...")
            roi_gdf = roi_gdf.to_crs("EPSG:4326")
        intersection_roi_strip_gdf = gpd.overlay(
            roi_gdf, strip_boundary_4326, how="intersection"
        )
        intersection_gdf = gpd.overlay(
            grounded_gdf, intersection_roi_strip_gdf, how="intersection"
        )
        if intersection_gdf.empty:
            print(
                "⚠️ Strip does not intersect slow-velocity polygons; "
                "falling back to grounded-ice control only."
            )
            intersection_gdf = gpd.overlay(
                grounded_gdf, strip_boundary_4326, how="intersection"
            )
            if intersection_gdf.empty:
                print(
                    "❌ Strip does not intersect grounded ice either. "
                    "Skipping download."
                )
                return None
        print("✅ Intersection region created. Generating region polygon for query...")

    dem_date = datetime.strptime(dem_center_date, "%Y-%m-%d")
    start_date = (dem_date - timedelta(days=time_window)).strftime("%Y-%m-%d")
    end_date = (dem_date + timedelta(days=time_window)).strftime("%Y-%m-%d")

    print(f"📅 Querying IceSat-2 data from {start_date} to {end_date}...")

    try:
        # Sliderule's server-side CMR enumeration defaults to 300 granules;
        # for long strips with overlapping orbits over a 10-day window this
        # gets exceeded and atl06p returns 500. Workaround: enumerate ATL03
        # granules client-side via earthdata.cmr (which honors
        # set_max_resources via the request body) and pass the explicit
        # list to atl06p so the server skips its own CMR query.
        from sliderule import earthdata
        earthdata.set_max_resources(2000)

        poly = sliderule.toregion(intersection_gdf)["poly"]
        parms = {
            "poly": poly,
            "time": [start_date, end_date],
            "srt": icesat2.SRT_LAND,
            "cnf": icesat2.CNF_SURFACE_HIGH,
            "ats": 7.0,
            "cnt": 10,
            "len": 40.0,
            "res": 20.0,
        }
        print("🔄 Enumerating ATL03 granules via CMR...")
        granules = earthdata.cmr(
            short_name="ATL03",
            polygon=poly,
            time_start=f"{start_date}T00:00:00Z",
            time_end=f"{end_date}T23:59:59Z",
        )
        print(f"  found {len(granules)} ATL03 granules")
        if not granules:
            print("❌ No ATL03 granules in time/region.")
            return None
        print("🔄 Querying Sliderule for IceSat-2 data...")
        results = icesat2.atl06p(parms, resources=granules)
        results = results.to_crs(epsg="4326+4979")

        if results.empty:
            print("❌ No IceSat-2 data found in the given region and time window.")
            return None

        print("🌍 Projecting IceSat-2 data to EPSG:3031...")
        results["easting"], results["northing"] = Transformer.from_crs(
            "EPSG:4326", "EPSG:3031", always_xy=True
        ).transform(results.geometry.x, results.geometry.y)

        results_gdf = gpd.GeoDataFrame(
            results,
            geometry=gpd.points_from_xy(results["easting"], results["northing"]),
            crs="EPSG:3031",
        )
        if use_sample_filter:
            filtered_results = _filter_is2_by_sampled_fields(
                results_gdf,
                velocity_path=velocity_path,
                bedmachine_path=bedmachine_path,
                max_speed_myr=max_speed_myr,
                vel_smooth_sigma_m=vel_smooth_sigma_m,
                grounded_mask_value=grounded_mask_value,
                dem_center_date=dem_center_date,
                displacement_budget_m=displacement_budget_m,
            )
        else:
            filtered_results = gpd.sjoin(
                results_gdf,
                intersection_gdf.to_crs("EPSG:3031"),
                how="inner",
            )

        if filtered_results.empty:
            print("❌ Filtered IceSat-2 data is empty after sampled filter.")
            return None

        # Per-strip H5 name to avoid races in parallel runs (two strips on the
        # same date would otherwise collide). Compressed HDF5 replaces the
        # legacy CSV format -- ~7x smaller and decoded as binary float64
        # rather than parsed as text. See stereo_melt.io.altimetry
        # write_control_h5 / read_control_h5 for the schema.
        from ..io.altimetry import write_control_h5

        suffix = dem_id if dem_id is not None else dem_center_date
        output_file = os.path.join(output_dir, f"icesat2_filtered_{suffix}.h5")
        # atl06p indexes by per-photon UTC time; expose it as a `time`
        # column so drift-mode tide / IBE corrections in the orchestrator
        # have something to evaluate against.
        out_df = filtered_results.reset_index()
        time_col = "time" if "time" in out_df.columns else out_df.columns[0]
        out_df = out_df.rename(columns={time_col: "time"})
        write_control_h5(
            out_df[["time", "easting", "northing", "h_mean"]], output_file
        )
        print(f"✅ Filtered IceSat-2 data saved to {output_file}")

        return output_file

    except Exception as e:
        print(f"❌ Failed to download IceSat-2 data: {e}")
        return None
