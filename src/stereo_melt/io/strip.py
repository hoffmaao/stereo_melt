# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""DEM-strip helpers: loading, spatial subsetting, outlier/smooth filtering."""

import numpy as np
import rioxarray as rioxr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon


def load_strip(file_path, center_time):
    r"""Load a downloaded DEM strip and return its data and bounding polygon.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the DEM strip GeoTIFF.
    center_time : datetime-like
        Fallback acquisition time used when the file has no
        ``acquisition_date`` attribute.

    Returns
    -------
    tuple
        ``(dem_data, boundary_polygon)`` where ``dem_data`` is a dict
        with ``x``, ``y``, ``z``, and ``strip_date`` entries and
        ``boundary_polygon`` is a :class:`shapely.geometry.Polygon`
        covering the strip extent.
    """
    dem = rioxr.open_rasterio(file_path, masked=True)
    dem_data = {
        "x": dem.x.values,
        "y": dem.y.values,
        "z": dem.values[0],  # Assuming single-band DEM
        "strip_date": np.datetime64(dem.attrs.get("acquisition_date", center_time)),
    }

    # Generate boundary polygon
    min_x, max_x = dem_data["x"].min(), dem_data["x"].max()
    min_y, max_y = dem_data["y"].min(), dem_data["y"].max()
    boundary_polygon = Polygon(
        [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, min_y),  # Close the polygon
        ]
    )

    print(f"Strip date: {dem_data['strip_date']}")
    return dem_data, boundary_polygon


def subset_strip(strip, x_min, x_max, y_min, y_max):
    r"""Return a DEM strip cropped to the given ``(x, y)`` bounds.

    The nodata sentinel ``-9999`` in the subset is replaced with NaN.
    """
    mask_x = (strip["x"] >= x_min) & (strip["x"] <= x_max)
    mask_y = (strip["y"] >= y_min) & (strip["y"] <= y_max)

    strip["x"] = strip["x"][mask_x]
    strip["y"] = strip["y"][mask_y]
    strip["z"] = strip["z"][np.ix_(mask_y, mask_x)]

    strip["z"][strip["z"] == -9999] = np.nan  # Set nodata values to NaN

    return strip


def filter_strip(strip, corrections, smooth_kern=5000, std_res=1000, std_kern=5000):
    r"""Filter a DEM strip against a correction mosaic to drop outliers.

    Points that deviate from the correction mosaic by more than the local
    residual spread, lie below ``-40`` m, or exceed the smoothed standard
    deviation envelope are set to NaN.

    Parameters
    ----------
    strip : dict
        DEM strip dict as returned by :func:`load_strip`. A copy of the
        original ``z`` is stored under ``z_orig``.
    corrections : object
        Correction mosaic with ``x``, ``y``, ``z`` attributes covering the
        strip footprint.
    smooth_kern, std_kern, std_res : float
        Gaussian kernel widths (meters) for the smoothing and
        standard-deviation filters.

    Returns
    -------
    dict
        The input ``strip`` updated in place with filtered ``z``.
    """
    strip_res = np.abs(strip["x"][1] - strip["x"][0])

    # Original data copy
    strip["z_orig"] = strip["z"].copy()

    X_grid, Y_grid = np.meshgrid(corrections.x, corrections.y)
    X_strip, Y_strip = np.meshgrid(strip["x"], strip["y"])

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

    # Smooth and standard deviation filtering
    z_smooth = gaussian_filter(strip["z"], smooth_kern / strip_res)
    z_smooth_resid = np.abs(strip["z"] - z_smooth)

    z_std = gaussian_filter(strip["z_orig"], std_kern / std_res)
    z_std[z_std < 0] = 0
    z_std_interp = griddata(
        (X_strip.ravel(), Y_strip.ravel()), z_std.ravel(), (X_strip, Y_strip), method="linear"
    )

    if np.nanmean(strip["z"]) > 0 and np.nanmean(z_std_interp) > 5:
        strip["z"][z_smooth_resid > np.nanmax(z_std_interp)] = np.nan

    return strip
