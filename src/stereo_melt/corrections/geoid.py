# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Geoid correction.

Converts WGS84-ellipsoid REMA DEMs to orthometric (geoid-referenced)
heights using the BedMachine geoid undulation field. Following
Shean 2019 Eq. 1 and the pdemtools convention,

.. math::
    h_\mathrm{ortho} = h_\mathrm{ellipsoid} - N

where :math:`N` is the geoid undulation above the WGS84 ellipsoid, read
from the BedMachine ``geoid`` variable.

The primary path is :func:`apply_geoid_correction_xr`, backed by
:meth:`pdemtools.DemAccessor.geoid_correct`. A numpy equivalent is
provided for the flat-array correction pipeline.
"""

import xarray as xr
import rioxarray  # noqa: F401 - registers .rio accessor
import pdemtools  # noqa: F401 - registers .pdt accessor

__all__ = [
    "load_bedmachine_geoid",
    "apply_geoid_correction_xr",
    "apply_geoid_correction_np",
]


def load_bedmachine_geoid(bm_path: str) -> xr.DataArray:
    r"""Return the BedMachine geoid field as an EPSG:3031 DataArray."""
    ds = xr.open_dataset(bm_path)
    geoid = ds["geoid"]
    return geoid.rio.write_crs("EPSG:3031", inplace=False)


def apply_geoid_correction_xr(dem: xr.DataArray, bm_path: str) -> xr.DataArray:
    r"""Return an orthometric DEM using the pdemtools geoid pipeline.

    Parameters
    ----------
    dem : xarray.DataArray
        DEM of ellipsoidal heights (REMA is WGS84). Must carry a rio CRS.
    bm_path : str
        Path to the BedMachine NetCDF providing the geoid field.

    Returns
    -------
    xarray.DataArray
        Orthometric (geoid-referenced) DEM on the same grid as ``dem``.
    """
    geoid = load_bedmachine_geoid(bm_path)
    return dem.pdt.geoid_correct(geoid)


def apply_geoid_correction_np(dem_flat, xs, ys, bedmachine_data):
    r"""Return an orthometric DEM for a flat-array pipeline.

    Parameters
    ----------
    dem_flat : numpy.ndarray
        DEM values above the WGS84 ellipsoid.
    xs, ys : numpy.ndarray
        Flat coordinate arrays in EPSG:3031.
    bedmachine_data : dict
        BedMachine dict from
        :func:`stereo_melt.io.bedmachine.load_bedmachine`.

    Returns
    -------
    numpy.ndarray
        Orthometric DEM values.
    """
    from scipy.interpolate import RegularGridInterpolator

    geoid_interp = RegularGridInterpolator(
        (bedmachine_data["y"], bedmachine_data["x"]),
        bedmachine_data["geoid"],
        bounds_error=False,
        fill_value=0,
    )
    geoid_N = geoid_interp((ys, xs))
    return dem_flat - geoid_N
