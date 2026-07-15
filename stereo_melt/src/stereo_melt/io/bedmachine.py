# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""BedMachine Antarctica loader.

Exposes the bed, thickness, surface, geoid, firn air content, and ice
mask fields from the BedMachine HDF5 product as a flat dict of numpy
arrays keyed by variable name.
"""

import h5py
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# BedMachine v3 mask integer codes.
MASK_OCEAN = 0
MASK_ICE_FREE_LAND = 1
MASK_GROUNDED_ICE = 2
MASK_FLOATING_ICE = 3
MASK_LAKE_VOSTOK = 4


def load_bedmachine(file_path):
    r"""Return BedMachine Antarctica fields as a dict of numpy arrays.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the BedMachine HDF5 file.

    Returns
    -------
    dict
        Arrays keyed by variable name: ``x``, ``y``, ``geoid``, ``bed``,
        ``thickness``, ``surface``, ``source``, ``errbed``, ``mask``,
        ``firn``.
    """
    with h5py.File(file_path, "r") as f:
        return {
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


def load_firn_on_grid(
    template: xr.DataArray,
    bedmachine_path,
    *,
    method: str = "linear",
) -> xr.DataArray:
    r"""Return BedMachine firn air content (FAC) interpolated onto ``template``.

    Bilinear interpolation of the BedMachine ``firn`` field (depth in
    meters of the column-integrated air voids inside the firn layer)
    from its native ~500 m grid onto the ``(y, x)`` coords of
    ``template``. Out-of-extent cells get NaN, which propagates through
    :func:`stereo_melt.freeboard.freeboard_to_thickness` and the
    downstream solvers — the right behavior, since we cannot constrain
    the ice-equivalent thickness without firn data.

    The returned field is the canonical input for the ``d=`` argument
    of every solver in :mod:`stereo_melt.melt` and
    :mod:`stereo_melt.dynamics`.

    Parameters
    ----------
    template : xarray.DataArray
        Any field on the basin's analysis grid (e.g. the mean stack).
        Only its ``x`` and ``y`` coords are used.
    bedmachine_path : str or pathlib.Path
        BedMachine HDF5/NetCDF file (the project's
        ``BEDMACHINE_NC``).
    method : {"linear", "nearest"}
        Interpolation kind. Linear (default) for the smoothly-varying
        firn field; nearest is rarely needed but kept for symmetry
        with :func:`interp_bedmachine_mask_at`.

    Returns
    -------
    xarray.DataArray, dims ``(y, x)``
        Firn air content in meters on the ``template`` grid.
    """
    if method not in ("linear", "nearest"):
        raise ValueError(f"method must be 'linear' or 'nearest', got {method!r}")
    bm = load_bedmachine(str(bedmachine_path))
    bm_x = np.asarray(bm["x"], dtype=np.float64)
    bm_y = np.asarray(bm["y"], dtype=np.float64)
    firn = np.asarray(bm["firn"], dtype=np.float64)
    if bm_y[0] > bm_y[-1]:  # need ascending y for RegularGridInterpolator
        bm_y = bm_y[::-1]
        firn = firn[::-1, :]
    interp = RegularGridInterpolator(
        (bm_y, bm_x), firn,
        method=method, bounds_error=False, fill_value=np.nan,
    )
    xs = np.asarray(template["x"].values, dtype=np.float64)
    ys = np.asarray(template["y"].values, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    firn_grid = interp(np.stack([yy, xx], axis=-1))
    return xr.DataArray(
        firn_grid,
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
        name="firn",
        attrs={
            "units": "m",
            "long_name": "firn air content (FAC)",
            "source": "BedMachine Antarctica",
            "interpolation": method,
        },
    )


def interp_bedmachine_mask_at(
    bedmachine_path,
    xs,
    ys,
    *,
    mask_value: int,
    out_of_range: bool = False,
) -> np.ndarray:
    r"""Boolean mask matching ``mask == mask_value`` at EPSG:3031 (xs, ys).

    Nearest-neighbour interpolation of the BedMachine v3 ``mask`` field
    onto an arbitrary EPSG:3031 point set. Use with constants like
    :data:`MASK_FLOATING_ICE` or :data:`MASK_GROUNDED_ICE`.

    Parameters
    ----------
    bedmachine_path : str or pathlib.Path
        BedMachine NetCDF (or HDF5) file. Either format is accepted via
        ``xarray.open_dataset``.
    xs, ys : array_like
        EPSG:3031 coordinates (any shape; broadcast together).
    mask_value : int
        BedMachine mask integer to test for equality.
    out_of_range : bool
        Value returned for points falling outside the BedMachine extent
        (default ``False`` -- treat as not matching).

    Returns
    -------
    numpy.ndarray of bool
        ``True`` where the nearest BedMachine cell has ``mask == mask_value``.
    """
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    bm = xr.open_dataset(str(bedmachine_path))
    try:
        mask_arr = bm["mask"].values
        bm_x = bm["x"].values.astype(np.float64)
        bm_y = bm["y"].values.astype(np.float64)
    finally:
        bm.close()

    if bm_y[0] > bm_y[-1]:
        bm_y = bm_y[::-1]
        mask_arr = mask_arr[::-1, :]

    interp = RegularGridInterpolator(
        (bm_y, bm_x),
        mask_arr.astype(np.float64),
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.column_stack([ys_arr.ravel(), xs_arr.ravel()])
    nearest = interp(pts).reshape(xs_arr.shape)
    out = np.where(np.isnan(nearest), out_of_range, nearest == mask_value)
    return out.astype(bool)
