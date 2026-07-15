# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Velocity ingestion.

Provides interpolation of MEaSUREs annual velocity products onto a
target grid, gap-filling of missing samples, and construction of a
time-stacked velocity field with per-pixel statistics, gradients, and
divergence suitable for the mass-budget stage.
"""

import os

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter

from .masks import create_polyshape


def load_nsidc_0525(path) -> xr.Dataset:
    r"""Load NSIDC-0525 (Scheuchl et al. 2012) Central Antarctica velocity.

    The NSIDC-0525 NetCDF ships without ``x``/``y`` coordinate axes;
    coordinates are reconstructed from the global attributes
    ``xmin``, ``ymax``, ``spacing`` (per the file header). The grid is
    EPSG:3031 polar stereographic south at 900 m, 3200 × 3200.

    Returns an ``xr.Dataset`` with dims ``(y, x)`` and variables
    ``vx``, ``vy`` in m/yr, suitable for slicing/interpolation onto
    a basin grid (matches the schema produced by
    :func:`run_melt._subset_velocity`).
    """
    raw = xr.open_dataset(path)
    # NSIDC-0525 stores xmin/ymax/spacing as strings with trailing " m" units
    # ("                900.0 m"); strip whitespace + non-numeric tail before parse.
    def _parse_meters(v):
        return float(str(v).strip().split()[0])
    nx = int(raw.attrs["nx"]); ny = int(raw.attrs["ny"])
    dx = _parse_meters(raw.attrs["spacing"])
    x_min = _parse_meters(raw.attrs["xmin"]); y_max = _parse_meters(raw.attrs["ymax"])
    x = x_min + np.arange(nx) * dx
    y = y_max - np.arange(ny) * dx  # descending y, north-up convention
    return xr.Dataset(
        {
            "vx": (("y", "x"), raw["vx"].values),
            "vy": (("y", "x"), raw["vy"].values),
        },
        coords={"x": x, "y": y},
        attrs={
            "source": "NSIDC-0525 (Scheuchl, Mouginot & Rignot 2012) "
                      "Central Antarctica Ice Velocity 2009, 900 m, EPSG:3031",
        },
    )


def measuresann_interp(variable, data_dir, xi, yi, method="linear", inpaint_nans=False):
    r"""Interpolate MEaSUREs annual velocity products onto target points.

    Parameters
    ----------
    variable : {"velocity", "error"}
        Which MEaSUREs field to read. ``"velocity"`` returns
        ``(vx, vy)``; ``"error"`` returns ``(vx_error, vy_error)``.
    data_dir : str
        Directory containing MEaSUREs annual ``.nc`` files.
    xi, yi : array-like
        Target coordinates in polar stereographic meters.
    method : str
        :class:`scipy.interpolate.RegularGridInterpolator` method.
    inpaint_nans : bool
        If True, fill NaNs via :func:`scipy.interpolate.griddata`.

    Returns
    -------
    tuple
        ``(values_x, values_y, t_values)`` for ``"velocity"`` and
        ``"error"``, or ``(values, t_values)`` for scalar variants.
    """
    variable_mapping = {
        "velocity": ("vx", "vy"),
        "error": ("vx_error", "vy_error"),
    }

    if variable.lower() not in variable_mapping:
        raise ValueError(
            f"Invalid variable '{variable}'. Choose from {list(variable_mapping.keys())}."
        )

    file_list = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")]
    )
    if not file_list:
        raise FileNotFoundError(f"No NetCDF files found in {data_dir}.")

    vars_to_read = variable_mapping[variable.lower()]
    is_velocity = len(vars_to_read) == 2

    values_x, values_y = [], []
    t_values = np.arange(2018, 2021)

    for file in file_list:
        with xr.open_dataset(file) as ds:
            x = ds["x"].values
            y = ds["y"].values

            data_x = ds[vars_to_read[0]].values
            if is_velocity:
                data_y = ds[vars_to_read[1]].values

            interpolator_x = RegularGridInterpolator(
                (y, x), data_x, method=method, bounds_error=False, fill_value=np.nan
            )
            interpolated_x = interpolator_x(np.column_stack((yi.ravel(), xi.ravel()))).reshape(
                xi.shape
            )

            if is_velocity:
                interpolator_y = RegularGridInterpolator(
                    (y, x), data_y, method=method, bounds_error=False, fill_value=np.nan
                )
                interpolated_y = interpolator_y(np.column_stack((yi.ravel(), xi.ravel()))).reshape(
                    xi.shape
                )

            if inpaint_nans:
                interpolated_x = _inpaint_nans(interpolated_x)
                if is_velocity:
                    interpolated_y = _inpaint_nans(interpolated_y)

            values_x.append(interpolated_x)
            if is_velocity:
                values_y.append(interpolated_y)

    values_x = np.stack(values_x, axis=-1)
    if is_velocity:
        values_y = np.stack(values_y, axis=-1)

    t_values = np.array(t_values).flatten()
    if variable.lower() == "speed":
        return np.sqrt(values_x**2 + values_y**2), t_values
    elif variable.lower() in ["velocity", "error"]:
        return values_x, values_y, t_values
    elif variable.lower() == "count":
        return values_x, t_values


def _inpaint_nans(data):
    r"""Fill NaN values in a 2-D or 3-D array using linear griddata interpolation.

    Parameters
    ----------
    data : numpy.ndarray
        Array with NaN entries to fill.

    Returns
    -------
    numpy.ndarray
        Array with NaNs replaced where interpolation was possible.
    """
    coords = np.array(
        np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij")
    )
    valid_mask = ~np.isnan(data)

    if data.ndim == 3:
        for i in range(data.shape[2]):
            time_slice = data[:, :, i]
            data[:, :, i] = griddata(
                coords[:, valid_mask[:, :, i]].T,
                time_slice[valid_mask[:, :, i]],
                coords.reshape(2, -1).T,
                method="linear",
            ).reshape(data.shape[:2])
    else:
        data = griddata(
            coords[:, valid_mask].T,
            data[valid_mask],
            coords.reshape(2, -1).T,
            method="linear",
        ).reshape(data.shape)

    return data


# TODO add ability to include other velocity products.


def create_velocity_stack(x, y, vel_dir, MYrs, std_thr=2.5, QAnn=None, QYrs=None):
    r"""Build a time-stacked velocity field with gradients and divergence.

    MEaSUREs annual velocities are interpolated onto a grid extended
    10 km outside the input extent, optionally concatenated with an
    auxiliary velocity product (``QAnn``), filtered by a standard-
    deviation threshold, smoothed with a Gaussian kernel, and
    differenced to produce :math:`\partial v_x/\partial x`,
    :math:`\partial v_y/\partial y`, and the divergence stack.

    Parameters
    ----------
    x, y : numpy.ndarray
        Original 1-D grid coordinates in polar stereographic meters.
    vel_dir : str
        Directory containing MEaSUREs annual ``.nc`` files.
    MYrs : array-like
        Years corresponding to the MEaSUREs annual velocities to select.
    std_thr : float
        Outlier threshold in multiples of the per-pixel standard
        deviation.
    QAnn : dict, optional
        Auxiliary velocity dict with ``vx`` and ``vy`` arrays.
    QYrs : array-like, optional
        Years corresponding to ``QAnn``.

    Returns
    -------
    dict
        Velocity fields, statistics, gradients, and divergence on the
        extended grid.
    """
    XB = [min(x) - 10000, max(x) + 10000]
    YB = [min(y) - 10000, max(y) + 10000]
    dx = abs(x[1] - x[0])
    x_new = np.arange(XB[0], XB[1] + dx, dx)
    y_new = np.arange(YB[0], YB[1] + dx, dx)

    xg, yg = np.meshgrid(x_new, y_new)

    MeasAnn_vx, MeasAnn_vy, MeasAnn_t = measuresann_interp(
        "velocity", vel_dir, xg, yg, method="linear", inpaint_nans=True
    )

    VelAnn = {
        "x": x_new,
        "y": y_new,
        "vx": [],
        "vy": [],
        "Years": [],
    }

    Mapidx = np.argmin(np.abs(MeasAnn_t - np.array(MYrs)[:, None]), axis=1)
    print(MeasAnn_vx)
    print(Mapidx)
    MeasAnn_vx = MeasAnn_vx[:, :, Mapidx]
    MeasAnn_vy = MeasAnn_vy[:, :, Mapidx]
    VelAnn["Years"].extend(MYrs)

    if QAnn and QYrs:
        VelAnn["vx"] = np.concatenate((MeasAnn_vx, QAnn["vx"]), axis=-1)
        VelAnn["vy"] = np.concatenate((MeasAnn_vy, QAnn["vy"]), axis=-1)
        VelAnn["Years"].extend(QYrs)
    else:
        VelAnn["vx"] = MeasAnn_vx
        VelAnn["vy"] = MeasAnn_vy

    VelAnn["vx_mean"] = np.nanmean(VelAnn["vx"], axis=-1)
    VelAnn["vy_mean"] = np.nanmean(VelAnn["vy"], axis=-1)
    VelAnn["vx_med"] = np.nanmedian(VelAnn["vx"], axis=-1)
    VelAnn["vy_med"] = np.nanmedian(VelAnn["vy"], axis=-1)
    VelAnn["vx_std"] = np.nanstd(VelAnn["vx"], axis=-1)
    VelAnn["vy_std"] = np.nanstd(VelAnn["vy"], axis=-1)

    for ii in range(VelAnn["vx"].shape[-1]):
        vx_tmp = VelAnn["vx"][:, :, ii]
        mask_vx = (vx_tmp > VelAnn["vx_mean"] + std_thr * VelAnn["vx_std"]) | (
            vx_tmp < VelAnn["vx_mean"] - std_thr * VelAnn["vx_std"]
        )
        vx_tmp[mask_vx | np.isnan(vx_tmp)] = VelAnn["vx_mean"][mask_vx | np.isnan(vx_tmp)]
        VelAnn["vx"][:, :, ii] = gaussian_filter(vx_tmp, sigma=30)

        vy_tmp = VelAnn["vy"][:, :, ii]
        mask_vy = (vy_tmp > VelAnn["vy_mean"] + std_thr * VelAnn["vy_std"]) | (
            vy_tmp < VelAnn["vy_mean"] - std_thr * VelAnn["vy_std"]
        )
        vy_tmp[mask_vy | np.isnan(vy_tmp)] = VelAnn["vy_mean"][mask_vy | np.isnan(vy_tmp)]
        VelAnn["vy"][:, :, ii] = gaussian_filter(vy_tmp, sigma=30)

    VelAnn["vMag"] = np.sqrt(VelAnn["vx"] ** 2 + VelAnn["vy"] ** 2)
    VelAnn["vMag_mean"] = np.nanmean(VelAnn["vMag"], axis=-1)
    VelAnn["vMag_med"] = np.nanmedian(VelAnn["vMag"], axis=-1)

    dx = np.diff(x_new)[0]
    dy = np.diff(y_new)[0]
    VelAnn["grad_vx"] = np.gradient(VelAnn["vx"], dx, axis=0)
    VelAnn["grad_vy"] = np.gradient(VelAnn["vy"], dy, axis=1)
    VelAnn["vdiv_stack"] = VelAnn["grad_vx"] + VelAnn["grad_vy"]

    return VelAnn


def compute_gradients_and_masks(VelAnn, Corrections):
    r"""Compute velocity gradients and refresh the rock-control polygon.

    Parameters
    ----------
    VelAnn : dict
        Velocity stack produced by :func:`create_velocity_stack`.
    Corrections : object
        Correction bundle carrying ``x``, ``y``, and ``mask`` arrays; a
        rock-control polygon derived from ``mask == 1`` is written back
        to ``Corrections["rockPoly"]``.

    Returns
    -------
    dict
        The input ``VelAnn`` updated in place with gradient and
        magnitude statistics.
    """
    dx = np.diff(VelAnn["x"])[0]
    dy = np.diff(VelAnn["y"])[0]

    VelAnn["grad_vx"], _ = np.gradient(VelAnn["vx"], dx, axis=0)
    _, VelAnn["grad_vy"] = np.gradient(VelAnn["vy"], dy, axis=1)
    VelAnn["vdiv_stack"] = VelAnn["grad_vx"] + VelAnn["grad_vy"]

    VelAnn["vMag"] = np.sqrt(VelAnn["vx"] ** 2 + VelAnn["vy"] ** 2)
    VelAnn["vMag_mean"] = np.nanmean(VelAnn["vMag"], axis=-1)
    VelAnn["vMag_med"] = np.nanmedian(VelAnn["vMag"], axis=-1)

    Corrections["rockPoly"] = create_polyshape(
        Corrections["mask"] == 1, Corrections["x"], Corrections["y"]
    )
    return VelAnn
