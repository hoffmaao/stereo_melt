# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Post-coregistration tide + IBE correction on the 25 m analysis stack.

Mirrors Shean 2019 ``stack_tidecorr.py``: per-epoch tide (spatially
varying, CATS2008) plus IBE (scalar from the ARCO point cube),
multiplied by a 3 km feathered floating-ice mask, subtracted from each
epoch of the stacked DEM. First step in the post-coreg correction
chain (Shean order: ``tide -> IBE -> MDT -> geoid -> tilt fit``).

ASP ``pc_align`` is anchored on grounded IS2 + rock GCPs, which are
tide-free by construction, so no pre-ASP tide / IBE correction is
required.

Tide is sampled on a ~2 km coarse grid and bilinear-resampled to the
25 m stack: CATS2008 has ~4 km native resolution, so 25 m direct
evaluation oversamples by 2-3 orders of magnitude with no information
gain. IBE varies on synoptic scales (100s of km) and is treated as a
single scalar per epoch from the ARCO point-mode cube; spatial
variation across a single basin is at the millimeter level.

Floating-ice mask: BedMachine v3 ``mask == 3``, nearest-resampled to
the stack grid, then tapered with a 3 km ``scipy.ndimage.uniform_filter``
so the grounding-zone transition is smooth instead of stepping by the
full correction magnitude (~0.4 m IBE + meters of tide). Without the
feather, the GL would inject a sawtooth into ``Dh/Dt`` exactly where
melt rates are most sensitive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
import xarray as xr
from rasterio.transform import from_bounds
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.ndimage import uniform_filter

from .ibe import (
    _METERS_PER_HPA,
    _REFERENCE_PRESSURE_HPA,
    _load_pressure_interpolator,
    _to_seconds_since_epoch,
)
from .tides import apply_tidal_correction

__all__ = ["apply_tide_ibe_to_stack"]


def apply_tide_ibe_to_stack(
    stack: xr.DataArray,
    *,
    climate_cache_path,
    bedmachine_path,
    tide_model: str = "CATS2008",
    tide_model_dir=None,
    tide_grid_step_m: float = 2000.0,
    fwidth_m: float = 3000.0,
) -> xr.DataArray:
    r"""Subtract per-epoch tide + IBE from each epoch of a stacked DEM.

    Parameters
    ----------
    stack : xarray.DataArray, dims ``(time, y, x)``
        Coregistered DEM stack on a common grid (EPSG:3031). Must
        carry ``time``, ``y``, ``x`` coords.
    climate_cache_path : str or pathlib.Path
        ARCO point-mode pressure cube produced by
        :func:`stereo_melt.pipeline.populate_climate_cache`.
    bedmachine_path : str or pathlib.Path
        BedMachine v3 NetCDF (used for the floating-ice mask).
    tide_model, tide_model_dir : str / pathlib.Path
        See :func:`stereo_melt.corrections.tides.apply_tidal_correction`.
        Default ``"CATS2008"`` for Antarctic shelves.
    tide_grid_step_m : float
        Spacing (m) of the coarse grid on which CATS2008 is evaluated
        before bilinear-resampling to the stack 25 m. Default
        ``2000`` (oversamples CATS native ~4 km by 2x).
    fwidth_m : float
        Spatial width (m) of the ``uniform_filter`` used to feather
        the floating-ice mask across the grounding zone. Default
        ``3000`` (Shean PIG).

    Returns
    -------
    xarray.DataArray
        Corrected stack with the same shape, dims, and coords as the
        input. The original ``attrs`` are preserved and a
        ``corrections`` entry is appended noting tide + IBE.
    """
    if "time" not in stack.dims:
        raise ValueError("stack must have a 'time' dimension")
    for dim in ("y", "x"):
        if dim not in stack.dims:
            raise ValueError(f"stack must have a '{dim}' dimension")

    x_axis = stack["x"].values.astype(np.float64)
    y_axis = stack["y"].values.astype(np.float64)
    times = stack["time"].values

    # Floating mask, feathered. Static in time -- compute once.
    mask = _build_feathered_floating_mask(
        x_axis, y_axis, bedmachine_path, fwidth_m,
    )
    n_floating = int((mask > 0).sum())
    print(
        f"🌊 post-coreg tide+IBE: floating-mask coverage "
        f"{n_floating}/{mask.size} ({100*n_floating/mask.size:.1f}% of grid)"
    )

    # Build IBE interpolator once (ARCO point cube is shared across epochs).
    ibe_interp = _load_pressure_interpolator(climate_cache_path, "surface_pressure")
    if ibe_interp._mode != "point":
        raise NotImplementedError(
            "apply_tide_ibe_to_stack expects a point-mode (1-D time) "
            f"climate cube; got mode={ibe_interp._mode!r}."
        )

    corrected = stack.copy(deep=True)
    data = corrected.values

    for k, t in enumerate(times):
        center_time = np.datetime64(t)
        tide_field = _tide_on_stack_grid(
            x_axis, y_axis, center_time,
            tide_model=tide_model, tide_model_dir=tide_model_dir,
            grid_step_m=tide_grid_step_m,
        )
        ibe_scalar = _scalar_ibe_at(ibe_interp, center_time)

        correction = (tide_field + ibe_scalar) * mask
        data[k] = data[k] - correction
        print(
            f"   epoch {k+1}/{len(times)}  {str(center_time)[:19]}  "
            f"tide(median)={np.nanmedian(tide_field):+.3f} m  "
            f"IBE={ibe_scalar:+.4f} m"
        )

    corrected.values = data
    existing = corrected.attrs.get("corrections", "")
    appended = "tide+IBE@stack(25m, 3km-feathered floating mask)"
    corrected.attrs["corrections"] = (
        f"{existing}; {appended}" if existing else appended
    )
    return corrected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tide_on_stack_grid(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    center_time: np.datetime64,
    *,
    tide_model: str,
    tide_model_dir,
    grid_step_m: float,
    buffer_m: float = 5000.0,
) -> np.ndarray:
    """Evaluate CATS2008 on a coarse grid, fill NaNs, bilinear-resample to (y, x).

    CATS2008 is defined on a ~4 km grid; sampling the analysis 25 m
    grid directly would do ~25 600 redundant pyTMD evaluations per
    25 m pixel cluster. ``grid_step_m=2000`` keeps the coarse grid
    well above Nyquist for CATS while reducing the pyTMD call count
    by ~6400x.

    pyTMD returns ``NaN`` over land / grounded ice, but valid values
    under the floating shelf where CATS2008 has cavity coverage.
    Bilinear-resampling a field with NaN holes near the GL would
    smear zeros into the shelf and break the correction near the
    grounding zone. We fill the NaNs by thin-plate-spline RBF
    extrapolation over the valid samples
    (``scipy.interpolate.RBFInterpolator``, ``kernel='thin_plate_spline'``)
    so the filled field is C^2-smooth across the valid/extrapolated
    boundary -- no Voronoi-cell edges to leak into the feathered
    transition zone.

    A small ``buffer_m`` extension of the coarse grid beyond the
    stack extent gives the RBF extra anchor points near the stack
    boundary.
    """
    # Build coarse axes that span the stack extent plus a buffer.
    x_min = float(x_axis.min()) - buffer_m
    x_max = float(x_axis.max()) + buffer_m
    y_min = float(y_axis.min()) - buffer_m
    y_max = float(y_axis.max()) + buffer_m
    n_cx = max(2, int(np.ceil((x_max - x_min) / grid_step_m)) + 1)
    n_cy = max(2, int(np.ceil((y_max - y_min) / grid_step_m)) + 1)
    coarse_x = np.linspace(x_min, x_max, n_cx)
    coarse_y = np.linspace(y_min, y_max, n_cy)
    cx, cy = np.meshgrid(coarse_x, coarse_y)
    tide_coarse_raw = apply_tidal_correction(
        cx.ravel(), cy.ravel(), center_time,
        model=tide_model, model_directory=tide_model_dir,
    ).reshape(cy.shape)

    # Fill NaNs by thin-plate-spline RBF over the valid CATS samples.
    # smoothing=0 reproduces valid samples exactly; the kernel is
    # C^2-continuous so the filled field has no edges where it
    # transitions from "interpolated" (inside the convex hull of the
    # valid samples) to "extrapolated" (outside).
    valid = np.isfinite(tide_coarse_raw)
    if not valid.any():
        print(
            f"⚠️  CATS2008 returned no valid points across the coarse grid for "
            f"{str(center_time)[:19]}; tide correction set to 0 for this epoch."
        )
        return np.zeros((len(y_axis), len(x_axis)))
    pts_all = np.column_stack([cx.ravel(), cy.ravel()])
    valid_flat = valid.ravel()
    rbf = RBFInterpolator(
        pts_all[valid_flat],
        tide_coarse_raw.ravel()[valid_flat],
        kernel="thin_plate_spline",
        smoothing=0.0,
    )
    tide_coarse = rbf(pts_all).reshape(tide_coarse_raw.shape)

    # RegularGridInterpolator wants strictly ascending axes; the stack
    # convention is y descending, so flip both the y axis and its
    # corresponding rows of the field before constructing the
    # interpolator.
    cy_axis = coarse_y
    field = tide_coarse
    if cy_axis[0] > cy_axis[-1]:
        cy_axis = cy_axis[::-1]
        field = field[::-1, :]
    interp = RegularGridInterpolator(
        (cy_axis, coarse_x), field,
        bounds_error=False, fill_value=None, method="linear",
    )
    xs2d, ys2d = np.meshgrid(x_axis, y_axis)
    pts = np.column_stack([ys2d.ravel(), xs2d.ravel()])
    return interp(pts).reshape(xs2d.shape)


def _scalar_ibe_at(interp, center_time: np.datetime64) -> float:
    """IBE displacement (m) from a point-mode ARCO interpolator."""
    t_secs = _to_seconds_since_epoch(np.datetime64(center_time))
    pressure_hpa = float(interp(np.array([[t_secs, 0.0, 0.0]]))[0])
    return -(pressure_hpa - _REFERENCE_PRESSURE_HPA) * _METERS_PER_HPA


def _build_feathered_floating_mask(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    bedmachine_path,
    fwidth_m: float,
) -> np.ndarray:
    """3 km-feathered floating-ice mask resampled to the stack grid.

    Returns a float array on ``(len(y), len(x))`` valued in ``[0, 1]``,
    1 deep on shelf, 0 on grounded ice / land / ocean, ramping in
    between with a transition width of ``fwidth_m`` (set by the
    ``scipy.ndimage.uniform_filter`` kernel size).
    """
    bedmachine_path = Path(bedmachine_path)
    bm = xr.open_dataset(bedmachine_path)
    try:
        mask_arr = bm["mask"].values.astype(np.int16)
        bm_x = bm["x"].values.astype(np.float64)
        bm_y = bm["y"].values.astype(np.float64)
    finally:
        bm.close()

    dx = float(bm_x[1] - bm_x[0])
    dy = float(bm_y[1] - bm_y[0])
    if dy < 0:
        west = float(bm_x[0]) - 0.5 * dx
        east = float(bm_x[-1]) + 0.5 * dx
        north = float(bm_y[0]) - 0.5 * dy
        south = float(bm_y[-1]) + 0.5 * dy
    else:
        west = float(bm_x[0]) - 0.5 * dx
        east = float(bm_x[-1]) + 0.5 * dx
        south = float(bm_y[0]) - 0.5 * dy
        north = float(bm_y[-1]) + 0.5 * dy
        mask_arr = mask_arr[::-1, :]
    src_transform = from_bounds(
        west, south, east, north, mask_arr.shape[1], mask_arr.shape[0],
    )
    src_crs = "EPSG:3031"

    # Stack grid extent.
    sx = float(x_axis[1] - x_axis[0])
    sy = float(y_axis[1] - y_axis[0])
    if sy > 0:
        # Stack is unusual: y ascending. Flip on output to north-up.
        flip_y = True
        x_axis_for_warp = x_axis
        y_axis_for_warp = y_axis[::-1]
    else:
        flip_y = False
        x_axis_for_warp = x_axis
        y_axis_for_warp = y_axis
    sw = float(x_axis_for_warp[0]) - 0.5 * sx
    se = float(x_axis_for_warp[-1]) + 0.5 * sx
    sn = float(y_axis_for_warp[0]) - 0.5 * abs(sy)
    ss = float(y_axis_for_warp[-1]) + 0.5 * abs(sy)
    dst_transform = from_bounds(
        sw, ss, se, sn, len(x_axis_for_warp), len(y_axis_for_warp),
    )

    dst = np.zeros((len(y_axis_for_warp), len(x_axis_for_warp)), dtype=np.int16)
    rasterio.warp.reproject(
        source=mask_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:3031",
        resampling=rasterio.warp.Resampling.nearest,
    )

    floating = (dst == 3).astype(np.float64)

    # Feather with uniform_filter. Kernel width in pixels = fwidth_m / pixel size.
    px = abs(sx)
    py = abs(sy)
    kx = max(1, int(round(fwidth_m / px)))
    ky = max(1, int(round(fwidth_m / py)))
    feathered = uniform_filter(floating, size=(ky, kx), mode="nearest")

    if flip_y:
        feathered = feathered[::-1, :]
    return feathered
