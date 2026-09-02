# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Greene et al. (2022) time-evolving Antarctic ice-mask loader.

Annual (24 epochs, 1997.75-2021.2) binary ice masks at 240 m from
Greene, Gardner, Schlegel & Fraser (2022), *Antarctic calving loss
rivals ice-shelf thinning*, Nature 609 — the observed coastline
product, so calving-front retreat and advance are both in it. Source
archive: Zenodo record 5903643, file ``icemask_composite.mat``
(MATLAB v7.3 = HDF5; read with h5py). The grid is Antarctic polar
stereographic with x ascending and y descending — the same axis
conventions as the basin stacks — and the mask is 1 = ice, 0 = ocean.

The HDF5 ``ice`` dataset is stored MATLAB column-major as
``(year, x, y)``; per-epoch hyperslabs are read and transposed to the
package's ``(y, x)`` orientation. The full cube is ~10 GB expanded, so
always crop to the template bbox before reading.
"""

import h5py
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


def decimal_year(when) -> float:
    """Convert a date-like (str, datetime64, pandas Timestamp) to decimal years."""
    if isinstance(when, (int, float)):
        return float(when)
    t = np.datetime64(when, "D")
    y0 = t.astype("datetime64[Y]")
    d0 = y0.astype("datetime64[D]")
    d1 = (y0 + 1).astype("datetime64[D]")
    frac = float((t - d0) / (d1 - d0))
    return float(y0.astype(int) + 1970) + frac


def load_greene_years(greene_path) -> np.ndarray:
    """Return the decimal-year epochs of the Greene ice-mask archive."""
    with h5py.File(str(greene_path), "r") as f:
        return np.asarray(f["year"]).ravel().astype(float)


def greene_min_extent_on_grid(
    template: xr.DataArray,
    greene_path,
    *,
    t0,
    t1,
) -> xr.DataArray:
    r"""Window-minimum observed ice extent on the ``template`` grid.

    Logical AND of every Greene annual ice mask with epoch inside
    ``[t0, t1]``, **plus the latest epoch at or before t0** (the
    coastline state the window starts from — without it a window that
    opens between Greene epochs, or after the archive ends at 2021.2,
    would use no initial coastline at all). A pixel survives only if it
    was ice at every one of those epochs, so anything calved before or
    during the window (through 2021.2) is excluded; post-archive calving
    must be handled by the caller (see
    :func:`stereo_melt.shelf_extent.min_shelf_extent`'s stack ocean
    test).

    Parameters
    ----------
    template : xarray.DataArray
        Any field on the basin's analysis grid; only ``x``/``y`` coords
        are used.
    greene_path : str or pathlib.Path
        The ``icemask_composite.mat`` archive.
    t0, t1 : date-like or float
        Analysis window (dates or decimal years).

    Returns
    -------
    xarray.DataArray of bool, dims ``(y, x)``
        True where ice was present at every selected Greene epoch.
    """
    t0d, t1d = decimal_year(t0), decimal_year(t1)
    if t1d < t0d:
        raise ValueError(f"t1 ({t1d}) precedes t0 ({t0d})")
    xs = np.asarray(template["x"].values, dtype=np.float64)
    ys = np.asarray(template["y"].values, dtype=np.float64)

    with h5py.File(str(greene_path), "r") as f:
        years = np.asarray(f["year"]).ravel().astype(float)
        gx = np.asarray(f["x"]).ravel().astype(np.float64)   # ascending
        gy = np.asarray(f["y"]).ravel().astype(np.float64)   # descending

        sel = (years >= t0d) & (years <= t1d)
        before = np.where(years <= t0d)[0]
        if before.size:
            sel[before[-1]] = True
        idx = np.where(sel)[0]
        if idx.size == 0:
            raise ValueError(
                f"no Greene epochs at or before the window [{t0d:.2f}, "
                f"{t1d:.2f}] (archive spans {years[0]:.2f}-{years[-1]:.2f})"
            )

        buf = 1000.0
        i0 = max(np.searchsorted(gx, xs.min() - buf) - 1, 0)
        i1 = min(np.searchsorted(gx, xs.max() + buf) + 1, gx.size)
        # gy descends; searchsorted needs ascending
        gy_asc = gy[::-1]
        j1r = min(np.searchsorted(gy_asc, ys.max() + buf) + 1, gy.size)
        j0r = max(np.searchsorted(gy_asc, ys.min() - buf) - 1, 0)
        j0 = gy.size - j1r
        j1 = gy.size - j0r

        acc = None
        for k in idx:
            # stored (year, x, y): hyperslab -> (nx, ny) -> transpose
            sub = np.asarray(f["ice"][k, i0:i1, j0:j1]).T.astype(bool)
            acc = sub if acc is None else (acc & sub)
        sub_x = gx[i0:i1]
        sub_y = gy[j0:j1]

    interp = RegularGridInterpolator(
        (sub_y[::-1], sub_x), acc[::-1, :].astype(np.float64),
        method="nearest", bounds_error=False, fill_value=0.0,
    )
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    on_grid = interp(np.stack([yy, xx], axis=-1)) > 0.5
    return xr.DataArray(
        on_grid,
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
        name="greene_min_extent",
        attrs={
            "source": "Greene et al. 2022 icemask_composite.mat (Zenodo 5903643)",
            "epochs_used": " ".join(f"{years[k]:.2f}" for k in idx),
            "window": f"{t0d:.3f}..{t1d:.3f}",
        },
    )
