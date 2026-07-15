# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Integrated basal-flux reporting and the grounding-line buffer.

The per-pixel melt-rate fields from :mod:`stereo_melt.melt` are noise-heavy
near the grounding line, where the flux-divergence term
:math:`\nabla\!\cdot(H u)` is amplified by sharp velocity/thickness gradients,
and on sparse repeat-DEM coverage, where the dh/dt regression on few epochs
leaks coregistration residual into thickness change through the ~9.4 hydrostatic
gain. Both inflate the *area integral* (a tail-sensitive quantity) far more than
the median. Reproducing Shean et al. 2019's integrated PIG melt (82-93 Gt/yr)
from our own DEMs required (a) excluding a buffer band inward from the grounding
line and (b) bracketing the integral against its heavy tail. This module
provides both as basin-agnostic helpers.

- :func:`grounding_buffer` — the floating-ice integration domain eroded a fixed
  distance back from grounded ice (keeps the calving front).
- :func:`integrate_basal_flux` — the bracketed area integral (robust / clipped /
  raw) plus the distribution diagnostics that say whether a near-Shean number is
  real channel melt or heavy-tailed pixel noise.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

__all__ = ["grounding_buffer", "integrate_basal_flux"]

RHO_ICE = 918.0  # kg m^-3, matches stereo_melt.constants.rhoi


def _to_bool(a) -> np.ndarray:
    if isinstance(a, xr.DataArray):
        a = a.values
    return np.asarray(a, dtype=bool)


def grounding_buffer(
    floating: "np.ndarray | xr.DataArray",
    grounded: "np.ndarray | xr.DataArray",
    buffer_m: float,
    res_m: float,
) -> np.ndarray:
    r"""Floating-ice domain eroded ``buffer_m`` back from grounded ice.

    The basal-melt continuity terms are least trustworthy in the grounding
    zone: the ice is partly grounded, the velocity field has its sharpest
    gradients, and the finite-difference :math:`\nabla\!\cdot(H u)` overshoots
    there (on clean Shean DEMs this alone drives the deepest "melt" to
    -1300 m/yr and inflates the PIG integral past 140 Gt/yr). Shean buffers the
    grounding line before integrating; this reproduces that.

    The buffer is measured as Euclidean distance from the nearest grounded-ice
    cell, so it follows the true grounding line rather than eroding every edge
    (the calving front is preserved — front pixels are kept).

    Parameters
    ----------
    floating, grounded : ndarray or xarray.DataArray of bool, shape (y, x)
        BedMachine floating-ice (mask==3) and grounded-ice (mask==2) masks on
        the analysis grid.
    buffer_m : float
        Buffer width in meters (e.g. 2000 for a 2 km grounding-line buffer).
        Non-positive returns the floating mask unchanged.
    res_m : float
        Grid resolution in meters (square pixels assumed).

    Returns
    -------
    ndarray of bool, shape (y, x)
        Floating cells whose distance from grounded ice exceeds ``buffer_m``.
    """
    floating = _to_bool(floating)
    grounded = _to_bool(grounded)
    if buffer_m <= 0:
        return floating
    # distance (in cells) from the nearest grounded cell, converted to meters
    dist_m = distance_transform_edt(~grounded) * res_m
    return floating & (dist_m > buffer_m)


def integrate_basal_flux(
    melt: "np.ndarray | xr.DataArray",
    domain: "np.ndarray | xr.DataArray",
    res_m: float,
    *,
    rho_i: float = RHO_ICE,
    cap_myr: float = 250.0,
    quality: "np.ndarray | xr.DataArray | None" = None,
) -> dict:
    r"""Bracketed area-integrated basal melt over ``domain``.

    Melt fields are heavy-tailed, so a single area integral is fragile: the raw
    sum is dominated by a few extreme pixels (real near-GL channels *and* noise
    spikes, indistinguishable per-pixel). This returns three integrals that
    bracket the truth, plus diagnostics:

    - ``gt_raw`` — unguarded sum; upper bound, tail-dominated.
    - ``gt_clip`` — pixels clipped to ``+-cap_myr`` (Shean's max real channel is
      ~250 m/yr); removes blunder spikes while keeping real channels.
    - ``gt_robust`` — ``median x area``; tail-free lower bound.

    Sign convention follows :mod:`stereo_melt.melt`: ``melt`` is negative for
    melt. All ``gt_*`` outputs are returned **positive for melt** (Gt ice / yr).

    Parameters
    ----------
    melt : ndarray or xarray.DataArray, shape (y, x)
        Basal melt rate in m ice / yr, negative = melt.
    domain : ndarray or xarray.DataArray of bool, shape (y, x)
        Integration domain (e.g. :func:`grounding_buffer` output).
    res_m : float
        Grid resolution in meters (square pixels).
    rho_i : float
        Ice density, kg m\ :sup:`-3`.
    cap_myr : float
        Clip magnitude for ``gt_clip``, m/yr.
    quality : ndarray or xarray.DataArray of bool, optional
        Extra per-pixel keep-mask (e.g. dh/dt count >= k AND rmse <= thresh);
        intersected with ``domain``.

    Returns
    -------
    dict
        ``n, area_km2, median_myr, mean_myr, p1_myr, p99_myr, deepest_myr,
        frac_top5, gt_robust, gt_clip, gt_raw``.
    """
    if isinstance(melt, xr.DataArray):
        melt = melt.values
    melt = np.asarray(melt, dtype=float)
    keep = _to_bool(domain)
    if quality is not None:
        keep = keep & _to_bool(quality)
    a = np.where(keep, melt, np.nan)
    v = a[np.isfinite(a)]
    if v.size == 0:
        return {"n": 0, "area_km2": 0.0}
    cell = res_m * res_m
    area_km2 = v.size * cell / 1e6
    med = float(np.median(v))
    gt = lambda arr: float(np.sum(-arr) * cell * rho_i / 1e12)  # noqa: E731 (positive=melt)
    # tail concentration: share of total melt carried by the 5% most-melting px
    melt_only = -v[v < 0]
    total = float(melt_only.sum()) if melt_only.size else np.nan
    k = max(1, int(0.05 * v.size))
    top5 = float(np.sort(melt_only)[::-1][:k].sum()) if melt_only.size else np.nan
    return {
        "n": int(v.size),
        "area_km2": area_km2,
        "median_myr": med,
        "mean_myr": float(np.mean(v)),
        "p1_myr": float(np.percentile(v, 1)),
        "p99_myr": float(np.percentile(v, 99)),
        "deepest_myr": float(np.min(v)),
        "frac_top5": float(top5 / total) if total and np.isfinite(total) else np.nan,
        "gt_robust": (-med) * area_km2 * 1e6 * rho_i / 1e12,
        "gt_clip": gt(np.clip(v, -cap_myr, cap_myr)),
        "gt_raw": gt(v),
    }
