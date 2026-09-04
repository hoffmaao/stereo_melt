# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Joint least-squares tilt optimizer for DEM stacks.

Following Shean 2019 (``ndinterp.py``), jointly estimates per-pixel
intercept, per-pixel linear trend, and per-epoch planar tilt
:math:`(\alpha_x, \alpha_y, \alpha_z)` from a stack of coregistered
DEMs. The observation model for pixel :math:`p` at epoch :math:`k` is

.. math::
    z_{p,k} = z^0_p + \dot h_p\,\tilde t_k
             + \alpha_{x,k}\,(x_p - \bar x_k)
             + \alpha_{y,k}\,(y_p - \bar y_k)
             + \alpha_{z,k}

where :math:`\tilde t_k = t_k - \bar t` is the **mean-centered** time
in **days** (matches Shean's normalization for numerical conditioning)
and :math:`(\bar x_k, \bar y_k)` is the centroid of the valid pixels at
epoch :math:`k`. With this convention the recovered :math:`z^0_p` is
elevation at the mean epoch and :math:`\dot h_p` is in **m / day**.

A per-pixel temporal-median reference :math:`\hat z_p =
\mathrm{median}_k(z_{p,k})` is subtracted from the observations
before the LSQ and added back to :math:`z^0_p` afterwards (mirrors
Shean ``ndinterp.py``). This is what lets the ``Eint`` Tikhonov prior
on :math:`z^0_p` carry sensible meters-of-residual semantics across
domains with kilometers of topographic relief: without the
subtraction the prior pulls the absolute intercept toward zero and
the LSQ smuggles topography into the per-epoch :math:`\alpha_z` and
the per-pixel trend :math:`\dot h_p`.

The system is solved in a single sparse LSQ with Tikhonov
regularization on each parameter block:

.. math::
    \min_{z^0, \dot h, \alpha}
        \| A\,\mathbf{x} - \mathbf{b} \|_2^2
        + \| E^{-1}\,\mathbf{x} \|_2^2

where :math:`E` holds the prior standard deviation of each unknown
(``Eint`` on intercepts, ``Edhdt`` on trends, ``Ex``, ``Ey``, ``Ez`` on
tilt coefficients). Default magnitudes mirror Shean's PIG values; the
trend prior is intentionally loose (1 m/day) so the work of separating
static control from dynamic ice falls on the input mask -- see
:func:`build_static_control_mask`.

Two solvers are exposed: ``"lsmr"`` (default) runs LSMR directly on the
augmented sparse system and scales to full-AOI joint problems;
``"spsolve"`` matches Shean's UMFPACK normal-equations path and is
faster at moderate sizes.

A stack of :math:`T=2` is a valid degenerate case: the per-pixel trend
collapses to a finite difference and the solver behaves as a pair-wise
plane fit.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import xarray as xr

from ..backend import asarray, backend, to_numpy, xp

__all__ = [
    "apply_tilt",
    "build_ice_domain_mask",
    "build_static_area_polygon_mask",
    "build_static_control_mask",
    "fit_tilt_stack",
]


def apply_tilt(
    dem: xr.DataArray,
    dx: float,
    dy: float,
    dz: float,
    xref: float = 0.0,
    yref: float = 0.0,
) -> xr.DataArray:
    r"""Return ``dem`` with a planar tilt subtracted.

    .. math::
        z_\mathrm{corr}(x, y) =
            z(x, y) - \left[
                \alpha_x\,(x - \bar x)
              + \alpha_y\,(y - \bar y)
              + \alpha_z
            \right]

    Parameters
    ----------
    dem : xarray.DataArray, dims ``(y, x)``
        Input DEM.
    dx, dy, dz : float
        Tilt coefficients :math:`\alpha_x, \alpha_y, \alpha_z`.
    xref, yref : float
        Centroid used during tilt fitting.

    Returns
    -------
    xarray.DataArray
        Tilt-corrected DEM on the input grid.
    """
    x = dem["x"].values
    y = dem["y"].values
    X, Y = np.meshgrid(x, y)
    tilt = dx * (X - xref) + dy * (Y - yref) + dz
    return dem - tilt


def build_static_area_polygon_mask(
    stack: xr.DataArray,
    bedmachine_path: Path | str,
    *,
    erode_grounded_m: float = 2000.0,
) -> xr.DataArray:
    r"""BedMachine rock + grounded-ice polygon resampled onto the stack grid.

    Used as the polygon-style restriction Shean's ``ndinterp.py`` calls
    ``pig_mainshelfmargins_upstreamtrunk_mask_for_tiltcorr.shp``: it
    zeros out the floating shelf and ocean before the temporal-stat
    filter in :func:`build_static_control_mask` runs. Pass the result as
    that function's ``shapefile_mask`` argument.

    Asymmetric grounding-zone erosion
    ---------------------------------
    BedMachine v3 has known sliver errors at the grounded/floating
    boundary: pixels 0-1 km from the grounding line that are mislabeled
    grounded but are actually floating tongue (or are real grounding-zone
    dynamic-thinning pixels the temporal-trend filter doesn't catch).
    These slivers carry +0.12-0.17 m/yr dh/dt that pollutes the per-strip
    :math:`\alpha_z` fit through the LSQ -> :math:`\times 9.42`
    hydrostatic gain -> fake -1.6 m/yr "freeze" in the recovered melt
    rate.

    The fix mirrors what Shean's hand-digitized PIG shapefile
    accomplished naturally: back grounded-ice pixels away from the
    non-grounded (ocean/floating/lake) boundary by ``erode_grounded_m``
    meters. **Rock outcrops are preserved as-is** — they are static
    regardless of where they sit, and they provide most of the spatial
    leverage for fitting per-epoch :math:`\alpha_x, \alpha_y` slopes.

    Empirical sweep on Nansen 2026-05-06 placed the bias minimum at
    ~2 km erosion (mean static-control dh/dt +0.062 -> +0.037 m/yr,
    eliminating ~+0.24 m/yr of fake "freeze" through the hydrostatic
    gain). Going past 5 km strips out the diluting deep-grounded-ice
    pixels and the bias regresses toward the rock-only floor of
    ~+0.05 m/yr (a separate effect, snow / mountain coreg residual).

    Parameters
    ----------
    stack : xarray.DataArray, dims ``(time, y, x)`` or ``(y, x)``
        Repeat-DEM stack on the analysis grid; only the ``x``/``y``
        coords are used.
    bedmachine_path : pathlib.Path or str
        Path to the BedMachine v3 NetCDF holding the ``mask`` variable
        with codes ``1``=rock, ``2``=grounded ice, ``3``=floating, ``4``=lake,
        ``0``=ocean.
    erode_grounded_m : float, keyword-only
        Minimum distance (in metres) from any non-grounded BedMachine
        cell that a grounded-ice pixel must satisfy to remain in the
        polygon. ``0`` recovers the legacy ``(bm == 1) | (bm == 2)``
        behaviour. Default ``2000`` (Nansen sweet spot).

    Returns
    -------
    xarray.DataArray
        Boolean mask on ``(y, x)`` with the input ``y, x`` coords.
        Attributes record the source string, ``erode_grounded_m``, and
        the kept-pixel counts for rock and grounded ice.
    """
    bm_path = Path(bedmachine_path)
    if not bm_path.exists():
        raise FileNotFoundError(f"BedMachine missing at {bm_path}")
    ds = xr.open_dataset(bm_path)
    buf = 2000.0
    x_min = float(stack["x"].min()) - buf
    x_max = float(stack["x"].max()) + buf
    y_min = float(stack["y"].min()) - buf
    y_max = float(stack["y"].max()) + buf
    by = ds["y"].values
    if by[0] > by[-1]:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
    else:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    bm = sub.load().interp(x=stack["x"], y=stack["y"], method="nearest").values

    rock = (bm == 1)
    grounded = (bm == 2)

    if erode_grounded_m > 0:
        from scipy import ndimage

        res = abs(float(stack["x"].values[1] - stack["x"].values[0]))
        kept = rock | grounded
        dist_m = ndimage.distance_transform_edt(kept).astype(float) * res
        polygon = rock | (grounded & (dist_m >= erode_grounded_m))
        attrs_source = (
            f"BedMachine v3: rock (mask==1) | grounded (mask==2) eroded "
            f"{erode_grounded_m:.0f} m from non-grounded boundary"
        )
    else:
        polygon = rock | grounded
        attrs_source = "BedMachine v3 mask in {1,2} (rock + grounded ice)"

    return xr.DataArray(
        polygon, dims=("y", "x"),
        coords={"y": stack["y"], "x": stack["x"]},
        name="static_polygon_mask",
        attrs={
            "source": attrs_source,
            "erode_grounded_m": float(erode_grounded_m),
            "n_rock": int(rock.sum()),
            "n_grounded_kept": int((grounded & polygon).sum()),
        },
    )


def build_ice_domain_mask(
    stack: xr.DataArray,
    bedmachine_path: Path | str,
) -> xr.DataArray:
    r"""Full ice + rock domain (BedMachine rock|grounded|floating) on the stack grid.

    Unlike :func:`build_static_area_polygon_mask` -- which keeps only the
    *static* rock + grounded margins that anchor the per-epoch tilts --
    this returns the whole non-ocean ice domain: rock (1), grounded ice
    (2) and floating shelf (3), excluding open ocean (0) and lakes (4).

    Intended as the ``observation_mask`` argument to
    :func:`fit_tilt_stack`. It admits the floating-shelf interior pixels
    as LSQ observations (their per-pixel ``intercept`` / ``dhdt`` nuisance
    parameters absorb the real basal-melt time trend) so the per-epoch
    tilt slopes get leverage from the full domain instead of being
    extrapolated off the grounded margins. Open ocean is deliberately
    excluded: geoid-referenced ocean rounds to ~0 m and would inject
    spurious near-zero observations into the fit.

    Parameters
    ----------
    stack : xarray.DataArray, dims ``(time, y, x)`` or ``(y, x)``
        Repeat-DEM stack on the analysis grid; only the ``x``/``y``
        coords are used.
    bedmachine_path : pathlib.Path or str
        Path to the BedMachine v3 NetCDF holding the ``mask`` variable
        with codes ``1``=rock, ``2``=grounded ice, ``3``=floating,
        ``4``=lake, ``0``=ocean.

    Returns
    -------
    xarray.DataArray
        Boolean mask on ``(y, x)`` with the input ``y, x`` coords.
    """
    bm_path = Path(bedmachine_path)
    if not bm_path.exists():
        raise FileNotFoundError(f"BedMachine missing at {bm_path}")
    ds = xr.open_dataset(bm_path)
    buf = 2000.0
    x_min = float(stack["x"].min()) - buf
    x_max = float(stack["x"].max()) + buf
    y_min = float(stack["y"].min()) - buf
    y_max = float(stack["y"].max()) + buf
    by = ds["y"].values
    if by[0] > by[-1]:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
    else:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    bm = sub.load().interp(x=stack["x"], y=stack["y"], method="nearest").values

    domain = (bm == 1) | (bm == 2) | (bm == 3)

    return xr.DataArray(
        domain, dims=("y", "x"),
        coords={"y": stack["y"], "x": stack["x"]},
        name="ice_domain_mask",
        attrs={
            "source": "BedMachine v3 mask in {1,2,3} (rock + grounded + floating ice)",
            "n_rock": int((bm == 1).sum()),
            "n_grounded": int((bm == 2).sum()),
            "n_floating": int((bm == 3).sum()),
        },
    )


def build_static_control_mask(
    stack: xr.DataArray,
    *,
    min_count: int = 4,
    min_ptp_years: float = 1.5,
    max_std: float = 4.0,
    min_z: float = 10.0,
    shapefile_mask: xr.DataArray | np.ndarray | None = None,
    detrended_residual_thresh: float = 3.0,
    abs_trend_thresh_myr: float = 2.0,
) -> xr.DataArray:
    r"""Stack-based mask of static-control pixels (Shean ``ndinterp.py``).

    Selects pixels whose elevation time series is well-observed and
    behaves like a static surface, so they can anchor the per-epoch
    planar tilt fit. Mirrors Shean 2019 ``ndinterp.py``: a pixel passes
    iff

    1. it has at least ``min_count`` finite observations,
    2. the temporal span between first and last observation is
       :math:`\geq` ``min_ptp_years``,
    3. its temporal standard deviation is :math:`\leq` ``max_std`` m,
    4. its mean elevation is :math:`>` ``min_z`` m,
    5. (optional) it lies inside ``shapefile_mask``,
    6. its detrended residual standard deviation is :math:`<`
       ``detrended_residual_thresh`` m AND the magnitude of its linear
       trend is :math:`<` ``abs_trend_thresh_myr`` m/yr.

    The trend / detrended-residual filter is the part that makes the
    mask "static": a pixel that survives the std/mean checks but has a
    coherent trend (e.g. dynamic thinning) is excluded so the joint
    LSQ in :func:`fit_tilt_stack` does not mistake it for control.

    Parameters
    ----------
    stack : xarray.DataArray, dims ``(time, y, x)``
        Coregistered DEM stack on a common grid.
    min_count : int
        Minimum number of finite epochs at a pixel. Default ``4``.
    min_ptp_years : float
        Minimum first-to-last temporal span (years). Default ``1.5``.
    max_std : float
        Maximum allowed temporal std (m). Default ``4.0``.
    min_z : float
        Minimum allowed temporal mean (m). Default ``10.0`` (rejects
        ocean / floating ice that round to ~0 in geoid-referenced DEMs).
    shapefile_mask : xarray.DataArray or numpy.ndarray, dims ``(y, x)``, optional
        Optional polygon-derived mask (e.g. grounded margins / upstream
        trunk, excluding the floating shelf). Multiplied into the
        intermediate result.
    detrended_residual_thresh : float
        Maximum allowed std of (z - linear_fit(t)) residuals (m).
        Default ``3.0`` (Shean PIG).
    abs_trend_thresh_myr : float
        Maximum allowed magnitude of the per-pixel linear trend (m/yr).
        Default ``2.0`` (Shean PIG).

    Returns
    -------
    xarray.DataArray
        Boolean mask on ``(y, x)`` with the input ``y, x`` coords.
    """
    if "time" not in stack.dims:
        raise ValueError("stack must have a 'time' dimension")

    z = stack.values  # (T, ny, nx)
    valid = np.isfinite(z)
    n_valid = valid.sum(axis=0)

    times = stack["time"].values
    t_years = (times - times[0]).astype("timedelta64[s]").astype(float) / (86400.0 * 365.25)
    t_3d = t_years[:, None, None]

    # Temporal ptp (first-to-last span among valid samples).
    t_masked = np.where(valid, t_3d, np.nan)
    with np.errstate(invalid="ignore"):
        t_max = np.nanmax(t_masked, axis=0)
        t_min = np.nanmin(t_masked, axis=0)
        ptp_years = t_max - t_min

    # Temporal mean / std using NaN masking.
    z_masked = np.where(valid, z, np.nan)
    with np.errstate(invalid="ignore"):
        z_mean = np.nanmean(z_masked, axis=0)
        z_std = np.nanstd(z_masked, axis=0)

    base = (
        (n_valid >= min_count)
        & (ptp_years >= min_ptp_years)
        & np.where(np.isfinite(z_std), z_std <= max_std, False)
        & np.where(np.isfinite(z_mean), z_mean > min_z, False)
    )

    if shapefile_mask is not None:
        sm = shapefile_mask.values if isinstance(shapefile_mask, xr.DataArray) else shapefile_mask
        sm = np.asarray(sm, dtype=bool)
        if sm.shape != base.shape:
            raise ValueError(
                f"shapefile_mask shape {sm.shape} != stack (y, x) {base.shape}"
            )
        base = base & sm

    # Per-pixel linear regression (vectorised) for the trend / residual filter.
    n_eff = n_valid.astype(float)
    safe_n = np.where(n_eff > 0, n_eff, 1.0)
    z_fill = np.where(valid, z, 0.0)
    t_fill = np.where(valid, t_3d, 0.0)
    sum_z = z_fill.sum(axis=0)
    sum_t = t_fill.sum(axis=0)
    sum_zt = (z_fill * t_fill).sum(axis=0)
    sum_tt = (t_fill * t_fill).sum(axis=0)
    mean_z = sum_z / safe_n
    mean_t = sum_t / safe_n
    cov_tz = sum_zt / safe_n - mean_t * mean_z
    var_t = sum_tt / safe_n - mean_t * mean_t
    slope = np.where(var_t > 0, cov_tz / var_t, 0.0)
    intercept = mean_z - slope * mean_t

    fit_3d = intercept[None, :, :] + slope[None, :, :] * t_3d
    resid = np.where(valid, z - fit_3d, np.nan)
    with np.errstate(invalid="ignore"):
        resid_std = np.nanstd(resid, axis=0)

    static = base & (
        np.where(np.isfinite(resid_std), resid_std < detrended_residual_thresh, False)
    ) & (np.abs(slope) < abs_trend_thresh_myr)

    return xr.DataArray(
        static,
        dims=("y", "x"),
        coords={"y": stack["y"], "x": stack["x"]},
        name="control_mask",
        attrs={
            "source": "Shean ndinterp.py-style stack-based static-control filter",
            "min_count": min_count,
            "min_ptp_years": min_ptp_years,
            "max_std_m": max_std,
            "min_z_m": min_z,
            "detrended_residual_thresh_m": detrended_residual_thresh,
            "abs_trend_thresh_myr": abs_trend_thresh_myr,
            "has_shapefile_mask": shapefile_mask is not None,
        },
    )


def fit_tilt_stack(
    stack: xr.DataArray,
    control_mask: xr.DataArray | np.ndarray | None = None,
    observation_mask: xr.DataArray | np.ndarray | None = None,
    Eint: float = 10.0,
    Edhdt: float = 1.0,
    Ex: float = 2e-6,
    Ey: float = 2e-6 / 3.0,
    Ez: float | np.ndarray = 0.1,
    min_width: float = 0.0,
    offset_only_epochs: np.ndarray | None = None,
    dhdt_smoothness: float | None = None,
    solver: str = "lsmr",
    lsmr_atol: float = 1e-8,
    lsmr_maxiter: int | None = None,
    robust: bool = True,
    robust_max_iter: int = 8,
    robust_tol: float = 1e-3,
    robust_c: float = 4.685,
) -> tuple[xr.Dataset, xr.DataArray]:
    r"""Fit per-pixel trend and per-epoch tilt jointly to a DEM stack.

    Solves the Tikhonov-regularized sparse LSQ system described in the
    module docstring and returns both the recovered parameters and the
    tilt-corrected stack.

    Parameters
    ----------
    stack : xarray.DataArray, dims ``(time, y, x)``
        Coregistered DEM stack on a common grid. Must carry ``time``,
        ``y``, ``x`` coordinates.
    control_mask : xarray.DataArray or numpy.ndarray, dims ``(y, x)``, optional
        Boolean mask marking pixels to include in the fit (rock
        outcrops + slow-velocity grounded ice are the canonical
        choice). When ``observation_mask`` is also None, this
        controls *both* which observations enter the LSQ and which
        pixels are reported as "control". Defaults to all valid pixels.
    observation_mask : xarray.DataArray or numpy.ndarray, dims ``(y, x)``, optional
        Boolean mask marking which pixels' observations to include in
        the joint LSQ. When None (default), uses ``control_mask`` —
        the legacy "static-control-only" mode that constrains the
        per-strip tilts using rock+grounded observations alone.
        Setting this to a wider mask (e.g. ``static | floating`` or the
        full grid where observations are finite) matches Shean 2019 /
        Smith ``ndinterp.py``: the per-pixel ``intercept`` and
        ``dhdt`` parameters become nuisance variables that absorb the
        real basal-melt time trend on floating ice while the
        per-strip tilt parameters get strong leverage from every
        observation. Tikhonov priors keep the per-pixel trend
        well-posed even on noisy floating pixels.
    Eint : float
        Prior std. dev. on per-pixel intercepts (meters). Larger =
        weaker regularization. Default ``10.0`` (Shean PIG).
    Edhdt : float
        Prior std. dev. on per-pixel trends (m / day). Default ``1.0``
        (Shean PIG -- effectively unregularized; the input mask is
        expected to do the heavy lifting of excluding dynamic pixels).
    Ex, Ey : float
        Prior std. dev. on per-epoch tilt slopes (m/m). Defaults
        ``2e-6`` and ``2e-6/3`` (Shean PIG: along-track Ex, ~3x
        tighter cross-track Ey).
    Ez : float or numpy.ndarray, dims ``(time,)``
        Prior std. dev. on per-epoch offsets (meters). Default ``0.1``
        (tightened from Shean PIG's 0.3 after the Nansen 2026-05-05
        dh/dt diagnosis showed a +0.07 m/yr static-control bias survived
        Ez=0.3; appropriate for IS2/ATM/LVIS-controlled strips). Pass a
        per-epoch array to loosen the prior for epochs whose ASP control
        was CryoSat-2-only — e.g. ``Ez=2.0`` for pre-IS2 (pre-Oct 2018)
        epochs and ``0.1`` for IS2-era epochs. Tight Ez over-shrinks the
        recovered :math:`\alpha_z` when the real coregistration residual
        is meter-scale (so loosen it for CS2-only epochs).
    min_width : float
        Minimum spatial spread (meters) required to fit the slope
        components :math:`\alpha_x, \alpha_y` at an epoch. Narrower
        epochs fit :math:`\alpha_z` only. Default ``0`` — no spread
        gate; every epoch fits slopes unless ``offset_only_epochs``
        says otherwise, and ill-conditioned slopes are damped
        continuously by the ``Ex``/``Ey`` Tikhonov prior rather than by
        a cliff. The old default was ``40000`` (inherited from Shean
        PIG), which all seven basin drivers already overrode with
        ``10000`` because 40 km disabled slope fitting almost
        everywhere; it was never a value production ran at, which is why
        the default change leaves PIG's published numbers untouched.
        Three call sites, in two non-production files, do NOT pass
        ``min_width`` and so take the new default.
        ``scripts/test_gpu_tilt_fit.py`` is a GPU segfault/timing smoke
        script that asserts nothing. ``tests/sanity_tilt_stack.py``
        (both of its fits) takes the new default DELIBERATELY and DOES
        assert on slopes -- tightly, at ``atol`` 1e-9 mean-removed and
        1e-6 for the T=2 case, since slope recovery is the whole point
        of that file. Its 10 km synthetic gives ``dist_ptp`` ~4.7 km,
        so under the old ``40000`` default ``fit_xy`` was False and the
        slope columns came back empty: the assertions were passing
        against a solver that had not fitted anything, which is exactly
        the stale-test bug this default change fixed. Raising the
        default again would break that registered gate. NOTE the gate still bites hard
        at the drivers' 10 km: 52.8 % of PIG's 513 epochs fit
        :math:`\alpha_z` only. Changing what the *drivers* pass is a
        science change that moves published melt numbers -- do it
        deliberately, with a re-run, not by editing this default.
    dhdt_smoothness : float, optional
        Weight of the Shean ``ndinterp.py`` spatial-smoothness
        constraint on the per-pixel trend field (his L574+ "Smoothness
        Constraint" block, ported 2026-07-11). For every observed pixel
        with both vertical (up/down) and/or both horizontal
        (left/right) observed neighbours, appends a second-difference
        row on the ``dhdt`` unknowns (center ``+2`` per direction pair,
        neighbours ``-1``; RHS 0), scaled by this weight. ``1.0``
        reproduces Shean's unit-weight rows. These rows are appended to
        the fixed (never IRLS-reweighted) regularization block. This is
        what stabilizes the per-pixel trend on sparsely-observed
        floating pixels and makes the shelf-inclusive observation
        domain solvable — without it, per-epoch tilt/offset trades
        against per-pixel trend wherever the temporal sampling is thin
        (the 2026-06-29 "manufactured shelf-front accretion" revert,
        and the 2026-07-11 nocorr αz≈0 failure, are the two faces of
        running Shean's domain without Shean's stabilizer).
        ``None``/``0`` (default) = off, bit-exact legacy system.
    solver : {"lsmr", "spsolve"}
        Linear solver. ``"spsolve"`` uses sparse LU via UMFPACK on the
        normal equations; can run out of memory when every valid pixel
        is used as an observation (cross-coupling between per-pixel
        intercepts/trends and per-epoch tilts produces large LU
        fill-in). ``"lsmr"`` (default) uses
        :func:`scipy.sparse.linalg.lsmr` directly on the augmented
        sparse system; memory scales with the non-zero count of
        ``A_reg``, not with LU fill-in.
    lsmr_atol : float
        Absolute residual tolerance for ``lsmr``.
    lsmr_maxiter : int, optional
        Iteration cap for ``lsmr``; defaults to ``min(M+N, 20*N)``
        which is what scipy uses internally.
    robust : bool
        If True, refit with iteratively reweighted least squares
        (IRLS) using a Tukey biweight on observation residuals. The
        residual scale is estimated from a normal-consistent MAD
        (``1.4826 * median(|r - median(r)|)``) so the only knob is
        ``robust_c`` (the rejection threshold in MAD-units). Tikhonov
        rows are not reweighted -- the prior is fixed. Outer iterations
        stop on relative parameter-norm change below ``robust_tol`` or
        after ``robust_max_iter``. Default ``True``.
    robust_max_iter : int
        Maximum IRLS outer iterations (each is a full LSMR solve).
        Default ``8`` -- Tukey biweight typically converges in 3-5.
    robust_tol : float
        Relative-norm change in the parameter vector below which IRLS
        terminates. Default ``1e-3``.
    robust_c : float
        Tukey biweight tuning constant (in MAD-units). Default
        ``4.685``, which gives 95% asymptotic Gaussian efficiency and
        zeros the weight beyond ~4.685 robust-sigmas.

    Returns
    -------
    params : xarray.Dataset
        Variables:

        - ``tilt_dx``, ``tilt_dy``, ``tilt_dz`` — per-epoch tilt
          coefficients (dims ``(time,)``).
        - ``xref``, ``yref`` — per-epoch centroids used for
          normalization (dims ``(time,)``).
        - ``fit_xy`` — boolean flag, True if slope components were
          fit at that epoch.
        - ``intercept`` — per-pixel intercept :math:`z^0_p`
          (m, elevation at the mean epoch; dims ``(y, x)``).
        - ``dhdt`` — per-pixel trend :math:`\dot h_p` (**m/day**;
          dims ``(y, x)``).
        - ``weight_mean`` — per-epoch mean of the final IRLS Tukey
          weights (dims ``(time,)``). 1.0 when ``robust=False``. Low
          values flag epochs whose static-control observations were
          mostly downweighted by the robust loss.
        - ``weight_frac_kept`` — per-epoch fraction of observations
          with non-zero IRLS weight (dims ``(time,)``).
    stack_corrected : xarray.DataArray
        The input stack with the per-epoch tilt subtracted.
    """
    if "time" not in stack.dims:
        raise ValueError("stack must have a 'time' dimension")
    if stack.sizes["time"] < 2:
        raise ValueError(f"stack must have >= 2 time samples, got {stack.sizes['time']}")

    times = stack["time"].values
    T = len(times)
    ny = stack.sizes["y"]
    nx = stack.sizes["x"]
    n_pix = ny * nx

    # Per-epoch Ez prior. Scalar broadcasts to length T; array must match.
    Ez_arr = np.broadcast_to(np.asarray(Ez, dtype=float), (T,)).astype(float, copy=True)

    # Per-epoch DOF override: force offset-only (alpha_z, no x/y plane) for
    # flagged epochs. Used to demote poorly-coregistered strips (high pc_align
    # end_p50) to a vertical-offset-only fit, where a full tilt plane would be
    # noise-dominated and inject a spurious ramp into dh/dt -- without dropping
    # the strip entirely. Length-T boolean; default all-False (legacy behaviour).
    if offset_only_epochs is None:
        offset_only = np.zeros(T, dtype=bool)
    else:
        offset_only = np.broadcast_to(np.asarray(offset_only_epochs, dtype=bool), (T,)).copy()

    x_coords = stack["x"].values
    y_coords = stack["y"].values

    # Time in days, mean-centered (matches Shean ``ndinterp.py``: ``tn = t - t_ref``).
    t_days = (times - times[0]).astype("timedelta64[s]").astype(float) / 86400.0
    t_centered = t_days - t_days.mean()

    # Control mask on (y, x)
    if control_mask is None:
        static_mask = np.ones((ny, nx), dtype=bool)
    else:
        raw = control_mask.values if isinstance(control_mask, xr.DataArray) else control_mask
        static_mask = np.asarray(raw, dtype=bool)
        if static_mask.shape != (ny, nx):
            raise ValueError(f"control_mask shape {static_mask.shape} != stack (y, x) {(ny, nx)}")

    if observation_mask is None:
        # Legacy behaviour: gate observations by control_mask.
        obs_mask = static_mask
    else:
        raw_o = (
            observation_mask.values if isinstance(observation_mask, xr.DataArray)
            else observation_mask
        )
        obs_mask = np.asarray(raw_o, dtype=bool)
        if obs_mask.shape != (ny, nx):
            raise ValueError(
                f"observation_mask shape {obs_mask.shape} != stack (y, x) {(ny, nx)}"
            )

    # Per-pixel temporal-median reference (Shean ``ndinterp.py``):
    #
    #     test_ref = median(test, axis=0); testn = test - test_ref
    #
    # Subtracted from observations before the LSQ so the per-pixel
    # intercept block carries only the *residual* elevation. Without
    # this step the ``Eint`` Tikhonov prior (10 m by default) acts on
    # absolute elevation, which collapses the intercept toward zero in
    # any domain with non-trivial topographic relief and forces the
    # mean elevation into per-epoch alpha_z and per-pixel dhdt --
    # producing fake hundreds-of-meters tilts on small-footprint
    # epochs and 1000+ m/yr trends on grounded ice.
    z_arr = stack.values
    valid_arr = np.isfinite(z_arr)
    with np.errstate(invalid="ignore"):
        z_ref = np.nanmedian(np.where(valid_arr, z_arr, np.nan), axis=0)
    has_ref = np.isfinite(z_ref)
    # Pixels with no finite obs anywhere: ref undefined; use 0 so the
    # subtraction in the assembly loop is safe (those pixels contribute
    # no LSQ rows). After solving we mark their intercept NaN.
    z_ref_safe = np.where(has_ref, z_ref, 0.0)

    # Total unknowns: n_pix intercepts + n_pix trends + 3*T tilt coefficients
    N = 2 * n_pix + 3 * T

    # Streaming assembly of the sparse design matrix in COO triplets
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    b_obs: list[np.ndarray] = []
    obs_epoch: list[np.ndarray] = []

    xrefs = np.zeros(T)
    yrefs = np.zeros(T)
    fit_tilt_xy = np.zeros(T, dtype=bool)
    # Pixels contributing >=1 observation row (Shean's Aidx key set);
    # the smoothness constraint is built over this support.
    pix_seen = np.zeros((ny, nx), dtype=bool)

    print(f"  Assembling sparse LSQ design matrix ({T} epochs, n_pix={n_pix:,})...", flush=True)
    asm_t0 = time.time()
    asm_print_every = max(1, T // 20)
    row_cursor = 0
    for k in range(T):
        z_k = stack.isel(time=k).values
        valid = np.isfinite(z_k) & obs_mask
        iy, ix = np.where(valid)
        n_obs = iy.size
        if n_obs == 0:
            continue
        pix_seen |= valid

        x_k = x_coords[ix]
        y_k = y_coords[iy]
        xref_k = float(x_k.mean())
        yref_k = float(y_k.mean())
        xrefs[k] = xref_k
        yrefs[k] = yref_k
        x_n = x_k - xref_k
        y_n = y_k - yref_k

        # Spatial-spread check controls whether dx, dy are fit at epoch k.
        # A per-epoch offset_only flag (e.g. poorly-aligned strips) forces the
        # vertical-offset-only fit regardless of spread.
        dist = np.sqrt(x_n**2 + y_n**2)
        dist_ptp = float(np.percentile(dist, 95) - np.percentile(dist, 5))
        fit_xy = (dist_ptp > min_width) and not bool(offset_only[k])
        fit_tilt_xy[k] = fit_xy

        pix_idx = iy * nx + ix
        r = np.arange(row_cursor, row_cursor + n_obs, dtype=np.int64)

        # Intercept column
        rows.append(r)
        cols.append(pix_idx.astype(np.int64))
        vals.append(np.ones(n_obs))

        # Trend column (scaled by mean-centered time in days).
        rows.append(r)
        cols.append((n_pix + pix_idx).astype(np.int64))
        vals.append(np.full(n_obs, t_centered[k]))

        col_base = 2 * n_pix + 3 * k
        if fit_xy:
            rows.append(r)
            cols.append(np.full(n_obs, col_base + 0, dtype=np.int64))
            vals.append(x_n)
            rows.append(r)
            cols.append(np.full(n_obs, col_base + 1, dtype=np.int64))
            vals.append(y_n)
        rows.append(r)
        cols.append(np.full(n_obs, col_base + 2, dtype=np.int64))
        vals.append(np.ones(n_obs))

        b_obs.append(z_k[iy, ix] - z_ref_safe[iy, ix])
        obs_epoch.append(np.full(n_obs, k, dtype=np.int32))
        row_cursor += n_obs
        if (k + 1) % asm_print_every == 0 or k == T - 1:
            elapsed = time.time() - asm_t0
            print(
                f"    assembly: epoch {k+1}/{T}  rows={row_cursor:,}  elapsed={elapsed:.1f}s",
                flush=True,
            )

    if row_cursor == 0:
        raise ValueError("No valid observations across stack; cannot fit tilt.")

    A_rows = np.concatenate(rows)
    A_cols = np.concatenate(cols)
    A_vals = np.concatenate(vals)
    b = np.concatenate(b_obs)
    epoch_of_row = np.concatenate(obs_epoch)
    M = row_cursor

    A = sp.coo_matrix((A_vals, (A_rows, A_cols)), shape=(M, N)).tocsr()

    # Tikhonov regularization: append E^{-1} I rows
    E = np.ones(N)
    E[0:n_pix] = Eint
    E[n_pix : 2 * n_pix] = Edhdt
    E[2 * n_pix + 0 :: 3] = Ex
    E[2 * n_pix + 1 :: 3] = Ey
    E[2 * n_pix + 2 :: 3] = Ez_arr
    rA = sp.diags(1.0 / E, 0, shape=(N, N))

    # Shean ndinterp.py smoothness constraint on the per-pixel trend
    # field: one row per observed pixel with an observed up/down and/or
    # left/right neighbour pair; center +2 per direction pair (so 4 when
    # both), neighbours -1, RHS 0. Rows live in the fixed (non-IRLS)
    # regularization block alongside the Tikhonov diag.
    n_sc = 0
    if dhdt_smoothness:
        w_sc = float(dhdt_smoothness)
        ud = np.zeros((ny, nx), dtype=bool)
        ud[1:-1, :] = pix_seen[1:-1, :] & pix_seen[:-2, :] & pix_seen[2:, :]
        lr = np.zeros((ny, nx), dtype=bool)
        lr[:, 1:-1] = pix_seen[:, 1:-1] & pix_seen[:, :-2] & pix_seen[:, 2:]
        sel = ud | lr
        p_sel = np.flatnonzero(sel.ravel()).astype(np.int64)
        n_sc = p_sel.size
        if n_sc:
            ud_s = ud.ravel()[p_sel]
            lr_s = lr.ravel()[p_sel]
            row_ids = np.arange(n_sc, dtype=np.int64)
            sc_rows = [row_ids]
            sc_cols = [n_pix + p_sel]
            sc_vals = [w_sc * (2.0 * ud_s + 2.0 * lr_s)]
            r_ud = row_ids[ud_s]
            p_ud = p_sel[ud_s]
            sc_rows += [r_ud, r_ud]
            sc_cols += [n_pix + p_ud - nx, n_pix + p_ud + nx]
            sc_vals += [np.full(p_ud.size, -w_sc), np.full(p_ud.size, -w_sc)]
            r_lr = row_ids[lr_s]
            p_lr = p_sel[lr_s]
            sc_rows += [r_lr, r_lr]
            sc_cols += [n_pix + p_lr - 1, n_pix + p_lr + 1]
            sc_vals += [np.full(p_lr.size, -w_sc), np.full(p_lr.size, -w_sc)]
            SC = sp.coo_matrix(
                (np.concatenate(sc_vals),
                 (np.concatenate(sc_rows), np.concatenate(sc_cols))),
                shape=(n_sc, N),
            ).tocsr()
            print(
                f"  dh/dt smoothness constraint: {n_sc:,} rows "
                f"(weight {w_sc:g}, Shean ndinterp L574+)",
                flush=True,
            )

    blocks = [A, rA] if n_sc == 0 else [A, rA, SC]
    A_reg = sp.vstack(blocks).tocsr()
    n_fixed = N + n_sc
    b_reg = np.concatenate([b, np.zeros(n_fixed)])
    print(
        f"  LSQ system built: A_reg shape={A_reg.shape}  nnz={A_reg.nnz:,}  "
        f"obs_rows={M:,}  reg_rows={n_fixed:,}  unknowns={N:,}  "
        f"(2*n_pix={2 * n_pix:,} pixel terms + 3*T={3 * T:,} tilt terms)",
        flush=True,
    )

    # Dispatch the LSQ solve + IRLS loop to GPU when STEREO_MELT_BACKEND=cupy.
    # Sparse assembly above stays on CPU/scipy (cheap; numpy concatenations
    # of COO triplets), but the heavy LSMR + per-iteration matrix reweighting
    # moves to cupyx.scipy.sparse on the device. The "spsolve" path stays
    # on CPU regardless — cupyx's spsolve uses cuSOLVER, which has different
    # numerics than scipy's UMFPACK, and we don't want a silent regression
    # when the user opts into spsolve.
    use_gpu = backend == "cupy" and solver == "lsmr"
    if use_gpu:
        import cupyx.scipy.sparse as _cusp
        import cupyx.scipy.sparse.linalg as _cuspla

        sp_mod = _cusp
        lsmr_fn = _cuspla.lsmr
        # cuSPARSE crashes hard on NaN in matrix data or RHS — fail fast with
        # a clear error rather than letting the driver segfault.
        if not np.all(np.isfinite(A_reg.data)) or not np.all(np.isfinite(b_reg)):
            raise RuntimeError(
                "NaN/Inf in design matrix or RHS; would crash cuSPARSE. "
                "Fix upstream nan-handling."
            )
        # Cast to float32 on the device — LSMR doesn't need float64 precision
        # for tilt parameters (m-scale physical units), and float32 halves
        # GPU memory footprint of the design matrix and IRLS reweighting copies.
        A_dev = _cusp.csr_matrix(A_reg.astype(np.float32))
        A_obs_dev = _cusp.csr_matrix(A.astype(np.float32))
        b_obs_dev = asarray(b.astype(np.float32))
        b_reg_dev = asarray(b_reg.astype(np.float32))
        # Fixed-weight tail = Tikhonov diag + smoothness rows; IRLS never
        # reweights either.
        ones_N_dev = xp.ones(n_fixed, dtype=xp.float32)
    else:
        sp_mod = sp
        lsmr_fn = spla.lsmr
        A_dev = A_reg
        A_obs_dev = A
        b_obs_dev = b
        b_reg_dev = b_reg
        ones_N_dev = np.ones(n_fixed)

    def _solve(A_sys, b_sys, label=""):
        t0 = time.time()
        if solver == "lsmr":
            # Iterative LSQR-style solve directly on the augmented system.
            # Memory scales with nnz(A_sys); avoids the LU fill-in that
            # makes spsolve fail at full-AOI scale (1.7M unknowns).
            res = lsmr_fn(
                A_sys, b_sys,
                atol=lsmr_atol, btol=lsmr_atol,
                maxiter=lsmr_maxiter,
            )
            x_out = res[0]
            # scipy/cupyx lsmr returns (x, istop, itn, normr, normar, ...);
            # surface enough to tell convergence-vs-iter-cap from the log.
            istop, itn, normr = int(res[1]), int(res[2]), float(res[3])
            elapsed = time.time() - t0
            print(
                f"    LSMR {label}: itn={itn} istop={istop} "
                f"normr={normr:.3e} elapsed={elapsed:.1f}s",
                flush=True,
            )
            return x_out
        elif solver == "spsolve":
            # Normal-equations solve via sparse LU. Fast for moderate-size
            # systems; can OOM on the full-AOI joint system. Always CPU
            # (use_gpu==False here by construction).
            AT = A_sys.T
            lhs = (AT @ A_sys).tocsc()
            rhs = AT @ b_sys
            try:
                x_out = spla.spsolve(lhs, rhs, use_umfpack=True)
            except TypeError:
                x_out = spla.spsolve(lhs, rhs)
            print(f"    spsolve {label}: elapsed={time.time() - t0:.1f}s", flush=True)
            return x_out
        else:
            raise ValueError(f"solver must be 'lsmr' or 'spsolve', got {solver!r}")

    # Initial unweighted solve.
    print("  Initial unweighted solve...", flush=True)
    x_lsq = _solve(A_dev, b_reg_dev, label="(initial)")
    weights_obs = xp.ones(M) if use_gpu else np.ones(M)
    n_irls_iter = 0

    if robust:
        # IRLS with Tukey biweight on observation residuals. Tikhonov
        # rows are not reweighted -- the prior std priors are fixed.
        # Scale is a normal-consistent MAD so the only knob is robust_c.
        print(
            f"  IRLS Tukey biweight (c={robust_c}, max_iter={robust_max_iter}, "
            f"tol={robust_tol})",
            flush=True,
        )
        irls_t0 = time.time()
        for it in range(robust_max_iter):
            iter_t0 = time.time()
            r_obs = b_obs_dev - A_obs_dev @ x_lsq
            if use_gpu:
                med = float(xp.median(r_obs))
                mad = float(xp.median(xp.abs(r_obs - med)))
            else:
                med = float(np.median(r_obs))
                mad = float(np.median(np.abs(r_obs - med)))
            scale = 1.4826 * mad
            if scale < 1e-9:
                # Residuals are already vanishing; no further reweighting helps.
                print(
                    f"    IRLS iter {it+1}: scale={scale:.2e} below floor; stopping.",
                    flush=True,
                )
                break
            u = r_obs / (robust_c * scale)
            if use_gpu:
                weights_obs = xp.where(xp.abs(u) < 1.0, (1.0 - u * u) ** 2, xp.asarray(0.0, dtype=xp.float32))
                sqw = xp.sqrt(weights_obs)
                full_w = xp.concatenate([sqw, ones_N_dev]).astype(xp.float32)
                # Row-scale A_dev by full_w directly on its CSR data array.
                # Equivalent to `diags(full_w) @ A_dev` but avoids the heavy
                # cuSPARSE spgemm allocation that OOMs on 11 GB GPUs at the
                # full-AOI joint scale (Nansen 11M cells, Beardmore 19M cells).
                # Map each non-zero index k to its row i via searchsorted on
                # indptr (cupy's `repeat` doesn't accept an ndarray as the
                # `repeats` argument; numpy does, but searchsorted works on
                # both backends and is also lower memory).
                nnz = A_dev.data.size
                row_idx = xp.searchsorted(
                    A_dev.indptr, xp.arange(nnz, dtype=A_dev.indptr.dtype), side="right",
                ) - 1
                scaled_data = A_dev.data * full_w[row_idx]
                A_reg_w = sp_mod.csr_matrix(
                    (scaled_data, A_dev.indices, A_dev.indptr),
                    shape=A_dev.shape,
                )
                b_reg_w = full_w * b_reg_dev
                x_new = _solve(A_reg_w, b_reg_w, label=f"(IRLS iter {it+1})")
                denom = max(float(xp.linalg.norm(x_lsq)), 1e-12)
                rel_change = float(xp.linalg.norm(x_new - x_lsq)) / denom
                mean_w = float(weights_obs.mean())
                kept_frac = float((weights_obs > 0).mean())
                # Free per-iteration scratch before the next IRLS allocation.
                # NB: keep weights_obs — used post-loop by `to_numpy(weights_obs)`.
                del A_reg_w, b_reg_w, full_w, sqw, u, r_obs, row_idx, scaled_data
                xp.get_default_memory_pool().free_all_blocks()
            else:
                weights_obs = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
                sqw = np.sqrt(weights_obs)
                full_w = np.concatenate([sqw, ones_N_dev])
                A_reg_w = (sp.diags(full_w) @ A_reg).tocsr()
                b_reg_w = full_w * b_reg
                x_new = _solve(A_reg_w, b_reg_w, label=f"(IRLS iter {it+1})")
                denom = max(float(np.linalg.norm(x_lsq)), 1e-12)
                rel_change = float(np.linalg.norm(x_new - x_lsq)) / denom
                mean_w = float(weights_obs.mean())
                kept_frac = float((weights_obs > 0).mean())
            iter_elapsed = time.time() - iter_t0
            print(
                f"    IRLS iter {it+1}/{robust_max_iter}: "
                f"scale={scale:.3f}m kept={kept_frac:.3f} mean_w={mean_w:.3f} "
                f"rel_change={rel_change:.2e} iter_elapsed={iter_elapsed:.1f}s",
                flush=True,
            )
            x_lsq = x_new
            n_irls_iter = it + 1
            if rel_change < robust_tol:
                print(
                    f"  IRLS converged (rel_change<{robust_tol}) after {n_irls_iter} iters",
                    flush=True,
                )
                break
        print(
            f"  IRLS total: {n_irls_iter} iters, {time.time() - irls_t0:.1f}s",
            flush=True,
        )

    # Bring solution + IRLS weights back to numpy for the postprocessing
    # block below (cheap numpy reductions; no GPU benefit).
    if use_gpu:
        x_lsq = to_numpy(x_lsq)
        weights_obs = to_numpy(weights_obs)

    tilt_dx = x_lsq[2 * n_pix + 0 :: 3]
    tilt_dy = x_lsq[2 * n_pix + 1 :: 3]
    tilt_dz = x_lsq[2 * n_pix + 2 :: 3]
    intercept = x_lsq[0:n_pix].reshape(ny, nx)
    dhdt = x_lsq[n_pix : 2 * n_pix].reshape(ny, nx)

    # Per-epoch IRLS weight summary -- mean weight and surviving fraction
    # per epoch. With Tukey biweight, weights == 0 for rejected obs.
    weight_mean = np.full(T, np.nan)
    weight_frac_kept = np.full(T, np.nan)
    for k in range(T):
        sel = epoch_of_row == k
        if sel.any():
            weight_mean[k] = float(weights_obs[sel].mean())
            weight_frac_kept[k] = float((weights_obs[sel] > 0.0).mean())

    # Mean-shift gauge fix on the *residual* intercept. The
    # (intercept, alpha_z) pair carries one degenerate global-offset
    # DOF: any constant c can be added to every intercept and
    # subtracted from every alpha_z without changing the data fit. The
    # Tikhonov prior breaks the degeneracy via the relative weights
    # ~ n_pix*Ez^2 vs T*Eint^2; with n_pix >> T this can park a small
    # nonzero mean inside alpha_z. Move that mean back into the
    # intercept so alpha_z sums to zero and the corrected stack
    # preserves input elevations. With the per-pixel reference
    # subtraction above, the residual intercept is ~zero-mean by
    # construction and alpha_z.mean() is typically ~0, so this is
    # essentially a no-op now -- kept for safety.
    tilt_dz_mean = float(tilt_dz.mean())
    tilt_dz = tilt_dz - tilt_dz_mean
    intercept = intercept + tilt_dz_mean

    # Add the per-pixel temporal-median reference back to the residual
    # intercept so the returned intercept is in absolute elevation
    # units (m, geoid-or-ellipsoid-referenced as the input stack was).
    # Pixels with no finite observations across the stack get NaN,
    # which is more honest than the prior-implied zero.
    intercept = np.where(has_ref, intercept + z_ref_safe, np.nan)

    # Reconstruct and subtract the per-epoch tilt
    X, Y = np.meshgrid(x_coords, y_coords)
    tilt_stack = np.zeros((T, ny, nx))
    for k in range(T):
        if fit_tilt_xy[k]:
            tilt_stack[k] = tilt_dx[k] * (X - xrefs[k]) + tilt_dy[k] * (Y - yrefs[k]) + tilt_dz[k]
        else:
            tilt_stack[k] = tilt_dz[k]

    tilt_da = xr.DataArray(
        tilt_stack,
        dims=("time", "y", "x"),
        coords={"time": stack["time"], "y": y_coords, "x": x_coords},
    )
    stack_corrected = stack - tilt_da

    params = xr.Dataset(
        {
            "tilt_dx": (("time",), tilt_dx),
            "tilt_dy": (("time",), tilt_dy),
            "tilt_dz": (("time",), tilt_dz),
            "xref": (("time",), xrefs),
            "yref": (("time",), yrefs),
            "fit_xy": (("time",), fit_tilt_xy),
            "intercept": (("y", "x"), intercept),
            "dhdt": (("y", "x"), dhdt),
            "weight_mean": (("time",), weight_mean),
            "weight_frac_kept": (("time",), weight_frac_kept),
            "Ez_per_epoch": (("time",), Ez_arr),
        },
        coords={"time": stack["time"], "y": y_coords, "x": x_coords},
        attrs={
            "Eint": Eint,
            "Edhdt": Edhdt,
            "Ex": Ex,
            "Ey": Ey,
            "Ez": float(np.mean(Ez_arr)),
            "min_width": min_width,
            "robust": int(robust),
            "robust_c": robust_c,
            "robust_max_iter": robust_max_iter,
            "robust_tol": robust_tol,
            "robust_n_iter": n_irls_iter,
        },
    )

    return params, stack_corrected
