# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Temporal and spatial derivatives for the mass-budget stage.

This module provides the kinematic building blocks consumed by
:func:`stereo_melt.melt.eulerian_melt_rate`:

- :func:`dh_dt` — per-pixel linear regression of a (time, y, x) stack.
- :func:`gradient` — :math:`(\partial/\partial x, \partial/\partial y)`
  using the xarray coord spacing, so non-uniform grids are supported.
- :func:`divergence` — :math:`\partial u/\partial x + \partial v/\partial y`.
- :func:`flux_divergence` — :math:`\nabla\!\cdot(H u)`, the flux term
  in Shean 2019 Eq. 4.

The flux divergence goes through a pluggable :class:`DivergenceEstimator`
protocol so a learned Helmholtz flux surrogate can slot in without
touching the melt-rate solver.

All array arithmetic honors the active backend selected in
:mod:`stereo_melt.backend` (numpy by default, cupy when
``STEREO_MELT_BACKEND=cupy``).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import xarray as xr

from .backend import asarray, to_numpy, xp

__all__ = [
    "common_epoch_mean",
    "dh_dt",
    "gradient",
    "divergence",
    "flux_divergence",
    "gaussian_smooth_nan",
    "clean_temporal_outliers",
    "DivergenceEstimator",
    "FiniteDifferenceDivergence",
    "HelmholtzDivergence",
    "SECONDS_PER_YEAR",
]

SECONDS_PER_YEAR = 86400.0 * 365.25


def dh_dt(
    stack: xr.DataArray,
    min_count: int = 3,
    robust: bool = False,
    robust_c: float = 4.685,
    robust_max_iter: int = 8,
    robust_tol: float = 1e-3,
) -> xr.Dataset:
    r"""Return the per-pixel linear trend of a time-indexed raster stack.

    For each pixel :math:`(y, x)`, fit :math:`z = m\,t + b` to the finite
    samples in ``stack`` and return the slope, intercept, residual RMS,
    and sample count.

    Parameters
    ----------
    stack : xarray.DataArray, dims (time, y, x)
        Values to regress against time. The ``time`` coordinate is
        converted to seconds since the earliest sample.
    min_count : int
        Minimum finite samples per pixel; pixels with fewer samples are
        returned as NaN.
    robust : bool, optional
        If True, refine the OLS fit with iteratively-reweighted-least-
        squares using a Tukey biweight on observation residuals. The
        scale is a basin-wide normal-consistent MAD so per-pixel weights
        remain stable even with sparse temporal coverage. Matches the
        IRLS convention in :func:`stereo_melt.coregister.tilt.fit_tilt_stack`.
    robust_c : float, optional
        Tukey biweight tuning constant; default 4.685 gives 95%
        efficiency at the Gaussian.
    robust_max_iter, robust_tol : optional
        Iteration cap and relative-change stopping criterion on the
        per-pixel slope.

    Returns
    -------
    xarray.Dataset
        Variables on the (y, x) grid:

        - ``slope`` — trend in stack units per second
        - ``intercept`` — y-intercept in stack units
        - ``count`` — number of finite samples per pixel
        - ``rmse`` — residual RMS error per pixel, stack units

    Notes
    -----
    Slopes are reported in units-per-second to make downstream unit
    handling explicit; multiply by :data:`SECONDS_PER_YEAR` for per-year.
    """
    if "time" not in stack.dims:
        raise ValueError("stack must have a 'time' dimension")
    if stack.sizes["time"] < min_count:
        raise ValueError(f"stack has {stack.sizes['time']} time samples; need >= {min_count}")

    # Time coordinate expressed as seconds since the first sample
    t_ns = stack["time"].values.astype("datetime64[ns]").astype("int64")
    t = (t_ns - t_ns.min()).astype("float64") * 1e-9

    # Upload the (time, y, x) cube to the active backend; element-wise
    # ops below run on numpy or cupy depending on STEREO_MELT_BACKEND.
    y = asarray(np.asarray(stack.values, dtype="float64"))
    finite = xp.isfinite(y)
    n = finite.sum(axis=0)

    T = asarray(t[:, None, None])
    mask = finite.astype("float64")
    Tw = T * mask
    Yw = xp.where(finite, y, 0.0)

    sumT = Tw.sum(axis=0)
    sumY = Yw.sum(axis=0)
    sumTT = (Tw * T).sum(axis=0)
    sumTY = (Tw * Yw).sum(axis=0)

    denom = n * sumTT - sumT * sumT
    nan_scalar = xp.asarray(np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = xp.where(denom > 0, (n * sumTY - sumT * sumY) / denom, nan_scalar)
        intercept = xp.where(n > 0, (sumY - slope * sumT) / xp.maximum(n, 1), nan_scalar)
        pred = slope[None, :, :] * T + intercept[None, :, :]
        residuals = xp.where(finite, y - pred, nan_scalar)
        ss_res = xp.nansum(residuals * residuals, axis=0)
        rmse = xp.where(n > 2, xp.sqrt(ss_res / xp.maximum(n - 2, 1)), nan_scalar)

    bad = n < min_count
    slope = xp.where(bad, nan_scalar, slope)
    intercept = xp.where(bad, nan_scalar, intercept)
    rmse = xp.where(bad, nan_scalar, rmse)

    if robust:
        # IRLS with Tukey biweight on observation residuals, basin-wide
        # MAD scale (per-pixel MAD is too noisy with 5-30 obs / pixel).
        # NaN cells are zero-weighted throughout.
        for _ in range(int(robust_max_iter)):
            pred = slope[None, :, :] * T + intercept[None, :, :]
            r_obs = xp.where(finite, y - pred, 0.0)
            abs_r = xp.where(finite, xp.abs(r_obs), xp.asarray(np.nan))
            med = xp.nanmedian(abs_r)
            mad = xp.nanmedian(xp.abs(abs_r - med))
            scale = 1.4826 * float(to_numpy(mad))
            if not np.isfinite(scale) or scale < 1e-9:
                break
            u = r_obs / (robust_c * scale)
            w = xp.where(
                finite & (xp.abs(u) < 1.0),
                (1.0 - u * u) ** 2,
                xp.asarray(0.0),
            )
            sumW = w.sum(axis=0)
            sumWT = (w * T).sum(axis=0)
            sumWY = (w * Yw).sum(axis=0)
            sumWTT = (w * T * T).sum(axis=0)
            sumWTY = (w * T * Yw).sum(axis=0)
            denom_w = sumW * sumWTT - sumWT * sumWT
            with np.errstate(invalid="ignore", divide="ignore"):
                new_slope = xp.where(
                    denom_w > 0,
                    (sumW * sumWTY - sumWT * sumWY) / denom_w,
                    slope,
                )
                new_intercept = xp.where(
                    sumW > 0,
                    (sumWY - new_slope * sumWT) / xp.maximum(sumW, 1e-12),
                    intercept,
                )
            denom_chk = float(to_numpy(xp.nanmax(xp.abs(slope))))
            if denom_chk > 0:
                rel_change = float(to_numpy(
                    xp.nanmax(xp.abs(new_slope - slope))
                )) / denom_chk
            else:
                rel_change = 1.0
            slope = xp.where(bad, nan_scalar, new_slope)
            intercept = xp.where(bad, nan_scalar, new_intercept)
            if rel_change < robust_tol:
                break
        # Recompute rmse with final slope/intercept
        pred = slope[None, :, :] * T + intercept[None, :, :]
        residuals = xp.where(finite, y - pred, nan_scalar)
        ss_res = xp.nansum(residuals * residuals, axis=0)
        rmse = xp.where(n > 2, xp.sqrt(ss_res / xp.maximum(n - 2, 1)), nan_scalar)
        rmse = xp.where(bad, nan_scalar, rmse)

    coords = {k: v for k, v in stack.coords.items() if k != "time"}
    dims = tuple(d for d in stack.dims if d != "time")
    return xr.Dataset(
        {
            "slope": (dims, to_numpy(slope)),
            "intercept": (dims, to_numpy(intercept)),
            "count": (dims, to_numpy(n)),
            "rmse": (dims, to_numpy(rmse)),
        },
        coords=coords,
        attrs={"slope_units": "input_units per second"},
    )


def gradient(
    da: xr.DataArray, dim_x: str = "x", dim_y: str = "y"
) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Return :math:`(\partial/\partial x, \partial/\partial y)` of a field.

    Uses :meth:`xarray.DataArray.differentiate`, which respects the
    coord spacing and therefore handles non-uniform grids.
    """
    return da.differentiate(dim_x), da.differentiate(dim_y)


def divergence(
    vx: xr.DataArray,
    vy: xr.DataArray,
    dim_x: str = "x",
    dim_y: str = "y",
) -> xr.DataArray:
    r"""Return :math:`\partial u/\partial x + \partial v/\partial y`
    of a vector field on xarray coords (central differences)."""
    return vx.differentiate(dim_x) + vy.differentiate(dim_y)


class DivergenceEstimator(Protocol):
    r"""Protocol for flux-divergence estimators.

    Any callable ``(H, vx, vy) -> DataArray`` that returns
    :math:`\nabla\!\cdot(H u)` on the input grid satisfies this
    interface. The default is :class:`FiniteDifferenceDivergence`; a
    learned flux surrogate would implement the same call signature and
    be passed to :func:`flux_divergence` via ``estimator=...``.
    """

    def __call__(self, H: xr.DataArray, vx: xr.DataArray, vy: xr.DataArray) -> xr.DataArray: ...


class FiniteDifferenceDivergence:
    r"""Central-difference flux divergence on xarray coords.

    Computes

    .. math::
        \nabla\!\cdot(H u)
            = \frac{\partial (H v_x)}{\partial x}
            + \frac{\partial (H v_y)}{\partial y}

    Works on non-uniform grids and propagates the active array backend
    transparently through xarray.
    """

    def __call__(self, H: xr.DataArray, vx: xr.DataArray, vy: xr.DataArray) -> xr.DataArray:
        qx = H * vx
        qy = H * vy
        return qx.differentiate("x") + qy.differentiate("y")


class HelmholtzDivergence:
    r"""Mass-consistent flux divergence from a Helmholtz-decomposed flux fit.

    The flux :math:`q = H u` is represented, on the mirror-doubled spectral
    grid (the same construction as the bridging operator), as

    .. math::
        q = J\nabla\psi + \nabla\phi,\qquad
        J\nabla\psi = (-\partial_y\psi,\ \partial_x\psi),

    a solenoidal (divergence-free) part carrying the bulk transport and a
    potential part carrying all of the convergence. Per wavenumber the
    decomposition is the orthogonal projection of :math:`\hat q` onto
    :math:`\hat k` and :math:`\hat k^\perp`, and the divergence is

    .. math::
        \nabla\!\cdot q = \nabla^2\phi \;\;\Longleftrightarrow\;\;
        \widehat{\nabla\!\cdot q}(k) = i\,k\cdot\hat q(k)\,F_\ell(k),

    i.e. the *exact* divergence of a continuous field (Gauss's theorem holds
    on the fitted flux to round-off; the solenoidal part cannot leak into
    melt), with the potential smoothed by a Tikhonov penalty on
    :math:`\nabla^2\phi`, :math:`F_\ell = 1/(1 + (|k|\ell)^4)`. This is
    the FluxNet construction (Bente et al. 2026: :math:`q = J\nabla\psi +
    \nabla\phi` from a coordinate network, divergence by autodiff) with a
    linear spectral representation in place of the network, so the
    smoothness is an explicit length :math:`\ell` instead of an implicit
    bandwidth — and :math:`\ell` is selected by closed-form GCV on the
    longitudinal flux component :math:`\hat k\cdot\hat q`, the only part the
    divergence sees (``ell="gcv"``, the default).

    Why it helps: on a DEM stack the observation-side noise of the budget is
    dominated by :math:`\nabla\!\cdot(\bar H u)` of the white noise in the
    mean thickness — :math:`u\,\delta H/\Delta x \sim 800\times1/200 = 4`
    m/yr per pixel — which finite differences pass straight through.

    Parameters
    ----------
    ell : float or "gcv"
        Smoothing length (m) of the potential; ``"gcv"`` picks it from
        ``ell_grid_px`` (in pixels) by GCV. ``0`` reproduces the spectral
        divergence of the unsmoothed flux.
    ell_grid_px : sequence of float
        Candidate lengths for GCV, in pixels (default 0.5 … 16, log-spaced).
    fill_sigma_px : float
        NaN-aware Gaussian fill scale for gaps before the transform; masked
        cells return NaN.
    """

    def __init__(self, ell="gcv", ell_grid_px=None, fill_sigma_px: float = 2.0):
        self.ell = ell
        self.ell_grid_px = (np.logspace(np.log10(0.5), np.log10(16.0), 16)
                            if ell_grid_px is None else np.asarray(ell_grid_px, float))
        self.fill_sigma_px = float(fill_sigma_px)
        self.last = {}   # diagnostics of the last call

    @staticmethod
    def _fill(a: np.ndarray, sigma: float) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        m = np.isfinite(a)
        if m.all():
            return a
        a0 = np.where(m, a, 0.0)
        num = gaussian_filter(a0, sigma)
        den = gaussian_filter(m.astype(float), sigma)
        filled = np.where(den > 1e-3, num / np.maximum(den, 1e-12), 0.0)
        # far from any data: iterate a wider fill so there is no hard edge
        far = den <= 1e-3
        if far.any():
            num2 = gaussian_filter(a0, 4 * sigma)
            den2 = gaussian_filter(m.astype(float), 4 * sigma)
            filled = np.where(far, num2 / np.maximum(den2, 1e-12), filled)
        return np.where(m, a, filled)

    def __call__(self, H: xr.DataArray, vx: xr.DataArray, vy: xr.DataArray) -> xr.DataArray:
        qx = to_numpy((H * vx).values).astype(float)
        qy = to_numpy((H * vy).values).astype(float)
        valid = np.isfinite(qx) & np.isfinite(qy)
        ny, nx = qx.shape
        x = np.asarray(H.x.values, float)
        y = np.asarray(H.y.values, float)
        dx = float(abs(x[1] - x[0]))
        dy = float(abs(y[1] - y[0]))
        sx = np.sign(x[1] - x[0])
        sy = np.sign(y[1] - y[0])
        fx = self._fill(qx, self.fill_sigma_px)
        fy = self._fill(qy, self.fill_sigma_px)
        # mirror-double so the periodic transform sees an even, seamless field
        px = np.concatenate([fx, fx[::-1]], 0)
        px = np.concatenate([px, px[:, ::-1]], 1)
        py = np.concatenate([fy, fy[::-1]], 0)
        py = np.concatenate([py, py[:, ::-1]], 1)
        kx = 2.0 * np.pi * np.fft.fftfreq(2 * nx, d=dx) * sx   # d/dx in COORDINATE direction
        ky = 2.0 * np.pi * np.fft.fftfreq(2 * ny, d=dy) * sy
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2
        Qx, Qy = np.fft.fft2(px), np.fft.fft2(py)
        div_hat = 1j * (KX * Qx + KY * Qy)            # exact spectral divergence
        Kmag = np.sqrt(K2)
        # longitudinal (potential) flux component: the scalar the divergence sees
        with np.errstate(invalid="ignore", divide="ignore"):
            p_hat = np.where(K2 > 0, (KX * Qx + KY * Qy) / np.where(K2 > 0, Kmag, 1.0), 0.0)
        if self.ell == "gcv":
            N = p_hat.size
            best, ell_m = None, 0.0
            gcv_curve = []
            for ell_px in self.ell_grid_px:
                ell = ell_px * 0.5 * (dx + dy)
                F = 1.0 / (1.0 + (Kmag * ell) ** 4)
                resid = float(np.sum(np.abs((1.0 - F) * p_hat) ** 2))
                dof = float(np.sum(F)) / N
                g = resid / max(1.0 - dof, 1e-9) ** 2
                gcv_curve.append((ell, g))
                if best is None or g < best:
                    best, ell_m = g, ell
            self.last["gcv"] = gcv_curve
        else:
            ell_m = float(self.ell)
        F = 1.0 / (1.0 + (Kmag * ell_m) ** 4) if ell_m > 0 else np.ones_like(K2)
        div = np.real(np.fft.ifft2(div_hat * F))[:ny, :nx]
        self.last.update(ell_m=ell_m, ell_px=ell_m / (0.5 * (dx + dy)))
        out = np.where(valid, div, np.nan)
        return xr.DataArray(asarray(out), dims=H.dims, coords=H.coords,
                            name="flux_divergence",
                            attrs={"estimator": "helmholtz", "ell_m": ell_m})


def flux_divergence(
    H: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    estimator: DivergenceEstimator | None = None,
) -> xr.DataArray:
    r"""Return :math:`\nabla\!\cdot(H u)`, the flux term of Shean 2019 Eq. 4.

    Parameters
    ----------
    H : xarray.DataArray
        Ice-equivalent thickness on a (y, x) grid, meters.
    vx, vy : xarray.DataArray
        Velocity components on the same grid. The output carries the
        same time units as the inputs (typically m yr\ :sup:`-1`).
    estimator : DivergenceEstimator, optional
        Alternative estimator; defaults to
        :class:`FiniteDifferenceDivergence`.

    Returns
    -------
    xarray.DataArray
        Flux divergence on the input grid.
    """
    if estimator is None:
        estimator = FiniteDifferenceDivergence()
    return estimator(H, vx, vy)


def common_epoch_mean(
    stack: xr.DataArray,
    slope: xr.DataArray,
    sigma_px: float = 2.0,
    t0: float | None = None,
) -> xr.DataArray:
    r"""Return the stack mean referred to a single common epoch.

    A repeat-DEM stack samples each pixel at whatever times its strips
    happen to cover, so the plain time-mean is the field evaluated at a
    per-pixel mean epoch :math:`\bar t(x, y)`, not at one instant. Where the
    surface is changing this makes the mean carry a spatially structured
    sampling artifact, :math:`\bar H - H(t_0) \simeq \dot H\,(\bar t - t_0)`,
    whose pattern follows strip footprints rather than the ice. The
    correction

    .. math::
        H(t_0) = \bar H - \mathcal{S}_\sigma[\dot H]\,\bigl(\bar t - t_0\bigr)

    refers every pixel to the same epoch, with the rate field spatially
    smoothed (:func:`gaussian_smooth_nan`, scale ``sigma_px``) so that
    poorly-sampled pixels borrow their neighbourhood's trend instead of
    extrapolating on their own noisy slope — the per-pixel fit alone is
    markedly rougher than the raw mean and over-corrects. Pixels whose own
    slope is undefined (fewer samples than the regression's ``min_count``)
    take the smoothed rate of their finite neighbours; where no neighbour
    lies within the smoother's support the plain mean is kept, so the
    result is finite wherever the plain mean is and the correction never
    reduces coverage. Caveat: that support is scipy's Gaussian truncation
    radius, 4 ``sigma_px`` from the nearest slope-defined pixel, so a
    contiguous block of sub-``min_count`` pixels wider than about
    ``8 * sigma_px`` receives the correction only on its rim while its
    interior keeps the plain mean — a step of :math:`\dot H\,(\bar t -
    t_0)` in the result that a divergence stencil will differentiate. This
    is the time-consistency of an interpolated DEM product (Shean 2019
    builds epoch mosaics for the same reason), at the linear order the
    steady-melt budget already assumes.

    Parameters
    ----------
    stack : xarray.DataArray, dims ``(time, y, x)``
        Field to average, e.g. ice-equivalent thickness.
    slope : xarray.DataArray
        Per-pixel trend in stack units per second, as returned by
        :func:`dh_dt` (``reg["slope"]``).
    sigma_px : float
        Gaussian smoothing scale of the rate field, in pixels.
    t0 : float, optional
        Reference epoch in seconds from the stack's earliest sample;
        defaults to the mean sample time, which is the epoch the plain
        mean of a fully-sampled stack already refers to (so the correction
        is identically zero under uniform sampling, however the epochs are
        spaced).

    Returns
    -------
    xarray.DataArray
        The mean of ``stack`` referred to ``t0``, on the ``(y, x)`` grid.
    """
    t = (stack.time.values - stack.time.values.min()) / np.timedelta64(1, "s")
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(to_numpy(stack.values))
    n = finite.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_bar = np.tensordot(t, finite, axes=(0, 0)) / np.maximum(n, 1)
    t_bar[n == 0] = np.nan
    t_ref = float(np.mean(t)) if t0 is None else float(t0)

    mean = stack.mean("time", skipna=True)
    slope_s = gaussian_smooth_nan(slope, sigma_px, keep_gaps=False)
    lever = xr.DataArray(t_bar - t_ref, dims=("y", "x"),
                         coords={"y": mean.y, "x": mean.x})
    out = mean - (slope_s * lever).fillna(0.0)
    out.attrs.update(common_epoch_s=t_ref, rate_sigma_px=float(sigma_px))
    return out


def gaussian_smooth_nan(
    field: xr.DataArray,
    sigma_px: float,
    keep_gaps: bool = True,
) -> xr.DataArray:
    r"""NaN-aware Gaussian smoothing of a 2-D field.

    Velocity mosaics carry data voids (NaN) that a plain Gaussian filter would
    smear zeros into. This normalizes by the smoothed validity mask
    (Knutsson-Westin style) so smoothing borrows only from finite neighbours,
    then (by default) restores the original NaN footprint. Used to apply
    velocity smoothing (~1-3.5 km) before the flux-divergence
    term, which tames the near-grounding-line :math:`\nabla\!\cdot(H u)`
    overshoot.

    Parameters
    ----------
    field : xarray.DataArray, dims (y, x)
        Field to smooth (e.g. a velocity component in m/yr).
    sigma_px : float
        Gaussian sigma in pixels (``smooth_m / res_m``). Non-positive returns
        ``field`` unchanged.
    keep_gaps : bool
        If True (default) the input's NaN cells stay NaN. If False the
        normalized estimate is also returned inside the gaps, i.e. a gap
        pixel takes the Gaussian-weighted mean of the finite pixels within
        the smoother's support; only cells with no finite neighbour in
        support remain NaN.

    Returns
    -------
    xarray.DataArray
        Smoothed field on the same coords.
    """
    from scipy.ndimage import gaussian_filter

    if sigma_px <= 0:
        return field
    a = np.asarray(field.values, dtype=float)
    finite = np.isfinite(a)
    a0 = np.where(finite, a, 0.0)
    num = gaussian_filter(a0, sigma_px, mode="nearest")
    den = gaussian_filter(finite.astype(float), sigma_px, mode="nearest")
    out = np.where(den > 1e-6, num / den, np.nan)
    if keep_gaps:
        out[~finite] = np.nan
    return xr.DataArray(out, dims=field.dims, coords=field.coords, attrs=field.attrs)


def clean_temporal_outliers(
    stack: xr.DataArray,
    n_sigma: float = 4.0,
    min_count: int = 3,
) -> tuple[xr.DataArray, dict]:
    r"""Per-pixel temporal NMAD blunder rejection on a repeat-DEM stack.

    For each pixel, NaN the time samples that deviate more than
    ``n_sigma * NMAD`` from the per-pixel temporal median, then NaN whole pixels
    left with fewer than ``min_count`` finite samples. This is Shean
    ``make_stack``-style blunder removal: individual bad strip pixels (cloud,
    blunder, mis-registered tile edge) corrupt the dh/dt regression and the
    Lagrangian path residual, inflating melt-rate noise through the ~9.4
    hydrostatic gain. Removing them lets more of the shelf clear a downstream
    melt quality gate.

    Parameters
    ----------
    stack : xarray.DataArray, dims (time, y, x)
        Coregistered surface-elevation stack.
    n_sigma : float
        Rejection threshold in normal-consistent MADs; non-positive disables
        rejection (only the ``min_count`` pixel drop is applied).
    min_count : int
        Minimum finite samples a pixel must retain; pixels below are all-NaN.

    Returns
    -------
    (xarray.DataArray, dict)
        The cleaned stack (same coords) and a stats dict with
        ``n_obs_rejected``, ``n_pix_dropped``, ``n_obs_before``.
    """
    a = np.asarray(stack.values, dtype=float).copy()  # (t, y, x)
    finite0 = np.isfinite(a)
    n_obs_before = int(finite0.sum())
    n_rej = 0
    if n_sigma > 0:
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(np.where(finite0, a, np.nan), axis=0)
            mad = np.nanmedian(np.abs(a - med[None]), axis=0) * 1.4826
            thresh = n_sigma * mad
            bad = finite0 & np.isfinite(thresh[None]) & (mad[None] > 0) & (
                np.abs(a - med[None]) > thresh[None]
            )
        a[bad] = np.nan
        n_rej = int(bad.sum())
    cnt = np.isfinite(a).sum(axis=0)
    drop_pix = cnt < min_count
    a[:, drop_pix] = np.nan
    stats = {
        "n_obs_before": n_obs_before,
        "n_obs_rejected": n_rej,
        "n_pix_dropped": int(drop_pix.sum()),
    }
    return xr.DataArray(a, dims=stack.dims, coords=stack.coords, attrs=stack.attrs), stats
