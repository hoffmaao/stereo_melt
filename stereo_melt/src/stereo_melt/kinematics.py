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

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import xarray as xr

from .backend import asarray, map_coordinates, to_numpy, xp

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
    "backward_advect_pixels",
    "build_streamline_dataset",
    "StreamlineDataset",
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


# ----------------------------------------------------------------------
# Streamline-frame trajectory primitives (consumed by streamline_pinn)
# ----------------------------------------------------------------------


@dataclass
class StreamlineDataset:
    r"""Per-observation training dataset for the streamline-frame PINN.

    Each row is one freeboard observation with its Lagrangian label
    :math:`(x_{gl}, y_{gl})` — the grounding-line crossing point — and
    its age :math:`\tau = t_{obs} - t_{gl}` since GL entry. Eulerian
    forcings are sampled at the observation pixel for the data-only
    loss formulation (see :func:`build_streamline_dataset`).

    ``strip_idx`` is the epoch index of the observation, used by the
    PINN's per-strip h offset embedding to absorb DC coregistration
    bias without polluting the recovered melt rate.
    """

    x_gl: np.ndarray
    y_gl: np.ndarray
    tau: np.ndarray
    h_obs: np.ndarray
    sigma: np.ndarray
    a_dot: np.ndarray
    vdiv: np.ndarray
    d_fac: np.ndarray
    x_obs: np.ndarray
    y_obs: np.ndarray
    strip_idx: np.ndarray
    n_strips: int
    rho_w: float
    rho_i: float

    def __len__(self) -> int:
        return self.h_obs.size


def backward_advect_pixels(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    grounded_mask: np.ndarray,
    floating_mask: np.ndarray,
    dt_yr: float = 0.05,
    max_tau_yr: float = 30.0,
) -> dict[str, np.ndarray]:
    r"""Backward-integrate every floating-ice pixel to its GL crossing.

    For each pixel currently floating, integrates the negated velocity
    field with forward-Euler substeps of ``dt_yr`` until the particle
    enters the grounded-ice mask (the GL crossing in Lagrangian time).
    Same numerical scheme as :func:`stereo_melt.melt.lagrangian_melt_rate`
    but stepping backward in time and terminating on a mask transition
    instead of running for a fixed pair-duration.

    Velocity is treated as time-stationary (column-mean from MEaSUREs /
    ITS_LIVE).

    Parameters
    ----------
    x_coords, y_coords : ndarray
        1-D EPSG:3031 coords. ``x`` ascending, ``y`` descending (north-up).
    vx, vy : ndarray, shape (Y, X)
        Time-mean column-averaged velocity in m / yr.
    grounded_mask : ndarray, shape (Y, X)
        Boolean — True where ice is grounded. Backward integration
        terminates the first substep a particle samples True here.
    floating_mask : ndarray, shape (Y, X)
        Boolean — True where ice is floating. Only floating pixels seed
        backward trajectories.
    dt_yr : float
        Integration substep, years. Default 0.05 matches
        :func:`lagrangian_melt_rate`.
    max_tau_yr : float
        Hard cap on trajectory age. Particles that don't reach the GL
        in ``max_tau_yr`` are flagged as failed (reason=2).

    Returns
    -------
    dict with arrays of shape (Y, X):

        tau : ndarray, float
            Years since GL crossing. NaN where the trajectory failed.
        x_gl, y_gl : ndarray, float
            GL crossing coordinates, meters. NaN on failure.
        reason : ndarray, int8
            0 success, 1 left domain, 2 hit max_tau, 3 seed not floating.
    """
    if vx.shape != vy.shape or vx.shape != grounded_mask.shape:
        raise ValueError("vx, vy, grounded_mask must share shape (Y, X)")
    if floating_mask.shape != grounded_mask.shape:
        raise ValueError("floating_mask must match grounded_mask shape")

    ny, nx = vx.shape
    res_x = float(x_coords[1] - x_coords[0])
    res_y = float(y_coords[0] - y_coords[1])

    vx_b = asarray(np.asarray(vx, dtype=np.float64))
    vy_b = asarray(np.asarray(vy, dtype=np.float64))
    grounded_b = asarray(np.asarray(grounded_mask, dtype=np.float64))

    y_seed_2d, x_seed_2d = xp.mgrid[0:ny, 0:nx]
    y_idx = y_seed_2d.ravel().astype(xp.float64)
    x_idx = x_seed_2d.ravel().astype(xp.float64)
    seed_floating = asarray(np.asarray(floating_mask.ravel(), dtype=bool))

    n = x_idx.size
    in_flight = seed_floating.copy()
    tau = xp.zeros(n, dtype=xp.float64)
    x_gl_idx = xp.full(n, xp.nan, dtype=xp.float64)
    y_gl_idx = xp.full(n, xp.nan, dtype=xp.float64)
    reason = xp.where(in_flight, xp.int8(0), xp.int8(3))

    n_steps = int(np.ceil(max_tau_yr / dt_yr))

    for _ in range(n_steps):
        if not bool(in_flight.any()):
            break
        # Sample velocity at current position (only for in-flight particles
        # — we still pay the kernel cost on all rows since map_coordinates
        # is vectorized; cheaper than re-gathering).
        vxs = map_coordinates(vx_b, [y_idx, x_idx], order=1, mode="nearest")
        vys = map_coordinates(vy_b, [y_idx, x_idx], order=1, mode="nearest")

        # Backward step: index-space increments are negated relative to
        # the forward path in lagrangian_melt_rate.
        dx_idx = -vxs * dt_yr / res_x
        dy_idx = vys * dt_yr / res_y
        x_new = x_idx + dx_idx
        y_new = y_idx + dy_idx

        oob = (x_new < 0) | (x_new > nx - 1) | (y_new < 0) | (y_new > ny - 1)
        # Sample grounded mask at the NEW position
        x_clip = xp.clip(x_new, 0.0, nx - 1)
        y_clip = xp.clip(y_new, 0.0, ny - 1)
        gm = map_coordinates(grounded_b, [y_clip, x_clip], order=0, mode="nearest")
        crossed = (gm > 0.5) & in_flight & (~oob)

        # Record GL crossing
        x_gl_idx = xp.where(crossed, x_new, x_gl_idx)
        y_gl_idx = xp.where(crossed, y_new, y_gl_idx)
        tau = xp.where(in_flight & ~crossed & ~oob, tau + dt_yr, tau)
        # Terminate crossed and OOB particles
        reason = xp.where(crossed, xp.int8(0), reason)
        reason = xp.where(oob & in_flight, xp.int8(1), reason)
        in_flight = in_flight & ~crossed & ~oob

        # Advance positions for still-in-flight particles only; freeze
        # the others to avoid sampling outside the domain.
        x_idx = xp.where(in_flight, x_new, x_idx)
        y_idx = xp.where(in_flight, y_new, y_idx)

    # Anything still in flight hit max_tau
    reason = xp.where(in_flight, xp.int8(2), reason)

    # Convert index-space GL position back to meters
    x_gl_idx_np = to_numpy(x_gl_idx)
    y_gl_idx_np = to_numpy(y_gl_idx)
    tau_np = to_numpy(tau)
    reason_np = to_numpy(reason)
    success = reason_np == 0
    tau_np = np.where(success, tau_np, np.nan)

    x_gl_m = np.full_like(x_gl_idx_np, np.nan)
    y_gl_m = np.full_like(y_gl_idx_np, np.nan)
    x_gl_m[success] = x_coords[0] + x_gl_idx_np[success] * res_x
    # y axis is descending: y_coord = y[0] - idx*res_y
    y_gl_m[success] = y_coords[0] - y_gl_idx_np[success] * res_y

    return {
        "tau": tau_np.reshape(ny, nx),
        "x_gl": x_gl_m.reshape(ny, nx),
        "y_gl": y_gl_m.reshape(ny, nx),
        "reason": reason_np.reshape(ny, nx),
    }


def build_streamline_dataset(
    h_stack: xr.DataArray,
    trajectories: dict[str, np.ndarray],
    sigma_per_epoch: np.ndarray,
    a_dot: xr.DataArray,
    vdiv: xr.DataArray,
    d_fac: xr.DataArray,
    rho_w: float = 1027.0,
    rho_i: float = 918.0,
    sigma_floor_m: float = 0.1,
    use_per_pixel_mad: bool = True,
) -> StreamlineDataset:
    r"""Assemble per-observation training rows for the streamline PINN.

    For each pixel with a successful GL trajectory and every epoch with
    a finite freeboard observation there, emit one row carrying:
    Lagrangian coords ``(x_gl, y_gl, tau)``, observed freeboard
    ``h_obs`` with its noise ``sigma``, and Eulerian forcings
    (``a_dot``, ``vdiv``, ``d_fac``) sampled at the observation pixel.

    Time enters only through the noise level: identical
    ``(x_gl, y_gl, tau)`` from different epochs produce multiple
    targets at the same streamline coordinate, which the PINN's
    h-only loss reduces to a per-coord weighted mean (the v1
    stationary assumption).

    Noise model
    -----------
    When ``use_per_pixel_mad=True`` (default) the per-row ``sigma`` is

    .. math::

        \sigma = \sqrt{\mathrm{MAD}_{\mathrm{pix}}^2 + \sigma_{\mathrm{epoch}}^2}

    where :math:`\mathrm{MAD}_{\mathrm{pix}}` is the per-pixel temporal
    median-absolute-deviation of ``h_stack`` (scaled by 1.4826 for
    Gaussian equivalence). This adapts the noise floor to the actual
    coregistration residual at each pixel rather than assuming a
    uniform scalar. Set ``use_per_pixel_mad=False`` to fall back to
    the old per-epoch-only behaviour.

    Parameters
    ----------
    h_stack : DataArray (time, y, x)
        Tilt-corrected freeboard stack, meters geoid-referenced.
    trajectories : dict
        Output of :func:`backward_advect_pixels`.
    sigma_per_epoch : ndarray (T,)
        Per-epoch GCP-source noise (e.g. ``EZ_PER_SOURCE_M`` lookup),
        meters of freeboard.
    a_dot, vdiv, d_fac : DataArray (y, x)
        Eulerian forcing fields. ``a_dot`` = SMB (m ice / yr),
        ``vdiv`` = ∇·u (1/yr), ``d_fac`` = firn air content (m).
    sigma_floor_m : float
        Minimum per-pixel MAD, meters. Single-epoch pixels and pixels
        whose stack happens to be quiet should not get unrealistically
        small ``sigma``. Default 0.1 m matches the IS2 noise floor.
    use_per_pixel_mad : bool
        Compute and combine per-pixel temporal MAD. Default True.
    """
    tau = trajectories["tau"]
    x_gl = trajectories["x_gl"]
    y_gl = trajectories["y_gl"]
    valid_pixel = np.isfinite(tau)

    h_arr = h_stack.values.astype(np.float64, copy=False)  # (T, Y, X)
    if h_arr.shape[1:] != tau.shape:
        raise ValueError(
            f"h_stack (y, x) {h_arr.shape[1:]} != trajectory grid {tau.shape}"
        )
    n_strips = int(h_arr.shape[0])
    if sigma_per_epoch.shape != (n_strips,):
        raise ValueError(
            f"sigma_per_epoch length {sigma_per_epoch.shape} != T={n_strips}"
        )

    a_dot_arr = a_dot.broadcast_like(h_stack.isel(time=0)).values.astype(np.float64)
    vdiv_arr = vdiv.broadcast_like(h_stack.isel(time=0)).values.astype(np.float64)
    d_arr = d_fac.broadcast_like(h_stack.isel(time=0)).values.astype(np.float64)

    finite_h = np.isfinite(h_arr)
    valid = finite_h & valid_pixel[None, :, :]  # (T, Y, X)

    # Per-pixel temporal MAD as the noise floor at each pixel — picks up
    # coregistration residuals that the per-epoch GCP sigma misses.
    # All-NaN-column warnings are expected over masked-out pixels; the
    # downstream `np.where(isfinite)` already clamps those to the floor.
    if use_per_pixel_mad:
        import warnings
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message="All-NaN slice encountered")
            med = np.nanmedian(h_arr, axis=0, keepdims=True)
            mad = np.nanmedian(np.abs(h_arr - med), axis=0)
        sigma_pix = 1.4826 * mad  # Gaussian-equivalent std
        sigma_pix = np.where(np.isfinite(sigma_pix), sigma_pix, sigma_floor_m)
        sigma_pix = np.maximum(sigma_pix, sigma_floor_m)
    else:
        sigma_pix = np.full(h_arr.shape[1:], 0.0, dtype=np.float64)

    # Eulerian observation coords broadcast to (T, Y, X)
    x_obs_2d, y_obs_2d = np.meshgrid(h_stack["x"].values, h_stack["y"].values)

    # Flatten
    t_idx, y_idx, x_idx = np.where(valid)
    if t_idx.size == 0:
        raise RuntimeError("No valid (pixel, epoch) pairs — check masks/inputs.")

    sigma_combined = np.sqrt(
        sigma_pix[y_idx, x_idx] ** 2 + sigma_per_epoch[t_idx] ** 2
    )

    return StreamlineDataset(
        x_gl=x_gl[y_idx, x_idx],
        y_gl=y_gl[y_idx, x_idx],
        tau=tau[y_idx, x_idx],
        h_obs=h_arr[t_idx, y_idx, x_idx],
        sigma=sigma_combined,
        a_dot=a_dot_arr[y_idx, x_idx],
        vdiv=vdiv_arr[y_idx, x_idx],
        d_fac=d_arr[y_idx, x_idx],
        x_obs=x_obs_2d[y_idx, x_idx],
        y_obs=y_obs_2d[y_idx, x_idx],
        strip_idx=t_idx.astype(np.int64),
        n_strips=n_strips,
        rho_w=rho_w,
        rho_i=rho_i,
    )


def common_epoch_mean(
    stack: xr.DataArray,
    slope: xr.DataArray,
    sigma_px: float = 2.0,
    t0: float | None = None,
) -> xr.DataArray:
    r"""Return the stack mean referred to a single common epoch.

    A repeat-DEM stack samples each pixel at whatever times its strips
    happen to cover, so the plain time-mean is the field evaluated at a
    per-pixel mean epoch :math:`ar t(x, y)`, not at one instant. Where the
    surface is changing this makes the mean carry a spatially structured
    sampling artifact, :math:`ar H - H(t_0) \simeq \dot H\,(ar t - t_0)`,
    whose pattern follows strip footprints rather than the ice. The
    correction

    .. math::
        H(t_0) = ar H - \mathcal{S}_\sigma[\dot H]\,igl(ar t - t_0igr)

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
    Shean-style velocity smoothing (~1-3.5 km) before the flux-divergence
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
