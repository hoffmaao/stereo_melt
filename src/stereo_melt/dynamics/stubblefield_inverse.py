# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Faithful Stubblefield, Wearing & Meyer (2023) linear non-hydrostatic inverse.

Recovers basal melt rate :math:`m` from the ice-shelf surface-elevation anomaly
:math:`h_{anom}` by directly inverting the linear forward operator
:math:`h \leftarrow m` (Proc. R. Soc. A 479:20230290; reference implementation
``agstub/linear-shelf-melt``). Unlike the hydrostatic inversion (``m`` is a
constant multiple of ``h``), the Stubblefield kernel is wavenumber-dependent: it
*damps* the surface expression of narrow (width :math:`\lesssim H`) channels, so
the inverse *re-sharpens* them -- recovering channel melt that hydrostatic
flotation underestimates -- and it carries the background advection that smears
the surface signature downstream of the melt source.

This is the **direct Fourier inverse** ``m_hat = h_hat / f(k)`` (NOT a
prior-regularized least-squares), faithful to the reference:

- ``R``/``B`` relaxation/buoyancy transfer functions with the ``1/(eps + 1/R)``
  small-k cap (so the k->0 limit is the hydrostatic ``R - B -> 1/4``, not a
  blow-up);
- the hydrostatic DC limit (``h = -2 m`` as ``k->0``), recovered exactly (no
  separate mean-splice);
- 2-D mean-flow advection ``q = i*2*pi*(kx*ax + ky*ay) - gamma`` generalizing the
  reference's 1-D ``2*pi*kx*alpha`` to an arbitrary flow direction
  (Stubblefield Eq. 3.12, advection ``alpha = u_bar t_r / H``, extension
  ``gamma = E t_r``).

A Tikhonov term controls short-wavelength noise amplification on real
(coregistration-limited) surface data.

It is a non-hydrostatic **correction** solver: it recovers the channel-scale melt
structure, a *different quantity* from the mass-budget Eulerian/Lagrangian melt,
and the three are not interchangeable. The recovered amplitude scales with the
assumed viscosity ``eta_bar`` through ``t_r = 2 eta_bar / (rho_i g H)``; the
spatial pattern does not.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.fft import fft2, fftfreq, ifft2
from scipy.ndimage import gaussian_filter

from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness

__all__ = ["stubblefield_inverse_melt_rate", "stubblefield_forward_steady"]

SECONDS_PER_YEAR = 86400.0 * 365.25
G_GRAVITY = 9.81
# Stubblefield's small-k regularization: caps the singular R, B at 1/eps so the
# long-wavelength limit R - B -> 1/4 is attained near k = 1e-3 (params.py).
_EPS = 2.529e-14


def _R(k):
    n = 2.0 * np.pi * k
    return 1.0 / (
        _EPS
        + n
        * (np.exp(4 * n) - 2 * (1 + 2 * n**2) * np.exp(2 * n) + 1)
        / (np.exp(4 * n) + 4 * n * np.exp(2 * n) - 1)
    )


def _B(k):
    n = 2.0 * np.pi * k
    return 1.0 / (
        _EPS
        + n
        * (np.exp(4 * n) - 2 * (1 + 2 * n**2) * np.exp(2 * n) + 1)
        / (2 * (n + 1) * np.exp(3 * n) + 2 * (n - 1) * np.exp(n))
    )


def _forward_kernel(K, KX, KY, delta, alphax, alphay, gamma):
    """Steady forward kernel ``f(k)`` with ``h_hat = f * m_hat`` (Eq. 3.12)."""
    R_, B_ = _R(K), _B(K)
    q = 1j * 2.0 * np.pi * (KX * alphax + KY * alphay) - gamma   # i*k'.alpha - gamma
    c0 = delta * (R_**2 - B_**2) + q * (delta + 1.0) * R_ + q**2
    f0 = -c0 / (delta * B_)
    return 1.0 / (1e-10 * _EPS + f0)


def _grids(ny, nx, dx_nd, dy_nd):
    kx = fftfreq(nx, dx_nd)
    ky = fftfreq(ny, dy_nd)
    kx[0] = 1e-10
    ky[0] = 1e-10
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    return KX, KY, np.sqrt(KX**2 + KY**2)


def stubblefield_forward_steady(m_nd, dx_nd, dy_nd, delta, alphax=0.0, alphay=0.0, gamma=0.0):
    """Non-dimensional steady forward ``h = f(k) m`` (for synthetic validation)."""
    ny, nx = m_nd.shape
    KX, KY, K = _grids(ny, nx, dx_nd, dy_nd)
    f = _forward_kernel(K, KX, KY, delta, alphax, alphay, gamma)
    h_ft = f * fft2(m_nd)
    h_ft[K < 10 * K.min()] = -2.0 * fft2(m_nd)[K < 10 * K.min()]
    return ifft2(h_ft).real


def _steady_inverse_nd(h_nd, dx_nd, dy_nd, delta, alphax, alphay, gamma, tik):
    """Non-dimensional steady inverse ``m = h / f(k)`` (Tikhonov-damped)."""
    ny, nx = h_nd.shape
    KX, KY, K = _grids(ny, nx, dx_nd, dy_nd)
    f = _forward_kernel(K, KX, KY, delta, alphax, alphay, gamma)
    h_ft = fft2(h_nd)
    if tik > 0:
        m_ft = h_ft * (np.conj(f) / (np.abs(f) ** 2 + tik))
    else:
        m_ft = h_ft / f
    m_ft[K < 10 * K.min()] = -h_ft[K < 10 * K.min()] / 2.0   # hydrostatic DC limit
    return ifft2(m_ft).real


def _nan_gauss(a, sigma):
    """NaN-aware Gaussian smoothing (smooth value / smooth mask)."""
    m = np.isfinite(a)
    a0 = np.where(m, a, 0.0)
    return gaussian_filter(a0, sigma) / np.maximum(gaussian_filter(m.astype(float), sigma), 1e-6)


def _tukey2d(ny, nx, frac):
    from scipy.signal.windows import tukey

    return np.outer(tukey(ny, frac), tukey(nx, frac))


def stubblefield_inverse_melt_rate(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    floating_mask: xr.DataArray | None = None,
    d: xr.DataArray | float = 0.0,
    eta_bar: float = 1e13,
    gamma: float = 0.0,
    sigma_hp_H: float = 5.0,
    tik: float = 1e-2,
    rho_w: float = rhow,
    rho_i: float = rhoi,
    g: float = G_GRAVITY,
    taper_frac: float = 0.25,
) -> xr.Dataset:
    r"""Basal melt rate from the surface-elevation anomaly (Stubblefield inverse).

    Pipeline: strip-robust time-**median** surface -> high-pass anomaly
    (remove the ``> sigma_hp_H * H_ref`` regional shape) -> NaN-fill + Tukey taper
    (FFT conditioning) -> non-dimensionalize by ``H_ref`` and
    ``t_r = 2 eta_bar/(rho_i g H_ref)`` -> faithful steady inverse with 2-D
    advection ``alpha = u_bar t_r / H_ref`` and Tikhonov ``tik`` -> dimensionalize
    ``m = m_nd H_ref / t_r`` -> Shean sign (negative = melt).

    Parameters
    ----------
    h_stack : xarray.DataArray ``(time, y, x)``
        Geoid-referenced, corrected surface-elevation stack, meters.
    vx, vy : xarray.DataArray
        Column-averaged velocity (m/yr); the time/shelf mean sets the advection
        direction. A ``time`` dimension is averaged.
    floating_mask : xarray.DataArray, optional
        Floating-ice mask; the inversion is reported only there.
    d : firn air content (m), for the reference thickness.
    eta_bar : float
        Newtonian viscosity (Pa s). Sets ``t_r`` and hence the recovered
        amplitude (``~1/eta_bar``); the pattern is independent of it. ``1e13`` is
        appropriate for warm/fast shelf ice (Stubblefield uses ``1e14``).
    gamma : float
        Dimensionless extensional thinning ``E t_r`` (default 0).
    sigma_hp_H : float
        High-pass scale in units of ``H_ref`` (default 5 ~ channel-scale).
    tik : float
        Tikhonov regularization for short-wavelength noise (default 1e-2).

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` (m ice/yr, Shean sign), plus ``H_ref_m``, ``t_r_yr``,
        ``alpha_x``/``alpha_y``, ``eta_bar``, ``gamma``, ``sigma_hp_H``, ``tik``.
    """
    delta = rho_w / rho_i - 1.0

    # strip-robust surface (median over time, not mean)
    h_med = h_stack.median("time", skipna=True)
    if floating_mask is not None:
        h_med = h_med.where(floating_mask)
    res = float(abs(h_stack["x"].values[1] - h_stack["x"].values[0]))

    fl = np.isfinite(h_med.values)
    if floating_mask is not None:
        fl &= floating_mask.values.astype(bool)
    if not fl.any():
        raise ValueError("no finite floating-shelf surface for the Stubblefield inverse")

    H_ref = float(freeboard_to_thickness(float(np.nanmean(h_med.values[fl])), d=0.0,
                                         rho_w=rho_w, rho_i=rho_i))
    if isinstance(d, xr.DataArray):
        H_ref = float(np.nanmean(
            freeboard_to_thickness(h_med, d=d, rho_w=rho_w, rho_i=rho_i).values[fl]))
    if not np.isfinite(H_ref) or H_ref <= 0:
        raise ValueError(f"derived H_ref={H_ref} is not positive finite")
    t_r_yr = (2.0 * eta_bar / (rho_i * g * H_ref)) / SECONDS_PER_YEAR

    # advection from the mean flow
    vxm = vx.mean("time") if "time" in vx.dims else vx
    vym = vy.mean("time") if "time" in vy.dims else vy
    u0x = float(vxm.where(xr.DataArray(fl, dims=h_med.dims, coords=h_med.coords)).mean(skipna=True))
    u0y = float(vym.where(xr.DataArray(fl, dims=h_med.dims, coords=h_med.coords)).mean(skipna=True))
    u0x = 0.0 if not np.isfinite(u0x) else u0x
    u0y = 0.0 if not np.isfinite(u0y) else u0y
    alphax = u0x * t_r_yr / H_ref
    alphay = u0y * t_r_yr / H_ref

    # crop to the floating bbox, build h_anom, condition for the FFT
    ys, xs = np.where(fl)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    hm = h_med.values[y0:y1, x0:x1]
    h_anom = hm - _nan_gauss(hm, sigma_hp_H * H_ref / res)
    h_anom = np.where(np.isfinite(h_anom), h_anom, 0.0)
    h_anom = h_anom * _tukey2d(*h_anom.shape, taper_frac)

    m_nd = _steady_inverse_nd(h_anom / H_ref, res / H_ref, res / H_ref,
                              delta, alphax, alphay, gamma, tik)
    # dimensionalize; Shean sign convention (Stubblefield m>0 = melt -> negate)
    m_dim = -(m_nd * H_ref / t_r_yr)

    m_full = np.full(h_med.shape, np.nan, dtype=np.float64)
    m_full[y0:y1, x0:x1] = m_dim
    m_full[~fl] = np.nan
    melt = xr.DataArray(m_full, dims=h_med.dims, coords=h_med.coords, name="melt_rate")

    return xr.Dataset(
        {"melt_rate": melt},
        attrs={
            "equation": "Stubblefield 2023 linear non-hydrostatic inverse, steady: m = h_anom / f(k)",
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "method": "direct Fourier inverse of the forward kernel (faithful, not a prior LSQ)",
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref, "t_r_yr": t_r_yr,
            "alpha_x": alphax, "alpha_y": alphay,
            "gamma": gamma, "sigma_hp_H": sigma_hp_H, "tik": tik,
            # aliases so this Dataset slots into run_melt's linv plot/save plumbing
            "gamma_dimless": gamma, "tr_yr": t_r_yr,
            "note": "non-hydrostatic CHANNEL correction; amplitude ~1/eta_bar, pattern independent",
        },
    )
