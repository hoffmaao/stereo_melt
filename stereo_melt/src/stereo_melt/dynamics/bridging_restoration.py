# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Bounded bridging-transfer restoration for hydrostatic melt inversions.

The hydrostatic (Eulerian / budget-Eulerian) inverses read the surface as
freeboard-scaled thickness, so their recovered melt inherits the ice shelf's
bridging transfer: for melt structure varying **along flow** at wavenumber
:math:`k` the recovered field is

.. math::
    \hat m_{\rm rec}(\mathbf k) \;\approx\; T(\mathbf k)\,\hat m(\mathbf k),
    \qquad
    T = \frac{G_h}{f_b\,(G_h - G_s)},

with :math:`G_h, G_s` the steady Stubblefield surface/basal Green's
functions (advected, :math:`\alpha \ne 0`) and :math:`f_b = 1-\rho_i/\rho_w`
the freeboard factor — melt narrower than :math:`\sim 6H` along flow comes
back too weak, displaced downstream, ringed by a fake accretion halo.
Across-flow-uniform structure (ridges along the flow direction) is recovered
**near-exactly** by the budget inverses — the melt signal travels through the
advective term :math:`u\,\partial H/\partial x`, whose secondary-flow
response the surface transfer does not describe (E2a ``cosy`` twin: recovery
1.07 where the surface transfer alone predicts 0.63–0.70).

This module applies the corresponding **restoration filter**

.. math::
    F(\mathbf k) = 1 + w(\mathbf k)\,\bigl(T^{-1}(\mathbf k) - 1\bigr),
    \qquad
    w = \frac{(\mathbf k\cdot\hat u)^2}{|\mathbf k|^2},

i.e. the full complex inverse transfer for along-flow wavenumbers
(amplitude lift **and** upstream phase restoration), smoothly reduced to the
identity for across-flow wavenumbers where the budget inverse needs no help.
The flow-projection weight :math:`w` is anchored by the two measured axes
(E1b ``cosx``: lift :math:`1/T`; E2a ``cosy``: lift 1); oblique wavenumbers
are interpolated and remain to be validated against 3-D truth.

Calibration
-----------

Twin-experiment validation (E1b full-Stokes flowline vs this kernel,
2026-07-19; see ``literature/stubblefield_applicability_prefactor.md``)
shows the depth-averaged theory overweights the advection parameter
:math:`\alpha = \bar u\,t_r/H` by :math:`\sim 3\times`: a single effective
viscosity :math:`\bar\eta_{\rm eff} = \bar\eta/3` (equivalently
``alpha_scale = 0.34``) reproduces the measured transfer in amplitude and
phase at :math:`\lambda/H = 3, 4, 6`. ``alpha_scale`` defaults to that
calibrated value; pass ``1.0`` for the uncalibrated theory.

Noise control, in three layers: the lift is **band-limited** — a
Butterworth roll-off closes the correction below ``band_lam_min``
(default :math:`2.5H`, the shortest wavelength with twin-experiment
calibration; below it the DEM signal is noise-dominated and
:math:`1/T \to \infty`) — the modulus of :math:`F` is additionally capped
at ``lift_cap``, and the DC mode is always passed through untouched (the
budget channel's spatial mean is exact and must stay authoritative). The
field is mirror-padded before the FFT so non-periodic tile edges don't
wrap through the filter.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..backend import asarray, to_numpy, xp
from ..constants import rhoi, rhow
from .linear_perturbation import (
    G_GRAVITY,
    SECONDS_PER_YEAR,
    LinearPerturbation,
    _wavenumber_grids,
)

__all__ = [
    "bridging_restoration",
    "bridging_restoration_filter",
    "bridging_inverse",
    "bridging_inverse_filter",
]

_TINY = 1e-30


def _mean_component(v):
    """Tile-mean velocity component (m/yr); 0 if no finite cells."""
    if isinstance(v, xr.DataArray):
        vals = v.values
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
    return float(v)


def _bridging_transfer(ny, nx, dx, dy, H, ux_myr, uy_myr, eta_bar, alpha_scale,
                       rho_i, rho_w, g, gamma, theta):
    r"""Raw advected bridging transfer ``T(k) = G_h/(f_b (G_h - G_s))``.

    Returns ``(T, kx, ky, fb, u_mag)`` on the ``(ny, nx)`` FFT wavenumber grid.
    ``T`` carries no anisotropy weight and no cap — the shared Stubblefield
    surface transfer behind both the restoration post-filter and the regularized
    inverse. ``alpha_{x,y} = alpha_scale * u_{x,y} * t_r / H`` (E1b calibration).
    """
    fb = 1.0 - rho_i / rho_w
    u_mag = float(np.hypot(ux_myr, uy_myr))
    model = LinearPerturbation(H=H, eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w,
                               g=g, gamma=gamma, theta=theta)
    a_fac = alpha_scale * model.tr / (H * SECONDS_PER_YEAR)
    model.alpha = float(ux_myr) * a_fac
    model.alpha_y = float(uy_myr) * a_fac
    kx, ky = _wavenumber_grids(nx, ny, dx, dy)
    G_h, G_s = model.steady_state_kernel(kx, ky)
    dH_resp = G_h - G_s
    dH_safe = xp.where(xp.abs(dH_resp) > _TINY, dH_resp, xp.asarray(_TINY + 0j))
    T = G_h / (fb * dH_safe)
    return T, kx, ky, fb, u_mag


def bridging_restoration_filter(
    ny: int,
    nx: int,
    dx: float,
    dy: float,
    H: float,
    ux_myr: float,
    uy_myr: float,
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    lift_cap: float = 6.0,
    band_lam_min: float | None = None,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    gamma: float = 0.0,
    theta: float = 1e-14,
):
    r"""Return the complex restoration filter ``F`` and model transfer ``T``.

    Both on the ``(ny, nx)`` FFT wavenumber grid (backend arrays).
    ``ux_myr, uy_myr`` are the tile-mean velocity components in m/yr; the
    advection parameters are :math:`\alpha_{x,y} = \texttt{alpha\_scale}
    \cdot u_{x,y}\,t_r/H`. ``F`` is Hermitian (real output fields) and
    equals 1 at ``k = 0``. ``band_lam_min`` (default ``2.5*H``) closes the
    lift smoothly below that wavelength (order-8 Butterworth in |k|;
    half-power at the band edge) so out-of-band noise is not deconvolved.
    """
    if band_lam_min is None:
        band_lam_min = 2.5 * H
    T, kx, ky, fb, u_mag = _bridging_transfer(
        ny, nx, dx, dy, H, ux_myr, uy_myr, eta_bar, alpha_scale,
        rho_i, rho_w, g, gamma, theta)

    # Flow-projection anisotropy weight: 1 for along-flow wavenumbers,
    # 0 for across-flow (budget dynamics recovers those without help).
    kmag2 = kx**2 + ky**2
    if u_mag > 0:
        k_dot_u = (kx * float(ux_myr) + ky * float(uy_myr)) / u_mag
        w = xp.where(kmag2 > 0, k_dot_u**2 / xp.where(kmag2 > 0, kmag2, 1.0), 0.0)
    else:
        w = xp.zeros_like(kmag2)

    T_safe = xp.where(xp.abs(T) > _TINY, T, xp.asarray(_TINY + 0j))
    # Band limit: only lift wavelengths the calibration covers; the
    # deconvolution grows without bound as |k| increases and would
    # otherwise amplify out-of-band noise straight to the cap.
    k_band = 2.0 * np.pi / float(band_lam_min)
    band = 1.0 / (1.0 + (xp.sqrt(kmag2) / k_band) ** 16)
    F = 1.0 + w * band * (1.0 / T_safe - 1.0)
    # Identity at DC and wherever the transfer model is degenerate.
    bad = (kmag2 <= 0) | ~xp.isfinite(F)
    F = xp.where(bad, xp.asarray(1.0 + 0j), F)
    # Modulus cap, phase-preserving.
    Fabs = xp.abs(F)
    over = Fabs > lift_cap
    F = xp.where(over, F * (lift_cap / xp.where(over, Fabs, 1.0)), F)
    return F, T


def bridging_restoration(
    m: xr.DataArray,
    vx: xr.DataArray | float,
    vy: xr.DataArray | float,
    H: float,
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    lift_cap: float = 6.0,
    band_lam_min: float | None = None,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    gamma: float = 0.0,
    theta: float = 1e-14,
) -> xr.DataArray:
    r"""Restore bridging-damped along-flow structure in a recovered melt field.

    Parameters
    ----------
    m : xarray.DataArray, dims ``(y, x)``
        Recovered melt rate from a hydrostatic inverse (Eulerian or
        budget-Eulerian), m ice yr\ :sup:`-1`, Shean sign. NaNs allowed
        (filled with the finite mean for the FFT, masked back after).
    vx, vy : xarray.DataArray or float
        Background velocity (m/yr). Fields are reduced to their finite
        means — the filter assumes tile-uniform flow, same as the kernel.
    H : float
        Reference ice thickness (m) for the tile.
    eta_bar : float
        Column-average viscosity (Pa s). On a real shelf, take the
        linearized effective viscosity from a hardness inversion,
        :math:`\bar\eta = \tfrac12 B\,\dot\varepsilon_e^{1/n - 1}`.
    alpha_scale : float
        Structural calibration on the advection parameter (see module
        docstring). Default 0.34 (E1b-calibrated); 1.0 = raw theory.
    lift_cap : float
        Maximum amplitude lift (modulus of the filter).
    band_lam_min : float, optional
        Shortest wavelength the lift is applied to (smooth Butterworth
        roll-off; half-power at this wavelength). Defaults to ``2.5*H`` —
        the calibrated band edge. Wavelengths below it pass unchanged.
    gamma, theta : float
        Extension parameter and long-wavelength regularization, as in
        :class:`~stereo_melt.dynamics.linear_perturbation.LinearPerturbation`.

    Returns
    -------
    xarray.DataArray
        Restored melt field, same grid/units/sign as ``m``.
    """
    if "time" in m.dims:
        raise ValueError("bridging_restoration expects a (y, x) field")

    ux, uy = _mean_component(vx), _mean_component(vy)
    x_coords = m["x"].values
    y_coords = m["y"].values
    ny, nx = m.sizes["y"], m.sizes["x"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))

    # Mirror-pad to 2x so non-periodic tile edges (inflow ramps, front
    # notches) don't wrap through the filter; the filter is built on the
    # padded grid and the result cropped back.
    F, _ = bridging_restoration_filter(
        2 * ny, 2 * nx, dx, dy, H, ux, uy, eta_bar=eta_bar,
        alpha_scale=alpha_scale, lift_cap=lift_cap,
        band_lam_min=band_lam_min, rho_i=rho_i, rho_w=rho_w, g=g,
        gamma=gamma, theta=theta,
    )

    vals = m.values.astype(np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        raise ValueError("melt field contains no finite cells")
    fill = float(vals[finite].mean())
    filled = np.where(finite, vals, fill)
    padded = asarray(np.pad(filled, ((0, ny), (0, nx)), mode="symmetric"))
    rest_pad = to_numpy(xp.fft.ifft2(F * xp.fft.fft2(padded)).real)
    restored = rest_pad[:ny, :nx]
    restored[~finite] = np.nan

    return xr.DataArray(
        restored, dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords}, name=m.name or "m",
        attrs={
            **m.attrs,
            "bridging_restoration": "1 + w*(1/T - 1), w = (k.u_hat)^2/|k|^2",
            "H": H, "eta_bar": eta_bar, "alpha_scale": alpha_scale,
            "lift_cap": lift_cap, "ux_myr": ux, "uy_myr": uy,
        },
    )


def bridging_inverse_filter(
    ny: int,
    nx: int,
    dx: float,
    dy: float,
    H: float,
    ux_myr: float,
    uy_myr: float,
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    lam: float = 1e-3,
    reg: str = "white",
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    gamma: float = 0.0,
    theta: float = 1e-14,
):
    r"""Return the regularized (Wiener/Tikhonov) inverse filter ``W`` and ``T``.

    ``W(k) = conj(T) / (|T|^2 + lam * L(k))`` on the ``(ny, nx)`` FFT grid, with
    ``W(0) = 1`` (DC passthrough). This is the per-wavenumber minimizer of
    ``||T m - m_rec||^2 + lam ||L m||^2`` for the diagonal transfer ``T`` — the
    well-posed counterpart to :func:`bridging_restoration_filter`'s
    ``1 + w(1/T - 1)``: where ``|T| -> 0`` the inverse is **suppressed**
    (``W -> 0``), not amplified to a cap. ``reg='white'`` uses ``L = 1`` (Wiener,
    white prior — validated best on E2a); ``reg='grad'`` uses ``L = (|k| H)^2``
    (roughness prior). ``lam`` trades resolution against noise (small on clean
    data; larger on noisy DEMs).
    """
    if reg not in ("white", "grad"):
        raise ValueError(f"unknown reg {reg!r} (use 'white' or 'grad')")
    T, kx, ky, fb, u_mag = _bridging_transfer(
        ny, nx, dx, dy, H, ux_myr, uy_myr, eta_bar, alpha_scale,
        rho_i, rho_w, g, gamma, theta)
    kmag2 = kx**2 + ky**2
    L = kmag2 * (H * H) if reg == "grad" else xp.ones_like(kmag2)
    W = xp.conj(T) / (xp.abs(T) ** 2 + lam * L)
    # DC passthrough (spatial mean authoritative) + degenerate-mode guard.
    bad = (kmag2 <= 0) | ~xp.isfinite(W)
    W = xp.where(bad, xp.asarray(1.0 + 0j), W)
    return W, T


def bridging_inverse(
    m: xr.DataArray,
    vx: xr.DataArray | float,
    vy: xr.DataArray | float,
    H: float,
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    lam: float = 1e-3,
    reg: str = "white",
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    gamma: float = 0.0,
    theta: float = 1e-14,
) -> xr.DataArray:
    r"""Well-posed inverse of the bridging transfer (regularized deconvolution).

    The variational counterpart to :func:`bridging_restoration`. Rather than the
    band-limited, capped post-filter ``1 + w(1/T - 1)``, solve

    .. math::
        \hat m = \arg\min_m \; \|T m - m_{\rm rec}\|^2 + \lambda\,\|L m\|^2,

    whose per-wavenumber minimizer is the Wiener filter
    ``W = conj(T)/(|T|^2 + lam |L|^2)``. Apply to a **plain Eulerian** recovered
    melt ``m_rec`` (Shean sign, m ice yr\ :sup:`-1`): the hydrostatic error there
    is exactly the transfer ``T`` on every axis, so ``W`` inverts it cleanly,
    restoring both amplitude and downstream phase. Modes the shelf cannot express
    (``|T| -> 0``) are damped by the prior instead of amplified, so unlike the
    post-filter this recovers broadband/isotropic melt without distortion
    (E2a gpatch: corr 0.42 -> 0.95, nrmse 1.95 -> 0.35). Prefer this to the
    budget-family output, which already beats ``T`` across-flow and would be
    over-lifted.

    Parameters mirror :func:`bridging_restoration`; ``lam`` (regularization
    strength) and ``reg`` (``'white'`` Wiener default, or ``'grad'`` roughness)
    replace ``lift_cap``/``band_lam_min``. NaNs allowed (filled with the finite
    mean for the FFT, masked back after).
    """
    if "time" in m.dims:
        raise ValueError("bridging_inverse expects a (y, x) field")
    ux, uy = _mean_component(vx), _mean_component(vy)
    x_coords = m["x"].values
    y_coords = m["y"].values
    ny, nx = m.sizes["y"], m.sizes["x"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))

    # Mirror-pad to 2x so non-periodic tile edges don't wrap through the filter.
    W, _ = bridging_inverse_filter(
        2 * ny, 2 * nx, dx, dy, H, ux, uy, eta_bar=eta_bar,
        alpha_scale=alpha_scale, lam=lam, reg=reg, rho_i=rho_i, rho_w=rho_w,
        g=g, gamma=gamma, theta=theta)

    vals = m.values.astype(np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        raise ValueError("melt field contains no finite cells")
    fill = float(vals[finite].mean())
    filled = np.where(finite, vals, fill)
    padded = asarray(np.pad(filled, ((0, ny), (0, nx)), mode="symmetric"))
    inv_pad = to_numpy(xp.fft.ifft2(W * xp.fft.fft2(padded)).real)
    inv = inv_pad[:ny, :nx]
    inv[~finite] = np.nan

    return xr.DataArray(
        inv, dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords}, name=m.name or "m",
        attrs={
            **m.attrs,
            "bridging_inverse": "conj(T)/(|T|^2 + lam*L)",
            "H": H, "eta_bar": eta_bar, "alpha_scale": alpha_scale,
            "lam": lam, "reg": reg, "ux_myr": ux, "uy_myr": uy,
        },
    )
