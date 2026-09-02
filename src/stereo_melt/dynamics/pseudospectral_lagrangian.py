"""Time-dependent linear-perturbation inverse in the Lagrangian frame.

Implements the Level-0 pipeline of ``literature/plan_lagrangian.md``:

1. Advect the observed surface stack into a comoving reference frame
   anchored at the first epoch (existing
   :func:`stereo_melt.dynamics.lagrangian_inverse.lagrangian_frame_stack`).
2. In the comoving frame, ``α = 0`` by construction (the bulk
   advection is absorbed into the frame transformation), so the
   pseudo-spectral kernel simplifies to the static-column response.
3. Build a :class:`PerturbationForwardOp` with ``α = 0`` and tile-mean
   ``H, γ``; apply :func:`cg_invert` to recover ``m'(ξ, η, t)``.

Distinct from :func:`stereo_melt.dynamics.lagrangian_inverse.linear_inverse_lagrangian_melt_rate`,
which uses the closed-form steady-state inverse and recovers a single
2-D field. This module recovers the time-dependent ``m'`` by solving a
genuine time-domain LS problem.

Plan steps left for follow-on:

- Step 2 (TV via lagged diffusivity) --- the current Tikhonov inner
  loop is the weighted-L2 sub-problem the eventual TV outer loop will
  call repeatedly.
- Step 4 (tiles in ξ-space) --- requires building one
  :class:`PerturbationForwardOp` per tile and blending in overlap.
- Level 2 (time-varying kernel along trajectory) --- requires
  switching from analytic time-convolution to explicit pseudo-spectral
  time-stepping with kernel updates per step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import SECONDS_PER_YEAR, divergence, flux_divergence
from .lagrangian_inverse import lagrangian_frame_stack
from .pseudospectral import (
    PerturbationForwardOp,
    cg_invert,
    cg_invert_basis,
    make_temporal_basis,
)

__all__ = ["pseudospectral_lagrangian_inverse"]

G_GRAVITY = 9.81


def pseudospectral_lagrangian_inverse(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    floating_mask: xr.DataArray | None = None,
    H_ref: float | xr.DataArray | None = None,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    gamma: float | None = None,
    theta: float = 1e-14,
    tikhonov: float = 1e-2,
    temporal_basis: str = "full",
    length_scale_m: float = 0.0,
    max_iter: int = 100,
    cg_tol: float = 1e-6,
    dt_yr: float = 0.05,
    d: xr.DataArray | float = 0.0,
    verbose: bool = False,
) -> xr.Dataset:
    r"""Recover time-dependent basal melt rate via the Stubblefield linear
    inverse in a Lagrangian co-moving frame.

    This is the time-dependent companion to
    :func:`linear_inverse_lagrangian_melt_rate`. Same frame, same
    physics, but the inverse recovers ``m'(t, ξ, η)`` rather than a
    single time-mean ``m(ξ, η)``.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Geoid-referenced corrected surface elevation stack, meters.
    vx, vy : xarray.DataArray
        Velocity field, m/yr. Time-mean used for the trajectory advection.
    floating_mask : xarray.DataArray, optional
        Boolean mask of floating ice. Used both to mask the recovered
        melt rate and to restrict the spatial means used for ``H_ref``
        and ``γ``.
    H_ref : float or xarray.DataArray, optional
        Reference ice-equivalent thickness, meters. If a DataArray,
        the spatial mean over the floating mask is used. If None,
        defaults to the spatial mean of the hydrostatic-converted
        time-mean surface over the floating mask.
    eta_bar, rho_i, rho_w, g, theta : float
        Physical constants and long-wavelength regularization.
    gamma : float, optional
        Dimensionless extension ``E·t_r``. Defaults to
        ``⟨∇·u⟩_{floating} · t_r``.
    tikhonov : float
        Per-wavenumber Tikhonov scale for the CG inverter; ``1e-3`` for
        clean synthetic, ``1e-2`` to ``1e-1`` for noisy real data.
    max_iter : int
        CG iteration cap.
    cg_tol : float
        Relative-residual stopping threshold for CG.
    dt_yr : float
        Forward-Euler sub-step (years) for the trajectory tracer.
    d : float or xarray.DataArray
        Firn air content (m); used only to derive the diagnostic
        ``H_f_mean`` and the default ``H_ref``.
    verbose : bool
        If True, print CG iteration progress.

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` --- m/yr, dims ``(time, y, x)`` on the Lagrangian
        reference grid (which coincides with the Eulerian grid at the
        anchor epoch).
        ``H_f_mean``, ``flux_div``, ``a_dot`` --- diagnostic
        broadcasts matching the existing solver Datasets.
        Attributes record ``H_ref``, ``gamma_dimless``, ``tr_yr``,
        Tikhonov scale, CG iteration count, and final residual.
    """
    if "time" not in h_stack.dims:
        raise ValueError("pseudospectral_lagrangian_inverse requires a 'time' dim on h_stack")
    if h_stack.sizes["time"] < 2:
        raise ValueError("need at least two epochs for the time-dependent inverse")

    if "time" in vx.dims:
        vx_mean = vx.mean("time", skipna=True)
    else:
        vx_mean = vx
    if "time" in vy.dims:
        vy_mean = vy.mean("time", skipna=True)
    else:
        vy_mean = vy

    h_lag = lagrangian_frame_stack(h_stack, vx_mean, vy_mean, dt_yr=dt_yr)

    H_f_stack = freeboard_to_thickness(h_stack, d=d, rho_w=rho_w, rho_i=rho_i)
    H_f_mean = H_f_stack.mean("time", skipna=True)

    if H_ref is None:
        if floating_mask is not None:
            H_ref_val = float(H_f_mean.where(floating_mask).mean(skipna=True))
        else:
            H_ref_val = float(H_f_mean.mean(skipna=True))
    elif isinstance(H_ref, xr.DataArray):
        if floating_mask is not None:
            H_ref_val = float(H_ref.where(floating_mask).mean(skipna=True))
        else:
            H_ref_val = float(H_ref.mean(skipna=True))
    else:
        H_ref_val = float(H_ref)
    if not np.isfinite(H_ref_val) or H_ref_val <= 0:
        raise ValueError(f"derived H_ref={H_ref_val} is not positive finite; supply explicitly")

    tr_seconds = 2.0 * eta_bar / (rho_i * g * H_ref_val)
    if gamma is None:
        div_u = divergence(vx_mean, vy_mean)
        if floating_mask is not None:
            div_per_yr = float(div_u.where(floating_mask).mean(skipna=True))
        else:
            div_per_yr = float(div_u.mean(skipna=True))
        if not np.isfinite(div_per_yr):
            div_per_yr = 0.0
        gamma_val = (div_per_yr / SECONDS_PER_YEAR) * tr_seconds
    else:
        gamma_val = float(gamma)

    # Anchor at the first epoch: h_lag[0] is the Eulerian h at t_0.
    # Subtract h_lag[0] so we feed an anomaly with h_anom(t=0) = 0
    # exactly, matching the forward operator's t=0 convention. Keep NaN so the
    # masked CG excludes coverage gaps; do NOT fillna(0), which would feed the
    # mask fake zero observations and reintroduce the zero-fill coverage cliff.
    h_anom = h_lag - h_lag.isel(time=0)

    x_coords = h_anom["x"].values
    y_coords = h_anom["y"].values
    nx = h_anom.sizes["x"]
    ny = h_anom.sizes["y"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))

    op = PerturbationForwardOp(
        H=H_ref_val,
        times=h_anom["time"].values,
        nx=nx, ny=ny, dx=dx, dy=dy,
        eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
        alpha=0.0, gamma=gamma_val, theta=theta,
    )

    if temporal_basis == "full":
        m_recovered, info = cg_invert(
            op, h_anom.values,
            tikhonov=tikhonov,
            max_iter=max_iter,
            tol=cg_tol,
            verbose=verbose,
        )
    else:
        # Reduced temporal basis: solve for J coefficient fields, reconstruct.
        phi, _ = make_temporal_basis(op.t_secs, temporal_basis)
        coeffs, info = cg_invert_basis(
            op, h_anom.values, phi,
            tikhonov=tikhonov, length_scale_m=length_scale_m,
            max_iter=max_iter, tol=cg_tol, verbose=verbose,
        )
        m_recovered = np.zeros((op.n_t, ny, nx), dtype=np.float64)
        for _i in range(op.n_t):
            for _j in range(coeffs.shape[0]):
                m_recovered[_i] += coeffs[_j] * float(phi[_i, _j])
    if verbose:
        print(
            f"  CG done: iters={info['n_iter']}  "
            f"‖r‖/‖r_0‖={info['residual_norm']/max(info['residual_norm_initial'],1e-30):.3e}  "
            f"converged={info['converged']}"
        )

    # Internal Stubblefield forward operator maps positive m -> downward
    # h (melt-as-positive). Negate to publish the package-wide Shean
    # convention (positive = accretion, negative = melt).
    m_recovered = -m_recovered
    coords = {"time": h_anom["time"].values, "y": y_coords, "x": x_coords}
    m_da = xr.DataArray(m_recovered, dims=("time", "y", "x"), coords=coords, name="melt_rate")
    if floating_mask is not None:
        m_da = m_da.where(floating_mask)
        H_f_mean = H_f_mean.where(floating_mask)

    fd = flux_divergence(H_f_mean, vx_mean, vy_mean)
    a_dot_field = xr.zeros_like(H_f_mean).rename("a_dot")

    return xr.Dataset(
        {
            "melt_rate": m_da,
            "H_f_mean": H_f_mean,
            "flux_div": fd,
            "a_dot": a_dot_field,
        },
        attrs={
            "equation": (
                "Stubblefield 2023 time-dependent linear inverse in Lagrangian frame; "
                "alpha=0, gamma=⟨∇·u⟩·t_r"
            ),
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref_val,
            "gamma_dimless": gamma_val,
            "tr_yr": tr_seconds / SECONDS_PER_YEAR,
            "alpha": 0.0,
            "tikhonov": tikhonov,
            "temporal_basis": temporal_basis,
            "length_scale_m": length_scale_m,
            "cg_iter": int(info["n_iter"]),
            "cg_residual_norm": float(info["residual_norm"]),
            "cg_converged": bool(info["converged"]),
            "anchor_epoch": str(h_stack["time"].values[0]),
        },
    )
