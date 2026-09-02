"""Time-dependent linear-perturbation inverse in the Eulerian frame.

Implements the Level-0 pipeline of ``literature/plan_eulerian.md``:

1. Use the observed surface stack on its native (fixed) grid --- no
   trajectory advection.
2. Derive the constant background ``(α_x, α_y, γ)`` from the
   tile-mean MEaSUREs velocity and its divergence over the floating
   mask. The 2-D ``(α_x, α_y)`` are needed because the Eulerian frame
   carries the full background advection in the spectral kernel
   (whereas the Lagrangian frame absorbs it into the trajectory
   transformation).
3. Build a :class:`PerturbationForwardOp` with the derived
   ``(α_x, α_y, γ)`` and a tile-mean ``H_ref``; apply
   :func:`cg_invert` to recover ``m'(x, y, t)`` on the same grid.

Companion to
:func:`stereo_melt.dynamics.pseudospectral_lagrangian.pseudospectral_lagrangian_inverse`.
Both share the same forward operator and CG inverter; the only
differences are the data preprocessing (none for Eulerian; trajectory
advection for Lagrangian) and how ``α`` is derived (from velocity
directly here; ``α = 0`` by construction in the comoving frame).

Plan steps left for follow-on:

- Step 2 (TV via lagged diffusivity) --- shared with Lagrangian.
- Step 4 (overlapping tiles) --- requires per-tile parameter
  estimation and Tukey-tapered blending.
- Levels 2--3 (δH expansion / WKB) --- require updating the kernel
  per spatial location, breaking the global FFT.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import SECONDS_PER_YEAR, divergence, flux_divergence
from .pseudospectral import (
    PerturbationForwardOp,
    cg_invert,
    cg_invert_basis,
    make_temporal_basis,
)

__all__ = ["pseudospectral_eulerian_inverse"]

G_GRAVITY = 9.81


def pseudospectral_eulerian_inverse(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    floating_mask: xr.DataArray | None = None,
    H_ref: float | xr.DataArray | None = None,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    alpha_x: float | None = None,
    alpha_y: float | None = None,
    gamma: float | None = None,
    theta: float = 1e-14,
    tikhonov: float = 1e-2,
    temporal_basis: str = "full",
    length_scale_m: float = 0.0,
    max_iter: int = 100,
    cg_tol: float = 1e-6,
    d: xr.DataArray | float = 0.0,
    verbose: bool = False,
) -> xr.Dataset:
    r"""Recover time-dependent basal melt rate via the Stubblefield linear
    inverse in the fixed Eulerian frame.

    No trajectory bookkeeping: the strip stack is fed directly to the
    pseudo-spectral forward operator with ``(α_x, α_y, γ)`` derived
    from the tile-mean MEaSUREs velocity.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Surface elevation perturbation, meters. The first epoch is
        treated as ``t = 0`` reference (anomaly definition consistent
        with the forward operator's causal kernel).
    vx, vy : xarray.DataArray
        Velocity components on the ``(y, x)`` grid, m/yr.
    floating_mask : xarray.DataArray, optional
        Boolean mask of floating ice. Used both to mask the recovered
        melt rate and to restrict the spatial means used to derive
        ``H_ref``, ``α_*``, and ``γ``.
    H_ref : float or xarray.DataArray, optional
        Reference ice-equivalent thickness, meters. If a DataArray,
        the spatial mean over the floating mask is used. If None,
        defaults to the spatial mean of the hydrostatic-converted
        time-mean surface over the floating mask.
    eta_bar, rho_i, rho_w, g, theta : float
        Physical constants and long-wavelength regularization.
    alpha_x, alpha_y : float, optional
        Dimensionless 2-D advection ``ū·t_r/H``. If None, derived from
        the tile-mean MEaSUREs velocity over the floating mask.
    gamma : float, optional
        Dimensionless extension ``E·t_r``. Defaults to
        ``⟨∇·u⟩_{floating} · t_r``.
    tikhonov : float
        Per-wavenumber Tikhonov scale for the CG inverter.
    max_iter : int
        CG iteration cap.
    cg_tol : float
        Relative-residual stopping threshold for CG.
    d : float or xarray.DataArray
        Firn air content (m).
    verbose : bool
        If True, print CG iteration progress.

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` --- m/yr, dims ``(time, y, x)`` on the input
        grid. Diagnostics: ``H_f_mean``, ``flux_div``, ``a_dot``.
        Attributes record ``H_ref``, ``alpha_x``, ``alpha_y``,
        ``gamma_dimless``, Tikhonov scale, and CG metadata.
    """
    if "time" not in h_stack.dims:
        raise ValueError("pseudospectral_eulerian_inverse requires a 'time' dim on h_stack")
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

    # Tile-mean velocity for (alpha_x, alpha_y) — restrict to floating mask
    # if provided so we don't pick up grounded-ice flow that's irrelevant
    # to the floating-shelf physics.
    if alpha_x is None:
        if floating_mask is not None:
            ux_mean = float(vx_mean.where(floating_mask).mean(skipna=True))
        else:
            ux_mean = float(vx_mean.mean(skipna=True))
        if not np.isfinite(ux_mean):
            ux_mean = 0.0
        alpha_x_val = (ux_mean / SECONDS_PER_YEAR) * tr_seconds / H_ref_val
    else:
        alpha_x_val = float(alpha_x)
    if alpha_y is None:
        if floating_mask is not None:
            uy_mean = float(vy_mean.where(floating_mask).mean(skipna=True))
        else:
            uy_mean = float(vy_mean.mean(skipna=True))
        if not np.isfinite(uy_mean):
            uy_mean = 0.0
        alpha_y_val = (uy_mean / SECONDS_PER_YEAR) * tr_seconds / H_ref_val
    else:
        alpha_y_val = float(alpha_y)

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

    # Anomaly relative to first epoch — matches the forward operator's
    # causal-kernel convention (h(t=0) = 0). Keep NaN so the masked CG excludes
    # coverage gaps; do NOT fillna(0), which would feed the mask fake zero
    # observations and reintroduce the zero-fill coverage cliff.
    h_anom = h_stack - h_stack.isel(time=0)

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
        alpha=alpha_x_val, alpha_y=alpha_y_val, gamma=gamma_val, theta=theta,
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
                "Stubblefield 2023 time-dependent linear inverse in Eulerian frame; "
                "(alpha_x, alpha_y) = (ū, v̄)·t_r/H, gamma=⟨∇·u⟩·t_r"
            ),
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref_val,
            "alpha_x": alpha_x_val,
            "alpha_y": alpha_y_val,
            "gamma_dimless": gamma_val,
            "tr_yr": tr_seconds / SECONDS_PER_YEAR,
            "tikhonov": tikhonov,
            "temporal_basis": temporal_basis,
            "length_scale_m": length_scale_m,
            "cg_iter": int(info["n_iter"]),
            "cg_residual_norm": float(info["residual_norm"]),
            "cg_converged": bool(info["converged"]),
            "anchor_epoch": str(h_stack["time"].values[0]),
        },
    )
