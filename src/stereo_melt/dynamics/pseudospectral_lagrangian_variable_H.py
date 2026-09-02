"""Stationary variable-H pseudospectral inverse in the Lagrangian frame (v3).

Companion to :func:`stationary_pseudospectral_lagrangian_inverse` (v2-CG):
relaxes the ``H_ref = constant`` assumption by Taylor-expanding the
Stubblefield kernel to first order in ``H`` around the spatial mean
``H_ref``. The expansion is implemented by
:class:`stereo_melt.dynamics.pseudospectral.LinearizedHForwardOp`, which
finite-differences ``∂K/∂H`` numerically by constructing two
:class:`PerturbationForwardOp` instances at ``H_ref·(1 ± eps)``.

Forward / adjoint structure (from the operator class):

  forward:  h(x,y) = IFFT[K0(k) · m̂(k)]  +  ΔH(x,y) · IFFT[K1(k) · m̂(k)]
  adjoint:  m_adj  = IFFT[conj(K0) · ĥ]  +  IFFT[conj(K1) · FFT(ΔH·h)]

For Beardmore where ``ΔH/H_ref ≈ ±15-20%``, the truncation error is
second-order ≈ 4% — well below the η_bar / γ uncertainty bands in the
sensitivity sweep.

Required input: ``H_total_field`` (per-pixel firn-corrected ice thickness,
m). Pass ``H_f_mean + (ρ_w/(ρ_w-ρ_i)) * firn`` from BedMachine.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import SECONDS_PER_YEAR, divergence, flux_divergence
from .lagrangian_inverse import lagrangian_frame_stack
from .pseudospectral import LinearizedHForwardOp, cg_invert_stationary

__all__ = ["variable_H_pseudospectral_lagrangian_inverse"]

G_GRAVITY = 9.81


def variable_H_pseudospectral_lagrangian_inverse(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    H_total_field: xr.DataArray,
    floating_mask: xr.DataArray | None = None,
    H_ref: float | None = None,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    gamma: float | None = None,
    theta: float = 1e-14,
    tikhonov: float = 1e-2,
    length_scale_m: float = 100.0,
    max_iter: int = 100,
    cg_tol: float = 1e-6,
    dt_yr: float = 0.05,
    d: xr.DataArray | float = 0.0,
    localize: bool = True,
    localize_ring_frac: float = 0.8,
    reference: str = "time_mean",
    eps_H: float = 0.05,
    dH_clip_rel: float = 0.3,
    h_lag_precomputed: xr.DataArray | None = None,
    verbose: bool = False,
) -> xr.Dataset:
    r"""Recover stationary ``m(x, y)`` with a position-dependent reference thickness.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims (time, y, x)
        Geoid-referenced corrected surface elevation stack, meters. NaN
        cells are honored via the W-mask in CG.
    vx, vy : xarray.DataArray
        Velocity field, m/yr; time-mean used for trajectory advection.
    H_total_field : xarray.DataArray, dims (y, x)
        Firn-corrected total ice thickness in meters. Typically
        ``H_f_mean + (rho_w/(rho_w - rho_i)) * firn`` where firn is from
        BedMachine. NaN cells outside the floating mask are filled with
        ``H_ref`` internally.
    floating_mask : optional
        Boolean mask of floating ice for output masking and statistics.
    H_ref : float, optional
        Anchor for the H-Taylor expansion. Defaults to spatial mean of
        ``H_total_field`` over the floating mask (the natural choice for
        the linearization).
    eta_bar, rho_i, rho_w, g, theta : physical constants.
    gamma : float, optional
        If supplied, overrides the derived ``γ = ⟨∇·u⟩·t_r``. The
        operator scales γ with H internally so the FD captures the
        ``γ ∝ 1/H`` dependence.
    tikhonov : float
        Dimensionless Tikhonov.
    max_iter, cg_tol : CG settings.
    dt_yr : float
        Trajectory sub-step (yr).
    d : float or DataArray
        Firn air content (m), used only for the ``H_f_mean`` /
        ``flux_div`` diagnostic outputs. Does NOT enter the kernel; the
        kernel uses ``H_total_field`` directly.
    localize : bool, default True
        Subtract the per-epoch outer-ring mean (lake-altimetry trick).
    eps_H : float, default 0.05
        Relative finite-difference step for ``∂K/∂H``.
    h_lag_precomputed : optional
        Pre-built Lagrangian-frame stack from
        :func:`lagrangian_frame_stack`. Reused across calls in sweeps.
    verbose : bool

    Returns
    -------
    xarray.Dataset
        ``melt_rate(y, x)`` recovered basal mass balance, m ice/yr
        (Shean convention: negative = melt, positive = accretion).
        ``H_f_mean``, ``flux_div``, ``a_dot``, plus the H-field used:
        ``H_total_used`` and ``dH``.
    """
    if "time" not in h_stack.dims:
        raise ValueError("variable_H_pseudospectral_lagrangian_inverse requires a 'time' dim")
    if h_stack.sizes["time"] < 2:
        raise ValueError("need at least two epochs")

    if "time" in vx.dims:
        vx_mean = vx.mean("time", skipna=True)
    else:
        vx_mean = vx
    if "time" in vy.dims:
        vy_mean = vy.mean("time", skipna=True)
    else:
        vy_mean = vy

    if h_lag_precomputed is not None:
        h_lag = h_lag_precomputed
    else:
        h_lag = lagrangian_frame_stack(h_stack, vx_mean, vy_mean, dt_yr=dt_yr)

    H_f_stack = freeboard_to_thickness(h_stack, d=d, rho_w=rho_w, rho_i=rho_i)
    H_f_mean = H_f_stack.mean("time", skipna=True)

    # Resolve H_ref from H_total_field over the floating mask if not supplied.
    if H_ref is None:
        if floating_mask is not None:
            H_ref_val = float(H_total_field.where(floating_mask).mean(skipna=True))
        else:
            H_ref_val = float(H_total_field.mean(skipna=True))
    else:
        H_ref_val = float(H_ref)
    if not np.isfinite(H_ref_val) or H_ref_val <= 0:
        raise ValueError(f"derived H_ref={H_ref_val} is not positive finite")

    tr_seconds = 2.0 * eta_bar / (rho_i * g * H_ref_val)

    # Spatial-mean div(u) for γ scaling. Used by the operator to recompute
    # γ at H_ref·(1 ± eps).
    div_u = divergence(vx_mean, vy_mean)
    if floating_mask is not None:
        div_per_yr = float(div_u.where(floating_mask).mean(skipna=True))
    else:
        div_per_yr = float(div_u.mean(skipna=True))
    if not np.isfinite(div_per_yr):
        div_per_yr = 0.0
    div_u_mean_si = div_per_yr / SECONDS_PER_YEAR
    if gamma is None:
        gamma_val = div_u_mean_si * tr_seconds
    else:
        gamma_val = float(gamma)

    # Build the time-mean anomaly + localize, exactly as in the v2 wrapper.
    h_mean = h_lag.mean("time", skipna=True)
    h_anom = h_lag - h_mean

    if localize and floating_mask is not None:
        fmask = floating_mask.values
        if fmask.any():
            yy_grid, xx_grid = np.meshgrid(
                h_anom["y"].values, h_anom["x"].values, indexing="ij"
            )
            y_c = float(yy_grid[fmask].mean())
            x_c = float(xx_grid[fmask].mean())
            r = np.sqrt((yy_grid - y_c) ** 2 + (xx_grid - x_c) ** 2)
            r_max = float(r[fmask].max())
            outer_ring = (r >= localize_ring_frac * r_max) & fmask
            ring_da = xr.DataArray(
                outer_ring, dims=("y", "x"),
                coords={"y": h_anom["y"], "x": h_anom["x"]},
            )
            far = h_anom.where(ring_da).mean(dim=("y", "x"), skipna=True)
            h_anom = h_anom - far

    # NaN-aware: cg_invert_stationary handles the W mask itself.
    h_obs = h_anom.values.astype(np.float64)

    x_coords = h_anom["x"].values
    y_coords = h_anom["y"].values
    nx = h_anom.sizes["x"]
    ny = h_anom.sizes["y"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))

    H_field_arr = np.asarray(H_total_field.values, dtype=np.float64)
    if H_field_arr.shape != (ny, nx):
        raise ValueError(
            f"H_total_field shape {H_field_arr.shape} != stack ({ny}, {nx})"
        )
    # Clip ΔH/H_ref to keep the first-order Taylor expansion valid. Outliers
    # at the floating-mask boundary often have unphysical hydrostatic-from-
    # freeboard thickness; these would dominate the operator's response and
    # drive the linearization out of its small-ΔH regime.
    dH = H_field_arr - H_ref_val
    finite = np.isfinite(dH)
    if dH_clip_rel > 0 and finite.any():
        clip = float(dH_clip_rel) * H_ref_val
        dH = np.where(finite, np.clip(dH, -clip, clip), dH)
        H_field_arr = np.where(finite, H_ref_val + dH, H_field_arr)
    dH_max_rel = float(np.nanmax(np.abs(dH[finite])) / H_ref_val) if finite.any() else 0.0

    op = LinearizedHForwardOp(
        H_field=H_field_arr,
        H_ref=H_ref_val,
        times=h_anom["time"].values,
        nx=nx, ny=ny, dx=dx, dy=dy,
        eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
        alpha=0.0, gamma=gamma_val, theta=theta,
        eps=eps_H,
        div_u_mean=div_u_mean_si,
    )

    m_recovered, info = cg_invert_stationary(
        op, h_obs,
        tikhonov=tikhonov,
        length_scale_m=length_scale_m,
        max_iter=max_iter,
        tol=cg_tol,
        reference=reference,
        verbose=verbose,
    )
    if verbose:
        print(
            f"  CG-varH done: iters={info['n_iter']}  "
            f"‖r‖/‖r_0‖={info['residual_norm']/max(info['residual_norm_initial'],1e-30):.3e}  "
            f"converged={info['converged']}"
        )

    # Internal Stubblefield forward operator maps positive m -> downward
    # h (melt-as-positive). Negate to publish the package-wide Shean
    # convention (positive = accretion, negative = melt).
    m_recovered = -m_recovered
    coords = {"y": y_coords, "x": x_coords}
    m_da = xr.DataArray(m_recovered, dims=("y", "x"), coords=coords, name="melt_rate")
    H_used = xr.DataArray(
        H_field_arr, dims=("y", "x"), coords=coords, name="H_total_used"
    )
    dH_da = xr.DataArray(
        np.where(np.isfinite(dH), dH, np.nan),
        dims=("y", "x"), coords=coords, name="dH",
    )
    if floating_mask is not None:
        m_da = m_da.where(floating_mask)
        H_f_mean = H_f_mean.where(floating_mask)
        H_used = H_used.where(floating_mask)
        dH_da = dH_da.where(floating_mask)

    fd = flux_divergence(H_f_mean, vx_mean, vy_mean)
    a_dot_field = xr.zeros_like(H_f_mean).rename("a_dot")

    return xr.Dataset(
        {
            "melt_rate": m_da,
            "H_f_mean": H_f_mean,
            "H_total_used": H_used,
            "dH": dH_da,
            "flux_div": fd,
            "a_dot": a_dot_field,
        },
        attrs={
            "equation": (
                "Stubblefield 2023 stationary CG-LSQ inverse in Lagrangian frame "
                "with first-order Taylor expansion of K(k, H) around H_ref "
                "(LinearizedHForwardOp); ΔH(x,y) enters as smooth real-space field"
            ),
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref_val,
            "dH_max_rel": dH_max_rel,
            "gamma_dimless": gamma_val,
            "tr_yr": tr_seconds / SECONDS_PER_YEAR,
            "alpha": 0.0,
            "tikhonov": tikhonov,
            "length_scale_m": float(length_scale_m),
            "eps_H": eps_H,
            "dH_clip_rel": float(dH_clip_rel),
            "cg_iter": int(info["n_iter"]),
            "cg_residual_norm": float(info["residual_norm"]),
            "cg_converged": bool(info["converged"]),
            "localize": bool(localize),
            "localize_ring_frac": float(localize_ring_frac) if localize else 0.0,
            "reference": reference,
            "anchor_epoch": str(h_stack["time"].values[0]),
        },
    )
