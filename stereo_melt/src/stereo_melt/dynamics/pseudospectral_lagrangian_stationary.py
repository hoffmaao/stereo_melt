"""Stationary CG-LSQ pseudospectral inverse in the Lagrangian frame.

The user-facing analogue of :func:`linear_inverse_lagrangian_melt_rate`
(closed-form steady-state inverse) and the time-dependent
:func:`pseudospectral_lagrangian_inverse`. Recovers a single 2-D
``m(x, y)`` field using a CG-LSQ minimisation that:

1. Treats every NaN cell honestly via the per-cell, per-epoch weight
   mask (``W=0`` where data is missing). No ``fillna(0)`` upstream —
   gap cells contribute zero to both the misfit gradient and the
   curvature, instead of being injected as fake zero observations.
2. Uses the same Stubblefield-2023 column-Stokes kernels as the other
   two solvers (alpha=0 in the Lagrangian frame; tile-mean
   ``γ = ⟨∇·u⟩·t_r``).
3. Exploits the stationary-unknown collapse so each CG iteration is
   ``O(n_t · ny · nx)`` FFTs rather than the ``O(n_t² · ny · nx)`` of
   the time-dependent CG.
4. Optional ``localize`` preprocessing (lifted from
   agstub/lake-altimetry-inversions) shrinks any residual cliff at the
   floating-mask boundary by per-epoch subtracting the outer-ring mean.

The strip-coverage-gappy regime where this matters: real DEM stacks
where every epoch has many NaN cells (the gaps between WorldView
swaths) and where you need the spatial pattern but cannot afford to
ask for ``m(t, x, y)`` -- there isn't enough simultaneous coverage.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import SECONDS_PER_YEAR, divergence, flux_divergence
from .lagrangian_inverse import lagrangian_frame_stack
from .perturbation_dct import PerturbationForwardOpDCT
from .pseudospectral import PerturbationForwardOp, cg_invert_stationary

__all__ = ["stationary_pseudospectral_lagrangian_inverse"]

G_GRAVITY = 9.81


def stationary_pseudospectral_lagrangian_inverse(
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
    length_scale_m: float = 100.0,
    max_iter: int = 100,
    cg_tol: float = 1e-6,
    dt_yr: float = 0.05,
    d: xr.DataArray | float = 0.0,
    localize: bool = True,
    localize_ring_frac: float = 0.8,
    reference: str = "time_mean",
    h_lag_precomputed: xr.DataArray | None = None,
    transform: str = "fft",
    verbose: bool = False,
) -> xr.Dataset:
    r"""Recover a stationary basal melt rate ``m(x, y)`` via CG-LSQ.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Geoid-referenced corrected surface elevation stack, meters.
        NaN cells (strip-coverage gaps, off-mask, etc.) are honored;
        they do **not** get filled with zero.
    vx, vy : xarray.DataArray
        Velocity field, m/yr. Time-mean is used for trajectory advection.
    floating_mask : xarray.DataArray, optional
        Boolean mask of floating ice. Used both to mask the recovered
        ``m`` and to anchor the spatial means used for ``H_ref``,
        ``γ``, and the ``localize`` preprocessing ring.
    H_ref : float or xarray.DataArray, optional
        Reference ice-equivalent thickness, m. Default: spatial mean
        of the hydrostatic-converted time-mean surface over the
        floating mask.
    eta_bar, rho_i, rho_w, g, theta : float
        Physical constants and long-wavelength regularization.
    gamma : float, optional
        Dimensionless extension ``E·t_r``. Default: ``⟨∇·u⟩_{floating} · t_r``.
    tikhonov : float
        Dimensionless Tikhonov scale for ``cg_invert_stationary``.
    max_iter : int
        CG iteration cap.
    cg_tol : float
        Relative-residual stopping threshold for CG.
    dt_yr : float
        Sub-step (years) for the trajectory tracer in
        :func:`lagrangian_frame_stack`.
    d : float or xarray.DataArray
        Firn air content (m), used only to derive ``H_f_mean`` and
        the default ``H_ref``.
    localize : bool, default True
        If True, per-epoch subtract the spatial mean of ``h_anom`` over
        the outer ring of the floating mask
        (``r > localize_ring_frac · r_max``) to remove any spatially-
        uniform temporal trend before the CG. Mirrors the ``localize()``
        step in agstub/lake-altimetry-inversions.
    localize_ring_frac : float, default 0.8
        Inner radial fraction of the floating-mask extent below which
        cells are excluded from the outer-ring mean.
    verbose : bool
        Print per-CG-iteration residual.

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` --- ``(y, x)`` recovered stationary melt, m ice/yr,
        masked to floating ice.
        ``H_f_mean``, ``flux_div``, ``a_dot`` --- diagnostic broadcasts
        matching the other solvers.
        Attrs record ``H_ref_m``, ``gamma_dimless``, ``tr_yr``,
        Tikhonov scale, CG iter / residual / convergence, and
        whether localize was applied.
    """
    if "time" not in h_stack.dims:
        raise ValueError(
            "stationary_pseudospectral_lagrangian_inverse requires a 'time' dim"
        )
    if h_stack.sizes["time"] < 2:
        raise ValueError("need at least two epochs for the stationary inverse")

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
        raise ValueError(
            f"derived H_ref={H_ref_val} is not positive finite; supply explicitly"
        )

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

    # Per-pixel time-mean anomaly: better than first-epoch anchor when
    # the anchor has many NaN cells (those would zero the anomaly there
    # even though the cell may have data at later epochs).
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

    # Critical: do NOT fillna(0). cg_invert_stationary masks NaN via W.
    h_obs = h_anom.values.astype(np.float64)

    x_coords = h_anom["x"].values
    y_coords = h_anom["y"].values
    nx = h_anom.sizes["x"]
    ny = h_anom.sizes["y"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))

    if transform == "fft":
        op_cls = PerturbationForwardOp
    elif transform == "dct":
        op_cls = PerturbationForwardOpDCT
    else:
        raise ValueError(f"transform must be 'fft' or 'dct', got {transform!r}")
    op = op_cls(
        H=H_ref_val,
        times=h_anom["time"].values,
        nx=nx, ny=ny, dx=dx, dy=dy,
        eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
        alpha=0.0, gamma=gamma_val, theta=theta,
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
            f"  CG-stat done: iters={info['n_iter']}  "
            f"‖r‖/‖r_0‖={info['residual_norm']/max(info['residual_norm_initial'],1e-30):.3e}  "
            f"converged={info['converged']}"
        )

    # Internal Stubblefield forward operator maps positive m -> downward
    # h (melt-as-positive). Negate to publish the package-wide Shean
    # convention (positive = accretion, negative = melt).
    m_recovered = -m_recovered
    coords = {"y": y_coords, "x": x_coords}
    m_da = xr.DataArray(m_recovered, dims=("y", "x"), coords=coords, name="melt_rate")
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
                "Stubblefield 2023 stationary CG-LSQ inverse in Lagrangian frame; "
                "alpha=0; W-mask honors NaN; lake-altimetry-inversions localize"
            ),
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref_val,
            "gamma_dimless": gamma_val,
            "tr_yr": tr_seconds / SECONDS_PER_YEAR,
            "alpha": 0.0,
            "tikhonov": tikhonov,
            "length_scale_m": float(length_scale_m),
            "cg_iter": int(info["n_iter"]),
            "cg_residual_norm": float(info["residual_norm"]),
            "cg_converged": bool(info["converged"]),
            "localize": bool(localize),
            "localize_ring_frac": float(localize_ring_frac) if localize else 0.0,
            "reference": reference,
            "transform": transform,
            "anchor_epoch": str(h_stack["time"].values[0]),
        },
    )
