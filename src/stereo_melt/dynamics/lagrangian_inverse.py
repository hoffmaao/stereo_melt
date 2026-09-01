"""Stubblefield linear inverse in a Lagrangian (co-moving) reference frame.

The Eulerian Stubblefield inverse (:func:`inverse_stationary`) assumes a
uniform across-channel inflow `α` and uniform along-channel extension
`γ`. For real grounding zones with curved channels and spatially-varying
flow, the uniform-`α` assumption is the most fragile piece of the model.

This module sidesteps that constraint by transforming the surface-elevation
stack into a frame co-moving with the time-mean velocity field. In that
frame:

- `α = 0` by construction (no residual uniform inflow)
- `γ` collapses to the spatial mean of `∇·u` over the floating tile
- The stress kernels `R, B` are unchanged (they describe a static column)

Relationship to altimetryFit
----------------------------

Smith's ``altimetryFit/lagrangian_functions.py`` advects each
*observation point* to its position at a chosen ``lagrangian_epoch``.
We advect a *fixed grid* (the Eulerian grid at the anchor epoch) so
the comoving-frame stack stays on a regular grid amenable to the
spectral inverse. Functionally equivalent for our use case (gridded
REMA stacks); the trade-off is grid-deformation accumulation over
long trajectories vs. irregular point distributions. We anchor at the
first observation epoch (``inverse_stationary`` requires
``h(t=0) = 0`` by convention); altimetryFit accepts an arbitrary
``lagrangian_epoch`` parameter and is more flexible at the cost of
extra trajectory bookkeeping.

The remaining approximations are: tile-mean `H` and `γ` (same uniform-
coefficient assumption Stubblefield makes, just in a moving frame), and
the implicit assumption that `m` is stationary *along each trajectory*
rather than at fixed Eulerian points. For short observation windows where
particle displacement is small relative to melt-feature scale, the two
notions of stationarity are close; for longer windows or stronger flow
the recovered `m` is best read as the trajectory-averaged melt rate.

Frame anchor convention
-----------------------

Trajectories are anchored at the first observation epoch
(`t_ref_index = 0`). Particles seeded at every Eulerian pixel of the
first epoch are advected forward through the time-mean velocity field;
each subsequent epoch is sampled at the particle's current position.
The resulting Lagrangian-frame stack `h_lag(t, ξ, η)` shares the spatial
grid of the first epoch, so the anomaly
`h_anom = h_lag - h_lag.isel(time=0)` satisfies the
`h(t=0) = 0` convention of :func:`inverse_stationary` exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import asarray, gaussian_filter, map_coordinates, to_numpy, xp
from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import SECONDS_PER_YEAR, divergence, flux_divergence
from .linear_perturbation import inverse_dhdt, inverse_stationary

__all__ = [
    "lagrangian_frame_stack",
    "linear_inverse_lagrangian_melt_rate",
    "linear_inverse_dhdt_lagrangian_melt_rate",
]


def _per_pixel_ols_slope(
    stack_vals,
    t_secs,
    robust: bool = False,
    robust_c: float = 4.685,
    robust_max_iter: int = 8,
    robust_tol: float = 1e-3,
):
    """Per-pixel OLS slope of stack_vals(t) vs t, NaN-safe.

    Parameters
    ----------
    stack_vals : (n_t, ny, nx) float
    t_secs : (n_t,) float, observation seconds (relative anchor)
    robust : bool
        If True, refine the slope with Tukey-biweight IRLS using a
        basin-wide MAD scale (matches :func:`stereo_melt.kinematics.dh_dt`
        with ``robust=True``).
    robust_c, robust_max_iter, robust_tol :
        IRLS tuning parameters.

    Returns
    -------
    slope : (ny, nx) float64, m/s
    n_obs : (ny, nx) int, finite-epoch count per pixel
    """
    n_t = stack_vals.shape[0]
    t = np.asarray(t_secs, dtype=np.float64).reshape(n_t, 1, 1)
    h = stack_vals.astype(np.float64)
    finite = np.isfinite(h)
    h0 = np.where(finite, h, 0.0)
    w = finite.astype(np.float64)

    sum_w = w.sum(axis=0)
    sum_t = (w * t).sum(axis=0)
    sum_h = (w * h0).sum(axis=0)
    sum_tt = (w * t * t).sum(axis=0)
    sum_th = (w * t * h0).sum(axis=0)

    det = sum_w * sum_tt - sum_t * sum_t
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (sum_w * sum_th - sum_t * sum_h) / det
        intercept = (sum_h - slope * sum_t) / np.maximum(sum_w, 1.0)
    slope = np.where(sum_w >= 2, slope, np.nan)
    n_obs = sum_w.astype(int)

    if not robust:
        return slope, n_obs

    intercept = np.where(sum_w >= 2, intercept, 0.0)
    for _ in range(int(robust_max_iter)):
        pred = slope[None, :, :] * t + intercept[None, :, :]
        r_obs = np.where(finite, h - pred, 0.0)
        abs_r = np.where(finite, np.abs(r_obs), np.nan)
        med = np.nanmedian(abs_r)
        mad = np.nanmedian(np.abs(abs_r - med))
        scale = 1.4826 * float(mad)
        if not np.isfinite(scale) or scale < 1e-9:
            break
        u = r_obs / (robust_c * scale)
        ww = np.where(finite & (np.abs(u) < 1.0), (1.0 - u * u) ** 2, 0.0)
        sumW = ww.sum(axis=0)
        sumWT = (ww * t).sum(axis=0)
        sumWH = (ww * h0).sum(axis=0)
        sumWTT = (ww * t * t).sum(axis=0)
        sumWTH = (ww * t * h0).sum(axis=0)
        denom_w = sumW * sumWTT - sumWT * sumWT
        with np.errstate(invalid="ignore", divide="ignore"):
            new_slope = np.where(
                denom_w > 0,
                (sumW * sumWTH - sumWT * sumWH) / denom_w,
                slope,
            )
            new_intercept = np.where(
                sumW > 0,
                (sumWH - new_slope * sumWT) / np.maximum(sumW, 1e-12),
                intercept,
            )
        with np.errstate(invalid="ignore"):
            denom_chk = np.nanmax(np.abs(slope))
            rel = np.nanmax(np.abs(new_slope - slope)) / max(float(denom_chk), 1e-12)
        slope = np.where(sum_w >= 2, new_slope, np.nan)
        intercept = new_intercept
        if rel < robust_tol:
            break
    return slope, n_obs


def linear_inverse_dhdt_lagrangian_melt_rate(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    floating_mask: xr.DataArray | None = None,
    H_ref: float | xr.DataArray | None = None,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = 9.81,
    gamma: float | None = None,
    theta: float = 1e-14,
    reg: float = 1e-3,
    min_n_obs: int = 3,
    dt_yr: float = 0.05,
    d: xr.DataArray | float = 0.0,
    transform: str = "fft",
    h_lag_precomputed: xr.DataArray | None = None,
    robust_dh_dt: bool = False,
    recover_dc: bool = True,
    a_dot_dc: float = 0.0,
) -> xr.Dataset:
    r"""dh/dt-reformulation of the Stubblefield linear inverse in the Lagrangian frame.

    Same physics as :func:`linear_inverse_lagrangian_melt_rate` but inverts
    a per-pixel OLS slope field rather than the full (time, y, x) stack.

    The closed-form per-wavenumber Tikhonov inverse fed the entire
    ``h_anom`` stack to an FFT/DCT and treated NaN-filled cells as
    legitimate "anomaly = 0" observations — see
    ``tests/diagnose_linear_inverse_coverage.py`` for the
    failure mode. Real REMA stacks have 23-47% per-pixel coverage on
    Beardmore/Nansen/PIG, so the closed form's full-coverage assumption
    is dramatically violated and the recovered ``m`` is dominated by
    boundary ringing.

    This wrapper instead

      1. builds the Lagrangian-frame stack (same as the closed-form
         path, so per-particle history is preserved),
      2. fits a per-pixel OLS slope of ``h_lag(t)`` vs ``t`` using
         only that pixel's valid epochs (so coverage gaps don't bleed
         into other pixels),
      3. masks pixels with fewer than ``min_n_obs`` valid epochs,
      4. calls :func:`inverse_dhdt` for a single-step Fourier inverse.

    The recovered ``m`` is on the original ``(y, x)`` grid (no
    further resampling needed) and represents the trajectory-averaged
    basal melt rate at the particle sitting at that position at
    ``t = 0``.

    Parameters
    ----------
    h_stack, vx, vy, floating_mask, H_ref, eta_bar, rho_i, rho_w, g,
    gamma, theta, reg, dt_yr, d, transform :
        Same meaning as :func:`linear_inverse_lagrangian_melt_rate`.
    min_n_obs : int
        Drop any pixel whose Lagrangian-frame stack has fewer than this
        many valid epochs after the frame transform. Default 3 — bare
        minimum for a slope estimate. Pixels below this threshold are
        infilled with the dh/dt mean for the FFT and then masked out at
        the end.
    h_lag_precomputed : xr.DataArray, optional
        If supplied, skip :func:`lagrangian_frame_stack` and use this
        directly. Shape must match ``h_stack``.

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` (the recovered ``m``), ``dh_dt`` (the per-pixel
        slope used as input), ``n_obs`` (per-pixel valid-epoch count),
        ``H_f_mean``, ``flux_div``.
    """
    if "time" not in h_stack.dims:
        raise ValueError("requires a 'time' dim on h_stack")
    if transform not in ("fft", "dct"):
        raise ValueError(f"transform must be 'fft' or 'dct', got {transform!r}")

    if "time" in vx.dims:
        vx_mean = vx.mean("time", skipna=True)
    else:
        vx_mean = vx
    if "time" in vy.dims:
        vy_mean = vy.mean("time", skipna=True)
    else:
        vy_mean = vy

    h_lag = (
        h_lag_precomputed if h_lag_precomputed is not None
        else lagrangian_frame_stack(h_stack, vx_mean, vy_mean, dt_yr=dt_yr)
    )

    H_f_stack = freeboard_to_thickness(h_stack, d=d, rho_w=rho_w, rho_i=rho_i)
    H_f_mean = H_f_stack.mean("time", skipna=True)

    if H_ref is None:
        if floating_mask is not None:
            H_ref_val = float(H_f_mean.where(floating_mask).mean(skipna=True))
        else:
            H_ref_val = float(H_f_mean.mean(skipna=True))
    elif isinstance(H_ref, xr.DataArray):
        H_ref_val = (
            float(H_ref.where(floating_mask).mean(skipna=True))
            if floating_mask is not None
            else float(H_ref.mean(skipna=True))
        )
    else:
        H_ref_val = float(H_ref)

    if not np.isfinite(H_ref_val) or H_ref_val <= 0:
        raise ValueError(f"derived H_ref={H_ref_val} is not a positive finite number")

    tr = 2.0 * eta_bar / (rho_i * g * H_ref_val)
    if gamma is None:
        div_u = divergence(vx_mean, vy_mean)
        if floating_mask is not None:
            div_per_yr = float(div_u.where(floating_mask).mean(skipna=True))
        else:
            div_per_yr = float(div_u.mean(skipna=True))
        if not np.isfinite(div_per_yr):
            div_per_yr = 0.0
        gamma_val = (div_per_yr / SECONDS_PER_YEAR) * tr
    else:
        gamma_val = float(gamma)

    # Per-pixel OLS slope of h_lag(t) ~ a + b·t over valid epochs.
    t_vals = h_lag["time"].values
    if np.issubdtype(np.asarray(t_vals).dtype, np.datetime64):
        t_secs = (np.asarray(t_vals) - np.asarray(t_vals)[0]).astype(
            "timedelta64[s]"
        ).astype(np.float64)
    else:
        t_secs = np.asarray(t_vals, dtype=np.float64)
        t_secs = t_secs - t_secs[0]

    slope, n_obs = _per_pixel_ols_slope(
        h_lag.values, t_secs, robust=robust_dh_dt,
    )
    valid = n_obs >= int(min_n_obs)
    slope_masked = np.where(valid, slope, np.nan)

    dh_dt = xr.DataArray(
        slope_masked,
        dims=("y", "x"),
        coords={"y": h_lag["y"].values, "x": h_lag["x"].values},
        name="dh_dt",
        attrs={"units": "m s^-1", "min_n_obs": int(min_n_obs)},
    )
    n_obs_da = xr.DataArray(
        n_obs.astype(np.int32),
        dims=("y", "x"),
        coords={"y": h_lag["y"].values, "x": h_lag["x"].values},
        name="n_obs",
    )

    m = inverse_dhdt(
        dh_dt, H=H_ref_val, t_secs=t_secs,
        eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
        alpha=0.0, alpha_y=0.0, gamma=gamma_val, theta=theta,
        reg=reg, transform=transform,
        # The Lagrangian-frame per-pixel slope's basin-mean carries the
        # advective term ⟨u·∇h⟩; the wrapper splice below uses the
        # Eulerian-frame stack instead, so disable the internal splice.
        recover_dc=False,
    )

    if floating_mask is not None:
        m = m.where(floating_mask)
        H_f_mean = H_f_mean.where(floating_mask)
        dh_dt = dh_dt.where(floating_mask)

    fd = flux_divergence(H_f_mean, vx_mean, vy_mean)
    a_dot_field = xr.zeros_like(H_f_mean).rename("a_dot")

    # Eulerian-frame DC splice in the Shean public convention
    # (positive = accretion). ⟨ḃ⟩ = R·⟨∂h/∂t⟩ + ⟨∇·(Hu)⟩ - ⟨ȧ⟩.
    # ⟨∂h/∂t⟩ is the basin-mean of per-pixel OLS slopes (robust IRLS) of
    # h_stack, NOT the OLS slope of basin-mean h(t) — the latter is
    # dominated by per-epoch coverage-population swings (different REMA
    # strips cover different parts of the basin at different freeboards).
    if recover_dc:
        from ..kinematics import dh_dt as _kine_dh_dt
        reg_pix = _kine_dh_dt(h_stack, min_count=3, robust=True)
        dh_pix_si = reg_pix["slope"].values
        if floating_mask is not None:
            fmask_arr = floating_mask.values.astype(bool)
        else:
            fmask_arr = np.ones(dh_pix_si.shape, dtype=bool)
        finite_pix = np.isfinite(dh_pix_si) & fmask_arr
        if finite_pix.any():
            mean_dh_dt_si = float(dh_pix_si[finite_pix].mean())
            R_hydro = rho_w / (rho_w - rho_i)
            if floating_mask is not None:
                fd_mean_yr = float(fd.where(floating_mask).mean(skipna=True))
            else:
                fd_mean_yr = float(fd.mean(skipna=True))
            if not np.isfinite(fd_mean_yr):
                fd_mean_yr = 0.0
            m_dc_yr = (
                R_hydro * mean_dh_dt_si * SECONDS_PER_YEAR
                + fd_mean_yr
                - float(a_dot_dc)
            )
            if floating_mask is not None:
                cur_mean = float(m.where(floating_mask).mean(skipna=True))
            else:
                cur_mean = float(m.mean(skipna=True))
            m = m - cur_mean + m_dc_yr

    return xr.Dataset(
        {
            "melt_rate": m.rename("melt_rate"),
            "dh_dt": dh_dt,
            "n_obs": n_obs_da,
            "H_f_mean": H_f_mean,
            "flux_div": fd,
            "a_dot": a_dot_field,
        },
        attrs={
            "equation": (
                "Stubblefield 2023 linearized inverse, dh/dt reformulation: "
                "per-pixel OLS slope of h_lag(t), then single-step Fourier inverse"
            ),
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref_val,
            "gamma_dimless": gamma_val,
            "tr_yr": tr / SECONDS_PER_YEAR,
            "alpha": 0.0, "theta": theta, "reg": reg,
            "dt_yr": dt_yr,
            "transform": transform,
            "min_n_obs": int(min_n_obs),
            "method": "dh_dt single-step Fourier inverse, Lagrangian frame",
        },
    )

G_GRAVITY = 9.81


def _nan_aware_gaussian_infill_2d(
    arr2d: np.ndarray,
    sigma_pix: tuple[float, float],
    max_iters: int = 5,
    support_threshold: float = 0.05,
) -> np.ndarray:
    r"""Iteratively widen a NaN-valued array's data support by Gaussian-weighted averaging.

    Each pass replaces NaN cells whose Gaussian-neighborhood mass exceeds
    ``support_threshold`` with the weighted-mean of nearby data. Filled cells
    contribute to the next pass at unit weight, so support grows by ~2.4 σ
    per pass and bridges interior DEM-strip holes that would otherwise become
    step cliffs in the FFT input. After the iterations, any cell still NaN
    has no nearby data and is set to zero anomaly — matching the v1
    ``fillna(0)`` semantics in deep no-data interior, but only after a smooth
    transition from the data envelope.
    """
    out = arr2d.astype(np.float64).copy()
    for _ in range(int(max_iters)):
        finite = np.isfinite(out)
        if finite.all():
            return out
        arr_z = np.where(finite, out, 0.0)
        weight = finite.astype(np.float64)
        sm_num = to_numpy(
            gaussian_filter(asarray(arr_z), sigma=sigma_pix, mode="nearest")
        )
        sm_den = to_numpy(
            gaussian_filter(asarray(weight), sigma=sigma_pix, mode="nearest")
        )
        fillable = (~finite) & (sm_den > support_threshold)
        if not fillable.any():
            break
        infilled_values = sm_num / np.maximum(sm_den, 1e-12)
        out = np.where(fillable, infilled_values, out)
    return np.where(np.isfinite(out), out, 0.0)


def _pad_xy_dataarray(
    da: xr.DataArray, P: int, fill_value: float = np.nan
) -> xr.DataArray:
    r"""Pad the (y, x) dims of a (time, y, x) or (y, x) DataArray by ``P`` cells per side.

    Coordinates are extended linearly (y descending, x ascending) so the
    returned grid is uniformly spaced. The pad region is filled with
    ``fill_value`` (NaN by default) so the downstream infill+FFT path treats
    pad cells as missing data, never as fake zero observations.
    """
    if P <= 0:
        return da
    y_coords = da["y"].values
    x_coords = da["x"].values
    dy = float(y_coords[0] - y_coords[1])  # descending
    dx = float(x_coords[1] - x_coords[0])

    y_pad_pre = y_coords[0] + dy * np.arange(P, 0, -1)
    y_pad_post = y_coords[-1] - dy * np.arange(1, P + 1)
    y_ext = np.concatenate([y_pad_pre, y_coords, y_pad_post])

    x_pad_pre = x_coords[0] - dx * np.arange(P, 0, -1)
    x_pad_post = x_coords[-1] + dx * np.arange(1, P + 1)
    x_ext = np.concatenate([x_pad_pre, x_coords, x_pad_post])

    if "time" in da.dims:
        n_t = da.sizes["time"]
        ny0 = da.sizes["y"]
        nx0 = da.sizes["x"]
        padded = np.full(
            (n_t, ny0 + 2 * P, nx0 + 2 * P), fill_value, dtype=np.float64
        )
        padded[:, P : P + ny0, P : P + nx0] = da.values
        coords = {"time": da["time"].values, "y": y_ext, "x": x_ext}
        dims = ("time", "y", "x")
    else:
        ny0 = da.sizes["y"]
        nx0 = da.sizes["x"]
        padded = np.full((ny0 + 2 * P, nx0 + 2 * P), fill_value, dtype=np.float64)
        padded[P : P + ny0, P : P + nx0] = da.values
        coords = {"y": y_ext, "x": x_ext}
        dims = ("y", "x")
    return xr.DataArray(
        padded, dims=dims, coords=coords, name=da.name, attrs=da.attrs
    )


def lagrangian_frame_stack(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    dt_yr: float = 0.05,
) -> xr.DataArray:
    r"""Resample a surface-elevation stack into a co-moving reference frame.

    Particles seeded at every pixel of ``h_stack.isel(time=0)`` are advected
    forward through the time-mean velocity field. At each subsequent epoch
    the surface is sampled at the particle's current position, so the
    returned stack ``h_lag(t, y, x)`` shares the spatial grid of the first
    epoch and ``h_lag.isel(time=0) == h_stack.isel(time=0)`` exactly.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Eulerian-frame surface elevation stack, meters.
    vx, vy : xarray.DataArray
        Velocity field, m/yr. If carrying a ``time`` dimension the
        time-mean is used. EPSG:3031 sign convention (x ascends,
        y descends).
    dt_yr : float
        Forward-Euler integration sub-step in years. Smaller values
        reduce integration error along curved trajectories.

    Returns
    -------
    xarray.DataArray, dims ``(time, y, x)``
        ``h_stack`` resampled at trajectory positions. Pixels whose
        trajectory leaves the domain are NaN.
    """
    if "time" in vx.dims:
        vx = vx.mean("time", skipna=True)
    if "time" in vy.dims:
        vy = vy.mean("time", skipna=True)

    x_coords = h_stack["x"].values
    y_coords = h_stack["y"].values
    ny = h_stack.sizes["y"]
    nx = h_stack.sizes["x"]
    res_x = float(x_coords[1] - x_coords[0])
    res_y = float(y_coords[0] - y_coords[1])  # descending

    h_arr = asarray(np.asarray(h_stack.values, dtype=np.float64))
    vx_arr = asarray(np.asarray(vx.values, dtype=np.float64))
    vy_arr = asarray(np.asarray(vy.values, dtype=np.float64))

    times = pd.to_datetime(h_stack["time"].values)
    t_years = np.array(
        [(t - times[0]).total_seconds() / SECONDS_PER_YEAR for t in times],
        dtype=np.float64,
    )
    T = len(times)

    h_traj = np.full((T, ny, nx), np.nan, dtype=np.float64)
    h_traj[0] = to_numpy(h_arr[0])

    ji, ii = xp.mgrid[0:ny, 0:nx]
    y0 = ji.astype(xp.float64)
    x0 = ii.astype(xp.float64)

    for k in range(1, T):
        dt_total = t_years[k]
        n_steps = max(1, int(np.ceil(dt_total / dt_yr)))
        actual_dt = dt_total / n_steps

        x_idx = x0.copy()
        y_idx = y0.copy()
        for _ in range(n_steps):
            vx_t = map_coordinates(vx_arr, [y_idx, x_idx], order=1, mode="nearest")
            vy_t = map_coordinates(vy_arr, [y_idx, x_idx], order=1, mode="nearest")
            x_idx = x_idx + vx_t * actual_dt / res_x
            y_idx = y_idx - vy_t * actual_dt / res_y

        h_k = map_coordinates(
            h_arr[k], [y_idx, x_idx], order=1, mode="constant", cval=float("nan")
        )
        h_traj[k] = to_numpy(h_k)

    return xr.DataArray(
        h_traj,
        dims=("time", "y", "x"),
        coords={
            "time": h_stack["time"].values,
            "y": y_coords,
            "x": x_coords,
        },
        name="h_lagrangian",
        attrs={
            "frame": "Lagrangian (co-moving with time-mean velocity)",
            "anchor_epoch": str(h_stack["time"].values[0]),
            "integrator": "forward-Euler",
            "dt_yr": dt_yr,
        },
    )


def linear_inverse_lagrangian_melt_rate(
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
    reg: float = 1e-3,
    dt_yr: float = 0.05,
    d: xr.DataArray | float = 0.0,
    prefilter_sigma_m: float | None = None,
    localize: bool = True,
    localize_ring_frac: float = 0.8,
    boundary_fix: str = "off",
    infill_sigma_m: float | None = None,
    infill_max_iters: int = 5,
    infill_support_threshold: float = 0.05,
    pad_pixels: int | None = None,
    transform: str = "fft",
    h_lag_precomputed: xr.DataArray | None = None,
    recover_dc: bool = True,
    a_dot_dc: float = 0.0,
) -> xr.Dataset:
    r"""Recover stationary basal melt rate via the Stubblefield inverse in
    a Lagrangian reference frame.

    Pipeline:

    1. Build a Lagrangian-frame surface stack by tracing particles forward
       from the first observation epoch through the time-mean velocity
       field (:func:`lagrangian_frame_stack`).
    2. Form the anomaly ``h_anom = h_lag - h_lag.isel(time=0)``, which
       satisfies the ``h(t=0) = 0`` convention of
       :func:`inverse_stationary` exactly.
    3. Pick a tile-mean reference thickness ``H_ref`` (from the
       hydrostatic-converted time-mean surface unless supplied) and a
       tile-mean dimensionless extension ``γ = E·t_r`` (from
       ``⟨∇·u⟩`` over the floating mask unless supplied).
    4. Run :func:`inverse_stationary` with ``α = 0`` (the bulk advection
       has been absorbed into the frame transformation) to recover
       ``m(ξ, η)`` on the reference grid.

    Because the reference grid coincides with the Eulerian grid at the
    first epoch, the recovered ``m`` is on the original
    ``(y, x)`` grid without further resampling. It is best interpreted
    as the trajectory-averaged basal melt rate at the particle currently
    sitting at that position at ``t = 0``.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Geoid-referenced, corrected surface elevation stack, meters.
    vx, vy : xarray.DataArray
        Velocity field on the ``(y, x)`` grid, m/yr.
    floating_mask : xarray.DataArray, optional
        Boolean mask of floating ice. If supplied, used both to mask the
        recovered melt rate and to restrict the spatial means used for
        ``H_ref`` and ``γ``.
    H_ref : float or xarray.DataArray, optional
        Reference ice-equivalent thickness, meters. If a DataArray,
        the spatial mean over the floating mask is used. If None,
        defaults to the spatial mean of the hydrostatic-converted
        time-mean surface over the floating mask.
    eta_bar : float
        Column-averaged dynamic viscosity, Pa s.
    rho_i, rho_w : float
        Ice and seawater densities, kg/m^3.
    g : float
        Gravitational acceleration, m/s^2.
    gamma : float, optional
        Dimensionless extensional parameter ``E·t_r``. If None,
        derived from ``⟨∇·u⟩`` over the floating tile.
    theta : float
        Long-wavelength regularization for the Stubblefield kernels.
    reg : float
        Per-wavenumber Tikhonov regularization in
        :func:`inverse_stationary`.
    dt_yr : float
        Forward-Euler sub-step for trajectory integration.
    d : xarray.DataArray or float
        Firn air content, meters. Used only to derive ``H_f_mean``
        diagnostic and the default ``H_ref``.
    prefilter_sigma_m : float, optional
        Standard deviation in meters of a Gaussian pre-filter applied
        per epoch to ``h_anom`` before the FFT. Two physical jobs:
        (i) attenuates sub-``H`` grid noise --- the Stubblefield model
        cannot resolve features smaller than ``H`` anyway --- and
        (ii) softens the floating-mask boundary discontinuity that
        otherwise injects broadband ringing into the FFT input. Default
        ``H_ref / 2``. Pass ``0`` to disable.
    localize : bool, default True
        If True, per epoch subtract the spatial mean of ``h_anom`` over
        the outer ring of the floating mask (cells whose radial distance
        from the floating-mask centroid exceeds
        ``localize_ring_frac * r_max``). Mirrors the ``localize()`` step
        in agstub/lake-altimetry-inversions. Removes any spatially-
        uniform temporal trend (e.g. regional ice thinning) so the
        field at the floating-mask boundary is ≈ 0; the subsequent
        ``fillna(0)`` cliff becomes small and FFT periodic-BC ringing
        is dramatically reduced.
    localize_ring_frac : float, default 0.8
        Inner radial fraction of the floating-mask extent below which
        cells are excluded from the outer-ring mean. ``0.8`` matches
        the lake-altimetry default and keeps the outermost ~20 % of
        the radial extent for the trend estimate.
    transform : {"fft", "dct"}
        Spatial basis for the closed-form Stubblefield inverse. ``"fft"``
        (default) uses the periodic 2-D FFT and is the historical path —
        usually paired with ``boundary_fix="infill+pad"`` to keep the
        periodic wrap from coupling the data envelope to its mirror.
        ``"dct"`` uses the DCT-II reflective basis, which extends the
        signal by mirroring rather than wrapping; this matches "no
        information past the data envelope" and removes the need for
        padding entirely. Pair ``"dct"`` with ``boundary_fix="off"`` for
        the simplest, fastest closed-form path.

    Returns
    -------
    xarray.Dataset
        Variables on the ``(y, x)`` grid:

        - ``melt_rate`` --- recovered stationary basal mass balance,
          m ice yr^-1 (Shean convention: negative = melt,
          positive = accretion)
        - ``H_f_mean`` --- time-mean ice-equivalent thickness, m
        - ``flux_div`` --- ``∇·(H_f u)`` on the time-mean field
          (diagnostic parity with the other two solvers)
        - ``a_dot`` --- broadcast SMB rate (zeros here; the Stubblefield
          model accounts for surface mass balance only implicitly through
          the steady-state assumption)
    """
    if "time" not in h_stack.dims:
        raise ValueError("linear_inverse_lagrangian_melt_rate requires a 'time' dim on h_stack")
    if h_stack.sizes["time"] < 2:
        raise ValueError("need at least two epochs for the linear inverse")

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
            f"derived H_ref={H_ref_val} is not a positive finite number; "
            "supply H_ref explicitly"
        )

    tr = 2.0 * eta_bar / (rho_i * g * H_ref_val)
    if gamma is None:
        div_u = divergence(vx_mean, vy_mean)
        if floating_mask is not None:
            div_per_yr = float(div_u.where(floating_mask).mean(skipna=True))
        else:
            div_per_yr = float(div_u.mean(skipna=True))
        if not np.isfinite(div_per_yr):
            div_per_yr = 0.0
        gamma_val = (div_per_yr / SECONDS_PER_YEAR) * tr
    else:
        gamma_val = float(gamma)

    # Per-pixel time-mean anomaly. REMA strip coverage at any single
    # epoch is patchy (often <30% of the floating mask at Beardmore),
    # so anchoring to the first epoch zeros out every pixel the anchor
    # didn't observe. Subtracting the per-pixel mean preserves the
    # original per-epoch finite fraction.
    h_mean = h_lag.mean("time", skipna=True)
    h_anom = h_lag - h_mean

    # Localize: subtract per-epoch spatial mean over the outer ring of the
    # floating mask. Lifted from agstub/lake-altimetry-inversions; pulls
    # the field near the floating-mask boundary toward zero so the
    # subsequent fillna(0) cliff is small and FFT periodic-BC ringing
    # collapses.
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

    valid_modes = ("off", "infill", "pad", "infill+pad")
    if boundary_fix not in valid_modes:
        raise ValueError(
            f"boundary_fix must be one of {valid_modes}, got {boundary_fix!r}"
        )
    use_infill = boundary_fix in ("infill", "infill+pad")
    use_pad = boundary_fix in ("pad", "infill+pad")

    x_coords0 = h_anom["x"].values
    y_coords0 = h_anom["y"].values
    dx = float(abs(x_coords0[1] - x_coords0[0]))
    dy = float(abs(y_coords0[0] - y_coords0[1]))
    ny0 = h_anom.sizes["y"]
    nx0 = h_anom.sizes["x"]

    if use_pad:
        P = (
            int(np.ceil(2.0 * H_ref_val / dx))
            if pad_pixels is None
            else int(pad_pixels)
        )
        if P < 0:
            raise ValueError(f"pad_pixels must be >= 0, got {P}")
        h_in = _pad_xy_dataarray(h_anom, P, fill_value=np.nan)
    else:
        P = 0
        h_in = h_anom

    if use_infill:
        sigma_infill_m = (
            float(H_ref_val) if infill_sigma_m is None else float(infill_sigma_m)
        )
        sigma_infill_pix = (sigma_infill_m / dy, sigma_infill_m / dx)
        infilled = np.empty_like(h_in.values, dtype=np.float64)
        for ti in range(h_in.sizes["time"]):
            arr = h_in.isel(time=ti).values.astype(np.float64)
            infilled[ti] = _nan_aware_gaussian_infill_2d(
                arr,
                sigma_infill_pix,
                max_iters=int(infill_max_iters),
                support_threshold=float(infill_support_threshold),
            )
        h_in = xr.DataArray(
            infilled,
            dims=h_in.dims,
            coords=h_in.coords,
            name=h_in.name,
            attrs=h_in.attrs,
        )
    else:
        sigma_infill_m = 0.0
        h_in = h_in.fillna(0.0)

    sigma_m = H_ref_val / 2.0 if prefilter_sigma_m is None else float(prefilter_sigma_m)
    if sigma_m > 0:
        sigma_pix = (sigma_m / dy, sigma_m / dx)
        # Smooth each epoch independently along the spatial axes only.
        smoothed = np.empty_like(h_in.values)
        for ti in range(h_in.sizes["time"]):
            arr = asarray(h_in.isel(time=ti).values.astype(np.float64))
            smoothed[ti] = to_numpy(gaussian_filter(arr, sigma=sigma_pix, mode="nearest"))
        h_in = xr.DataArray(
            smoothed, dims=h_in.dims, coords=h_in.coords, name=h_in.name
        )

    if transform not in ("fft", "dct"):
        raise ValueError(f"transform must be 'fft' or 'dct', got {transform!r}")

    m = inverse_stationary(
        h_in,
        H=H_ref_val,
        eta_bar=eta_bar,
        rho_i=rho_i,
        rho_w=rho_w,
        g=g,
        alpha=0.0,
        gamma=gamma_val,
        theta=theta,
        reg=reg,
        reference="time_mean",
        transform=transform,
        recover_dc=False,  # we splice DC at the wrapper level using un-subtracted h_lag
    )

    if P > 0:
        m = m.isel(y=slice(P, P + ny0), x=slice(P, P + nx0))
        m = m.assign_coords(y=y_coords0, x=x_coords0)

    if floating_mask is not None:
        m = m.where(floating_mask)
        H_f_mean = H_f_mean.where(floating_mask)

    fd = flux_divergence(H_f_mean, vx_mean, vy_mean)

    # Mass-balance DC mode (k=0). The Stubblefield kernel zeros R, B at
    # k=0 by construction, so inverse_stationary loses the spatial-mean
    # melt rate. The recovery uses the *Eulerian* mass-balance, in the
    # Shean public convention (positive = accretion):
    #     ⟨ḃ⟩ = R·⟨∂h/∂t⟩ + ⟨∇·(Hu)⟩ - ⟨ȧ⟩      (R = ρ_w/(ρ_w-ρ_i))
    # ⟨∂h/∂t⟩ is the basin-mean of *per-pixel* OLS slopes (with robust
    # IRLS), not the OLS slope of basin-mean h(t). Per-pixel slopes are
    # immune to coverage-population swings: each pixel uses only its
    # own valid epochs. Without this, the basin-mean h(t) jumps around
    # by tens of m as different REMA strips cover different parts of
    # the basin at different freeboards (Nansen 2019-2023: spurious
    # ~−6 m/yr basin-mean slope from population shifts alone).
    if recover_dc:
        from ..kinematics import dh_dt as _kine_dh_dt
        reg_pix = _kine_dh_dt(h_stack, min_count=3, robust=True)
        dh_pix_si = reg_pix["slope"].values  # m/s freeboard, per pixel
        if floating_mask is not None:
            fmask_arr = floating_mask.values.astype(bool)
        else:
            fmask_arr = np.ones(dh_pix_si.shape, dtype=bool)
        finite_pix = np.isfinite(dh_pix_si) & fmask_arr
        if finite_pix.any():
            mean_dh_dt_si = float(dh_pix_si[finite_pix].mean())
            R_hydro = rho_w / (rho_w - rho_i)
            if floating_mask is not None:
                fd_mean_yr = float(fd.where(floating_mask).mean(skipna=True))
            else:
                fd_mean_yr = float(fd.mean(skipna=True))
            if not np.isfinite(fd_mean_yr):
                fd_mean_yr = 0.0
            m_dc_yr = (
                R_hydro * mean_dh_dt_si * SECONDS_PER_YEAR
                + fd_mean_yr
                - float(a_dot_dc)
            )
            if floating_mask is not None:
                cur_mean = float(m.where(floating_mask).mean(skipna=True))
            else:
                cur_mean = float(m.mean(skipna=True))
            m = m - cur_mean + m_dc_yr
    a_dot_field = xr.zeros_like(H_f_mean).rename("a_dot")

    return xr.Dataset(
        {
            "melt_rate": m.rename("melt_rate"),
            "H_f_mean": H_f_mean,
            "flux_div": fd,
            "a_dot": a_dot_field,
        },
        attrs={
            "equation": (
                "Stubblefield 2023 linearized inverse in Lagrangian frame; "
                "alpha=0 by construction, gamma from <div(u)>"
            ),
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "rho_w": rho_w,
            "rho_i": rho_i,
            "eta_bar": eta_bar,
            "H_ref_m": H_ref_val,
            "gamma_dimless": gamma_val,
            "tr_yr": tr / SECONDS_PER_YEAR,
            "alpha": 0.0,
            "theta": theta,
            "reg": reg,
            "dt_yr": dt_yr,
            "prefilter_sigma_m": sigma_m,
            "localize": bool(localize),
            "localize_ring_frac": float(localize_ring_frac) if localize else 0.0,
            "boundary_fix": boundary_fix,
            "infill_sigma_m": float(sigma_infill_m),
            "infill_max_iters": int(infill_max_iters) if use_infill else 0,
            "pad_pixels": int(P),
            "transform": transform,
            "anchor_epoch": str(h_stack["time"].values[0]),
        },
    )
