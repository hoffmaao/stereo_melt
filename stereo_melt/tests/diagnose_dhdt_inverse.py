"""Validate the dh/dt reformulation: per-pixel OLS slope + 1-step Fourier inverse.

Reformulation:
  Step 1 (per-pixel, handles coverage gaps natively):
    b(y, x) = OLS slope of h_lag(t) vs t over each pixel's valid epochs
  Step 2 (single 2D field, no temporal masking):
    m̂(k) = b̂(k) · K_dot*(k) / (|K_dot(k)|² + λ²)
  where K_dot(k) is the OLS-slope kernel:
    K_dot(k) = Σ_i (t_i - t̄) · A_i(k) / Σ_i (t_i - t̄)²
    A_i(k) = tr · I_h(k, t_i)

Test cases (matched to diagnose_linear_inverse_coverage.py for direct comparison):
  - Gaussian +5 m/yr blob, sigma=1 km
  - Coverage levels: dense / strip(70%) / strip(50%) / strip(30%)

Expected: dh/dt inverse should recover ≈+5 at center across all coverage levels,
because per-pixel OLS handles each pixel's gap pattern independently.
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import numpy as np
import xarray as xr

from stereo_melt.dynamics.linear_perturbation import (
    LinearPerturbation, _wavenumber_grids, _dct_wavenumber_grids, _dctn, _idctn, forward,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_linear_inverse_coverage import (
    DX, DY, ETA, H_REF, NX, NY, make_strip_masks, r, x, y,
)

SECONDS_PER_YEAR = 86400.0 * 365.25


def per_pixel_ols_slope(h_stack, t_secs):
    """Vectorized per-pixel OLS slope of h(t) vs t, ignoring NaNs.

    Returns
    -------
    slope : (ny, nx) float64, m/s (h/time units)
    n_obs : (ny, nx) int, number of valid epochs per pixel
    """
    n_t, ny, nx = h_stack.shape
    t = t_secs.astype(np.float64).reshape(n_t, 1, 1)  # (n_t, 1, 1)
    h = h_stack.astype(np.float64)
    finite = np.isfinite(h)
    # Treat NaN as zero in sums; mask weights via `finite`.
    h0 = np.where(finite, h, 0.0)
    w = finite.astype(np.float64)

    # Weighted moment sums per pixel
    sum_w = w.sum(axis=0)
    sum_t = (w * t).sum(axis=0)
    sum_h = (w * h0).sum(axis=0)
    sum_tt = (w * t * t).sum(axis=0)
    sum_th = (w * t * h0).sum(axis=0)

    det = sum_w * sum_tt - sum_t * sum_t
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (sum_w * sum_th - sum_t * sum_h) / det
    slope = np.where(sum_w >= 2, slope, np.nan)
    return slope, sum_w.astype(int)


def kdot_kernel(model, kx, ky, t_secs):
    """K_dot(k) = Σ_i (t_i - t̄) · A_i(k) / Σ_i (t_i - t̄)²

    A_i(k) = tr · I_h(k, t_i / tr).  Returns same shape as kx.
    """
    t_centered = t_secs - t_secs.mean()
    denom = float((t_centered ** 2).sum())
    K = np.zeros_like(kx, dtype=np.complex128)
    for i, t_s in enumerate(t_secs):
        I_h, _ = model.kernel_time_integral_stationary(kx, ky, t_s / model.tr)
        A_i = model.tr * I_h
        K = K + t_centered[i] * np.asarray(A_i)
    return K / denom


def inverse_dhdt(dh_dt, H, t_secs, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
                 theta=1e-14, reg=1e-3, transform="fft"):
    """Single-step Fourier inverse of a per-pixel dh/dt field for stationary m.

    Parameters
    ----------
    dh_dt : xr.DataArray, dims (y, x), units m/s
    H : reference thickness (m)
    t_secs : np.ndarray, the observation times in seconds (relative)
    """
    model = LinearPerturbation(
        H=H, eta_bar=eta_bar, alpha=alpha, alpha_y=alpha_y, gamma=gamma, theta=theta,
    )
    x_coords = dh_dt["x"].values
    y_coords = dh_dt["y"].values
    ny, nx = dh_dt.sizes["y"], dh_dt.sizes["x"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))

    if transform == "fft":
        kx, ky = _wavenumber_grids(nx, ny, dx, dy)
    else:
        kx, ky = _dct_wavenumber_grids(nx, ny, dx, dy)
    kx = np.asarray(kx); ky = np.asarray(ky)

    K = kdot_kernel(model, kx, ky, t_secs)
    if transform == "dct":
        K = K.real

    # FFT/DCT of dh_dt (NaN-safe: fill with mean of dh_dt before transform)
    dh = dh_dt.values.astype(np.float64)
    finite_mask = np.isfinite(dh)
    if not finite_mask.any():
        raise ValueError("dh_dt is entirely NaN")
    fill_val = float(dh[finite_mask].mean())
    dh_filled = np.where(finite_mask, dh, fill_val)

    if transform == "fft":
        b_hat = np.fft.fft2(dh_filled)
        m_hat = b_hat * np.conj(K) / (np.abs(K) ** 2 + reg ** 2)
        m_si = np.fft.ifft2(m_hat).real
    else:
        b_hat = _dctn(dh_filled)
        m_hat = b_hat * K / (K * K + reg ** 2)
        m_si = _idctn(m_hat)

    m_m_per_yr = m_si * SECONDS_PER_YEAR
    return xr.DataArray(
        m_m_per_yr, dims=("y", "x"), coords={"y": y_coords, "x": x_coords},
        name="m",
        attrs={"units": "m ice yr^-1; positive = melt", "transform": transform,
               "reg": reg, "H": H},
    )


def run_case(label, m_truth_2d, n_epochs, coverage_kind):
    """Forward + masking + dh/dt inverse, report center vs truth."""
    times = np.arange(n_epochs).astype("datetime64[Y]").astype("datetime64[ns]")
    t_secs = (times - times[0]).astype("timedelta64[s]").astype(np.float64)
    m_da = xr.DataArray(m_truth_2d, dims=("y", "x"), coords={"y": y, "x": x})
    h = forward(
        m_da, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        times=xr.DataArray(times, dims="time"), stationary=True,
    )

    # Apply mask
    if coverage_kind == "dense":
        obs = np.ones((n_epochs, NY, NX), dtype=bool)
    elif coverage_kind.startswith("strip"):
        target_p = float(coverage_kind.split(":")[1])
        # tweak the strip-mask helper to hit target coverage by repeated calls
        rng = np.random.default_rng(7)
        obs = np.zeros((n_epochs, NY, NX), dtype=bool)
        yy_pix, xx_pix = np.mgrid[0:NY, 0:NX]
        for i in range(n_epochs):
            em = np.zeros((NY, NX), dtype=bool)
            attempts = 0
            while em.mean() < target_p * 0.95 and attempts < 12:
                ang = rng.uniform(0, np.pi)
                cx = rng.uniform(0, NX); cy = rng.uniform(0, NY)
                w = rng.uniform(40, 80)
                nrm = (xx_pix - cx) * np.sin(ang) - (yy_pix - cy) * np.cos(ang)
                em |= np.abs(nrm) <= w
                attempts += 1
            obs[i] = em

    h_v = h.values.copy()
    h_v[~obs] = np.nan
    h_sparse = xr.DataArray(h_v, dims=h.dims, coords=h.coords)

    # Per-pixel OLS slope
    slope, n_obs = per_pixel_ols_slope(h_v, t_secs)
    slope_da = xr.DataArray(slope, dims=("y", "x"), coords={"y": y, "x": x})
    slope_da = slope_da.where(n_obs >= 3)

    # Run inverse for FFT and DCT
    m_fft = inverse_dhdt(slope_da, H=H_REF, t_secs=t_secs, transform="fft").values
    m_dct = inverse_dhdt(slope_da, H=H_REF, t_secs=t_secs, transform="dct").values

    cov_pp = obs.mean(axis=0)
    print(f"[{label:<22s}]  mean per-pixel cov={cov_pp.mean():.2f}  "
          f"min n_obs={int(n_obs.min())}/{n_epochs}  "
          f"FFT center={m_fft[NY//2,NX//2]:+.2f} (truth +5.00)  "
          f"DCT center={m_dct[NY//2,NX//2]:+.2f}  "
          f"FFT abs_max={np.abs(m_fft).max():.1f}  "
          f"DCT abs_max={np.abs(m_dct).max():.1f}")


# ----------------------------------------------------------------------
# Test sweep
# ----------------------------------------------------------------------
mB = 5.0 * np.exp(-0.5 * (r / 1000.0) ** 2)

print("Gaussian +5 m/yr melt blob — dh/dt reformulation under coverage gaps")
print("=" * 95)
for n_t in [5, 15, 30]:
    for kind in ["dense", "strip:0.70", "strip:0.50", "strip:0.30"]:
        run_case(f"n={n_t}, {kind}", mB, n_t, kind)
    print()
