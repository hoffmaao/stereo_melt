"""Probe: does fit_tilt_stack with observation_mask=full grid actually
recover the injected per-strip tilt coefficients (αx, αy, αz)?

This bypasses the inverses and inspects params.tilt_dx/dy/dz directly.
The synthetic test showed mean_err didn't change much between B and C;
that could mean either (a) the tilts cancel in the per-pixel OLS by
accident, or (b) the LSQ isn't recovering the tilts. This separates them.
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.coregister.tilt import fit_tilt_stack
from stereo_melt.dynamics import forward

SECONDS_PER_YEAR = 86400.0 * 365.25


def main():
    H = 500.0
    n_t = 11
    t_window_yr = 4.0
    times_s = np.linspace(0.0, t_window_yr * SECONDS_PER_YEAR, n_t)
    times_dt = pd.to_datetime("2020-01-01") + pd.to_timedelta(times_s, unit="s")

    L = 60_000.0
    res = 500.0
    x = np.arange(-L / 2, L / 2, res)
    y = np.arange(L / 2, -L / 2, -res)
    X, Y = np.meshgrid(x, y)
    m_true = 3.0 + 5.0 * np.exp(-0.5 * ((X - 8000) ** 2 + (Y + 6000) ** 2) / 4000.0**2)
    m_da = xr.DataArray(m_true, dims=("y", "x"), coords={"y": y, "x": x})

    # Forward (noiseless)
    h_xr = forward(
        m_da, H=H, alpha=0.0, gamma=0.0,
        stationary=True, times=xr.DataArray(times_s, dims="t"),
    )
    h_clean = xr.DataArray(
        h_xr.values, dims=("time", "y", "x"),
        coords={"time": times_dt, "y": y, "x": x},
    )

    # Pin static-region pixels to t=0 — real rock is time-invariant. Without
    # this, every pixel evolves under `forward()` and the rock region carries
    # no DC information, leaving αz_mean unidentifiable in the joint LSQ.
    ny, nx = m_true.shape
    static_mask = np.zeros((ny, nx), dtype=bool)
    static_mask[:, : nx // 2] = True
    h_clean.values[1:, static_mask] = h_clean.values[0, static_mask]
    print(f"pinned static rock to t=0 over left half ({static_mask.mean():.0%} of grid)")

    # Inject KNOWN tilts
    rng = np.random.default_rng(42)
    ax_true = rng.normal(0, 0.5, n_t)
    ay_true = rng.normal(0, 0.5, n_t)
    az_true = rng.normal(0, 0.5, n_t)
    print(f"injected: αz mean={az_true.mean():+.4f}, std={az_true.std():.3f}")
    print(f"          αx std={ax_true.std():.3f}, αy std={ay_true.std():.3f}")

    XN = (x[None, :] - x.mean()) / (0.5 * (x.max() - x.min()))
    YN = (y[:, None] - y.mean()) / (0.5 * (y.max() - y.min()))
    h_noisy = h_clean.values.copy()
    for k in range(n_t):
        h_noisy[k] = h_noisy[k] + ax_true[k] * XN + ay_true[k] * YN + az_true[k]
    h_noisy = xr.DataArray(h_noisy, dims=h_clean.dims, coords=h_clean.coords)

    # 30% rectangular coverage (same recipe as the synthetic test)
    rng2 = np.random.default_rng(11)
    for k in range(n_t):
        target = int(0.3 * ny * nx)
        h_box = max(int(np.sqrt(target * rng2.uniform(0.5, 2.0))), 8)
        h_box = min(h_box, ny)
        w_box = max(target // h_box, 8)
        w_box = min(w_box, nx)
        y0 = rng2.integers(0, ny - h_box + 1)
        x0 = rng2.integers(0, nx - w_box + 1)
        keep = np.zeros((ny, nx), dtype=bool)
        keep[y0:y0 + h_box, x0:x0 + w_box] = True
        h_noisy.values[k] = np.where(keep, h_noisy.values[k], np.nan)

    static_da = xr.DataArray(static_mask, dims=("y", "x"), coords={"y": y, "x": x})
    obs_full = xr.DataArray(np.ones((ny, nx), dtype=bool), dims=("y", "x"),
                            coords={"y": y, "x": x})

    # Run BOTH flavours
    print("\n--- B. static-only LSQ ---")
    params_b, h_corr_b = fit_tilt_stack(
        h_noisy, control_mask=static_da, min_width=5000.0,
    )
    az_b = params_b["tilt_dz"].values
    ax_b = params_b["tilt_dx"].values
    ay_b = params_b["tilt_dy"].values

    print("\n--- C. joint LSQ (full obs mask) ---")
    params_c, h_corr_c = fit_tilt_stack(
        h_noisy, control_mask=static_da, observation_mask=obs_full, min_width=5000.0,
    )
    az_c = params_c["tilt_dz"].values
    ax_c = params_c["tilt_dx"].values
    ay_c = params_c["tilt_dy"].values

    # αx / αy units: tilt_dx is m / m (slope per meter of x, after centering).
    # Injected ax_true is in m, applied as ax * X_norm where X_norm in [-1, +1].
    # So injected slope-per-m = ax_true / (0.5 * Lx).
    half_x = 0.5 * (x.max() - x.min())
    half_y = 0.5 * (y.max() - y.min())
    ax_inj_perm = ax_true / half_x
    ay_inj_perm = ay_true / half_y

    print(f"\nαz (offset, m):")
    print(f"  injected:  mean={az_true.mean():+.4f}, std={az_true.std():.3f}")
    print(f"  rec (B):   mean={az_b.mean():+.4f}, std={az_b.std():.3f}, rms_err={np.sqrt(np.mean((az_b - az_true)**2)):.3f}")
    print(f"  rec (C):   mean={az_c.mean():+.4f}, std={az_c.std():.3f}, rms_err={np.sqrt(np.mean((az_c - az_true)**2)):.3f}")

    print(f"\nαx (slope per meter):")
    print(f"  injected:  std={ax_inj_perm.std():.2e}")
    print(f"  rec (B):   std={ax_b.std():.2e}, rms_err={np.sqrt(np.mean((ax_b - ax_inj_perm)**2)):.2e}")
    print(f"  rec (C):   std={ax_c.std():.2e}, rms_err={np.sqrt(np.mean((ax_c - ax_inj_perm)**2)):.2e}")

    print(f"\nαy (slope per meter):")
    print(f"  injected:  std={ay_inj_perm.std():.2e}")
    print(f"  rec (B):   std={ay_b.std():.2e}, rms_err={np.sqrt(np.mean((ay_b - ay_inj_perm)**2)):.2e}")
    print(f"  rec (C):   std={ay_c.std():.2e}, rms_err={np.sqrt(np.mean((ay_c - ay_inj_perm)**2)):.2e}")

    # Also check: how close is the corrected stack to the clean ground truth?
    diff_b = h_corr_b.values - h_clean.values
    diff_c = h_corr_c.values - h_clean.values
    fb = np.isfinite(diff_b); fc = np.isfinite(diff_c)
    print(f"\n(corrected − clean) over finite pixels (per epoch should be ~zero if tilts removed):")
    print(f"  B:  mean={float(np.mean(diff_b[fb])):+.4f},  std={float(np.std(diff_b[fb])):.3f}")
    print(f"  C:  mean={float(np.mean(diff_c[fc])):+.4f},  std={float(np.std(diff_c[fc])):.3f}")

    # Per-epoch residual (mean of corrected - clean within epoch). If LSQ
    # got αz right, this should be ~0. If it offloaded into per-pixel
    # intercept instead, we'll see structured residuals.
    print("\nper-epoch (corrected − clean) mean and std:")
    print("  k    az_true   az_B    az_C    bias_B   bias_C   std_B   std_C")
    for k in range(n_t):
        d_b = h_corr_b.values[k] - h_clean.values[k]
        d_c = h_corr_c.values[k] - h_clean.values[k]
        f_b = np.isfinite(d_b); f_c = np.isfinite(d_c)
        bb = float(np.mean(d_b[f_b])) if f_b.any() else np.nan
        bc = float(np.mean(d_c[f_c])) if f_c.any() else np.nan
        sb = float(np.std(d_b[f_b])) if f_b.any() else np.nan
        sc = float(np.std(d_c[f_c])) if f_c.any() else np.nan
        print(f"  {k:2d}  {az_true[k]:+7.3f}  {az_b[k]:+7.3f} {az_c[k]:+7.3f}  {bb:+7.3f}  {bc:+7.3f}  {sb:6.3f}  {sc:6.3f}")


if __name__ == "__main__":
    main()
