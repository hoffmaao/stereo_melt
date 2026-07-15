"""Sanity test for the Lagrangian-frame Stubblefield inverse.

Two synthetic checks:

1. **No advection** (`vx = vy = 0`, `alpha = 0`). The Lagrangian frame
   coincides with the Eulerian frame, so the new solver should recover
   the prescribed stationary `m` to the same accuracy as
   :func:`inverse_stationary` itself.

2. **Uniform along-x advection** (`vx = u_x > 0`, prescribed melt
   forward-modelled with the matching `alpha = u_x * t_r / H`). The new
   solver runs internally with `alpha = 0` because the bulk advection
   is absorbed into the trajectory transformation. Recovery of the
   prescribed `m` should be close (within ~20%) for a window short
   enough that trajectory-averaging doesn't smooth out the Gaussian.
"""

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics import (
    forward,
    linear_inverse_lagrangian_melt_rate,
)

print("Lagrangian inverse module imports OK.")

# ---- Geometry / physics ----
H = 500.0  # m
eta_bar = 1e14  # Pa s
g = 9.81
delta = rhow / rhoi - 1.0
tr_s = 2.0 * eta_bar / (rhoi * g * H)
SECONDS_PER_YEAR = 86400.0 * 365.25
tr_yr = tr_s / SECONDS_PER_YEAR
print(f"H={H} m, eta_bar={eta_bar:.1e} Pa s, t_r={tr_yr:.1f} yr, delta={delta:.4f}")

# Domain: 60 km wide, 250 m grid (matches sanity_linear_perturbation)
L = 60_000.0
res = 250.0
x = np.arange(-L / 2, L / 2, res)
y = np.arange(L / 2, -L / 2, -res)  # descending (EPSG:3031)
X, Y = np.meshgrid(x, y)

# Prescribed stationary melt: wide Gaussian (sigma = 10/3 H ≈ 1.67 km),
# amplitude ~5 m/yr — same regime as the existing sanity test.
# m_true is in Stubblefield convention (positive=melt) since the
# library forward operator (forward(), steady_state()) takes positive
# m as a melt forcing. The library's PUBLIC inverse wrappers return
# Shean convention (negative=melt) — define m_true_shean for the
# round-trip comparison.
sigma = (10.0 / 3.0) * H
m0_dim_m_per_yr = 0.014 * H / tr_s * SECONDS_PER_YEAR
m_arr = m0_dim_m_per_yr * np.exp(-0.5 * ((X) ** 2 + (Y) ** 2) / sigma**2)
m_true = xr.DataArray(m_arr, dims=("y", "x"), coords={"y": y, "x": x}, name="m")
m_true_shean = -m_true  # Shean public convention: negative = melt
print(f"Prescribed m: peak={m0_dim_m_per_yr:.2f} m/yr, sigma={sigma/1000:.2f} km")

# Observation epochs: 5 evenly-spaced epochs over one t_r so the stack
# has clear time-build-up of the surface anomaly.
n_t = 5
t_window_s = tr_s
times_s = np.linspace(0.0, t_window_s, n_t)
times_dt = pd.to_datetime("2020-01-01") + pd.to_timedelta(times_s, unit="s")
print(f"Observation epochs: {n_t} over {t_window_s/SECONDS_PER_YEAR:.1f} yr")


def run_recovery(alpha_forward: float, vx_value: float, label: str) -> tuple[float, float]:
    """Forward-model h_stack with alpha_forward, invert in Lagrangian frame.

    Returns (peak_recovery_ratio, rmse_in_central_window).
    """
    # 1. Forward: build h_stack(t, y, x) on Eulerian grid using Stubblefield
    h_stack_xr = forward(
        m_true,
        H=H,
        eta_bar=eta_bar,
        alpha=alpha_forward,
        gamma=0.0,
        stationary=True,
        times=xr.DataArray(times_s, dims="t"),
        return_basal=False,
    )
    # forward() emits a 'time' dim with whatever we passed in; rebuild with
    # the datetime axis the new solver expects.
    h_stack = xr.DataArray(
        h_stack_xr.values,
        dims=("time", "y", "x"),
        coords={"time": times_dt, "y": y, "x": x},
        name="h",
    )

    # 2. Velocity field (uniform along-x for the advection branch)
    vx = xr.DataArray(
        np.full_like(X, vx_value, dtype=np.float64),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )
    vy = xr.zeros_like(vx)

    # 3. Run new solver
    out = linear_inverse_lagrangian_melt_rate(
        h_stack,
        vx,
        vy,
        H_ref=H,
        eta_bar=eta_bar,
        gamma=0.0,        # we know it's zero in the synthetic
        reg=1e-3,
        dt_yr=0.05,
    )
    m_rec = out.melt_rate

    # Recovery diagnostics on the central 30 km × 30 km box (avoid FFT-edge
    # effects from the periodic assumption). Inverse output is Shean
    # convention (negative = melt) so we compare against m_true_shean
    # (= -m_true). The Gaussian's "peak" in Shean convention is its most
    # negative value at the center.
    box = (np.abs(X) < L / 4) & (np.abs(Y) < L / 4)
    peak_ratio = float(m_rec.values[box].min() / m_true_shean.values[box].min())
    rmse = float(np.sqrt(np.nanmean((m_rec.values[box] - m_true_shean.values[box]) ** 2)))
    rel_rmse = rmse / m0_dim_m_per_yr
    print(
        f"\n[{label}] alpha_fwd={alpha_forward:.3f}, vx={vx_value:.1f} m/yr "
        f"(implied alpha_lag=0)\n"
        f"  peak m_rec / m_true = {peak_ratio:.3f} (target ~1.0)\n"
        f"  RMSE in central box = {rmse:.3f} m/yr  ({100*rel_rmse:.1f}% of peak)"
    )
    return peak_ratio, rel_rmse


# ---- Test 1: no advection ----
peak1, rel_rmse1 = run_recovery(alpha_forward=0.0, vx_value=0.0, label="no advection")
assert abs(peak1 - 1.0) < 0.05, f"no-advection peak recovery off: {peak1:.3f}"
assert rel_rmse1 < 0.05, f"no-advection rmse too high: {rel_rmse1:.3f}"
print("  PASS: no-advection recovery within 5%.")

# ---- Test 2: uniform along-x advection ----
# u_x chosen so total displacement over the window is sigma/4 — keeps
# trajectory-averaging from heavily smoothing the Gaussian.
u_x = (sigma / 4.0) / (t_window_s / SECONDS_PER_YEAR)  # m/yr
alpha_fwd = u_x * (1.0 / SECONDS_PER_YEAR) * tr_s / H
peak2, rel_rmse2 = run_recovery(alpha_forward=alpha_fwd, vx_value=u_x, label="uniform u_x")
assert abs(peak2 - 1.0) < 0.20, f"with-advection peak recovery off: {peak2:.3f}"
assert rel_rmse2 < 0.20, f"with-advection rmse too high: {rel_rmse2:.3f}"
print("  PASS: with-advection recovery within 20% (frame transform absorbs alpha).")

# ---- Test 3: time_mean reference handles per-pixel mean-subtracted input ----
# Forward-model a no-advection h_stack, then mean-subtract per pixel before
# inverting with reference="time_mean". Should recover the prescribed m
# essentially as well as the first-epoch reference.
from stereo_melt.dynamics import inverse_stationary

h_stack_xr = forward(
    m_true, H=H, eta_bar=eta_bar, alpha=0.0, gamma=0.0,
    stationary=True, times=xr.DataArray(times_s, dims="t"),
)
h_stack = xr.DataArray(
    h_stack_xr.values,
    dims=("time", "y", "x"),
    coords={"time": times_dt, "y": y, "x": x},
)
h_anom_mean = h_stack - h_stack.mean("time", skipna=True)
m_rec_mean = inverse_stationary(
    h_anom_mean.fillna(0.0), H=H, eta_bar=eta_bar,
    alpha=0.0, gamma=0.0, reg=1e-3, reference="time_mean",
)
box = (np.abs(X) < L / 4) & (np.abs(Y) < L / 4)
# inverse_stationary returns Shean convention; compare against m_true_shean.
peak3 = float(m_rec_mean.values[box].min() / m_true_shean.values[box].min())
rmse3 = float(np.sqrt(np.nanmean((m_rec_mean.values[box] - m_true_shean.values[box]) ** 2)))
rel3 = rmse3 / m0_dim_m_per_yr
print(
    f"\n[time_mean reference] no-advection + per-pixel mean anomaly\n"
    f"  peak m_rec / m_true = {peak3:.3f}\n"
    f"  RMSE in central box = {rmse3:.3f} m/yr  ({100*rel3:.1f}% of peak)"
)
assert abs(peak3 - 1.0) < 0.05, f"time_mean peak recovery off: {peak3:.3f}"
assert rel3 < 0.05, f"time_mean rmse too high: {rel3:.3f}"
print("  PASS: time_mean reference recovers within 5%.")

print("\nAll Lagrangian-inverse sanity checks passed.")
