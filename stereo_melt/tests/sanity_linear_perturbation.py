"""Sanity test for the Stubblefield 2023 linear perturbation forward model.

Reproduces the qualitative behavior of paper Figures 7-8:

- Wide Gaussian melt anomaly (sigma = 10/3 * H): surface h closely
  matches the perfect-flotation prediction -delta*s.
- Narrow Gaussian (sigma = 1/3 * H): surface expression is diminished
  and h != -delta*s (non-hydrostatic).
- Adding across-channel inflow (alpha > 0): surface expression is
  damped and asymmetric.
"""

import numpy as np
import xarray as xr

from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics import steady_state

print("Linear perturbation module imports OK.")

# Paper's reference parameters (Stubblefield 2023, Section 4)
H = 500.0  # m
eta_bar = 1e14  # Pa s
delta = rhow / rhoi - 1.0
print(f"H={H} m, eta_bar={eta_bar:.1e} Pa s, delta={delta:.4f}")

# Paper's m0 = 0.014 (non-dimensional) corresponds to ~5 m/yr
# Convert to dimensional: m_dim = m_ndim * H / tr
tr = 2.0 * eta_bar / (rhoi * 9.81 * H)
m0_dim_m_per_s = 0.014 * H / tr
m0_dim_m_per_yr = m0_dim_m_per_s * 86400.0 * 365.25
print(f"tr = {tr / (86400*365.25):.1f} yr; m0_dim = {m0_dim_m_per_yr:.2f} m/yr")

# Domain: 60 km wide, 250 m grid. Wavenumbers resolve sigma in {10/3, 1/3}*H.
L = 60_000.0
res = 250.0
x = np.arange(-L / 2, L / 2, res)
y = np.arange(L / 2, -L / 2, -res)  # descending (EPSG:3031 convention)
X, Y = np.meshgrid(x, y)


def gaussian_melt(sigma_over_H: float, m0: float) -> xr.DataArray:
    sigma = sigma_over_H * H
    m_arr = m0 * np.exp(-0.5 * (X / sigma) ** 2)
    return xr.DataArray(m_arr, dims=("y", "x"), coords={"y": y, "x": x})


# ---- Wide channel (sigma = 10/3 H) — expect h ~ -delta * s ----
m_wide = gaussian_melt(10.0 / 3.0, m0_dim_m_per_yr)
h_wide, s_wide = steady_state(m_wide, H=H, eta_bar=eta_bar, alpha=0.0, gamma=0.0, return_basal=True)
center = (len(y) // 2, len(x) // 2)
h_c = float(h_wide.values[center])
s_c = float(s_wide.values[center])
print(f"\nWide channel (sigma = 10/3 H):")
print(f"  h_center  = {h_c:.3e} m")
print(f"  s_center  = {s_c:.3e} m")
print(f"  -delta*s  = {-delta * s_c:.3e} m")
print(f"  |h + delta*s| / |h| = {abs(h_c + delta * s_c) / abs(h_c):.3e}")

assert np.sign(h_c) == -np.sign(
    s_c
), f"h and s should have opposite signs for basal melt: h={h_c}, s={s_c}"
assert abs(h_c + delta * s_c) / abs(h_c) < 0.05, "wide channel should be near-hydrostatic"
print("  PASS: wide channel is near-hydrostatic (|h + delta*s|/|h| < 5%).")

# ---- Narrow channel (sigma = 1/3 H) — expect diminished h and h != -delta*s ----
m_narrow = gaussian_melt(1.0 / 3.0, m0_dim_m_per_yr)
h_narrow, s_narrow = steady_state(
    m_narrow, H=H, eta_bar=eta_bar, alpha=0.0, gamma=0.0, return_basal=True
)
h_n = float(h_narrow.values[center])
s_n = float(s_narrow.values[center])
hyd_n = -delta * s_n
rel_dev = abs(h_n - hyd_n) / abs(hyd_n)
print(f"\nNarrow channel (sigma = 1/3 H):")
print(f"  h_center = {h_n:.3e} m")
print(f"  s_center = {s_n:.3e} m")
print(f"  -delta*s = {hyd_n:.3e} m")
print(f"  h / (-delta*s) = {h_n / hyd_n:.3f}  (ideal hydrostatic = 1.0)")

assert np.sign(h_n) == -np.sign(s_n), "narrow channel: h and s opposite signs"
assert (
    rel_dev > 0.05
), f"narrow channel should break hydrostatic flotation but only deviated by {rel_dev:.3e}"
print(f"  PASS: narrow channel breaks hydrostatic flotation ({rel_dev * 100:.1f}% deviation).")

# ---- Effect of inflow alpha -> surface expression is damped ----
h_no_inflow = float(steady_state(m_narrow, H=H, eta_bar=eta_bar, alpha=0.0).values[center])
# alpha = 1/2 in Stubblefield corresponds to ubar_0 ~ 193 m/yr
h_inflow = float(steady_state(m_narrow, H=H, eta_bar=eta_bar, alpha=0.5).values[center])
print(f"\nNarrow channel + inflow:")
print(f"  |h_center| (alpha=0)   = {abs(h_no_inflow):.3e} m")
print(f"  |h_center| (alpha=0.5) = {abs(h_inflow):.3e} m")
print(f"  damping factor = {abs(h_inflow) / abs(h_no_inflow):.3f}")
assert abs(h_inflow) < abs(
    h_no_inflow
), "inflow should diminish the narrow-channel surface expression"
print("  PASS: inflow mutes narrow-channel surface topography.")

# ---- Stationary forward at large t should match the steady state ----
from stereo_melt.dynamics import forward

# Long-wavelength relaxation time scale t_e = 2 tr (1 + 1/delta) ≈ 19 tr
# for delta ≈ 0.11, so a wide-channel steady state needs many t_e.
t_many = tr * 200.0
h_forward = forward(
    m_wide,
    H=H,
    eta_bar=eta_bar,
    stationary=True,
    times=np.array([t_many]),
    return_basal=False,
)
h_ss = steady_state(m_wide, H=H, eta_bar=eta_bar)
ratio = float(h_forward.isel(time=0).values[center]) / float(h_ss.values[center])
print(f"\nforward(t = 200 tr) / steady_state at wide-channel center = {ratio:.4f} (want ~1.0)")
assert abs(ratio - 1.0) < 0.02, f"forward at large t should match steady_state, got ratio={ratio}"
print("  PASS: forward(t -> infty) agrees with steady_state.")

print("\nAll linear-perturbation sanity checks passed.")
