"""Reproduce Figure 4 of the COMBINED_Proposal: forward + stationary inverse.

Setup:
  1. Prescribe a stationary Gaussian basal melt-rate anomaly m_true(x, y).
  2. Run the Stubblefield 2023 forward model to get h(x, y, t) at four
     time steps.
  3. Add Gaussian measurement noise.
  4. Invert the noisy h-stack back to a stationary m(x, y) via the
     closed-form per-wavenumber Tikhonov LSQ (dynamics.inverse_stationary).
  5. Save a 2-row figure: h_obs(t) on top, recovered m(t) on bottom
     (bottom panels show the single recovered m broadcast over time, as
     in proposal Fig. 4).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.dynamics import forward, inverse_stationary

# -------------------------------------------------------------------
# Physical parameters — chosen so that t = 1 yr is a non-trivial
# fraction of the relaxation time and h reaches O(1 m) amplitudes,
# matching the qualitative ranges in proposal Fig. 4.
# -------------------------------------------------------------------
H = 500.0  # reference ice thickness (m)
eta_bar = 3.0e13  # viscosity (Pa s) — moderate, gives tr ~ few months
alpha = 0.0
gamma = 0.0

# -------------------------------------------------------------------
# Domain — 80 km x 80 km as in proposal Fig. 4 (axes range -40 to +40 km)
# -------------------------------------------------------------------
L_km = 80.0
res_m = 500.0
x = np.arange(-L_km * 500, L_km * 500, res_m)  # centered at 0, ±40 km
y = np.arange(L_km * 500, -L_km * 500, -res_m)
ny, nx = y.size, x.size
X, Y = np.meshgrid(x, y)

# -------------------------------------------------------------------
# Prescribed stationary melt-rate anomaly — Gaussian peak, ~50 m/yr,
# a few kilometres wide.
# -------------------------------------------------------------------
m_peak = 50.0  # m/yr
sigma_m = 2500.0  # m (≈ 2.5 km)
m_true_arr = m_peak * np.exp(-0.5 * (X**2 + Y**2) / sigma_m**2)
m_true = xr.DataArray(m_true_arr, dims=("y", "x"), coords={"y": y, "x": x})

# -------------------------------------------------------------------
# Forward model at four time steps — produces synthetic h_obs
# -------------------------------------------------------------------
times_yr = np.array([0.25, 0.50, 0.75, 1.00])
SECONDS_PER_YEAR = 86400.0 * 365.25
times_s = times_yr * SECONDS_PER_YEAR

h_clean = forward(
    m_true,
    H=H,
    eta_bar=eta_bar,
    alpha=alpha,
    gamma=gamma,
    stationary=True,
    times=times_s,
)

rng = np.random.default_rng(42)
noise_std = 0.05 * float(np.abs(h_clean).max())  # 5% of peak signal
h_obs_arr = h_clean.values + rng.normal(0.0, noise_std, h_clean.shape)
h_obs = xr.DataArray(
    h_obs_arr,
    dims=h_clean.dims,
    coords=h_clean.coords,
    name="h_obs",
)
print(f"Forward: h peak = {float(np.abs(h_clean).max()):.2f} m, " f"noise std = {noise_std:.3f} m")

# -------------------------------------------------------------------
# Inversion — recover stationary m_recovered from the noisy h_obs
# stack. Input's first epoch is interpreted as t=0, so we prepend a
# zero layer at t=0 to match the forward convention.
# -------------------------------------------------------------------
h_obs_with_zero = xr.concat(
    [
        xr.zeros_like(h_obs.isel(time=0)).expand_dims(time=[0.0]),
        h_obs.assign_coords(time=times_s),
    ],
    dim="time",
)
m_recovered = inverse_stationary(
    h_obs_with_zero,
    H=H,
    eta_bar=eta_bar,
    alpha=alpha,
    gamma=gamma,
    reg=1e-2,
)

peak_true = float(m_true.max())
peak_rec = float(m_recovered.max())
print(f"Inversion: true peak m = {peak_true:.2f} m/yr, recovered peak = {peak_rec:.2f} m/yr")

# -------------------------------------------------------------------
# Plot: top row h_obs, bottom row m_recovered (broadcast over the 4
# epochs so the layout matches proposal Fig. 4; the recovered m is
# stationary by construction).
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
x_km = x / 1000.0
y_km = y / 1000.0
extent = [x_km.min(), x_km.max(), y_km.min(), y_km.max()]

h_vmin, h_vmax = float(h_obs.min()), float(h_obs.max())
m_vmin, m_vmax = 0.0, float(max(peak_true, peak_rec))

for i, t_yr in enumerate(times_yr):
    ax = axes[0, i]
    im = ax.imshow(
        h_obs.isel(time=i).values,
        extent=extent,
        origin="upper",
        cmap="Blues_r",
        vmin=h_vmin,
        vmax=h_vmax,
        aspect="equal",
    )
    ax.set_title(f"t = {t_yr:.2f} yr")
    if i == 0:
        ax.set_ylabel("y (km)")
    if i == 3:
        plt.colorbar(im, ax=ax, label="elevation anomaly (m)")

    ax = axes[1, i]
    im2 = ax.imshow(
        m_recovered.values,
        extent=extent,
        origin="upper",
        cmap="Reds",
        vmin=m_vmin,
        vmax=m_vmax,
        aspect="equal",
    )
    ax.set_xlabel("x (km)")
    if i == 0:
        ax.set_ylabel("y (km)")
    if i == 3:
        plt.colorbar(im2, ax=ax, label="melt rate (m/yr)")

fig.suptitle(
    "Proposal Fig. 4 reproduction: synthetic h_obs (top) and recovered stationary m (bottom)",
    fontsize=11,
)

out_path = Path(__file__).parent / "figures" / "proposal_fig4_recreation.png"
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Wrote {out_path}")
