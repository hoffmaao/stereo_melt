"""Plot the coverage-gap diagnostic — visualizes ringing collapse pattern."""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.dynamics.linear_perturbation import forward, inverse_stationary
from stereo_melt.dynamics.lagrangian_inverse import _nan_aware_gaussian_infill_2d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_linear_inverse_coverage import (
    DX, DY, ETA, H_REF, NX, NY, N_EPOCHS, make_strip_masks, r, times, x, y,
)


def invert(h_da, transform="fft"):
    h_anom = h_da - h_da.mean("time", skipna=True)
    h_anom = h_anom.fillna(0.0)
    return inverse_stationary(
        h_anom, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        reg=1e-3, reference="time_mean", transform=transform,
    ).values


m_truth = 5.0 * np.exp(-0.5 * (r / 1000.0) ** 2)
m_da = xr.DataArray(m_truth, dims=("y", "x"), coords={"y": y, "x": x}, name="m")
h = forward(
    m_da, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
    times=xr.DataArray(times, dims="time"), stationary=True,
)

obs_mask = make_strip_masks(N_EPOCHS, NY, NX, seed=7)
h_sparse_vals = h.values.copy()
h_sparse_vals[~obs_mask] = np.nan
h_sparse = xr.DataArray(h_sparse_vals, dims=h.dims, coords=h.coords)

m_dense = invert(h, transform="fft")
m_anom_fill = invert(h_sparse, transform="fft")

# Infill regime
sigma_pix = (H_REF / DY, H_REF / DX)
infilled = np.empty_like(h_sparse_vals)
for ti in range(N_EPOCHS):
    infilled[ti] = _nan_aware_gaussian_infill_2d(
        h_sparse_vals[ti], sigma_pix, max_iters=8, support_threshold=0.05,
    )
h_infill = xr.DataArray(infilled, dims=h.dims, coords=h.coords)
m_infill = invert(h_infill, transform="fft")

cov = obs_mask.mean(axis=0)

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)

vmax_signal = 5.5
im0 = axes[0].imshow(m_truth, cmap="RdBu_r", vmin=-vmax_signal, vmax=vmax_signal,
                     extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
axes[0].set_title(f"truth m\nGaussian +5 m/yr blob")
plt.colorbar(im0, ax=axes[0], fraction=0.045)

im1 = axes[1].imshow(cov, cmap="viridis", vmin=0, vmax=1,
                     extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
axes[1].set_title(f"per-pixel coverage\n(mean {cov.mean():.2f}, min {cov.min():.2f})")
plt.colorbar(im1, ax=axes[1], fraction=0.045)

im2 = axes[2].imshow(m_dense, cmap="RdBu_r", vmin=-vmax_signal, vmax=vmax_signal,
                     extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
axes[2].set_title(f"dense recovery\ncenter={m_dense[NY//2,NX//2]:+.2f}  "
                  f"abs_max={np.abs(m_dense).max():.1f}")
plt.colorbar(im2, ax=axes[2], fraction=0.045)

# For sparse cases, use a saturated colorbar so the ringing structure is visible
vmax_sparse = 8.0
im3 = axes[3].imshow(m_anom_fill, cmap="RdBu_r", vmin=-vmax_sparse, vmax=vmax_sparse,
                     extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
axes[3].set_title(f"sparse + anom→fill0\ncenter={m_anom_fill[NY//2,NX//2]:+.2f}  "
                  f"abs_max={np.abs(m_anom_fill).max():.1f}")
plt.colorbar(im3, ax=axes[3], fraction=0.045)

vmax_infill = 30.0
im4 = axes[4].imshow(m_infill, cmap="RdBu_r", vmin=-vmax_infill, vmax=vmax_infill,
                     extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
axes[4].set_title(f"sparse + Gaussian infill\ncenter={m_infill[NY//2,NX//2]:+.2f}  "
                  f"abs_max={np.abs(m_infill).max():.1f}")
plt.colorbar(im4, ax=axes[4], fraction=0.045)

for a in axes:
    a.set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")

fig.suptitle(
    "Stubblefield closed-form inverse: recovery vs per-pixel coverage gaps\n"
    "(synthetic Gaussian +5 m/yr melt blob, 5 epochs, ~70% per-pixel coverage)",
    fontsize=12,
)
out = "/wd2/projects/stereo_melt/tests/diagnose_linear_inverse_coverage.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved: {out}")
