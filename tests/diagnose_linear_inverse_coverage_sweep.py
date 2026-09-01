"""How does closed-form recovery scale with per-pixel coverage?

Sweep per-pixel coverage from 0.30 to 1.00 in 0.05 steps. For each coverage
level, generate a sparse observation mask, run forward + inverse on the same
Gaussian +5 m/yr blob, and report:
  - center recovery (truth = +5.0)
  - peak abs ringing
  - RMSE vs truth
  - spatial mean recovered (truth = +0.048)

Two masking modes for contrast:
  - "uniform"   : each pixel observed at random fraction p of epochs (IID)
  - "strip"     : REMA-like elongated strips (correlated coverage)

If LSQ over temporal epochs is genuinely helping, increasing n_epochs should
recover the signal even at low p. Plot center recovery vs p for n_epochs in
{5, 15, 30, 60} to test that.
"""
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

NX = NY = 256
DX = DY = 100.0
H_REF = 500.0
ETA = 1e14

x = (np.arange(NX) - NX // 2) * DX
y = (np.arange(NY) - NY // 2) * DY
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx ** 2 + yy ** 2)
m_truth = 5.0 * np.exp(-0.5 * (r / 1000.0) ** 2)


def make_uniform_mask(n_epochs, ny, nx, p, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n_epochs, ny, nx)) < p


def make_strip_mask(n_epochs, ny, nx, p_target, seed):
    """REMA-like strip masks tuned to hit per-pixel mean coverage ≈ p_target."""
    rng = np.random.default_rng(seed)
    # Per-epoch coverage fraction needed to hit per-pixel mean p_target
    # roughly equals p_target (since per-pixel = average of per-epoch).
    yy_pix, xx_pix = np.mgrid[0:ny, 0:nx]
    mask = np.zeros((n_epochs, ny, nx), dtype=bool)
    for i in range(n_epochs):
        em = np.zeros((ny, nx), dtype=bool)
        # Tune number of strips and width to hit p_target
        target_area = p_target * ny * nx
        achieved = 0
        attempts = 0
        while achieved < 0.85 * target_area and attempts < 12:
            angle = rng.uniform(0, np.pi)
            cx = rng.uniform(0, nx)
            cy = rng.uniform(0, ny)
            half_w = rng.uniform(40, 80)
            nrm = (xx_pix - cx) * np.sin(angle) - (yy_pix - cy) * np.cos(angle)
            em |= np.abs(nrm) <= half_w
            achieved = em.sum()
            attempts += 1
        mask[i] = em
    return mask


def invert(h_da, transform="fft"):
    h_anom = h_da - h_da.mean("time", skipna=True)
    h_anom = h_anom.fillna(0.0)
    return inverse_stationary(
        h_anom, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        reg=1e-3, reference="time_mean", transform=transform,
    ).values


def run_sweep(n_epochs, mask_kind="strip"):
    times = np.arange(n_epochs).astype("datetime64[Y]").astype("datetime64[ns]")
    m_da = xr.DataArray(m_truth, dims=("y", "x"), coords={"y": y, "x": x})
    h = forward(
        m_da, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        times=xr.DataArray(times, dims="time"), stationary=True,
    )

    coverage_targets = np.arange(0.30, 1.001, 0.05)
    rows = []
    for p in coverage_targets:
        if mask_kind == "uniform":
            m_mask = make_uniform_mask(n_epochs, NY, NX, p, seed=7)
        else:
            m_mask = make_strip_mask(n_epochs, NY, NX, p, seed=7)
        h_v = h.values.copy()
        h_v[~m_mask] = np.nan
        h_sparse = xr.DataArray(h_v, dims=h.dims, coords=h.coords)
        cov_actual = m_mask.mean()
        m_inv = invert(h_sparse, transform="fft")
        center = m_inv[NY // 2, NX // 2]
        spatial_mean = m_inv.mean()
        abs_max = np.abs(m_inv).max()
        rmse = float(np.sqrt(np.mean((m_inv - m_truth) ** 2)))
        rows.append((cov_actual, center, spatial_mean, abs_max, rmse))
    return np.array(rows)


print(f"{'n_epochs':>8s} {'kind':>8s} {'cov':>5s} {'center':>7s} "
      f"{'<m>':>7s} {'absmax':>8s} {'RMSE':>7s}  (truth: center=+5.0, <m>=+0.048)")
results = {}
for n_t in [5, 15, 30, 60]:
    for kind in ["strip", "uniform"]:
        arr = run_sweep(n_t, mask_kind=kind)
        results[(n_t, kind)] = arr
        for cov, c, mm, amax, rmse in arr:
            print(f"{n_t:>8d} {kind:>8s} {cov:>5.2f} {c:>+7.2f} "
                  f"{mm:>+7.3f} {amax:>8.1f} {rmse:>7.3f}")
        print()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
colors = {5: "C0", 15: "C1", 30: "C2", 60: "C3"}

for kind, ls in [("strip", "-"), ("uniform", "--")]:
    for n_t in [5, 15, 30, 60]:
        arr = results[(n_t, kind)]
        cov, center, mean_m, amax, rmse = arr.T
        axes[0].plot(cov, center, color=colors[n_t], linestyle=ls,
                     marker="o", label=f"n={n_t} {kind}")
        axes[1].plot(cov, amax, color=colors[n_t], linestyle=ls,
                     marker="o", label=f"n={n_t} {kind}")
        axes[2].plot(cov, rmse, color=colors[n_t], linestyle=ls,
                     marker="o", label=f"n={n_t} {kind}")

axes[0].axhline(5.0, color="k", linestyle=":", alpha=0.5, label="truth +5.0")
axes[0].set_xlabel("per-pixel coverage fraction")
axes[0].set_ylabel("recovered center value (m/yr)")
axes[0].set_title("Center recovery (truth = +5.0)")
axes[0].set_ylim(0, 6)
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8, ncol=2)

axes[1].axhline(5.0, color="k", linestyle=":", alpha=0.5, label="signal magnitude")
axes[1].set_xlabel("per-pixel coverage fraction")
axes[1].set_ylabel("|m| max (m/yr)")
axes[1].set_title("Peak ringing")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3, which="both")

axes[2].set_xlabel("per-pixel coverage fraction")
axes[2].set_ylabel("RMSE (m/yr)")
axes[2].set_title("RMSE vs truth")
axes[2].set_yscale("log")
axes[2].grid(alpha=0.3, which="both")

# Mark the actual coverage of each basin
for label, p in [("Nansen 23%", 0.23), ("Beardmore IS2 47%", 0.47), ("Beardmore CS2 42%", 0.42)]:
    for ax in axes:
        ax.axvline(p, color="red", alpha=0.3, linestyle=":")
    axes[0].text(p, 5.5, label.split()[0][:3], color="red", fontsize=8,
                 ha="center", rotation=90)

fig.suptitle(
    "Closed-form Stubblefield inverse: how recovery scales with coverage and n_epochs\n"
    "(stationary +5 m/yr Gaussian blob; solid=strip masks, dashed=uniform random)",
    fontsize=11,
)
out = "/wd2/projects/stereo_melt/tests/diagnose_coverage_sweep.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved: {out}")
