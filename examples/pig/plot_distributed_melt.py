#!/usr/bin/env python
"""Distributed spatial melt-rate map for PIG.

The native 250 m Eulerian field is flux-divergence-noise-dominated (per-pixel std
~250 m/yr, but MAD ~3-9 m/yr -> a heavy outlier tail from d/dx(H u)). A usable
distributed product needs robust spatial aggregation: here a ~1 km block-MEDIAN
(4x4), which rejects the tail while preserving the coherent melt pattern.

Shean convention: negative = melt, positive = freeze-on. Red = melt.
"""
import warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
BASE = f"{_REPO}/examples/pig/results/"
OUT = f"{_REPO}/examples/pig/figures/melt_distributed_1km_is2ctempo.png"
F = 4  # 250 m -> 1 km

def block_median(a, f=F):
    ny, nx = a.shape
    ny2, nx2 = (ny // f) * f, (nx // f) * f
    b = a[:ny2, :nx2].reshape(ny2 // f, f, nx2 // f, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        return np.nanmedian(b, axis=(1, 3))

def block_mean(c, f=F):
    n = (len(c) // f) * f
    return c[:n].reshape(-1, f).mean(1)

def block_frac(a, f=F):  # fraction of valid (for mask outline)
    ny, nx = a.shape
    ny2, nx2 = (ny // f) * f, (nx // f) * f
    return a[:ny2, :nx2].reshape(ny2 // f, f, nx2 // f, f).mean(axis=(1, 3))

def load(tag, fn, var="melt_rate_eulerian"):
    ds = xr.open_dataset(BASE + fn)
    da = ds[var].transpose("y", "x")
    x = block_mean(ds["x"].values) / 1e3  # km
    y = block_mean(ds["y"].values) / 1e3
    m = block_median(da.values.astype("float64"))
    fmask = ds["floating_mask"].transpose("y", "x").values.astype("float64")
    fmask = block_frac(np.nan_to_num(fmask, nan=0.0))
    return x, y, m, fmask

panels = [
    ("Full record  2010–2024", "pig_melt_250m_is2ctempo_2010-01-01_2024-01-10.nc", "melt_rate_eulerian"),
    ("Full record — Lagrangian", "pig_melt_250m_is2ctempo_2010-01-01_2024-01-10.nc", "melt_rate_lagrangian"),
    ("W3  2017–2020 (B44→B49)", "pig_melt_250m_is2ctempo_2017-09-01_2020-02-08.nc", "melt_rate_eulerian"),
    ("W4  2020–2024 (post-B49)", "pig_melt_250m_is2ctempo_2020-02-09_2024-01-10.nc", "melt_rate_eulerian"),
]

# shared, robust symmetric color limit from the full-record aggregated Eulerian
x0, y0, m0, _ = load(*panels[0])
vlim = float(np.nanpercentile(np.abs(m0), 96))
vlim = min(max(vlim, 10.0), 30.0)
norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
print(f"color limit = +/-{vlim:.1f} m/yr (red = melt)")

fig, axes = plt.subplots(2, 2, figsize=(12.5, 12.0), constrained_layout=True)
for ax, (title, fn, var) in zip(axes.ravel(), panels):
    x, y, m, fmask = load(title, fn, var)
    pm = ax.pcolormesh(x, y, m, cmap="RdBu", norm=norm, shading="nearest")
    ax.contour(x, y, fmask, levels=[0.5], colors="k", linewidths=0.6, alpha=0.6)
    ax.set_aspect("equal")
    nfin = int(np.isfinite(m).sum())
    med = np.nanmedian(m)
    ax.set_title(f"{title}\n1 km-median  •  median={med:+.2f} m/yr  •  {nfin} cells", fontsize=10.5)
    ax.set_xlabel("x (km, PS-71)", fontsize=9)
    ax.set_ylabel("y (km)", fontsize=9)
    ax.tick_params(labelsize=8)

cb = fig.colorbar(pm, ax=axes, shrink=0.6, extend="both", location="right", pad=0.02)
cb.set_label("basal melt rate  (m ice / yr)\n← melt          freeze →", fontsize=11)
fig.suptitle("Pine Island distributed basal melt — 250 m solve, 1 km block-median  (IS2+CryoTEMPO control)",
             fontsize=13)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
