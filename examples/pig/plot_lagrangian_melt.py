#!/usr/bin/env python
"""Distributed PIG Lagrangian basal-melt map at NATIVE 250 m (no block-aggregation).

The Lagrangian solver uses Shean's frame: melt = Dh/Dt + H*div(u) -- the clean
form (no u*grad(H)). This plots melt_rate_lagrangian straight at 250 m, the
resolution it was solved at. Shean convention: negative = melt -> red.

Panels: full-record (2010-2024, raw vel) = most complete distributed estimate;
W4 (2020-2024) raw vs FUSED velocity = the Kalman+EOF+1km A/B on the one window
the overnight sweep has finished.
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

R = "/wd2/projects/stereo_melt/examples/pig/results/"
OUT = "/wd2/projects/stereo_melt/examples/pig/figures/melt_lagrangian_250m_native.png"
V = "melt_rate_lagrangian"

panels = [
    ("Full record 2010–2024  (raw vel)", R + "pig_melt_250m_is2ctempo_2010-01-01_2024-01-10.nc"),
    ("W4 2020–2024  (raw vel)",          R + "pig_melt_250m_is2ctempo_2020-02-09_2024-01-10.nc"),
    ("W4 2020–2024  (FUSED vel)",        R + "pig_melt_250m_is2ctempo_fusedvel_2020-02-09_2024-01-10.nc"),
]

def load(fn):
    ds = xr.open_dataset(fn)
    m = ds[V].transpose("y", "x").values.astype("float64")
    x = ds["x"].values / 1e3
    y = ds["y"].values / 1e3
    fm = ds["floating_mask"].transpose("y", "x").values.astype("float64")
    return x, y, m, fm

# shared robust color limit + crop box from the full-record field.
# Shean-style: melt shown as POSITIVE magnitude on plasma (bright = high melt).
x0c, y0c, m0, _ = load(panels[0][1])
vhi = float(np.nanpercentile(np.abs(m0), 96))
vhi = min(max(vhi, 15.0), 50.0)
norm = Normalize(vmin=0.0, vmax=vhi)
fin0 = np.isfinite(m0)
cols = np.where(fin0.any(axis=0))[0]
rows = np.where(fin0.any(axis=1))[0]
mgn = 8.0  # km margin
xlim = (x0c[cols.min()] - mgn, x0c[cols.max()] + mgn)
ylim = (y0c[rows.min()] - mgn, y0c[rows.max()] + mgn)
print(f"plasma melt scale 0..{vhi:.1f} m/yr (bright = melt); crop x{tuple(round(v) for v in sorted(xlim))} "
      f"y{tuple(round(v) for v in sorted(ylim))} km")

fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4), constrained_layout=True)
for ax, (title, fn) in zip(axes, panels):
    x, y, m, fm = load(fn)
    melt = -m  # melt as positive magnitude (Shean convention; bright = high melt)
    pm = ax.pcolormesh(x, y, melt, cmap="plasma", norm=norm, shading="nearest")
    ax.contour(x, y, np.nan_to_num(fm), levels=[0.5], colors="w", linewidths=0.5, alpha=0.7)
    nfin = int(np.isfinite(m).sum())
    med = float(np.nanmedian(melt))
    p25, p75 = (float(v) for v in np.nanpercentile(melt[np.isfinite(melt)], [25, 75]))
    ax.set_aspect("equal")
    ax.set_xlim(sorted(xlim))
    ax.set_ylim(sorted(ylim))
    ax.set_title(f"{title}\nnative 250 m  •  melt median={med:+.2f}  IQR=[{p25:+.1f},{p75:+.1f}]  •  {nfin} cells",
                 fontsize=10)
    ax.set_xlabel("x (km, PS-71)", fontsize=9)
    ax.tick_params(labelsize=8)
    print(f"  {title:34s} melt median={med:+.2f}  IQR=[{p25:+.1f},{p75:+.1f}]  n={nfin}")
axes[0].set_ylabel("y (km)", fontsize=9)
cb = fig.colorbar(pm, ax=axes, shrink=0.8, extend="both", location="right", pad=0.015)
cb.set_label("Lagrangian basal melt rate  (m ice / yr)\nbright = high melt  (plasma, Shean-style)", fontsize=11)
fig.suptitle("Pine Island Lagrangian basal melt — native 250 m, Shean H·∇·u form  (IS2+CryoTEMPO+ATM/LVIS control)",
             fontsize=12.5)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
