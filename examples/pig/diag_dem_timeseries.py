#!/usr/bin/env python
"""Diagnostic: are we faithfully recreating the DEM elevation time series?

Computes per-pixel dh/dt (the exact dh_dt the Eulerian melt uses) on the
tilt-corrected is2ctempo stack, reports its magnitude/pattern over shelf vs
grounded, the per-pixel fit RMSE (reconstruction residual), and the actual
elevation time series h(t) at representative pixels (does it show coherent
thinning, or scatter?). This is the Dh/Dt side of melt = -rho*Dh/Dt - H*div(u) + a.
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pig.run_melt import load_stack, load_floating_mask
from stereo_melt.kinematics import dh_dt, SECONDS_PER_YEAR

_REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
OUT = f"{_REPO}/examples/pig/figures/diag_dem_timeseries.png"

print("loading stack...", flush=True)
stack = load_stack(stack_prefix="pig_stack_250m_is2ctempo")
floating = load_floating_mask(stack)
fm = floating.values.astype(bool)
t = pd.DatetimeIndex(stack["time"].values)
tyr = (t - t[0]).days / 365.25

print("regressing dh/dt (min_count=4)...", flush=True)
reg = dh_dt(stack, min_count=4)
slope = reg["slope"].values * SECONDS_PER_YEAR  # m/yr
count = reg["count"].values
rmse = reg["rmse"].values

def pct(a, m):
    v = a[m & np.isfinite(a)]
    return np.percentile(v, [5, 25, 50, 75, 95]) if v.size else np.full(5, np.nan), int(v.size if v.size else 0)

sh_p, sh_n = pct(slope, fm)
gr_p, gr_n = pct(slope, ~fm)
rm_p, _ = pct(rmse, fm)
print(f"\nshelf   dh/dt m/yr p[5,25,50,75,95]={np.round(sh_p,2)}  n={sh_n}")
print(f"grounded dh/dt m/yr p[5,25,50,75,95]={np.round(gr_p,2)}  n={gr_n}")
print(f"shelf   fit RMSE m  p[5,25,50,75,95]={np.round(rm_p,2)}")
print(f"shelf   obs/pixel   p[5,50,95]={np.round(np.percentile(count[fm & (count>0)],[5,50,95]),0)}")

# representative high-count floating pixels spread across x (GL -> outer shelf)
ys, xs = np.where(fm & (count >= 8) & np.isfinite(slope))
xc = stack["x"].values[xs]
reps = []
for q in (0.15, 0.4, 0.6, 0.85):  # x-quantiles across the shelf
    xt = np.quantile(xc, q)
    j = np.argmin(np.abs(xc - xt))
    reps.append((ys[j], xs[j]))

fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
# crop to shelf
finc = np.isfinite(slope) & fm
cols = np.where(finc.any(0))[0]; rows = np.where(finc.any(1))[0]
xkm = stack["x"].values / 1e3; ykm = stack["y"].values / 1e3
xl = (xkm[cols.min()] - 8, xkm[cols.max()] + 8); yl = (ykm[rows.min()] - 8, ykm[rows.max()] + 8)

ax0 = fig.add_subplot(gs[0])
sl_sh = np.where(fm, slope, np.nan)
pm = ax0.pcolormesh(xkm, ykm, sl_sh, cmap="RdBu", vmin=-4, vmax=4, shading="nearest")
ax0.set_title("dh/dt on shelf (m/yr)\n(surface lowering rate)", fontsize=10)
fig.colorbar(pm, ax=ax0, shrink=0.8, label="m/yr")
ax1 = fig.add_subplot(gs[1])
pm1 = ax1.pcolormesh(xkm, ykm, np.where(fm, rmse, np.nan), cmap="magma", vmin=0, vmax=3, shading="nearest")
ax1.set_title("per-pixel fit RMSE (m)\n(reconstruction residual)", fontsize=10)
fig.colorbar(pm1, ax=ax1, shrink=0.8, label="m")
for ax in (ax0, ax1):
    ax.set_aspect("equal"); ax.set_xlim(xl); ax.set_ylim(yl); ax.tick_params(labelsize=7)
    for (yi, xi) in reps:
        ax.plot(xkm[xi], ykm[yi], "kx", ms=7, mew=1.5)

ax2 = fig.add_subplot(gs[2])
for n, (yi, xi) in enumerate(reps):
    h = stack.values[:, yi, xi]
    ok = np.isfinite(h)
    s = slope[yi, xi]; r = rmse[yi, xi]; c = int(count[yi, xi])
    ax2.plot(tyr[ok], h[ok], "o", ms=4, label=f"px{n}: dh/dt={s:+.2f} m/yr, rmse={r:.2f}, n={c}")
    # fitted line
    b = reg["intercept"].values[yi, xi]
    ax2.plot(tyr, b + reg["slope"].values[yi, xi] * (tyr * SECONDS_PER_YEAR), "-", lw=1, alpha=0.6)
ax2.set_xlabel("years since " + str(t[0].date())); ax2.set_ylabel("surface elev (m)")
ax2.set_title("DEM elevation time series at marked pixels", fontsize=10)
ax2.legend(fontsize=7.5, loc="best")
fig.suptitle("PIG DEM time-series reconstruction (is2ctempo 250 m, tilt-corrected)", fontsize=12)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("\nwrote", OUT, flush=True)
