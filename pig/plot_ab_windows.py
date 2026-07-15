#!/usr/bin/env python
"""Cross-window calving-split melt figure for one alignment tag.

Renders the three 2015-anchored windows (whole / before-B49 / after-B49) as a
3x3 grid:  Eulerian melt | Lagrangian melt | Eulerian per-pixel coverage.

All panels share the full stack extent and carry the whole-window floating-ice
outline, so the post-2020 coverage collapse (and the melt structure that
survives it) are both legible at a glance.

    usage: python -m pig.plot_ab_windows [tag]      # tag defaults to is2ctempo
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = sys.argv[1] if len(sys.argv) > 1 else "is2ctempo"
RES = "250m"
RESULTS = Path("/wd2/projects/stereo_melt/pig/results")
FIGDIR = Path("/wd2/projects/stereo_melt/pig/figures")
CLIM = 20.0  # m ice/yr; channels saturate but shelf-wide gradient stays visible

WINDOWS = [
    ("whole 2015-2023", "2015-01-01_2023-12-31"),
    ("before B-49 (->2020-02-08)", "2015-01-01_2020-02-08"),
    ("after B-49 (2020-02-09->)", "2020-02-09_2023-12-31"),
]


def load(win):
    return xr.open_dataset(RESULTS / f"pig_melt_{RES}_{TAG}_{win}.nc")


dss = [load(w) for _, w in WINDOWS]

x = dss[0]["x"].values
y = dss[0]["y"].values
dx = abs(float(x[1] - x[0]))
px_km2 = (dx * dx) / 1e6
extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
Xg, Yg = np.meshgrid(x, y)
whole_float = dss[0]["floating_mask"].values.astype(float)


def stats(ds):
    fm = ds["floating_mask"].values
    e = ds["melt_rate_eulerian"].values
    l = ds["melt_rate_lagrangian"].values
    e_ok = fm & np.isfinite(e)
    l_ok = fm & np.isfinite(l)
    return dict(
        e_med=float(np.median(e[e_ok])) if e_ok.any() else float("nan"),
        l_med=float(np.median(l[l_ok])) if l_ok.any() else float("nan"),
        e_area=e_ok.sum() * px_km2,
        l_cells=int(l_ok.sum()),
    )


fig, axes = plt.subplots(3, 3, figsize=(16, 15), constrained_layout=True)
ims = [None, None, None]

for r, (label, _) in enumerate(WINDOWS):
    ds = dss[r]
    st = stats(ds)
    fm = ds["floating_mask"].values
    e = np.where(fm, ds["melt_rate_eulerian"].values, np.nan)
    l = np.where(fm, ds["melt_rate_lagrangian"].values, np.nan)
    cnt = np.where(fm, ds["eulerian_count"].values, np.nan)

    ims[0] = axes[r, 0].imshow(e, extent=extent, origin="upper",
                               cmap="RdBu_r", vmin=-CLIM, vmax=CLIM, aspect="equal")
    ims[1] = axes[r, 1].imshow(l, extent=extent, origin="upper",
                               cmap="RdBu_r", vmin=-CLIM, vmax=CLIM, aspect="equal")
    ims[2] = axes[r, 2].imshow(cnt, extent=extent, origin="upper",
                               cmap="viridis", vmin=0, aspect="equal")

    for c in range(3):
        axes[r, c].contour(Xg, Yg, whole_float, levels=[0.5],
                           colors="k", linewidths=0.6)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])

    axes[r, 0].set_ylabel(label, fontsize=11)
    axes[r, 0].set_title(f"Eulerian   med={st['e_med']:+.1f}   area={st['e_area']:.0f} km$^2$",
                         fontsize=10)
    axes[r, 1].set_title(f"Lagrangian   med={st['l_med']:+.1f}   {st['l_cells']:,} cells",
                         fontsize=10)
    axes[r, 2].set_title("Eulerian coverage (epochs/pixel)", fontsize=10)

for c, lab in [(0, "melt rate (m ice/yr)"), (1, "melt rate (m ice/yr)"),
               (2, "epochs/pixel")]:
    cb = fig.colorbar(ims[c], ax=axes[:, c], location="bottom",
                      fraction=0.045, pad=0.02, shrink=0.9)
    cb.set_label(lab, fontsize=9)

fig.suptitle(
    f"PIG calving-split melt A/B  -  alignment = {TAG}\n"
    f"RdBu_r +/-{CLIM:.0f} m ice/yr (blue = melt, red = accretion); "
    f"black = whole-window floating-ice outline",
    fontsize=13,
)
out = FIGDIR / f"ab_windows_{RES}_{TAG}.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)

print(f"wrote {out}")
print(f"pixel size = {dx:.0f} m  ({px_km2*1e6:.0f} m^2/px)")
print(f"{'window':32s} {'Eul med':>8s} {'Eul area':>10s} {'Lag med':>8s} {'Lag cells':>10s}")
for (label, _), ds in zip(WINDOWS, dss):
    st = stats(ds)
    print(f"{label:32s} {st['e_med']:+8.2f} {st['e_area']:9.0f}k {st['l_med']:+8.2f} {st['l_cells']:>10,}")
