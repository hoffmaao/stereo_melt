"""Plot melt rates from run_melt outputs.

Operates on the saved ``pig_melt_<res>m_<tag>_<start>_<end>.nc`` files (post-hoc,
decoupled from run_melt so it can run while the solver pipeline is in flight).

Two modes:
  single run :  python -m pig.plot_melt --tag is2ctempo --start S --end E
                -> Eulerian map | Lagrangian map | distribution
  A/B compare:  python -m pig.plot_melt --compare is2ctempo densetie --start S --end E
                -> 2x3 grid: {Eul,Lag} x {base, new, new-base difference}

Maps auto-crop to the floating-shelf data extent. Shean convention:
melt_rate negative = melt, positive = accretion.
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm

RESULTS = Path("/wd2/projects/stereo_melt/pig/results")
FIG = Path("/wd2/projects/stereo_melt/pig/figures")
EUL, LAG = "melt_rate_eulerian", "melt_rate_lagrangian"


def _path(tag, res, start, end):
    p = RESULTS / f"pig_melt_{res}m_{tag}_{start}_{end}.nc"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _imshow(ax, da, **kw):
    x, y = da.x.values, da.y.values
    extent = [x.min(), x.max(), y.min(), y.max()]
    origin = "upper" if y[0] > y[-1] else "lower"
    return ax.imshow(da.values, extent=extent, origin=origin, aspect="equal", **kw)


def _stats(da):
    v = np.asarray(da.values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), v.size


def _lbl(name, da):
    m, q1, q3, n = _stats(da)
    return f"{name}\nmed={m:.2f}  IQR=[{q1:.1f}, {q3:.1f}]  n={n}"


def _crop(axes, ref, margin=4000.0):
    """Zoom all map axes to the finite-data bounding box of ``ref`` (+margin)."""
    m = np.isfinite(np.asarray(ref.values, float))
    if not m.any():
        return
    ys, xs = np.where(m)
    x, y = ref.x.values, ref.y.values
    xlo, xhi = sorted([float(x[xs.min()]), float(x[xs.max()])])
    ylo, yhi = sorted([float(y[ys.min()]), float(y[ys.max()])])
    for a in np.atleast_1d(axes).ravel():
        a.set_xlim(xlo - margin, xhi + margin)
        a.set_ylim(ylo - margin, yhi + margin)


def plot_single(tag, res, start, end, clim):
    ds = xr.open_dataset(_path(tag, res, start, end))
    fig, ax = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)

    for a, var, name in [(ax[0], EUL, "Eulerian"), (ax[1], LAG, "Lagrangian")]:
        im = _imshow(a, ds[var], cmap=melt_cmap(), norm=melt_norm(vmax=clim))
        a.set_title(_lbl(name, ds[var]))
        add_melt_colorbar(fig, im, ax=a, fraction=0.046)
        a.set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    _crop(ax[:2], ds[EUL])

    for var, name, c in [(EUL, "Eulerian", "C3"), (LAG, "Lagrangian", "C0")]:
        v = np.asarray(ds[var].values, float)
        v = v[np.isfinite(v)]
        ax[2].hist(np.clip(v, -clim, clim), bins=80, histtype="step",
                   color=c, label=f"{name} (med {np.median(v):.2f})")
    ax[2].axvline(0, color="k", lw=0.6)
    ax[2].set_xlabel("melt (m ice/yr)"); ax[2].set_ylabel("pixels")
    ax[2].set_title("distribution"); ax[2].legend(fontsize=9)

    fig.suptitle(f"PIG melt rate — {tag} — {start} to {end}", fontsize=13)
    out = FIG / f"meltrate_{res}m_{tag}_{start}_{end}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")
    return out


def plot_ab(base, new, res, start, end, clim, dclim):
    dsA, dsB = xr.open_dataset(_path(base, res, start, end)), xr.open_dataset(_path(new, res, start, end))
    fig, ax = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

    for r, (var, name) in enumerate([(EUL, "Eulerian"), (LAG, "Lagrangian")]):
        for c, (ds, tag) in enumerate([(dsA, base), (dsB, new)]):
            im = _imshow(ax[r, c], ds[var], cmap=melt_cmap(),
                         norm=melt_norm(vmax=clim))
            ax[r, c].set_title(_lbl(f"{name} — {tag}", ds[var]))
            add_melt_colorbar(fig, im, ax=ax[r, c], fraction=0.046)
        diff = dsB[var] - dsA[var]
        im = _imshow(ax[r, 2], diff, cmap="PuOr", vmin=-dclim, vmax=dclim)
        ax[r, 2].set_title(f"{name}  Δ ({new} − {base})")
        fig.colorbar(im, ax=ax[r, 2], fraction=0.046)
    for a in ax[1, :]:
        a.set_xlabel("x (m)")
    ax[0, 0].set_ylabel("y (m)"); ax[1, 0].set_ylabel("y (m)")
    _crop(ax, dsA[EUL])

    fig.suptitle(f"PIG melt A/B  {new} vs {base}  —  {start} to {end}", fontsize=13)
    out = FIG / f"meltrate_AB_{res}m_{new}_vs_{base}_{start}_{end}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=250)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"))
    ap.add_argument("--clim", type=float, default=20.0)
    ap.add_argument("--dclim", type=float, default=10.0, help="diff-map color limit")
    a = ap.parse_args()
    if a.compare:
        plot_ab(a.compare[0], a.compare[1], a.res, a.start, a.end, a.clim, a.dclim)
    elif a.tag:
        plot_single(a.tag, a.res, a.start, a.end, a.clim)
    else:
        ap.error("need --tag (single) or --compare BASE NEW (A/B)")
