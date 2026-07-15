"""Side-by-side Eulerian / Lagrangian / stationary-spectral melt rate plot.

Loads the three result files written by run_melt, run_stationary, and (optional)
run_pseudospectral, and renders a comparison grid for the Beardmore AOI.

Run:

    python -m beardmore.plot_inversions
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from beardmore import config


CLIM = (-5.0, 5.0)
DIFF_CLIM = (-2.5, 2.5)


def _imshow_xr(ax, da: xr.DataArray, *, cmap, vmin=None, vmax=None):
    im = ax.imshow(
        da.values,
        extent=[
            float(da["x"].min()),
            float(da["x"].max()),
            float(da["y"].min()),
            float(da["y"].max()),
        ],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    return im


def _pick_tikhonov(stat: xr.Dataset) -> tuple[xr.DataArray, float]:
    """Pick the Tikhonov index whose IQR-width is closest to the Lagrangian
    target; falls back to the middle one."""
    tiks = np.asarray(stat.attrs["tikhonov_values"], dtype=float)
    vars_sorted = sorted(
        [v for v in stat.data_vars if v.startswith("melt_rate_tik_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    # Choose the one with the largest finite IQR before it blows up: that's
    # usually the elbow. We score each candidate's IQR width; the "good" ones
    # are non-degenerate (IQR>~0.5) and not yet huge (<~50).
    best_idx = None
    best_score = -np.inf
    for i, v in enumerate(vars_sorted):
        a = stat[v]
        finite = a.where(np.isfinite(a))
        if int(finite.notnull().sum()) == 0:
            continue
        q25, q75 = float(finite.quantile(0.25)), float(finite.quantile(0.75))
        iqr = q75 - q25
        if iqr < 0.5 or iqr > 60.0:
            continue
        score = -abs(iqr - 10.0)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx is None:
        best_idx = len(vars_sorted) // 2
    return stat[vars_sorted[best_idx]], float(tiks[best_idx])


def main() -> None:
    config.ensure_output_dirs()

    melt_path = (
        config.RESULTS_DIR
        / f"beardmore_melt_{config.START_TIME}_{config.END_TIME}.nc"
    )
    stat_path = (
        config.RESULTS_DIR
        / f"beardmore_stationary_{config.START_TIME}_{config.END_TIME}.nc"
    )
    ps_path = (
        config.RESULTS_DIR
        / f"beardmore_pseudospectral_{config.START_TIME}_{config.END_TIME}.nc"
    )

    if not melt_path.exists():
        raise SystemExit(f"missing run_melt output: {melt_path}")
    if not stat_path.exists():
        raise SystemExit(f"missing run_stationary output: {stat_path}")

    print(f"Loading {melt_path.name}...")
    melt = xr.open_dataset(melt_path)
    print(f"Loading {stat_path.name}...")
    stat = xr.open_dataset(stat_path)
    have_ps = ps_path.exists()
    if have_ps:
        print(f"Loading {ps_path.name}...")
        ps = xr.open_dataset(ps_path)

    eul = melt["melt_rate_eulerian"]
    lag = melt["melt_rate_lagrangian"]

    stat_da, tik = _pick_tikhonov(stat)
    print(f"  using stationary Tikhonov={tik:.0e}")

    def _stats(name, a):
        f = a.where(np.isfinite(a))
        n = int(f.notnull().sum())
        if n == 0:
            print(f"  {name}: no finite cells")
            return
        med = float(f.median())
        q25, q75 = float(f.quantile(0.25)), float(f.quantile(0.75))
        print(f"  {name}: median={med:+.2f}  IQR=[{q25:+.2f}, {q75:+.2f}]  N={n}")

    _stats("Eulerian", eul)
    _stats("Lagrangian", lag)
    _stats(f"Stationary (tik={tik:.0e})", stat_da)
    if have_ps:
        _stats("Pseudospectral Lagrangian", ps["melt_rate_lagrangian_ps"].mean("time"))

    nrows = 2 if have_ps else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])

    im0 = _imshow_xr(axes[0, 0], eul, cmap="RdBu_r", vmin=CLIM[0], vmax=CLIM[1])
    axes[0, 0].set_title("Eulerian (m ice/yr)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.045)

    im1 = _imshow_xr(axes[0, 1], lag, cmap="RdBu_r", vmin=CLIM[0], vmax=CLIM[1])
    axes[0, 1].set_title("Lagrangian (m ice/yr)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.045)

    im2 = _imshow_xr(axes[0, 2], stat_da, cmap="RdBu_r", vmin=CLIM[0], vmax=CLIM[1])
    axes[0, 2].set_title(f"Stationary spectral (Tik={tik:.0e})")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.045)

    if have_ps:
        ps_lag_mean = ps["melt_rate_lagrangian_ps"].mean("time", skipna=True)
        ps_eul_mean = ps["melt_rate_eulerian_ps"].mean("time", skipna=True)
        im3 = _imshow_xr(axes[1, 0], ps_eul_mean, cmap="RdBu_r", vmin=CLIM[0], vmax=CLIM[1])
        axes[1, 0].set_title("Pseudospectral Eulerian (time-mean)")
        fig.colorbar(im3, ax=axes[1, 0], fraction=0.045)

        im4 = _imshow_xr(axes[1, 1], ps_lag_mean, cmap="RdBu_r", vmin=CLIM[0], vmax=CLIM[1])
        axes[1, 1].set_title("Pseudospectral Lagrangian (time-mean)")
        fig.colorbar(im4, ax=axes[1, 1], fraction=0.045)

        linv = melt.get("melt_rate_linear_inverse")
        if linv is not None:
            im5 = _imshow_xr(axes[1, 2], linv, cmap="RdBu_r", vmin=CLIM[0], vmax=CLIM[1])
            axes[1, 2].set_title("Linear-inverse (Stubblefield, Lag-frame)")
            fig.colorbar(im5, ax=axes[1, 2], fraction=0.045)
        else:
            axes[1, 2].set_visible(False)

    for ax in axes.ravel():
        ax.set_xlabel("x (m)")
    for r in range(nrows):
        axes[r, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Beardmore melt rate inversions — {config.START_TIME} to {config.END_TIME}",
        fontsize=12,
    )

    out = config.FIGURES_DIR / "inversion_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
