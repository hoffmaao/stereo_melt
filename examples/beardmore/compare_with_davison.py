"""Compare our 5-method Beardmore melt-rate inversions against Davison 2023.

Beardmore is a Ross-system tributary glacier and is **not** in Davison's
55-shelf per-shelf CSV catalog (which uses Greene 2022 named shelves).
So we sample Davison's gridded basal-melt TIFF over our floating mask
only — no per-shelf time-series scalar.

Layout: 2x4 figure mirroring :mod:`nansen.compare_with_davison`.

Run::

    python -m beardmore.compare_with_davison
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

from stereo_melt.io.davison import load_davison_gridded_in_shean

from beardmore import config

CMAP_RANGE = 5.0  # m/yr; tighter than Nansen


def _imshow(ax, da: xr.DataArray, *, vmin, vmax, cmap="RdBu_r"):
    x = da["x"].values
    y = da["y"].values
    return ax.imshow(
        da.values,
        origin="upper" if y[0] > y[-1] else "lower",
        extent=(x.min(), x.max(), y.min(), y.max()),
        cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="nearest",
    )


def _stats_str(da: xr.DataArray, mask: np.ndarray) -> str:
    v = da.values[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "no data"
    return (
        f"med={np.median(v):+.2f}  mean={np.mean(v):+.2f}  "
        f"IQR=[{np.percentile(v, 25):+.2f}, {np.percentile(v, 75):+.2f}]"
    )


def main() -> None:
    nc_path = (
        config.RESULTS_DIR
        / f"beardmore_five_methods_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Loading -> {nc_path}")
    ds = xr.open_dataset(nc_path)
    floating = ds.floating_mask.astype(bool)
    floating_arr = floating.values

    print("Reprojecting Davison 2023 gridded TIFF onto Beardmore 25 m grid...")
    # `load_davison_gridded_in_shean` flips Davison's native positive=melt
    # into the Shean public convention so panels and stats are directly
    # comparable to ours.
    davison = load_davison_gridded_in_shean(ds.melt_rate_eulerian)
    davison_f = davison.where(floating)
    finite_frac = float(np.isfinite(davison_f.values[floating_arr]).mean())
    print(f"  Davison finite-fraction over our floating mask: {finite_frac:.3f}")

    eul = ds.melt_rate_eulerian.where(floating)
    lagr = ds.melt_rate_lagrangian.where(floating)
    cdct = ds.melt_rate_closed_dct.where(floating)
    ddct = ds.melt_rate_dhdt_dct.where(floating)

    print("\nBasin-mean / median comparison (m ice/yr; Shean convention: negative = melt):")
    print(f"  Eulerian      : {_stats_str(eul,  floating_arr)}")
    print(f"  Lagrangian    : {_stats_str(lagr, floating_arr)}")
    print(f"  closed-DCT    : {_stats_str(cdct, floating_arr)}")
    print(f"  dh/dt-DCT     : {_stats_str(ddct, floating_arr)}")
    print(f"  Davison grid  : {_stats_str(davison_f, floating_arr)}")

    fig, axes = plt.subplots(2, 4, figsize=(4 * 4.6, 2 * 5.6), constrained_layout=True)
    panels_top = [
        ("Eulerian (ours)", eul),
        ("Lagrangian path-int (ours)", lagr),
        ("closed-DCT (ours)", cdct),
        ("dh/dt-DCT (ours)", ddct),
    ]
    for ax, (title, da) in zip(axes[0], panels_top):
        im = _imshow(ax, da, vmin=-CMAP_RANGE, vmax=CMAP_RANGE)
        ax.set_title(f"{title}\n{_stats_str(da, floating_arr)}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, label="m ice/yr")
        ax.set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")

    diff_lag = (lagr - davison_f)
    diff_cdct = (cdct - davison_f)
    diff_ddct = (ddct - davison_f)
    panels_bot = [
        ("Davison 2023 gridded\n(1 km, RACMO-FAC corr.)", davison_f, -CMAP_RANGE, CMAP_RANGE),
        ("Lag − Davison",  diff_lag,  -CMAP_RANGE, CMAP_RANGE),
        ("closed-DCT − Davison",  diff_cdct, -CMAP_RANGE, CMAP_RANGE),
        ("dh/dt-DCT − Davison",   diff_ddct, -CMAP_RANGE, CMAP_RANGE),
    ]
    for ax, (title, da, vmin, vmax) in zip(axes[1], panels_bot):
        im = _imshow(ax, da, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n{_stats_str(da, floating_arr)}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, label="m ice/yr")
        ax.set_xlabel("x (m)")
    axes[1, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Beardmore melt-rate vs Davison 2023  "
        f"({config.START_TIME} → {config.END_TIME}; "
        f"no per-shelf CSV — Beardmore is a Ross-system tributary, "
        f"sampled from gridded TIFF only)",
        fontsize=12,
    )
    fig_path = config.FIGURES_DIR / "compare_with_davison.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
