"""Replot the 5-method comparison from the saved NetCDF.

Reads ``beardmore/results/beardmore_five_methods_<window>.nc`` (produced by the retired
``beardmore.compare_five_methods``) and builds a 2x3 figure with all six
melt-rate fields on a single ±5 m/yr RdBu scale, so panels are directly
comparable to the rest of the Beardmore figures.

Run::

    python -m beardmore.replot_five_methods
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

from beardmore import config


def _imshow(ax, da: xr.DataArray, *, cmap: str, vmin: float, vmax: float):
    x = da["x"].values
    y = da["y"].values
    extent = (x.min(), x.max(), y.min(), y.max())
    return ax.imshow(
        da.values,
        origin="upper" if y[0] > y[-1] else "lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )


def _stats(da: xr.DataArray) -> str:
    v = da.values
    finite = np.isfinite(v)
    if not finite.any():
        return "no data"
    v = v[finite]
    return (
        f"med={np.median(v):+.2f}  "
        f"IQR=[{np.percentile(v, 25):+.2f}, {np.percentile(v, 75):+.2f}]  "
        f"|max|={np.max(np.abs(v)):.0f}"
    )


def main() -> None:
    nc_path = (
        config.RESULTS_DIR
        / f"beardmore_five_methods_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Loading -> {nc_path}")
    ds = xr.open_dataset(nc_path)

    floating = ds.floating_mask.astype(bool)
    panels = [
        ("1. Eulerian\n(Shean Eq. 10)", ds.melt_rate_eulerian.where(floating)),
        ("2. Lagrangian path-int\n(Shean Eq. 7)", ds.melt_rate_lagrangian.where(floating)),
        ("3. closed-form FFT\n(infill+pad, reg=0.1)", ds.melt_rate_closed_fft.where(floating)),
        ("4. closed-form DCT\n(reflective, reg=0.1)", ds.melt_rate_closed_dct.where(floating)),
        ("5a. dh/dt FFT\n(per-pixel OLS, reg=10)", ds.melt_rate_dhdt_fft.where(floating)),
        ("5b. dh/dt DCT\n(per-pixel OLS, reg=10)", ds.melt_rate_dhdt_dct.where(floating)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(3 * 4.6, 2 * 5.6), constrained_layout=True)
    for ax, (title, da) in zip(axes.flat, panels):
        im = _imshow(ax, da, cmap="RdBu_r", vmin=-5, vmax=5)
        ax.set_title(f"{title}\n{_stats(da)}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, label="m ice/yr")
        ax.set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Beardmore melt-rate comparison — 5 methods  "
        f"({config.START_TIME} → {config.END_TIME}, "
        f"vel={ds.attrs.get('velocity_source', '?')})",
        fontsize=12,
    )

    fig_path = config.FIGURES_DIR / "five_methods_comparison_v2.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
