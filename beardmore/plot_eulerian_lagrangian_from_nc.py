"""Plot 2-panel Eulerian vs Lagrangian melt rate (Nansen-style) from a saved NetCDF.

Reads any NetCDF that contains ``melt_rate_eulerian``, ``melt_rate_lagrangian``,
and ``floating_mask`` (currently: ``beardmore_five_methods_<R>m_*.nc`` from
``compare_five_methods``, or ``beardmore_eulerian_lagrangian_<R>m_*.nc`` from
``plot_eulerian_lagrangian``) and emits the standardized 2-panel figure.

Shean convention throughout: negative = melt, positive = accretion.

Run::

    python -m beardmore.plot_eulerian_lagrangian_from_nc <path/to/results.nc>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from beardmore import config


def _med_iqr(da: xr.DataArray) -> str:
    med = float(da.median())
    q25, q75 = float(da.quantile(0.25)), float(da.quantile(0.75))
    n = int(da.notnull().sum())
    return f"median={med:+.2f}  IQR=[{q25:+.2f}, {q75:+.2f}]  N={n}"


def main(nc_path: Path, fig_path: Path | None = None) -> Path:
    ds = xr.open_dataset(nc_path)

    floating = ds["floating_mask"].astype(bool)
    eul = ds["melt_rate_eulerian"].where(floating)
    lag = ds["melt_rate_lagrangian"].where(floating)

    grid_res_m = float(abs(float(ds["x"].values[1]) - float(ds["x"].values[0])))
    res_label = f"{int(round(grid_res_m))} m"

    if fig_path is None:
        # Mirror nansen/figures/eulerian_lagrangian_<R>m.png layout.
        m = re.search(r"_(\d+)m_", nc_path.name)
        suffix = f"_{m.group(1)}m" if m else ""
        fig_path = config.FIGURES_DIR / f"eulerian_lagrangian{suffix}.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    clim = (-16.0, 16.0)
    extent = [
        float(ds["x"].min()), float(ds["x"].max()),
        float(ds["y"].min()), float(ds["y"].max()),
    ]

    im0 = axes[0].imshow(
        eul.values, extent=extent, origin="upper",
        cmap="RdBu_r", vmin=clim[0], vmax=clim[1], aspect="equal",
    )
    axes[0].set_title(
        f"Eulerian  (Shean Eq. 10)\n{_med_iqr(eul)} m ice/yr", fontsize=11
    )
    fig.colorbar(
        im0, ax=axes[0], fraction=0.045,
        label="melt rate (m ice/yr)\nneg = melt, pos = accretion",
    )

    im1 = axes[1].imshow(
        lag.values, extent=extent, origin="upper",
        cmap="RdBu_r", vmin=clim[0], vmax=clim[1], aspect="equal",
    )
    axes[1].set_title(
        f"Lagrangian path-integral  (Shean Eq. 7)\n{_med_iqr(lag)} m ice/yr", fontsize=11
    )
    fig.colorbar(
        im1, ax=axes[1], fraction=0.045,
        label="melt rate (m ice/yr)\nneg = melt, pos = accretion",
    )

    for ax in axes:
        ax.set_xlabel("x (m, EPSG:3031)")
    axes[0].set_ylabel("y (m, EPSG:3031)")

    window_start = ds.attrs.get("window_start", config.START_TIME)
    window_end = ds.attrs.get("window_end", config.END_TIME)
    fig.suptitle(
        f"Beardmore {res_label} pilot — Eulerian vs Lagrangian melt rate "
        f"({window_start} → {window_end})",
        fontsize=13,
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    print(f"wrote {fig_path}")
    return fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nc", type=Path, help="Input NetCDF path.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output figure path (default: derived from <nc> name).")
    args = parser.parse_args()
    main(args.nc, args.out)
