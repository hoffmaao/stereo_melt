"""Quick 2-panel comparison of Eulerian vs Lagrangian melt rate (250 m pilot).

Shean convention throughout: negative = melt, positive = accretion.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from nansen import config


def main() -> None:
    nc = config.RESULTS_DIR / f"nansen_melt_250m_{config.START_TIME}_{config.END_TIME}.nc"
    ds = xr.open_dataset(nc)

    floating = ds["floating_mask"].astype(bool)
    eul = ds["melt_rate_eulerian"].where(floating)
    lag = ds["melt_rate_lagrangian"].where(floating)

    def med_iqr(da: xr.DataArray) -> str:
        med = float(da.median())
        q25, q75 = float(da.quantile(0.25)), float(da.quantile(0.75))
        n = int(da.notnull().sum())
        return f"median={med:+.2f}  IQR=[{q25:+.2f}, {q75:+.2f}]  N={n}"

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    clim = (-5.0, 5.0)
    extent = [float(ds["x"].min()), float(ds["x"].max()),
              float(ds["y"].min()), float(ds["y"].max())]

    im0 = axes[0].imshow(
        eul.values, extent=extent, origin="upper",
        cmap="RdBu_r", vmin=clim[0], vmax=clim[1], aspect="equal",
    )
    axes[0].set_title(
        f"Eulerian  (Shean Eq. 10)\n{med_iqr(eul)} m ice/yr", fontsize=11
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.045, label="melt rate (m ice/yr)\nneg = melt, pos = accretion")

    im1 = axes[1].imshow(
        lag.values, extent=extent, origin="upper",
        cmap="RdBu_r", vmin=clim[0], vmax=clim[1], aspect="equal",
    )
    axes[1].set_title(
        f"Lagrangian path-integral  (Shean Eq. 7)\n{med_iqr(lag)} m ice/yr", fontsize=11
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.045, label="melt rate (m ice/yr)\nneg = melt, pos = accretion")

    for ax in axes:
        ax.set_xlabel("x (m, EPSG:3031)")
    axes[0].set_ylabel("y (m, EPSG:3031)")

    fig.suptitle(
        f"Nansen 250 m pilot — Eulerian vs Lagrangian melt rate "
        f"({config.START_TIME} → {config.END_TIME})",
        fontsize=13,
    )

    out_png = config.FIGURES_DIR / "eulerian_lagrangian_250m.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    ds.close()


if __name__ == "__main__":
    main()
