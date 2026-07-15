"""Plot 7-panel five-methods + Davison comparison from a saved NetCDF (PIG).

Decouples replotting (cheap) from re-running the full compute (expensive).
Reads ``pig_five_methods_<R>m_*.nc`` produced by ``compare_five_methods`` and
emits the same layout, with a tunable color range.

Run::

    python -m pig.plot_five_methods_from_nc <path/to/results.nc> [--vlim 20]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pig import config


def _imshow_xr(ax, da: xr.DataArray, *, cmap, vmin, vmax):
    return ax.imshow(
        da.values,
        extent=[
            float(da["x"].min()), float(da["x"].max()),
            float(da["y"].min()), float(da["y"].max()),
        ],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )


def main(nc_path: Path, vlim: float = 20.0, fig_path: Path | None = None) -> Path:
    ds = xr.open_dataset(nc_path)
    floating = ds["floating_mask"].astype(bool)

    eul        = ds["melt_rate_eulerian"].where(floating)
    lagr       = ds["melt_rate_lagrangian"].where(floating)
    closed_fft = ds["melt_rate_closed_fft"].where(floating)
    closed_dct = ds["melt_rate_closed_dct"].where(floating)
    dhdt_fft   = ds["melt_rate_dhdt_fft"].where(floating)
    dhdt_dct   = ds["melt_rate_dhdt_dct"].where(floating)
    davison    = ds["melt_rate_davison"].where(floating)

    window_start = ds.attrs.get("window_start", config.START_TIME)
    window_end   = ds.attrs.get("window_end", config.END_TIME)
    H_ref_m      = float(ds.attrs.get("H_ref_m", float("nan")))

    if fig_path is None:
        m = re.search(r"_(\d+)m_", nc_path.name)
        suffix = f"_{m.group(1)}m" if m else ""
        fig_path = config.FIGURES_DIR / f"five_methods_comparison{suffix}.png"

    panels = [
        ("1. Eulerian mass-cons.\n(Shean Eq. 10)",    eul),
        ("2. Lagrangian path-int.\n(Shean Eq. 7)",    lagr),
        ("3. closed-form FFT\n(infill+pad, reg=0.1)", closed_fft),
        ("4. closed-form DCT\n(reflective, reg=0.1)", closed_dct),
        ("5a. dh/dt FFT\n(per-pixel OLS, reg=10)",    dhdt_fft),
        ("5b. dh/dt DCT\n(per-pixel OLS, reg=10)",    dhdt_dct),
        ("Davison 2023\n(gridded, RACMO-FAC corr.)",  davison),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 4, 6.5), constrained_layout=True)
    for col, (title, da) in enumerate(panels):
        im = _imshow_xr(axes[col], da, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        axes[col].set_title(
            f"{title}\nmedian={float(da.median()):+.2f}  "
            f"IQR=[{float(da.quantile(0.25)):+.2f}, {float(da.quantile(0.75)):+.2f}]  "
            f"abs_max={float(np.abs(da).max()):.0f}",
            fontsize=10,
        )
        fig.colorbar(im, ax=axes[col], fraction=0.045)
        axes[col].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"PIG IS2 melt-rate comparison — 5 methods + Davison 2023 "
        f"({window_start} → {window_end}, H_ref={H_ref_m:.0f} m, clim=±{int(vlim)} m/yr)",
        fontsize=12,
    )
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    print(f"wrote {fig_path}")
    return fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nc", type=Path, help="Input NetCDF path.")
    parser.add_argument("--vlim", type=float, default=20.0,
                        help="Symmetric color limit (default 20 m ice/yr).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output figure path (default: derived from <nc> name).")
    args = parser.parse_args()
    main(args.nc, vlim=args.vlim, fig_path=args.out)
