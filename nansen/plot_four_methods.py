"""4-panel comparison: Eulerian, Lagrangian path-integration, linear-FFT, linear-DCT.

Loads pre-computed melt fields from the Nansen run_melt NetCDF — does not re-run any
solvers. Reuses the floating mask and ``_imshow_xr`` helper from ``run_melt``.
"""

from __future__ import annotations

import argparse
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

from nansen import config
from nansen.run_melt import _imshow_xr


def _summary(name: str, da: xr.DataArray) -> None:
    print(
        f"[{name:<24}] median={float(da.median()):+.2f}  "
        f"IQR=[{float(da.quantile(0.25)):+.2f}, {float(da.quantile(0.75)):+.2f}]  "
        f"p05/p95=[{float(da.quantile(0.05)):+.2f}, {float(da.quantile(0.95)):+.2f}]  "
        f"abs_max={float(np.abs(da).max()):.1f}  m ice/yr"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--melt-nc",
        default=None,
        help="Path to nansen_melt_<window>.nc (defaults to config window).",
    )
    ap.add_argument("--vmin", type=float, default=-8.0)
    ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--out", default=None, help="Output PNG path.")
    args = ap.parse_args()

    melt_nc = Path(args.melt_nc) if args.melt_nc else (
        config.RESULTS_DIR
        / f"nansen_melt_{config.START_TIME}_{config.END_TIME}.nc"
    )
    if not melt_nc.exists():
        raise SystemExit(f"Melt NetCDF not found: {melt_nc}")

    print(f"Loading 4 methods from {melt_nc.name}")
    ds = xr.open_dataset(melt_nc)
    floating = ds["floating_mask"].astype(bool)
    eulerian = ds["melt_rate_eulerian"].where(floating)
    lagrangian = ds["melt_rate_lagrangian"].where(floating)
    linear_fft = ds["melt_rate_linear_fft"].where(floating)
    linear_dct = ds["melt_rate_linear_dct"].where(floating)

    _summary("Eulerian", eulerian)
    _summary("Lagrangian path-int.", lagrangian)
    _summary("linear-FFT (Stubblefield)", linear_fft)
    _summary("linear-DCT (Stubblefield)", linear_dct)

    fig, axes = plt.subplots(2, 2, figsize=(13, 14), constrained_layout=True)
    panels = [
        (axes[0, 0], eulerian, "Eulerian\n(Shean 2019 Eq. 10)"),
        (axes[0, 1], lagrangian, "Lagrangian path-integration\n(Shean 2019 Eq. 7)"),
        (axes[1, 0], linear_fft, "linear-inverse FFT\n(Stubblefield 2023, infill+pad)"),
        (axes[1, 1], linear_dct, "linear-inverse DCT\n(Stubblefield 2023, reflective)"),
    ]
    for ax, da, title in panels:
        im = _imshow_xr(ax, da, cmap="RdBu_r", vmin=args.vmin, vmax=args.vmax)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, label="m ice / yr")
    for ax in axes[-1, :]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")

    fig.suptitle(
        f"Nansen basal melt rate — 4-method comparison ({config.START_TIME} → {config.END_TIME})",
        fontsize=13,
    )

    out_path = Path(args.out) if args.out else (
        config.FIGURES_DIR / f"nansen_four_methods_{config.START_TIME}_{config.END_TIME}.png"
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
