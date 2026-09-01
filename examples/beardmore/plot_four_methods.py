"""4-panel comparison: Eulerian, Lagrangian path-integration, closed-FFT, closed-DCT.

Loads pre-computed melt fields from existing result NetCDFs — does not re-run any
solvers. Reuses the floating mask and ``_imshow_xr`` helper from ``run_melt``.

Sources:
  Eulerian, Lagrangian → ``beardmore_melt_<window>.nc`` (run_melt)
  closed-FFT, closed-DCT → ``beardmore_dct_vs_fft_<window>_tk*_L*.nc`` (compare_dct_vs_fft)
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

from beardmore import config
from beardmore.run_melt import _imshow_xr


def _summary(name: str, da: xr.DataArray) -> None:
    print(
        f"[{name:<24}] median={float(da.median()):+.2f}  "
        f"IQR=[{float(da.quantile(0.25)):+.2f}, {float(da.quantile(0.75)):+.2f}]  "
        f"p05/p95=[{float(da.quantile(0.05)):+.2f}, {float(da.quantile(0.95)):+.2f}]  "
        f"abs_max={float(np.abs(da).max()):.1f}  m ice/yr"
    )


def _resolve_dct_fft_path(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"--dct-fft-nc not found: {p}")
        return p
    candidates = sorted(config.RESULTS_DIR.glob("beardmore_dct_vs_fft_*.nc"))
    if not candidates:
        raise SystemExit(
            f"No beardmore_dct_vs_fft_*.nc in {config.RESULTS_DIR}; "
            "run beardmore.compare_dct_vs_fft first or pass --dct-fft-nc."
        )
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--melt-nc",
        default=None,
        help="Path to beardmore_melt_<window>.nc (defaults to config window).",
    )
    ap.add_argument(
        "--dct-fft-nc",
        default=None,
        help="Path to beardmore_dct_vs_fft_*.nc (defaults to most recent in results/).",
    )
    ap.add_argument("--vmin", type=float, default=-8.0)
    ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--out", default=None, help="Output PNG path.")
    args = ap.parse_args()

    melt_nc = Path(args.melt_nc) if args.melt_nc else (
        config.RESULTS_DIR
        / f"beardmore_melt_{config.START_TIME}_{config.END_TIME}.nc"
    )
    if not melt_nc.exists():
        raise SystemExit(f"Eulerian/Lagrangian source not found: {melt_nc}")
    dct_fft_nc = _resolve_dct_fft_path(args.dct_fft_nc)

    print(f"Loading Eulerian + Lagrangian from {melt_nc.name}")
    melt_ds = xr.open_dataset(melt_nc)
    floating = melt_ds["floating_mask"].astype(bool)
    eulerian = melt_ds["melt_rate_eulerian"].where(floating)
    lagrangian = melt_ds["melt_rate_lagrangian"].where(floating)

    print(f"Loading closed-FFT + closed-DCT from {dct_fft_nc.name}")
    cf_ds = xr.open_dataset(dct_fft_nc)
    cf_floating = cf_ds["floating_mask"].astype(bool)
    closed_fft = cf_ds["melt_rate_closed_fft"].where(cf_floating)
    closed_dct = cf_ds["melt_rate_closed_dct"].where(cf_floating)

    _summary("Eulerian", eulerian)
    _summary("Lagrangian path-int.", lagrangian)
    _summary("closed-FFT (Tikhonov)", closed_fft)
    _summary("closed-DCT (Tikhonov)", closed_dct)

    fig, axes = plt.subplots(2, 2, figsize=(13, 14), constrained_layout=True)
    panels = [
        (axes[0, 0], eulerian, f"Eulerian\n({melt_nc.stem.split('_')[-2]} → {melt_nc.stem.split('_')[-1]})"),
        (axes[0, 1], lagrangian, "Lagrangian path-integration\n(Shean 2019 Eq. 7)"),
        (axes[1, 0], closed_fft, f"closed-FFT (Stubblefield 2023)\n({dct_fft_nc.stem})"),
        (axes[1, 1], closed_dct, "closed-DCT (Stubblefield 2023)\n(reflective basis)"),
    ]
    for ax, da, title in panels:
        im = _imshow_xr(ax, da, cmap="RdBu_r", vmin=args.vmin, vmax=args.vmax)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, label="m ice / yr")
    for ax in axes[-1, :]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")

    fig.suptitle("Beardmore basal melt rate — 4-method comparison", fontsize=13)

    out_path = Path(args.out) if args.out else (
        config.FIGURES_DIR / f"beardmore_four_methods_{config.START_TIME}_{config.END_TIME}.png"
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
