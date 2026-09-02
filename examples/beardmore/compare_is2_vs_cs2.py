"""4-panel comparison: IS2 v1 / IS2 v2 / CS2 v1 / CS2 v2 from the linear-inverse compare NetCDFs.

Reads the existing per-variant comparison NetCDFs produced by
:mod:`beardmore.compare_linear_inverse` (one for ``--variant is2`` and one
for ``--variant cs2``) and renders a unified 2x2 figure on a shared color
scale, plus an absolute-difference panel (CS2_v2 - IS2_v2) so the
ASP-coregistration source effect can be read at a glance.

Run after both compare runs are finished:

    python -m beardmore.compare_is2_vs_cs2
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from beardmore import config
from beardmore.run_melt import _imshow_xr


def _summary(name: str, da: xr.DataArray) -> None:
    print(
        f"[{name}] median={float(da.median()):.2f}  "
        f"IQR=[{float(da.quantile(0.25)):.2f}, {float(da.quantile(0.75)):.2f}]  "
        f"p05/p95=[{float(da.quantile(0.05)):.2f}, {float(da.quantile(0.95)):.2f}]  "
        f"abs_max={float(np.abs(da).max()):.1f}  m ice/yr"
    )


def main() -> None:
    config.ensure_output_dirs()

    is2_nc = (
        config.BASIN_DIR / "results"
        / f"beardmore_linear_inverse_compare_{config.START_TIME}_{config.END_TIME}.nc"
    )
    cs2_nc = (
        config.BASIN_DIR / "results" / "cs2"
        / f"beardmore_linear_inverse_compare_{config.START_TIME}_{config.END_TIME}.nc"
    )
    for p in (is2_nc, cs2_nc):
        if not p.exists():
            raise SystemExit(f"missing input {p}")

    print(f"Loading IS2 compare from {is2_nc.name}")
    is2 = xr.open_dataset(is2_nc)
    print(f"Loading CS2 compare from {cs2_nc.name}")
    cs2 = xr.open_dataset(cs2_nc)

    is2_v1 = is2["melt_rate_v1"]
    is2_v2 = is2["melt_rate_v2"]
    cs2_v1 = cs2["melt_rate_v1"]
    cs2_v2 = cs2["melt_rate_v2"]
    floating = is2["floating_mask"].astype(bool)

    # CS2 grid is the same target grid (25 m, same AOI), so the fields
    # are directly comparable — but verify shapes match before differencing.
    if is2_v2.shape != cs2_v2.shape:
        raise SystemExit(
            f"shape mismatch: IS2 v2 {is2_v2.shape} vs CS2 v2 {cs2_v2.shape}"
        )
    diff = (cs2_v2 - is2_v2).where(floating)

    print()
    _summary("IS2 v1", is2_v1)
    _summary("IS2 v2", is2_v2)
    _summary("CS2 v1", cs2_v1)
    _summary("CS2 v2", cs2_v2)
    _summary("CS2 v2 - IS2 v2", diff)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    clim = (-8.0, 8.0)

    im00 = _imshow_xr(axes[0, 0], is2_v1, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[0, 0].set_title(
        f"IS2 v1 (fillna(0))\nmedian={float(is2_v1.median()):.2f}  "
        f"p95={float(is2_v1.quantile(0.95)):.2f}  abs_max={float(np.abs(is2_v1).max()):.0f}"
    )
    fig.colorbar(im00, ax=axes[0, 0], fraction=0.045)

    im01 = _imshow_xr(axes[0, 1], is2_v2, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[0, 1].set_title(
        f"IS2 v2 (infill+pad)\nmedian={float(is2_v2.median()):.2f}  "
        f"p95={float(is2_v2.quantile(0.95)):.2f}  abs_max={float(np.abs(is2_v2).max()):.0f}"
    )
    fig.colorbar(im01, ax=axes[0, 1], fraction=0.045)

    axes[0, 2].set_visible(False)

    im10 = _imshow_xr(axes[1, 0], cs2_v1, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[1, 0].set_title(
        f"CS2 v1 (fillna(0))\nmedian={float(cs2_v1.median()):.2f}  "
        f"p95={float(cs2_v1.quantile(0.95)):.2f}  abs_max={float(np.abs(cs2_v1).max()):.0f}"
    )
    fig.colorbar(im10, ax=axes[1, 0], fraction=0.045)

    im11 = _imshow_xr(axes[1, 1], cs2_v2, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[1, 1].set_title(
        f"CS2 v2 (infill+pad)\nmedian={float(cs2_v2.median()):.2f}  "
        f"p95={float(cs2_v2.quantile(0.95)):.2f}  abs_max={float(np.abs(cs2_v2).max()):.0f}"
    )
    fig.colorbar(im11, ax=axes[1, 1], fraction=0.045)

    im12 = _imshow_xr(axes[1, 2], diff, cmap="PuOr", vmin=-3.0, vmax=3.0)
    axes[1, 2].set_title(
        f"CS2 v2 − IS2 v2\nmedian={float(diff.median()):.2f}  "
        f"std={float(diff.std()):.2f}"
    )
    fig.colorbar(im12, ax=axes[1, 2], fraction=0.045)

    for ax in axes.ravel():
        ax.set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Beardmore IS2 vs CS2 ASP-coregistration — linear-inverse v1/v2 "
        f"({config.START_TIME} → {config.END_TIME})",
        fontsize=12,
    )

    fig_path = config.BASIN_DIR / "figures" / "is2_vs_cs2_linear_inverse.png"
    print(f"\nSaving figure -> {fig_path}")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("done.")


if __name__ == "__main__":
    main()
