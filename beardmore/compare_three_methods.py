"""3-way melt-rate comparison: v1 (legacy), v2 (boundary-fixed), Lagrangian path integration.

Reuses ``melt_rate_v1`` / ``melt_rate_v2`` from
``results/beardmore_linear_inverse_compare_<window>.nc`` (produced by
:mod:`beardmore.compare_linear_inverse`) and computes the Shean-2019
Lagrangian path-integration field on the same bad-epoch-dropped stack.
The three are then plotted side by side with a consistent color scale.

GPU is default if cupy is installed; override with ``STEREO_MELT_BACKEND=numpy``.
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

if "STEREO_MELT_BACKEND" not in os.environ:
    try:
        import cupy as _cp  # noqa: F401
        os.environ["STEREO_MELT_BACKEND"] = "cupy"
    except ImportError:
        pass

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.backend import backend as _BACKEND
from stereo_melt.melt import lagrangian_melt_rate

from beardmore import config
from beardmore.run_melt import (
    _imshow_xr,
    load_floating_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
)


def _summary(name: str, da: xr.DataArray) -> None:
    print(
        f"[{name}] median={float(da.median()):.2f}  "
        f"IQR=[{float(da.quantile(0.25)):.2f}, {float(da.quantile(0.75)):.2f}]  "
        f"p05/p95=[{float(da.quantile(0.05)):.2f}, {float(da.quantile(0.95)):.2f}]  "
        f"abs_max={float(np.abs(da).max()):.1f}  m ice/yr"
    )


def _plot_three(
    v1: xr.DataArray,
    v2: xr.DataArray,
    lagr: xr.DataArray,
    out_path: Path,
    clim: tuple[float, float] = (-8.0, 8.0),
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)

    im0 = _imshow_xr(axes[0], v1, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[0].set_title("v1: linear-inverse, fillna(0)\n(legacy ringing)")
    fig.colorbar(im0, ax=axes[0], fraction=0.045)

    im1 = _imshow_xr(axes[1], v2, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[1].set_title("v2: linear-inverse, infill+pad\n(boundary-fixed)")
    fig.colorbar(im1, ax=axes[1], fraction=0.045)

    im2 = _imshow_xr(axes[2], lagr, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[2].set_title("Lagrangian path integration\n(Shean 2019 Eq. 7, hydrostatic)")
    fig.colorbar(im2, ax=axes[2], fraction=0.045)

    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    fig.suptitle(
        f"Beardmore IS2 melt-rate comparison "
        f"({config.START_TIME} → {config.END_TIME})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config.ensure_output_dirs()
    print(f"stereo_melt backend: {_BACKEND}")

    compare_nc = (
        config.RESULTS_DIR
        / f"beardmore_linear_inverse_compare_{config.START_TIME}_{config.END_TIME}.nc"
    )
    if not compare_nc.exists():
        raise SystemExit(
            f"Run beardmore.compare_linear_inverse first; expected {compare_nc}"
        )
    print(f"Loading existing v1/v2 from {compare_nc.name}...")
    cmp_ds = xr.open_dataset(compare_nc)
    v1 = cmp_ds["melt_rate_v1"]
    v2 = cmp_ds["melt_rate_v2"]
    floating = cmp_ds["floating_mask"].astype(bool)

    print("Loading IS2 tilt-corrected stack (BAD_EPOCHS applied)...")
    stack = load_stack()
    print(
        f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}"
    )
    stack = stack.where(floating)

    print("Loading velocity + SMB...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    a_dot = load_smb_on_grid(stack)

    print("\n=== Computing Lagrangian path-integration melt rate ===")
    lagr_ds = lagrangian_melt_rate(
        stack, vx, vy,
        a_dot=a_dot,
        dt_yr=0.05,
        seed_stride=2,
        pairs="all",
        min_dt_yr=2.0 / 12.0,
    )
    lagr = lagr_ds.melt_rate.where(floating)

    _summary("v1 (linear, fillna)", v1)
    _summary("v2 (linear, infill+pad)", v2)
    _summary("Lagrangian path", lagr)

    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_three_methods_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving 3-method NetCDF -> {out_nc}")
    out_ds = xr.Dataset(
        {
            "melt_rate_v1": v1.rename("melt_rate_v1"),
            "melt_rate_v2": v2.rename("melt_rate_v2"),
            "melt_rate_lagrangian": lagr.rename("melt_rate_lagrangian"),
            "floating_mask": floating,
        },
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "velocity_source": vel_source,
            "lagrangian_pairs": "all",
            "lagrangian_min_dt_yr": 2.0 / 12.0,
            "v1_source": str(compare_nc),
            "v2_source": str(compare_nc),
        },
    )
    out_ds.to_netcdf(out_nc)

    fig_path = config.FIGURES_DIR / "three_methods_comparison.png"
    print(f"Writing 3-panel figure -> {fig_path}")
    _plot_three(v1, v2, lagr, fig_path)
    print("done.")


if __name__ == "__main__":
    main()
