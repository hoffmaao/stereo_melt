"""Run the Stubblefield linear inverse on a chosen ASP variant's stack twice
(``boundary_fix="off"`` v1 and ``"infill+pad"`` v2) and write a side-by-side
comparison NetCDF + figure.

Variants (mirrors ``beardmore.run_variant``):

  - ``is2``     : canonical IS2-only ASP coregistration (legacy top-level dirs)
  - ``cs2``     : CS2-only forward test (``processed/cs2/`` etc.)
  - ``is2+cs2`` : combined IS2+CS2 control (``processed/is2cs2/`` etc.)

Run:

    python -m beardmore.compare_linear_inverse                 # is2 (default)
    python -m beardmore.compare_linear_inverse --variant cs2
    python -m beardmore.compare_linear_inverse --variant is2+cs2
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

# GPU by default if cupy is installed. Override with STEREO_MELT_BACKEND=numpy.
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
from stereo_melt.melt import linear_inverse_lagrangian_melt_rate

from beardmore import config
from beardmore.run_melt import (
    _imshow_xr,
    load_floating_mask,
    load_stack,
    load_velocity_on_grid,
)


def _summary(name: str, ds: xr.Dataset) -> None:
    mr = ds.melt_rate
    finite = mr.notnull()
    print(
        f"[{name}] median={float(mr.median()):.2f}  "
        f"IQR=[{float(mr.quantile(0.25)):.2f}, {float(mr.quantile(0.75)):.2f}]  "
        f"p05/p95=[{float(mr.quantile(0.05)):.2f}, {float(mr.quantile(0.95)):.2f}]  "
        f"abs_max={float(np.abs(mr).max()):.1f}  finite={int(finite.sum())} m ice/yr"
    )


def _plot(
    v1: xr.Dataset,
    v2: xr.Dataset,
    out_path: Path,
    clim: tuple[float, float] = (-8.0, 8.0),
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)

    im0 = _imshow_xr(axes[0], v1.melt_rate, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[0].set_title(
        f"v1: boundary_fix='off' (fillna(0))\n"
        f"H_ref={v1.attrs['H_ref_m']:.0f} m, gamma={v1.attrs['gamma_dimless']:.2e}"
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.045)

    im1 = _imshow_xr(axes[1], v2.melt_rate, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[1].set_title(
        f"v2: boundary_fix='infill+pad'\n"
        f"infill σ={v2.attrs['infill_sigma_m']:.0f} m, pad={v2.attrs['pad_pixels']} px"
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.045)

    diff = v2.melt_rate - v1.melt_rate
    im2 = _imshow_xr(axes[2], diff, cmap="PuOr", vmin=-3.0, vmax=3.0)
    axes[2].set_title("v2 − v1 (m ice/yr)")
    fig.colorbar(im2, ax=axes[2], fraction=0.045)

    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    fig.suptitle(
        f"Beardmore IS2 linear inverse — boundary fix comparison "
        f"({config.START_TIME} → {config.END_TIME})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


VARIANTS = {
    "is2": ("ASP", ""),
    "cs2": ("ASP_cs2", "cs2"),
    "is2+cs2": ("ASP_is2cs2", "is2cs2"),
}


def _patch_config_for_variant(variant: str) -> None:
    """Repoint config paths to the chosen variant (mirrors run_variant._patch_config_for_variant)."""
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; pick from {list(VARIANTS)}")
    asp_subdir, suffix = VARIANTS[variant]
    config.STRIP_ALIGNED_DIR = config.STRIPS_DIR / asp_subdir / "asp_aligned"
    if suffix:
        config.PROCESSED_DIR = config.BASIN_DIR / "processed" / suffix
        config.FIGURES_DIR = config.BASIN_DIR / "figures" / suffix
        config.RESULTS_DIR = config.BASIN_DIR / "results" / suffix
    config.ensure_output_dirs()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="is2", choices=list(VARIANTS),
                        help="ASP variant to run on (default: is2).")
    args = parser.parse_args()

    _patch_config_for_variant(args.variant)
    print(f"variant: {args.variant}")
    print(f"  PROCESSED_DIR: {config.PROCESSED_DIR}")
    print(f"  RESULTS_DIR:   {config.RESULTS_DIR}")
    print(f"  FIGURES_DIR:   {config.FIGURES_DIR}")
    print(f"stereo_melt backend: {_BACKEND}")

    print(f"\nLoading {args.variant} tilt-corrected stack...")
    stack = load_stack()
    print(
        f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}"
    )

    print("Building floating-ice mask (BedMachine v3)...")
    floating = load_floating_mask(stack)
    print(f"  floating-ice fraction of AOI: {float(floating.mean()):.3f}")
    stack = stack.where(floating)

    print("Loading velocity...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  source: {vel_source}")

    common_kwargs = dict(
        floating_mask=floating,
        reg=1e-1,
        dt_yr=0.05,
    )

    print("\n=== Running v1 (boundary_fix='off' — legacy fillna(0)) ===")
    v1 = linear_inverse_lagrangian_melt_rate(
        stack, vx, vy, boundary_fix="off", **common_kwargs
    )
    _summary("v1", v1)

    print("\n=== Running v2 (boundary_fix='infill+pad') ===")
    v2 = linear_inverse_lagrangian_melt_rate(
        stack,
        vx,
        vy,
        boundary_fix="infill+pad",
        infill_max_iters=5,
        **common_kwargs,
    )
    _summary("v2", v2)

    diff = v2.melt_rate - v1.melt_rate
    print(
        f"\n[v2 - v1] median={float(diff.median()):.2f}  "
        f"abs_max={float(np.abs(diff).max()):.2f}  "
        f"std={float(diff.std()):.2f} m ice/yr"
    )

    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_linear_inverse_compare_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving comparison NetCDF -> {out_nc}")
    out_ds = xr.Dataset(
        {
            "melt_rate_v1": v1.melt_rate.rename("melt_rate_v1"),
            "melt_rate_v2": v2.melt_rate.rename("melt_rate_v2"),
            "H_f_mean": v1.H_f_mean,
            "flux_div": v1.flux_div,
            "floating_mask": floating,
        },
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "grid_res_m": float(config.RES),
            "velocity_source": vel_source,
            "v1_boundary_fix": v1.attrs["boundary_fix"],
            "v2_boundary_fix": v2.attrs["boundary_fix"],
            "v2_infill_sigma_m": v2.attrs["infill_sigma_m"],
            "v2_infill_max_iters": v2.attrs["infill_max_iters"],
            "v2_pad_pixels": v2.attrs["pad_pixels"],
            "H_ref_m": v1.attrs["H_ref_m"],
            "gamma_dimless": v1.attrs["gamma_dimless"],
            "tr_yr": v1.attrs["tr_yr"],
            "reg": v1.attrs["reg"],
        },
    )
    out_ds.to_netcdf(out_nc)

    fig_path = config.FIGURES_DIR / "linear_inverse_v1_vs_v2.png"
    print(f"Writing comparison figure -> {fig_path}")
    _plot(v1, v2, fig_path)
    print("done.")


if __name__ == "__main__":
    main()
