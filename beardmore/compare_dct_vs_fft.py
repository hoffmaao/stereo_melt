"""DCT vs FFT comparison for the stationary linear-perturbation inversion.

Compares up to four flavors of the stationary Stubblefield inverse on the
same Beardmore stack:

  1. ``closed-FFT`` — closed-form Tikhonov in the FFT (periodic) basis,
     with Gaussian NaN infill and edge padding to keep the periodic wrap
     from coupling the data envelope to its mirror.
  2. ``closed-DCT`` — closed-form Tikhonov in the DCT-II (reflective) basis;
     no padding required because the implicit boundary is ``∂f/∂n = 0``,
     not periodic. The "what works quickly and is easiest to understand"
     baseline.
  3. ``CG-FFT`` — masked conjugate-gradient with the FFT operator
     (skipped with ``--skip-cg``).
  4. ``CG-DCT`` — masked conjugate-gradient with the DCT-II operator
     (skipped with ``--skip-cg``).

Common knobs (``H_ref``, viscosity, Tikhonov, length-scale) are identical
across paths, so tightening of the result or stability gain at low
Tikhonov is attributable to the spatial basis or solver, not the data.

Run::

    # Fast: just the two closed-form paths (default for quick iteration)
    python -m beardmore.compare_dct_vs_fft --variant is2 --skip-cg

    # Full: include the CG paths
    python -m beardmore.compare_dct_vs_fft --variant is2 --tikhonov 1e-2 --length-scale 100

The figure is written next to the NetCDF.
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

if "STEREO_MELT_BACKEND" not in os.environ:
    try:
        import cupy as _cp  # noqa: F401

        os.environ["STEREO_MELT_BACKEND"] = "cupy"
    except ImportError:
        pass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.backend import backend as _BACKEND
from stereo_melt.dynamics import (
    lagrangian_frame_stack,
    stationary_pseudospectral_lagrangian_inverse,
)
from stereo_melt.melt import linear_inverse_lagrangian_melt_rate

from beardmore import config
from beardmore.compare_v2_vs_v3 import VARIANTS, _patch_config_for_variant
from beardmore.run_melt import (
    _imshow_xr,
    load_floating_mask,
    load_stack,
    load_velocity_on_grid,
)


def _summary(name: str, da: xr.DataArray) -> None:
    print(
        f"[{name:<22s}] median={float(da.median()):+.2f}  "
        f"IQR=[{float(da.quantile(0.25)):+.2f}, {float(da.quantile(0.75)):+.2f}]  "
        f"p05/p95=[{float(da.quantile(0.05)):+.2f}, {float(da.quantile(0.95)):+.2f}]  "
        f"abs_max={float(np.abs(da).max()):.1f}  m ice/yr"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", default="is2", choices=list(VARIANTS),
        help="ASP variant (default: is2).",
    )
    parser.add_argument(
        "--tikhonov", type=float, default=1e-2,
        help="Dimensionless Tikhonov for both CG paths (default 1e-2).",
    )
    parser.add_argument(
        "--length-scale", dest="length_scale", type=float, default=100.0,
        help="Sobolev-H¹ smoothness length L in meters; 0 disables (default 100).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=80,
        help="CG iteration cap (default 80).",
    )
    parser.add_argument(
        "--cg-tol", type=float, default=1e-6,
        help="CG relative-residual tolerance (default 1e-6).",
    )
    parser.add_argument(
        "--skip-closed", action="store_true",
        help="Skip the closed-form Tikhonov baselines (useful for fast iteration).",
    )
    parser.add_argument(
        "--skip-cg", action="store_true",
        help="Skip the iterative CG paths and compare only the two closed-form solvers.",
    )
    parser.add_argument(
        "--closed-reg", dest="closed_reg", type=float, default=1e-1,
        help="Tikhonov reg for the closed-form solvers (default 1e-1).",
    )
    args = parser.parse_args()
    _patch_config_for_variant(args.variant)
    print(f"variant: {args.variant}")
    print(f"  PROCESSED_DIR: {config.PROCESSED_DIR}")
    print(f"  RESULTS_DIR:   {config.RESULTS_DIR}")
    print(f"  FIGURES_DIR:   {config.FIGURES_DIR}")
    print(f"  Tikhonov={args.tikhonov:g}  length_scale={args.length_scale} m")
    print(f"stereo_melt backend: {_BACKEND}")

    print(f"\nLoading {args.variant} tilt-corrected stack...")
    stack = load_stack()
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    floating = load_floating_mask(stack)
    stack_f = stack.where(floating)
    print(f"  floating-ice fraction: {float(floating.mean()):.3f}")

    vx, vy, vel_source = load_velocity_on_grid(stack_f)
    print(f"  velocity: {vel_source}")
    if "time" in vx.dims:
        vx_m = vx.mean("time", skipna=True)
    else:
        vx_m = vx
    if "time" in vy.dims:
        vy_m = vy.mean("time", skipna=True)
    else:
        vy_m = vy

    print("Building Lagrangian-frame stack (one-time)...")
    h_lag = lagrangian_frame_stack(stack_f, vx_m, vy_m, dt_yr=0.05)

    common = dict(
        floating_mask=floating,
        dt_yr=0.05,
        h_lag_precomputed=h_lag,
    )

    closed_fft = None
    closed_dct = None
    if not args.skip_closed:
        print("\n=== closed-form, transform=FFT (infill+pad NaN-fill baseline) ===")
        closed_fft = linear_inverse_lagrangian_melt_rate(
            stack_f, vx, vy,
            boundary_fix="infill+pad",
            transform="fft",
            reg=args.closed_reg,
            **common,
        )
        _summary("closed_FFT_infillpad", closed_fft.melt_rate)

        print("\n=== closed-form, transform=DCT (reflective boundary, no pad/infill) ===")
        closed_dct = linear_inverse_lagrangian_melt_rate(
            stack_f, vx, vy,
            boundary_fix="off",
            transform="dct",
            reg=args.closed_reg,
            **common,
        )
        _summary("closed_DCT", closed_dct.melt_rate)

    cg_fft = None
    cg_dct = None
    if not args.skip_cg:
        print("\n=== CG-LSQ, transform=FFT (periodic boundary, masked CG) ===")
        cg_fft = stationary_pseudospectral_lagrangian_inverse(
            stack_f, vx, vy,
            tikhonov=args.tikhonov,
            length_scale_m=args.length_scale,
            max_iter=args.max_iter,
            cg_tol=args.cg_tol,
            transform="fft",
            verbose=True,
            **common,
        )
        _summary("CG_FFT", cg_fft.melt_rate)

        print("\n=== CG-LSQ, transform=DCT (reflective boundary, masked CG) ===")
        cg_dct = stationary_pseudospectral_lagrangian_inverse(
            stack_f, vx, vy,
            tikhonov=args.tikhonov,
            length_scale_m=args.length_scale,
            max_iter=args.max_iter,
            cg_tol=args.cg_tol,
            transform="dct",
            verbose=True,
            **common,
        )
        _summary("CG_DCT", cg_dct.melt_rate)

    panels = []
    if closed_fft is not None:
        panels.append(("closed-FFT (infill+pad)", closed_fft.melt_rate))
    if closed_dct is not None:
        panels.append(("closed-DCT (reflective)", closed_dct.melt_rate))
    if cg_fft is not None:
        panels.append(("CG-FFT (periodic)", cg_fft.melt_rate))
    if cg_dct is not None:
        panels.append(("CG-DCT (reflective)", cg_dct.melt_rate))

    if not panels:
        print("Nothing to plot — both --skip-closed and --skip-cg were set.")
        return

    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_dct_vs_fft_{config.START_TIME}_{config.END_TIME}_"
          f"tk{args.tikhonov:g}_L{int(args.length_scale)}.nc"
    )
    print(f"\nSaving -> {out_nc}")
    out_vars = {"floating_mask": floating}
    if closed_fft is not None:
        out_vars["melt_rate_closed_fft"] = closed_fft.melt_rate.rename("melt_rate_closed_fft")
    if closed_dct is not None:
        out_vars["melt_rate_closed_dct"] = closed_dct.melt_rate.rename("melt_rate_closed_dct")
    if cg_fft is not None:
        out_vars["melt_rate_cg_fft"] = cg_fft.melt_rate.rename("melt_rate_cg_fft")
    if cg_dct is not None:
        out_vars["melt_rate_cg_dct"] = cg_dct.melt_rate.rename("melt_rate_cg_dct")

    # Pull H_ref from whichever solver ran.
    h_ref_attr = float("nan")
    for ds in (closed_fft, closed_dct, cg_fft, cg_dct):
        if ds is not None:
            h_ref_attr = float(ds.attrs.get("H_ref_m", h_ref_attr))
            break

    out_attrs = {
        "shelf": config.SHELF,
        "window_start": config.START_TIME,
        "window_end": config.END_TIME,
        "velocity_source": vel_source,
        "tikhonov": args.tikhonov,
        "length_scale_m": float(args.length_scale),
        "closed_reg": float(args.closed_reg),
        "cg_max_iter": int(args.max_iter),
        "H_ref_m": h_ref_attr,
    }
    if cg_fft is not None:
        out_attrs["cg_fft_iter"] = int(cg_fft.attrs.get("cg_iter", -1))
        out_attrs["cg_fft_converged"] = int(cg_fft.attrs.get("cg_converged", 0))
    if cg_dct is not None:
        out_attrs["cg_dct_iter"] = int(cg_dct.attrs.get("cg_iter", -1))
        out_attrs["cg_dct_converged"] = int(cg_dct.attrs.get("cg_converged", 0))
    xr.Dataset(out_vars, attrs=out_attrs).to_netcdf(out_nc)

    fig_path = (
        config.FIGURES_DIR
        / f"dct_vs_fft_tk{args.tikhonov:g}_L{int(args.length_scale)}.png"
    )
    print(f"Plotting -> {fig_path}")
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                             squeeze=False, constrained_layout=True)
    clim = (-8.0, 8.0)

    for col, (title, da) in enumerate(panels):
        im = _imshow_xr(axes[0, col], da, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
        axes[0, col].set_title(
            f"{title}\nmedian={float(da.median()):+.2f}  "
            f"IQR=[{float(da.quantile(0.25)):+.2f}, {float(da.quantile(0.75)):+.2f}]  "
            f"abs_max={float(np.abs(da).max()):.0f}"
        )
        fig.colorbar(im, ax=axes[0, col], fraction=0.045)
        axes[0, col].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Beardmore {args.variant} — DCT-II reflective vs FFT periodic boundary "
        f"(tk={args.tikhonov:g}, L={int(args.length_scale)} m, closed_reg={args.closed_reg:g}) "
        f"— {config.START_TIME} → {config.END_TIME}",
        fontsize=12,
    )
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("done.")


if __name__ == "__main__":
    main()
