"""3-way comparison: v2 (closed-form, constant H) vs v3 (variable-H pseudo-spectral CG) vs the existing constant-H pseudo-spectral CG.

Loads the IS2 tilt-corrected stack (BAD_EPOCHS applied), builds a
firn-corrected `H_total(x,y)` field from BedMachine, then runs:

  - v2_const_closed   : linear_inverse_lagrangian_melt_rate (boundary_fix=infill+pad)
                         — closed-form Tikhonov, constant H_ref, no mask.
  - v2_const_psCG     : stationary_pseudospectral_lagrangian_inverse
                         — CG, mask-aware, constant H_ref.
  - v3_varH_psCG      : variable_H_pseudospectral_lagrangian_inverse
                         — CG, mask-aware, first-order Taylor expansion in H(x,y).

Writes a NetCDF with all three fields and a side-by-side figure.

Run:

    python -m beardmore.compare_v2_vs_v3
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
from scipy.interpolate import RegularGridInterpolator

from stereo_melt.backend import backend as _BACKEND
from stereo_melt.dynamics import (
    lagrangian_frame_stack,
    stationary_pseudospectral_lagrangian_inverse,
    variable_H_pseudospectral_lagrangian_inverse,
)
from stereo_melt.io.bedmachine import load_bedmachine
from stereo_melt.melt import linear_inverse_lagrangian_melt_rate

from beardmore import config
from beardmore.run_melt import (
    _imshow_xr,
    load_floating_mask,
    load_stack,
    load_velocity_on_grid,
)


def build_H_total_field(
    H_f_mean: xr.DataArray, rho_w: float = 1027.0, rho_i: float = 918.0
) -> xr.DataArray:
    """Return firn-corrected total ice column thickness on the stack grid.

    H_total = H_f_mean + (rho_w / (rho_w - rho_i)) * firn(BedMachine)

    NaN cells outside the floating mask are kept as NaN; the v3 wrapper
    fills them with H_ref internally.
    """
    print("Loading BedMachine firn ...")
    bm = load_bedmachine(config.BEDMACHINE_NC)
    bm_x, bm_y, firn = bm["x"], bm["y"], bm["firn"]
    if bm_y[0] > bm_y[-1]:  # paper has y descending; we need ascending for interp
        bm_y = bm_y[::-1]
        firn = firn[::-1, :]
    interp = RegularGridInterpolator(
        (bm_y, bm_x), firn, bounds_error=False, fill_value=np.nan
    )
    xs = H_f_mean.x.values
    ys = H_f_mean.y.values
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    firn_grid = interp(np.stack([yy, xx], axis=-1))
    firn_da = xr.DataArray(
        firn_grid, dims=("y", "x"), coords={"y": ys, "x": xs}, name="firn"
    )
    factor = rho_w / (rho_w - rho_i)
    H_total = H_f_mean + factor * firn_da
    H_total.name = "H_total"
    return H_total, firn_da


def _summary(name: str, da: xr.DataArray) -> None:
    print(
        f"[{name}] median={float(da.median()):.2f}  "
        f"IQR=[{float(da.quantile(0.25)):.2f}, {float(da.quantile(0.75)):.2f}]  "
        f"p05/p95=[{float(da.quantile(0.05)):.2f}, {float(da.quantile(0.95)):.2f}]  "
        f"abs_max={float(np.abs(da).max()):.1f}  m ice/yr"
    )


VARIANTS = {
    "is2": ("ASP", ""),
    "cs2": ("ASP_cs2", "cs2"),
    "is2+cs2": ("ASP_is2cs2", "is2cs2"),
}


def _patch_config_for_variant(variant: str) -> None:
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
    parser.add_argument("--tikhonov", type=float, default=1e-2,
                        help="Dimensionless Tikhonov for the CG solvers (v2_psCG and v3).")
    parser.add_argument("--length-scale", dest="length_scale", type=float, default=100.0,
                        help="Sobolev-H¹ smoothness length L in meters; 0 disables.")
    args = parser.parse_args()
    _patch_config_for_variant(args.variant)
    print(f"variant: {args.variant}")
    print(f"  PROCESSED_DIR: {config.PROCESSED_DIR}")
    print(f"  RESULTS_DIR:   {config.RESULTS_DIR}")
    print(f"  FIGURES_DIR:   {config.FIGURES_DIR}")
    print(f"stereo_melt backend: {_BACKEND}")

    print(f"\nLoading {args.variant} tilt-corrected stack...")
    stack = load_stack()
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    floating = load_floating_mask(stack)
    stack_f = stack.where(floating)
    print(f"  floating-ice fraction: {float(floating.mean()):.3f}")

    vx, vy, vel_source = load_velocity_on_grid(stack_f)
    print(f"  velocity: {vel_source}")

    # Lagrangian-frame stack built ONCE for all three solvers.
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

    # Build H_total field (firn-corrected) from BedMachine.
    from stereo_melt.freeboard import freeboard_to_thickness
    H_f_mean = freeboard_to_thickness(stack_f, d=0.0).mean("time", skipna=True)
    H_total, firn = build_H_total_field(H_f_mean)
    H_ref = float(H_total.where(floating).mean(skipna=True))
    H_max = float(H_total.where(floating).max())
    H_min = float(H_total.where(floating).min())
    dH_max_rel = max(abs(H_max - H_ref), abs(H_min - H_ref)) / H_ref
    print(
        f"H_total over floating: H_ref={H_ref:.1f} m  "
        f"min/max=[{H_min:.0f}, {H_max:.0f}]  "
        f"max|ΔH|/H_ref={dH_max_rel:.2f}"
    )

    common = dict(
        floating_mask=floating,
        dt_yr=0.05,
        h_lag_precomputed=h_lag,
    )

    print("\n=== v2 closed-form (constant H, infill+pad, no W-mask) ===")
    v2_close = linear_inverse_lagrangian_melt_rate(
        stack_f, vx, vy,
        H_ref=H_ref,  # use H_ref from H_total so all three solvers anchor on the same H
        boundary_fix="infill+pad",
        reg=1e-1,
        **common,
    )
    _summary("v2_const_closed", v2_close.melt_rate)

    print("\n=== v2 pseudospectral CG (constant H, mask-aware) ===")
    v2_psCG = stationary_pseudospectral_lagrangian_inverse(
        stack_f, vx, vy,
        H_ref=H_ref,
        tikhonov=args.tikhonov,
        length_scale_m=args.length_scale,
        max_iter=80,
        cg_tol=1e-6,
        verbose=True,
        **common,
    )
    _summary("v2_const_psCG", v2_psCG.melt_rate)

    print("\n=== v3 variable-H pseudospectral CG ===")
    v3 = variable_H_pseudospectral_lagrangian_inverse(
        stack_f, vx, vy,
        H_total_field=H_total,
        H_ref=H_ref,
        tikhonov=args.tikhonov,
        length_scale_m=args.length_scale,
        max_iter=80,
        cg_tol=1e-6,
        eps_H=0.05,
        verbose=True,
        **common,
    )
    _summary("v3_varH_psCG", v3.melt_rate)

    # Save
    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_v2_vs_v3_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving -> {out_nc}")
    out_ds = xr.Dataset(
        {
            "melt_rate_v2_closed": v2_close.melt_rate.rename("melt_rate_v2_closed"),
            "melt_rate_v2_psCG":   v2_psCG.melt_rate.rename("melt_rate_v2_psCG"),
            "melt_rate_v3_varH":   v3.melt_rate.rename("melt_rate_v3_varH"),
            "H_total_used":        v3.H_total_used,
            "dH":                  v3.dH,
            "floating_mask":       floating,
        },
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "velocity_source": vel_source,
            "H_ref_m": H_ref,
            "dH_max_rel": dH_max_rel,
            "v2_closed_boundary_fix": v2_close.attrs["boundary_fix"],
            "v2_psCG_iter": int(v2_psCG.attrs["cg_iter"]),
            "v2_psCG_converged": int(v2_psCG.attrs["cg_converged"]),
            "v3_iter": int(v3.attrs["cg_iter"]),
            "v3_converged": int(v3.attrs["cg_converged"]),
            "v3_eps_H": v3.attrs["eps_H"],
        },
    )
    out_ds.to_netcdf(out_nc)

    # Plot
    fig_path = config.FIGURES_DIR / "v2_vs_v3_variableH.png"
    print(f"Plotting -> {fig_path}")
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    clim = (-8.0, 8.0)

    panels = [
        ("v2 closed-form\n(constant H, infill+pad)", v2_close.melt_rate),
        ("v2 pseudospectral CG\n(constant H, mask-aware)", v2_psCG.melt_rate),
        ("v3 variable-H pseudospectral CG\n(linearized in H)", v3.melt_rate),
    ]
    for col, (title, da) in enumerate(panels):
        im = _imshow_xr(axes[0, col], da, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
        axes[0, col].set_title(
            f"{title}\nmedian={float(da.median()):.2f}  "
            f"IQR=[{float(da.quantile(0.25)):.2f}, {float(da.quantile(0.75)):.2f}]  "
            f"p95={float(da.quantile(0.95)):.2f}"
        )
        fig.colorbar(im, ax=axes[0, col], fraction=0.045)

    # Difference: v3 - v2_psCG (isolates the variable-H effect)
    diff_var = (v3.melt_rate - v2_psCG.melt_rate).where(floating)
    im_diff = _imshow_xr(axes[1, 0], diff_var, cmap="PuOr", vmin=-3, vmax=3)
    axes[1, 0].set_title(
        f"v3 − v2_psCG (isolated variable-H effect)\n"
        f"median={float(diff_var.median()):.2f}  std={float(diff_var.std()):.2f}"
    )
    fig.colorbar(im_diff, ax=axes[1, 0], fraction=0.045)

    # H_total field for context
    im_H = _imshow_xr(axes[1, 1], v3.H_total_used, cmap="viridis", vmin=400, vmax=1100)
    axes[1, 1].set_title(
        f"H_total (firn-corrected)\nH_ref={H_ref:.0f} m  max|ΔH|={dH_max_rel*100:.0f}% of H_ref"
    )
    fig.colorbar(im_H, ax=axes[1, 1], fraction=0.045, label="m")

    # ΔH for context
    im_dH = _imshow_xr(axes[1, 2], v3.dH, cmap="RdBu_r", vmin=-200, vmax=200)
    axes[1, 2].set_title(f"ΔH = H_total − H_ref")
    fig.colorbar(im_dH, ax=axes[1, 2], fraction=0.045, label="m")

    for ax in axes.ravel():
        ax.set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Beardmore IS2 — variable-H pseudo-spectral inverse (v3) vs constant-H baselines "
        f"({config.START_TIME} → {config.END_TIME})",
        fontsize=12,
    )
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("done.")


if __name__ == "__main__":
    main()
