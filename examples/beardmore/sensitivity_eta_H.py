"""eta_bar and H_ref sensitivity sweep for the v2 (boundary-fixed) linear inverse.

The Stubblefield model has two scalar tile-mean parameters that have to be
guessed: column-averaged dynamic viscosity ``eta_bar`` (Pa s) and reference
ice thickness ``H_ref`` (m). They combine into the relaxation timescale
``t_r = 2 eta_bar / (rho_i g H_ref)``, which sets the spatial reach of
the kernels and the dimensionless extension ``gamma = E·t_r``. We expect
roughly:

  - large eta_bar / large H_ref -> longer t_r -> larger smoothing scale
    in the recovered ``m``, lower-amplitude fine-scale features.
  - small eta_bar / small H_ref -> shorter t_r -> sharper recovered
    ``m``, more sensitive to noise.

Two 1-D sweeps:
  - eta_bar in {1e13, 1e14, 1e15} Pa s at default H_ref.
  - H_ref in {0.5x, 1x, 1.5x} default at default eta_bar.

The Lagrangian-frame stack is built once (most of the per-run cost) and
reused across the sweep via the new ``h_lag_precomputed`` kwarg.
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
from stereo_melt.dynamics.lagrangian_inverse import lagrangian_frame_stack
from stereo_melt.melt import linear_inverse_lagrangian_melt_rate

from beardmore import config
from beardmore.run_melt import (
    _imshow_xr,
    load_floating_mask,
    load_stack,
    load_velocity_on_grid,
)


def _bulk_stats(da: xr.DataArray) -> dict:
    return {
        "median": float(da.median()),
        "p25": float(da.quantile(0.25)),
        "p75": float(da.quantile(0.75)),
        "p05": float(da.quantile(0.05)),
        "p95": float(da.quantile(0.95)),
        "abs_max": float(np.abs(da).max()),
    }


def _summary(name: str, s: dict) -> str:
    return (
        f"[{name}]  median={s['median']:.2f}  "
        f"IQR=[{s['p25']:.2f}, {s['p75']:.2f}]  "
        f"p05/p95=[{s['p05']:.2f}, {s['p95']:.2f}]  "
        f"abs_max={s['abs_max']:.1f} m ice/yr"
    )


def main() -> None:
    config.ensure_output_dirs()
    print(f"stereo_melt backend: {_BACKEND}")

    print("Loading IS2 tilt-corrected stack (BAD_EPOCHS applied)...")
    stack = load_stack()
    print(
        f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}"
    )

    print("Building floating-ice mask (BedMachine v3)...")
    floating = load_floating_mask(stack)
    stack_f = stack.where(floating)

    print("Loading velocity...")
    vx, vy, vel_source = load_velocity_on_grid(stack_f)
    print(f"  source: {vel_source}")

    if "time" in vx.dims:
        vx_m = vx.mean("time", skipna=True)
    else:
        vx_m = vx
    if "time" in vy.dims:
        vy_m = vy.mean("time", skipna=True)
    else:
        vy_m = vy

    print("\nBuilding Lagrangian-frame stack ONCE (will be reused across all sweep points)...")
    h_lag = lagrangian_frame_stack(stack_f, vx_m, vy_m, dt_yr=0.05)
    print(f"  h_lag built: time={h_lag.sizes['time']}, y={h_lag.sizes['y']}, x={h_lag.sizes['x']}")

    # Resolve default H_ref from the stack so the H sweep is anchored to it.
    from stereo_melt.freeboard import freeboard_to_thickness
    Hf_stack = freeboard_to_thickness(stack_f, d=0.0)
    H_ref_default = float(Hf_stack.mean("time", skipna=True).where(floating).mean(skipna=True))
    eta_default = 1e14
    print(f"  default H_ref = {H_ref_default:.1f} m, default eta_bar = {eta_default:.0e} Pa s")

    eta_values = [1e13, 1e14, 1e15]
    H_factors = [0.5, 1.0, 1.5]
    H_values = [H_ref_default * f for f in H_factors]

    common = dict(
        floating_mask=floating,
        reg=1e-1,
        dt_yr=0.05,
        boundary_fix="infill+pad",
        infill_max_iters=5,
        h_lag_precomputed=h_lag,
    )

    runs = []  # list of (label, group, value, ds)

    print("\n=== eta_bar sweep (H_ref = default) ===")
    for eta in eta_values:
        label = f"eta={eta:.0e}"
        print(f"  running {label} ...")
        ds = linear_inverse_lagrangian_melt_rate(
            stack_f, vx, vy,
            H_ref=H_ref_default,
            eta_bar=eta,
            **common,
        )
        s = _bulk_stats(ds.melt_rate)
        print("    " + _summary(label, s))
        runs.append(("eta", eta, label, ds, s))

    print("\n=== H_ref sweep (eta_bar = default 1e14) ===")
    for f, H in zip(H_factors, H_values):
        label = f"H={H:.0f}m({f:.2f}x)"
        print(f"  running {label} ...")
        ds = linear_inverse_lagrangian_melt_rate(
            stack_f, vx, vy,
            H_ref=H,
            eta_bar=eta_default,
            **common,
        )
        s = _bulk_stats(ds.melt_rate)
        print("    " + _summary(label, s))
        runs.append(("H", H, label, ds, s))

    # Save
    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_linear_inverse_sensitivity_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving sweep NetCDF -> {out_nc}")
    out_vars = {}
    for group, val, label, ds, s in runs:
        var_name = label.replace(" ", "").replace("=", "_").replace("(", "_").replace(")", "")
        out_vars[f"melt_{var_name}"] = ds.melt_rate.rename(f"melt_{var_name}")
    out_ds = xr.Dataset(
        out_vars,
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "velocity_source": vel_source,
            "boundary_fix": "infill+pad",
            "H_ref_default_m": H_ref_default,
            "eta_default_Pa_s": eta_default,
            "eta_sweep": ", ".join(f"{e:.0e}" for e in eta_values),
            "H_factors": ", ".join(f"{f:.2f}" for f in H_factors),
        },
    )
    out_ds.to_netcdf(out_nc)

    # Plot 2 x 3 grid
    fig_path = config.FIGURES_DIR / "linear_inverse_sensitivity.png"
    print(f"Writing sweep figure -> {fig_path}")
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)
    clim = (-8.0, 8.0)
    for i, (group, val, label, ds, s) in enumerate(runs):
        row = 0 if group == "eta" else 1
        col = i % 3
        ax = axes[row, col]
        im = _imshow_xr(ax, ds.melt_rate, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
        ax.set_title(
            f"{label}\nmedian={s['median']:.2f}  IQR=[{s['p25']:.1f},{s['p75']:.1f}]  "
            f"p95={s['p95']:.1f}",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, fraction=0.045)
        if col == 0:
            ax.set_ylabel("y (m)")
        if row == 1:
            ax.set_xlabel("x (m)")
    axes[0, 0].set_title("eta=1e13 — fast relaxation\n" + axes[0, 0].get_title(), fontsize=10)
    axes[1, 0].set_title("0.5x H_ref\n" + axes[1, 0].get_title(), fontsize=10)
    fig.suptitle(
        f"Beardmore IS2 linear-inverse v2 — eta_bar / H_ref sensitivity "
        f"({config.START_TIME} → {config.END_TIME})",
        fontsize=12,
    )
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("done.")


if __name__ == "__main__":
    main()
