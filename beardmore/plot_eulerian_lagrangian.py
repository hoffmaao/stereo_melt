"""Eulerian + Lagrangian melt-rate 2-panel for Beardmore (Nansen-style).

Lightweight side-job: runs just the two physics-based solvers (no
linear-inverse stages), saves a small NetCDF, and writes a 2-panel
figure styled like ``nansen/figures/eulerian_lagrangian_250m.png``.

Shean convention throughout: negative = melt, positive = accretion.

Run::

    python -m beardmore.plot_eulerian_lagrangian [--res 250]
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.backend import backend as _BACKEND
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.melt import eulerian_melt_rate, lagrangian_melt_rate

from beardmore import config
from beardmore.run_melt import (
    load_floating_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
)


def _summary(name: str, da: xr.DataArray) -> None:
    print(
        f"[{name:<22s}] median={float(da.median()):+.2f}  "
        f"IQR=[{float(da.quantile(0.25)):+.2f}, {float(da.quantile(0.75)):+.2f}]  "
        f"p05/p95=[{float(da.quantile(0.05)):+.2f}, {float(da.quantile(0.95)):+.2f}]  "
        f"N={int(da.notnull().sum())}  m ice/yr"
    )


def _med_iqr(da: xr.DataArray) -> str:
    med = float(da.median())
    q25, q75 = float(da.quantile(0.25)), float(da.quantile(0.75))
    n = int(da.notnull().sum())
    return f"median={med:+.2f}  IQR=[{q25:+.2f}, {q75:+.2f}]  N={n}"


def main(res_override: float | None = None) -> None:
    config.ensure_output_dirs()
    print(f"stereo_melt backend: {_BACKEND}")

    if res_override is not None:
        stack_prefix = f"beardmore_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
        res_label = f"{int(round(res_override))} m"
    else:
        stack_prefix = "beardmore_stack"
        out_suffix = ""
        res_label = f"{int(round(config.RES))} m"

    print("\nLoading Beardmore tilt-corrected stack...")
    stack = load_stack(stack_prefix=stack_prefix)
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    floating = load_floating_mask(stack)
    print(f"  floating-ice fraction: {float(floating.mean()):.3f}")
    stack_f = stack.where(floating)

    print("Loading velocity + SMB...")
    vx, vy, vel_source = load_velocity_on_grid(stack_f)
    a_dot = load_smb_on_grid(stack_f)
    print(f"  velocity: {vel_source}")

    print("Loading firn air content (BedMachine, static)...")
    firn = load_firn_on_grid(stack_f, config.BEDMACHINE_NC)

    print("\n=== 1/2 Eulerian mass-conservation ===")
    eul_ds = eulerian_melt_rate(stack_f, vx, vy, a_dot=a_dot, d=firn, robust_dh_dt=True)
    eul = eul_ds.melt_rate.where(floating)
    _summary("eulerian", eul)

    grid_res_m = float(abs(stack_f["x"][1] - stack_f["x"][0]))
    seed_stride = max(1, round(50.0 / grid_res_m))
    print(f"\n=== 2/2 Lagrangian path integration (seed_stride={seed_stride}, {grid_res_m:.0f} m grid) ===")
    lagr_ds = lagrangian_melt_rate(
        stack_f, vx, vy, a_dot=a_dot, d=firn,
        dt_yr=0.05, seed_stride=seed_stride, pairs="all",
        min_dt_yr=2.0 / 12.0,
    )
    lagr = lagr_ds.melt_rate.where(floating)
    _summary("lagrangian", lagr)

    # ---- Save NetCDF (eul + lag + floating mask only) ----
    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_eulerian_lagrangian{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving -> {out_nc}")
    xr.Dataset(
        {
            "melt_rate_eulerian":   eul.rename("melt_rate_eulerian"),
            "melt_rate_lagrangian": lagr.rename("melt_rate_lagrangian"),
            "floating_mask":        floating,
        },
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "grid_res_m": grid_res_m,
            "velocity_source": vel_source,
            "convention": "Shean: negative=melt, positive=accretion",
        },
    ).to_netcdf(out_nc)

    # ---- Plot (Nansen-style 2-panel) ----
    fig_path = config.FIGURES_DIR / f"eulerian_lagrangian{out_suffix}.png"
    print(f"Plotting -> {fig_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    clim = (-16.0, 16.0)
    extent = [
        float(stack_f["x"].min()), float(stack_f["x"].max()),
        float(stack_f["y"].min()), float(stack_f["y"].max()),
    ]

    im0 = axes[0].imshow(
        eul.values, extent=extent, origin="upper",
        cmap="RdBu_r", vmin=clim[0], vmax=clim[1], aspect="equal",
    )
    axes[0].set_title(
        f"Eulerian  (Shean Eq. 10)\n{_med_iqr(eul)} m ice/yr", fontsize=11
    )
    fig.colorbar(
        im0, ax=axes[0], fraction=0.045,
        label="melt rate (m ice/yr)\nneg = melt, pos = accretion",
    )

    im1 = axes[1].imshow(
        lagr.values, extent=extent, origin="upper",
        cmap="RdBu_r", vmin=clim[0], vmax=clim[1], aspect="equal",
    )
    axes[1].set_title(
        f"Lagrangian path-integral  (Shean Eq. 7)\n{_med_iqr(lagr)} m ice/yr", fontsize=11
    )
    fig.colorbar(
        im1, ax=axes[1], fraction=0.045,
        label="melt rate (m ice/yr)\nneg = melt, pos = accretion",
    )

    for ax in axes:
        ax.set_xlabel("x (m, EPSG:3031)")
    axes[0].set_ylabel("y (m, EPSG:3031)")

    fig.suptitle(
        f"Beardmore {res_label} pilot — Eulerian vs Lagrangian melt rate "
        f"({config.START_TIME} → {config.END_TIME})",
        fontsize=13,
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help=(
            "Resolution variant to load (meters). When set, looks up "
            "beardmore_stack_<N>m*_<start>_<end>.nc and writes outputs at *<N>m* paths."
        ),
    )
    args = parser.parse_args()
    main(res_override=args.res)
