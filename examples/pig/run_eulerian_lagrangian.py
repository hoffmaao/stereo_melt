"""Quick two-solver preview: Eulerian + path-integrated Lagrangian only.

A stripped-down :mod:`pig.run_melt` that skips its opt-in inverse stages
and flux integration and plots a 2-panel comparison. Intended for fast
iteration on the wider 2010-2024 stack; uses ``pairs="consecutive"`` by
default so the Lagrangian loop runs in tens of minutes instead of hours.

Run:

    python -m pig.run_eulerian_lagrangian --res 250
    python -m pig.run_eulerian_lagrangian --res 250 --pairs all   # full
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force env-local proj.db before pyproj imports.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.melt import eulerian_melt_rate, lagrangian_melt_rate

from pig import config
from pig.run_melt import (
    _imshow_xr,
    load_floating_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
    plot_inputs,
)


def plot_two_panel(
    euler: xr.Dataset,
    lagr: xr.Dataset,
    out_path: Path,
    clim: tuple[float, float] = (-60.0, 60.0),
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    mcmap = melt_cmap()
    mnorm = melt_norm(vmax=max(abs(clim[0]), abs(clim[1])))
    im0 = _imshow_xr(axes[0], euler.melt_rate, cmap=mcmap, norm=mnorm)
    axes[0].set_title(
        f"Eulerian melt (m ice/yr)\n"
        f"median={float(euler.melt_rate.median()):.2f}"
    )
    add_melt_colorbar(fig, im0, ax=axes[0], fraction=0.045)

    im1 = _imshow_xr(axes[1], lagr.melt_rate, cmap=mcmap, norm=mnorm)
    axes[1].set_title(
        f"Lagrangian (path-integrated) melt (m ice/yr)\n"
        f"median={float(lagr.melt_rate.median()):.2f}"
    )
    add_melt_colorbar(fig, im1, ax=axes[1], fraction=0.045)

    diff = lagr.melt_rate - euler.melt_rate
    im2 = _imshow_xr(axes[2], diff, cmap="PuOr", vmin=-2.0, vmax=2.0)
    axes[2].set_title("Lagrangian − Eulerian")
    fig.colorbar(im2, ax=axes[2], fraction=0.045)

    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    fig.suptitle(
        f"Pine Island Eulerian + path-integrated Lagrangian — "
        f"{config.START_TIME} to {config.END_TIME}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(res_override: float | None = None, pairs: str = "consecutive") -> None:
    config.ensure_output_dirs()

    if res_override is not None:
        stack_prefix = f"pig_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "pig_stack"
        out_suffix = ""

    print("Loading stack...")
    stack = load_stack(stack_prefix=stack_prefix)
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    print("Building floating-ice mask (BedMachine v3)...")
    floating = load_floating_mask(stack)
    print(f"  floating-ice fraction of AOI: {float(floating.mean()):.3f}")
    stack = stack.where(floating)

    print("Loading velocity...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  velocity source: {vel_source}")

    print("Loading SMB (RACMO2.4p1)...")
    a_dot = load_smb_on_grid(stack)
    print(
        f"  a_dot range: {float(a_dot.min()):.3f} .. {float(a_dot.max()):.3f} m ice/yr"
    )

    print("Loading firn air content (BedMachine, static)...")
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)
    finite = firn.values[np.isfinite(firn.values)]
    print(f"  firn median={float(np.median(finite)):.2f} m")

    inputs_png = config.FIGURES_DIR / f"melt_inputs{out_suffix}.png"
    plot_inputs(stack, vx, vy, a_dot, firn, inputs_png)
    print(f"  wrote {inputs_png}")

    print("Running Eulerian solver (robust dh/dt IRLS)...")
    euler = eulerian_melt_rate(stack, vx, vy, a_dot=a_dot, d=firn, robust_dh_dt=True)
    print(
        f"  melt_rate: median={float(euler.melt_rate.median()):.2f}  "
        f"IQR=[{float(euler.melt_rate.quantile(0.25)):.2f}, "
        f"{float(euler.melt_rate.quantile(0.75)):.2f}] m ice/yr"
    )

    print(f"Running path-integrated Lagrangian solver (pairs={pairs!r})...")
    lagr = lagrangian_melt_rate(
        stack, vx, vy,
        a_dot=a_dot, d=firn,
        dt_yr=0.05, seed_stride=2,
        pairs=pairs, min_dt_yr=2.0 / 12.0,
    )
    print(
        f"  melt_rate: median={float(lagr.melt_rate.median()):.2f}  "
        f"IQR=[{float(lagr.melt_rate.quantile(0.25)):.2f}, "
        f"{float(lagr.melt_rate.quantile(0.75)):.2f}] m ice/yr  "
        f"finite={int(lagr.melt_rate.notnull().sum())} cells"
    )

    euler_melt = euler.melt_rate.where(floating)
    lagr_melt = lagr.melt_rate.where(floating)

    out_nc = (
        config.RESULTS_DIR
        / f"pig_melt_eul_lagr{out_suffix}_{pairs}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Saving -> {out_nc}")
    ds = xr.Dataset(
        {
            "melt_rate_eulerian": euler_melt,
            "melt_rate_lagrangian": lagr_melt,
            "dHdt": euler.dHdt.where(floating),
            "flux_div": euler.flux_div.where(floating),
            "H_f_mean": euler.H_f_mean.where(floating),
            "a_dot": a_dot,
            "floating_mask": floating,
            "lagrangian_count": lagr["count"].where(floating),
        },
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "grid_res_m": float(config.RES),
            "velocity_source": vel_source,
            "lagrangian_pairs": pairs,
            "smb_source": "RACMO2.4p1 smbgl (Zenodo 19255213), window-integrated",
        },
    )
    ds.to_netcdf(out_nc)

    fig_png = config.FIGURES_DIR / f"melt_eul_lagr{out_suffix}_{pairs}.png"
    plot_two_panel(euler, lagr, fig_png)
    print(f"  wrote {fig_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res", type=float, default=None,
        help="Resolution variant to load (e.g. 250 for pig_stack_250m_*).",
    )
    parser.add_argument(
        "--pairs", choices=("consecutive", "all"), default="consecutive",
        help="Lagrangian pair generation. 'consecutive' is fast (~100 pairs); "
        "'all' is C(T,2) pairs and can take hours on the wider 2010-2024 stack.",
    )
    args = parser.parse_args()
    main(res_override=args.res, pairs=args.pairs)
