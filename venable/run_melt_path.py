"""Venable production-config Lagrangian PATH melt rate (Shean pair_median).

The old ``venable.run_melt`` / ``venable.run_pseudospectral`` drivers predate
the 2026-07-02 solver verdict (keep the path solver as production; the
pseudospectral linear-perturbation inverses are retired as melt products).
This driver runs ONLY the production Lagrangian path solver with the exact
PIG configuration on the curated stack:

    lagrangian_melt_rate(output="path", seed_stride=1,
                         aggregator="pair_median", pairs="all",
                         min_dt_yr=1.5, max_dt_yr=2.5, dt_yr=0.05)

Caveats specific to Venable:
- Velocity is the static MEaSUREs phase map (no time-varying product covers
  the Bellingshausen sector). Per the 2026-07-03 velocity-representation
  finding, static velocity inflates trajectory-solver flux on NONSTATIONARY
  fast ice; Venable is slow (~<1 km/yr), so the risk is modest — flag it in
  write-ups rather than blocking on it.
- ``load_velocity_on_grid`` zero-fills residual velocity NaNs (mostly
  ocean/ice-free cells). Zero-fill freezes parcels rather than dropping
  them and can ring ∇·u at fill edges; the GL buffer excludes the worst of
  it. Revisit if the QC shows edge artefacts.

Run (125 m curated stack, disconnect-safe):

    cd /wd2/projects/stereo_melt
    nohup /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u \
        -m venable.run_melt_path --res 125 \
        > venable/logs/run_melt_path_125m.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.flux import grounding_buffer, integrate_basal_flux
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.melt import lagrangian_melt_rate
from stereo_melt.stack import load_basin_stack

from venable import config
from venable.run_melt import (
    load_floating_mask,
    load_smb_on_grid,
    load_velocity_on_grid,
)


def load_stack_res(res_m: int) -> xr.DataArray:
    prefix = "venable_stack" if res_m == config.RES else f"venable_stack_{res_m}m"
    stack, _ = load_basin_stack(
        config.PROCESSED_DIR,
        prefix,
        config.START_TIME,
        config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    return stack


def load_grounded_mask(stack: xr.DataArray) -> xr.DataArray:
    """BedMachine mask==2 (grounded ice) on the stack grid (nearest)."""
    ds = xr.open_dataset(config.BEDMACHINE_NC)
    buf = 2000.0
    x_min = float(stack["x"].min()) - buf
    x_max = float(stack["x"].max()) + buf
    y_min = float(stack["y"].min()) - buf
    y_max = float(stack["y"].max()) + buf
    by = ds["y"].values
    if by[0] > by[-1]:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
    else:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    on_grid = sub.load().interp(x=stack["x"], y=stack["y"], method="nearest")
    grounded = (on_grid == 2)
    grounded.name = "grounded_mask"
    return grounded


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res", type=int, default=125,
                   help="stack resolution suffix to load (default 125)")
    args = p.parse_args()
    tag = f"{args.res}m"
    t0 = time.time()

    print(f"Loading curated {tag} stack...")
    stack = load_stack_res(args.res)
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    floating = load_floating_mask(stack)
    grounded = load_grounded_mask(stack)
    stack = stack.where(floating)
    print(f"  floating cells: {int(floating.sum()):,}")

    vx, vy, vel_source = load_velocity_on_grid(stack)
    a_dot = load_smb_on_grid(stack)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    print("Running production Lagrangian path solver (narrates)...")
    lagr = lagrangian_melt_rate(
        stack, vx, vy, a_dot=a_dot, d=firn, dt_yr=0.05,
        seed_stride=1, output="path", aggregator="pair_median",
        pairs="all", min_dt_yr=1.5, max_dt_yr=2.5,
    )
    mr = lagr.melt_rate.where(floating)
    print(
        f"  melt_rate: median={float(mr.median()):.2f}  "
        f"IQR=[{float(mr.quantile(0.25)):.2f}, {float(mr.quantile(0.75)):.2f}] "
        f"m ice/yr  finite-cells={int(mr.notnull().sum()):,}"
    )

    res_m = abs(float(stack["x"].values[1] - stack["x"].values[0]))
    dom = grounding_buffer(
        np.asarray(floating.values, bool), np.asarray(grounded.values, bool),
        2000.0, res_m,
    )
    q = np.nan_to_num(lagr["count"].values, nan=0) >= 10
    b = integrate_basal_flux(mr.values, dom, res_m, quality=q)
    print(
        f"  GL-2km clip flux: area={b['area_km2']:.0f} km2  med={b['median_myr']:+.2f}  "
        f"Gt/yr: CLIP={b['gt_clip']:.2f}  raw={b['gt_raw']:.2f}  robust={b['gt_robust']:.2f}"
    )

    out = xr.Dataset(
        {
            "melt_rate_lagrangian": mr,
            "lagrangian_count": lagr["count"],
            "lagrangian_rmse": lagr["rmse"],
            "H_f_mean": lagr["H_f_mean"].where(floating),
            "a_dot": lagr["a_dot"],
            "floating_mask": floating,
        },
        attrs={
            "solver": "lagrangian_melt_rate output=path seed_stride=1 "
                      "pair_median pairs=all 1.5-2.5yr dt=0.05 (PIG production config)",
            "velocity_source": vel_source + " (static)",
            "units": "m ice yr^-1; Shean convention: negative = melt",
            "flux_clip_gt_yr": float(b["gt_clip"]),
        },
    )
    out_nc = (
        config.PROCESSED_DIR
        / f"venable_melt_path_{tag}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    comp = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_nc, encoding=comp)
    print(f"Saved -> {out_nc}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    im = axes[0].imshow(mr.values, cmap=melt_cmap(), norm=melt_norm(vmax=10.0),
                        interpolation="nearest")
    axes[0].set_title(f"Venable path melt {tag} (m ice/yr)", fontsize=10)
    add_melt_colorbar(fig, im, ax=axes[0], shrink=0.8)
    im2 = axes[1].imshow(np.where(floating.values, lagr["count"].values, np.nan),
                         cmap="viridis", interpolation="nearest")
    axes[1].set_title("path deposit count", fontsize=10)
    fig.colorbar(im2, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    out_png = config.FIGURES_DIR / f"melt_path_{tag}.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_png}")
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
