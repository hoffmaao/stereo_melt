"""Apply both pseudo-spectral linear-perturbation inverses to Beardmore_Shelf.

Loads the existing Beardmore_Shelf stack + floating mask + velocity, runs the
time-dependent Lagrangian and Eulerian linear-perturbation inverses,
writes per-time-slice melt-rate fields to NetCDF, and produces a
comparison figure with time-means and selected epochs.

Separate from :mod:`beardmore_shelf.run_melt` (which runs the three existing
classic solvers) so the pseudo-spectral pipeline can be iterated
without disturbing the existing driver.

Run:

    python -m beardmore_shelf.run_pseudospectral
"""

from __future__ import annotations

import os
import sys

# PROJ_DATA fix for this conda env's broken base proj.db.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.dynamics import (
    pseudospectral_eulerian_inverse,
    pseudospectral_lagrangian_inverse,
)
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.stack import load_basin_stack

from beardmore_shelf import config
from beardmore_shelf.run_melt import (
    load_floating_mask,
    load_velocity_on_grid,
)


def _load_any_stack() -> tuple[xr.DataArray, Path]:
    return load_basin_stack(
        config.PROCESSED_DIR,
        "beardmore_shelf_stack",
        config.START_TIME,
        config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )

SECONDS_PER_YEAR = 86400.0 * 365.25


def _imshow_xr(ax, da: xr.DataArray, *, cmap, vmin=None, vmax=None, norm=None):
    # `norm` and `vmin`/`vmax` are mutually exclusive in matplotlib; melt-rate
    # panels pass the symmetric-log `melt_norm`, everything else stays linear.
    kw = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    im = ax.imshow(
        da.values,
        extent=[
            float(da["x"].min()),
            float(da["x"].max()),
            float(da["y"].min()),
            float(da["y"].max()),
        ],
        origin="upper", cmap=cmap, aspect="equal", **kw,
    )
    return im


def plot_pseudospectral_summary(
    lag: xr.Dataset,
    eul: xr.Dataset,
    out_path: Path,
    clim=(-5.0, 5.0),
) -> None:
    """Top row: time-mean melt from each solver + their difference.
    Bottom row: first / mid / last epoch of the Lagrangian solver."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    lag_mean = lag.melt_rate.mean("time", skipna=True)
    eul_mean = eul.melt_rate.mean("time", skipna=True)

    # LADDIE symmetric-log melt scale (black at zero, log decades outward).
    mcmap = melt_cmap()
    mnorm = melt_norm(vmax=max(abs(clim[0]), abs(clim[1])))
    im0 = _imshow_xr(axes[0, 0], lag_mean, cmap=mcmap, norm=mnorm)
    axes[0, 0].set_title("Lagrangian PS time-mean (m ice/yr)")
    add_melt_colorbar(fig, im0, ax=axes[0, 0], fraction=0.045)

    im1 = _imshow_xr(axes[0, 1], eul_mean, cmap=mcmap, norm=mnorm)
    axes[0, 1].set_title("Eulerian PS time-mean (m ice/yr)")
    add_melt_colorbar(fig, im1, ax=axes[0, 1], fraction=0.045)

    diff = eul_mean - lag_mean
    im2 = _imshow_xr(axes[0, 2], diff, cmap="PuOr", vmin=-2.0, vmax=2.0)
    axes[0, 2].set_title("Eulerian − Lagrangian (time-mean)")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.045)

    n_t = lag.sizes["time"]
    sel = [0, n_t // 2, max(0, n_t - 2)]  # first / mid / second-to-last
    for col, ti in enumerate(sel):
        im = _imshow_xr(axes[1, col], lag.melt_rate.isel(time=ti),
                        cmap=mcmap, norm=mnorm)
        t_label = str(lag["time"].values[ti])[:10]
        axes[1, col].set_title(f"Lagrangian PS  t={t_label}")
        add_melt_colorbar(fig, im, ax=axes[1, col], fraction=0.045)

    for ax in axes.ravel():
        ax.set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Pseudo-spectral linear-perturbation inverses — Beardmore_Shelf "
        f"{config.START_TIME} to {config.END_TIME}", fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config.ensure_output_dirs()

    print("Loading stack...")
    stack, stack_path = _load_any_stack()
    print(f"  loaded {stack_path.name}")
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    print("Building floating-ice mask...")
    floating = load_floating_mask(stack)
    stack = stack.where(floating)

    print("Loading velocity...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  velocity source: {vel_source}")
    print("Loading firn air content (BedMachine, static)...")
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)
    firn_finite = firn.values[np.isfinite(firn.values)]
    print(
        f"  firn (m): median={float(np.median(firn_finite)):.2f}  "
        f"IQR=[{float(np.percentile(firn_finite, 25)):.2f}, "
        f"{float(np.percentile(firn_finite, 75)):.2f}]"
    )


    # ------------------------------------------------------------------
    # Lagrangian-frame pseudo-spectral inverse
    # ------------------------------------------------------------------
    print("\nRunning Lagrangian pseudo-spectral inverse...")
    lag = pseudospectral_lagrangian_inverse(
        stack, vx, vy,
        floating_mask=floating, d=firn,
        tikhonov=1e-1,
        max_iter=80,
        cg_tol=1e-5,
        dt_yr=0.05,
        verbose=True,
    )
    print(
        f"  H_ref={lag.attrs['H_ref_m']:.1f} m  γ={lag.attrs['gamma_dimless']:.3e}  "
        f"t_r={lag.attrs['tr_yr']:.1f} yr"
    )
    mr = lag.melt_rate
    print(
        f"  Lagrangian PS time-mean: median={float(mr.mean('time').median()):.2f}  "
        f"IQR=[{float(mr.mean('time').quantile(0.25)):.2f}, "
        f"{float(mr.mean('time').quantile(0.75)):.2f}] m ice/yr"
    )

    # ------------------------------------------------------------------
    # Eulerian pseudo-spectral inverse
    # ------------------------------------------------------------------
    print("\nRunning Eulerian pseudo-spectral inverse...")
    eul = pseudospectral_eulerian_inverse(
        stack, vx, vy,
        floating_mask=floating, d=firn,
        tikhonov=1e-1,
        max_iter=80,
        cg_tol=1e-5,
        verbose=True,
    )
    print(
        f"  H_ref={eul.attrs['H_ref_m']:.1f} m  "
        f"(α_x, α_y)=({eul.attrs['alpha_x']:.3f}, {eul.attrs['alpha_y']:.3f})  "
        f"γ={eul.attrs['gamma_dimless']:.3e}"
    )
    mr_e = eul.melt_rate
    print(
        f"  Eulerian PS time-mean: median={float(mr_e.mean('time').median()):.2f}  "
        f"IQR=[{float(mr_e.mean('time').quantile(0.25)):.2f}, "
        f"{float(mr_e.mean('time').quantile(0.75)):.2f}] m ice/yr"
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_shelf_pseudospectral_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving -> {out_nc}")
    ds_out = xr.Dataset(
        {
            "melt_rate_lagrangian_ps": lag.melt_rate.rename("melt_rate_lagrangian_ps"),
            "melt_rate_eulerian_ps": eul.melt_rate.rename("melt_rate_eulerian_ps"),
            "H_f_mean": eul.H_f_mean,
            "floating_mask": floating,
        },
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "grid_res_m": float(config.RES),
            "velocity_source": vel_source,
            "lagrangian_H_ref_m": lag.attrs["H_ref_m"],
            "lagrangian_gamma_dimless": lag.attrs["gamma_dimless"],
            "lagrangian_cg_iter": lag.attrs["cg_iter"],
            "lagrangian_cg_converged": int(lag.attrs["cg_converged"]),
            "eulerian_H_ref_m": eul.attrs["H_ref_m"],
            "eulerian_alpha_x": eul.attrs["alpha_x"],
            "eulerian_alpha_y": eul.attrs["alpha_y"],
            "eulerian_gamma_dimless": eul.attrs["gamma_dimless"],
            "eulerian_cg_iter": eul.attrs["cg_iter"],
            "eulerian_cg_converged": int(eul.attrs["cg_converged"]),
        },
    )
    ds_out.to_netcdf(out_nc)

    out_png = config.FIGURES_DIR / "pseudospectral_summary.png"
    plot_pseudospectral_summary(lag, eul, out_png)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
