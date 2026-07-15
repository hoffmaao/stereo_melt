"""Run the Venable Lagrangian melt-rate pipeline on the built stack.

Stage 4+ of the Venable pipeline. Loads the already-built DEM stack,
fetches matching velocity and SMB, and runs
:func:`stereo_melt.melt.lagrangian_melt_rate` (and Eulerian as a
side-by-side comparison). Writes a NetCDF with every variable of the
returned Dataset plus QC plots to ``venable/figures/``.

Run:

    python -m venable.run_melt

Assumes :mod:`venable.build_stack` has already produced
``processed/venable_stack_<start>_<end>.nc``.
"""

from __future__ import annotations

import os
import sys

# Force PROJ database to the active env before any pyproj import. This env's
# base-install proj.db has stale metadata that breaks EPSG code lookups.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.io.smb import smb_over_window
from stereo_melt.melt import (
    eulerian_melt_rate,
    lagrangian_melt_rate,
    linear_inverse_lagrangian_melt_rate,
)
from stereo_melt.stack import load_basin_stack

from venable import config

SECONDS_PER_YEAR = 86400.0 * 365.25


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------


def load_stack() -> xr.DataArray:
    stack, _ = load_basin_stack(
        config.PROCESSED_DIR,
        "venable_stack",
        config.START_TIME,
        config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    return stack


def _subset_velocity(
    ds: xr.Dataset, stack: xr.DataArray, vx_name: str, vy_name: str, buf_m: float = 2000.0
) -> xr.Dataset:
    """Sub-select a velocity dataset around the stack bbox, respecting
    y-coord direction (ascending vs descending)."""
    x_min = float(stack["x"].min()) - buf_m
    x_max = float(stack["x"].max()) + buf_m
    y_min = float(stack["y"].min()) - buf_m
    y_max = float(stack["y"].max()) + buf_m
    vy_src = ds["y"].values
    if vy_src[0] > vy_src[-1]:  # descending
        sub = ds.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
    else:
        sub = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    return sub[[vx_name, vy_name]].rename({vx_name: "vx", vy_name: "vy"}).load()


def load_velocity_on_grid(stack: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, str]:
    """Load velocity ``vx``/``vy`` cropped + resampled onto the stack grid.

    Velocity source is selected by the ``VENABLE_VELOCITY`` env var:
    ``measures`` (default, NSIDC-0754 phase map at 450 m), ``nsidc-0525``
    (Scheuchl 2012 Central Antarctica 2009 mosaic at 900 m), or
    ``its_live`` (annual mosaic at 120 m). Remaining NaN gaps are filled
    by nearest-valid neighbor.
    """
    requested = os.environ.get("VENABLE_VELOCITY", "measures").strip().lower()
    source = None
    if requested == "nsidc-0525":
        from stereo_melt.io.velocity import load_nsidc_0525
        if not config.NSIDC_0525_2009_NC.exists():
            raise SystemExit(
                f"NSIDC-0525 requested but file missing at {config.NSIDC_0525_2009_NC}"
            )
        ds = load_nsidc_0525(config.NSIDC_0525_2009_NC)
        sub = _subset_velocity(ds, stack, vx_name="vx", vy_name="vy")
        source = f"NSIDC-0525 (Scheuchl 2012, 2009 mosaic) 900 m (finite frac {float(sub['vx'].notnull().mean()):.2f})"
    elif requested == "its_live":
        if not config.ITS_LIVE_2019.exists():
            raise SystemExit(f"ITS_LIVE requested but file missing at {config.ITS_LIVE_2019}")
        ds = xr.open_dataset(config.ITS_LIVE_2019)
        sub = _subset_velocity(ds, stack, vx_name="vx", vy_name="vy")
        source = f"ITS_LIVE 2019 120 m (finite frac {float(sub['vx'].notnull().mean()):.2f})"
    else:
        # Default path: MEaSUREs preferred, ITS_LIVE fallback.
        if config.MEASURES_PHASE_NC.exists():
            ds = xr.open_dataset(config.MEASURES_PHASE_NC)
            sub = _subset_velocity(ds, stack, vx_name="VX", vy_name="VY")
            finite_frac = float(sub["vx"].notnull().mean())
            if finite_frac > 0.5:
                source = f"MEaSUREs 450 m (finite frac {finite_frac:.2f})"
        if source is None and config.ITS_LIVE_2019.exists():
            ds = xr.open_dataset(config.ITS_LIVE_2019)
            sub = _subset_velocity(ds, stack, vx_name="vx", vy_name="vy")
            source = f"ITS_LIVE 2019 120 m (finite frac {float(sub['vx'].notnull().mean()):.2f})"
    if source is None:
        raise SystemExit("No usable velocity source found — install MEaSUREs or ITS_LIVE.")

    print(f"  velocity source: {source}")

    # Resample onto the 25 m stack grid via linear interpolation.
    vx = sub["vx"].interp(x=stack["x"], y=stack["y"], method="linear")
    vy = sub["vy"].interp(x=stack["x"], y=stack["y"], method="linear")

    # Fill residual NaN pixels with 0 m/yr — a reasonable placeholder
    # over ice-free cells near the grounding zone where velocity is
    # undefined. Small fraction (< 5 %); won't bias the flow-divergence.
    vx = vx.fillna(0.0)
    vy = vy.fillna(0.0)

    return vx, vy, source


def load_floating_mask(stack: xr.DataArray) -> xr.DataArray:
    """Return a boolean floating-ice mask on the stack grid.

    Reads BedMachine Antarctica v3 ``mask`` (flag_values 0=ocean,
    1=ice_free_land, 2=grounded_ice, 3=floating_ice, 4=lake_vostok),
    crops to the stack bbox, and resamples to the 25 m target grid
    via nearest-neighbor (the mask is categorical).
    """
    if not config.BEDMACHINE_NC.exists():
        raise SystemExit(
            f"BedMachine not found at {config.BEDMACHINE_NC}. "
            "Cannot build floating-ice mask."
        )
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
    sub = sub.load()
    on_grid = sub.interp(x=stack["x"], y=stack["y"], method="nearest")
    floating = (on_grid == 3)
    floating.attrs = {
        "source": "BedMachine Antarctica v3 mask == 3 (floating_ice)",
        "flag_values": "0=ocean 1=ice_free_land 2=grounded_ice 3=floating_ice 4=lake_vostok",
    }
    floating.name = "floating_mask"
    return floating


def load_smb_on_grid(stack: xr.DataArray) -> xr.DataArray:
    """Integrate RACMO2.4p1 SMB over the stack time window and regrid to 25 m."""
    start = config.START_TIME
    end = config.END_TIME
    m_ice_cumulative = smb_over_window(
        str(config.RACMO_SMB_NC),
        stack["x"].values,
        stack["y"].values,
        start=start,
        end=end,
        method="linear",
    )
    # Cumulative m ice over the window → mean rate m ice / yr
    dt_years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / SECONDS_PER_YEAR
    m_ice_per_yr = m_ice_cumulative / dt_years
    return xr.DataArray(
        m_ice_per_yr,
        dims=("y", "x"),
        coords={"y": stack["y"].values, "x": stack["x"].values},
        name="a_dot",
        attrs={
            "units": "m ice yr^-1",
            "integration_window": f"{start} to {end}",
            "source": "RACMO2.4p1 smbgl (Zenodo 19255213)",
        },
    )


# ----------------------------------------------------------------------
# QC plotting
# ----------------------------------------------------------------------


def _imshow_xr(ax, da: xr.DataArray, *, cmap, vmin=None, vmax=None):
    im = ax.imshow(
        da.values,
        extent=[
            float(da["x"].min()),
            float(da["x"].max()),
            float(da["y"].min()),
            float(da["y"].max()),
        ],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    return im


def plot_inputs(stack, vx, vy, a_dot, firn, out_path: Path) -> None:
    """QC: time-mean surface, velocity magnitude, SMB, firn air content."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    im0 = _imshow_xr(axes[0], stack.mean("time", skipna=True), cmap="terrain")
    axes[0].set_title("time-mean surface (m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.045)

    speed = np.sqrt(vx**2 + vy**2)
    im1 = _imshow_xr(axes[1], speed, cmap="viridis")
    axes[1].set_title(f"|v| (m/yr)")
    fig.colorbar(im1, ax=axes[1], fraction=0.045)

    im2 = _imshow_xr(axes[2], a_dot, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axes[2].set_title(f"SMB rate (m ice/yr)\n{config.START_TIME}→{config.END_TIME}")
    fig.colorbar(im2, ax=axes[2], fraction=0.045)

    im3 = _imshow_xr(axes[3], firn, cmap="viridis")
    axes[3].set_title("firn air content (m)\nBedMachine static FAC")
    fig.colorbar(im3, ax=axes[3], fraction=0.045)

    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    fig.suptitle("Venable melt-rate inputs", fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_melt_comparison(
    euler: xr.Dataset,
    lagr: xr.Dataset,
    linv: xr.Dataset | None,
    out_path: Path,
    clim=(-5.0, 5.0),
) -> None:
    """QC: side-by-side Eulerian vs Lagrangian vs linear-inverse melt."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    im0 = _imshow_xr(axes[0, 0], euler.melt_rate, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[0, 0].set_title("Eulerian melt_rate (m ice/yr)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.045)

    im1 = _imshow_xr(axes[0, 1], lagr.melt_rate, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
    axes[0, 1].set_title("Lagrangian melt_rate (m ice/yr)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.045)

    if linv is not None:
        im2 = _imshow_xr(axes[0, 2], linv.melt_rate, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
        axes[0, 2].set_title("Linear inverse (Lagrangian-frame Stubblefield)")
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.045)
    else:
        axes[0, 2].set_visible(False)

    diff_lag = lagr.melt_rate - euler.melt_rate
    im3 = _imshow_xr(axes[1, 0], diff_lag, cmap="PuOr", vmin=-2.0, vmax=2.0)
    axes[1, 0].set_title("Lagrangian − Eulerian")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.045)

    if linv is not None:
        diff_lin = linv.melt_rate - lagr.melt_rate
        im4 = _imshow_xr(axes[1, 1], diff_lin, cmap="PuOr", vmin=-2.0, vmax=2.0)
        axes[1, 1].set_title("Linear inverse − Lagrangian")
        fig.colorbar(im4, ax=axes[1, 1], fraction=0.045)
    else:
        axes[1, 1].set_visible(False)

    im5 = _imshow_xr(axes[1, 2], euler.flux_div, cmap="RdBu", vmin=-5.0, vmax=5.0)
    axes[1, 2].set_title("∇·(H_f u) (m ice/yr)")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.045)

    for ax in axes.ravel():
        ax.set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Venable melt rate — {config.START_TIME} to {config.END_TIME}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def main() -> None:
    config.ensure_output_dirs()

    print("Loading stack...")
    stack = load_stack()
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    print("Building floating-ice mask (BedMachine v3)...")
    floating = load_floating_mask(stack)
    frac_floating = float(floating.mean())
    print(f"  floating-ice fraction of AOI: {frac_floating:.3f}")
    stack = stack.where(floating)

    print("Loading velocity...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(
        f"  vx range: {float(vx.min()):.1f} .. {float(vx.max()):.1f} m/yr  "
        f"vy range: {float(vy.min()):.1f} .. {float(vy.max()):.1f} m/yr"
    )

    print("Loading SMB (RACMO2.4p1)...")
    a_dot = load_smb_on_grid(stack)
    print(
        f"  a_dot range: {float(a_dot.min()):.3f} .. {float(a_dot.max()):.3f} m ice/yr  "
        f"(window-mean)"
    )

    print("Loading firn air content (BedMachine, static)...")
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)
    firn_finite = firn.values[np.isfinite(firn.values)]
    print(
        f"  firn (m): median={float(np.median(firn_finite)):.2f}  "
        f"IQR=[{float(np.percentile(firn_finite, 25)):.2f}, "
        f"{float(np.percentile(firn_finite, 75)):.2f}]  "
        f"finite frac={float(np.isfinite(firn.values).mean()):.3f}"
    )

    plot_inputs(stack, vx, vy, a_dot, firn, config.FIGURES_DIR / "melt_inputs.png")
    print(f"  wrote {config.FIGURES_DIR / 'melt_inputs.png'}")

    print("Running Eulerian solver...")
    euler = eulerian_melt_rate(stack, vx, vy, a_dot=a_dot, d=firn)
    print(f"  melt_rate: median={float(euler.melt_rate.median()):.2f}  "
          f"IQR=[{float(euler.melt_rate.quantile(0.25)):.2f}, "
          f"{float(euler.melt_rate.quantile(0.75)):.2f}] m ice/yr")

    print("Running Lagrangian solver (this can take a minute)...")
    # pairs="all" with a 2-month dt floor: short baselines amplify
    # coregistration residuals as noise / dt (see diagnose_pair_dhdt.py);
    # the 2019 stack has no consecutive pair with an annual baseline,
    # so we need cross-cluster pairs to get into Shean's regime.
    lagr = lagrangian_melt_rate(
        stack,
        vx,
        vy,
        a_dot=a_dot,
        d=firn,
        dt_yr=0.05,
        seed_stride=2,
        pairs="all",
        min_dt_yr=1.5,  # Shean 2019 1.5-2.5 yr window (raised from 2 mo 2026-06-15)
    )
    mr = lagr.melt_rate
    print(f"  melt_rate: median={float(mr.median()):.2f}  "
          f"IQR=[{float(mr.quantile(0.25)):.2f}, {float(mr.quantile(0.75)):.2f}] m ice/yr  "
          f"finite-cell-count={int((mr.notnull()).sum())}")

    print("Running linear-inverse solver (Lagrangian-frame Stubblefield)...")
    # reg=1e-1 from a sweep on the real Venable stack: lower reg leaves
    # broadband strip-residual noise (p95 outliers >100 m/yr at reg=1e-3),
    # higher reg over-smooths and biases the median away from zero. Pre-filter
    # default (sigma = H_ref/2) handles sub-H grid noise.
    try:
        linv = linear_inverse_lagrangian_melt_rate(
            stack,
            vx,
            vy,
            floating_mask=floating,
            d=firn,
            reg=1e-1,
            dt_yr=0.05,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"  linear-inverse FAILED: {exc}")
        print("  continuing with Eulerian + Lagrangian only.")
        linv = None

    if linv is not None:
        linv_mr = linv.melt_rate
        print(
            f"  H_ref={linv.attrs['H_ref_m']:.1f} m  "
            f"gamma_dimless={linv.attrs['gamma_dimless']:.3e}  "
            f"t_r={linv.attrs['tr_yr']:.1f} yr"
        )
        print(
            f"  melt_rate: median={float(linv_mr.median()):.2f}  "
            f"IQR=[{float(linv_mr.quantile(0.25)):.2f}, "
            f"{float(linv_mr.quantile(0.75)):.2f}] m ice/yr"
        )

    euler_melt = euler.melt_rate.where(floating)
    lagr_melt = lagr.melt_rate.where(floating)
    linv_melt = linv.melt_rate if linv is not None else None
    print(
        f"  floating-only Eulerian median={float(euler_melt.median()):.2f}  "
        f"Lagrangian median={float(lagr_melt.median()):.2f}"
        + (f"  Linear-inverse median={float(linv_melt.median()):.2f}" if linv is not None else "")
        + " m ice/yr"
    )

    out_nc = (
        config.RESULTS_DIR
        / f"venable_melt_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Saving results -> {out_nc}")
    ds_vars = {
        "melt_rate_eulerian": euler_melt,
        "melt_rate_lagrangian": lagr_melt,
        "dHdt": euler.dHdt.where(floating),
        "flux_div": euler.flux_div.where(floating),
        "H_f_mean": euler.H_f_mean.where(floating),
        "a_dot": a_dot,
        "floating_mask": floating,
        "lagrangian_count": lagr["count"].where(floating),
        "lagrangian_rmse": lagr["rmse"].where(floating),
    }
    ds_attrs = {
        "shelf": config.SHELF,
        "window_start": config.START_TIME,
        "window_end": config.END_TIME,
        "grid_res_m": float(config.RES),
        "velocity_source": vel_source,
        "smb_source": "RACMO2.4p1 smbgl (Zenodo 19255213), window-integrated",
    }
    if linv is not None:
        ds_vars["melt_rate_linear_inverse"] = linv_melt
        ds_attrs["linear_inverse_H_ref_m"] = linv.attrs["H_ref_m"]
        ds_attrs["linear_inverse_gamma_dimless"] = linv.attrs["gamma_dimless"]
        ds_attrs["linear_inverse_tr_yr"] = linv.attrs["tr_yr"]
    ds_out = xr.Dataset(ds_vars, attrs=ds_attrs)
    ds_out.to_netcdf(out_nc)

    plot_melt_comparison(euler, lagr, linv, config.FIGURES_DIR / "melt_comparison.png")
    print(f"  wrote {config.FIGURES_DIR / 'melt_comparison.png'}")


if __name__ == "__main__":
    main()
