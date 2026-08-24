"""Run the Beardmore Lagrangian melt-rate pipeline on the built stack.

Stage 4+ of the Beardmore pipeline. Loads the already-built DEM stack,
fetches matching velocity and SMB, and runs
:func:`stereo_melt.melt.lagrangian_melt_rate` (and Eulerian as a
side-by-side comparison). Writes a NetCDF with every variable of the
returned Dataset plus QC plots to ``beardmore/figures/``.

Run:

    python -m beardmore.run_melt

Assumes :mod:`beardmore.build_stack` has already produced
``processed/beardmore_stack_<start>_<end>.nc``.
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

from stereo_melt.dynamics import lagrangian_frame_stack
from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.io.davison import load_davison_gridded_in_shean
from stereo_melt.io.smb import smb_over_window
from stereo_melt.melt import (
    eulerian_melt_rate,
    lagrangian_melt_rate,
    linear_inverse_dhdt_lagrangian_melt_rate,
)
from stereo_melt.stack import load_basin_stack

from beardmore import config

SECONDS_PER_YEAR = 86400.0 * 365.25


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------


def load_stack(stack_prefix: str = "beardmore_stack") -> xr.DataArray:
    stack, _ = load_basin_stack(
        config.PROCESSED_DIR,
        stack_prefix,
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

    Velocity source is selected by the ``BEARDMORE_VELOCITY`` env var:
    ``measures`` (default, NSIDC-0754 phase map at 450 m), ``nsidc-0525``
    (Scheuchl 2012 Central Antarctica 2009 mosaic at 900 m), or
    ``its_live`` (annual mosaic at 120 m). Remaining NaN gaps are filled
    by nearest-valid neighbor.
    """
    requested = os.environ.get("BEARDMORE_VELOCITY", "measures").strip().lower()
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
        origin="upper",
        cmap=cmap,
        aspect="equal",
        **kw,
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

    fig.suptitle("Beardmore melt-rate inputs", fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _panel_stats(da: xr.DataArray) -> str:
    v = da.values
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "no data"
    return (
        f"med={float(np.median(v)):+.2f}  "
        f"IQR=[{float(np.percentile(v, 25)):+.2f}, {float(np.percentile(v, 75)):+.2f}]"
    )


def plot_melt_comparison(
    panels: list[tuple[str, xr.DataArray]],
    out_path: Path,
    clim: tuple[float, float] = (-5.0, 5.0),
) -> None:
    """QC plot: one row of melt-rate panels on a shared ±5 m/yr RdBu scale.

    Standardized panel set (Eulerian, Lagrangian path-int, dh/dt FFT,
    dh/dt DCT, D-masked CG DCT, Davison) — closed-form per-(t,k)
    Tikhonov variants are intentionally excluded; they filter most of
    the real signal and are not regularized in a coverage-aware way.
    See ``project_linear_inverse_coverage_diagnosis``.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 4.4, 5.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    # LADDIE symmetric-log scale: black at zero, log decades outward. A linear
    # shared stretch buries the sub-m/yr structure that dominates these shelves.
    mcmap = melt_cmap()
    mnorm = melt_norm(vmax=max(abs(clim[0]), abs(clim[1])))
    for ax, (title, da) in zip(axes, panels):
        im = _imshow_xr(ax, da, cmap=mcmap, norm=mnorm)
        ax.set_title(f"{title}\n{_panel_stats(da)}", fontsize=10)
        add_melt_colorbar(fig, im, ax=ax, fraction=0.045)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"Beardmore melt rate — {config.START_TIME} to {config.END_TIME}",
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
    # See nansen.run_melt for rationale. robust_dh_dt=True enables Tukey-
    # biweight IRLS so per-strip residual stripes don't get amplified by
    # the ×9.42 hydrostatic gain into spurious dh/dt.
    euler = eulerian_melt_rate(stack, vx, vy, a_dot=a_dot, d=firn, robust_dh_dt=True)
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

    euler_melt = euler.melt_rate.where(floating)
    lagr_melt = lagr.melt_rate.where(floating)

    print("Building Lagrangian-frame stack (shared by dh/dt + masked-CG inverses)...")
    h_lag = lagrangian_frame_stack(stack, vx, vy, dt_yr=0.05)

    print("Running dh/dt-reformulation linear inverse (per-pixel OLS, transform=FFT)...")
    dhdt_fft_melt = None; dhdt_dct_melt = None
    H_ref_m = None
    try:
        dhdt_fft = linear_inverse_dhdt_lagrangian_melt_rate(
            stack, vx, vy,
            floating_mask=floating, d=firn, dt_yr=0.05, h_lag_precomputed=h_lag,
            transform="fft", reg=10.0, min_n_obs=3, robust_dh_dt=True,
        )
        dhdt_fft_melt = dhdt_fft.melt_rate.where(floating)
        H_ref_m = float(dhdt_fft.attrs["H_ref_m"])
        print(
            f"  H_ref={H_ref_m:.1f} m  "
            f"gamma_dimless={dhdt_fft.attrs['gamma_dimless']:.3e}  "
            f"t_r={dhdt_fft.attrs['tr_yr']:.1f} yr"
        )
        print(
            f"  melt_rate: median={float(dhdt_fft_melt.median()):.2f}  "
            f"IQR=[{float(dhdt_fft_melt.quantile(0.25)):.2f}, "
            f"{float(dhdt_fft_melt.quantile(0.75)):.2f}] m ice/yr"
        )
    except (ValueError, RuntimeError) as e:
        print(f"  ⚠ linear-inverse FFT failed: {e}; skipping FFT/DCT/CG stages")

    if H_ref_m is not None:
        print("Running dh/dt-reformulation linear inverse (per-pixel OLS, transform=DCT)...")
        try:
            dhdt_dct = linear_inverse_dhdt_lagrangian_melt_rate(
                stack, vx, vy,
                floating_mask=floating, d=firn, dt_yr=0.05, h_lag_precomputed=h_lag,
                transform="dct", reg=10.0, min_n_obs=3, robust_dh_dt=True,
            )
            dhdt_dct_melt = dhdt_dct.melt_rate.where(floating)
            print(
                f"  melt_rate: median={float(dhdt_dct_melt.median()):.2f}  "
                f"IQR=[{float(dhdt_dct_melt.quantile(0.25)):.2f}, "
                f"{float(dhdt_dct_melt.quantile(0.75)):.2f}] m ice/yr"
            )
        except (ValueError, RuntimeError) as e:
            print(f"  ⚠ linear-inverse DCT failed: {e}; skipping")

    print("Loading Davison 2023 gridded melt rate (regridded to stack, Shean convention)...")
    davison = load_davison_gridded_in_shean(euler.melt_rate).where(floating)
    print(
        f"  davison: median={float(davison.median()):.2f}  "
        f"IQR=[{float(davison.quantile(0.25)):.2f}, "
        f"{float(davison.quantile(0.75)):.2f}] m ice/yr"
    )

    def _median_str(da):
        return f"{float(da.median()):.2f}" if da is not None else "—"
    print(
        f"  floating-only medians: "
        f"Eulerian={_median_str(euler_melt)}  "
        f"Lagrangian={_median_str(lagr_melt)}  "
        f"dh/dt-FFT={_median_str(dhdt_fft_melt)}  "
        f"dh/dt-DCT={_median_str(dhdt_dct_melt)}  "
        f"Davison={_median_str(davison)} m ice/yr"
    )

    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_melt_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Saving results -> {out_nc}")
    ds_vars = {
        "melt_rate_eulerian":      euler_melt,
        "melt_rate_lagrangian":    lagr_melt,
        "melt_rate_dhdt_fft":      dhdt_fft_melt,
        "melt_rate_dhdt_dct":      dhdt_dct_melt,
        "melt_rate_davison":       davison,
        "dHdt":                    euler.dHdt.where(floating),
        "flux_div":                euler.flux_div.where(floating),
        "H_f_mean":                euler.H_f_mean.where(floating),
        "a_dot":                   a_dot,
        "floating_mask":           floating,
        "lagrangian_count":        lagr["count"].where(floating),
        "lagrangian_rmse":         lagr["rmse"].where(floating),
    }
    ds_attrs = {
        "shelf": config.SHELF,
        "window_start": config.START_TIME,
        "window_end": config.END_TIME,
        "grid_res_m": float(config.RES),
        "velocity_source": vel_source,
        "smb_source": "RACMO2.4p1 smbgl (Zenodo 19255213), window-integrated",
        "linear_inverse_H_ref_m": H_ref_m,
        "linear_inverse_gamma_dimless": float(dhdt_fft.attrs["gamma_dimless"]),
        "linear_inverse_tr_yr": float(dhdt_fft.attrs["tr_yr"]),
        "dhdt_reg": 10.0,
    }
    ds_out = xr.Dataset(ds_vars, attrs=ds_attrs)
    ds_out.to_netcdf(out_nc)

    panels = [
        ("1. Eulerian\n(Shean Eq. 10)",                       euler_melt),
        ("2. Lagrangian path-int\n(Shean Eq. 7)",             lagr_melt),
        ("3. dh/dt FFT\n(per-pixel OLS, λ=10)",               dhdt_fft_melt),
        ("4. dh/dt DCT\n(per-pixel OLS, λ=10)",               dhdt_dct_melt),
        ("Davison 2023\n(gridded, RACMO-FAC)",                davison),
    ]
    plot_melt_comparison(panels, config.FIGURES_DIR / "melt_comparison.png")
    print(f"  wrote {config.FIGURES_DIR / 'melt_comparison.png'}")


if __name__ == "__main__":
    main()
