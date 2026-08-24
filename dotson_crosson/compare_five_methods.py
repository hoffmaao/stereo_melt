"""5-method melt-rate comparison + Davison overlay on Dotson + Crosson IS2 stack.

Mirrors :mod:`nansen.compare_five_methods`: runs all currently-implemented
stationary inversions on the same tilt-corrected stack, regrids Davison
2023 onto the same 25 m grid, plots seven panels side by side, and writes
a single NetCDF.

Methods (left to right in the figure):
  1. Eulerian mass-conservation (Shean 2019 Eq. 10 — Eulerian)
  2. Lagrangian path integration  (Shean 2019 — Lagrangian)
  3. Closed-form Stubblefield linear inverse, FFT  (per-(t, k) Tikhonov)
  4. Closed-form Stubblefield linear inverse, DCT  (per-(t, k) Tikhonov)
  5a. dh/dt reformulation, FFT                    (per-pixel OLS slope, then
                                                   1-step Fourier inverse)
  5b. dh/dt reformulation, DCT                    (same, reflective extension)
  6.  Davison 2023 gridded basal melt              (RACMO-FAC corrected,
                                                   regridded to our grid)

Run::

    python -m dotson_crosson.compare_five_methods
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

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.backend import backend as _BACKEND
from stereo_melt.dynamics import lagrangian_frame_stack
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.io.davison import load_davison_gridded_in_shean
from stereo_melt.melt import (
    eulerian_melt_rate,
    lagrangian_melt_rate,
    linear_inverse_dhdt_lagrangian_melt_rate,
    linear_inverse_lagrangian_melt_rate,
)

from dotson_crosson import config
from dotson_crosson.run_melt import (
    _imshow_xr,
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
        f"abs_max={float(np.abs(da).max()):.1f}  m ice/yr"
    )


def main() -> None:
    config.ensure_output_dirs()
    print(f"stereo_melt backend: {_BACKEND}")

    print("\nLoading Dotson + Crosson IS2 tilt-corrected stack...")
    stack = load_stack()
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
    firn_finite = firn.values[np.isfinite(firn.values)]
    print(
        f"  firn (m): median={float(np.median(firn_finite)):.2f}  "
        f"IQR=[{float(np.percentile(firn_finite, 25)):.2f}, "
        f"{float(np.percentile(firn_finite, 75)):.2f}]  "
        f"finite frac={float(np.isfinite(firn.values).mean()):.3f}"
    )

    if "time" in vx.dims:
        vx_m = vx.mean("time", skipna=True)
    else:
        vx_m = vx
    if "time" in vy.dims:
        vy_m = vy.mean("time", skipna=True)
    else:
        vy_m = vy

    print("\nBuilding Lagrangian-frame stack (one-time)...")
    h_lag = lagrangian_frame_stack(stack_f, vx_m, vy_m, dt_yr=0.05)

    common_lin = dict(floating_mask=floating, dt_yr=0.05, h_lag_precomputed=h_lag, d=firn)

    print("\n=== 1/5 Eulerian mass-conservation ===")
    eul_ds = eulerian_melt_rate(stack_f, vx, vy, a_dot=a_dot, d=firn, robust_dh_dt=True)
    eul = eul_ds.melt_rate.where(floating)
    _summary("eulerian", eul)

    print("\n=== 2/5 Lagrangian path integration ===")
    lagr_ds = lagrangian_melt_rate(
        stack_f, vx, vy, a_dot=a_dot, d=firn,
        dt_yr=0.05, seed_stride=2, pairs="all",
        min_dt_yr=2.0 / 12.0,
    )
    lagr = lagr_ds.melt_rate.where(floating)
    _summary("lagrangian", lagr)

    print("\n=== 3/5 closed-form linear inverse (FFT, infill+pad) ===")
    closed_fft_ds = linear_inverse_lagrangian_melt_rate(
        stack_f, vx, vy,
        boundary_fix="infill+pad", transform="fft", reg=1e-1,
        **common_lin,
    )
    closed_fft = closed_fft_ds.melt_rate.where(floating)
    _summary("closed-FFT", closed_fft)

    print("\n=== 4/5 closed-form linear inverse (DCT, off) ===")
    closed_dct_ds = linear_inverse_lagrangian_melt_rate(
        stack_f, vx, vy,
        boundary_fix="off", transform="dct", reg=1e-1,
        **common_lin,
    )
    closed_dct = closed_dct_ds.melt_rate.where(floating)
    _summary("closed-DCT", closed_dct)

    print("\n=== 5/5 dh/dt reformulation (per-pixel OLS, FFT) ===")
    dhdt_fft_ds = linear_inverse_dhdt_lagrangian_melt_rate(
        stack_f, vx, vy, transform="fft", reg=10.0, min_n_obs=3,
        robust_dh_dt=True,
        **common_lin,
    )
    dhdt_fft = dhdt_fft_ds.melt_rate.where(floating)
    _summary("dh/dt-FFT", dhdt_fft)

    print("\n=== 5b dh/dt reformulation (per-pixel OLS, DCT) ===")
    dhdt_dct_ds = linear_inverse_dhdt_lagrangian_melt_rate(
        stack_f, vx, vy, transform="dct", reg=10.0, min_n_obs=3,
        robust_dh_dt=True,
        **common_lin,
    )
    dhdt_dct = dhdt_dct_ds.melt_rate.where(floating)
    _summary("dh/dt-DCT", dhdt_dct)

    print("\n=== Davison 2023 (gridded, regridded to our 25 m grid; Shean convention) ===")
    davison = load_davison_gridded_in_shean(eul).where(floating)
    _summary("davison-2023", davison)

    out_nc = (
        config.RESULTS_DIR
        / f"dotson_crosson_five_methods_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving -> {out_nc}")
    out_ds = xr.Dataset(
        {
            "melt_rate_eulerian":   eul.rename("melt_rate_eulerian"),
            "melt_rate_lagrangian": lagr.rename("melt_rate_lagrangian"),
            "melt_rate_closed_fft": closed_fft.rename("melt_rate_closed_fft"),
            "melt_rate_closed_dct": closed_dct.rename("melt_rate_closed_dct"),
            "melt_rate_dhdt_fft":   dhdt_fft.rename("melt_rate_dhdt_fft"),
            "melt_rate_dhdt_dct":   dhdt_dct.rename("melt_rate_dhdt_dct"),
            "melt_rate_davison":    davison.rename("melt_rate_davison"),
            "dh_dt_input":          dhdt_fft_ds.dh_dt.rename("dh_dt_input"),
            "n_obs_per_pixel":      dhdt_fft_ds.n_obs.rename("n_obs_per_pixel"),
            "floating_mask":        floating,
        },
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "velocity_source": vel_source,
            "H_ref_m": float(closed_fft_ds.attrs.get("H_ref_m", float("nan"))),
        },
    )
    out_ds.to_netcdf(out_nc)

    fig_path = config.FIGURES_DIR / "five_methods_comparison.png"
    print(f"Plotting -> {fig_path}")

    panels = [
        ("1. Eulerian mass-cons.\n(Shean Eq. 10)", eul, (-5, 5)),
        ("2. Lagrangian path-int.\n(Shean Eq. 7)", lagr, (-5, 5)),
        ("3. closed-form FFT\n(infill+pad, reg=0.1)", closed_fft, (-5, 5)),
        ("4. closed-form DCT\n(reflective, reg=0.1)", closed_dct, (-5, 5)),
        ("5a. dh/dt FFT\n(per-pixel OLS, reg=10)", dhdt_fft, (-5, 5)),
        ("5b. dh/dt DCT\n(per-pixel OLS, reg=10)", dhdt_dct, (-5, 5)),
        ("Davison 2023\n(gridded, RACMO-FAC corr.)", davison, (-5, 5)),
    ]

    fig, axes = plt.subplots(1, 7, figsize=(7 * 4, 6.5), constrained_layout=True)
    for col, (title, da, clim) in enumerate(panels):
        im = _imshow_xr(axes[col], da, cmap=melt_cmap(),
                        norm=melt_norm(vmax=max(abs(clim[0]), abs(clim[1]))))
        axes[col].set_title(
            f"{title}\nmedian={float(da.median()):+.2f}  "
            f"IQR=[{float(da.quantile(0.25)):+.2f}, {float(da.quantile(0.75)):+.2f}]  "
            f"abs_max={float(np.abs(da).max()):.0f}",
            fontsize=10,
        )
        add_melt_colorbar(fig, im, ax=axes[col], fraction=0.045)
        axes[col].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"Dotson + Crosson IS2 melt-rate comparison — 5 methods + Davison 2023 "
        f"({config.START_TIME} → {config.END_TIME})",
        fontsize=12,
    )
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("done.")


if __name__ == "__main__":
    main()
