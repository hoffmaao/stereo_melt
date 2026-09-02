"""Stationary CG-LSQ pseudospectral inverse driver for PineIsland.

Calls :func:`stereo_melt.dynamics.stationary_pseudospectral_lagrangian_inverse`
on the existing PIG stack. Sweeps Tikhonov regularization so the
operator point at the right magnitude can be picked from the recovered
medians and the residual norm.

Run:

    python -m pig.run_stationary
"""

from __future__ import annotations

import os
import sys

# PROJ_DATA fix.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.dynamics import stationary_pseudospectral_lagrangian_inverse
from stereo_melt.freeboard import freeboard_to_thickness
from stereo_melt.constants import rhoi, rhow
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.stack import load_basin_stack

from pig import config
from pig.run_melt import load_floating_mask, load_velocity_on_grid


def _load_any_stack() -> tuple[xr.DataArray, Path]:
    return load_basin_stack(
        config.PROCESSED_DIR,
        "pig_stack",
        config.START_TIME,
        config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )


def _imshow_xr(ax, da, *, cmap, vmin=None, vmax=None, norm=None):
    # `norm` and `vmin`/`vmax` are mutually exclusive in matplotlib; melt-rate
    # panels pass the symmetric-log `melt_norm`, everything else stays linear.
    kw = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    return ax.imshow(
        da.values,
        extent=[
            float(da["x"].min()), float(da["x"].max()),
            float(da["y"].min()), float(da["y"].max()),
        ],
        origin="upper", cmap=cmap, aspect="equal", **kw,
    )


def plot_tikhonov_sweep(results: list[tuple[float, xr.Dataset]], out_path: Path) -> None:
    """One panel per Tikhonov value, common colormap."""
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    clim = (-5.0, 5.0)
    for i, (tik, ds) in enumerate(results):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        m = ds.melt_rate
        im = _imshow_xr(ax, m, cmap=melt_cmap(),
                        norm=melt_norm(vmax=max(abs(clim[0]), abs(clim[1]))))
        med = float(m.median())
        iqr = (float(m.quantile(0.25)), float(m.quantile(0.75)))
        ax.set_title(
            f"tikhonov={tik:.0e}  median={med:+.2f}\n"
            f"IQR=[{iqr[0]:+.2f}, {iqr[1]:+.2f}]  "
            f"cg_iter={ds.attrs['cg_iter']} conv={ds.attrs['cg_converged']}",
            fontsize=10,
        )
        add_melt_colorbar(fig, im, ax=ax, fraction=0.045)
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")
    fig.suptitle(
        f"Stationary CG-LSQ — Pine Island "
        f"{config.START_TIME} to {config.END_TIME}  "
        f"({results[0][1].attrs['H_ref_m']:.0f} m H_ref)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config.ensure_output_dirs()

    print("Loading stack...")
    stack, stack_path = _load_any_stack()
    print(f"  loaded {stack_path.name}")
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    print("Loading velocity (for advection)...")
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


    # Derive H_ref from the RAW (pre-tilt-correction) stack. The tilt fit
    # collapses many low-coverage pixels' absolute elevations toward 0
    # (LSQ can't separate z0_p from αz when a pixel is observed at one
    # epoch), so the corrected stack's per-pixel time-mean is no longer
    # a reliable thickness estimate even though the time-varying anomaly
    # is still useful. Anchor H_ref against the raw absolute elevations.
    if "tilt_corrected" in stack_path.name:
        raw_path = (
            config.PROCESSED_DIR
            / f"pig_stack_{config.START_TIME}_{config.END_TIME}.nc"
        )
        if not raw_path.exists():
            candidates = [
                p for p in sorted(config.PROCESSED_DIR.glob("pig_stack_*.nc"))
                if "tilt_corrected" not in p.name
            ]
            if not candidates:
                raise SystemExit(
                    "Need the raw stack to derive H_ref but none is on disk."
                )
            raw_path = candidates[-1]
        # load_dataset (not open_dataset) reads everything into memory and
        # closes the file handle immediately — important because we hold
        # H_ref_val for the lifetime of main() and other processes
        # (build_stack rebuild, tilt_fit) need to write back to this path.
        raw_ds = xr.load_dataset(raw_path)
        try:
            raw_da = raw_ds[[v for v in raw_ds.data_vars if v != "spatial_ref"][0]]
            raw_floating = load_floating_mask(raw_da)
            raw_h_mean = raw_da.where(raw_floating).mean("time", skipna=True)
            raw_H_f = freeboard_to_thickness(raw_h_mean, d=0.0, rho_w=rhow, rho_i=rhoi)
            H_ref_val = float(raw_H_f.where(raw_floating).mean(skipna=True))
        finally:
            raw_ds.close()
        print(
            f"  H_ref={H_ref_val:.1f} m (derived from raw {raw_path.name})"
        )
    else:
        H_ref_val = None  # let the inverse derive from current stack
        print("  H_ref will be derived from the loaded stack inside the solver")

    print("Applying floating-ice mask...")
    floating = load_floating_mask(stack)
    stack = stack.where(floating)

    # The defaults of cg_invert_stationary scale Tikhonov against the
    # spectral diagonal of G^T G (centered cumulative kernel). For the
    # 4-year PIG stack, the right operating scale is well below
    # 1e-3; sweep wide and let the figure tell us.
    tikhonov_grid = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
    print(f"\nSweeping Tikhonov over {tikhonov_grid}...")

    results: list[tuple[float, xr.Dataset]] = []
    for tik in tikhonov_grid:
        print(f"\n--- tikhonov={tik:.0e} ---")
        ds = stationary_pseudospectral_lagrangian_inverse(
            stack, vx, vy,
            floating_mask=floating, d=firn,
            H_ref=H_ref_val,
            tikhonov=tik,
            max_iter=120,
            cg_tol=1e-7,
            dt_yr=0.05,
            localize=True,
            verbose=True,
        )
        m = ds.melt_rate
        print(
            f"  result: median={float(m.median()):+.3f}  "
            f"IQR=[{float(m.quantile(0.25)):+.3f}, {float(m.quantile(0.75)):+.3f}]  "
            f"H_ref={ds.attrs['H_ref_m']:.1f}  γ={ds.attrs['gamma_dimless']:.3e}  "
            f"converged={ds.attrs['cg_converged']}"
        )
        results.append((tik, ds))

    out_nc = config.RESULTS_DIR / f"pig_stationary_{config.START_TIME}_{config.END_TIME}.nc"
    print(f"\nSaving Tikhonov-grid output -> {out_nc}")
    out_ds = xr.Dataset(
        {f"melt_rate_tik_{i}": ds.melt_rate for i, (_, ds) in enumerate(results)},
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "grid_res_m": float(config.RES),
            "velocity_source": vel_source,
            "stack_path": str(stack_path),
            "tikhonov_values": [t for t, _ in results],
            "H_ref_m": results[0][1].attrs["H_ref_m"],
            "gamma_dimless": results[0][1].attrs["gamma_dimless"],
        },
    )
    out_ds.to_netcdf(out_nc)

    out_png = config.FIGURES_DIR / "stationary_tikhonov_sweep.png"
    plot_tikhonov_sweep(results, out_png)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
