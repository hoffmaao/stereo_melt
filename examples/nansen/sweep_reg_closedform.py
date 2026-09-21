"""Sweep Tikhonov regularization for the closed-form FFT/DCT linear inverse.

The closed-form path through ``inverse_stationary`` has its
own conditioning, so its plateau in ``reg`` is not the same number as
the CG ``tikhonov`` plateau. Production currently uses ``reg=1e-1``,
which is two orders of magnitude above the function default of
``1e-3`` and lands the closed-form medians at ~ -0.2 m/yr while the
Lagrangian path-integral and stationary CG-LSQ both park near
-3 to -4 m/yr. This script finds the ``reg`` value that closes that
gap.

Sweeps ``reg ∈ {1e-1, 1e-3, 1e-5, 1e-7}`` for both ``transform="fft"``
(``boundary_fix="infill+pad"``) and ``transform="dct"``
(``boundary_fix="off"``). All other knobs match the production
``run_melt.py`` call (``localize=True``, default ``prefilter_sigma_m
= H_ref/2``, ``dt_yr=0.05``).

Run:

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -m nansen.sweep_reg_closedform
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.dynamics import (
    lagrangian_frame_stack,
    linear_inverse_lagrangian_melt_rate,
)

from nansen import config
from nansen.run_melt import (
    load_floating_mask,
    load_stack,
    load_velocity_on_grid,
)


REG_GRID = (1e-1, 1e-3, 1e-5, 1e-7)
TRANSFORM_CONFIG = (
    ("fft", "infill+pad"),
    ("dct", "off"),
)
LOCALIZE_GRID = (True, False)


def _stats(da: xr.DataArray) -> dict:
    """Return floating-only summary stats for a melt-rate panel."""
    vals = da.values
    finite = np.isfinite(vals)
    if not finite.any():
        return {k: float("nan") for k in
                ("median", "q25", "q75", "p05", "p95", "abs_max", "l2norm", "n")}
    v = vals[finite]
    return {
        "median": float(np.median(v)),
        "q25": float(np.quantile(v, 0.25)),
        "q75": float(np.quantile(v, 0.75)),
        "p05": float(np.quantile(v, 0.05)),
        "p95": float(np.quantile(v, 0.95)),
        "abs_max": float(np.max(np.abs(v))),
        "l2norm": float(np.sqrt(np.mean(v * v))),
        "n": int(finite.sum()),
    }


def _print_stats(label: str, s: dict) -> None:
    print(
        f"  [{label:<32}] median={s['median']:+.3f}  "
        f"IQR=[{s['q25']:+.3f}, {s['q75']:+.3f}]  "
        f"p05/p95=[{s['p05']:+.3f}, {s['p95']:+.3f}]  "
        f"abs_max={s['abs_max']:.1f}  "
        f"||m||_rms={s['l2norm']:.3f}  n={s['n']}"
    )


def plot_sweep(
    panels: dict[tuple[str, float, bool], xr.DataArray],
    eulerian: xr.DataArray | None,
    lagrangian: xr.DataArray | None,
    out_path: Path,
) -> None:
    """Rows = (eulerian, lagrangian, then one per (transform, localize)); cols = reg values."""
    transform_loc_pairs = [
        (transform, boundary, loc)
        for transform, boundary in TRANSFORM_CONFIG
        for loc in LOCALIZE_GRID
    ]
    nrows = 2 + len(transform_loc_pairs)
    ncols = len(REG_GRID)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.4 * ncols, 4.0 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    clim = (-5.0, 5.0)

    def _imshow(ax, da, title):
        if da is None:
            ax.axis("off")
            return None
        im = ax.imshow(
            da.values,
            extent=[
                float(da["x"].min()), float(da["x"].max()),
                float(da["y"].min()), float(da["y"].max()),
            ],
            origin="upper", cmap="RdBu_r", vmin=clim[0], vmax=clim[1],
            aspect="equal",
        )
        s = _stats(da)
        ax.set_title(
            f"{title}\nmedian={s['median']:+.2f}  "
            f"IQR=[{s['q25']:+.2f}, {s['q75']:+.2f}]  "
            f"|max|={s['abs_max']:.0f}",
            fontsize=9,
        )
        return im

    # Row 0: Eulerian
    for c in range(ncols):
        if c == 0:
            im = _imshow(axes[0, c], eulerian, "Eulerian (production)")
            if im is not None:
                fig.colorbar(im, ax=axes[0, c], fraction=0.045)
        else:
            axes[0, c].axis("off")

    # Row 1: Lagrangian
    for c in range(ncols):
        if c == 0:
            im = _imshow(axes[1, c], lagrangian, "Lagrangian path-int (production)")
            if im is not None:
                fig.colorbar(im, ax=axes[1, c], fraction=0.045)
        else:
            axes[1, c].axis("off")

    for ri, (transform, boundary, loc) in enumerate(transform_loc_pairs):
        r = 2 + ri
        loc_str = "loc=on" if loc else "loc=off"
        for c, reg in enumerate(REG_GRID):
            da = panels.get((transform, reg, loc))
            title = f"{transform.upper()} ({boundary}, {loc_str})\nreg={reg:.0e}"
            im = _imshow(axes[r, c], da, title)
            if im is not None:
                fig.colorbar(im, ax=axes[r, c], fraction=0.045)

    fig.suptitle(
        f"Closed-form linear inverse — Tikhonov × localize sweep — Nansen "
        f"{config.START_TIME} to {config.END_TIME}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config.ensure_output_dirs()

    print("Loading stack + floating mask + velocity...")
    stack = load_stack()
    floating = load_floating_mask(stack)
    stack_floating = stack.where(floating)
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  velocity source: {vel_source}")
    print(f"  stack: time={stack.sizes['time']}  y={stack.sizes['y']}  x={stack.sizes['x']}")

    print("Building Lagrangian-frame stack (shared across reg values)...")
    h_lag = lagrangian_frame_stack(
        stack_floating,
        vx.where(floating),
        vy.where(floating),
        dt_yr=0.05,
    )

    # Production references for the comparison panels.
    ref_nc = (
        config.RESULTS_DIR
        / f"nansen_melt_{config.START_TIME}_{config.END_TIME}.nc"
    )
    ref_eul = ref_lag = None
    if ref_nc.exists():
        print(f"Loading production reference: {ref_nc.name}")
        ref_ds = xr.open_dataset(ref_nc)
        if "melt_rate_eulerian" in ref_ds:
            ref_eul = ref_ds["melt_rate_eulerian"].where(floating)
        if "melt_rate_lagrangian" in ref_ds:
            ref_lag = ref_ds["melt_rate_lagrangian"].where(floating)
        ref_ds.close()
        if ref_eul is not None:
            _print_stats("Eulerian (production)", _stats(ref_eul))
        if ref_lag is not None:
            _print_stats("Lagrangian (production)", _stats(ref_lag))
    else:
        print(f"  (no production NC at {ref_nc.name}; skipping reference panels)")

    panels: dict[tuple[str, float, bool], xr.DataArray] = {}
    panel_attrs: dict[tuple[str, float, bool], dict] = {}
    panel_stats: list[dict] = []

    print("\n=== Sweep ===")
    for transform, boundary in TRANSFORM_CONFIG:
        for loc in LOCALIZE_GRID:
            for reg in REG_GRID:
                loc_str = "on" if loc else "off"
                label = f"{transform.upper()} reg={reg:.0e} loc={loc_str}"
                print(f"\n--- {label}  (boundary_fix={boundary}) ---")
                ds = linear_inverse_lagrangian_melt_rate(
                    stack_floating, vx, vy,
                    floating_mask=floating,
                    reg=reg,
                    dt_yr=0.05,
                    h_lag_precomputed=h_lag,
                    transform=transform,
                    boundary_fix=boundary,
                    localize=loc,
                )
                m_floating = ds.melt_rate.where(floating)
                panels[(transform, reg, loc)] = m_floating
                panel_attrs[(transform, reg, loc)] = {
                    "transform": transform,
                    "boundary_fix": boundary,
                    "reg": float(reg),
                    "localize": bool(loc),
                    "H_ref_m": float(ds.attrs["H_ref_m"]),
                    "gamma_dimless": float(ds.attrs["gamma_dimless"]),
                    "tr_yr": float(ds.attrs["tr_yr"]),
                }
                s = _stats(m_floating)
                panel_stats.append({
                    "transform": transform, "reg": reg, "localize": loc, **s,
                })
                _print_stats(label, s)

    print("\n=== Sweep summary ===")
    print(
        f"{'transform':<8} {'loc':>5} {'reg':>10} "
        f"{'median':>9} {'IQR_lo':>9} {'IQR_hi':>9} "
        f"{'p05':>8} {'p95':>8} {'|max|':>10} {'rms':>10}"
    )
    for row in panel_stats:
        loc_str = "on" if row["localize"] else "off"
        print(
            f"{row['transform']:<8} {loc_str:>5} {row['reg']:>10.0e} "
            f"{row['median']:+9.3f} {row['q25']:+9.3f} {row['q75']:+9.3f} "
            f"{row['p05']:+8.2f} {row['p95']:+8.2f} "
            f"{row['abs_max']:10.1f} {row['l2norm']:10.3f}"
        )

    out_nc = (
        config.RESULTS_DIR
        / f"nansen_sweep_reg_closedform_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving sweep NetCDF -> {out_nc.name}")
    data_vars = {}
    for (transform, reg, loc), m in panels.items():
        loc_str = "on" if loc else "off"
        key = f"melt_{transform}_loc{loc_str}_reg_{reg:.0e}".replace("-", "m")
        data_vars[key] = m
    if ref_eul is not None:
        data_vars["melt_eulerian"] = ref_eul
    if ref_lag is not None:
        data_vars["melt_lagrangian"] = ref_lag
    data_vars["floating_mask"] = floating
    out_ds = xr.Dataset(
        data_vars,
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "reg_grid": list(REG_GRID),
            "transforms": [t for t, _ in TRANSFORM_CONFIG],
            "H_ref_m": panel_attrs[(TRANSFORM_CONFIG[0][0], REG_GRID[0], LOCALIZE_GRID[0])]["H_ref_m"],
            "gamma_dimless": panel_attrs[(TRANSFORM_CONFIG[0][0], REG_GRID[0], LOCALIZE_GRID[0])]["gamma_dimless"],
            "tr_yr": panel_attrs[(TRANSFORM_CONFIG[0][0], REG_GRID[0], LOCALIZE_GRID[0])]["tr_yr"],
            "velocity_source": vel_source,
        },
    )
    out_ds.to_netcdf(out_nc)

    out_png = config.FIGURES_DIR / "closedform_reg_sweep.png"
    print(f"Saving sweep figure   -> {out_png.name}")
    plot_sweep(panels, ref_eul, ref_lag, out_png)
    print("\nDone.")


if __name__ == "__main__":
    main()
