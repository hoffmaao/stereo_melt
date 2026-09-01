"""Nansen correction statistics — Shean 2019 Fig 4 analog.

Loads the raw (un-tilt-corrected) Nansen stack, recomputes each
correction component from the same lib helpers ``tilt_fit`` used, and
plots a 6-panel distribution figure:

    (a) per-epoch tide median (floating-ice mask, m)
    (b) per-epoch IBE scalar (m)
    (c) MDT spatial distribution over the AOI (m)
    (d) geoid undulation distribution over the AOI (m)
    (e) per-epoch αz (m)  -- from tilt_params
    (f) per-epoch αx, αy (m/m)

Output: ``nansen/figures/corrections_fig4.png``.

Run:

    python -m nansen.plot_corrections_fig4
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

from stereo_melt.corrections.geoid import apply_geoid_correction_xr
from stereo_melt.corrections.post_coreg import (
    _build_feathered_floating_mask,
    _load_pressure_interpolator,
    _scalar_ibe_at,
    _tide_on_stack_grid,
)
from stereo_melt.pipeline import _grounded_mask_for_stack, _interp_mdt_to_stack_grid
from stereo_melt.stack import load_basin_stack

from nansen import config


def _summarize(values, label):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return f"{label}: (no finite values)"
    return (
        f"{label}: median={np.median(finite):+.3f}  "
        f"IQR=[{np.percentile(finite, 25):+.3f}, {np.percentile(finite, 75):+.3f}]  "
        f"range=[{finite.min():+.3f}, {finite.max():+.3f}]  "
        f"N={finite.size}"
    )


def _compute_corrections(stack: xr.DataArray):
    """Return a dict of correction fields/series matched to ``stack``."""
    x_axis = stack["x"].values.astype(np.float64)
    y_axis = stack["y"].values.astype(np.float64)
    times = stack["time"].values
    n_t = len(times)

    print("Building feathered floating-ice mask...")
    floating = _build_feathered_floating_mask(
        x_axis, y_axis, str(config.BEDMACHINE_NC), 3000.0
    )
    floating_bool = floating > 0

    print("Loading IBE pressure interpolator...")
    ibe_interp = _load_pressure_interpolator(config.ERA5_CACHE_NC, "surface_pressure")

    tide_series = np.full(n_t, np.nan, dtype=np.float64)
    ibe_series = np.full(n_t, np.nan, dtype=np.float64)
    print(f"Computing tide + IBE for {n_t} epochs...")
    for k, t in enumerate(times):
        center_time = np.datetime64(t)
        tide_field = _tide_on_stack_grid(
            x_axis, y_axis, center_time,
            tide_model=config.TIDE_MODEL,
            tide_model_dir=config.TIDE_MODEL_DIR,
            grid_step_m=2000.0,
        )
        if floating_bool.sum():
            tide_series[k] = float(np.nanmedian(tide_field[floating_bool]))
        ibe_series[k] = float(_scalar_ibe_at(ibe_interp, center_time))
        print(
            f"  epoch {k+1}/{n_t}  {str(center_time)[:10]}  "
            f"tide(median over floating)={tide_series[k]:+.3f}  "
            f"IBE={ibe_series[k]:+.4f} m"
        )

    print("Interpolating DTU22 MDT onto stack grid (floating-ice only)...")
    mdt_grid = _interp_mdt_to_stack_grid(stack, config.DTU22_MDT_XYZ)
    grounded = _grounded_mask_for_stack(stack, str(config.BEDMACHINE_NC), 2)
    mdt_grid = np.where(grounded, np.nan, mdt_grid)

    print("Loading geoid undulation from BedMachine (via apply_geoid_correction_xr)...")
    # apply_geoid_correction_xr does dem - N. Pass a zero DEM so the
    # output is -N regridded to our stack grid; negate to get +N.
    zero_slab = xr.zeros_like(stack.isel(time=0)).rio.write_crs("EPSG:3031", inplace=False)
    geoid_grid = -apply_geoid_correction_xr(zero_slab, str(config.BEDMACHINE_NC)).values

    print("Loading tilt parameters...")
    tilt_path = (
        config.PROCESSED_DIR
        / f"nansen_tilt_params_{config.START_TIME}_{config.END_TIME}.nc"
    )
    tilt = xr.open_dataset(tilt_path)
    print(f"  loaded {tilt_path.name}: {tilt.sizes.get('time', 0)} epochs")

    return {
        "tide": tide_series,
        "ibe": ibe_series,
        "mdt": np.asarray(mdt_grid),
        "geoid": np.asarray(geoid_grid),
        "tilt_dz": np.asarray(tilt["tilt_dz"].values),
        "tilt_dx": np.asarray(tilt["tilt_dx"].values),
        "tilt_dy": np.asarray(tilt["tilt_dy"].values),
        "times": np.asarray(times),
    }


def _hist_panel(ax, values, *, title, xlabel, color, bins=30):
    finite = values[np.isfinite(values)]
    if not finite.size:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.hist(finite, bins=bins, color=color, edgecolor="k", alpha=0.85)
    med = float(np.median(finite))
    iqr = (float(np.percentile(finite, 25)), float(np.percentile(finite, 75)))
    ax.axvline(med, color="k", lw=1.4, linestyle="--")
    ax.set_title(
        f"{title}\nmed={med:+.3f}  IQR=[{iqr[0]:+.3f}, {iqr[1]:+.3f}]  N={finite.size}",
        fontsize=10,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def main() -> None:
    config.ensure_output_dirs()

    print("Loading raw Nansen stack (bad-epochs filtered)...")
    stack, src_path = load_basin_stack(
        config.PROCESSED_DIR,
        "nansen_stack",
        config.START_TIME,
        config.END_TIME,
        prefer_tilt_corrected=False,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    print(f"  loaded {src_path.name}: {stack.sizes['time']} epochs")

    cr = _compute_corrections(stack)

    print()
    print(_summarize(cr["tide"], "tide        (per-epoch median, m)"))
    print(_summarize(cr["ibe"], "IBE         (per-epoch scalar,  m)"))
    print(_summarize(cr["mdt"], "MDT         (floating pixels,   m)"))
    print(_summarize(cr["geoid"], "geoid       (all pixels,        m)"))
    print(_summarize(cr["tilt_dz"], "tilt αz     (per-epoch,         m)"))
    print(_summarize(cr["tilt_dx"], "tilt αx     (per-epoch,       m/m)"))
    print(_summarize(cr["tilt_dy"], "tilt αy     (per-epoch,       m/m)"))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    _hist_panel(
        axes[0, 0], cr["tide"],
        title="(a) tide (per-epoch median, floating)",
        xlabel="tide [m]", color="#3C8DAD",
    )
    _hist_panel(
        axes[0, 1], cr["ibe"],
        title="(b) IBE (per-epoch scalar)",
        xlabel="IBE [m]", color="#76B947",
    )
    _hist_panel(
        axes[0, 2], cr["mdt"],
        title="(c) MDT (DTU22, floating pixels)",
        xlabel="MDT [m]", color="#F0A04B", bins=60,
    )
    _hist_panel(
        axes[1, 0], cr["geoid"],
        title="(d) geoid undulation (BedMachine)",
        xlabel="geoid [m]", color="#9D4EDD", bins=60,
    )
    _hist_panel(
        axes[1, 1], cr["tilt_dz"],
        title=r"(e) tilt $\alpha_z$ (per-epoch)",
        xlabel=r"$\alpha_z$ [m]", color="#E63946",
    )
    ax = axes[1, 2]
    finite_dx = cr["tilt_dx"][np.isfinite(cr["tilt_dx"])]
    finite_dy = cr["tilt_dy"][np.isfinite(cr["tilt_dy"])]
    ax.hist(finite_dx, bins=20, color="#264653", edgecolor="k", alpha=0.7, label=r"$\alpha_x$")
    ax.hist(finite_dy, bins=20, color="#E76F51", edgecolor="k", alpha=0.7, label=r"$\alpha_y$")
    ax.axvline(0.0, color="k", lw=1.0, linestyle=":")
    ax.set_title(
        f"(f) tilt slopes (per-epoch)\n"
        f"αx med={np.median(finite_dx):+.2e}  "
        f"αy med={np.median(finite_dy):+.2e}",
        fontsize=10,
    )
    ax.set_xlabel(r"$\alpha_{x,y}$ [m/m]")
    ax.set_ylabel("count")
    ax.legend(loc="upper right")

    fig.suptitle(
        f"Nansen — correction distributions (Shean 2019 Fig 4 analog)\n"
        f"{config.START_TIME} → {config.END_TIME}, "
        f"{stack.sizes['time']} epochs",
        fontsize=12,
    )

    out_path = config.FIGURES_DIR / "corrections_fig4.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out_path}")

    # Also dump the per-epoch correction series as an .nc for downstream
    # plotting / table-making.
    out_nc = config.FIGURES_DIR / "corrections_fig4_data.nc"
    ds = xr.Dataset(
        {
            "tide_median_floating": ("time", cr["tide"]),
            "ibe_scalar": ("time", cr["ibe"]),
            "tilt_dz": ("time", cr["tilt_dz"]),
            "tilt_dx": ("time", cr["tilt_dx"]),
            "tilt_dy": ("time", cr["tilt_dy"]),
        },
        coords={"time": cr["times"]},
    )
    ds.to_netcdf(out_nc)
    print(f"Wrote {out_nc}")


if __name__ == "__main__":
    main()
