"""Per-epoch residual-tilt correction for the Venable DEM stack.

Stage 3.5 of the Venable pipeline. Loads the raw stack produced by
:mod:`venable.build_stack`, builds a Shean ``ndinterp.py``-style
static-control mask (BedMachine rock + grounded ice, intersected with
the per-pixel temporal-statistics filter from
:func:`stereo_melt.coregister.tilt.build_static_control_mask`), runs
:func:`stereo_melt.coregister.tilt.fit_tilt_stack` with library
defaults (Shean 2019 Venable priors), and writes a tilt-corrected stack
as a new NetCDF that downstream melt-rate solvers consume directly.

The Venable stack covers a wider domain than ``VENABLE_AOI_SHP``
(see ``VENABLE_STACK_AOI_SHP``: ~30 km extension into the Queen
Alexandra Range, ~47% grounded + ~7% rock outcrop) for exactly this
reason: the joint LSQ needs a healthy population of static-control
pixels to anchor the per-epoch tilts.

Saving the corrected stack on disk lets us iterate on the melt-rate
inverse (sweep Tikhonov, swap regularization terms, etc.) without
repeating the tilt LSQ every time.

Run:

    python -m venable.tilt_fit
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
import pandas as pd
import xarray as xr

from stereo_melt.coregister.control_source import build_per_epoch_ez
from stereo_melt.coregister.tilt import (
    build_static_area_polygon_mask,
    build_static_control_mask,
    fit_tilt_stack,
)
from stereo_melt.corrections.post_coreg import apply_tide_ibe_to_stack
from stereo_melt.pipeline import apply_geoid_to_stack, apply_mdt_to_stack
from stereo_melt.stack import load_basin_stack, save_stack

from venable import config


# --------------------------------------------------------------------------
# Stack loaders
# --------------------------------------------------------------------------


def _load_raw_stack(stack_prefix: str = "venable_stack") -> tuple[xr.DataArray, Path]:
    return load_basin_stack(
        config.PROCESSED_DIR,
        stack_prefix,
        config.START_TIME,
        config.END_TIME,
        prefer_tilt_corrected=False,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )


# --------------------------------------------------------------------------
# QC plotting
# --------------------------------------------------------------------------


def plot_tilt_params(params: xr.Dataset, control: xr.DataArray, out_path: Path) -> None:
    """QC: per-epoch tilt time series + control mask.

    Top: time series of (αx, αy, αz). Bottom: control mask coverage map
    + per-pixel dhdt recovered from the joint LSQ.
    """
    times = pd.to_datetime(params["time"].values)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(times, params["tilt_dx"].values * 1e6, "o-", label=r"$\alpha_x$")
    ax.plot(times, params["tilt_dy"].values * 1e6, "s-", label=r"$\alpha_y$")
    ax.set_ylabel("slope (m / 1000 km × 1e6, i.e. unit-less × 1e6)")
    ax.set_xlabel("epoch")
    ax.set_title("per-epoch tilt slopes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, params["tilt_dz"].values, "o-")
    ax.set_ylabel("offset (m)")
    ax.set_xlabel("epoch")
    ax.set_title(r"per-epoch offset $\alpha_z$")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    im = ax.imshow(
        control.values, extent=[
            float(control["x"].min()), float(control["x"].max()),
            float(control["y"].min()), float(control["y"].max()),
        ], origin="upper", cmap="Greys_r", aspect="equal",
    )
    ax.set_title(
        f"any-time coverage "
        f"({int(control.values.sum())} / {control.values.size} cells, "
        f"{100 * float(control.mean()):.1f}%)"
    )
    fig.colorbar(im, ax=ax, fraction=0.045)

    ax = axes[1, 1]
    dhdt = params["dhdt"].values
    finite = dhdt[np.isfinite(dhdt)]
    if finite.size:
        vmax = float(np.percentile(np.abs(finite), 95))
    else:
        vmax = 1.0
    im = ax.imshow(
        dhdt, extent=[
            float(params["x"].min()), float(params["x"].max()),
            float(params["y"].min()), float(params["y"].max()),
        ], origin="upper", cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="equal",
    )
    ax.set_title(r"recovered $\dot h$ (m/day, Shean PIG convention)")
    fig.colorbar(im, ax=ax, fraction=0.045)

    fig.suptitle(
        f"Venable tilt fit — {config.START_TIME} to {config.END_TIME} "
        f"({params.sizes['time']} epochs)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main(res_override: float | None = None) -> None:
    config.ensure_output_dirs()

    # Resolution-pilot suffix (feedback_res_suffix_convention): --res N
    # loads venable_stack_<N>m_*.nc and writes *_<N>m_* outputs so the
    # config.RES production products aren't clobbered.
    if res_override is not None:
        stack_prefix = f"venable_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "venable_stack"
        out_suffix = ""

    print("Loading raw stack...")
    stack, src_path = _load_raw_stack(stack_prefix=stack_prefix)
    print(f"  loaded {src_path.name}")
    print(
        f"  dims: time={stack.sizes['time']}, "
        f"y={stack.sizes['y']}, x={stack.sizes['x']}"
    )

    # Shean 2019 post-coreg correction order:
    #   tide -> IBE -> MDT -> geoid -> tilt fit
    # Venable sits at ~-75°S, north of DTU22's -79°S coverage limit, so
    # MDT applies (unlike Beardmore which is geoid-only). All static-
    # field corrections run BEFORE the per-epoch tilt LSQ.

    # 1/4: tide + IBE (per-epoch, floating-mask-gated).
    print("Applying post-coreg tide + IBE corrections...")
    stack = apply_tide_ibe_to_stack(
        stack,
        climate_cache_path=config.ERA5_CACHE_NC,
        bedmachine_path=config.BEDMACHINE_NC,
        tide_model=config.TIDE_MODEL,
        tide_model_dir=config.TIDE_MODEL_DIR,
    )

    # 2/4: DTU22 MDT (steady SSH-geoid offset, floating-only).
    print("Applying MDT to stack...")
    stack = apply_mdt_to_stack(
        stack,
        mdt_path=config.DTU22_MDT_XYZ,
        bedmachine_mask_path=config.BEDMACHINE_NC,
    )

    # 3/4: geoid undulation (BedMachine, everywhere).
    print("Applying geoid to stack...")
    stack = apply_geoid_to_stack(
        stack,
        bedmachine_path=config.BEDMACHINE_NC,
    )

    # Shean ndinterp.py-style static-control mask:
    #   1. Polygon restriction to BedMachine rock + grounded ice
    #      (the Venable stack extent is built wide enough -- via
    #      VENABLE_STACK_AOI_SHP -- to include ~47% grounded + ~7%
    #      rock margins of the Queen Alexandra Range).
    #   2. Temporal-statistics filter on the stack itself
    #      (count, ptp, std, mean, detrended residual, |trend|).
    print("Building static-control mask (BedMachine rock+grounded ∩ Shean temporal filter)...")
    polygon_mask = build_static_area_polygon_mask(stack, config.BEDMACHINE_NC)
    control = build_static_control_mask(stack, shapefile_mask=polygon_mask)
    print(
        f"  polygon mask: {int(polygon_mask.sum())}/{polygon_mask.size} "
        f"({100*float(polygon_mask.mean()):.1f}%)"
    )
    print(
        f"  static control: {int(control.sum())}/{control.size} "
        f"({100*float(control.mean()):.1f}%)"
    )

    # 4/4: per-epoch tilt LSQ on the geoid+MDT-corrected stack.
    # Venable stack is ~23 km × 54 km; Shean's 40 km Venable default would
    # disable x/y slope fitting at every epoch, collapsing the LSQ to a
    # constant-only fit and shoving cross-grid tilt into αz. 10 km lets
    # the LSQ engage slopes when the static control spans ≥10 km.
    #
    # Ez (per-epoch αz prior) scales with the ALIGNMENT control source
    # actually used by ASP, parsed from <asp_root>/asp_aligned/<dem_id>.sources.json
    # (or inferred from the ASP-root suffix for strips aligned before the
    # sidecar was added). High-precision GCPs (IS2 / ATM / LVIS) → Ez=0.1 m
    # (tightened 2026-05-05 from Shean's 0.3 m); CS2-only → Ez=2.0 m. See
    # feedback_ez_per_gcp_source.
    Ez_per_epoch, ez_summary = build_per_epoch_ez(
        stack["time"].values,
        config.ASP_ROOT,
        suffix_default="",
    )
    print(f"Per-epoch Ez (best-source breakdown): {dict(ez_summary)}")
    print("Fitting per-epoch residual tilts (Shean 2019 / Smith ndinterp.py-style)...")
    params, stack_corr = fit_tilt_stack(
        stack,
        control_mask=control,
        min_width=10000,
        Ez=Ez_per_epoch,
    )
    n_xy = int(params["fit_xy"].sum())
    print(
        f"  per-epoch fit: {n_xy}/{stack.sizes['time']} epochs got x/y slopes"
    )
    print(
        f"  |αx| range: [{float(np.abs(params['tilt_dx']).min()):.2e}, "
        f"{float(np.abs(params['tilt_dx']).max()):.2e}]"
    )
    print(
        f"  |αy| range: [{float(np.abs(params['tilt_dy']).min()):.2e}, "
        f"{float(np.abs(params['tilt_dy']).max()):.2e}]"
    )
    print(
        f"  |αz| range: [{float(np.abs(params['tilt_dz']).min()):.2f}, "
        f"{float(np.abs(params['tilt_dz']).max()):.2f}] m"
    )
    if int(params.attrs.get("robust", 0)):
        print(
            f"  IRLS (Tukey c={params.attrs.get('robust_c')}): "
            f"{params.attrs.get('robust_n_iter')} iterations"
        )
        wm = params["weight_mean"].values
        wk = params["weight_frac_kept"].values
        for k, t in enumerate(params["time"].values):
            print(
                f"    epoch {k:2d} {str(t)[:10]}  weight_mean={wm[k]:.3f}  "
                f"frac_kept={wk[k]:.3f}  "
                f"αz={float(params['tilt_dz'].values[k]):+8.2f} m  "
                f"αx={float(params['tilt_dx'].values[k]):+.2e}"
            )

    out_nc = (
        config.PROCESSED_DIR
        / f"{stack_prefix}_tilt_corrected_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Saving tilt-corrected (tide+IBE+MDT+geoid+tilt) stack -> {out_nc}")
    save_stack(stack_corr, out_nc)

    params_nc = (
        config.PROCESSED_DIR
        / f"venable_tilt_params{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Saving tilt parameters -> {params_nc}")
    params.to_netcdf(params_nc)

    fig_path = config.FIGURES_DIR / f"tilt_fit_qc{out_suffix}.png"
    plot_tilt_params(params, control, fig_path)
    print(f"  wrote {fig_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help=(
            "Resolution variant to load (meters). When set, looks up "
            "venable_stack_<N>m_<start>_<end>.nc built by "
            "`venable.build_stack --res <N>`; outputs land at *_<N>m_* "
            "paths so config.RES production isn't clobbered."
        ),
    )
    args = parser.parse_args()
    main(res_override=args.res)
