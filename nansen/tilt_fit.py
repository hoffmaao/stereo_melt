"""Per-epoch residual-tilt correction for the Nansen DEM stack.

Stage 3.5 of the Nansen pipeline. Loads the raw stack produced by
:mod:`nansen.build_stack`, builds a Shean ``ndinterp.py``-style
static-control mask (BedMachine rock + grounded ice, intersected with
the per-pixel temporal-statistics filter from
:func:`stereo_melt.coregister.tilt.build_static_control_mask`), runs
:func:`stereo_melt.coregister.tilt.fit_tilt_stack` with library
defaults (Shean 2019 PIG priors), and writes a tilt-corrected stack
as a new NetCDF that downstream melt-rate solvers consume directly.

The Nansen AOI already covers ~40% grounded ice + ~16% rock outcrop,
so no separate stack-extent shapefile is needed -- the AOI itself
gives the joint LSQ a healthy population of static-control pixels.

Saving the corrected stack on disk lets us iterate on the melt-rate
inverse (sweep Tikhonov, swap regularization terms, etc.) without
repeating the tilt LSQ every time.

Run:

    python -m nansen.tilt_fit
"""

from __future__ import annotations

import os
import sys

# PROJ_DATA fix for this conda env's broken base proj.db.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

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

from nansen import config


# --------------------------------------------------------------------------
# Stack loader
# --------------------------------------------------------------------------


def _load_raw_stack(prefix: str = "nansen_stack") -> tuple[xr.DataArray, Path]:
    return load_basin_stack(
        config.PROCESSED_DIR,
        prefix,
        config.START_TIME,
        config.END_TIME,
        prefer_tilt_corrected=False,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )


# --------------------------------------------------------------------------
# QC plotting
# --------------------------------------------------------------------------


def plot_tilt_params(params: xr.Dataset, coverage: xr.DataArray, out_path: Path) -> None:
    """QC: per-epoch tilt time series + any-time coverage + recovered dhdt."""
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
        coverage.values, extent=[
            float(coverage["x"].min()), float(coverage["x"].max()),
            float(coverage["y"].min()), float(coverage["y"].max()),
        ], origin="upper", cmap="Greys_r", aspect="equal",
    )
    ax.set_title(
        f"any-time coverage "
        f"({int(coverage.values.sum())} / {coverage.values.size} cells, "
        f"{100 * float(coverage.mean()):.1f}%)"
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
        f"Nansen tilt fit — {config.START_TIME} to {config.END_TIME} "
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

    # When `--res` is passed, look up the matching nansen_stack_<N>m_*.nc
    # built by `nansen.build_stack --res <N>`. Otherwise stay on the
    # canonical "nansen_stack_*.nc" path that the production 25 m pipeline
    # uses.
    if res_override is not None:
        stack_prefix = f"nansen_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "nansen_stack"
        out_suffix = ""

    print("Loading raw stack...")
    stack, src_path = _load_raw_stack(prefix=stack_prefix)
    print(f"  loaded {src_path.name}")
    print(
        f"  dims: time={stack.sizes['time']}, "
        f"y={stack.sizes['y']}, x={stack.sizes['x']}"
    )

    # Shean 2019 post-coreg correction order:
    #   tide -> IBE -> MDT -> geoid -> tilt fit
    # All static-field corrections (MDT, geoid) run BEFORE the per-epoch
    # tilt LSQ so the per-pixel intercept block in fit_tilt_stack carries
    # residual elevation around the orthometric/MSL surface, leaving
    # alpha_z to absorb per-strip coregistration drift only.

    # 1/4: tide + IBE (per-epoch, spatially varying CATS2008 tide on the
    # 25 m grid via thin-plate-RBF over valid CATS samples, plus per-epoch
    # IBE scalar from ARCO; both gated by a 3 km feathered floating mask).
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

    # Shean ndinterp.py-style static-control mask: BedMachine rock +
    # grounded ice (~56% of the Nansen AOI) intersected with the
    # temporal-statistics filter (count, ptp, std, mean, detrended
    # residual, |trend|).
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

    # 4/4: per-epoch tilt LSQ on the geoid+MDT-corrected stack (Shean
    # 2019 / Smith ndinterp.py-style). Static-only joint LSQ: per-strip
    # tilts constrained by rock+grounded observations. Observation_mask=
    # full-grid was tested and rejected — Ez=0.1 m is much tighter than
    # Eint=10 m so the LSQ pins αz_mean to ~0 and absorbs the DC into
    # per-pixel intercepts. See project_joint_lsq_exploration_2026_05_03.
    # min_width=10 km lets αx/αy fit on Nansen's narrow strips.
    #
    # Ez (per-epoch αz prior) scales with the ALIGNMENT control source
    # actually used by ASP, parsed from <asp_root>/asp_aligned/<dem_id>.sources.json
    # (or inferred from the ASP-root suffix for strips aligned before
    # the sidecar was added). High-precision GCPs (IS2 / ATM / LVIS) →
    # Ez=0.1 m (tightened 2026-05-05 from 0.3 m after the dh/dt diagnosis
    # showed +0.07 m/yr static-control bias survived Ez=0.3); CS2-only →
    # Ez=2.0 m. See feedback_ez_per_gcp_source.
    Ez_per_epoch, ez_summary = build_per_epoch_ez(
        stack["time"].values,
        config.ASP_ROOT,
        suffix_default="",  # per-basin ASP root → IS2 + rock by default
    )
    # ez_summary is histogram-by-winning-source, e.g. {"is2": 25, "missing": 0}.
    # The label is whichever altimetric source supplied the minimum Ez for
    # each strip's control mix (is2/atm/lvis at 0.3 m; cs2/rock at 2.0 m).
    print(f"Per-epoch Ez (best-source breakdown): {dict(ez_summary)}")
    print("Fitting per-epoch residual tilts (Shean 2019 / Smith ndinterp.py-style)...")
    params, stack_corr = fit_tilt_stack(
        stack,
        control_mask=control,
        min_width=10000.0,
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

    out_nc = (
        config.PROCESSED_DIR
        / f"{stack_prefix}_tilt_corrected_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Saving tilt-corrected (tide+IBE+MDT+geoid+tilt) stack -> {out_nc}")
    save_stack(stack_corr, out_nc)

    params_nc = (
        config.PROCESSED_DIR
        / f"nansen_tilt_params{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
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
            "nansen_stack_<N>m_*.nc built by `nansen.build_stack --res <N>`. "
            "Outputs land at *<N>m* paths so 25 m production isn't clobbered."
        ),
    )
    args = parser.parse_args()
    main(res_override=args.res)
