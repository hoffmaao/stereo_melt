"""Per-epoch residual-tilt correction for the Beardmore_Shelf DEM stack.

Stage 3.5 of the Beardmore_Shelf pipeline. Loads the raw stack produced by
:mod:`beardmore_shelf.build_stack`, builds a Shean ``ndinterp.py``-style
static-control mask (BedMachine rock + grounded ice, intersected with
the per-pixel temporal-statistics filter from
:func:`stereo_melt.coregister.tilt.build_static_control_mask`), runs
:func:`stereo_melt.coregister.tilt.fit_tilt_stack` with library
defaults (Shean 2019 Beardmore_Shelf priors), and writes a tilt-corrected stack
as a new NetCDF that downstream melt-rate solvers consume directly.

The Beardmore_Shelf stack covers a wider domain than ``BEARDMORE_SHELF_AOI_SHP``
(see ``BEARDMORE_SHELF_STACK_AOI_SHP``: ~30 km extension into the Queen
Alexandra Range, ~47% grounded + ~7% rock outcrop) for exactly this
reason: the joint LSQ needs a healthy population of static-control
pixels to anchor the per-epoch tilts.

Saving the corrected stack on disk lets us iterate on the melt-rate
inverse (sweep Tikhonov, swap regularization terms, etc.) without
repeating the tilt LSQ every time.

Run:

    python -m beardmore_shelf.tilt_fit
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
    build_ice_domain_mask,
    build_static_area_polygon_mask,
    build_static_control_mask,
    fit_tilt_stack,
)
from stereo_melt.corrections.post_coreg import apply_tide_ibe_to_stack
from stereo_melt.pipeline import apply_geoid_to_stack
from stereo_melt.stack import load_basin_stack, save_stack

from beardmore_shelf import config


# --------------------------------------------------------------------------
# Stack loaders
# --------------------------------------------------------------------------


def _load_raw_stack(
    stack_prefix: str = "beardmore_shelf_stack",
) -> tuple[xr.DataArray, Path]:
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
        f"Beardmore_Shelf tilt fit — {config.START_TIME} to {config.END_TIME} "
        f"({params.sizes['time']} epochs)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main(res_override: float | None = None, tag: str | None = None) -> None:
    config.ensure_output_dirs()

    # Variant suffix (feedback_res_suffix_convention): --res N and/or --tag
    # <t> load beardmore_shelf_stack[_<N>m][_<t>]_*.nc and write matching
    # outputs so the config.RES production chain isn't clobbered.
    out_suffix = f"_{int(round(res_override))}m" if res_override is not None else ""
    if tag:
        out_suffix += f"_{tag}"
    stack_prefix = f"beardmore_shelf_stack{out_suffix}"

    print("Loading raw stack...")
    stack, src_path = _load_raw_stack(stack_prefix=stack_prefix)
    print(f"  loaded {src_path.name}")
    print(
        f"  dims: time={stack.sizes['time']}, "
        f"y={stack.sizes['y']}, x={stack.sizes['x']}"
    )

    # Shean 2019 post-coreg correction order:
    #   tide -> IBE -> MDT -> geoid -> tilt fit
    # MDT is skipped here (lat ~-84°S, poleward of DTU22's -79°S coverage
    # limit — same as the beardmore basin), so the chain reduces to:
    #   tide -> IBE -> geoid -> tilt fit
    # All static-field corrections run BEFORE the per-epoch tilt LSQ.

    # 1/3: tide + IBE (per-epoch, floating-mask-gated).
    print("Applying post-coreg tide + IBE corrections...")
    stack = apply_tide_ibe_to_stack(
        stack,
        climate_cache_path=config.ERA5_CACHE_NC,
        bedmachine_path=config.BEDMACHINE_NC,
        tide_model=config.TIDE_MODEL,
        tide_model_dir=config.TIDE_MODEL_DIR,
    )

    # 2/3: geoid undulation (BedMachine, everywhere). MDT skipped.
    print("Applying geoid to stack (MDT skipped: south of DTU22)...")
    stack = apply_geoid_to_stack(
        stack,
        bedmachine_path=config.BEDMACHINE_NC,
    )

    # Shean ndinterp.py-style static-control mask:
    #   1. Polygon restriction to BedMachine rock + grounded ice
    #      (the v15 stack extent is grown upstream over the Queen
    #      Alexandra Range specifically so this mask has grounded-ice +
    #      rock surfaces to anchor to).
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

    # Tilt observation domain. Two modes (BEARDMORE_SHELF_TILT_DOMAIN):
    #
    # "static" (default) -- per-epoch tilt LSQ on the STATIC control only,
    # floating shelf EXCLUDED, matching pig/tilt_fit.py production. Without a
    # trend regularizer this is the only stable choice: admitting the shelf
    # opens the joint-LSQ nullspace in which a per-epoch tilt growing linearly
    # in time aliases the real along-flow melt gradient, manufacturing
    # accretion at the shelf front (the 2026-06-29 revert). BUT it gives
    # control-free (nocorr) epochs ZERO observation rows, so their αz
    # collapses to the prior mean and their datum error goes uncorrected
    # (the 2026-07-11 nocorr A/B failure).
    #
    # "full" -- Shean's actual production system (vendor ndinterp.py:
    # clip_to_shelfmask=False, both=True): ALL valid ice pixels including the
    # floating shelf. MUST be paired with the dh/dt smoothness constraint
    # (BEARDMORE_SHELF_TILT_DHDT_SMOOTH, Shean L574+ unit weight = 1.0) --
    # that is what stabilizes per-pixel trends on sparse floating pixels and
    # lets cross-epoch self-consistency set nocorr αz without the aliasing
    # nullspace.
    tilt_domain = os.environ.get("BEARDMORE_SHELF_TILT_DOMAIN", "static").lower()
    dhdt_smooth = float(os.environ.get("BEARDMORE_SHELF_TILT_DHDT_SMOOTH", "0"))
    # IRLS iteration cap. Both 2026-07-11 full+smooth runs plateaued by
    # iteration 4 (scale drift <1% after), so the ~630-epoch full-record
    # solve can stop at 5 without measurable quality loss.
    irls_max = int(os.environ.get("BEARDMORE_SHELF_TILT_IRLS_MAX", "8"))
    if irls_max != 8:
        print(f"  IRLS max iterations: {irls_max} (BEARDMORE_SHELF_TILT_IRLS_MAX)")
    if tilt_domain == "full":
        ice_domain = build_ice_domain_mask(stack, config.BEDMACHINE_NC)
        obs_domain = ice_domain
        print(
            f"  tilt-fit observation domain = full ice incl. shelf (Shean "
            f"ndinterp both-mode): {int(obs_domain.values.sum())}/{obs_domain.size} "
            f"({100*float(obs_domain.values.mean()):.1f}%)"
        )
        if not dhdt_smooth:
            print(
                "  ⚠ full domain WITHOUT dh/dt smoothness reproduces the "
                "2026-06-29 aliasing nullspace — set "
                "BEARDMORE_SHELF_TILT_DHDT_SMOOTH=1.0 unless deliberately "
                "reproducing it."
            )
    elif tilt_domain == "static":
        obs_domain = control
        print(
            f"  tilt-fit observation domain = static control (shelf excluded): "
            f"{int(control.sum())}/{control.size} ({100*float(control.mean()):.1f}%)"
        )
    else:
        raise SystemExit(
            f"BEARDMORE_SHELF_TILT_DOMAIN must be 'static' or 'full', "
            f"got {tilt_domain!r}"
        )
    if dhdt_smooth:
        print(f"  dh/dt smoothness weight: {dhdt_smooth:g} (Shean ndinterp L574+)")

    # 3/3: per-epoch tilt LSQ on the geoid-corrected stack.
    # Shean's 40 km min_width would disable x/y slope fitting at epochs
    # whose static control spans less than 40 km, collapsing the LSQ to a
    # constant-only fit and shoving cross-grid tilt into αz. 10 km (the
    # venable/mcmurdo default) lets slopes engage once control spans ≥10 km.
    #
    # Ez (per-epoch αz prior) scales with the ALIGNMENT control source
    # actually used by ASP, parsed from <asp_root>/asp_aligned/<dem_id>.sources.json
    # (or inferred from the ASP-root suffix for strips aligned before the
    # sidecar was added). High-precision GCPs (IS2 / ATM / LVIS) → Ez=0.1 m
    # (tightened 2026-05-05 from Shean's 0.3 m); CS2-only → Ez=2.0 m. See
    # feedback_ez_per_gcp_source.
    # Ez roots follow config.STRIP_SOURCES so the combined 2012-2024 stack
    # (is2cs2 + pre-IS2 ctempoatm) resolves each epoch against the root that
    # actually aligned it: IS2-era epochs -> ASP/ (is2/atm -> Ez 0.1), pre-IS2
    # epochs -> ASP_ctempoatm/ (cryotempo -> Ez 1.0). Dates never overlap across
    # eras, so there is no cross-root contamination. Baseline (1 source) passes
    # the bare ASP_ROOT exactly as before.
    ez_roots = [src_dir.parent for src_dir, _variant in config.STRIP_SOURCES]
    Ez_per_epoch, ez_summary = build_per_epoch_ez(
        stack["time"].values,
        ez_roots if len(ez_roots) > 1 else ez_roots[0],
        # Resolve each layer by ITS OWN strip: without this, a nocorr layer
        # sharing a date with an is2 strip inherits the is2 Ez 0.1 prior
        # (35/99 nocorr layers on the 07-11 stack).
        epoch_dem_ids=(stack["dem_id"].values if "dem_id" in stack.coords else None),
        suffix_default="",
    )
    print(f"Per-epoch Ez (best-source breakdown): {dict(ez_summary)}")

    # P3 (pig/tilt_fit.py parity): demote poorly-coregistered strips (high
    # pc_align end_p50) to an offset-only (alpha_z) tilt fit rather than dropping
    # them -- a full x/y plane on a strip pc_align couldn't lock is
    # noise-dominated and injects a spurious ramp into dh/dt. Threshold via
    # BEARDMORE_SHELF_OFFSET_ONLY_END_P50_M (default 3 m); strips above the
    # find_bad_epochs drop gate are already gone via BAD_STRIPS.
    offset_only = None
    offset_thresh = float(
        os.environ.get("BEARDMORE_SHELF_OFFSET_ONLY_END_P50_M", "3.0")
    )
    try:
        from stereo_melt.coregister.alignment_quality import aggregate_basin_quality

        aq = aggregate_basin_quality(config.STRIP_SOURCES)
        if not aq.empty:
            aq = aq.copy()
            aq["d"] = pd.to_datetime(aq["date"]).dt.normalize()
            per = aq.groupby("d")["end_p50"].max()
            stimes = pd.to_datetime(stack["time"].values).normalize()
            ep = np.array([per.get(d, np.nan) for d in stimes], dtype=float)
            offset_only = np.isfinite(ep) & (ep > offset_thresh)
            print(
                f"  P3 offset-only: {int(offset_only.sum())}/{len(offset_only)} "
                f"epochs have end_p50>{offset_thresh:.1f}m -> alpha_z-only fit"
            )
    except Exception as exc:
        print(f"  P3 offset-only gate skipped: {exc}")

    print("Fitting per-epoch residual tilts (Shean 2019 / Smith ndinterp.py-style)...")
    params, stack_corr = fit_tilt_stack(
        stack,
        control_mask=control,
        observation_mask=obs_domain,
        min_width=10000,
        Ez=Ez_per_epoch,
        offset_only_epochs=offset_only,
        dhdt_smoothness=dhdt_smooth or None,
        robust_max_iter=irls_max,
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
    print(f"Saving tilt-corrected (tide+IBE+geoid+tilt) stack -> {out_nc}")
    save_stack(stack_corr, out_nc)

    params_nc = (
        config.PROCESSED_DIR
        / f"beardmore_shelf_tilt_params{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
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
            "beardmore_shelf_stack_<N>m_<start>_<end>.nc built by "
            "`beardmore_shelf.build_stack --res <N>`; outputs land at "
            "*_<N>m_* paths so config.RES production isn't clobbered."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help=(
            "Variant tag (after any _<N>m) matching a `build_stack --tag` "
            "run, e.g. --tag is2ctempo."
        ),
    )
    args = parser.parse_args()
    main(res_override=args.res, tag=args.tag)
