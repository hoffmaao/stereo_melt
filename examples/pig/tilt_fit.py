"""Per-epoch residual-tilt correction for the Pine Island DEM stack.

Stage 3.5 of the Pine Island pipeline. Loads the raw stack produced by
:mod:`pig.build_stack`, builds a Shean ``ndinterp.py``-style
static-control mask (BedMachine rock + grounded ice, intersected with
the per-pixel temporal-statistics filter from
:func:`stereo_melt.coregister.tilt.build_static_control_mask`), runs
:func:`stereo_melt.coregister.tilt.fit_tilt_stack` with library
defaults (Shean 2019 PIG priors), and writes a tilt-corrected stack
as a new NetCDF that downstream melt-rate solvers consume directly.

Opt-in env switches, each defaulting to the behaviour above so the canon
products come out unchanged: ``PIG_TILT_DOMAIN=full`` fits over all ice pixels
including the shelf (Shean ndinterp both-mode) and must be paired with
``PIG_TILT_DHDT_SMOOTH=1.0``; ``PIG_TILT_IRLS_MAX`` caps the Tukey IRLS
iterations; ``PIG_TILT_EZ_BY_DEM_ID=1`` resolves per-epoch Ez per layer rather
than by date union (implied whenever a nocorr root is active). What each is
for is in the comments at its call site below.

The PIG stack covers a wider domain than ``PIG_AOI_SHP``
(see ``PIG_STACK_AOI_SHP``: ~30 km extension into the Queen
Alexandra Range, ~47% grounded + ~7% rock outcrop) for exactly this
reason: the joint LSQ needs a healthy population of static-control
pixels to anchor the per-epoch tilts.

Saving the corrected stack on disk lets us iterate on the melt-rate
inverse (sweep Tikhonov, swap regularization terms, etc.) without
repeating the tilt LSQ every time.

Run:

    python -m pig.tilt_fit
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
from stereo_melt.pipeline import apply_geoid_to_stack, apply_mdt_to_stack
from stereo_melt.stack import load_basin_stack, save_stack

from pig import config


# --------------------------------------------------------------------------
# Stack loaders
# --------------------------------------------------------------------------


def _load_raw_stack(stack_prefix: str = "pig_stack") -> tuple[xr.DataArray, Path]:
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
        f"PIG tilt fit — {config.START_TIME} to {config.END_TIME} "
        f"({params.sizes['time']} epochs)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main(
    res_override: float | None = None,
    tag: str | None = None,
    pre_is2_asp: str | None = None,
    is2_asp: str | None = None,
) -> None:
    config.ensure_output_dirs()

    if res_override is not None:
        stack_prefix = f"pig_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "pig_stack"
        out_suffix = ""
    if tag:
        stack_prefix += f"_{tag}"
        out_suffix += f"_{tag}"
    if pre_is2_asp:
        variant = pre_is2_asp.lstrip("_")
        config.STRIP_SOURCES = [
            config.STRIP_SOURCES[0],
            (config.BASIN_DIR / "data" / f"ASP_{variant}" / "asp_aligned", variant),
        ] + config.STRIP_SOURCES[2:]
    if is2_asp:
        variant = is2_asp.lstrip("_")
        config.STRIP_SOURCES = [
            (config.BASIN_DIR / "data" / f"ASP_{variant}" / "asp_aligned", variant),
            config.STRIP_SOURCES[1],
        ] + config.STRIP_SOURCES[2:]

    print("Loading raw stack...")
    stack, src_path = _load_raw_stack(stack_prefix=stack_prefix)
    print(f"  loaded {src_path.name}")
    print(
        f"  dims: time={stack.sizes['time']}, "
        f"y={stack.sizes['y']}, x={stack.sizes['x']}"
    )

    # Shean 2019 post-coreg correction order:
    #   tide -> IBE -> MDT -> geoid -> tilt fit
    # PIG sits at ~-75°S, north of DTU22's -79°S coverage limit, so
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
    #      (the PIG stack extent is built wide enough -- via
    #      PIG_STACK_AOI_SHP -- to include ~47% grounded + ~7%
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

    # Shean-faithful tilt domain: fit the per-epoch tilt LSQ on the STATIC
    # control ONLY (the validmask above) -- floating shelf + upstream trunk
    # EXCLUDED -- exactly as Shean's ndinterp_tiltcorr.py masks the main
    # shelf and any |trend|>2 m/yr pixel out of its tilt fit. Admitting the
    # shelf as observations (the 2026-06-07 DEM-interior widening, reverted
    # here) opens the joint-LSQ nullspace in which a per-epoch tilt growing
    # linearly in time aliases the real along-flow melt gradient (max
    # thinning at the GL, tapering to the front); removing that tilt then
    # manufactures accretion at the shelf front. All strips are retained:
    # every epoch contributes its control-overlap pixels, and strips with
    # too little static overlap degrade to offset-only via min_width + the
    # Ez/Ex/Ey priors -- none are dropped.
    #
    # PIG_TILT_DOMAIN=full (2026-09-06 nocorr test; mirrors beardmore_shelf)
    # switches to Shean's actual production system (vendor ndinterp.py
    # clip_to_shelfmask=False, both=True): ALL valid ice pixels including the
    # floating shelf. It MUST be paired with the dh/dt smoothness rows
    # (PIG_TILT_DHDT_SMOOTH, Shean L574+ unit weight = 1.0), which is what
    # closes the aliasing nullspace above and lets cross-epoch self-consistency
    # set the datum of control-free (nocorr) epochs, which the static domain
    # gives ZERO observation rows. Default remains static.
    tilt_domain = os.environ.get("PIG_TILT_DOMAIN", "static").lower()
    dhdt_smooth = float(os.environ.get("PIG_TILT_DHDT_SMOOTH", "0"))
    irls_max = int(os.environ.get("PIG_TILT_IRLS_MAX", "8"))
    if irls_max != 8:
        print(f"  IRLS max iterations: {irls_max} (PIG_TILT_IRLS_MAX)")
    if tilt_domain == "full":
        obs_domain = build_ice_domain_mask(stack, config.BEDMACHINE_NC)
        print(
            f"  tilt-fit observation domain = full ice incl. shelf (Shean "
            f"ndinterp both-mode): {int(obs_domain.values.sum())}/{obs_domain.size} "
            f"({100*float(obs_domain.values.mean()):.1f}%)"
        )
        if not dhdt_smooth:
            print(
                "  ⚠ full domain WITHOUT dh/dt smoothness reproduces the "
                "2026-06-07 aliasing nullspace — set PIG_TILT_DHDT_SMOOTH=1.0 "
                "unless deliberately reproducing it."
            )
    elif tilt_domain == "static":
        obs_domain = control
        print(
            f"  tilt-fit observation domain = static control (shelf excluded): "
            f"{int(control.sum())}/{control.size} ({100*float(control.mean()):.1f}%)"
        )
    else:
        raise SystemExit(
            f"PIG_TILT_DOMAIN must be 'static' or 'full', got {tilt_domain!r}"
        )
    if dhdt_smooth:
        print(f"  dh/dt smoothness weight: {dhdt_smooth:g} (Shean ndinterp L574+)")

    # 4/4: per-epoch tilt LSQ on the geoid+MDT-corrected stack.
    # PIG stack is ~23 km × 54 km; Shean's 40 km PIG default would
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
    # Multi-root form: walk every ASP root in STRIP_SOURCES so the pre-IS2
    # IS2+ATM+LVIS era (ASP_is2atmlvis/) and the post-Oct 2018 IS2+CS2 era
    # (ASP/) both contribute sources.json sidecars.
    # Layers resolve by DATE UNION by default -- every strip sharing a timestamp
    # contributes, min Ez wins -- which is how the canon PIG products were built,
    # so that stays the default and they remain byte-identical. Resolve by dem_id
    # instead when a nocorr root is active (or PIG_TILT_EZ_BY_DEM_ID=1): REMA
    # filenames carry no time-of-day, so a control-free nocorr layer sharing a date
    # with an aligned strip would otherwise inherit its Ez 0.1 prior instead of the
    # nocorr 1.0 tier, pinning the one datum the recipe needs the LSQ to estimate
    # (beardmore_shelf 2026-07-11: 35 of 99 nocorr layers leaked that way).
    ez_by_dem_id = (os.environ.get("PIG_TILT_EZ_BY_DEM_ID", "") == "1"
                    or any(variant == "nocorr" for _d, variant in config.STRIP_SOURCES))
    ez_dem_ids = (stack["dem_id"].values
                  if ez_by_dem_id and "dem_id" in stack.coords else None)
    if ez_by_dem_id and ez_dem_ids is None:
        print("WARNING: by-dem_id Ez resolution requested but this stack carries no dem_id "
              "coord, so Ez falls back to date union -- a nocorr layer sharing a REMA date "
              "with an aligned strip will inherit that strip's tighter Ez. Rebuild the stack "
              "with build_stack to close the leak.")
    Ez_per_epoch, ez_summary = build_per_epoch_ez(
        stack["time"].values,
        [(p.parent, suf) for p, suf in config.STRIP_SOURCES],
        epoch_dem_ids=ez_dem_ids,
    )
    print(f"Per-epoch Ez (best-source breakdown, resolved by "
          f"{'dem_id' if ez_dem_ids is not None else 'date'}): {dict(ez_summary)}")

    # P3: demote poorly-coregistered strips (high pc_align end_p50) to an
    # offset-only (alpha_z) tilt fit rather than dropping them -- a full x/y
    # plane on a strip pc_align couldn't lock is noise-dominated and injects a
    # spurious ramp into dh/dt. Threshold via PIG_OFFSET_ONLY_END_P50_M (default
    # 3 m); strips above the find_bad_epochs drop gate (~5 m) are already gone.
    offset_only = None
    offset_thresh = float(os.environ.get("PIG_OFFSET_ONLY_END_P50_M", "3.0"))
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
            print(f"  P3 offset-only: {int(offset_only.sum())}/{len(offset_only)} "
                  f"epochs have end_p50>{offset_thresh:.1f}m -> alpha_z-only fit")
    except Exception as exc:
        print(f"  P3 offset-only gate skipped: {exc}")
    print("Fitting per-epoch residual tilts (Shean 2019 / Smith ndinterp.py-style)...")
    print(f"  tilt-plane priors: Ex={config.TILT_EX:.3g} Ey={config.TILT_EY:.3g} m/m "
          "(config.TILT_EX/EY; control-residual calibration 2026-09-02)")
    params, stack_corr = fit_tilt_stack(
        stack,
        control_mask=control,
        observation_mask=obs_domain,
        min_width=10000,
        Ex=config.TILT_EX,
        Ey=config.TILT_EY,
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
    print(f"Saving tilt-corrected (tide+IBE+MDT+geoid+tilt) stack -> {out_nc}")
    save_stack(stack_corr, out_nc)

    params_nc = (
        config.PROCESSED_DIR
        / f"pig_tilt_params{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
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
            "pig_stack_<N>m_<start>_<end>.nc built by `pig.build_stack --res <N>`. "
            "Outputs land at *<N>m* paths so 25 m production isn't clobbered."
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Experiment tag matching the `pig.build_stack --tag` build: "
            "loads pig_stack_<N>m_<tag>_*.nc and writes "
            "*_<tag>_tilt_corrected_*.nc."
        ),
    )
    parser.add_argument(
        "--pre-is2-asp",
        default=None,
        help=(
            "Override the pre-IS2 strip source for per-epoch Ez sidecar "
            "lookup (e.g. ctempoatmlvis); must match the build_stack run. "
            "Replaces only that entry: any further roots (PIG_SOURCES=nocorr "
            "appends one) stay in the list, so by-dem_id Ez still sees them."
        ),
    )
    parser.add_argument(
        "--is2-asp",
        default=None,
        help=(
            "Override the IS2-era strip source for per-epoch Ez sidecar "
            "lookup; must match the build_stack run, and leaves any further "
            "roots in place. Point both --is2-asp "
            "and --pre-is2-asp at the same variant for a uniform-control "
            "stack (e.g. ctempoatmlvis)."
        ),
    )
    args = parser.parse_args()
    main(res_override=args.res, tag=args.tag, pre_is2_asp=args.pre_is2_asp,
         is2_asp=args.is2_asp)
