"""A/B: budget-corrected pair-banded linear inverse vs the production path solver.

The linear-inverse framework was rebuilt (2026-07-02,
``literature/plan_match_linear_inverse.md``) to match
:func:`stereo_melt.melt.lagrangian_melt_rate` by construction: same Shean
pair banding and two-level median, per-pixel budget correction
(``H_f·∇·u``, ``ȧ``, ``u·∇d``), hydrostatic/non-hydrostatic split. The
synthetic gate (``stereo_melt/tests/gate_match_lagrangian.py``) passes; this
script is the real-data leg.

Loads the SAME inputs as ``pig.run_melt`` (250 m is2ctempo tilt-corrected
stack, ase-quarterly velocity, RACMO SMB, BedMachine firn + floating mask),
runs :func:`stereo_melt.dynamics.linear_inverse_budget_melt_rate`, and
compares against ``melt_rate_lagrangian`` from the saved production
baseline: spatial correlation (raw / 2 km / 5 km, NaN-aware), median/IQR,
and the GL-2 km clip flux vs Shean's 82-93 Gt/yr.

v2 (2026-07-03): velocity passes through TIME-VARYING (auto-detected in the
solver; per-sub-step interp mirrors the production path solver) and the fan
strain-H uses the record trend at fan mid-time. v1 (time-mean velocity,
fan-mean H) landed at flux 120.6 Gt/yr vs REF 88.6 with a 1.6x IQR stretch;
``pig/diag_path_meanvel.py`` showed the production path solver itself gives
125.2 Gt/yr and the same stretch when fed time-MEAN velocity — velocity
representation, not estimator physics, was the gap. Outputs carry a ``_tv``
tag so the v1 files are preserved.

The production path solver is untouched; this writes a separate results
file (``pig_lininv_budget_…``) and figure.

Run (long job — keep it disconnect-safe):

    cd /wd2/projects/stereo_melt
    nohup /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u \
        -m pig.compare_budget_inverse > pig/logs/compare_budget_inverse.log 2>&1 &

Env knobs: ``LININV_REG`` (default 0.1), ``LININV_CORR_SIGMA_M`` (default
H_ref), ``LININV_ETA_BAR`` (default 1e14), ``LININV_FANH``
(trend|fan-mean), ``LININV_ESTIMATOR`` (split|irls), ``LININV_CORR_AGG``
(pooled|fan-median), ``LININV_ATTRIBUTION`` (seed|path), ``LININV_OUT_TAG``
(output/figure tag; keep one per configuration so prior legs are
preserved), ``LININV_FAN_CACHE`` (fan-maps memo path; "" disables),
``PIG_VELOCITY`` (default ase-quarterly to mirror the baseline).

Leg ledger (2026-07-03): ``_tv`` seed/split; ``_tv_irls``; ``_tv_corrfm``;
``_tv_path``; ``_tv_gapfix`` + ``_tv_path_gapfix`` = same after the
c-integral mosaic-gap fix (design note §9) — the gapfix legs supersede the
earlier tv numbers; pre-fix files are artefact exhibits.
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

# Mirror the 2026-07-02 baseline unless the caller overrides.
os.environ.setdefault("PIG_VELOCITY", "ase-quarterly")

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from stereo_melt.dynamics import linear_inverse_budget_melt_rate
from stereo_melt.flux import grounding_buffer, integrate_basal_flux
from stereo_melt.io.bedmachine import load_firn_on_grid

from pig import config
from pig.run_melt import (
    load_floating_mask,
    load_grounded_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
)

BASELINE_NC = (
    config.RESULTS_DIR
    / "pig_melt_250m_is2ctempo_parcellsq_2010-01-01_2024-01-10.nc"
)
STACK_PREFIX = "pig_stack_250m_is2ctempo"
OUT_TAG = os.environ.get("LININV_OUT_TAG", "250m_is2ctempo_tv")


def nan_gauss(a, sigma_pix):
    if sigma_pix <= 0:
        return a
    m = np.isfinite(a)
    a0 = np.where(m, a, 0.0)
    num = gaussian_filter(a0, sigma_pix, mode="nearest")
    den = gaussian_filter(m.astype(float), sigma_pix, mode="nearest")
    out = num / np.maximum(den, 1e-9)
    return np.where(den > 0.05, out, np.nan)


def scorr(a, b, sigma_pix):
    aa = nan_gauss(np.asarray(a, float), sigma_pix)
    bb = nan_gauss(np.asarray(b, float), sigma_pix)
    f = np.isfinite(aa) & np.isfinite(bb)
    if f.sum() < 100:
        return np.nan
    av = aa[f] - aa[f].mean()
    bv = bb[f] - bb[f].mean()
    den = np.sqrt((av**2).sum() * (bv**2).sum())
    return float((av * bv).sum() / den) if den > 0 else np.nan


def corr_row(name, v, ref, res_m):
    return (
        f"{name:<22} vs Lagrangian: raw {scorr(v, ref, 0):.3f}  "
        f"2km {scorr(v, ref, 2000.0 / res_m):.3f}  "
        f"5km {scorr(v, ref, 5000.0 / res_m):.3f}"
    )


def main() -> None:
    config.ensure_output_dirs()
    t0 = time.time()

    print("Loading stack...")
    stack = load_stack(stack_prefix=STACK_PREFIX)
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    print("Building floating-ice mask (BedMachine v3)...")
    floating = load_floating_mask(stack)
    stack = stack.where(floating)

    print("Loading velocity...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    tv = "time" in vx.dims and vx.sizes.get("time", 1) > 1
    print(f"  velocity source: {vel_source} "
          f"({'TIME-VARYING, per-sub-step interp' if tv else 'static'})")

    print("Loading SMB (RACMO2.4p1)...")
    a_dot = load_smb_on_grid(stack)

    print("Loading firn air content (BedMachine, static)...")
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    reg = float(os.environ.get("LININV_REG", "0.1"))
    eta_bar = float(os.environ.get("LININV_ETA_BAR", "1e14"))
    sigma_env = os.environ.get("LININV_CORR_SIGMA_M", "").strip()
    corr_sigma = float(sigma_env) if sigma_env else None
    fan_H = os.environ.get("LININV_FANH", "trend").strip()
    estimator = os.environ.get("LININV_ESTIMATOR", "split").strip()
    corr_agg = os.environ.get("LININV_CORR_AGG", "pooled").strip()
    slope_agg = os.environ.get("LININV_SLOPE_AGG", "pooled-pairs").strip()
    attribution = os.environ.get("LININV_ATTRIBUTION", "seed").strip()

    print(
        f"Running budget-corrected pair-banded linear inverse "
        f"(reg={reg:g}, eta_bar={eta_bar:g}, corr_sigma="
        f"{'H_ref' if corr_sigma is None else corr_sigma}, fan_H={fan_H}, "
        f"estimator={estimator}, corr_aggregate={corr_agg}, "
        f"slope_aggregate={slope_agg}, attribution={attribution})..."
    )
    fan_cache = os.environ.get(
        "LININV_FAN_CACHE",
        str(config.PROCESSED_DIR
            / f"pig_lininv_fanmaps_{STACK_PREFIX}_tv_{fan_H}_{attribution}"
              f"_{slope_agg}.npz"),
    ).strip() or None
    print(f"  fan-maps cache: {fan_cache}")
    ds = linear_inverse_budget_melt_rate(
        stack, vx, vy, a_dot=a_dot, d=firn, floating_mask=floating,
        eta_bar=eta_bar, reg=reg, transform="dct",
        min_pair_dt_yr=1.5, max_pair_dt_yr=2.5, dt_yr=0.05,
        corr_prefilter_sigma_m=corr_sigma, fan_H=fan_H,
        estimator=estimator, corr_aggregate=corr_agg,
        slope_aggregate=slope_agg,
        attribution=attribution, fan_cache=fan_cache,
        progress=True, progress_interval_s=30.0,
    )
    mr = ds.melt_rate
    print(
        f"  lin_budget melt_rate: median={float(mr.median()):.2f}  "
        f"IQR=[{float(mr.quantile(0.25)):.2f}, {float(mr.quantile(0.75)):.2f}] "
        f"m ice/yr  finite-cell-count={int(mr.notnull().sum())}  "
        f"({ds.attrs['n_starts']} fans, {ds.attrs['n_pairs']} pairs, "
        f"mean dt {ds.attrs['mean_pair_dt_yr']:.2f} yr)  "
        f"[{time.time() - t0:.0f}s]"
    )

    print(f"Loading baseline {BASELINE_NC.name} ...")
    base = xr.open_dataset(BASELINE_NC)
    lagr = base["melt_rate_lagrangian"]
    eul = base["melt_rate_eulerian"]
    lagr_count = base["lagrangian_count"]
    res_m = abs(float(stack["x"].values[1] - stack["x"].values[0]))

    lag_v = np.asarray(lagr.values, float)
    print("\n=== spatial agreement (NaN-aware smoothed Pearson) ===")
    print(corr_row("eulerian (benchmark)", np.asarray(eul.values, float), lag_v, res_m))
    print(corr_row("lin_budget", np.asarray(mr.values, float), lag_v, res_m))
    print(corr_row("lin_budget_hydro", np.asarray(ds.melt_rate_hydro.values, float), lag_v, res_m))

    print("\n=== distribution ===")
    for name, da in (
        ("lagrangian(REF)", lagr), ("eulerian", eul),
        ("lin_budget", mr), ("lin_budget_hydro", ds.melt_rate_hydro),
    ):
        print(
            f"{name:<18} median={float(da.median()):+7.2f}  "
            f"IQR=[{float(da.quantile(0.25)):+7.2f}, {float(da.quantile(0.75)):+7.2f}]  "
            f"cells={int(da.notnull().sum()):,}"
        )

    print("\n=== GL-2km clip flux (Shean 82-93 Gt/yr; count>=10, |melt|<250 clip) ===")
    grounded = load_grounded_mask(stack)
    dom = grounding_buffer(floating, grounded, 2000.0, res_m)
    rows = [
        ("Lagrangian(REF)", lagr, np.nan_to_num(lagr_count.values, nan=0) >= 10),
        ("lin_budget", mr, np.nan_to_num(ds["count"].values, nan=0) >= 10),
        ("lin_budget_hydro", ds.melt_rate_hydro,
         np.nan_to_num(ds["count"].values, nan=0) >= 10),
        ("lin_budget|lagr-gate", mr, np.nan_to_num(lagr_count.values, nan=0) >= 10),
    ]
    flux_attrs = {}
    for name, field, qual in rows:
        b = integrate_basal_flux(field, dom, res_m, quality=qual)
        print(
            f"{name:<22} area={b['area_km2']:7.0f} km2  med={b['median_myr']:+6.2f}  "
            f"Gt/yr: CLIP={b['gt_clip']:.1f}  raw={b['gt_raw']:.1f}  "
            f"robust={b['gt_robust']:.1f}"
        )
        flux_attrs[f"flux_clip_{name}"] = float(b["gt_clip"])

    out_nc = (
        config.RESULTS_DIR
        / f"pig_lininv_budget_{OUT_TAG}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    ds_out = ds.copy()
    ds_out.attrs.update(
        {
            "baseline": BASELINE_NC.name,
            "velocity_source": vel_source
            + (" (time-varying, per-sub-step interp)" if tv else " (static)"),
            **{k: v for k, v in flux_attrs.items()},
        }
    )
    print(f"\nSaving -> {out_nc}")
    comp = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(out_nc, encoding=comp)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True)
    panels = [
        ("Lagrangian path (REF)", lag_v, 25),
        ("lin_budget", np.asarray(mr.values, float), 25),
        ("hydrostatic part", np.asarray(ds.melt_rate_hydro.values, float), 25),
        ("non-hydro correction", np.asarray(ds.nonhydro_corr.values, float), 8),
    ]
    for ax, (name, v, vmax) in zip(axes, panels):
        im = ax.imshow(v, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(
        "PIG 250 m is2ctempo — budget-corrected linear inverse vs production path solver "
        "(m ice/yr, negative = melt)", fontsize=11,
    )
    out_png = config.FIGURES_DIR / f"compare_budget_inverse_{OUT_TAG}.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_png}")
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
