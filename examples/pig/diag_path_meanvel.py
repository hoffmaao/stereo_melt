"""Single-variable test: production path solver with TIME-MEAN ase-quarterly
velocity vs the time-varying production baseline.

Motivation (2026-07-03): the budget-corrected linear inverse (seed-attributed,
time-MEAN velocity) showed a ~1.6x anomaly stretch and +32 Gt/yr vs the
production path solver (time-VARYING quarterly velocity) on identical gates,
while the synthetic amplitude gate (tests/gate_flux_amplitude.py) reproduces
neither under a SHARED steady velocity (path dilution is mild, beta~0.85-0.93;
seed attribution adds no stretch). The one config variable left is velocity
time-variation: mean-u trajectories misplace 2-yr parcels by |dv|*dt ~ 150-500 m
across PIG's steep near-GL gradients, inflating per-fan slope variance where
melt is deep. This script isolates that variable with the TRUSTED production
code path: everything identical to the 2026-07-02 baseline except
vx/vy collapsed to the time mean.

Read: if IQR/flux inflate toward the linear inverse's numbers -> velocity
time-variation drives the A/B gap (fix: time-varying velocity in the inverse).
If it stays at the baseline -> the gap is in the inverse's internals.

Run:
    cd /wd2/projects/stereo_melt
    nohup /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u \
        -m pig.diag_path_meanvel > pig/logs/diag_path_meanvel.log 2>&1 &
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

os.environ.setdefault("PIG_VELOCITY", "ase-quarterly")

import time

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from stereo_melt.flux import grounding_buffer, integrate_basal_flux
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.melt import lagrangian_melt_rate

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
LININV_NC = (
    config.RESULTS_DIR
    / "pig_lininv_budget_250m_is2ctempo_2010-01-01_2024-01-10.nc"
)
OUT_NC = (
    config.RESULTS_DIR
    / f"pig_diag_path_meanvel_250m_is2ctempo_{config.START_TIME}_{config.END_TIME}.nc"
)


def nan_gauss(a, sp):
    if sp <= 0:
        return a
    m = np.isfinite(a)
    a0 = np.where(m, a, 0.0)
    num = gaussian_filter(a0, sp, mode="nearest")
    den = gaussian_filter(m.astype(float), sp, mode="nearest")
    return np.where(den > 0.05, num / np.maximum(den, 1e-9), np.nan)


def scorr(a, b, sp=0.0):
    aa, bb = nan_gauss(np.asarray(a, float), sp), nan_gauss(np.asarray(b, float), sp)
    f = np.isfinite(aa) & np.isfinite(bb)
    if f.sum() < 100:
        return np.nan
    av, bv = aa[f] - aa[f].mean(), bb[f] - bb[f].mean()
    d = np.sqrt((av**2).sum() * (bv**2).sum())
    return float((av * bv).sum() / d) if d > 0 else np.nan


def main() -> None:
    t0 = time.time()
    print("Loading stack...")
    stack = load_stack(stack_prefix="pig_stack_250m_is2ctempo")
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")
    floating = load_floating_mask(stack)
    stack = stack.where(floating)
    vx, vy, vel_source = load_velocity_on_grid(stack)
    if "time" not in vx.dims or vx.sizes.get("time", 1) < 2:
        raise SystemExit(f"expected time-varying velocity, got: {vel_source}")
    nt = vx.sizes["time"]
    vx_mean = vx.mean("time", skipna=True)
    vy_mean = vy.mean("time", skipna=True)
    print(f"  velocity: {vel_source}\n  -> COLLAPSED to time-mean over {nt} quarters")
    a_dot = load_smb_on_grid(stack)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    print("Running production path solver with TIME-MEAN velocity...")
    lagr = lagrangian_melt_rate(
        stack, vx_mean, vy_mean, a_dot=a_dot, d=firn, dt_yr=0.05,
        seed_stride=1, output="path", aggregator="pair_median",
        pairs="all", min_dt_yr=1.5, max_dt_yr=2.5,
        vdiv_clip=(float(os.environ.get("PIG_VDIV_CLIP", "0") or 0) or None),
    )
    mr = lagr.melt_rate
    mv = np.asarray(mr.values, float)
    print(
        f"  meanvel path: median={float(mr.median()):.2f}  "
        f"IQR=[{float(mr.quantile(0.25)):.2f}, {float(mr.quantile(0.75)):.2f}]  "
        f"cells={int(mr.notnull().sum()):,}  [{time.time() - t0:.0f}s]"
    )

    base = xr.open_dataset(BASELINE_NC)
    ref = np.asarray(base.melt_rate_lagrangian.values, float)
    step_count = np.nan_to_num(base.lagrangian_count.values, nan=0)
    lin_ds = xr.open_dataset(LININV_NC)
    lin = np.asarray(lin_ds.melt_rate.values, float)
    fan_count = np.nan_to_num(lin_ds["count"].values, nan=0).astype(int)
    fmask = np.asarray(base.floating_mask.values, bool)
    res_m = abs(float(stack["x"].values[1] - stack["x"].values[0]))

    print("\n=== vs baseline (time-varying velocity) ===")
    print(f"  REF(tv)   median={np.nanmedian(ref):.2f}  IQR=[{np.nanpercentile(ref,25):.2f}, {np.nanpercentile(ref,75):.2f}]")
    print(f"  corr(meanvel, tv): raw {scorr(mv, ref):.3f}  2km {scorr(mv, ref, 2000/res_m):.3f}  5km {scorr(mv, ref, 5000/res_m):.3f}")
    print(f"  corr(meanvel, lininv): raw {scorr(mv, lin):.3f}  2km {scorr(mv, lin, 2000/res_m):.3f}")

    print("\n=== binned by lininv fan count (floating, all three finite) ===")
    both = np.isfinite(mv) & np.isfinite(ref) & np.isfinite(lin) & fmask
    print(f"{'fans':>7} {'n':>7} {'medMV':>8} {'q25MV':>8} {'medTV':>8} {'q25TV':>8} {'medLIN':>8} {'q25LIN':>8} {'IQRw MV':>8} {'IQRw TV':>8}")
    for lo, hi in ((1, 2), (3, 5), (6, 9), (10, 19), (20, 999)):
        m = both & (fan_count >= lo) & (fan_count <= hi)
        if m.sum() < 100:
            continue
        M, T, L = mv[m], ref[m], lin[m]
        iqM = np.percentile(M, 75) - np.percentile(M, 25)
        iqT = np.percentile(T, 75) - np.percentile(T, 25)
        print(f"{lo:>3}-{hi:<3} {M.size:>7} {np.median(M):>8.2f} {np.percentile(M,25):>8.2f} "
              f"{np.median(T):>8.2f} {np.percentile(T,25):>8.2f} "
              f"{np.median(L):>8.2f} {np.percentile(L,25):>8.2f} {iqM:>8.1f} {iqT:>8.1f}")

    print("\n=== GL-2km clip flux ===")
    grounded = load_grounded_mask(stack)
    dom = grounding_buffer(fmask, np.asarray(grounded.values, bool), 2000.0, res_m)
    my_count = np.nan_to_num(lagr["count"].values, nan=0)
    for name, field, q in (
        ("REF tv (baseline)", ref, step_count >= 10),
        ("path mean-vel", mv, my_count >= 10),
        ("path mean-vel|tv-gate", mv, step_count >= 10),
        ("lininv (context)", lin, step_count >= 10),
    ):
        b = integrate_basal_flux(field, dom, res_m, quality=q)
        print(f"  {name:<22} area={b['area_km2']:7.0f} km2  med={b['median_myr']:+6.2f}  "
              f"CLIP={b['gt_clip']:.1f}  raw={b['gt_raw']:.1f}  robust={b['gt_robust']:.1f}")

    out = lagr[["melt_rate", "count", "rmse"]].copy()
    out.attrs.update({
        "note": "production path solver with TIME-MEAN ase-quarterly velocity "
                "(single-variable diagnostic vs time-varying baseline)",
        "velocity_source": vel_source + " -> time-mean",
        "baseline": BASELINE_NC.name,
    })
    comp = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(OUT_NC, encoding=comp)
    print(f"\nSaved -> {OUT_NC}\nDONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
