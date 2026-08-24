"""PIG: the bridging-aware melt solvers on the production DEM stack.

Runs, on EXACTLY the inputs of :mod:`pig.run_melt` (tilt-corrected stack,
fused/production velocity, RACMO SMB, BedMachine firn, min-extent floating
mask):

* ``eulerian``            the production reference (Shean Eq. 10, robust dh/dt)
* ``eulerian_helm``       + the mass-consistent Helmholtz flux divergence
* ``restored``            restore-then-budget, global H_ref
* ``restored_local``      restore-then-budget, per-(H, u) local filters,
                          trunk-guarded (no lift where u > ``--lift-umax-myr``)
* ``restored_local_helm`` + the Helmholtz divergence (the 2026-08-24
                          headline: 90.4 Gt/yr, fused velocity)
* ``monolithic``          budget+bridging fit @ the ML-II lam, local bins,
                          PIG eta field, Helmholtz divergence, and the
                          CORRECTED forward model (``flux_restored``,
                          "monolithic v2"); ``--mono-v1`` reproduces the
                          historical placement of the flux divergence

and writes one NetCDF of melt maps + a figure (LADDIE symlog maps, differences
vs the Eulerian, and shelf-integrated fluxes). The point (2026-08-24): the
twins say the additions only matter in the across-flow lambda <~ 3H band —
this run measures whether that band changes PIG's melt at all.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    OMP_NUM_THREADS=14 nohup $PY -u -m pig.run_melt_bridging > pig/logs/run_melt_bridging.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/wd2/projects/stereo_melt")
sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")

from stereo_melt import envsetup  # noqa: F401,E402  (PROJ fix first)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from pig import config  # noqa: E402
from pig.run_melt import (  # noqa: E402
    apply_min_extent,
    load_floating_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
)
from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm  # noqa: E402
from stereo_melt.dynamics.bridging_restoration import restored_budget_melt_rate  # noqa: E402
from stereo_melt.dynamics.budget_bridging import budget_bridging_melt_rate  # noqa: E402
from stereo_melt.io.bedmachine import load_firn_on_grid  # noqa: E402
from stereo_melt.kinematics import HelmholtzDivergence  # noqa: E402
from stereo_melt.melt import eulerian_melt_rate  # noqa: E402

RHO_I = 918.0  # stereo_melt.constants
ETA_NPZ = Path("/wd2/projects/stereo_melt/pig/processed/"
               "pig_eta_field_250m_dual_20260730_t0era5.npz")
ML2_LAM = 3.2e-2   # the ML-II pick on the survey-realistic rungs


def gt_per_yr(melt: xr.DataArray, floating: xr.DataArray, res_m: float) -> float:
    """Shelf-integrated melt flux (Gt/yr, positive = mass loss to melt)."""
    m = melt.where(floating).values
    return float(-np.nansum(m) * res_m * res_m * RHO_I / 1e12)


def main() -> int:
    # Production configuration by default: this driver exists to reproduce the
    # production bridging products, and production velocity is the Kalman-EOF
    # fusion (pig/CLAUDE.md). PIG_VELOCITY in the environment still overrides.
    os.environ.setdefault("PIG_VELOCITY", "fused")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=float, default=250.0)
    ap.add_argument("--tag", default="is2ctempo_sheltilt")
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--lift-umax-myr", type=float, default=1500.0,
                    help="trunk guard: bins faster than this get no lift (0 = off)")
    ap.add_argument("--mono-iters", type=int, default=300)
    ap.add_argument("--mono-v1", action="store_true",
                    help="historical monolithic forward model (flux from observed H)")
    ap.add_argument("--skip-mono", action="store_true")
    args = ap.parse_args()

    t00 = time.time()
    file_start, file_end = str(config.START_TIME), str(config.END_TIME)
    res_i = int(round(args.res))
    stack_prefix = f"pig_stack_{res_i}m_{args.tag}"
    print(f"[inputs] stack {stack_prefix} tilt_corrected "
          f"{file_start}..{file_end}", flush=True)
    stack = load_stack(stack_prefix)

    floating = load_floating_mask(stack)
    mask_suffix = f"_{res_i}m_{args.tag}"
    floating = apply_min_extent(floating, mask_suffix, file_start, file_end)
    stack = stack.where(floating)
    print(f"  floating cells: {int(floating.sum())}", flush=True)

    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  velocity: {vel_source}", flush=True)
    a_dot = load_smb_on_grid(stack)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    out = {}
    fluxes = {}

    def record(name, ds, t0):
        m = ds.melt_rate if hasattr(ds, "melt_rate") else ds
        out[name] = m
        fluxes[name] = gt_per_yr(m, floating, args.res)
        med = float(m.where(floating).median())
        print(f"  [{name}] median {med:+.2f} m/yr  shelf flux {fluxes[name]:.1f} Gt/yr  "
              f"({time.time() - t0:.0f}s)", flush=True)

    print("[1/6] Eulerian (production reference)...", flush=True)
    t0 = time.time()
    eul = eulerian_melt_rate(stack, vx, vy, a_dot=a_dot, d=firn, robust_dh_dt=True)
    record("eulerian", eul, t0)

    print("[2/6] Eulerian + Helmholtz divergence...", flush=True)
    t0 = time.time()
    eul_h = eulerian_melt_rate(stack, vx, vy, a_dot=a_dot, d=firn, robust_dh_dt=True,
                               estimator=HelmholtzDivergence())
    record("eulerian_helm", eul_h, t0)

    print("[3/6] restored budget (global H_ref)...", flush=True)
    t0 = time.time()
    rb = restored_budget_melt_rate(stack, vx, vy, a_dot=a_dot, floating_mask=floating,
                                   d=firn, robust_dh_dt=True)
    print(f"    H_ref {rb.attrs['H_ref_m']:.0f} m  u0 ({rb.attrs['u0x_myr']:.0f}, "
          f"{rb.attrs['u0y_myr']:.0f}) m/yr  band {rb.attrs['band_lam_min_m']:.0f} m",
          flush=True)
    record("restored", rb, t0)

    guard = args.lift_umax_myr if args.lift_umax_myr > 0 else None
    print(f"[4/6] restored budget (local, {args.n_bins} bins, guard {guard})...",
          flush=True)
    t0 = time.time()
    rbl = restored_budget_melt_rate(stack, vx, vy, a_dot=a_dot, floating_mask=floating,
                                    d=firn, robust_dh_dt=True, n_bins=args.n_bins,
                                    lift_umax_myr=guard)
    print(f"    bins: {rbl.attrs['bin_geometry']}", flush=True)
    record("restored_local", rbl, t0)

    print("[5/6] restored budget (local) + Helmholtz divergence...", flush=True)
    t0 = time.time()
    rblh = restored_budget_melt_rate(stack, vx, vy, a_dot=a_dot, floating_mask=floating,
                                     d=firn, robust_dh_dt=True, n_bins=args.n_bins,
                                     lift_umax_myr=guard,
                                     estimator=HelmholtzDivergence())
    record("restored_local_helm", rblh, t0)

    if not args.skip_mono:
        print(f"[6/6] monolithic @ lam {ML2_LAM:g}, {args.n_bins} bins, eta field, "
              f"Helmholtz...", flush=True)
        t0 = time.time()
        eta_field = None
        if ETA_NPZ.exists():
            z = np.load(ETA_NPZ)
            eta_field = xr.DataArray(z["eta_pas"], dims=("y", "x"),
                                     coords={"y": z["y"], "x": z["x"]})
            eta_field = eta_field.reindex_like(stack.isel(time=0), method="nearest")
            print(f"    eta field: median {float(eta_field.median()):.2e} Pa s", flush=True)
        mono = budget_bridging_melt_rate(
            stack, vx, vy, a_dot=a_dot, floating_mask=floating, d=firn,
            robust_dh_dt=True, bridging=True, transfer="flotation",
            flux_restored=not args.mono_v1,
            restore_kwargs=(None if args.mono_v1
                            else dict(lift_umax_myr=guard)),
            eta_field=eta_field, n_bins=args.n_bins, lam=ML2_LAM,
            iters=args.mono_iters, converge_tol=1e-9, log_every=50,
            estimator=HelmholtzDivergence())
        record("monolithic" if args.mono_v1 else "monolithic_v2", mono, t0)

    # ------------------------------------------------------------ outputs
    win = f"{file_start}_{file_end}"
    ds_out = xr.Dataset({k: v for k, v in out.items()})
    ds_out.attrs.update(velocity=vel_source, tag=args.tag, res_m=args.res,
                        min_extent_mask=floating.attrs["min_extent_mask"],
                        ml2_lam=ML2_LAM, n_bins=args.n_bins,
                        fluxes_gt_yr=";".join(f"{k}={v:.2f}" for k, v in fluxes.items()))
    out_nc = config.PROCESSED_DIR / f"pig_melt_bridging_{res_i}m_{args.tag}_{win}.nc"
    ds_out.to_netcdf(out_nc)
    print(f"wrote {out_nc}", flush=True)

    names = list(out)
    cmap, norm = melt_cmap(), melt_norm(vmax=100.0)
    n = len(names)
    fig, axs = plt.subplots(2, n, figsize=(3.1 * n + 1.5, 9.5), squeeze=False)
    im = imd = None
    x, y = stack.x.values / 1e3, stack.y.values / 1e3
    for j, name in enumerate(names):
        ax = axs[0, j]
        im = ax.pcolormesh(x, y, out[name].where(floating).values, cmap=cmap,
                           norm=norm, shading="nearest", rasterized=True)
        ax.set_title(f"{name}\n{fluxes[name]:.1f} Gt/yr", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axs[1, j]
        if name == "eulerian":
            ax.set_axis_off()
            ax.text(0.5, 0.5, "reference", transform=ax.transAxes, ha="center")
            continue
        dmap = (out[name] - out["eulerian"]).where(floating).values
        imd = ax.pcolormesh(x, y, dmap, cmap="RdBu_r", vmin=-10, vmax=10,
                            shading="nearest", rasterized=True)
        rms = float(np.sqrt(np.nanmean(dmap ** 2)))
        ax.set_title(f"Δ vs Eulerian  rms {rms:.2f} m/yr\n"
                     f"Δflux {fluxes[name] - fluxes['eulerian']:+.1f} Gt/yr", fontsize=8.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    add_melt_colorbar(fig, im, ax=axs[0].tolist(), shrink=0.85, pad=0.01,
                      label="ḃ (m ice a⁻¹)")
    fig.colorbar(imd, ax=axs[1].tolist(), shrink=0.85, pad=0.01,
                 label="Δ vs Eulerian (m a⁻¹)")
    fig.suptitle(f"PIG {res_i} m {args.tag} {win} — bridging-aware melt solvers on the "
                 f"production stack ({vel_source})", fontsize=11)
    out_png = config.FIGURES_DIR / f"melt_bridging_{res_i}m_{args.tag}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}", flush=True)
    print("[done] fluxes (Gt/yr): " + "  ".join(f"{k} {v:.1f}" for k, v in fluxes.items()))
    print(f"[done] total {time.time() - t00:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
