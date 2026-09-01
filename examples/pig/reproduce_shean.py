"""Reproduce Shean et al. 2019's melt fits by running OUR solvers on HIS DEMs.

The point: our real-data melt field is noise-limited per-pixel (Eulerian p1/p99
~ +-500 m/yr from x9.4 hydrostatic amplification of coreg residuals on sparse
REMA strips), so the *integral* is tail-dominated and not a clean test of the
solver. Shean's distributed annual thickness mosaics (2008-2015, gap-filled,
512 m) are clean. Feeding them through our production Eulerian + Lagrangian
solvers isolates the solver from our data noise:

  if our solver on his DEMs -> ~82-93 Gt/yr, ~200-250 m/yr GL channels,
  10-30 m/yr main shelf  =>  the solver is sound and our noisy real-data
  integral is a coregistration/coverage problem, not a solver problem.

Method
------
His "_freeboard_thickness" is already firn-corrected ice-equivalent H, so we do
NOT re-apply firn. To reuse the production solvers (which take a *surface* stack
and invert hydrostatically with firn d), we map H -> pseudo-surface via
thickness_to_freeboard(H, d=0) and call the solvers with d=0; they invert it
straight back to H. Velocity = MEaSUREs (optionally Gaussian-smoothed to ~1 km
to match Shean's velocity smoothing, SHEAN_VEL_SMOOTH_KM). SMB = RACMO over
2008-2015. Domain = IceShelf_v02 Pine_Island polygon intersected with finite,
non-grounded (H < ceiling) Shean thickness.

Run:
    python -m pig.reproduce_shean
    SHEAN_VEL_SMOOTH_KM=0  python -m pig.reproduce_shean   # no velocity smoothing
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj
os.environ.setdefault("PIG_VELOCITY", "measures")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_erosion, gaussian_filter

from stereo_melt.freeboard import thickness_to_freeboard
from stereo_melt.io.smb import smb_over_window
from stereo_melt.kinematics import SECONDS_PER_YEAR
from stereo_melt.melt import eulerian_melt_rate, lagrangian_melt_rate
from stereo_melt.constants import rhoi, rhow

from pig import config
from pig.compare_shean2019 import _rasterize_shelf_polygon, _shean_stack
from pig.run_melt import load_floating_mask, load_velocity_on_grid

ALL_YEARS = (2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015)  # Shean's full record
H_CEILING_M = 1200.0  # drop grounded ice (PIG shelf H < ~700 m; >1200 m = grounded)
RHO_I = 918.0
CAP_MYR = 250.0  # Shean's max real channel melt; clip noise beyond this for a bracketed integral
# Defaults reproduce Shean's treatment: ~3.5 km velocity smoothing + a 2 km GL
# buffer. Without these the Eulerian finite-difference flux divergence overshoots
# near the grounding line (deepest -1300 m/yr, integral 140+ Gt/yr); with them
# the integral lands at 91-105 Gt/yr and Lagrangian channels reach -297 m/yr,
# both in/near Shean's published 82-93 Gt/yr and 200-250 m/yr.
SMOOTH_KM = float(os.environ.get("SHEAN_VEL_SMOOTH_KM", "3.5"))
ERODE_KM = float(os.environ.get("SHEAN_GL_BUFFER_KM", "2.0"))  # erode domain edges (GL + front)

# Shean 2019 published targets (TC 13:2633)
SHEAN = {"integrated_Gt_yr": "82-93", "channels_myr": "200-250", "main_shelf_myr": "10-30", "ns_shelf_myr": "0-10"}


def _smooth_velocity(v: xr.DataArray, sigma_px: float) -> xr.DataArray:
    """NaN-aware Gaussian smoothing of a velocity component."""
    a = v.values.astype(float)
    m = np.isfinite(a)
    a0 = np.where(m, a, 0.0)
    num = gaussian_filter(a0, sigma_px)
    den = gaussian_filter(m.astype(float), sigma_px)
    out = np.where(den > 1e-6, num / den, np.nan)
    out[~m] = np.nan  # keep original gaps as gaps
    return xr.DataArray(out, dims=v.dims, coords=v.coords)


def _integral_bracket(melt: xr.DataArray, dom: np.ndarray, dx: float, dy: float) -> dict:
    a = melt.values.copy()
    a[~dom] = np.nan
    v = a[np.isfinite(a)]
    if v.size == 0:
        return {}
    cell = dx * dy
    area_km2 = v.size * cell / 1e6
    med = float(np.median(v))
    raw = float(np.sum(-v) * cell * RHO_I / 1e12)
    clip = float(np.sum(-np.clip(v, -CAP_MYR, CAP_MYR)) * cell * RHO_I / 1e12)
    robust = float((-med) * area_km2 * 1e6 * RHO_I / 1e12)
    return {
        "area_km2": area_km2, "median_myr": med, "mean_myr": float(np.mean(v)),
        "p1": float(np.percentile(v, 1)), "p01": float(np.percentile(v, 0.1)),
        "max_melt_myr": float(np.min(v)),  # most negative = deepest melt
        "Gt_robust": -robust, "Gt_clip": clip, "Gt_raw": raw,
    }


def main() -> None:
    config.ensure_output_dirs()
    print("=" * 78)
    print("REPRODUCE SHEAN 2019 FITS  --  our solvers on HIS annual DEMs (2008-2015)")
    print("=" * 78)

    # grid + floating from the production product (cheap; avoids loading the big stack)
    prod = config.RESULTS_DIR / "pig_melt_250m_is2ctempo_2010-01-01_2024-01-10.nc"
    pd_ds = xr.open_dataset(prod)
    like = pd_ds["melt_rate_eulerian"]  # 2D (y,x) on our 250 m grid
    xs, ys = like["x"].values, like["y"].values
    dx, dy = abs(float(xs[1] - xs[0])), abs(float(ys[1] - ys[0]))
    print(f"[grid] {xs.size}x{ys.size} @ {dx:.0f} m (from {prod.name})")

    # Shean thickness stack, all 8 years, on our grid
    print(f"[shean] loading {len(ALL_YEARS)} annual thickness mosaics ...")
    H_stack = _shean_stack("thickness", like, years=ALL_YEARS)
    print(f"        H stack: t={H_stack.sizes['time']}  "
          f"{str(H_stack['time'].values.min())[:10]}..{str(H_stack['time'].values.max())[:10]}")
    H_mean = H_stack.mean("time", skipna=True)

    # domain: fixed shelf polygon AND finite, non-grounded Shean thickness
    poly = _rasterize_shelf_polygon(like)
    finite_nongrounded = np.isfinite(H_mean.values) & (H_mean.values < H_CEILING_M) & (H_mean.values > 5.0)
    if poly is not None:
        dom = poly.values.astype(bool) & finite_nongrounded
        dom_src = "IceShelf_v02 poly & finite-Shean-H<1200m"
    else:
        dom = load_floating_mask(like).values.astype(bool) & finite_nongrounded
        dom_src = "BedMachine floating & finite-Shean-H<1200m"
    if ERODE_KM > 0:
        ep = max(1, int(round(ERODE_KM * 1000.0 / dx)))
        dom = binary_erosion(dom, iterations=ep)
        dom_src += f" eroded {ERODE_KM:.1f}km ({ep}px) off GL+front"
    print(f"[domain] {dom_src}: {int(dom.sum())} px = {dom.sum()*dx*dy/1e6:.0f} km2")
    print(f"         Shean H over domain: median={np.nanmedian(H_mean.values[dom]):.0f} m  "
          f"IQR=[{np.nanpercentile(H_mean.values[dom],25):.0f},{np.nanpercentile(H_mean.values[dom],75):.0f}]")

    # pseudo-surface so production solvers invert straight back to Shean H (d=0)
    pseudo_h = thickness_to_freeboard(H_stack, d=0.0, rho_w=rhow, rho_i=rhoi)

    # velocity (MEaSUREs), optionally smoothed to ~Shean scale
    print(f"[vel] loading velocity ...")
    vx, vy, vel_source = load_velocity_on_grid(like)
    if SMOOTH_KM > 0:
        sig = SMOOTH_KM * 1000.0 / dx
        vx = _smooth_velocity(vx, sig).fillna(0.0)
        vy = _smooth_velocity(vy, sig).fillna(0.0)
        vel_source += f" + {SMOOTH_KM:.1f}km Gaussian"
    print(f"      {vel_source}")

    # SMB over Shean's window
    print(f"[smb] RACMO 2008-2015 ...")
    smb_cum = smb_over_window(str(config.RACMO_SMB_NC), xs, ys, start="2008-01-01", end="2015-12-31")
    dt_years = (pd.Timestamp("2015-12-31") - pd.Timestamp("2008-01-01")).total_seconds() / SECONDS_PER_YEAR
    a_dot = xr.DataArray(smb_cum / dt_years, dims=("y", "x"), coords={"y": ys, "x": xs})
    print(f"      a_dot median={float(a_dot.median()):.3f} m ice/yr")

    domx = xr.DataArray(dom, dims=("y", "x"), coords={"y": ys, "x": xs})

    # --- Eulerian (Shean Eq.10) on his DEMs ---
    print(f"\n[Eulerian] running on Shean DEMs (robust dh/dt) ...")
    eul = eulerian_melt_rate(pseudo_h, vx, vy, a_dot=a_dot, d=0.0, robust_dh_dt=True)
    eul_m = eul.melt_rate.where(domx)
    be = _integral_bracket(eul_m, dom, dx, dy)

    # --- Lagrangian (Shean Eq.7) on his DEMs ---
    print(f"[Lagrangian] running on Shean DEMs (path integration; 0.75-2.5 yr pairs) ...")
    lag = lagrangian_melt_rate(pseudo_h, vx, vy, a_dot=a_dot, d=0.0,
                               dt_yr=0.05, seed_stride=2, pairs="all",
                               min_dt_yr=0.75, max_dt_yr=2.5)
    lag_m = lag.melt_rate.where(domx)
    bl = _integral_bracket(lag_m, dom, dx, dy)

    # --- report vs Shean ---
    print("\n" + "=" * 78)
    print("RESULT  --  our solvers on Shean's DEMs   vs   Shean 2019 published")
    print("=" * 78)
    print(f"  {'quantity':22s} {'Eulerian':>14s} {'Lagrangian':>14s}   Shean published")
    print(f"  {'main-shelf median':22s} {be['median_myr']:+14.1f} {bl['median_myr']:+14.1f}   {SHEAN['main_shelf_myr']} m/yr")
    print(f"  {'deepest (channel)':22s} {be['max_melt_myr']:+14.0f} {bl['max_melt_myr']:+14.0f}   -{SHEAN['channels_myr']} m/yr")
    print(f"  {'p1 (high-melt tail)':22s} {be['p1']:+14.1f} {bl['p1']:+14.1f}")
    print(f"  {'mean':22s} {be['mean_myr']:+14.1f} {bl['mean_myr']:+14.1f}")
    print(f"  {'integral robust':22s} {be['Gt_robust']:14.1f} {bl['Gt_robust']:14.1f}   {SHEAN['integrated_Gt_yr']} Gt/yr")
    print(f"  {'integral clip|250|':22s} {be['Gt_clip']:14.1f} {bl['Gt_clip']:14.1f}   {SHEAN['integrated_Gt_yr']} Gt/yr")
    print(f"  {'integral raw':22s} {be['Gt_raw']:14.1f} {bl['Gt_raw']:14.1f}   {SHEAN['integrated_Gt_yr']} Gt/yr")
    print(f"  {'area':22s} {be['area_km2']:14.0f} {bl['area_km2']:14.0f}   km2")
    print("  (melt negative; Gt/yr positive=melt; CAP=250 m/yr = Shean's max real channel)")

    # --- save ---
    out_nc = config.RESULTS_DIR / "pig_reproduce_shean.nc"
    xr.Dataset(
        {"melt_eulerian_sheanDEM": eul_m, "melt_lagrangian_sheanDEM": lag_m,
         "shean_H_mean": H_mean.where(domx), "dHdt_eulerian": eul.dHdt.where(domx),
         "flux_div_eulerian": eul.flux_div.where(domx), "domain": domx},
        attrs={"desc": "our Eulerian+Lagrangian solvers on Shean 2019 annual DEMs 2008-2015",
               "velocity": vel_source, "smb": "RACMO2.4p1 2008-2015",
               "shean_published": str(SHEAN)},
    ).to_netcdf(out_nc)
    print(f"\n[save] {out_nc}")

    def ext(da):
        return [float(da["x"].min()), float(da["x"].max()), float(da["y"].min()), float(da["y"].max())]
    fig, ax = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    mv = dict(cmap="RdBu_r", vmin=-60, vmax=60, origin="upper", aspect="equal")
    im0 = ax[0].imshow(eul_m.values, extent=ext(eul_m), **mv); ax[0].set_title("Eulerian on Shean DEMs (m/yr)")
    im1 = ax[1].imshow(lag_m.values, extent=ext(lag_m), **mv); ax[1].set_title("Lagrangian on Shean DEMs (m/yr)")
    im2 = ax[2].imshow(H_mean.where(domx).values, extent=ext(H_mean), cmap="viridis", vmin=0, vmax=800,
                       origin="upper", aspect="equal"); ax[2].set_title("Shean thickness mean (m)")
    for a_, im in zip(ax, (im0, im1, im2)):
        fig.colorbar(im, ax=a_, fraction=0.045); a_.set_xlabel("x (m)")
    fig.suptitle(f"PIG melt: our solvers on Shean 2019 DEMs (2008-2015) | {vel_source}", fontsize=13)
    png = config.FIGURES_DIR / "reproduce_shean.png"
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[save] {png}")


if __name__ == "__main__":
    main()
