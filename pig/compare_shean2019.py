"""Spatial A/B of our PIG product against Shean et al. 2019 gridded data.

Shean's distributed product (the two zips the user uploaded, unpacked to
``data/Shean2019/512m_annual_interp/``) is NOT his final melt grid -- it is his
annual DEM mosaics plus the derived hydrostatic fields, one set per water year
2008--2015:

    <YYYY>0101_<prev>0701-<YYYY>0630_mos-tile-0.tif                 surface DEM
    ..._mos-tile-0_freeboard_thickness.tif                          ice thickness H_f
    ..._mos-tile-0_bottom_elev.tif                                  ice draft

All EPSG:3031, 512 m, NoData 1e20 (surface) / -9999 (derived). This script does
the two comparisons that data supports:

PART A -- data-level A/B (no velocity, no solver):
    his annual mean surface  vs ours          -> vertical-datum / DEM check
    his annual mean thickness vs our H_f       -> hydrostatic-input check
  over the 2010--2015 overlap, on our 250 m grid, over the floating shelf.

PART B -- reconstruct melt from HIS thickness with OUR Eulerian solver:
    dH/dt (his annual H series) + div(H_his u) - a_dot   (Shean Eq. 10)
  with MEaSUREs velocity + RACMO SMB, integrated over (1) the present
  BedMachine floating mask and (2) the fixed IceShelf_v02 Pine_Island polygon.
  Same solver run on OUR thickness over the same era/grid/velocity is the
  controlled A/B; the published target is 82--93 Gt/yr.

  Caveat logged in the report: Shean's *published* melt is Lagrangian from strip
  pairs; differencing his annual mosaics is the matching Eulerian estimator
  (same continuity, coarser channel detail).

Run:
    python -m pig.compare_shean2019
"""

from __future__ import annotations

import os
import sys

# PROJ shim (mirror run_melt) -- must precede any pyproj import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj
os.environ.setdefault("PIG_VELOCITY", "measures")  # multiyear mosaic for 2008-2015

import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers .rio accessor + open_rasterio)
import xarray as xr

from stereo_melt.freeboard import freeboard_to_thickness
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.io.smb import smb_over_window
from stereo_melt.kinematics import SECONDS_PER_YEAR, dh_dt, flux_divergence
from stereo_melt.stack import load_basin_stack

from pig import config
from pig.run_melt import load_floating_mask, load_velocity_on_grid

SHEAN_DIR = config.MAIN_DIR / "data" / "Shean2019" / "512m_annual_interp"
OVERLAP_YEARS = (2010, 2011, 2012, 2013, 2014, 2015)  # Shean years overlapping our record
STACK_PREFIX = "pig_stack_250m_is2ctempo"
RHO_I = 918.0  # kg/m^3, matches stereo_melt.constants.rhoi
OUT_DIR = config.FIGURES_DIR
NC_OUT = config.RESULTS_DIR / "pig_shean2019_comparison.nc"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def _robust_stats(diff: np.ndarray, label: str) -> dict:
    d = diff[np.isfinite(diff)]
    if d.size == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": int(d.size),
        "median": float(np.median(d)),
        "mad": _mad(d),
        "mean": float(np.mean(d)),
        "rms": float(np.sqrt(np.mean(d**2))),
        "p5": float(np.percentile(d, 5)),
        "p95": float(np.percentile(d, 95)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def _shean_year_path(year: int, kind: str) -> str | None:
    """kind in {'surface','thickness','draft'}."""
    suffix = {
        "surface": "_mos-tile-0.tif",
        "thickness": "_mos-tile-0_freeboard_thickness.tif",
        "draft": "_mos-tile-0_bottom_elev.tif",
    }[kind]
    hits = glob.glob(str(SHEAN_DIR / f"{year}0101_*{suffix}"))
    # exclude the derived-product names when asking for the bare surface
    if kind == "surface":
        hits = [h for h in hits if h.endswith("_mos-tile-0.tif")]
    return hits[0] if hits else None


def _load_shean_on_grid(year: int, kind: str, like: xr.DataArray) -> xr.DataArray | None:
    p = _shean_year_path(year, kind)
    if p is None:
        return None
    da = rioxarray.open_rasterio(p, masked=True).squeeze("band", drop=True)
    # same CRS (EPSG:3031) -> bilinear interp onto our grid; NoData already NaN.
    out = da.interp(x=like["x"], y=like["y"], method="linear")
    return out.reset_coords(drop=True)


def _shean_mean(kind: str, like: xr.DataArray, years=OVERLAP_YEARS) -> tuple[xr.DataArray, list[int]]:
    grids, got = [], []
    for y in years:
        g = _load_shean_on_grid(y, kind, like)
        if g is not None:
            grids.append(g)
            got.append(y)
    if not grids:
        raise SystemExit(f"no Shean {kind} grids found for {years} in {SHEAN_DIR}")
    stacked = xr.concat(grids, dim="syear").assign_coords(syear=got)
    return stacked.mean("syear", skipna=True), got


def _shean_stack(kind: str, like: xr.DataArray, years=OVERLAP_YEARS) -> xr.DataArray:
    """Time-stacked Shean field for dH/dt (mid-water-year timestamps)."""
    grids, tcoords = [], []
    for y in years:
        g = _load_shean_on_grid(y, kind, like)
        if g is not None:
            grids.append(g)
            tcoords.append(pd.Timestamp(f"{y - 1}-12-31"))  # mosaic centre ~ Jan 1 of label year
    da = xr.concat(grids, dim="time").assign_coords(time=pd.DatetimeIndex(tcoords))
    return da.reset_coords(drop=True)


def _integrate_gt(melt: xr.DataArray, domain: xr.DataArray, dx: float, dy: float) -> dict:
    """melt (m ice/yr, negative=melt) over boolean `domain` -> integrated Gt/yr (melt positive)."""
    m = melt.where(domain)
    arr = m.values
    fin = np.isfinite(arr)
    n = int(fin.sum())
    area_km2 = n * dx * dy / 1e6
    if n == 0:
        return {"n": 0, "area_km2": 0.0, "mean_myr": np.nan, "gt_per_yr": np.nan}
    mean_myr = float(np.nanmean(arr))  # negative = melt
    vol_m3 = float(np.nansum(-arr)) * dx * dy  # m^3 ice/yr, positive = melt
    gt = vol_m3 * RHO_I / 1e12
    return {"n": n, "area_km2": area_km2, "mean_myr": mean_myr, "gt_per_yr": gt}


def _rasterize_shelf_polygon(like: xr.DataArray) -> xr.DataArray | None:
    """IceShelf_v02 Pine_Island feature -> boolean mask on `like` grid."""
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
    except Exception as exc:  # pragma: no cover
        print(f"  [shelf-polygon] geopandas/rasterio unavailable ({exc}); skipping domain-2")
        return None
    if not config.ICESHELF_FEATURE_SHP.exists():
        print(f"  [shelf-polygon] {config.ICESHELF_FEATURE_SHP} missing; skipping domain-2")
        return None
    gdf = gpd.read_file(config.ICESHELF_FEATURE_SHP)
    name_col = next((c for c in gdf.columns if gdf[c].astype(str).str.contains("Pine_Island").any()), None)
    if name_col is None:
        print("  [shelf-polygon] Pine_Island feature not found; skipping domain-2")
        return None
    feat = gdf[gdf[name_col].astype(str) == config.ICESHELF_FEATURE_NAME]
    if feat.empty:
        feat = gdf[gdf[name_col].astype(str).str.contains("Pine_Island")]
    feat = feat.to_crs("EPSG:3031")
    xs = like["x"].values
    ys = like["y"].values
    dx = abs(float(xs[1] - xs[0]))
    dy = abs(float(ys[1] - ys[0]))
    transform = from_origin(xs.min() - dx / 2, ys.max() + dy / 2, dx, dy)
    mask = rasterize(
        [(geom, 1) for geom in feat.geometry],
        out_shape=(ys.size, xs.size),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    # rasterize assumes north-up rows (y descending from top); our ys descend -> ok
    return xr.DataArray(mask.astype(bool), dims=("y", "x"), coords={"y": ys, "x": xs})


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    config.ensure_output_dirs()
    report: list[str] = []

    def say(s=""):
        print(s)
        report.append(s)

    say("=" * 78)
    say("SHEAN 2019 SPATIAL A/B  (PIG, 2010-2015 overlap)")
    say("=" * 78)

    # --- load our stack (is2ctempo tilt-corrected) ---
    say("\n[load] our is2ctempo tilt-corrected stack ...")
    stack, stack_path = load_basin_stack(
        config.PROCESSED_DIR,
        STACK_PREFIX,
        config.START_TIME,
        config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    say(f"       {Path(stack_path).name}  dims t={stack.sizes['time']} y={stack.sizes['y']} x={stack.sizes['x']}")
    xs, ys = stack["x"].values, stack["y"].values
    dx, dy = abs(float(xs[1] - xs[0])), abs(float(ys[1] - ys[0]))
    say(f"       grid res dx={dx:.0f} m dy={dy:.0f} m")

    # subset to the Shean overlap window
    sub = stack.sel(time=slice("2010-01-01", "2015-12-31"))
    say(f"       overlap subset 2010..2015: t={sub.sizes['time']} epochs")

    # --- masks ---
    floating = load_floating_mask(stack).astype(bool)
    say(f"       BedMachine floating fraction of AOI: {float(floating.mean()):.3f}")
    shelf_poly = _rasterize_shelf_polygon(stack)
    if shelf_poly is not None:
        say(f"       IceShelf_v02 Pine_Island polygon fraction of AOI: {float(shelf_poly.mean()):.3f}")

    # --- firn for thickness conversion ---
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    # ==================================================================
    # PART A -- data-level A/B
    # ==================================================================
    say("\n" + "-" * 78)
    say("PART A  --  thickness & surface, ours vs Shean, over floating shelf")
    say("-" * 78)

    shean_surf, ys_surf = _shean_mean("surface", stack)
    shean_thk, ys_thk = _shean_mean("thickness", stack)
    say(f"  Shean years used: surface={ys_surf}  thickness={ys_thk}")

    our_surf = sub.mean("time", skipna=True)
    H_f_sub = freeboard_to_thickness(sub, d=firn)
    our_thk = H_f_sub.mean("time", skipna=True)

    dom = floating.values
    surf_diff = (our_surf - shean_surf).where(floating)
    thk_diff = (our_thk - shean_thk).where(floating)

    s_surf = _robust_stats(surf_diff.values[dom], "surface ours-Shean (m)")
    s_thk = _robust_stats(thk_diff.values[dom], "thickness ours-Shean (m)")
    r_surf, n_surf = _corr(our_surf.where(floating).values, shean_surf.where(floating).values)
    r_thk, n_thk = _corr(our_thk.where(floating).values, shean_thk.where(floating).values)

    for s in (s_surf, s_thk):
        if s.get("n", 0):
            say(
                f"  {s['label']:32s} n={s['n']:6d}  median={s['median']:+8.2f}  "
                f"MAD={s['mad']:7.2f}  RMS={s['rms']:7.2f}  p5/p95={s['p5']:+7.1f}/{s['p95']:+7.1f}"
            )
    say(f"  spatial corr (floating): surface r={r_surf:.3f} (n={n_surf})   thickness r={r_thk:.3f} (n={n_thk})")
    say("  NOTE surface median diff ~ vertical-datum/geoid offset (his mos-tile may be ellipsoid);")
    say("       thickness is the datum-robust comparison (both freeboard-derived, x~9.3-9.4 gain).")

    # ==================================================================
    # PART B -- reconstruct melt from HIS thickness, our Eulerian solver
    # ==================================================================
    say("\n" + "-" * 78)
    say("PART B  --  Eulerian melt from Shean's annual thickness vs from ours")
    say("           m = dH/dt + div(H u) - a_dot   (Shean Eq.10, neg=melt)")
    say("-" * 78)

    say("  [vel] loading velocity (MEaSUREs multiyear) ...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    say(f"        {vel_source}")

    say("  [smb] integrating RACMO over 2010-2015 ...")
    smb_cum = smb_over_window(str(config.RACMO_SMB_NC), xs, ys, start="2010-01-01", end="2015-12-31", method="linear")
    dt_years = (pd.Timestamp("2015-12-31") - pd.Timestamp("2010-01-01")).total_seconds() / SECONDS_PER_YEAR
    a_dot = xr.DataArray(smb_cum / dt_years, dims=("y", "x"), coords={"y": ys, "x": xs})
    say(f"        a_dot median={float(a_dot.median()):.3f} m ice/yr")

    def eulerian_from_H(H_stack: xr.DataArray, label: str) -> xr.DataArray:
        reg = dh_dt(H_stack, min_count=3, robust=True)
        dHdt = reg["slope"] * SECONDS_PER_YEAR
        H_mean = H_stack.mean("time", skipna=True)
        fd = flux_divergence(H_mean, vx, vy)
        melt = dHdt + fd - a_dot
        say(
            f"  [{label}] dH/dt med={float(dHdt.median()):+.3f}  div(Hu) med={float(fd.median()):+.3f}  "
            f"-> melt med={float(melt.where(floating).median()):+.3f} m/yr (floating)"
        )
        return melt

    # his thickness stack (annual) and our thickness stack (the same primitive
    # the production Eulerian solver uses internally), both -> Eulerian melt.
    shean_H_stack = _shean_stack("thickness", stack)
    say(f"  Shean thickness stack: t={shean_H_stack.sizes['time']} annual epochs")
    melt_shean = eulerian_from_H(shean_H_stack, "Shean-H")
    melt_ours = eulerian_from_H(H_f_sub, "our-H ")

    # integrate on both domains
    say("\n  integrated basal melt (positive = melt):")
    say(f"  {'domain':28s} {'source':9s} {'area_km2':>9s} {'mean_m/yr':>10s} {'Gt/yr':>9s}")
    domains = [("BedMachine floating (present)", floating)]
    if shelf_poly is not None:
        domains.append(("IceShelf_v02 Pine_Island (fixed)", shelf_poly))
    rows = {}
    for dname, dmask in domains:
        for src, melt in (("Shean-H", melt_shean), ("our-H", melt_ours)):
            r = _integrate_gt(melt, dmask, dx, dy)
            rows[(dname, src)] = r
            say(
                f"  {dname:28s} {src:9s} {r['area_km2']:9.0f} {r['mean_myr']:+10.2f} "
                f"{(-r['gt_per_yr']) if np.isfinite(r['gt_per_yr']) else float('nan'):9.1f}"
            )
    say("  (Gt/yr printed positive=melt; Shean published 82-93 Gt/yr over the full PIG shelf)")
    say("  CAVEAT: present BedMachine floating mask is post-calving (smaller) than Shean's")
    say("          2008-2015 shelf -> compare mean m/yr + pattern; use fixed polygon for Gt/yr.")

    # spatial agreement of the two reconstructed melt fields
    for dname, dmask in domains:
        a = melt_shean.where(dmask).values
        b = melt_ours.where(dmask).values
        r, n = _corr(a, b)
        ms = _robust_stats((melt_ours - melt_shean).where(dmask).values, f"melt our-Shean ({dname})")
        say(
            f"  [{dname}] melt corr r={r:.3f} (n={n})  "
            f"our-Shean median={ms.get('median', float('nan')):+.2f} MAD={ms.get('mad', float('nan')):.2f} m/yr"
        )

    # ==================================================================
    # save NetCDF + figure
    # ==================================================================
    ds = xr.Dataset(
        {
            "shean_surface": shean_surf.where(floating),
            "our_surface": our_surf.where(floating),
            "surface_diff": surf_diff,
            "shean_thickness": shean_thk.where(floating),
            "our_thickness": our_thk.where(floating),
            "thickness_diff": thk_diff,
            "melt_shean_H": melt_shean.where(floating),
            "melt_our_H": melt_ours.where(floating),
            "floating_mask": floating,
        },
        attrs={
            "comparison": "PIG ours vs Shean et al. 2019 (TC 13:2633), 2010-2015 overlap",
            "shean_source": "512m_annual_interp (annual DEM mosaics + hydrostatic fields)",
            "shean_published_integrated_Gt_per_yr": "82-93",
            "note": "Part B Eulerian melt from Shean annual thickness vs ours; MEaSUREs vel, RACMO SMB",
        },
    )
    if shelf_poly is not None:
        ds["shelf_polygon_mask"] = shelf_poly
    NC_OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(NC_OUT)
    say(f"\n[save] {NC_OUT}")

    _plot(shean_surf, our_surf, surf_diff, shean_thk, our_thk, thk_diff,
          melt_shean, melt_ours, floating, OUT_DIR / "shean2019_comparison.png")
    say(f"[save] {OUT_DIR / 'shean2019_comparison.png'}")

    (config.RESULTS_DIR / "pig_shean2019_comparison.txt").write_text("\n".join(report) + "\n")
    say(f"[save] {config.RESULTS_DIR / 'pig_shean2019_comparison.txt'}")


def _plot(shean_surf, our_surf, surf_diff, shean_thk, our_thk, thk_diff,
          melt_shean, melt_ours, floating, out_path: Path) -> None:
    def ext(da):
        return [float(da["x"].min()), float(da["x"].max()), float(da["y"].min()), float(da["y"].max())]

    def show(ax, da, **kw):
        return ax.imshow(da.where(floating).values, extent=ext(da), origin="upper", aspect="equal", **kw)

    fig, ax = plt.subplots(3, 3, figsize=(16, 15), constrained_layout=True)
    # surfaces
    sv = dict(cmap="terrain", vmin=0, vmax=120)
    fig.colorbar(show(ax[0, 0], shean_surf, **sv), ax=ax[0, 0], fraction=0.045); ax[0, 0].set_title("Shean surface (m)")
    fig.colorbar(show(ax[0, 1], our_surf, **sv), ax=ax[0, 1], fraction=0.045); ax[0, 1].set_title("our surface (m)")
    fig.colorbar(show(ax[0, 2], surf_diff, cmap="RdBu_r", vmin=-30, vmax=30), ax=ax[0, 2], fraction=0.045)
    ax[0, 2].set_title("surface ours-Shean (m)")
    # thickness
    tv = dict(cmap="viridis", vmin=0, vmax=800)
    fig.colorbar(show(ax[1, 0], shean_thk, **tv), ax=ax[1, 0], fraction=0.045); ax[1, 0].set_title("Shean thickness (m)")
    fig.colorbar(show(ax[1, 1], our_thk, **tv), ax=ax[1, 1], fraction=0.045); ax[1, 1].set_title("our thickness (m)")
    fig.colorbar(show(ax[1, 2], thk_diff, cmap="RdBu_r", vmin=-80, vmax=80), ax=ax[1, 2], fraction=0.045)
    ax[1, 2].set_title("thickness ours-Shean (m)")
    # melt
    mv = dict(cmap="RdBu_r", vmin=-60, vmax=60)
    fig.colorbar(show(ax[2, 0], melt_shean, **mv), ax=ax[2, 0], fraction=0.045); ax[2, 0].set_title("melt from Shean-H (m/yr)")
    fig.colorbar(show(ax[2, 1], melt_ours, **mv), ax=ax[2, 1], fraction=0.045); ax[2, 1].set_title("melt from our-H (m/yr)")
    fig.colorbar(show(ax[2, 2], (melt_ours - melt_shean), cmap="PuOr", vmin=-20, vmax=20), ax=ax[2, 2], fraction=0.045)
    ax[2, 2].set_title("melt our-Shean (m/yr)")
    for a in ax.ravel():
        a.set_xlabel("x (m)")
    fig.suptitle("PIG: ours vs Shean et al. 2019 (2010-2015 overlap)", fontsize=14)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
