"""Beardmore_Shelf solver suite: Eulerian + the TWO budget linear inverses
vs the production Lagrangian path solver.

Port of ``pig/compare_budget_inverse.py`` extended for the 2026-07-10 ask:
the production path-melt map's under-constrained regions (the RED accretion
stripe, cols ~225-300, and the far grid-WEST band, cols 0-90) coincide with
collapsed path-deposit counts — few 1.5-2.5 yr DEM pairs constrain those
cells, so the cross-pair median cannot reject strip-datum biases there. This
driver runs, on the SAME curated stack the production product used:

1. the production EULERIAN estimate (``eulerian_melt_rate``, Tukey-robust
   dh/dt — PIG production config): per-cell regression over ALL epochs, a
   different (denser) data model than the pair band;
2. the PATH-integrated budget linear inverse
   (:func:`stereo_melt.dynamics.linear_inverse_budget_melt_rate`,
   ``attribution="path"``, pooled-pairs, trend-H — the PIG ``_tv_path_dfix``
   production config);
3. the EULERIAN budget linear inverse
   (:func:`stereo_melt.dynamics.linear_inverse_eulerian_budget_melt_rate`):
   the Eulerian estimate is its hydrostatic channel EXACTLY, plus the same
   band-limited non-hydrostatic kernel correction the path inverse uses.

and compares all of them against the saved production path REF — spatial
correlation (raw / 2 km / 5 km), distributions, GL-2 km clip flux, and
per-band statistics inside the two under-constrained bands (reusing
``diag_stripe_strips.BANDS``) with each estimator's own count semantics
(path deposits vs banded pairs vs epochs).

Velocity is the static MEaSUREs phase map (NSIDC-0754) — the budget inverse
takes its static path (bit-identical to constant time-varying input).

Run (disconnect-safe):

    cd /wd2/projects/stereo_melt/examples
    nohup /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u \
        -m beardmore_shelf.compare_budget_inverse \
        > beardmore_shelf/logs/compare_budget_inverse_125m.log 2>&1 &

Env knobs (PIG parity): ``LININV_REG`` (0.1), ``LININV_ETA_BAR`` (1e14),
``LININV_CORR_SIGMA_M`` (default H_ref), ``LININV_FANH`` (trend|fan-mean),
``LININV_ESTIMATOR`` (split|irls), ``LININV_CORR_AGG`` (pooled|fan-median),
``LININV_SLOPE_AGG`` (pooled-pairs|fan-median), ``LININV_ATTRIBUTION``
(path|seed; default path = the PIG production attribution),
``LININV_OUT_TAG`` (default the --res/--tag suffix), ``LININV_FAN_CACHE``
("" disables).
"""
from __future__ import annotations

import argparse
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from stereo_melt.dynamics import (
    linear_inverse_budget_melt_rate,
    linear_inverse_eulerian_budget_melt_rate,
)
from stereo_melt.flux import grounding_buffer, integrate_basal_flux
from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.io.bedmachine import load_firn_on_grid

from beardmore_shelf import config
from beardmore_shelf.diag_stripe_strips import BANDS
from beardmore_shelf.run_melt import (
    load_floating_mask,
    load_smb_on_grid,
    load_velocity_on_grid,
)
from beardmore_shelf.run_melt_path import load_grounded_mask, load_stack_res


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
        f"{name:<22} vs path REF: raw {scorr(v, ref, 0):.3f}  "
        f"2km {scorr(v, ref, 2000.0 / res_m):.3f}  "
        f"5km {scorr(v, ref, 5000.0 / res_m):.3f}"
    )


def dist_row(name, v):
    v = np.asarray(v, float)
    f = np.isfinite(v)
    return (
        f"{name:<22} median={np.median(v[f]):+7.2f}  "
        f"IQR=[{np.percentile(v[f], 25):+7.2f}, {np.percentile(v[f], 75):+7.2f}]  "
        f"cells={int(f.sum()):,}"
    )


def band_row(name, field, count, band_mask):
    v = np.asarray(field, float)[band_mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return f"    {name:<20} (no finite cells)"
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    c = np.asarray(count, float)[band_mask]
    c_med = float(np.nanmedian(c))
    return (
        f"    {name:<20} med={med:+7.2f}  MAD={mad:6.2f}  "
        f"count_med={c_med:7.0f}  cells={v.size:,}"
    )


def _render_compare_figure(melt_panels, aux_panels, out_png, tag) -> None:
    """Draw the 8-panel comparison (4 melt maps + 4 aux) and save.

    Melt panels use the shared LADDIE symmetric-log melt colormap
    (``stereo_melt.colormaps``, negative = melt -> warm); aux panels keep
    their own linear maps. Shared by the solve path and ``--replot``.
    """
    fig, axes = plt.subplots(2, 4, figsize=(19, 10), constrained_layout=True)
    mcmap, mnorm = melt_cmap(), melt_norm(vmax=10.0)
    for ax, (name, v) in zip(axes[0], melt_panels):
        im = ax.imshow(v, cmap=mcmap, norm=mnorm, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        add_melt_colorbar(fig, im, ax=ax, shrink=0.75)
    for ax, (name, v, cmap, vmax) in zip(axes[1], aux_panels):
        kw = dict(vmin=-vmax, vmax=vmax) if vmax else {}
        im = ax.imshow(v, cmap=cmap, interpolation="nearest", **kw)
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for c0, c1, r0, r1 in BANDS.values():
            ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0,
                                   fill=False, ec="k", lw=0.8, ls="--"))
    fig.suptitle(
        f"Beardmore_Shelf {tag} — Eulerian + budget linear inverses vs production "
        "path solver (m ice/yr, negative = melt; dashed = under-constrained bands)",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_png}")


def _replot_from_disk(tag, out_tag, ref_nc) -> None:
    """Rebuild ``compare_budget_inverse_{out_tag}.png`` from the saved
    eul/path/REF products (no solve), using the current colormap."""
    eul_nc = (
        config.PROCESSED_DIR
        / f"beardmore_shelf_lininv_budget_{out_tag}_eul_{config.START_TIME}_{config.END_TIME}.nc"
    )
    pth_nc = (
        config.PROCESSED_DIR
        / f"beardmore_shelf_lininv_budget_{out_tag}_path_{config.START_TIME}_{config.END_TIME}.nc"
    )
    for f in (eul_nc, pth_nc, ref_nc):
        if not f.exists():
            raise SystemExit(f"--replot: missing product {f}")
    print(f"Replot from:\n  {eul_nc.name}\n  {pth_nc.name}\n  {ref_nc.name}")
    eul = xr.open_dataset(eul_nc)
    pth = xr.open_dataset(pth_nc)
    base = xr.open_dataset(ref_nc)
    flo = np.asarray(load_floating_mask(eul).values, bool)
    lag_v = np.asarray(base["melt_rate_lagrangian"].values, float)
    ref_count = np.asarray(base["lagrangian_count"].values, float)
    eul_count = np.asarray(eul["count"].values, float)
    pth_count = np.asarray(pth["count"].values, float)
    melt_panels = [
        ("Lagrangian path (REF)", lag_v),
        ("Eulerian (robust dh/dt)", np.asarray(eul.melt_rate_hydro.values, float)),
        ("lininv EULERIAN", np.asarray(eul.melt_rate.values, float)),
        ("lininv PATH", np.asarray(pth.melt_rate.values, float)),
    ]
    aux_panels = [
        ("REF path deposit count", ref_count, "viridis", None),
        ("epoch count (dh/dt regression)", np.where(flo, eul_count, np.nan),
         "viridis", None),
        ("lininv_eul nonhydro corr", np.asarray(eul.nonhydro_corr.values, float),
         "PuOr", 5),
        ("lininv_path pair count", np.where(flo, pth_count, np.nan),
         "viridis", None),
    ]
    out_png = config.FIGURES_DIR / f"compare_budget_inverse_{out_tag}.png"
    _render_compare_figure(melt_panels, aux_panels, out_png, tag)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res", type=int, default=config.RES)
    p.add_argument("--tag", type=str, default=None,
                   help="variant tag matching a `--tag` build/tilt run; "
                        "carried into input/output names")
    p.add_argument("--replot", action="store_true",
                   help="skip the solve; rebuild the figure from the saved "
                        "eul/path/REF products with the current colormap")
    args = p.parse_args()
    tag = f"{args.res}m" + (f"_{args.tag}" if args.tag else "")
    out_tag = os.environ.get("LININV_OUT_TAG", tag)
    t0 = time.time()

    ref_nc = (
        config.PROCESSED_DIR
        / f"beardmore_shelf_melt_path_{tag}_{config.START_TIME}_{config.END_TIME}.nc"
    )

    if args.replot:
        _replot_from_disk(tag, out_tag, ref_nc)
        return

    print(f"Loading curated {tag} stack...")
    stack = load_stack_res(args.res, tag=args.tag)
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    floating = load_floating_mask(stack)
    grounded = load_grounded_mask(stack)
    stack = stack.where(floating)
    flo = np.asarray(floating.values, bool)
    print(f"  floating cells: {int(flo.sum()):,}")

    vx, vy, vel_source = load_velocity_on_grid(stack)
    tv = "time" in vx.dims and vx.sizes.get("time", 1) > 1
    print(f"  velocity source: {vel_source} ({'time-varying' if tv else 'static'})")
    a_dot = load_smb_on_grid(stack)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    res_m = abs(float(stack["x"].values[1] - stack["x"].values[0]))
    ny, nx = stack.sizes["y"], stack.sizes["x"]

    reg = float(os.environ.get("LININV_REG", "0.1"))
    eta_bar = float(os.environ.get("LININV_ETA_BAR", "1e14"))
    sigma_env = os.environ.get("LININV_CORR_SIGMA_M", "").strip()
    corr_sigma = float(sigma_env) if sigma_env else None
    fan_H = os.environ.get("LININV_FANH", "trend").strip()
    estimator = os.environ.get("LININV_ESTIMATOR", "split").strip()
    corr_agg = os.environ.get("LININV_CORR_AGG", "pooled").strip()
    slope_agg = os.environ.get("LININV_SLOPE_AGG", "pooled-pairs").strip()
    attribution = os.environ.get("LININV_ATTRIBUTION", "path").strip()

    # ---- 1) Eulerian budget linear inverse (hydro channel == production
    #         Eulerian estimate, Tukey-robust dh/dt = PIG run_melt config) ----
    print(
        f"Running EULERIAN budget linear inverse (robust dh/dt, reg={reg:g}, "
        f"eta_bar={eta_bar:g}, corr_sigma={'H_ref' if corr_sigma is None else corr_sigma})..."
    )
    eul = linear_inverse_eulerian_budget_melt_rate(
        stack, vx, vy, a_dot=a_dot, d=firn, floating_mask=floating,
        eta_bar=eta_bar, reg=reg, transform="dct",
        corr_prefilter_sigma_m=corr_sigma,
        robust_dh_dt=True, progress=True,
    )
    print(
        f"  eulerian (hydro):   {dist_row('', eul.melt_rate_hydro.values)}"
    )
    print(
        f"  lininv_eul:         {dist_row('', eul.melt_rate.values)}  "
        f"[{time.time() - t0:.0f}s]"
    )

    # ---- 2) PATH-integrated budget linear inverse (PIG production config:
    #         attribution=path, pooled-pairs, trend-H, split/pooled) ----------
    fan_cache = os.environ.get(
        "LININV_FAN_CACHE",
        str(config.PROCESSED_DIR
            / f"beardmore_shelf_lininv_fanmaps_{tag}_{attribution}_{slope_agg}.npz"),
    ).strip() or None
    print(
        f"Running PATH budget linear inverse (attribution={attribution}, "
        f"slope_aggregate={slope_agg}, fan_H={fan_H}, estimator={estimator}, "
        f"corr_aggregate={corr_agg})..."
    )
    print(f"  fan-maps cache: {fan_cache}")
    pth = linear_inverse_budget_melt_rate(
        stack, vx, vy, a_dot=a_dot, d=firn, floating_mask=floating,
        eta_bar=eta_bar, reg=reg, transform="dct",
        min_pair_dt_yr=1.5, max_pair_dt_yr=2.5, dt_yr=0.05,
        corr_prefilter_sigma_m=corr_sigma, fan_H=fan_H,
        estimator=estimator, corr_aggregate=corr_agg,
        slope_aggregate=slope_agg, attribution=attribution,
        fan_cache=fan_cache, progress=True, progress_interval_s=30.0,
    )
    print(
        f"  lininv_path:        {dist_row('', pth.melt_rate.values)}  "
        f"({pth.attrs['n_starts']} fans, {pth.attrs['n_pairs']} pairs, "
        f"mean dt {pth.attrs['mean_pair_dt_yr']:.2f} yr)  "
        f"[{time.time() - t0:.0f}s]"
    )

    # ---- REF: the saved production path product --------------------------
    print(f"Loading path REF {ref_nc.name} ...")
    base = xr.open_dataset(ref_nc)
    lagr = base["melt_rate_lagrangian"]
    lagr_count = base["lagrangian_count"]
    lag_v = np.asarray(lagr.values, float)

    print("\n=== spatial agreement (NaN-aware smoothed Pearson) ===")
    print(corr_row("eulerian (hydro)", eul.melt_rate_hydro.values, lag_v, res_m))
    print(corr_row("lininv_eul", eul.melt_rate.values, lag_v, res_m))
    print(corr_row("lininv_path", pth.melt_rate.values, lag_v, res_m))
    print(corr_row("lininv_path_hydro", pth.melt_rate_hydro.values, lag_v, res_m))

    print("\n=== distribution (m ice/yr, negative = melt) ===")
    print(dist_row("path REF", lag_v))
    print(dist_row("eulerian (hydro)", eul.melt_rate_hydro.values))
    print(dist_row("lininv_eul", eul.melt_rate.values))
    print(dist_row("lininv_path", pth.melt_rate.values))

    # ---- under-constrained band diagnostics ------------------------------
    print("\n=== band diagnostics (count semantics: REF=path deposits, "
          "eul=epochs, path=banded pairs) ===")
    band_masks = {}
    for name, (c0, c1, r0, r1) in BANDS.items():
        m = np.zeros((ny, nx), bool)
        m[r0:r1, c0:c1] = True
        m &= flo
        band_masks[name] = m
    band_masks["REF_floating"] = flo.copy()
    eul_count = np.asarray(eul["count"].values, float)
    pth_count = np.asarray(pth["count"].values, float)
    ref_count = np.asarray(lagr_count.values, float)
    for bname, bmask in band_masks.items():
        print(f"  {bname} ({int(bmask.sum()):,} cells):")
        print(band_row("path REF", lag_v, ref_count, bmask))
        print(band_row("eulerian (hydro)", eul.melt_rate_hydro.values, eul_count, bmask))
        print(band_row("lininv_eul", eul.melt_rate.values, eul_count, bmask))
        print(band_row("lininv_path", pth.melt_rate.values, pth_count, bmask))

    # ---- GL-2km clip flux -------------------------------------------------
    print("\n=== GL-2km clip flux (count>=10 own-gate; '|lagr-gate' = REF's gate) ===")
    dom = grounding_buffer(
        np.asarray(floating.values, bool), np.asarray(grounded.values, bool),
        2000.0, res_m,
    )
    q_ref = np.nan_to_num(ref_count, nan=0) >= 10
    q_eul = np.nan_to_num(eul_count, nan=0) >= 10
    q_pth = np.nan_to_num(pth_count, nan=0) >= 10
    rows = [
        ("path REF", lag_v, q_ref),
        ("eulerian (hydro)", np.asarray(eul.melt_rate_hydro.values, float), q_eul),
        ("lininv_eul", np.asarray(eul.melt_rate.values, float), q_eul),
        ("lininv_path", np.asarray(pth.melt_rate.values, float), q_pth),
        ("lininv_eul|lagr-gate", np.asarray(eul.melt_rate.values, float), q_ref),
        ("lininv_path|lagr-gate", np.asarray(pth.melt_rate.values, float), q_ref),
    ]
    flux_attrs = {}
    for name, field, qual in rows:
        b = integrate_basal_flux(field, dom, res_m, quality=qual)
        print(
            f"{name:<22} area={b['area_km2']:7.0f} km2  med={b['median_myr']:+6.2f}  "
            f"Gt/yr: CLIP={b['gt_clip']:.2f}  raw={b['gt_raw']:.2f}  "
            f"robust={b['gt_robust']:.2f}"
        )
        flux_attrs[f"flux_clip_{name}"] = float(b["gt_clip"])

    # ---- save -------------------------------------------------------------
    common_attrs = {
        "baseline": ref_nc.name,
        "velocity_source": vel_source + (" (time-varying)" if tv else " (static)"),
        **flux_attrs,
    }
    eul_out = eul.copy()
    eul_out.attrs.update(common_attrs)
    eul_nc = (
        config.PROCESSED_DIR
        / f"beardmore_shelf_lininv_budget_{out_tag}_eul_{config.START_TIME}_{config.END_TIME}.nc"
    )
    comp = {v: {"zlib": True, "complevel": 4} for v in eul_out.data_vars}
    eul_out.to_netcdf(eul_nc, encoding=comp)
    print(f"\nSaved -> {eul_nc}")

    pth_out = pth.copy()
    pth_out.attrs.update(common_attrs)
    pth_nc = (
        config.PROCESSED_DIR
        / f"beardmore_shelf_lininv_budget_{out_tag}_path_{config.START_TIME}_{config.END_TIME}.nc"
    )
    comp = {v: {"zlib": True, "complevel": 4} for v in pth_out.data_vars}
    pth_out.to_netcdf(pth_nc, encoding=comp)
    print(f"Saved -> {pth_nc}")

    # ---- figure -----------------------------------------------------------
    melt_panels = [
        ("Lagrangian path (REF)", lag_v),
        ("Eulerian (robust dh/dt)", np.asarray(eul.melt_rate_hydro.values, float)),
        ("lininv EULERIAN", np.asarray(eul.melt_rate.values, float)),
        ("lininv PATH", np.asarray(pth.melt_rate.values, float)),
    ]
    aux_panels = [
        ("REF path deposit count", ref_count, "viridis", None),
        ("epoch count (dh/dt regression)", np.where(flo, eul_count, np.nan),
         "viridis", None),
        ("lininv_eul nonhydro corr", np.asarray(eul.nonhydro_corr.values, float),
         "PuOr", 5),
        ("lininv_path pair count", np.where(flo, pth_count, np.nan),
         "viridis", None),
    ]
    out_png = config.FIGURES_DIR / f"compare_budget_inverse_{out_tag}.png"
    _render_compare_figure(melt_panels, aux_panels, out_png, tag)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
