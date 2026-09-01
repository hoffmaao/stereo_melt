"""PIG melt rate as a function of wavenumber (Zinck-style radial PSD).

Compares the radially averaged power spectral density of the melt-rate maps
over the common PIG shelf mask (our min-extent floating mask):

* ours (250 m, 2010-2024): eulerian, eulerian + Helmholtz, restored budget
  local guarded + Helmholtz, monolithic v2 (corrected forward model)
* Shean 2019 reproduction (``results/pig_shean2019_comparison.nc`` —
  melt from his annual DEM mosaics; 250 m grid, ~512 m effective, 2008-2015)
* Zinck et al. NCC "channelized melt underestimated" BURGEE product
  (``data/Zinck2024/PineIsland.tif``, 50 m, REMA+CS2)
* Davison et al. 2023 (1 km, 2010-2017 steady state)

Adusumilli et al. 2020 is NOT on disk (UCSD download is browser-gated).

Method: each product stays on its NATIVE grid; the common mask (ours,
nearest/block-mapped to the product grid) is apodized with a Gaussian taper,
the field demeaned under the taper, zero-filled outside, 2-D FFT, Welch
normalization (variance-preserving), then radial averaging in log-spaced
|k| bins (cycles/km). Sign is irrelevant to the PSD; maps are shown in our
convention (negative = melt).

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY -m pig.plot_melt_spectra
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/wd2/projects/stereo_melt/examples")
sys.path.insert(0, "/wd2/projects/stereo_melt/src")

from stereo_melt import envsetup  # noqa: F401,E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import xarray as xr  # noqa: E402

from pig import config  # noqa: E402
from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm  # noqa: E402
from stereo_melt.spectra import radial_psd  # noqa: E402

RHO_I = 918.0
NC = config.PROCESSED_DIR / "pig_melt_bridging_250m_is2ctempo_sheltilt_2010-01-01_2024-01-10.nc"
NC_GUARD = config.PROCESSED_DIR / "pig_melt_rb_guarded_250m_sheltilt.nc"
NC_MONOV2 = config.PROCESSED_DIR / "pig_melt_mono_v2_250m_sheltilt.nc"
NC_SHEAN = config.BASIN_DIR / "results" / "pig_shean2019_comparison.nc"
TIF_ZINCK = "/wd2/projects/stereo_melt/data/Zinck2024/PineIsland.tif"
TIF_DAVISON = ("/wd2/projects/stereo_melt/data/Davison2023/data/basal_melt/"
               "basal_melt_map_racmo_firn_air_corrected.tif")
# 3H annotations: H_ref 438 m (run log, shelf median) and trunk bin H=1015 m
LAM_3H_SHELF_KM = 3 * 0.438
LAM_3H_TRUNK_KM = 3 * 1.015


def map_mask_nearest(mask, x_src, y_src, x_dst, y_dst):
    """Nearest-neighbour lookup of a (y, x) mask onto another regular grid."""
    ix = np.clip(np.round((x_dst - x_src[0]) / (x_src[1] - x_src[0])).astype(int),
                 0, len(x_src) - 1)
    iy = np.clip(np.round((y_dst - y_src[0]) / (y_src[1] - y_src[0])).astype(int),
                 0, len(y_src) - 1)
    inx = (x_dst >= min(x_src[0], x_src[-1])) & (x_dst <= max(x_src[0], x_src[-1]))
    iny = (y_dst >= min(y_src[0], y_src[-1])) & (y_dst <= max(y_src[0], y_src[-1]))
    out = mask[np.ix_(iy, ix)]
    return out & iny[:, None] & inx[None, :]


def read_tif_window(path, bounds):
    with rasterio.open(path) as r:
        win = rasterio.windows.from_bounds(*bounds, transform=r.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, r.width, r.height))
        win = win.round_offsets().round_lengths()
        a = r.read(1, window=win, masked=True).filled(np.nan)
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        a[np.abs(a) > 1e30] = np.nan
        tr = r.window_transform(win)
        nx, ny = int(win.width), int(win.height)
        x = tr.c + tr.a * (np.arange(nx) + 0.5)
        y = tr.f + tr.e * (np.arange(ny) + 0.5)
        return a, x, y, abs(tr.a)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--vmax", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = xr.open_dataset(NC)
    ours = {
        "Eulerian (ours, 250 m)": ds.eulerian,
        "Eulerian + Helmholtz (ours)": ds.eulerian_helm,
        "restored budget + Helm (ours)": (
            ds.restored_local_helm if "restored_local_helm" in ds
            else xr.open_dataset(NC_GUARD).melt_rate),
        "monolithic v2 (ours)": (
            ds.monolithic_v2 if "monolithic_v2" in ds
            else xr.open_dataset(NC_MONOV2).melt_rate),
    }
    x0 = ds.x.values
    y0 = ds.y.values
    mask0 = np.isfinite(ds.eulerian.values)

    # crop everything to the shelf bbox (+small pad) — also the Zinck bounds
    ys, xs = np.where(mask0)
    pad = 8
    r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, len(y0))
    c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, len(x0))
    xw, yw = x0[c0:c1], y0[r0:r1]
    mask = mask0[r0:r1, c0:c1]
    bounds = (xw.min() - 125, yw.min() - 125, xw.max() + 125, yw.max() + 125)

    sh = xr.open_dataset(NC_SHEAN)
    shean = sh.melt_shean_H.where((sh.floating_mask > 0) & (sh.shelf_polygon_mask > 0)).values
    lo, hi = np.nanpercentile(shean, [1, 99])
    shean = np.clip(shean, lo, hi)  # kill the -5000 m/yr edge artifacts

    zk, zx, zy, zres = read_tif_window(TIF_ZINCK, bounds)
    dv, dxv, dyv, dres = read_tif_window(TIF_DAVISON, bounds)
    # external products use positive = melt; flip to our convention for maps
    print(f"  Zinck native median (shelf px): {np.nanmedian(zk):+.2f}  -> "
          f"positive=melt assumed" , flush=True)
    print(f"  Davison native median: {np.nanmedian(dv):+.2f}  -> positive=melt", flush=True)
    zk_ours = -zk
    dv_ours = -dv

    mask_z = map_mask_nearest(mask, xw, yw, zx, zy)
    mask_d = map_mask_nearest(mask, xw, yw, dxv, dyv)

    # ---------------------------------------------------------------- PSDs
    curves = []  # (label, k, psd, style)
    styles_ours = [dict(color="#1f77b4", lw=1.8),
                   dict(color="#17becf", lw=1.6),
                   dict(color="#2ca02c", lw=1.8),
                   dict(color="#9467bd", lw=1.8)]
    fmin = 1.0 / 40.0
    stats = []
    def add(label, f, msk, dxkm, taper, stl, flip):
        k, p, st = radial_psd(f, msk, dxkm, taper_px=taper, fmin=fmin)
        curves.append((label, k, p, stl))
        stats.append((label, -st["mean"] if flip else st["mean"],
                      np.sqrt(st["var_total"]),
                      100 * st["var_band"] / st["var_total"]))

    for (label, da), stl in zip(ours.items(), styles_ours):
        add(label, da.values[r0:r1, c0:c1], mask, 0.25, 6, stl, True)
    add("Shean 2019 (repro, 250 m grid)", shean[r0:r1, c0:c1], mask, 0.25, 6,
        dict(color="0.25", lw=1.6, ls="--"), True)
    add("Zinck NCC BURGEE (50 m)", zk, mask_z, zres / 1e3, 30,
        dict(color="#d62728", lw=1.8, ls="--"), False)
    add("Davison 2023 (1 km)", dv, mask_d, dres / 1e3, 1.5,
        dict(color="#ff7f0e", lw=1.6, ls="--"), False)

    print("\n  product                        mean melt   sigma    var% at lam<3km")
    for label, mn, sd, bf in stats:
        print(f"  {label:30s} {mn:+8.2f}   {sd:7.2f}   {bf:6.1f}")

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(16.5, 11.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.16, wspace=0.06)
    cmap, norm = melt_cmap(), melt_norm(vmax=args.vmax)
    maps = [("monolithic v2 (ours, 250 m)",
             ours["monolithic v2 (ours)"].values[r0:r1, c0:c1], xw, yw),
            ("Shean 2019 reproduction (250 m grid)", shean[r0:r1, c0:c1], xw, yw),
            ("Zinck NCC BURGEE (50 m)", zk_ours, zx, zy),
            ("Davison 2023 (1 km)", dv_ours, dxv, dyv)]
    im = None
    for j, (title, a, xm, ym) in enumerate(maps):
        ax = fig.add_subplot(gs[0, j])
        msk = {0: mask, 1: mask, 2: mask_z, 3: mask_d}[j]
        im = ax.pcolormesh(np.asarray(xm) / 1e3, np.asarray(ym) / 1e3,
                           np.where(msk, a, np.nan), cmap=cmap, norm=norm,
                           shading="nearest", rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xw.min() / 1e3, xw.max() / 1e3)
        ax.set_ylim(yw.min() / 1e3, yw.max() / 1e3)
    add_melt_colorbar(fig, im, ax=[fig.axes[i] for i in range(4)], shrink=0.85,
                      pad=0.008, label="ḃ (m ice a⁻¹)  negative = melt")

    ax = fig.add_subplot(gs[1, :])
    for label, k, p, stl in curves:
        ok = np.isfinite(p) & (p > 0)
        ax.loglog(k[ok], p[ok], label=label, **stl)
    ax.axvspan(1 / LAM_3H_TRUNK_KM, 1 / LAM_3H_SHELF_KM, color="0.92", zorder=0)
    ax.axvline(1 / LAM_3H_TRUNK_KM, color="0.65", lw=0.8, ls=":")
    ax.axvline(1 / LAM_3H_SHELF_KM, color="0.65", lw=0.8, ls=":")
    ax.text(1 / LAM_3H_TRUNK_KM, ax.get_ylim()[0], "  3H trunk (3.0 km)",
            rotation=90, va="bottom", ha="right", fontsize=8, color="0.4")
    ax.text(1 / LAM_3H_SHELF_KM, ax.get_ylim()[0], "  3H shelf (1.3 km)",
            rotation=90, va="bottom", ha="right", fontsize=8, color="0.4")
    ax.set_xlabel("wavenumber (cycles km⁻¹)")
    ax.set_ylabel("radially averaged PSD of melt rate  (m² a⁻² km²)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, ncol=2, framealpha=0.9)
    sec = ax.secondary_xaxis("top", functions=(lambda f: 1 / np.maximum(f, 1e-9),
                                               lambda lam: 1 / np.maximum(lam, 1e-9)))
    sec.set_xlabel("wavelength (km)")
    fig.suptitle("PIG melt rate vs wavenumber — ours (2010–2024) vs Shean 2019, "
                 "Zinck NCC BURGEE, Davison 2023 (common shelf mask, native grids; "
                 "Adusumilli 2020 not on disk — download is browser-gated)",
                 fontsize=12)
    out = args.out or (config.FIGURES_DIR / "melt_spectra_250m_is2ctempo_sheltilt.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
