"""PIG: melt-product signal vs noise as a function of wavenumber.

Reads the half-stack solutions from :mod:`pig.run_noise_floor` and the shipped
full-stack products, and separates each product's radial spectrum into signal
and DEM/strip noise:

    PSD_noise(k) = PSD[m_A - m_B](k) / 4     (independent halves, 1/N scaling)
    PSD_signal(k) = PSD_full(k) - PSD_noise(k)
    SNR(k) = PSD_signal(k) / PSD_noise(k)

The wavelength where SNR = 1 is the resolution limit of the product. Zinck's
BURGEE 50 m PIG melt and the 3H bridging band are drawn for reference.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY -m pig.plot_noise_floor
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/wd2/projects/stereo_melt")
sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")

from stereo_melt import envsetup  # noqa: F401,E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from pig import config  # noqa: E402
from pig.plot_melt_spectra import TIF_ZINCK, map_mask_nearest, read_tif_window  # noqa: E402
from stereo_melt.spectra import radial_psd  # noqa: E402

NC_FULL = (config.PROCESSED_DIR /
           "pig_melt_bridging_250m_is2ctempo_sheltilt_2010-01-01_2024-01-10.nc")
NC_HALF = config.PROCESSED_DIR / "pig_noise_floor_250m_is2ctempo_sheltilt.nc"
LAM_3H_SHELF_KM, LAM_3H_TRUNK_KM = 3 * 0.438, 3 * 1.015
PAIRS = [("Eulerian", "eulerian", "eulerian_A", "eulerian_B", "#1f77b4"),
         ("restored budget + Helm", "restored_local_helm", "rb_A", "rb_B", "#2ca02c")]


def crossing(k, snr):
    """Wavenumber where SNR first falls below 1.

    Interpolates the un-logged SNR linearly in log-k between the last bin at
    or above 1 and the first bin below it. A non-positive SNR (product PSD
    below the noise floor) is a bin definitively below unity and is kept as
    such; only non-finite bins are dropped. Returns NaN when SNR never falls
    below 1 in band, or is already below 1 at the longest in-band
    wavelength, where the crossing lies outside the band and no in-band
    interpolation is defined.
    """
    ok = np.isfinite(snr)
    kk, ss = k[ok], snr[ok]
    below = np.where(ss < 1.0)[0]
    if not len(below) or below[0] == 0:
        return np.nan
    i = below[0]
    f = (ss[i - 1] - 1.0) / (ss[i - 1] - ss[i])
    return float(np.exp(np.log(kk[i - 1]) + f * (np.log(kk[i]) - np.log(kk[i - 1]))))


def available_pairs(full, half, full_var_suffix=""):
    """The PAIRS whose product and half-stack variables all exist.

    ``full_var_suffix`` is appended to every full-product variable name, so
    halves solved with a variant (e.g. ``--common-epoch`` -> ``_ce``) are
    compared against the SAME variant of the full product. A requested
    variant that the full product lacks is an error, never a silent fall
    back to the default product: the noise floor is only meaningful when
    both sides come from the same instrument.

    ``run_noise_floor --skip-rb`` writes Eulerian halves only, so a pair
    whose half-stack variables are absent is dropped (loudly) rather than
    raising a KeyError.
    """
    pairs = []
    for label, fk, ak, bk, colour in PAIRS:
        missing = [v for v in (ak, bk) if v not in half]
        if missing:
            print(f"  skipping {label!r}: halves have no {', '.join(missing)}", flush=True)
            continue
        fk = fk + full_var_suffix
        if fk not in full:
            if full_var_suffix:
                raise SystemExit(
                    f"full product has no variable {fk!r} for --full-var-suffix "
                    f"{full_var_suffix!r}; available: {', '.join(sorted(full.data_vars))}")
            print(f"  skipping {label!r}: full product has no {fk!r}", flush=True)
            continue
        pairs.append((label, fk, ak, bk, colour))
    if not pairs:
        raise SystemExit("no product/half-stack pair is complete; nothing to plot")
    return pairs


def provenance(ds, var):
    """``(common_epoch, velocity)`` recorded for ``var``.

    Variable attrs take precedence over dataset attrs. ``common_epoch``
    absent means the default path (0); ``velocity`` absent is ``None``.
    """
    ce = ds[var].attrs.get("common_epoch", ds.attrs.get("common_epoch", 0))
    vel = ds[var].attrs.get("velocity", ds.attrs.get("velocity"))
    return int(ce), (None if vel is None else str(vel))


def check_provenance(full, half, pairs, *, full_name="full product",
                     purpose="the noise floor",
                     hint="re-solve the full product with the halves' settings (or "
                          "pick a --half-suffix / --full-var-suffix pair that match)"):
    """Refuse to compare halves and a full product solved with different instruments.

    ``run_noise_floor`` stamps ``common_epoch`` and ``velocity`` on the halves
    and the bridging driver stamps ``velocity`` on the full product, so the
    files themselves say which solver settings produced them. Comparing
    common-epoch halves against the default-path full product is the
    mismatched instrument that produced the recorded prototype numbers, so a
    disagreement is an error regardless of which flags the caller passed.
    Returns the agreed ``(common_epoch, velocity)`` for labelling the output.

    ``full_name``, ``purpose`` and ``hint`` only change the wording, so other
    comparisons of ``run_noise_floor`` products against a reference (e.g. the
    quarters of the count ladder) can share the check.
    """
    problems, agreed = [], set()
    for label, fk, ak, bk, _ in pairs:
        fce, fvel = provenance(full, fk)
        for hk in (ak, bk):
            hce, hvel = provenance(half, hk)
            if hce != fce:
                problems.append(f"{label}: halves {hk} common_epoch={hce} "
                                f"vs {full_name} {fk} common_epoch={fce}")
            if hvel is None or fvel is None:
                print(f"  WARNING {label}: velocity provenance missing "
                      f"(halves {hvel!r}, {full_name} {fvel!r}); cannot verify",
                      flush=True)
            elif hvel != fvel:
                problems.append(f"{label}: halves {hk} velocity={hvel!r} "
                                f"vs {full_name} {fk} velocity={fvel!r}")
            agreed.add((fce, fvel))
    if problems:
        raise SystemExit(
            f"instrument mismatch between halves and {full_name} — {purpose} "
            "is only meaningful when both come from the same solver settings:\n  "
            + "\n  ".join(problems) + "\n  " + hint)
    return sorted(agreed)


def stratify(full, half, mask, xw, yw, r0, r1, c0, c1, pairs, label_prefix=""):
    """SNR vs wavenumber for one sub-region; returns rows + curves."""
    fmin = 1.0 / 40.0
    out = []
    for label, fk, ak, bk, colour in pairs:
        mf = full[fk].values[r0:r1, c0:c1]
        d = (half[ak] - half[bk]).values[r0:r1, c0:c1]
        k, psd_f, st_f = radial_psd(mf, mask, 0.25, taper_px=6, fmin=fmin)
        _, psd_d, st_d = radial_psd(d, mask, 0.25, taper_px=6, fmin=fmin)
        psd_n = psd_d / 4.0
        snr = (psd_f - psd_n) / psd_n
        kc = crossing(k, snr)
        out.append(dict(label=label_prefix + label, k=k, snr=snr, colour=colour,
                        lam_c=(np.nan if not np.isfinite(kc) else 1.0 / kc),
                        noise_band=st_d["var_band"] / 4.0, var_band=st_f["var_band"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--stratify", action="store_true",
                    help="also split into fast trunk / slow shelf and compare "
                         "the resolution limit with each region's own 3H")
    ap.add_argument("--half-suffix", default="",
                    help="use pig_noise_floor_...<suffix>.nc (e.g. _ce)")
    ap.add_argument("--full-var-suffix", default="",
                    help="suffix of the full-product variables to compare against "
                         "(e.g. _ce); every PAIRS variable must exist with it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    full = xr.open_dataset(NC_FULL)
    half_nc = (NC_HALF if not args.half_suffix else
               config.PROCESSED_DIR /
               f"pig_noise_floor_250m_is2ctempo_sheltilt{args.half_suffix}.nc")
    half = xr.open_dataset(half_nc)
    print(f"  halves: {half_nc.name}", flush=True)
    pairs = available_pairs(full, half, args.full_var_suffix)
    print("  full-product vars: " + ", ".join(p[1] for p in pairs), flush=True)
    prov = check_provenance(full, half, pairs)
    prov_label = "; ".join(f"velocity={v}, common_epoch={ce}" for ce, v in prov)
    print(f"  provenance (halves == full): {prov_label}", flush=True)
    tag = args.half_suffix + (f"_vs{args.full_var_suffix}" if args.full_var_suffix else "")

    # common mask: every field finite (halves lose thin-coverage pixels)
    m = np.isfinite(full.eulerian.values)
    for k in half.data_vars:
        m &= np.isfinite(half[k].values)
    for _, fk, _, _, _ in pairs:
        m &= np.isfinite(full[fk].values)
    ys, xs = np.where(m)
    pad = 8
    r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0])
    c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])
    mask = m[r0:r1, c0:c1]
    xw, yw = full.x.values[c0:c1], full.y.values[r0:r1]
    print(f"  common mask: {int(mask.sum())} px "
          f"({100 * mask.sum() / np.isfinite(full.eulerian.values).sum():.0f} % of the shelf)",
          flush=True)

    fmin = 1.0 / 40.0
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.2))
    rows = []
    for label, fk, ak, bk, colour in pairs:
        mf = full[fk].values[r0:r1, c0:c1]
        d = (half[ak] - half[bk]).values[r0:r1, c0:c1]
        k, psd_f, st_f = radial_psd(mf, mask, 0.25, taper_px=6, fmin=fmin)
        _, psd_d, st_d = radial_psd(d, mask, 0.25, taper_px=6, fmin=fmin)
        psd_n = psd_d / 4.0
        psd_s = psd_f - psd_n
        snr = psd_s / psd_n
        kc = crossing(k, snr)
        lam_c = np.nan if not np.isfinite(kc) else 1.0 / kc
        # band variance split (lambda < 3 km)
        noise_band = st_d["var_band"] / 4.0
        rows.append((label, np.sqrt(st_f["var_total"]), np.sqrt(st_d["var_total"]) / 2.0,
                     100 * noise_band / max(st_f["var_band"], 1e-30), lam_c))
        axs[0].loglog(k, psd_f, color=colour, lw=1.9, label=f"{label} — product")
        axs[0].loglog(k, psd_n, color=colour, lw=1.4, ls=":", label=f"{label} — noise floor")
        ok = psd_s > 0
        axs[0].loglog(k[ok], psd_s[ok], color=colour, lw=1.2, ls="--", alpha=0.8,
                      label=f"{label} — signal")
        axs[1].loglog(k, snr, color=colour, lw=1.9, label=label)
        if np.isfinite(lam_c):
            axs[1].plot([kc], [1.0], "o", color=colour, ms=7, zorder=5)

    # Zinck reference on the same mask
    bounds = (xw.min() - 125, yw.min() - 125, xw.max() + 125, yw.max() + 125)
    zk, zx, zy, zres = read_tif_window(TIF_ZINCK, bounds)
    kz, psz, _ = radial_psd(zk, map_mask_nearest(mask, xw, yw, zx, zy), zres / 1e3,
                            taper_px=30, fmin=fmin)
    axs[0].loglog(kz, psz, color="#d62728", lw=1.6, ls="--", label="Zinck BURGEE 50 m")

    for ax in axs:
        ax.axvspan(1 / LAM_3H_TRUNK_KM, 1 / LAM_3H_SHELF_KM, color="0.92", zorder=0)
        ax.set_xlabel("wavenumber (cycles km⁻¹)")
        ax.grid(True, which="both", alpha=0.25)
        sec = ax.secondary_xaxis("top", functions=(lambda f: 1 / np.maximum(f, 1e-9),
                                                   lambda lam: 1 / np.maximum(lam, 1e-9)))
        sec.set_xlabel("wavelength (km)")
    axs[0].set_ylabel("radially averaged PSD (m² a⁻² km²)")
    axs[0].set_title("melt spectra: product = signal + DEM/strip noise")
    axs[0].legend(fontsize=8, ncol=1)
    axs[1].axhline(1.0, color="0.3", lw=1.0, ls="-")
    axs[1].set_ylabel("SNR  =  signal PSD / noise PSD")
    axs[1].set_title("signal-to-noise vs wavenumber (● = SNR 1, the resolution limit)")
    axs[1].legend(fontsize=9)

    # shelf-integrated flux uncertainty from the same split:
    # Var[flux_A - flux_B] = 2 sigma_half^2, sigma_half^2 ~ 2 sigma_full^2
    print("\n  shelf-flux DEM-noise uncertainty (1 sigma, from the half difference):")
    for label, fk, ak, bk, _ in pairs:
        fa = -np.nansum(np.where(mask, half[ak].values[r0:r1, c0:c1], np.nan)) * 62500 * 918 / 1e12
        fb = -np.nansum(np.where(mask, half[bk].values[r0:r1, c0:c1], np.nan)) * 62500 * 918 / 1e12
        ff = -np.nansum(np.where(mask, full[fk].values[r0:r1, c0:c1], np.nan)) * 62500 * 918 / 1e12
        sig_full = abs(fa - fb) / 2.0
        print(f"  {label:24s} full {ff:6.1f}  halves {fa:6.1f}/{fb:6.1f}  "
              f"=> sigma_full ~ {sig_full:.1f} Gt/yr ({100 * sig_full / abs(ff):.1f} %)")

    print(f"\n  {'product':24s} {'sigma':>8s} {'noise':>8s} {'noise% of':>10s} {'SNR=1':>9s}")
    print(f"  {'':24s} {'(m/yr)':>8s} {'(m/yr)':>8s} {'<3km var':>10s} {'(km)':>9s}")
    for label, s, n, pct, lam in rows:
        lam_s = "—" if not np.isfinite(lam) else f"{lam:.2f}"
        print(f"  {label:24s} {s:8.1f} {n:8.1f} {pct:9.0f}% {lam_s:>9s}")

    if args.stratify:
        _stratified_figure(full, half, mask, xw, yw, r0, r1, c0, c1, pairs, args, tag)

    fig.suptitle("PIG melt products: how much of the short-wavelength power is real?\n"
                 "noise floor from independent half-stacks (alternating epochs, disjoint "
                 "strips); shaded = the 3H bridging band (1.3–3.0 km)\n"
                 f"halves and full product solved alike: {prov_label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = args.out or (config.FIGURES_DIR /
                       f"melt_noise_floor_250m_is2ctempo_sheltilt{tag}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


def _stratified_figure(full, half, mask, xw, yw, r0, r1, c0, c1, pairs, args, tag=""):
    """Is the lambda <~ 3H bridging band observable ANYWHERE at PIG?

    The bridging correction acts below ~3H. 3H is 1.3 km on the thin shelf but
    3.0 km in the deep trunk, so the trunk is the one place the band might sit
    above the noise. Split by speed (the trunk is the fast ice) and compare
    each region's resolution limit against its OWN 3H.
    """
    import sys as _sys
    _sys.path.insert(0, "/wd2/projects/stereo_melt")
    from pig.run_melt import load_stack  # noqa: E402
    from stereo_melt.freeboard import freeboard_to_thickness  # noqa: E402

    z = np.load(config.PROCESSED_DIR / "pig_eta_field_250m_dual_20260730_t0era5.npz")
    u = xr.DataArray(np.hypot(z["u_model_x"], z["u_model_y"]), dims=("y", "x"),
                     coords={"y": z["y"], "x": z["x"]})
    u = u.reindex_like(full.eulerian, method="nearest").values[r0:r1, c0:c1]
    st = load_stack("pig_stack_250m_is2ctempo_sheltilt")
    H = freeboard_to_thickness(st.mean("time", skipna=True)).values[r0:r1, c0:c1]

    regions = [("fast trunk (|u| ≥ 1 km/yr)", mask & np.isfinite(u) & (u >= 1000.0), "#d62728"),
               ("slow shelf (|u| < 1 km/yr)", mask & np.isfinite(u) & (u < 1000.0), "#1f77b4")]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    print(f"\n  {'region':28s} {'px':>7s} {'med H':>7s} {'3H':>6s} {'SNR=1':>7s} {'band':>10s}")
    print(f"  {'':28s} {'':>7s} {'(m)':>7s} {'(km)':>6s} {'(km)':>7s} {'observable':>10s}")
    for rlabel, rmask, colour in regions:
        if rmask.sum() < 2000:
            continue
        Hmed = float(np.nanmedian(H[rmask]))
        lam3H = 3 * Hmed / 1e3
        rows = stratify(full, half, rmask, xw, yw, r0, r1, c0, c1, pairs)
        for i, r in enumerate(rows):
            ls = "-" if i == 0 else "--"
            ax.loglog(r["k"], r["snr"], color=colour, lw=1.8, ls=ls,
                      label=f"{rlabel} — {r['label']}")
            if np.isfinite(r["lam_c"]):
                ax.plot([1 / r["lam_c"]], [1.0], "o", color=colour, ms=7, zorder=5)
            obs = ("YES" if np.isfinite(r["lam_c"]) and r["lam_c"] < lam3H else "no")
            print(f"  {rlabel[:26]:28s} {int(rmask.sum()):7d} {Hmed:7.0f} {lam3H:6.2f} "
                  f"{r['lam_c']:7.2f} {obs:>10s}   ({r['label']})")
        ax.axvline(1 / lam3H, color=colour, lw=1.0, ls=":", alpha=0.8)
        ax.text(1 / lam3H, 30, f" 3H = {lam3H:.1f} km", rotation=90, fontsize=8,
                color=colour, va="top")
    ax.axhline(1.0, color="0.3", lw=1.0)
    ax.set_xlabel("wavenumber (cycles km⁻¹)")
    ax.set_ylabel("SNR = signal PSD / noise PSD")
    sec = ax.secondary_xaxis("top", functions=(lambda f: 1 / np.maximum(f, 1e-9),
                                               lambda lam: 1 / np.maximum(lam, 1e-9)))
    sec.set_xlabel("wavelength (km)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Is the bridging band (λ ≲ 3H) observable at PIG?\n"
                 "dotted line = each region's own 3H; ● = SNR 1. Bridging is measurable "
                 "only where ● lies LEFT of the dotted line.", fontsize=10.5)
    out = config.FIGURES_DIR / f"melt_noise_floor_regions_250m_is2ctempo_sheltilt{tag}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
