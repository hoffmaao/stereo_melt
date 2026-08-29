"""Replot the PIG bridging-solver comparison at presentation quality.

Reads ``processed/pig_melt_bridging_*.nc`` (plus the trunk-guarded local
variant if present), crops to the data bounding box, and draws the melt maps
on the LADDIE symlog scale with a range wide enough for the trunk
(``--vmax``, default 300 m/yr — the 250 m Eulerian spans p1 −295 / p99 +186)
and Δ-vs-Eulerian maps, at high dpi.

Shelf fluxes (panel titles and the printed table) are LIKE-FOR-LIKE: summed
over the pixels finite in EVERY field. Fields that cover more of the shelf
(the Helmholtz variants fill stencil gaps the finite-difference divergence
drops) also report their own-domain flux and extra pixel count, so a
coverage gain is never read as a solver difference.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY -m pig.plot_melt_bridging [--vmax 300] [--dpi 250]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/wd2/projects/stereo_melt")
sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from pig import config  # noqa: E402
from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm  # noqa: E402

RHO_I = 918.0
NC = config.PROCESSED_DIR / "pig_melt_bridging_250m_is2ctempo_sheltilt_2010-01-01_2024-01-10.nc"
NC_GUARD = config.PROCESSED_DIR / "pig_melt_rb_guarded_250m_sheltilt.nc"
NC_MONOV2 = config.PROCESSED_DIR / "pig_melt_mono_v2_250m_sheltilt.nc"
PANELS = [("eulerian", "Eulerian\n(production reference)"),
          ("eulerian_helm", "Eulerian\n+ Helmholtz divergence"),
          ("restored", "restored budget\n(global filter)"),
          ("restored_local_guarded", "restored budget local + Helm\n(no lift where u > 1.5 km/yr)"),
          ("monolithic", "monolithic v1 @ ML-II lam\nlocal bins, η field, Helmholtz"),
          ("monolithic_v2", "monolithic v2 (corrected model)\nflux from restored H, guard, η, Helm")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vmax", type=float, default=300.0)
    ap.add_argument("--dvmax", type=float, default=25.0)
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = xr.open_dataset(NC)
    fields = {k: ds[k] for k in ds.data_vars}
    # since 2026-08-24 the driver writes the guarded local restored budget as
    # restored_local_helm and the corrected monolithic as monolithic_v2 (all
    # on the production fused velocity); the scratch ncs were MEaSUREs-velocity
    # and are only a fallback for pre-audit files.
    if "restored_local_helm" in fields:
        fields["restored_local_guarded"] = fields["restored_local_helm"]
    elif Path(NC_GUARD).exists():
        fields["restored_local_guarded"] = xr.open_dataset(NC_GUARD)["melt_rate"]
    if "monolithic_v2" not in fields and Path(NC_MONOV2).exists():
        fields["monolithic_v2"] = xr.open_dataset(NC_MONOV2)["melt_rate"]
    ref = fields["eulerian"]
    fin = np.isfinite(ref.values)
    ys, xs = np.where(fin)
    pad = 12
    r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad, ref.sizes["y"])
    c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad, ref.sizes["x"])

    def crop(a):
        return a[r0:r1, c0:c1]

    x = ref.x.values[c0:c1] / 1e3
    y = ref.y.values[r0:r1] / 1e3

    # Flux MUST be compared on a common pixel set: the Helmholtz estimator
    # fills stencil gaps the finite-difference divergence drops, so an
    # own-mask flux mixes a coverage gain (+5.4 % of the shelf at PIG) into
    # what reads as a method difference. `flux` = like-for-like; `flux_own`
    # = the field's own domain, reported separately with its pixel count.
    common = np.ones(ref.shape, bool)
    for _k, _v in fields.items():
        common &= np.isfinite(_v.values)

    def flux(m):
        return float(-np.nansum(np.where(common, m.values, np.nan))
                     * 250 * 250 * RHO_I / 1e12)

    def flux_own(m):
        return float(-np.nansum(m.values) * 250 * 250 * RHO_I / 1e12)

    def npx(m):
        return int(np.isfinite(m.values).sum())

    n_common = int(common.sum())
    extra_pct = 100.0 * (max(npx(_v) for _v in fields.values()) - n_common) / n_common
    print(f"  common mask {n_common} px; "
          f"like-for-like vs own-domain fluxes (Gt/yr):", flush=True)
    for _k, _v in fields.items():
        print(f"    {_k:24s} {flux(_v):6.1f}  |  own {flux_own(_v):6.1f} "
              f"({npx(_v)} px)", flush=True)

    cmap, norm = melt_cmap(), melt_norm(vmax=args.vmax)
    n = len(PANELS)
    fig, axs = plt.subplots(2, n, figsize=(4.6 * n + 1.8, 12.5), squeeze=False)
    im = imd = None
    for j, (key, title) in enumerate(PANELS):
        if key not in fields:
            axs[0, j].set_axis_off()
            axs[1, j].set_axis_off()
            continue
        m = fields[key]
        ax = axs[0, j]
        im = ax.pcolormesh(x, y, crop(m.values), cmap=cmap, norm=norm,
                           shading="nearest", rasterized=True)
        extra = npx(m) - n_common
        cov = "" if extra <= 0 else f"  (+{extra} px → {flux_own(m):.1f} own)"
        ax.set_title(f"{title}\n{flux(m):.1f} Gt/yr like-for-like{cov}", fontsize=9.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axs[1, j]
        if key == "eulerian":
            ax.set_axis_off()
            ax.text(0.5, 0.5, "reference", transform=ax.transAxes, ha="center", fontsize=11)
            continue
        d = crop((m - ref).values)
        imd = ax.pcolormesh(x, y, d, cmap="RdBu_r", vmin=-args.dvmax, vmax=args.dvmax,
                            shading="nearest", rasterized=True)
        ax.set_title(f"Δ vs Eulerian   rms {np.sqrt(np.nanmean(d ** 2)):.1f} m/yr, "
                     f"Δflux {flux(m) - flux(ref):+.1f} Gt/yr", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    add_melt_colorbar(fig, im, ax=axs[0].tolist(), shrink=0.9, pad=0.008,
                      label="ḃ (m ice a⁻¹)\nnegative = melt")
    fig.colorbar(imd, ax=axs[1].tolist(), shrink=0.9, pad=0.008,
                 label=f"Δ vs Eulerian (m a⁻¹, ±{args.dvmax:g})")
    fig.suptitle("PIG 250 m is2ctempo_sheltilt 2010–2024 — bridging-aware melt solvers on the "
                 f"production stack ({ds.attrs.get('velocity', '')})\n"
                 "fluxes on the COMMON pixel set; the Helmholtz variants additionally "
                 f"cover up to {extra_pct:.1f} % more shelf (stencil gaps), reported as 'own'",
                 fontsize=11)
    out = args.out or (config.FIGURES_DIR / "melt_bridging_250m_is2ctempo_sheltilt.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
