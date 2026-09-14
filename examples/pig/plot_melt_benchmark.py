"""Newest bridging-aware melt solver against the Eulerian and Lagrangian benchmarks.

Reads the bridging product for a tag
(``processed/pig_melt_bridging_250m_<tag>_<window>.nc``: ``monolithic_v2`` is
the newest / most complete solver — local (H, u, η) bins, flux from the
restored thickness, trunk guard, Helmholtz divergence — and ``eulerian`` is the
Eulerian budget on the SAME inputs) and a ``run_melt`` product that supplies
the Lagrangian path solver (``melt_rate_lagrangian``) and its own Eulerian.

Melt maps use the LADDIE symlog scale (``--vmax``); the Δ maps are linear
(``--dvmax``). Fluxes in the titles are LIKE-FOR-LIKE: summed over the pixels
finite in all three fields, with each field's own-domain flux beside it so a
coverage difference is never read as a solver difference. When the benchmark
product was made on other inputs (another tilt fit or velocity), the title
says so and quantifies it with the agreement between the two Eulerians.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    cd examples && $PY -m pig.plot_melt_benchmark [--tag is2ctempo_sheltilt_qcey]
        [--benchmark-nc results/pig_melt_250m_<tag>_<window>.nc] [--vmax 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pig import config  # noqa: E402
from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm  # noqa: E402

# Asked of the backend that will do the writing, so --out is validated against
# what this matplotlib can actually render rather than a hand-kept list.
FIGURE_SUFFIXES = frozenset(Figure().canvas.get_supported_filetypes())
RHO_I = 918.0
CANON_BENCHMARK = "pig_melt_250m_is2ctempo_minext_2010-01-01_2024-01-10.nc"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="is2ctempo_sheltilt_qcey")
    ap.add_argument("--benchmark-nc", default=None,
                    help="run_melt product with melt_rate_lagrangian; default: the "
                         "same-tag product if it exists, else the canonical production one")
    ap.add_argument("--vmax", type=float, default=300.0)
    ap.add_argument("--dvmax", type=float, default=100.0)
    ap.add_argument("--zoom", default="auto",
                    help="trunk zoom for the second figure: 'auto' (largest connected patch "
                         "of 1 km-smoothed Eulerian melt < -30 m/yr), 'x0,x1,y0,y1' in km, or 'none'")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--out", default=None,
                    help="output path for the full-shelf figure; must carry an image "
                         "extension matplotlib can write, since the trunk zoom is "
                         "written beside it as <stem>_trunk<ext>")
    args = ap.parse_args()

    window = f"{config.START_TIME}_{config.END_TIME}"
    nc = config.PROCESSED_DIR / f"pig_melt_bridging_250m_{args.tag}_{window}.nc"
    same_tag = config.RESULTS_DIR / f"pig_melt_250m_{args.tag}_{window}.nc"
    if args.benchmark_nc:
        bench = Path(args.benchmark_nc)
    elif same_tag.exists():
        bench = same_tag
    else:
        bench = config.RESULTS_DIR / CANON_BENCHMARK
    same_inputs = bench.resolve() == same_tag.resolve()

    b = xr.open_dataset(nc)
    p = xr.open_dataset(bench)
    if not (np.array_equal(b.x.values, p.x.values) and np.array_equal(b.y.values, p.y.values)):
        raise SystemExit(f"grid mismatch between {nc.name} and {bench.name}")

    newest = b["monolithic_v2"]
    eul = b["eulerian"]
    lag = p["melt_rate_lagrangian"]
    eul_bench = p["melt_rate_eulerian"]

    fields = {"newest": newest, "eulerian": eul, "lagrangian": lag}
    common = np.ones(eul.shape, bool)
    for v in fields.values():
        common &= np.isfinite(v.values)
    n_common = int(common.sum())

    def flux(m):
        return float(-np.nansum(np.where(common, m.values, np.nan)) * 250 * 250 * RHO_I / 1e12)

    def flux_own(m):
        return float(-np.nansum(m.values) * 250 * 250 * RHO_I / 1e12)

    def npx(m):
        return int(np.isfinite(m.values).sum())

    def rms(a):
        return float(np.sqrt(np.nanmean(a ** 2)))

    # How different are the benchmark's inputs? Compare the two Eulerians.
    m2 = np.isfinite(eul.values) & np.isfinite(eul_bench.values)
    corr_e = float(np.corrcoef(eul.values[m2], eul_bench.values[m2])[0, 1])
    rms_e = rms(eul.values[m2] - eul_bench.values[m2])

    print(f"bridging product : {nc.name}")
    print(f"benchmark product: {bench.name}  (same inputs: {same_inputs})")
    print(f"  Eulerian(bridging) vs Eulerian(benchmark): corr {corr_e:.3f}, rms {rms_e:.1f} m/yr")
    print(f"  common mask {n_common} px; like-for-like | own-domain fluxes (Gt/yr):")
    for k, v in fields.items():
        print(f"    {k:10s} {flux(v):6.1f} | own {flux_own(v):6.1f} ({npx(v)} px)")

    ref = eul
    fin = np.isfinite(ref.values)
    ys, xs = np.where(fin)
    pad = 12
    full = (max(ys.min() - pad, 0), min(ys.max() + pad, ref.sizes["y"]),
            max(xs.min() - pad, 0), min(xs.max() + pad, ref.sizes["x"]))

    cmap, norm = melt_cmap(), melt_norm(vmax=args.vmax)
    top = [
        ("newest", "monolithic v2 — newest / most complete\n"
                   "local (H, u, η) bins, restored-H flux, trunk guard, Helmholtz"),
        ("eulerian", "Eulerian budget\n(same inputs as the newest solver)"),
        ("lagrangian", "Lagrangian path solver\n"
                       + ("(same inputs)" if same_inputs else "(production benchmark, other inputs)")),
    ]
    bottom = [
        ("newest", "eulerian", "newest − Eulerian"),
        ("newest", "lagrangian", "newest − Lagrangian"),
        ("lagrangian", "eulerian", "Lagrangian − Eulerian"),
    ]
    inputs_note = ("all three on the same tilt-corrected stack and velocity"
                   if same_inputs else
                   f"benchmark from {bench.name} (other tilt fit / velocity): "
                   f"its Eulerian agrees with ours at corr {corr_e:.3f}, rms {rms_e:.0f} m/yr")

    def render(box, out, what):
        r0, r1, c0, c1 = box

        def crop(a):
            return a[r0:r1, c0:c1]

        x = ref.x.values[c0:c1] / 1e3
        y = ref.y.values[r0:r1] / 1e3
        sub = np.zeros(ref.shape, bool)
        sub[r0:r1, c0:c1] = True
        inbox = common & sub

        def bflux(m):
            return float(-np.nansum(np.where(inbox, m.values, np.nan)) * 250 * 250 * RHO_I / 1e12)

        fig, axs = plt.subplots(2, 3, figsize=(16.5, 10.2), squeeze=False,
                                constrained_layout=True)
        im = imd = None
        for j, (key, title) in enumerate(top):
            m = fields[key]
            ax = axs[0, j]
            im = ax.pcolormesh(x, y, crop(m.values), cmap=cmap, norm=norm,
                               shading="nearest", rasterized=True)
            extra = npx(m) - n_common
            cov = "" if (extra <= 0 or box != full) else f"  (+{extra} px → {flux_own(m):.1f} own)"
            ax.set_title(f"{title}\n{bflux(m):.1f} Gt/yr like-for-like{cov}", fontsize=9.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        for j, (ka, kb, title) in enumerate(bottom):
            a, bb = fields[ka], fields[kb]
            d = crop(np.where(common, a.values - bb.values, np.nan))
            ax = axs[1, j]
            imd = ax.pcolormesh(x, y, d, cmap="RdBu_r", vmin=-args.dvmax, vmax=args.dvmax,
                                shading="nearest", rasterized=True)
            ax.set_title(f"{title}\nrms {rms(d):.1f} m/yr · Δflux {bflux(a) - bflux(bb):+.1f} Gt/yr",
                         fontsize=9.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        add_melt_colorbar(fig, im, ax=axs[0].tolist(), shrink=0.85, pad=0.01,
                          label="ḃ (m ice a⁻¹)\nnegative = melt")
        fig.colorbar(imd, ax=axs[1].tolist(), shrink=0.85, pad=0.01,
                     label=f"Δ (m a⁻¹, ±{args.dvmax:g})")
        fig.suptitle(f"PIG 250 m {args.tag} {config.START_TIME}–{config.END_TIME} — {what}: "
                     f"newest solver vs the Eulerian / Lagrangian benchmarks\n"
                     f"velocity: {b.attrs.get('velocity', '')}\n"
                     f"fluxes on the COMMON pixel set ({int(inbox.sum())} px); {inputs_note}",
                     fontsize=10)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {out}")

    stem = None
    if args.out:
        out_full = Path(args.out)
        if out_full.suffix.lower().lstrip(".") not in FIGURE_SUFFIXES:
            raise SystemExit(
                f"--out {args.out!r} has no image extension matplotlib can write "
                f"(one of: {', '.join('.' + e for e in sorted(FIGURE_SUFFIXES))}). "
                f"The trunk zoom is written beside it as <stem>_trunk{out_full.suffix or '.png'}, "
                f"so an extensionless path would send it to the canonical figures directory "
                f"instead of where you asked."
            )
        stem = out_full.with_suffix("")
    else:
        out_full = config.FIGURES_DIR / f"melt_benchmark_250m_{args.tag}.png"
    render(full, out_full, "full shelf")

    if args.zoom != "none":
        if args.zoom == "auto":
            # The fast trunk = the largest connected patch of strong melt in
            # the 1 km-smoothed Eulerian field (raw pixels < -30 m/yr are
            # scattered noise all over the shelf; the smoothed patch is the
            # deep trunk by the grounding line).
            from scipy import ndimage

            v = np.where(np.isfinite(ref.values), ref.values, 0.0)
            w = np.isfinite(ref.values).astype(float)
            sm = ndimage.gaussian_filter(v, 4) / np.maximum(ndimage.gaussian_filter(w, 4), 1e-6)
            lab, nlab = ndimage.label((sm < -30.0) & (w > 0))
            sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
            zy, zx = np.where(lab == (1 + int(np.argmax(sizes))))
            zp = 8
            box = (max(zy.min() - zp, 0), min(zy.max() + zp, ref.sizes["y"]),
                   max(zx.min() - zp, 0), min(zx.max() + zp, ref.sizes["x"]))
        else:
            x0, x1, y0, y1 = (float(v) * 1e3 for v in args.zoom.split(","))
            xi = np.where((ref.x.values >= min(x0, x1)) & (ref.x.values <= max(x0, x1)))[0]
            yi = np.where((ref.y.values >= min(y0, y1)) & (ref.y.values <= max(y0, y1)))[0]
            box = (yi.min(), yi.max() + 1, xi.min(), xi.max() + 1)
        print(f"  trunk zoom box rows {box[0]}:{box[1]} cols {box[2]}:{box[3]} "
              f"= x {ref.x.values[box[2]]/1e3:.0f}..{ref.x.values[box[3]-1]/1e3:.0f} km, "
              f"y {ref.y.values[box[1]-1]/1e3:.0f}..{ref.y.values[box[0]]/1e3:.0f} km")
        out_zoom = (stem.with_name(stem.name + "_trunk").with_suffix(out_full.suffix)
                    if stem is not None
                    else config.FIGURES_DIR / f"melt_benchmark_trunk_250m_{args.tag}.png")
        render(box, out_zoom, "fast trunk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
