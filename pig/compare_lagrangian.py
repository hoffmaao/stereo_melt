"""Lagrangian-anchored melt-rate comparison for Pine Island (250 m).

The Lagrangian path-integral is one of the two trusted mass-budget solvers.
This script holds it fixed as the reference and asks, for every other
estimate, "how close is it to the Lagrangian, in distribution and in
spatial pattern?"

Estimates compared (all Shean convention, negative = melt):
  * Eulerian mass-conservation   (the other trusted mass-budget solver)
  * masked-CG linear inverse     (diagnostic; see project_linear_inverse_fix)
  * closed-form linear inverse   (the collapsed spectral sibling)
  * Davison 2023 gridded         (external, RACMO-FAC corrected)

Inputs are read from ``pig_maskedcg_250m_arrays.npz`` (eulerian, lagrangian,
masked_cg, closed_form on one 886x723 grid) so nothing is recomputed; Davison
is regridded onto that same grid via the library helper.

Run::

    python -m pig.compare_lagrangian
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import matplotlib.pyplot as plt

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr, spearmanr

from pig import config

NPZ = config.RESULTS_DIR / "pig_maskedcg_250m_arrays.npz"
GRID_RES_M = 250.0


def nan_box_smooth(a: np.ndarray, size_px: int) -> np.ndarray:
    """NaN-aware box mean: local average over finite cells only."""
    if size_px <= 1:
        return a
    m = np.isfinite(a)
    af = np.where(m, a, 0.0)
    num = uniform_filter(af, size=size_px, mode="constant", cval=0.0)
    den = uniform_filter(m.astype(float), size=size_px, mode="constant", cval=0.0)
    out = np.full_like(a, np.nan)
    ok = den > 0.2  # require >=20% window support
    out[ok] = num[ok] / den[ok]
    return out


def _dist(a: np.ndarray) -> str:
    f = a[np.isfinite(a)]
    return (
        f"median={np.median(f):+6.2f}  "
        f"IQR=[{np.percentile(f, 25):+6.2f},{np.percentile(f, 75):+6.2f}]  "
        f"p05/p95=[{np.percentile(f, 5):+6.2f},{np.percentile(f, 95):+6.2f}]  "
        f"n={f.size}"
    )


def _agreement(ref: np.ndarray, est: np.ndarray, sizes_px: dict[str, int]) -> dict:
    """Lagrangian-anchored agreement at native + smoothed scales."""
    out: dict[str, float] = {}
    both = np.isfinite(ref) & np.isfinite(est)
    diff = (est - ref)[both]
    out["n_overlap"] = int(both.sum())
    out["bias"] = float(np.median(diff))          # est - lagrangian
    out["mad"] = float(np.median(np.abs(diff - np.median(diff))))
    r, e = ref[both], est[both]
    out["spearman"] = float(spearmanr(r, e).statistic)
    out["pearson_native"] = float(pearsonr(r, e).statistic)
    for label, px in sizes_px.items():
        rs, es = nan_box_smooth(ref, px), nan_box_smooth(est, px)
        b = np.isfinite(rs) & np.isfinite(es)
        out[f"pearson_{label}"] = float(pearsonr(rs[b], es[b]).statistic)
    return out


def main() -> None:
    config.ensure_output_dirs()
    print(f"Loading {NPZ.name}")
    d = np.load(NPZ)
    x, y = d["x"], d["y"]
    lagr = d["lagrangian"]
    estimates = {
        "Eulerian":    d["eulerian"],
        "masked-CG":   d["masked_cg"],
        "closed-form": d["closed_form"],
    }

    # External reference: Davison 2023 regridded onto the npz grid.
    try:
        from stereo_melt.io.davison import load_davison_gridded_in_shean

        template = xr.DataArray(
            estimates["Eulerian"], coords={"y": y, "x": x}, dims=("y", "x"),
        )
        dav = load_davison_gridded_in_shean(template).values
        # Davison only where we have a Lagrangian estimate (floating footprint).
        dav = np.where(np.isfinite(lagr), dav, np.nan)
        estimates["Davison-2023"] = dav
        print("  Davison 2023 regridded onto npz grid (external reference)")
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip Davison: {exc}]")

    sizes_px = {"2km": round(2000 / GRID_RES_M), "5km": round(5000 / GRID_RES_M)}

    print("\n" + "=" * 78)
    print("DISTRIBUTIONS (m ice/yr, Shean convention: negative = melt)")
    print("=" * 78)
    print(f"  {'Lagrangian (ref)':<16s} {_dist(lagr)}")
    for name, a in estimates.items():
        print(f"  {name:<16s} {_dist(a)}")

    print("\n" + "=" * 78)
    print("SPATIAL AGREEMENT vs LAGRANGIAN")
    print("=" * 78)
    header = (
        f"  {'estimate':<14s} {'Spearman':>9s} {'Pear@nat':>9s} "
        f"{'Pear@2km':>9s} {'Pear@5km':>9s} {'bias':>8s} {'MAD':>7s} {'n':>8s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = {}
    for name, a in estimates.items():
        g = _agreement(lagr, a, sizes_px)
        rows[name] = g
        print(
            f"  {name:<14s} {g['spearman']:>9.3f} {g['pearson_native']:>9.3f} "
            f"{g['pearson_2km']:>9.3f} {g['pearson_5km']:>9.3f} "
            f"{g['bias']:>+8.2f} {g['mad']:>7.2f} {g['n_overlap']:>8d}"
        )
    print(
        "\n  bias = median(estimate - Lagrangian); MAD = robust spread of that "
        "difference.\n  Eulerian is the control: it is the *other* trusted "
        "mass-budget solver, so its\n  row is the bar the linear inverses "
        "should be judged against."
    )

    # ---- Figure: maps (row 1) + diff-from-Lagrangian (row 2) + density (row 3)
    fields = [("Lagrangian (ref)", lagr)] + list(estimates.items())
    ncol = len(fields)
    vlim = 20.0
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    fig, axes = plt.subplots(3, ncol, figsize=(ncol * 3.6, 11),
                             constrained_layout=True)
    if ncol == 1:
        axes = axes[:, None]

    # Row 1: melt-rate maps
    for c, (name, a) in enumerate(fields):
        ax = axes[0, c]
        im = ax.imshow(a, extent=extent, origin="upper", cmap=melt_cmap(),
                       norm=melt_norm(vmax=vlim), aspect="equal")
        f = a[np.isfinite(a)]
        ax.set_title(f"{name}\nmed={np.median(f):+.2f} "
                     f"IQR=[{np.percentile(f,25):+.1f},{np.percentile(f,75):+.1f}]",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if c == ncol - 1:
            add_melt_colorbar(fig, im, ax=axes[0, :].tolist(), fraction=0.012)
    axes[0, 0].set_ylabel("melt rate", fontsize=10)

    # Row 2: estimate - Lagrangian difference maps (anchor column blank)
    dlim = 15.0
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "difference\nfrom\nLagrangian\n(est - ref)",
                    ha="center", va="center", fontsize=11,
                    transform=axes[1, 0].transAxes)
    for c, (name, a) in enumerate(estimates.items(), start=1):
        ax = axes[1, c]
        im = ax.imshow(a - lagr, extent=extent, origin="upper", cmap="PuOr_r",
                       vmin=-dlim, vmax=dlim, aspect="equal")
        ax.set_title(f"{name} - Lagrangian\nbias={rows[name]['bias']:+.2f} "
                     f"MAD={rows[name]['mad']:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if c == ncol - 1:
            fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.012,
                         label="est - Lagrangian (m ice/yr)")

    # Row 3: density scatter at 5 km, Lagrangian (x) vs estimate (y)
    lagr_s = nan_box_smooth(lagr, sizes_px["5km"])
    axes[2, 0].axis("off")
    axes[2, 0].text(0.5, 0.5, "Lagrangian vs estimate\n(5 km smoothed)\n"
                    "1:1 = dashed", ha="center", va="center", fontsize=11,
                    transform=axes[2, 0].transAxes)
    s = 12.0
    for c, (name, a) in enumerate(estimates.items(), start=1):
        ax = axes[2, c]
        est_s = nan_box_smooth(a, sizes_px["5km"])
        b = np.isfinite(lagr_s) & np.isfinite(est_s)
        ax.hexbin(lagr_s[b], est_s[b], gridsize=60, bins="log",
                  cmap="viridis", extent=[-s, s, -s, s])
        ax.plot([-s, s], [-s, s], "r--", lw=1)
        ax.set_xlim(-s, s); ax.set_ylim(-s, s)
        ax.set_xlabel("Lagrangian", fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.set_title(f"r@5km={rows[name]['pearson_5km']:+.2f}", fontsize=9)
        ax.set_aspect("equal")

    fig.suptitle(
        "PIG 250 m — Lagrangian-anchored melt-rate comparison "
        f"({config.START_TIME} -> {config.END_TIME})", fontsize=13)
    out = config.FIGURES_DIR / "lagrangian_anchored_comparison_250m.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
