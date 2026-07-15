"""Validation figure: variable-H (v3) correction is below the inversion's noise floor.

Reads a `compare_v2_vs_v3` NetCDF (produced by `beardmore.compare_v2_vs_v3`) and
makes a 4-panel figure that motivates returning to the constant-H̄ (v2) framing:

    (a) scatter m_v3 vs m_v2_psCG with 1:1 line, OLS slope/intercept, R²
    (b) histogram of (m_v3 − m_v2_psCG) with median/MAD/std
    (c) spatial map of ΔH = H(x,y) − H_ref (context: H is *not* uniform)
    (d) spatial map of (m_v3 − m_v2_psCG) (where, if anywhere, varying H matters)

The pairing is v3_varH_psCG vs v2_const_psCG — same CG/length-scale/Tikhonov
hyperparameters, only the H treatment differs, so the residual isolates the
linearised-in-H correction.

Usage:

    python -m beardmore.plot_v2_vs_v3_validation \
        --input results/beardmore_v2_vs_v3_2019-01-01_2023-03-01_L2000.nc

    # Or by variant (looks up the file under results/<variant_subdir>/):
    python -m beardmore.plot_v2_vs_v3_validation --variant is2+cs2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from beardmore import config
from beardmore.compare_v2_vs_v3 import VARIANTS, _patch_config_for_variant
from beardmore.run_melt import _imshow_xr


def _resolve_input(args) -> Path:
    if args.input:
        p = Path(args.input)
        if not p.is_absolute():
            p = config.BASIN_DIR / p
        if not p.exists():
            raise SystemExit(f"input not found: {p}")
        return p
    _patch_config_for_variant(args.variant)
    candidate = (
        config.RESULTS_DIR
        / f"beardmore_v2_vs_v3_{config.START_TIME}_{config.END_TIME}.nc"
    )
    if not candidate.exists():
        raise SystemExit(
            f"no comparison file at {candidate} — run "
            f"`python -m beardmore.compare_v2_vs_v3 --variant {args.variant}` first."
        )
    return candidate


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, R²) for y vs x."""
    A = np.column_stack([x, np.ones_like(x)])
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="is2",
        choices=list(VARIANTS),
        help="ASP variant (used only to locate the default NetCDF).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Explicit path to a compare_v2_vs_v3 NetCDF. Overrides --variant lookup.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path. Default: <FIGURES_DIR>/v2_vs_v3_validation.png "
        "(or alongside --input if given).",
    )
    parser.add_argument(
        "--clim-m",
        type=float,
        default=3.0,
        help="Symmetric color limit for the residual map (m/yr). Default 3.0.",
    )
    parser.add_argument(
        "--clim-dH",
        type=float,
        default=400.0,
        help="Symmetric color limit for the ΔH map (m). Default 400.",
    )
    args = parser.parse_args()

    nc_path = _resolve_input(args)
    print(f"reading: {nc_path}")
    ds = xr.open_dataset(nc_path)

    H_ref = float(ds.attrs.get("H_ref_m", float("nan")))
    dH_max_rel = float(ds.attrs.get("dH_max_rel", float("nan")))
    window = (ds.attrs.get("window_start", "?"), ds.attrs.get("window_end", "?"))
    print(f"  H_ref = {H_ref:.1f} m  max|ΔH|/H_ref = {dH_max_rel:.2f}")

    floating = ds["floating_mask"].astype(bool)
    m_v2 = ds["melt_rate_v2_psCG"].where(floating)
    m_v3 = ds["melt_rate_v3_varH"].where(floating)
    dH = ds["dH"].where(floating)

    # Flatten and drop NaNs for the scatter / histogram.
    v2_flat = m_v2.values.ravel()
    v3_flat = m_v3.values.ravel()
    keep = np.isfinite(v2_flat) & np.isfinite(v3_flat)
    v2f = v2_flat[keep]
    v3f = v3_flat[keep]
    print(f"  pairs on floating mask: {keep.sum():,}")

    slope, intercept, r2 = _ols(v2f, v3f)
    diff = v3f - v2f
    diff_med = float(np.median(diff))
    diff_mad = float(np.median(np.abs(diff - diff_med)))
    diff_std = float(np.std(diff))
    print(
        f"  v3 = {slope:.4f} * v2 + {intercept:+.3f}  R²={r2:.4f}  "
        f"diff: median={diff_med:+.3f}  MAD={diff_mad:.3f}  std={diff_std:.3f} m/yr"
    )

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)

    # (a) scatter: m_v3 vs m_v2_psCG
    ax = axes[0, 0]
    lim = max(np.abs(np.percentile(v2f, [1, 99])).max(),
              np.abs(np.percentile(v3f, [1, 99])).max()) * 1.05
    if v2f.size > 200_000:
        idx = np.random.default_rng(0).choice(v2f.size, 200_000, replace=False)
        ax.scatter(v2f[idx], v3f[idx], s=1, alpha=0.15, color="steelblue", rasterized=True)
    else:
        ax.scatter(v2f, v3f, s=1, alpha=0.15, color="steelblue", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=1.0, label="1:1")
    xs = np.linspace(-lim, lim, 200)
    ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.2,
            label=f"OLS: slope={slope:.3f}, b={intercept:+.2f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$m_{v2}$ (constant $\bar H$, CG) [m/yr]")
    ax.set_ylabel(r"$m_{v3}$ (variable $H(x,y)$, CG) [m/yr]")
    ax.set_title(f"(a) v3 vs v2: $R^2={r2:.4f}$")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # (b) histogram of v3 - v2
    ax = axes[0, 1]
    hi = max(abs(np.percentile(diff, 1)), abs(np.percentile(diff, 99))) * 1.2
    ax.hist(diff, bins=200, range=(-hi, hi), color="slategray", edgecolor="none")
    ax.axvline(0.0, color="black", lw=1.0)
    ax.axvline(diff_med, color="crimson", lw=1.2,
               label=f"median = {diff_med:+.3f} m/yr")
    ax.set_xlabel(r"$m_{v3} - m_{v2}$ [m/yr]")
    ax.set_ylabel("pixel count")
    ax.set_title(
        f"(b) variable-H residual: MAD={diff_mad:.3f}, std={diff_std:.3f} m/yr"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) ΔH map (context: H is genuinely non-uniform)
    ax = axes[1, 0]
    im_dH = _imshow_xr(ax, dH, cmap="RdBu_r", vmin=-args.clim_dH, vmax=args.clim_dH)
    ax.set_title(
        rf"(c) $\Delta H = H(x,y) - \bar H$ ($\bar H={H_ref:.0f}$ m, "
        rf"$\max|\Delta H|/\bar H={dH_max_rel:.2f}$)"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im_dH, ax=ax, fraction=0.045, label="m")

    # (d) residual map (v3 - v2)
    ax = axes[1, 1]
    diff_da = (m_v3 - m_v2).where(floating)
    im_r = _imshow_xr(ax, diff_da, cmap="PuOr",
                      vmin=-args.clim_m, vmax=args.clim_m)
    ax.set_title(rf"(d) $m_{{v3}} - m_{{v2}}$ map [m/yr]")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im_r, ax=ax, fraction=0.045, label="m/yr")

    fig.suptitle(
        rf"Variable-H correction below noise floor "
        rf"(Beardmore, {window[0]} → {window[1]}) — "
        rf"justifies constant-$\bar H$ (v2) framing",
        fontsize=12,
    )

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = config.BASIN_DIR / out_path
    else:
        if args.input:
            out_path = nc_path.with_name(nc_path.stem + "_validation.png")
        else:
            out_path = config.FIGURES_DIR / "v2_vs_v3_validation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
