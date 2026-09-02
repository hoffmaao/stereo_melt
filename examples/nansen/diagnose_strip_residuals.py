"""Strip-residual diagnostic for recovered melt fields on Nansen.

Roughness analysis of the recovered melt rate is only meaningful if the
spatial structure isn't dominated by per-REMA-strip residuals (constant
or near-constant offset within a single strip's footprint that pc_align
+ tilt_fit didn't fully absorb). This script measures and visualises
that contamination.

For each recovered melt field ``m(x, y)`` and each epoch ``t`` (≈ one
REMA strip), we compute the per-epoch-footprint median ``μ_t`` and
broadcast it back to the footprint shape. The strip-coherent component
at pixel ``p`` is the mean of ``μ_t`` over the strips covering ``p``::

    m_strip(p) = mean_{t : p ∈ footprint_t} μ_t

The residual ``m_resid = m - m_strip`` is what's left after the
per-strip mean is removed -- this is the "honest" spatial structure you
can do roughness analysis on. Compares baseline closed-DCT, D-DCT
(λ=1e-1, L=H_ref/2), and E-DCT (λ=1e-1, L=100 m).

Run::

    python -m nansen.diagnose_strip_residuals
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from nansen import config
from nansen.run_melt import _imshow_xr, load_stack


def _strip_decomposition(
    melt: np.ndarray,
    footprints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decompose ``melt`` into ``m_strip`` (per-strip-median broadcast) and
    ``m_resid = melt - m_strip``.

    Parameters
    ----------
    melt : (ny, nx) float64
    footprints : (n_t, ny, nx) bool
        ``footprints[t]`` is the finite-data mask of epoch ``t``.

    Returns
    -------
    m_strip : (ny, nx) float64
    m_resid : (ny, nx) float64
    diag : dict with summary statistics.
    """
    n_t = footprints.shape[0]
    valid_pixel = np.isfinite(melt)

    # Per-epoch footprint median of melt (skipping NaN melt cells).
    per_epoch_median = np.full(n_t, np.nan, dtype=np.float64)
    for t in range(n_t):
        cells = footprints[t] & valid_pixel
        if cells.any():
            per_epoch_median[t] = np.median(melt[cells])

    # Strip-coherent reconstruction: at each pixel, mean of per-epoch
    # medians weighted by footprint indicator.
    coverage_count = footprints.astype(np.int32).sum(axis=0)
    accum = np.zeros_like(melt, dtype=np.float64)
    n_finite_t = 0
    for t in range(n_t):
        if not np.isfinite(per_epoch_median[t]):
            continue
        n_finite_t += 1
        accum += footprints[t].astype(np.float64) * per_epoch_median[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        m_strip = np.where(coverage_count > 0, accum / coverage_count, np.nan)
    m_resid = melt - m_strip

    diag = {
        "n_t_with_data": int(n_finite_t),
        "per_epoch_median": per_epoch_median,
        "median_of_medians": float(np.nanmedian(per_epoch_median)),
        "std_of_medians": float(np.nanstd(per_epoch_median)),
        "p05_p95_medians": (
            float(np.nanpercentile(per_epoch_median, 5)),
            float(np.nanpercentile(per_epoch_median, 95)),
        ),
    }
    return m_strip, m_resid, diag


def _ratio(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    sa = float(np.std(a))
    sb = float(np.std(b))
    return sa / sb if sb > 0 else float("nan")


def main() -> None:
    config.ensure_output_dirs()

    variants_nc = (
        config.RESULTS_DIR
        / f"nansen_closed_form_variants_{config.START_TIME}_{config.END_TIME}.nc"
    )
    five_nc = (
        config.RESULTS_DIR
        / f"nansen_five_methods_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Loading variants -> {variants_nc}")
    var = xr.open_dataset(variants_nc)
    print(f"Loading five-methods baseline -> {five_nc}")
    five = xr.open_dataset(five_nc)

    print("Loading tilt-corrected stack for footprints...")
    stack = load_stack()
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    # Per-epoch finite footprints from the stack.
    footprints = np.isfinite(stack.values)  # (time, y, x)
    coverage = footprints.sum(axis=0)
    print(f"  coverage: median={int(np.median(coverage))} epochs/pixel, max={int(coverage.max())}")

    floating = var.floating_mask.astype(bool).values

    # Methods come from two source datasets: baseline closed-DCT lives in
    # five-methods, the masked-CG variants live in variants.nc.
    methods = [
        ("baseline closed-DCT",        five, "melt_rate_closed_dct"),
        ("D-DCT (λ=1e-1, L=H_ref/2)",  var,  "melt_rate_masked_cg_dct_d"),
    ]
    if "melt_rate_masked_cg_dct_e" in var.data_vars:
        methods.append(
            ("E-DCT (λ=1e-1, L=100m)", var, "melt_rate_masked_cg_dct_e")
        )

    rows = []
    for label, src, vname in methods:
        m = src[vname].values.astype(np.float64)
        m = np.where(floating, m, np.nan)
        m_strip, m_resid, diag = _strip_decomposition(m, footprints)
        m_strip = np.where(floating, m_strip, np.nan)
        m_resid = np.where(floating, m_resid, np.nan)
        r_strip = _ratio(m_strip, m)
        r_resid = _ratio(m_resid, m)
        print(
            f"  [{label}]  std(m)={np.nanstd(m):.2f}  "
            f"std(m_strip)/std(m)={r_strip:.2f}  "
            f"std(m_resid)/std(m)={r_resid:.2f}  "
            f"per-epoch median: med={diag['median_of_medians']:+.2f} "
            f"std={diag['std_of_medians']:.2f} "
            f"p05/p95=[{diag['p05_p95_medians'][0]:+.2f}, {diag['p05_p95_medians'][1]:+.2f}]"
        )
        rows.append((label, m, m_strip, m_resid, diag, r_strip, r_resid))

    # ---- Figure: 3 cols (m, m_strip, m_resid) x N rows (methods) ----
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(3 * 4.6, n_rows * 5.2), constrained_layout=True
    )
    if n_rows == 1:
        axes = axes[None, :]
    clim_full = (-5.0, 5.0)
    clim_resid = (-5.0, 5.0)

    template = five.melt_rate_closed_dct  # (y, x) DataArray for coords
    for ri, (label, m, m_strip, m_resid, diag, r_strip, r_resid) in enumerate(rows):
        m_da = template.copy(data=m).rename("m")
        s_da = template.copy(data=m_strip).rename("m_strip")
        r_da = template.copy(data=m_resid).rename("m_resid")

        im0 = _imshow_xr(axes[ri, 0], m_da, cmap="RdBu_r", vmin=clim_full[0], vmax=clim_full[1])
        axes[ri, 0].set_title(
            f"{label}\nm(x,y) — std={np.nanstd(m):.2f}", fontsize=10,
        )
        fig.colorbar(im0, ax=axes[ri, 0], fraction=0.045)

        im1 = _imshow_xr(axes[ri, 1], s_da, cmap="RdBu_r", vmin=clim_full[0], vmax=clim_full[1])
        axes[ri, 1].set_title(
            f"strip-coherent component\n"
            f"std/std(m)={r_strip:.2f}  "
            f"per-strip med std={diag['std_of_medians']:.2f}",
            fontsize=10,
        )
        fig.colorbar(im1, ax=axes[ri, 1], fraction=0.045)

        im2 = _imshow_xr(axes[ri, 2], r_da, cmap="RdBu", vmin=clim_resid[0], vmax=clim_resid[1])
        axes[ri, 2].set_title(
            f"residual = m − strip-coherent\n"
            f"std/std(m)={r_resid:.2f}  std={np.nanstd(m_resid):.2f}",
            fontsize=10,
        )
        fig.colorbar(im2, ax=axes[ri, 2], fraction=0.045)

        for ax in axes[ri]:
            ax.set_xlabel("x (m)")
        axes[ri, 0].set_ylabel("y (m)")
    fig.suptitle(
        f"Nansen — strip-residual decomposition  "
        f"({config.START_TIME} → {config.END_TIME}; "
        f"n_epochs={stack.sizes['time']})",
        fontsize=12,
    )
    fig_path = config.FIGURES_DIR / "strip_residual_diagnostic.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {fig_path}")

    # ---- Per-epoch median time series ----
    fig2, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)
    times = stack["time"].values
    for label, m, m_strip, m_resid, diag, r_strip, r_resid in rows:
        ax.plot(times, diag["per_epoch_median"], "-o", markersize=4, label=label)
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("time (epoch)")
    ax.set_ylabel("per-epoch-footprint median melt (m ice/yr)")
    ax.set_title("Per-strip-median trajectory — flat ⇒ no strip-coherent contamination")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig2_path = config.FIGURES_DIR / "strip_median_trajectory.png"
    fig2.savefig(fig2_path, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {fig2_path}")


if __name__ == "__main__":
    main()
