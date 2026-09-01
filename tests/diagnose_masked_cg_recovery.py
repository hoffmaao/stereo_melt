"""Does the masked-CG inverse recover *melt* or just *spectral structure*?

The decisive controlled test the prior diagnostics never ran. Earlier
synthetic work exercised only the closed-form on dense data; the
masked-CG-vs-everyone-else spatial-correlation finding (PIG, 2026-06-13)
was on *real* data with no ground truth. So we could not tell whether the
production masked-CG path (``stationary_pseudospectral_lagrangian_inverse``)
is broken or is faithfully recovering a band-limited / DC-removed quantity
that genuinely differs from the mass-budget solvers.

This script closes that gap. On a KNOWN ``m_true`` field, forward-modelled
and then degraded by the same failure-mode ladder as
``synthetic_realistic_recovery`` (coverage gaps, tilt, non-planar
residual), it runs all five solvers including the masked-CG and computes
the SAME spatial-agreement matrix used on real PIG:

  Spearman rank corr, plus nan-aware Gaussian-smoothed Pearson at 2 km
  and 5 km, for every solver pair AND every solver vs truth.

The interpretation key:
  * masked-CG corr-with-TRUTH high  -> it recovers melt; real-data
    failure is a data/noise problem, not the algorithm.
  * masked-CG corr-with-truth low but corr-with-closed-form high, even on
    synthetic with known truth -> structural: it recovers the kernel's
    recoverable spectral band, not melt. Diagnostic-only is correct.

Run::

    python -u tests/diagnose_masked_cg_recovery.py
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from stereo_melt.constants import rhoi
from stereo_melt.dynamics import (
    forward,
    lagrangian_frame_stack,
    linear_inverse_lagrangian_melt_rate,
    stationary_pseudospectral_lagrangian_inverse,
)
from stereo_melt.melt import eulerian_melt_rate, lagrangian_melt_rate

# Reuse the validated noise/coverage injectors from the existing ladder.
# tests/ is not a package (no __init__.py); import the sibling by name,
# matching diagnose_dhdt_inverse.py / plot_coverage_diagnostic.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_realistic_recovery import (  # noqa: E402
    add_per_strip_nonplanar_residual,
    add_per_strip_tilt,
    add_strip_coverage_mask,
    build_truth,
)

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
SECONDS_PER_YEAR = 86400.0 * 365.25


def nan_smooth(a: np.ndarray, sigma_px: float) -> np.ndarray:
    """NaN-aware Gaussian smooth: gaussian(a*mask)/gaussian(mask)."""
    m = np.isfinite(a).astype(np.float64)
    a0 = np.where(m > 0, a, 0.0)
    num = gaussian_filter(a0, sigma_px, mode="constant")
    den = gaussian_filter(m, sigma_px, mode="constant")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.05)
    return out


def corr_pair(a: np.ndarray, b: np.ndarray, mask: np.ndarray, dx: float) -> dict:
    """Spearman + smoothed-Pearson at 2 km / 5 km over the common footprint."""
    base = mask & np.isfinite(a) & np.isfinite(b)
    out = {}
    if base.sum() >= 16:
        out["spearman"] = float(spearmanr(a[base], b[base]).statistic)
    else:
        out["spearman"] = np.nan
    for label, scale_m in (("p2km", 2_000.0), ("p5km", 5_000.0)):
        sig = scale_m / dx
        asm = nan_smooth(np.where(mask, a, np.nan), sig)
        bsm = nan_smooth(np.where(mask, b, np.nan), sig)
        g = mask & np.isfinite(asm) & np.isfinite(bsm)
        if g.sum() >= 16:
            out[label] = float(np.corrcoef(asm[g], bsm[g])[0, 1])
        else:
            out[label] = np.nan
    return out


def run_scenario(scenario: str, m_true: xr.DataArray, *, H=500.0,
                 n_epochs=15, t_window_yr=4.0, vx_value=0.0):
    """Forward-model, degrade per scenario, run all five solvers."""
    times_s = np.linspace(0.0, t_window_yr * SECONDS_PER_YEAR, n_epochs)
    times_dt = pd.to_datetime("2020-01-01") + pd.to_timedelta(times_s, unit="s")
    alpha_forward = vx_value * (2.0 * 1e14 / (rhoi * 9.81 * H)) / H
    h_xr = forward(
        m_true, H=H, eta_bar=1e14, alpha=alpha_forward, gamma=0.0,
        stationary=True, times=xr.DataArray(times_s, dims="t"),
    )
    h_stack = xr.DataArray(
        h_xr.values, dims=("time", "y", "x"),
        coords={"time": times_dt, "y": m_true["y"].values, "x": m_true["x"].values},
        name="h_freeboard",
    )
    if scenario in ("S1", "S2", "S3"):
        h_stack = add_strip_coverage_mask(h_stack, coverage_frac=0.30, seed=11)
    if scenario in ("S2", "S3"):
        h_stack = add_per_strip_tilt(h_stack, tilt_amp_m=0.5, seed=23)
    if scenario in ("S3",):
        h_stack = add_per_strip_nonplanar_residual(h_stack, amp_m=0.2, seed=37)

    vx = xr.DataArray(np.full_like(m_true.values, vx_value),
                      dims=m_true.dims, coords=m_true.coords)
    vy = xr.zeros_like(vx)
    floating = xr.DataArray(np.ones_like(m_true.values, dtype=bool),
                            dims=m_true.dims, coords=m_true.coords)
    a_dot = xr.zeros_like(m_true)

    eul = eulerian_melt_rate(h_stack, vx, vy, a_dot=a_dot,
                             robust_dh_dt=True).melt_rate
    lagr = lagrangian_melt_rate(h_stack, vx, vy, a_dot=a_dot, dt_yr=0.05,
                                seed_stride=2, pairs="all",
                                min_dt_yr=2.0 / 12.0).melt_rate
    h_lag = lagrangian_frame_stack(h_stack, vx, vy, dt_yr=0.05)
    closed = linear_inverse_lagrangian_melt_rate(
        h_stack, vx, vy, floating_mask=floating, H_ref=H,
        boundary_fix="off", transform="dct", reg=1e-1,
        dt_yr=0.05, h_lag_precomputed=h_lag,
    ).melt_rate
    # Production masked-CG params (pig/run_melt.py): DCT, lambda=1e-1,
    # length_scale=H/2, max_iter=200. H_ref supplied (synthetic h is anomaly).
    cg = stationary_pseudospectral_lagrangian_inverse(
        h_stack, vx, vy, floating_mask=floating, H_ref=H, d=0.0,
        tikhonov=1e-1, length_scale_m=H / 2.0,
        max_iter=200, cg_tol=1e-6, dt_yr=0.05,
        transform="dct", h_lag_precomputed=h_lag, verbose=False,
    )
    print(f"    masked-CG: iters={cg.attrs['cg_iter']} "
          f"converged={cg.attrs['cg_converged']}")
    return {
        "truth": -m_true,  # Shean convention (negative = melt)
        "Eulerian": eul,
        "Lagrangian": lagr,
        "closed-form": closed,
        "masked-CG": cg.melt_rate,
    }


def report(scenario: str, fields: dict, dx: float) -> pd.DataFrame:
    mask = np.isfinite(fields["truth"].values)
    names = list(fields.keys())
    arrs = {k: v.values for k, v in fields.items()}

    # median table
    print(f"\n  [{scenario}] median / IQR (Shean: negative = melt):")
    for k in names:
        v = arrs[k][mask & np.isfinite(arrs[k])]
        if v.size:
            print(f"    {k:<12s} median={np.median(v):+7.2f}  "
                  f"IQR=[{np.percentile(v, 25):+7.2f}, {np.percentile(v, 75):+7.2f}]")

    # correlation matrix (p5km) + key pairs at all scales
    print(f"\n  [{scenario}] spatial agreement (Spearman | p2km | p5km):")
    rows = []
    for a, b in combinations(names, 2):
        c = corr_pair(arrs[a], arrs[b], mask, dx)
        rows.append({"pair": f"{a} vs {b}", **c})
        print(f"    {a:>11s} vs {b:<12s}  "
              f"{c['spearman']:+.3f} | {c['p2km']:+.3f} | {c['p5km']:+.3f}")
    return pd.DataFrame(rows)


def plot(scenario: str, fields: dict):
    names = list(fields.keys())
    m_true = fields["truth"]
    extent = (float(m_true["x"].min()), float(m_true["x"].max()),
              float(m_true["y"].min()), float(m_true["y"].max()))
    fig, axes = plt.subplots(1, len(names), figsize=(len(names) * 4.2, 5.0),
                             constrained_layout=True)
    for ax, k in zip(axes, names):
        im = ax.imshow(fields[k].values, extent=extent, origin="upper",
                       cmap="RdBu_r", vmin=-10, vmax=10, aspect="equal",
                       interpolation="nearest")
        ax.set_title(k, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.045, label="m ice/yr")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(f"masked-CG recovery diagnostic — {scenario}", fontsize=12)
    out = OUT_DIR / f"diagnose_masked_cg_{scenario}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    m_true = build_truth()
    dx = float(abs(m_true["x"].values[1] - m_true["x"].values[0]))
    print(f"m_true: mean(Shean)={-float(m_true.mean()):+.2f}  "
          f"peak(Shean)={-float(m_true.max()):+.2f} m/yr  dx={dx:.0f} m")
    scenarios = [
        ("S0", "dense, clean (sanity: all should recover truth)"),
        ("S1", "+ 30% per-strip coverage gaps"),
        ("S3", "+ tilt + non-planar residual (realistic)"),
    ]
    for scenario, desc in scenarios:
        print(f"\n===== {scenario}: {desc} =====")
        fields = run_scenario(scenario, m_true)
        report(scenario, fields, dx)
        plot(scenario, fields)


if __name__ == "__main__":
    main()
