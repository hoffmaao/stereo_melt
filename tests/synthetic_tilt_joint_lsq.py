"""Test the Shean/ndinterp-style joint LSQ tilt-fit on synthetic data.

Generates a stack with known per-strip planar tilt residuals (the failure
mode that drives the −1.6 m/yr Eulerian bias on Nansen), then compares
two flavours of `fit_tilt_stack`:

  - **static-only** (legacy): observations gated by `control_mask`
    (static-control pixels — rock + grounded). Per-strip tilts
    constrained only by static observations.
  - **joint-LSQ** (Shean): observations from *all* valid pixels (static
    + floating). Per-pixel intercept + dhdt are nuisance variables.
    Per-strip tilts get leverage from every floating observation too.

For each flavour, we apply the recovered tilts to the noisy stack, then
run all four operational solvers and report mean melt-rate recovery.

Run::

    python -u tests/synthetic_tilt_joint_lsq.py
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.coregister.tilt import fit_tilt_stack
from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics import (
    forward,
    lagrangian_frame_stack,
    linear_inverse_dhdt_lagrangian_melt_rate,
    linear_inverse_lagrangian_melt_rate,
)
from stereo_melt.melt import eulerian_melt_rate, lagrangian_melt_rate

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
SECONDS_PER_YEAR = 86400.0 * 365.25


def build_truth(L=60_000.0, res=250.0, m_bg=3.0, m_peak=5.0, sigma=4_000.0):
    x = np.arange(-L / 2, L / 2, res)
    y = np.arange(L / 2, -L / 2, -res)
    X, Y = np.meshgrid(x, y)
    bump = m_peak * np.exp(-0.5 * ((X - 8000) ** 2 + (Y + 6000) ** 2) / sigma**2)
    return xr.DataArray(
        m_bg + bump, dims=("y", "x"),
        coords={"y": y, "x": x}, name="m_true",
    )


def add_per_strip_tilt_with_known(h_stack, tilt_amp_m=0.5, seed=23):
    """Add per-epoch αx, αy, αz and return them so we can verify recovery."""
    rng = np.random.default_rng(seed)
    out = h_stack.values.copy()
    x = h_stack["x"].values; y = h_stack["y"].values
    Lx = x.max() - x.min(); Ly = y.max() - y.min()
    XN = (x[None, :] - x.mean()) / (0.5 * Lx)
    YN = (y[:, None] - y.mean()) / (0.5 * Ly)
    nt = out.shape[0]
    ax_true = np.zeros(nt); ay_true = np.zeros(nt); az_true = np.zeros(nt)
    for k in range(nt):
        ax = rng.normal(0, tilt_amp_m); ay = rng.normal(0, tilt_amp_m); az = rng.normal(0, tilt_amp_m)
        ax_true[k] = ax; ay_true[k] = ay; az_true[k] = az
        out[k] = out[k] + ax * XN + ay * YN + az
    return (
        xr.DataArray(out, dims=h_stack.dims, coords=h_stack.coords, name=h_stack.name),
        ax_true, ay_true, az_true,
    )


def add_strip_coverage_mask(h_stack, coverage_frac=0.30, seed=11):
    """Each epoch keeps a random ~`coverage_frac` rectangular sub-slab."""
    rng = np.random.default_rng(seed)
    out = h_stack.values.copy()
    nt, ny, nx = out.shape
    for k in range(nt):
        target_pix = int(coverage_frac * ny * nx)
        h = max(int(np.sqrt(target_pix * (rng.uniform(0.5, 2.0)))), 8)
        h = min(h, ny)
        w = max(target_pix // max(h, 1), 8)
        w = min(w, nx)
        y0 = rng.integers(0, ny - h + 1)
        x0 = rng.integers(0, nx - w + 1)
        keep = np.zeros((ny, nx), dtype=bool)
        keep[y0:y0 + h, x0:x0 + w] = True
        out[k] = np.where(keep, out[k], np.nan)
    return xr.DataArray(out, dims=h_stack.dims, coords=h_stack.coords, name=h_stack.name)


def stats(name, m, m_true, mask):
    v = m.values; t = m_true.values
    finite = np.isfinite(v) & np.isfinite(t) & mask
    if not finite.any():
        return {"name": name, "mean_err": np.nan, "rmse": np.nan}
    iy, ix = np.unravel_index(np.argmax(t), t.shape)
    pw = v[max(iy-2,0):iy+3, max(ix-2,0):ix+3]
    return {
        "name": name,
        "mean_rec": float(np.mean(v[finite])),
        "mean_err": float(np.mean(v[finite]) - np.mean(t[finite])),
        "peak_rec": float(np.nanmax(pw)),
        "peak_err": float(np.nanmax(pw) - float(t.max())),
        "rmse": float(np.sqrt(np.mean((v[finite] - t[finite]) ** 2))),
    }


def run_inverses(h_stack, m_true, H, vx_value=0.0):
    """Run all four inverses on a given (corrected) stack."""
    vx = xr.DataArray(np.full_like(m_true.values, vx_value), dims=m_true.dims, coords=m_true.coords)
    vy = xr.zeros_like(vx)
    floating = xr.DataArray(np.ones_like(m_true.values, dtype=bool), dims=m_true.dims, coords=m_true.coords)
    a_dot = xr.zeros_like(m_true)

    eul = eulerian_melt_rate(h_stack, vx, vy, a_dot=a_dot, robust_dh_dt=True).melt_rate
    lagr = lagrangian_melt_rate(
        h_stack, vx, vy, a_dot=a_dot, dt_yr=0.05, seed_stride=2, pairs="all", min_dt_yr=2/12
    ).melt_rate
    h_lag = lagrangian_frame_stack(h_stack, vx, vy, dt_yr=0.05)
    closed_dct = linear_inverse_lagrangian_melt_rate(
        h_stack, vx, vy, floating_mask=floating, H_ref=H,
        boundary_fix="off", transform="dct", reg=1e-1, dt_yr=0.05,
        h_lag_precomputed=h_lag,
    ).melt_rate
    dhdt_dct = linear_inverse_dhdt_lagrangian_melt_rate(
        h_stack, vx, vy, floating_mask=floating, H_ref=H,
        transform="dct", reg=10.0, min_n_obs=3, robust_dh_dt=True,
        dt_yr=0.05, h_lag_precomputed=h_lag,
    ).melt_rate
    mask = floating.values.astype(bool)
    return [
        stats("Eulerian", eul, m_true, mask),
        stats("Lagrangian", lagr, m_true, mask),
        stats("closed-DCT", closed_dct, m_true, mask),
        stats("dh/dt-DCT", dhdt_dct, m_true, mask),
    ]


def main():
    H = 500.0
    n_epochs = 11
    t_window_yr = 4.0

    m_true = build_truth()
    print(f"m_true: mean={float(m_true.mean()):+.3f}, peak={float(m_true.max()):+.2f}")

    times_s = np.linspace(0.0, t_window_yr * SECONDS_PER_YEAR, n_epochs)
    times_dt = pd.to_datetime("2020-01-01") + pd.to_timedelta(times_s, unit="s")
    h_xr = forward(m_true, H=H, eta_bar=1e14, alpha=0.0, gamma=0.0,
                   stationary=True, times=xr.DataArray(times_s, dims="t"))
    h_stack_clean = xr.DataArray(
        h_xr.values, dims=("time", "y", "x"),
        coords={"time": times_dt, "y": m_true["y"].values, "x": m_true["x"].values},
        name="h",
    )

    # Static control mask: assume the "rock" region is the left half.
    # Floating mask: the rest. This mimics REMA's situation: ~half static
    # control, ~half floating, with the floating area being where we want
    # the melt-rate inversion to apply.
    ny, nx = m_true.shape
    static = np.zeros((ny, nx), dtype=bool)
    static[:, :nx // 2] = True   # left half = "static" (rock + grounded)
    floating = ~static
    print(f"\nstatic mask: {static.mean():.0%},  floating mask: {floating.mean():.0%}")

    # Pin static-region pixels to t=0 — real rock is time-invariant. Without
    # this, every pixel evolves under `forward()` and rock provides no DC
    # information, leaving αz_mean unidentifiable in the joint LSQ.
    h_stack_clean.values[1:, static] = h_stack_clean.values[0, static]
    print(f"pinned static rock to t=0 across all epochs")

    # Realistic noise: 30% per-epoch coverage gaps + per-strip planar tilts.
    # The coverage gaps are critical — without them, random per-strip tilts
    # cancel in the per-pixel OLS slope and don't bias the mean. With
    # coverage gaps, per-pixel time series are unbalanced and tilts bias
    # individual slopes that don't cancel basin-wide.
    h_with_gaps = add_strip_coverage_mask(h_stack_clean, coverage_frac=0.30, seed=11)
    h_noisy, ax_true, ay_true, az_true = add_per_strip_tilt_with_known(h_with_gaps)
    print(f"Injected per-strip tilts:")
    print(f"  σ_αx = {ax_true.std():.3f} m,  σ_αy = {ay_true.std():.3f} m,  σ_αz = {az_true.std():.3f} m")
    print(f"  per-epoch coverage: 30% rectangular sub-slabs")

    # --- Reference: no tilt removal at all (truth-only baseline) ---
    print("\n=== A. NO TILT FIT (reference: just the noisy stack) ===")
    rows_a = run_inverses(h_noisy, m_true, H)
    for r in rows_a:
        print(f"  {r['name']:<11s}  mean_err={r['mean_err']:+.3f}  rmse={r['rmse']:.2f}")

    # --- Static-only LSQ ---
    print("\n=== B. STATIC-ONLY LSQ (legacy: control_mask gates observations) ===")
    static_da = xr.DataArray(static, dims=("y", "x"), coords={"y": m_true["y"], "x": m_true["x"]})
    params_b, h_corr_b = fit_tilt_stack(
        h_noisy, control_mask=static_da, min_width=10000.0,
    )
    az_b = params_b["tilt_dz"].values
    print(f"  recovered αz: rms_err = {np.sqrt(np.mean((az_b - az_true)**2)):.3f} m")
    rows_b = run_inverses(h_corr_b, m_true, H)
    for r in rows_b:
        print(f"  {r['name']:<11s}  mean_err={r['mean_err']:+.3f}  rmse={r['rmse']:.2f}")

    # --- Shean-style joint LSQ with all observations ---
    print("\n=== C. JOINT LSQ (Shean: observation_mask = full grid) ===")
    obs_full = xr.DataArray(np.ones((ny, nx), dtype=bool), dims=("y", "x"),
                            coords={"y": m_true["y"], "x": m_true["x"]})
    params_c, h_corr_c = fit_tilt_stack(
        h_noisy, control_mask=static_da, observation_mask=obs_full, min_width=10000.0,
    )
    az_c = params_c["tilt_dz"].values
    print(f"  recovered αz: rms_err = {np.sqrt(np.mean((az_c - az_true)**2)):.3f} m")
    rows_c = run_inverses(h_corr_c, m_true, H)
    for r in rows_c:
        print(f"  {r['name']:<11s}  mean_err={r['mean_err']:+.3f}  rmse={r['rmse']:.2f}")

    # --- Summary plot ---
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    width = 0.27
    methods = [r["name"] for r in rows_a]
    x_pos = np.arange(len(methods))
    a = [r["mean_err"] for r in rows_a]
    b = [r["mean_err"] for r in rows_b]
    c = [r["mean_err"] for r in rows_c]
    ax.bar(x_pos - width, a, width, label="A. no tilt fit (raw noisy)")
    ax.bar(x_pos,         b, width, label="B. static-only LSQ (legacy)")
    ax.bar(x_pos + width, c, width, label="C. joint LSQ (Shean / ndinterp)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x_pos); ax.set_xticklabels(methods)
    ax.set_ylabel("mean(recovered) − mean(true)  [m ice/yr]")
    ax.set_title("Joint-LSQ tilt-fit kills the per-strip tilt bias\n"
                 f"injected σ_αz = {az_true.std():.2f} m, σ_αx,y = {ax_true.std():.2f} m\n"
                 f"αz rms_err: B={np.sqrt(np.mean((az_b-az_true)**2)):.3f} m, "
                 f"C={np.sqrt(np.mean((az_c-az_true)**2)):.3f} m")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = OUT_DIR / "synthetic_tilt_joint_lsq.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
