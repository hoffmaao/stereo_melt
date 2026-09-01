"""Realistic synthetic recovery test mirroring the REMA-strip failure modes.

The earlier ``synthetic_recovery_demo.py`` had every epoch fully covered
by clean iid noise, so it could not expose the actual bias mechanism on
real basins. This test layers, one at a time, the failure modes we
suspect drive the Nansen disagreement with Davison:

  S0  truth-only — known m_true(y, x), forward-modelled stack
  S1  + per-strip random sub-rectangle coverage masks (30% per epoch)
  S2  + per-strip linear tilt residual (αx, αy, αz drawn from N(0, σ))
  S3  + per-strip non-planar residual (low-frequency Gaussian fields)
  S4  S3 with advection (uniform vx) so flux divergence is non-zero
       and ∇·(Hu) must be subtracted in the DC splice

We run all four operational solvers (Eulerian, Lagrangian path-int,
closed-DCT, dh/dt-DCT) at each level. Each level tells us which fix
suffices and which problem it leaves behind.

Outputs:

  tests/figures/synthetic_realistic_<S0..S4>.png   per-scenario panel
  tests/figures/synthetic_realistic_summary.png    bar-chart of biases

Run::

    python -u tests/synthetic_realistic_recovery.py
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
from scipy.ndimage import gaussian_filter

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
R_HYDRO = rhow / (rhow - rhoi)


def build_truth(L=60_000.0, res=250.0, m_bg=3.0, m_peak=5.0, sigma=4_000.0):
    x = np.arange(-L / 2, L / 2, res)
    y = np.arange(L / 2, -L / 2, -res)
    X, Y = np.meshgrid(x, y)
    bump = m_peak * np.exp(-0.5 * ((X - 8000) ** 2 + (Y + 6000) ** 2) / sigma**2)
    return xr.DataArray(
        m_bg + bump, dims=("y", "x"),
        coords={"y": y, "x": x}, name="m_true",
    )


def add_strip_coverage_mask(h_stack, coverage_frac=0.30, seed=11):
    """Each epoch keeps only a random ~`coverage_frac` rectangular sub-slab."""
    rng = np.random.default_rng(seed)
    out = h_stack.values.copy()
    nt, ny, nx = out.shape
    for k in range(nt):
        # Random axis-aligned rectangle covering coverage_frac of pixels
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


def add_per_strip_tilt(h_stack, tilt_amp_m=0.5, seed=23):
    """Add per-epoch linear plane (αx·x + αy·y + αz)."""
    rng = np.random.default_rng(seed)
    out = h_stack.values.copy()
    x = h_stack["x"].values; y = h_stack["y"].values
    Lx = x.max() - x.min(); Ly = y.max() - y.min()
    XN = (x[None, :] - x.mean()) / (0.5 * Lx)
    YN = (y[:, None] - y.mean()) / (0.5 * Ly)
    for k in range(out.shape[0]):
        ax = rng.normal(0, tilt_amp_m)
        ay = rng.normal(0, tilt_amp_m)
        az = rng.normal(0, tilt_amp_m)
        plane = ax * XN + ay * YN + az
        out[k] = out[k] + plane
    return xr.DataArray(out, dims=h_stack.dims, coords=h_stack.coords, name=h_stack.name)


def add_per_strip_nonplanar_residual(h_stack, amp_m=0.2, scale_pix=20, seed=37):
    """Add per-epoch low-frequency Gaussian field (non-planar residual)."""
    rng = np.random.default_rng(seed)
    out = h_stack.values.copy()
    nt, ny, nx = out.shape
    for k in range(nt):
        noise = rng.normal(0.0, 1.0, (ny, nx))
        smooth = gaussian_filter(noise, sigma=scale_pix, mode="reflect")
        smooth /= max(float(smooth.std()), 1e-12)
        out[k] = out[k] + amp_m * smooth
    return xr.DataArray(out, dims=h_stack.dims, coords=h_stack.coords, name=h_stack.name)


def stats_vs_truth(name, m, m_true_shean, mask):
    """Recovery statistics in the Shean public convention.

    ``m_true_shean`` carries negative=melt; the localized melt feature is
    the most-negative cell. ``np.argmin`` finds it; ``nanmin`` reads the
    recovered peak. Pass the Shean-convention truth — caller is responsible
    for negating the Stubblefield-convention prescription.
    """
    v = m.values
    t = m_true_shean.values
    finite = np.isfinite(v) & np.isfinite(t) & mask
    if not finite.any():
        return {"name": name, "mean_err": np.nan, "peak_err": np.nan, "rmse": np.nan}
    iy, ix = np.unravel_index(np.argmin(t), t.shape)
    iy0, iy1 = max(iy-2, 0), iy+3
    ix0, ix1 = max(ix-2, 0), ix+3
    peak_window = v[iy0:iy1, ix0:ix1]
    return {
        "name": name,
        "mean_rec": float(np.mean(v[finite])),
        "mean_err": float(np.mean(v[finite]) - np.mean(t[finite])),
        "peak_rec": float(np.nanmin(peak_window)),
        "peak_err": float(np.nanmin(peak_window) - float(t.min())),
        "rmse": float(np.sqrt(np.mean((v[finite] - t[finite]) ** 2))),
    }


def run_one_scenario(scenario, m_true, *, vx_value=0.0, H=500.0, n_epochs=11,
                     t_window_yr=4.0, vx_nonuniform=False):
    """Build the synthetic stack with per-scenario noise, run all 4 methods."""
    times_s = np.linspace(0.0, t_window_yr * SECONDS_PER_YEAR, n_epochs)
    times_dt = pd.to_datetime("2020-01-01") + pd.to_timedelta(times_s, unit="s")

    alpha_forward = vx_value * (2.0 * 1e14 / (rhoi * 9.81 * H)) / H  # nondim
    h_xr = forward(
        m_true, H=H, eta_bar=1e14,
        alpha=alpha_forward, gamma=0.0,
        stationary=True,
        times=xr.DataArray(times_s, dims="t"),
    )
    h_stack = xr.DataArray(
        h_xr.values, dims=("time", "y", "x"),
        coords={"time": times_dt, "y": m_true["y"].values, "x": m_true["x"].values},
        name="h_freeboard",
    )

    # Layer noise per scenario
    if scenario in ("S1", "S2", "S3", "S4", "S5", "S6"):
        h_stack = add_strip_coverage_mask(h_stack, coverage_frac=0.30, seed=11)
    if scenario in ("S2", "S3", "S4", "S6"):
        h_stack = add_per_strip_tilt(h_stack, tilt_amp_m=0.5, seed=23)
    if scenario in ("S3", "S4", "S6"):
        h_stack = add_per_strip_nonplanar_residual(h_stack, amp_m=0.2, seed=37)

    # Velocity field
    if vx_nonuniform:
        # Across-channel ramp: u_x grows from 100 to 300 m/yr along +x.
        # Spatially-varying velocity violates the Lagrangian-frame assumption
        # that bulk advection can be absorbed by translating with ⟨u⟩, leaving
        # a residual α(x,y) the constant-α spectral kernel can't honour.
        x = m_true["x"].values
        u_profile = 100.0 + 200.0 * (x - x.min()) / (x.max() - x.min())
        vx_field = np.broadcast_to(u_profile[None, :], m_true.shape).copy()
        vx = xr.DataArray(vx_field, dims=m_true.dims, coords=m_true.coords)
    else:
        vx = xr.DataArray(
            np.full_like(m_true.values, vx_value),
            dims=m_true.dims, coords=m_true.coords,
        )
    vy = xr.zeros_like(vx)
    floating = xr.DataArray(
        np.ones_like(m_true.values, dtype=bool),
        dims=m_true.dims, coords=m_true.coords,
    )
    a_dot = xr.zeros_like(m_true)

    # Run inverses
    eul = eulerian_melt_rate(
        h_stack, vx, vy, a_dot=a_dot, robust_dh_dt=True,
    ).melt_rate
    lagr = lagrangian_melt_rate(
        h_stack, vx, vy, a_dot=a_dot,
        dt_yr=0.05, seed_stride=2, pairs="all",
        min_dt_yr=2.0/12.0,
    ).melt_rate
    h_lag = lagrangian_frame_stack(h_stack, vx, vy, dt_yr=0.05)
    closed_dct = linear_inverse_lagrangian_melt_rate(
        h_stack, vx, vy,
        floating_mask=floating, H_ref=H,
        boundary_fix="off", transform="dct", reg=1e-1,
        dt_yr=0.05, h_lag_precomputed=h_lag,
    ).melt_rate
    dhdt_dct = linear_inverse_dhdt_lagrangian_melt_rate(
        h_stack, vx, vy,
        floating_mask=floating, H_ref=H,
        transform="dct", reg=10.0, min_n_obs=3,
        robust_dh_dt=True,
        dt_yr=0.05, h_lag_precomputed=h_lag,
    ).melt_rate

    mask = floating.values.astype(bool)
    # m_true is the Stubblefield-convention prescription (positive=melt) the
    # forward model needs. Inverses now publish Shean convention; compare
    # against the negated truth.
    m_true_shean = -m_true
    rows = [
        stats_vs_truth("Eulerian", eul, m_true_shean, mask),
        stats_vs_truth("Lagrangian", lagr, m_true_shean, mask),
        stats_vs_truth("closed-DCT", closed_dct, m_true_shean, mask),
        stats_vs_truth("dh/dt-DCT", dhdt_dct, m_true_shean, mask),
    ]
    return rows, eul, lagr, closed_dct, dhdt_dct


def plot_scenario(scenario, m_true, eul, lagr, closed_dct, dhdt_dct, rows):
    fig, axes = plt.subplots(1, 5, figsize=(5 * 4.4, 5.2), constrained_layout=True)
    extent = (
        float(m_true["x"].min()), float(m_true["x"].max()),
        float(m_true["y"].min()), float(m_true["y"].max()),
    )
    m_true_shean = -m_true
    panels = [
        ("m_true (Shean)", m_true_shean,
         f"true: mean={float(m_true_shean.mean()):+.2f}, peak={float(m_true_shean.min()):+.2f}"),
        ("Eulerian", eul, _annot(rows[0])),
        ("Lagrangian path-int", lagr, _annot(rows[1])),
        ("closed-form DCT", closed_dct, _annot(rows[2])),
        ("dh/dt-DCT", dhdt_dct, _annot(rows[3])),
    ]
    for ax, (title, da, sub) in zip(axes, panels):
        im = ax.imshow(
            da.values, extent=extent, origin="upper",
            cmap="RdBu_r", vmin=-10, vmax=10,
            aspect="equal", interpolation="nearest",
        )
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, label="m ice/yr")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(_scenario_caption(scenario), fontsize=11)
    out = OUT_DIR / f"synthetic_realistic_{scenario}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _annot(r):
    return f"mean_err={r['mean_err']:+.2f}  peak_err={r['peak_err']:+.2f}  rmse={r['rmse']:.2f}"


def _scenario_caption(scenario):
    descs = {
        "S0": "S0 — clean forward model, no noise, no coverage gaps, vx=0",
        "S1": "S1 — S0 + 30% per-strip random rectangular coverage",
        "S2": "S2 — S1 + per-strip planar tilt residual (σ_αz=0.5 m)",
        "S3": "S3 — S2 + per-strip non-planar residual (low-freq, σ=0.2 m)",
        "S4": "S4 — S3 + uniform along-x advection (vx=200 m/yr) — flux div nonzero",
        "S5": "S5 — S1 + non-uniform vx (100→300 m/yr ramp) — tests Stokes inflow",
        "S6": "S6 — S5 + per-strip planar+nonplanar tilt — kitchen sink",
    }
    return descs.get(scenario, scenario)


def main():
    m_true = build_truth()
    print(
        f"m_true (Stubblefield convention, fed to forward()): "
        f"mean={float(m_true.mean()):+.3f}, peak={float(m_true.max()):+.2f} m/yr; "
        f"Shean form: mean={-float(m_true.mean()):+.3f}, peak={-float(m_true.max()):+.2f} m/yr"
    )
    summary = []
    scenarios = [
        ("S0", 0.0,   False),
        ("S1", 0.0,   False),
        ("S2", 0.0,   False),
        ("S3", 0.0,   False),
        ("S4", 200.0, False),
        ("S5", 0.0,   True),   # non-uniform vx, no per-strip tilt yet
        ("S6", 0.0,   True),   # non-uniform vx + per-strip tilt + nonplanar
    ]
    for scenario, vx_value, vx_nonuniform in scenarios:
        print(f"\n===== {scenario} =====")
        rows, *fields = run_one_scenario(
            scenario, m_true, vx_value=vx_value, vx_nonuniform=vx_nonuniform,
        )
        plot_scenario(scenario, m_true, *fields, rows)
        for r in rows:
            r2 = dict(r); r2["scenario"] = scenario; summary.append(r2)
        print(f"{'method':<13s}  {'mean_err':>10s}  {'peak_err':>10s}  {'rmse':>8s}")
        for r in rows:
            print(f"  {r['name']:<11s}  {r['mean_err']:>+10.3f}  "
                  f"{r['peak_err']:>+10.3f}  {r['rmse']:>8.3f}")

    # Summary bar chart: mean_err per (scenario, method)
    df = pd.DataFrame(summary)
    pivot = df.pivot(index="scenario", columns="name", values="mean_err")
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    pivot.plot.bar(ax=ax)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean(recovered) − mean(true)  [m ice/yr]")
    ax.set_title(
        "Synthetic recovery — mean-melt-rate bias by failure mode\n"
        "S0=clean  S1=+coverage  S2=+plane  S3=+nonplanar  S4=+advection"
    )
    ax.grid(True, alpha=0.3)
    out = OUT_DIR / "synthetic_realistic_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsummary -> {out}")
    print("\n=== Cross-scenario mean_err table ===")
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    main()
