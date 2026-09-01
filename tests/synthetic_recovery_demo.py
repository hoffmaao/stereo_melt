"""End-to-end synthetic recovery demonstration on a known melt field.

Builds a 2-D basal melt-rate field with two distinguishable signals:

  - **Mean melt** (spatial constant) at +3 m ice/yr — the basin-wide
    background that should be recovered to within a fraction of m/yr.
  - **Local anomaly** (Gaussian bump) at +5 m ice/yr, sigma = 4 km,
    placed off-center — the localized signal that should be recovered
    in both location and amplitude.

Forward-models the corresponding freeboard stack via
:func:`stereo_melt.dynamics.forward`, optionally adds per-epoch tilt
residual + per-pixel noise, then inverts with the four methods we use
on real basins (Eulerian, Lagrangian path-int, closed-DCT, dh/dt-DCT).

For each method we report

  - ``mean(recovered) - mean(true)``   (mean-melt-rate accuracy)
  - peak amplitude of recovered anomaly at the true bump location
  - rms over the central ROI (combined mean+anomaly accuracy)

Outputs a 1x5 figure: m_true alongside the four recoveries on a single
±10 m/yr RdBu scale, with stats annotated per panel.

Run::

    python -m stereo_melt.tests.synthetic_recovery_demo
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


def build_truth(
    L: float = 60_000.0,
    res: float = 250.0,
    m_bg: float = 3.0,
    m_peak: float = 5.0,
    sigma: float = 4_000.0,
    bump_xy: tuple[float, float] = (8_000.0, -6_000.0),
) -> xr.DataArray:
    """Return ``m_true(y, x)`` = uniform mean + Gaussian anomaly."""
    x = np.arange(-L / 2, L / 2, res)
    y = np.arange(L / 2, -L / 2, -res)  # descending
    X, Y = np.meshgrid(x, y)
    bx, by = bump_xy
    bump = m_peak * np.exp(-0.5 * ((X - bx) ** 2 + (Y - by) ** 2) / sigma**2)
    m = m_bg + bump
    return xr.DataArray(m, dims=("y", "x"), coords={"y": y, "x": x}, name="m_true")


def add_per_strip_noise(
    h_stack: xr.DataArray,
    seed: int = 7,
    tilt_amp_m: float = 0.5,
    pixel_sigma_m: float = 0.3,
) -> xr.DataArray:
    """Inject per-epoch random plane tilt + per-pixel iid Gaussian noise.

    Mimics the per-strip coregistration residual that the basin
    pipeline's tilt-fit absorbs. ``tilt_amp_m`` is the standard
    deviation of the per-epoch αz constant offset.
    """
    rng = np.random.default_rng(seed)
    nt, ny, nx = h_stack.shape
    out = h_stack.values.copy()
    x = h_stack["x"].values
    y = h_stack["y"].values
    Lx = x.max() - x.min()
    Ly = y.max() - y.min()
    Xn = (x - x.mean()) / (0.5 * Lx)
    Yn = (y - y.mean()) / (0.5 * Ly)
    XN, YN = np.meshgrid(Xn, Yn)
    for k in range(nt):
        ax = rng.normal(0.0, tilt_amp_m / Lx) * Lx
        ay = rng.normal(0.0, tilt_amp_m / Ly) * Ly
        az = rng.normal(0.0, tilt_amp_m)
        out[k] += ax * XN + ay * YN + az
        out[k] += rng.normal(0.0, pixel_sigma_m, (ny, nx))
    return xr.DataArray(
        out, dims=h_stack.dims, coords=h_stack.coords, name=h_stack.name,
    )


def main(*, with_noise: bool = True) -> None:
    H = 500.0
    eta_bar = 1e14
    n_epochs = 11
    t_window_yr = 4.0  # 11 epochs evenly spaced over 4 years

    # ---- truth ----
    # build_truth() returns m_true in Stubblefield convention (positive
    # Gaussian = melt feature) which is what the forward operator expects.
    # The library's PUBLIC inverse wrappers return Shean convention
    # (negative = melt), so we define m_true_shean for comparison and
    # plotting against the Shean-convention recoveries.
    m_true = build_truth()
    m_true_shean = -m_true
    m_true_shean.name = "m_true_shean"
    print(f"Built m_true: mean={float(m_true.mean()):+.3f}, "
          f"max={float(m_true.max()):+.3f} m ice/yr (Stubblefield/positive=melt); "
          f"shean form: mean={float(m_true_shean.mean()):+.3f}, "
          f"min={float(m_true_shean.min()):+.3f}, "
          f"shape={m_true.shape}")

    # ---- forward model: stationary m, n_epochs DEMs over t_window_yr ----
    times_s = np.linspace(0.0, t_window_yr * SECONDS_PER_YEAR, n_epochs)
    times_dt = pd.to_datetime("2020-01-01") + pd.to_timedelta(times_s, unit="s")
    print(f"Forward-modelling h_stack over {t_window_yr} yr, n={n_epochs}...")
    h_xr = forward(
        m_true, H=H, eta_bar=eta_bar, alpha=0.0, gamma=0.0,
        stationary=True,
        times=xr.DataArray(times_s, dims="t"),
    )
    h_stack = xr.DataArray(
        h_xr.values, dims=("time", "y", "x"),
        coords={"time": times_dt, "y": m_true["y"].values, "x": m_true["x"].values},
        name="h_freeboard",
    )

    if with_noise:
        h_stack = add_per_strip_noise(h_stack)
        print("  + per-strip tilt residual + per-pixel noise injected")

    # ---- velocity (zero), floating mask (full), SMB (zero) ----
    vx = xr.zeros_like(m_true)
    vy = xr.zeros_like(m_true)
    floating = xr.DataArray(
        np.ones_like(m_true.values, dtype=bool), dims=m_true.dims,
        coords=m_true.coords,
    )
    a_dot = xr.zeros_like(m_true)

    # ---- run inverses ----
    print("\n=== Eulerian ===")
    eul = eulerian_melt_rate(
        h_stack, vx, vy, a_dot=a_dot, robust_dh_dt=True,
    ).melt_rate

    print("=== Lagrangian path-int ===")
    lagr = lagrangian_melt_rate(
        h_stack, vx, vy, a_dot=a_dot,
        dt_yr=0.05, seed_stride=2, pairs="all",
        min_dt_yr=2.0 / 12.0,
    ).melt_rate

    print("=== closed-form DCT ===")
    h_lag = lagrangian_frame_stack(h_stack, vx, vy, dt_yr=0.05)
    # Synthetic h is freeboard *anomaly*, not absolute freeboard, so the
    # auto-derived H_ref via hydrostatic inversion is meaningless. Supply
    # the truth H so the kernel is normalized correctly.
    closed_dct = linear_inverse_lagrangian_melt_rate(
        h_stack, vx, vy,
        floating_mask=floating, H_ref=H,
        boundary_fix="off", transform="dct", reg=1e-1,
        dt_yr=0.05, h_lag_precomputed=h_lag,
    ).melt_rate

    print("=== dh/dt-DCT ===")
    dhdt_dct = linear_inverse_dhdt_lagrangian_melt_rate(
        h_stack, vx, vy,
        floating_mask=floating, H_ref=H,
        transform="dct", reg=10.0, min_n_obs=3,
        robust_dh_dt=True,
        dt_yr=0.05, h_lag_precomputed=h_lag,
    ).melt_rate

    # ---- evaluation (Shean convention; "peak melt" = min, not max) ----
    def evaluate(name: str, m: xr.DataArray) -> dict:
        v = m.values
        finite = np.isfinite(v)  # noqa: F841
        true = m_true_shean.values
        # peak at the truth-bump location: most-negative cell under Shean.
        bump_y = float(
            m_true_shean["y"].values[np.unravel_index(np.argmin(true), true.shape)[0]]
        )
        bump_x = float(
            m_true_shean["x"].values[np.unravel_index(np.argmin(true), true.shape)[1]]
        )
        # 5x5 window around peak
        iy = np.argmin(np.abs(m["y"].values - bump_y))
        ix = np.argmin(np.abs(m["x"].values - bump_x))
        peak_window = v[max(iy-2, 0):iy+3, max(ix-2, 0):ix+3]
        peak_val = float(np.nanmin(peak_window)) if peak_window.size else float("nan")
        return {
            "name": name,
            "mean_recovered": float(np.nanmean(v)),
            "mean_error": float(np.nanmean(v) - np.mean(true)),
            "peak_recovered": peak_val,
            "peak_error": peak_val - float(true.min()),
            "rmse": float(np.sqrt(np.nanmean((v - true) ** 2))),
        }

    rows = [
        evaluate("Eulerian", eul),
        evaluate("Lagrangian", lagr),
        evaluate("closed-DCT", closed_dct),
        evaluate("dh/dt-DCT", dhdt_dct),
    ]
    print("\nRecovery summary, Shean convention (m_true_shean mean={:+.2f}, peak={:+.2f} m/yr):".format(
        float(m_true_shean.mean()), float(m_true_shean.min()),
    ))
    print(f"{'method':<15s}  {'mean_rec':>10s}  {'mean_err':>10s}  "
          f"{'peak_rec':>10s}  {'peak_err':>10s}  {'rmse':>8s}")
    for r in rows:
        print(
            f"{r['name']:<15s}  "
            f"{r['mean_recovered']:>+10.3f}  {r['mean_error']:>+10.3f}  "
            f"{r['peak_recovered']:>+10.3f}  {r['peak_error']:>+10.3f}  "
            f"{r['rmse']:>8.3f}"
        )

    # ---- plot ----
    fig, axes = plt.subplots(1, 5, figsize=(5 * 4.4, 5.4), constrained_layout=True)
    panels = [
        ("m_true (Shean)", m_true_shean, "true: mean={:+.2f}, peak={:+.2f}".format(
            float(m_true_shean.mean()), float(m_true_shean.min())
        )),
        ("Eulerian", eul, _annot(rows[0])),
        ("Lagrangian path-int", lagr, _annot(rows[1])),
        ("closed-form DCT", closed_dct, _annot(rows[2])),
        ("dh/dt-DCT", dhdt_dct, _annot(rows[3])),
    ]
    extent = (
        float(m_true["x"].min()), float(m_true["x"].max()),
        float(m_true["y"].min()), float(m_true["y"].max()),
    )
    for ax, (title, da, sub) in zip(axes, panels):
        im = ax.imshow(
            da.values,
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=-10, vmax=10,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045, label="m ice/yr")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"Synthetic recovery (Shean: negative=melt): mean melt -3 m/yr + "
        f"Gaussian anomaly (-5 m/yr, σ=4 km).  Forward model: stationary "
        f"Stubblefield over {n_epochs} epochs, {t_window_yr:.1f} yr, "
        f"H={H} m{', + per-strip + per-pixel noise' if with_noise else ', noiseless'}.",
        fontsize=12,
    )
    out = OUT_DIR / (
        "synthetic_recovery_noise.png" if with_noise else "synthetic_recovery_clean.png"
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


def _annot(r: dict) -> str:
    return (
        f"mean_err={r['mean_error']:+.2f}  "
        f"peak_err={r['peak_error']:+.2f}  "
        f"rmse={r['rmse']:.2f}"
    )


if __name__ == "__main__":
    print("=== noiseless ===")
    main(with_noise=False)
    print("\n=== with noise (per-strip tilt + per-pixel iid) ===")
    main(with_noise=True)
