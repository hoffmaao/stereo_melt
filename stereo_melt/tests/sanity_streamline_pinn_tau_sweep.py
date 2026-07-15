r"""Velocity-uncertainty resilience as a function of max_tau_yr.

The noisy synthetic test showed that 10% velocity uncertainty kills the
PINN recovery completely (corr drops 0.94 → 0.03). The hypothesis is
that velocity errors amplify into streamline-label errors linearly with
trajectory length. If true, shorter max_tau should restore recovery.

This script re-runs ONE noise configuration (per-strip 5 m + per-pixel
0.3 m + 10% velocity multiplicative) at a sweep of max_tau values, and
reports (corr, RMSE, coverage). If corr recovers at small max_tau, the
fix for Nansen is to bound max_tau to a physically realistic shelf
residence time rather than the current 200 yr.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xarray as xr

from sanity_streamline_pinn_noisy import (  # noqa: E402
    build_clean_shelf,
    build_noisy_stack,
    perturb_velocity,
)

from stereo_melt.dynamics.streamline_pinn import (
    fit_streamline_pinn,
    predict_melt_grid,
)
from stereo_melt.kinematics import (
    backward_advect_pixels,
    build_streamline_dataset,
    divergence,
)


def run_once(h_stack, vx, vy, grounded, floating, m_true, x_coords, y_coords,
             *, max_tau_yr: float, n_epochs: int = 30):
    vx_da = xr.DataArray(vx, dims=("y", "x"), coords={"y": y_coords, "x": x_coords})
    vy_da = xr.DataArray(vy, dims=("y", "x"), coords={"y": y_coords, "x": x_coords})

    trajectories = backward_advect_pixels(
        x_coords=x_coords, y_coords=y_coords,
        vx=vx, vy=vy,
        grounded_mask=grounded, floating_mask=floating,
        dt_yr=0.05, max_tau_yr=max_tau_yr,
    )
    n_success = int((trajectories["reason"] == 0).sum())
    n_floating = int(floating.sum())

    a_dot = xr.DataArray(np.zeros_like(vx), dims=("y", "x"),
                         coords={"y": y_coords, "x": x_coords})
    d_fac = xr.DataArray(np.zeros_like(vx), dims=("y", "x"),
                         coords={"y": y_coords, "x": x_coords})
    vdiv = divergence(vx_da, vy_da)
    sigma_per_epoch = np.full(h_stack.sizes["time"], 0.1, dtype=np.float64)

    dataset = build_streamline_dataset(
        h_stack=h_stack, trajectories=trajectories,
        sigma_per_epoch=sigma_per_epoch,
        a_dot=a_dot, vdiv=vdiv, d_fac=d_fac,
    )

    result = fit_streamline_pinn(
        dataset, hidden_dim=192, n_hidden_layers=5, n_fourier_features=16,
        n_epochs=n_epochs, batch_size=4096, lr=2e-3,
        test_frac=0.1, device="cpu", verbose=False,
    )

    pred = predict_melt_grid(
        model=result.model, trajectories=trajectories,
        a_dot=a_dot, vdiv=vdiv, d_fac=d_fac, device="cpu",
    )
    interior = (slice(5, -5), slice(5, -5))
    m_hat_i = pred.melt_rate.values[interior]
    m_true_i = m_true[interior]
    valid = np.isfinite(m_hat_i) & np.isfinite(m_true_i)
    err = m_hat_i[valid] - m_true_i[valid]
    return {
        "max_tau": max_tau_yr,
        "coverage": 100 * n_success / max(n_floating, 1),
        "corr": float(np.corrcoef(m_hat_i[valid], m_true_i[valid])[0, 1]),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "loss": float(result.train_losses[-1]),
    }


def main():
    print("Building clean shelf, then applying tier-3 noise...")
    x_coords, y_coords, vx_2d, vy_2d, grounded, floating, m_true, h_true, times = (
        build_clean_shelf(n_epochs=10)
    )
    rng = np.random.default_rng(42)
    h_stack = build_noisy_stack(
        h_true=h_true, x_coords=x_coords, y_coords=y_coords, times=times,
        floating=floating, strip_offset_sigma=5.0, pixel_noise_sigma=0.3, rng=rng,
    )
    vx_n, vy_n = perturb_velocity(vx_2d, vy_2d, mult_sigma=0.10, gap_idx=None, rng=rng)

    rows = []
    for max_tau in [15.0, 30.0, 60.0, 100.0, 200.0, 300.0]:
        print(f"\n--- max_tau_yr = {max_tau} ---")
        r = run_once(
            h_stack=h_stack, vx=vx_n, vy=vy_n,
            grounded=grounded, floating=floating, m_true=m_true,
            x_coords=x_coords, y_coords=y_coords,
            max_tau_yr=max_tau, n_epochs=30,
        )
        print(f"  coverage={r['coverage']:.1f}%  corr={r['corr']:+.3f}  "
              f"RMSE={r['rmse']:.3f}  bias={r['bias']:+.3f}  loss={r['loss']:.2e}")
        rows.append(r)

    print("\n" + "=" * 72)
    print(f"{'max_tau (yr)':>14} {'cov%':>8} {'corr':>8} {'RMSE':>10} {'bias':>10} {'loss':>12}")
    print("=" * 72)
    for r in rows:
        print(f"{r['max_tau']:>14.0f} {r['coverage']:>7.1f}% {r['corr']:>+8.3f} "
              f"{r['rmse']:>9.3f} {r['bias']:>+9.3f} {r['loss']:>12.2e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
