r"""Velocity noise: pixel-independent vs spatially-correlated.

Pixel-independent 10% velocity perturbation collapsed recovery in
sanity_streamline_pinn_noisy (corr 0.94 → 0.03). The mechanism is the
spurious ∇·u this creates — std(∂v/∂x) ~ ε·v/dx ~ 0.08/yr, multiplied
by H ≈ 400 m, gives ~32 m/yr of spurious mass-balance term that
saturates m̂.

Real MEaSUREs velocity uncertainty is spatially correlated (phase
coherence scales of a few km). This script applies smooth velocity
perturbation at varying correlation lengths and reports recovery, to
test whether the architecture is robust to *physical* velocity
uncertainty as opposed to *unphysical* white noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from sanity_streamline_pinn_noisy import (  # noqa: E402
    build_clean_shelf,
    build_noisy_stack,
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


def perturb_velocity_smooth(vx, vy, *, mult_sigma: float,
                            correlation_length_cells: float, rng):
    """Smooth multiplicative velocity perturbation.

    Generates pixel-independent N(0, mult_sigma) noise, smooths it with
    a Gaussian filter at ``correlation_length_cells`` σ, renormalizes
    to recover the requested ``mult_sigma`` after smoothing, and
    applies multiplicatively to both vx and vy.
    """
    raw = rng.normal(0.0, 1.0, size=vx.shape)
    if correlation_length_cells > 0:
        smooth = gaussian_filter(raw, sigma=correlation_length_cells, mode="reflect")
    else:
        smooth = raw
    # Renormalize so the (smoothed) field has the requested std
    s = float(np.std(smooth))
    if s > 0:
        smooth = smooth / s * mult_sigma
    return vx * (1.0 + smooth), vy * (1.0 + smooth)


def run_once(h_stack, vx, vy, grounded, floating, m_true, x_coords, y_coords,
             *, max_tau_yr: float = 300.0, n_epochs: int = 30):
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
    vdiv_field = vdiv.values
    return {
        "coverage": 100 * n_success / max(n_floating, 1),
        "corr": float(np.corrcoef(m_hat_i[valid], m_true_i[valid])[0, 1]),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "loss": float(result.train_losses[-1]),
        "vdiv_std": float(np.nanstd(vdiv_field)),
    }


def main():
    print("Building clean shelf with tier-3 h noise (strip 5m + pix 0.3m)...")
    x_coords, y_coords, vx_2d, vy_2d, grounded, floating, m_true, h_true, times = (
        build_clean_shelf(n_epochs=10)
    )
    rng = np.random.default_rng(42)
    h_stack = build_noisy_stack(
        h_true=h_true, x_coords=x_coords, y_coords=y_coords, times=times,
        floating=floating, strip_offset_sigma=5.0, pixel_noise_sigma=0.3, rng=rng,
    )

    # correlation_length_cells: 0 = pixel-independent, 4 = ~1 km, 20 = ~5 km, 40 = ~10 km
    rows = []
    for corr_cells in [0.0, 2.0, 4.0, 8.0, 20.0, 40.0]:
        vx_n, vy_n = perturb_velocity_smooth(
            vx_2d, vy_2d, mult_sigma=0.10, correlation_length_cells=corr_cells, rng=rng,
        )
        r = run_once(
            h_stack=h_stack, vx=vx_n, vy=vy_n,
            grounded=grounded, floating=floating, m_true=m_true,
            x_coords=x_coords, y_coords=y_coords, max_tau_yr=300.0, n_epochs=30,
        )
        r["corr_cells"] = corr_cells
        r["corr_km"] = corr_cells * 250 / 1000
        rows.append(r)
        print(f"  corr_len={corr_cells:>4.1f} cells ({r['corr_km']:>4.1f} km)  "
              f"vdiv_std={r['vdiv_std']:.4f}/yr  cov={r['coverage']:>5.1f}%  "
              f"corr={r['corr']:>+6.3f}  RMSE={r['rmse']:>6.3f}  loss={r['loss']:.2e}")

    print("\n" + "=" * 80)
    print(f"{'corr len (km)':>14} {'vdiv std (/yr)':>17} {'cov%':>8} "
          f"{'corr':>8} {'RMSE':>10}")
    print("=" * 80)
    for r in rows:
        print(f"{r['corr_km']:>14.2f} {r['vdiv_std']:>17.4f} {r['coverage']:>7.1f}% "
              f"{r['corr']:>+8.3f} {r['rmse']:>9.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
