r"""Noisy synthetic recovery — does the v2 streamline PINN survive
Nansen-grade contamination?

Same curved-GL + Gaussian-m_true synthetic as
`sanity_streamline_pinn_spatial.py`, but with progressively-added
contamination that mimics real-data quality issues at Nansen:

    tier 0  clean (baseline)
    tier 1  + per-strip h offset  N(0, 5 m)      ← coreg DC bias
    tier 2  + per-pixel obs noise N(0, 0.3 m)
    tier 3  + 10% multiplicative velocity perturbation
    tier 4  + velocity gap (zero stripe through fast-flow zone)

Reports spatial correlation, RMSE, and bias of recovered m̂ vs m_true at
each tier. If tier 1 holds at corr > 0.9, the per-strip offset embedding
is doing its job. If tier 4 holds at corr > 0.7, Nansen's residual is
data-recoverable. If recovery collapses at tier 1-2 already, the
architecture has a limit the clean test missed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import cumulative_trapezoid

from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics.streamline_pinn import (
    fit_streamline_pinn,
    predict_melt_grid,
)
from stereo_melt.kinematics import (
    backward_advect_pixels,
    build_streamline_dataset,
    divergence,
)


def build_clean_shelf(n_epochs: int = 10):
    x = np.arange(-20000.0, 20000.0, 250.0)
    y = np.arange(20000.0, -20000.0, -250.0)
    Y, X = np.meshgrid(y, x, indexing="ij")

    vx_2d = np.full_like(X, 200.0)
    vy_2d = np.zeros_like(X)
    v = 200.0

    x_gl_y = -15000.0 + 5000.0 * (Y / 20000.0) ** 2
    grounded = X < x_gl_y
    floating = ~grounded

    m_max = -3.0
    x_c, y_c = -5000.0, 0.0
    sigma_m = 5000.0
    m_true = m_max * np.exp(-((X - x_c) ** 2 + (Y - y_c) ** 2) / (2 * sigma_m**2))
    m_true = np.where(floating, m_true, np.nan)

    H_GL = 500.0
    H_true = np.full_like(X, np.nan)
    for j in range(X.shape[0]):
        x_gl_j = float(x_gl_y[j, 0])
        i_gl = int(np.searchsorted(x, x_gl_j))
        if i_gl >= len(x):
            continue
        m_row = np.where(np.isnan(m_true[j, i_gl:]), 0.0, m_true[j, i_gl:])
        integrated = cumulative_trapezoid(m_row, x[i_gl:], initial=0.0)
        H_true[j, i_gl:] = H_GL + integrated / v
    H_true = np.where(floating, H_true, np.nan)
    h_true = H_true * (rhow - rhoi) / rhow

    times = pd.date_range("2019-01-01", periods=n_epochs, freq="6MS")
    return x, y, vx_2d, vy_2d, grounded, floating, m_true, h_true, times


def build_noisy_stack(
    h_true, x_coords, y_coords, times, floating, *,
    strip_offset_sigma: float = 0.0,
    pixel_noise_sigma: float = 0.0,
    rng=None,
):
    rng = rng if rng is not None else np.random.default_rng(0)
    layers = []
    for _ in times:
        layer = h_true.copy()
        if strip_offset_sigma > 0:
            offset = float(rng.normal(0.0, strip_offset_sigma))
            layer = layer + offset
        if pixel_noise_sigma > 0:
            noise = rng.normal(0.0, pixel_noise_sigma, size=layer.shape)
            layer = layer + noise
        layer = np.where(floating, layer, np.nan)
        layers.append(
            xr.DataArray(layer, dims=("y", "x"), coords={"y": y_coords, "x": x_coords})
        )
    return xr.concat(layers, dim=pd.Index(times, name="time"))


def perturb_velocity(
    vx, vy, *, mult_sigma: float = 0.0, gap_idx=None, rng=None
):
    rng = rng if rng is not None else np.random.default_rng(0)
    vx_out = vx.copy()
    vy_out = vy.copy()
    if mult_sigma > 0:
        # multiplicative perturbation, same factor on vx and vy per pixel
        m = rng.normal(1.0, mult_sigma, size=vx.shape)
        vx_out = vx_out * m
        vy_out = vy_out * m
    if gap_idx is not None:
        j0, j1, i0, i1 = gap_idx
        vx_out[j0:j1, i0:i1] = 0.0
        vy_out[j0:j1, i0:i1] = 0.0
    return vx_out, vy_out


def run_recovery(
    h_stack,
    vx,
    vy,
    grounded,
    floating,
    m_true,
    x_coords,
    y_coords,
    *,
    n_epochs: int = 30,
):
    vx_da = xr.DataArray(vx, dims=("y", "x"), coords={"y": y_coords, "x": x_coords})
    vy_da = xr.DataArray(vy, dims=("y", "x"), coords={"y": y_coords, "x": x_coords})

    trajectories = backward_advect_pixels(
        x_coords=x_coords,
        y_coords=y_coords,
        vx=vx,
        vy=vy,
        grounded_mask=grounded,
        floating_mask=floating,
        dt_yr=0.05,
        max_tau_yr=300.0,
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
        h_stack=h_stack,
        trajectories=trajectories,
        sigma_per_epoch=sigma_per_epoch,
        a_dot=a_dot,
        vdiv=vdiv,
        d_fac=d_fac,
    )

    result = fit_streamline_pinn(
        dataset,
        hidden_dim=192,
        n_hidden_layers=5,
        n_fourier_features=16,
        n_epochs=n_epochs,
        batch_size=4096,
        lr=2e-3,
        test_frac=0.1,
        device="cpu",
        verbose=False,
    )

    pred = predict_melt_grid(
        model=result.model,
        trajectories=trajectories,
        a_dot=a_dot,
        vdiv=vdiv,
        d_fac=d_fac,
        device="cpu",
    )

    interior = (slice(5, -5), slice(5, -5))
    m_hat_i = pred.melt_rate.values[interior]
    m_true_i = m_true[interior]
    valid = np.isfinite(m_hat_i) & np.isfinite(m_true_i)
    err = m_hat_i[valid] - m_true_i[valid]
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(m_hat_i[valid], m_true_i[valid])[0, 1])
    return {
        "corr": corr,
        "rmse": rmse,
        "bias": bias,
        "n_success_pct": 100 * n_success / max(n_floating, 1),
        "final_train_loss": float(result.train_losses[-1]),
        "delta_strip": (
            result.model.strip_offset.detach().cpu().numpy()
            if result.model.strip_offset is not None
            else None
        ),
        "pred": pred,
    }


def main():
    print("Building clean synthetic shelf...")
    x_coords, y_coords, vx_2d, vy_2d, grounded, floating, m_true, h_true, times = (
        build_clean_shelf(n_epochs=10)
    )
    ny, nx = floating.shape
    print(f"  grid {ny}x{nx}  floating frac {float(floating.mean()):.2f}  "
          f"m_true ∈ [{np.nanmin(m_true):+.2f}, {np.nanmax(m_true):+.2f}] m/yr")

    # Velocity gap: zero stripe at y near 0 in the fast-flow zone (x > 0).
    # 250 m grid, 160x160: rows 75-90 = y ∈ [-2.5 km, +1 km],
    # cols 80-140 = x ∈ [0, +15 km]. ~3.75 km × 15 km gap, fast-flow side.
    gap_idx = (75, 90, 80, 140)

    configs = [
        dict(name="0 clean",            strip=0.0, pix=0.0, v_mult=0.0, v_gap=None),
        dict(name="1 +strip 5m",        strip=5.0, pix=0.0, v_mult=0.0, v_gap=None),
        dict(name="2 +pix 0.3m",        strip=5.0, pix=0.3, v_mult=0.0, v_gap=None),
        dict(name="3 +v 10% mult",      strip=5.0, pix=0.3, v_mult=0.10, v_gap=None),
        dict(name="4 +v gap",           strip=5.0, pix=0.3, v_mult=0.10, v_gap=gap_idx),
    ]

    rng = np.random.default_rng(42)
    results = []
    for cfg in configs:
        print(f"\n=== {cfg['name']} ===")
        h_stack = build_noisy_stack(
            h_true=h_true, x_coords=x_coords, y_coords=y_coords, times=times,
            floating=floating, strip_offset_sigma=cfg["strip"],
            pixel_noise_sigma=cfg["pix"], rng=rng,
        )
        vx_n, vy_n = perturb_velocity(
            vx_2d, vy_2d, mult_sigma=cfg["v_mult"], gap_idx=cfg["v_gap"], rng=rng,
        )
        r = run_recovery(
            h_stack=h_stack, vx=vx_n, vy=vy_n,
            grounded=grounded, floating=floating, m_true=m_true,
            x_coords=x_coords, y_coords=y_coords, n_epochs=30,
        )
        print(f"  coverage:      {r['n_success_pct']:.1f}% of floating")
        print(f"  train_loss:    {r['final_train_loss']:.3e}")
        if r["delta_strip"] is not None:
            d = r["delta_strip"]
            print(f"  δ_strip:       mean={d.mean():+.2f}  std={d.std():.2f}  "
                  f"range=[{d.min():+.2f}, {d.max():+.2f}] m")
        print(f"  corr={r['corr']:.3f}  RMSE={r['rmse']:.3f} m/yr  bias={r['bias']:+.3f}")
        results.append({"name": cfg["name"], **r})

    # Summary table
    print("\n" + "=" * 68)
    print(f"{'config':<22} {'cov%':>6} {'corr':>7} {'RMSE':>8} {'bias':>8}  {'δ_str std':>9}")
    print("=" * 68)
    for r in results:
        ds = r["delta_strip"]
        ds_str = f"{ds.std():.2f}" if ds is not None else "—"
        print(
            f"{r['name']:<22} {r['n_success_pct']:>5.1f}%  {r['corr']:>6.3f} "
            f"{r['rmse']:>7.3f}  {r['bias']:>+7.3f}  {ds_str:>9}"
        )
    print("=" * 68)

    # 5-panel plot of recovered m̂
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=True)
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    for ax, r in zip(axes, results):
        im = ax.imshow(
            r["pred"].melt_rate.values, cmap="RdBu_r", vmin=-3.5, vmax=1.0,
            extent=extent, origin="upper", aspect="equal",
        )
        ax.set_title(f"{r['name']}\ncorr={r['corr']:.2f}  RMSE={r['rmse']:.2f}",
                     fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    # Also plot m_true for reference at the leftmost — actually replace tier-0 visualization
    # No — keep tier-0 as recovered (the test). Put m_true in suptitle context.
    fig.suptitle(
        f"Streamline PINN recovery vs progressive noise (m_true ∈ {np.nanmin(m_true):.1f} to "
        f"{np.nanmax(m_true):.1f} m/yr)",
        fontsize=12,
    )
    out_path = Path(__file__).parent / "sanity_streamline_pinn_noisy.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path.name}")


if __name__ == "__main__":
    main()
