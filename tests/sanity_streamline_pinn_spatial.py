r"""Spatial-recovery sanity test for the streamline-frame PINN.

The companion `sanity_streamline_pinn.py` exercises a CONSTANT melt
rate — any h field with the right global slope passes, so it doesn't
actually test whether the network can recover spatial structure. This
test ramps up:

- Curved (parabolic) grounding line — `x_gl(y) = -15 km + 5 km · (y/20 km)²`
- Constant velocity 200 m/yr in +x, zero in y, zero SMB, zero FAC
- True basal melt field with spatial structure:

      m_true(x, y) = m_max · exp(-((x - x_c)² + (y - y_c)²) / (2 σ²))

  Shean convention: negative = melt. Peak m_max = -3 m/yr.

Steady-state thickness is the streamline integral of m from the GL:

      H(x, y) = H_GL + (1/v) ∫[x_gl(y) to x] m(x', y) dx'

The PINN sees the resulting h_obs stack, recovers m̂ via the autodiff
residual `∂H/∂τ + H·∇·u - ḃ_s` (with ∇·u = ḃ_s = 0), and we score the
recovery with 2D RMSE and spatial correlation against m_true.

Pass: spatial correlation > 0.7 and RMSE < 1.0 m/yr (peak is 3 m/yr).
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


def _build_synthetic_shelf():
    # 40 km × 40 km, 250 m grid
    x = np.arange(-20000.0, 20000.0, 250.0)
    y = np.arange(20000.0, -20000.0, -250.0)  # descending (north-up)
    Y, X = np.meshgrid(y, x, indexing="ij")

    # Constant +x velocity field
    vx_2d = np.full_like(X, 200.0)
    vy_2d = np.zeros_like(X)
    v = 200.0  # m/yr

    # Curved (parabolic) GL: bulges into the shelf at large |y|
    x_gl_y = -15000.0 + 5000.0 * (Y / 20000.0) ** 2
    grounded = X < x_gl_y
    floating = ~grounded

    # True m: Gaussian bump (negative = melt; Shean convention)
    m_max = -3.0
    x_c, y_c = -5000.0, 0.0
    sigma_m = 5000.0
    m_true = m_max * np.exp(-((X - x_c) ** 2 + (Y - y_c) ** 2) / (2 * sigma_m**2))
    m_true = np.where(floating, m_true, np.nan)

    # Forward-solve steady-state H by streamline integration along each row.
    # At fixed y: dH/dτ = m,  τ = (x - x_gl) / v  →  dH = m · dx / v
    H_GL = 500.0  # m at the GL
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
    h_true = H_true * (rhow - rhoi) / rhow  # hydrostatic, d=0

    return x, y, X, Y, vx_2d, vy_2d, grounded, floating, m_true, H_true, h_true


def main():
    print("Building synthetic shelf (curved GL, Gaussian m_true)...")
    x_coords, y_coords, X, Y, vx_2d, vy_2d, grounded, floating, m_true, H_true, h_true = (
        _build_synthetic_shelf()
    )
    print(f"  domain: {X.shape[0]}x{X.shape[1]}  floating frac: {float(floating.mean()):.3f}")
    print(f"  m_true:  [{np.nanmin(m_true):+.2f}, {np.nanmax(m_true):+.2f}] m/yr")
    print(f"  H_true:  [{np.nanmin(H_true):.1f}, {np.nanmax(H_true):.1f}] m")
    print(f"  h_true:  [{np.nanmin(h_true):.1f}, {np.nanmax(h_true):.1f}] m")

    vx = xr.DataArray(vx_2d, dims=("y", "x"), coords={"y": y_coords, "x": x_coords})
    vy = xr.DataArray(vy_2d, dims=("y", "x"), coords={"y": y_coords, "x": x_coords})

    # 10-epoch steady-state stack (identical layers — the dataset's redundancy
    # at fixed (x_gl, y_gl, τ) tests that the PINN converges to the mean cleanly)
    times = pd.date_range("2019-01-01", periods=10, freq="6MS")
    layers = [
        xr.DataArray(h_true.copy(), dims=("y", "x"), coords={"y": y_coords, "x": x_coords})
        for _ in times
    ]
    h_stack = xr.concat(layers, dim=pd.Index(times, name="time"))

    print("Backward-advecting every floating pixel to its GL crossing...")
    trajectories = backward_advect_pixels(
        x_coords=x_coords,
        y_coords=y_coords,
        vx=vx.values,
        vy=vy.values,
        grounded_mask=grounded,
        floating_mask=floating,
        dt_yr=0.05,
        max_tau_yr=300.0,
    )
    reason = trajectories["reason"]
    n_total = int(reason.size)
    n_success = int((reason == 0).sum())
    n_floating = int(floating.sum())
    print(
        f"  trajectories: {n_success}/{n_floating} floating pixels success "
        f"({100*n_success/max(n_floating, 1):.1f}%)"
    )

    a_dot = xr.DataArray(
        np.zeros_like(vx_2d), dims=("y", "x"), coords={"y": y_coords, "x": x_coords}
    )
    d_fac = xr.DataArray(
        np.zeros_like(vx_2d), dims=("y", "x"), coords={"y": y_coords, "x": x_coords}
    )
    vdiv = divergence(vx, vy)  # zero everywhere for constant velocity

    sigma_per_epoch = np.full(h_stack.sizes["time"], 0.1, dtype=np.float64)

    print("Building streamline dataset...")
    dataset = build_streamline_dataset(
        h_stack=h_stack,
        trajectories=trajectories,
        sigma_per_epoch=sigma_per_epoch,
        a_dot=a_dot,
        vdiv=vdiv,
        d_fac=d_fac,
    )
    print(f"  rows: {len(dataset)}")

    print("Training streamline PINN (60 epochs, hidden=192, layers=5)...")
    result = fit_streamline_pinn(
        dataset,
        hidden_dim=192,
        n_hidden_layers=5,
        n_fourier_features=16,
        n_epochs=60,
        batch_size=4096,
        lr=2e-3,
        test_frac=0.1,
        device="cpu",
        verbose=False,
    )
    print(
        f"  final train_loss={result.train_losses[-1]:.4e}  "
        f"test_loss={result.test_losses[-1]:.4e}"
    )

    print("Predicting m on Eulerian grid...")
    pred = predict_melt_grid(
        model=result.model,
        trajectories=trajectories,
        a_dot=a_dot,
        vdiv=vdiv,
        d_fac=d_fac,
        device="cpu",
    )
    m_hat = pred.melt_rate.values

    # Score recovery on the interior (skip 5 pixels at each edge to avoid
    # boundary artifacts in the autodiff residual)
    interior = (slice(5, -5), slice(5, -5))
    m_hat_i = m_hat[interior]
    m_true_i = m_true[interior]
    valid_both = np.isfinite(m_hat_i) & np.isfinite(m_true_i)
    err = m_hat_i[valid_both] - m_true_i[valid_both]
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(m_hat_i[valid_both], m_true_i[valid_both])[0, 1])
    print("\n  Recovery metrics (interior):")
    print(f"    RMSE = {rmse:.3f} m/yr")
    print(f"    bias = {bias:+.3f} m/yr")
    print(f"    spatial correlation = {corr:.3f}")

    # 3-panel comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    vlim = (-3.5, 1.0)
    im0 = axes[0].imshow(
        m_true, cmap="RdBu_r", vmin=vlim[0], vmax=vlim[1], extent=extent, origin="upper", aspect="equal"
    )
    axes[0].set_title("m_true (analytic)")
    fig.colorbar(im0, ax=axes[0], fraction=0.045)
    im1 = axes[1].imshow(
        m_hat, cmap="RdBu_r", vmin=vlim[0], vmax=vlim[1], extent=extent, origin="upper", aspect="equal"
    )
    axes[1].set_title("m_hat (PINN)")
    fig.colorbar(im1, ax=axes[1], fraction=0.045)
    im2 = axes[2].imshow(
        m_hat - m_true,
        cmap="PuOr_r",
        vmin=-1.5,
        vmax=1.5,
        extent=extent,
        origin="upper",
        aspect="equal",
    )
    axes[2].set_title("residual (m_hat - m_true)")
    fig.colorbar(im2, ax=axes[2], fraction=0.045)
    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"Streamline PINN spatial recovery — RMSE={rmse:.2f} m/yr, corr={corr:.2f}, bias={bias:+.2f}"
    )
    out_path = Path(__file__).parent / "sanity_streamline_pinn_spatial.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")

    # Pass criteria — peak m is 3 m/yr in magnitude, so RMSE 1 m/yr is generous
    # but not trivial; corr > 0.7 means the network clearly captures the bump.
    if corr < 0.7 or rmse > 1.0:
        print(
            f"\n  ✗ Spatial recovery FAILED (corr={corr:.3f} target >0.7, "
            f"RMSE={rmse:.3f} target <1.0)"
        )
        raise SystemExit(1)
    print(f"\n  ✓ Spatial recovery PASSED (corr={corr:.3f}, RMSE={rmse:.3f})")


if __name__ == "__main__":
    main()
