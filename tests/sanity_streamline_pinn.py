r"""End-to-end synthetic sanity test for the streamline-frame PINN.

Steady-state shelf identical to ``sanity_lagrangian_melt.py``: linear
thickness gradient :math:`H = 300 - 0.005\,x`, uniform velocity
:math:`v_x = 200` m/yr in +x, zero SMB, zero FAC. A grounded "wall" at
the upstream (-x) edge serves as the synthetic grounding line; the
remainder of the domain is floating. Under these conditions Shean
Eq. 10 gives ``m = -1 m/yr`` everywhere (Shean convention:
negative = melt) and the analytic Lagrangian age is
:math:`\tau(x) = (x - x_{gl}) / v_x`.

The test gates the PINN against the Lagrangian solver before any real
data is touched. If recovered ``m`` differs from -1 m/yr by more than a
tolerance on the interior, the architecture or training loop is wrong.

Run:

    python -m tests.sanity_streamline_pinn
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

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
    # 20 km x 20 km domain, 250 m grid (same as sanity_lagrangian_melt)
    x = np.arange(-10000.0, 10000.0, 250.0)
    y = np.arange(10000.0, -10000.0, -250.0)
    Y, X = np.meshgrid(y, x, indexing="ij")

    # Linear thickness, hydrostatic freeboard
    H0 = 300.0 - 0.005 * X
    h0 = H0 * (rhow - rhoi) / rhow

    # Constant velocity field
    vx = xr.DataArray(np.full_like(H0, 200.0), dims=("y", "x"), coords={"y": y, "x": x})
    vy = xr.DataArray(np.zeros_like(H0), dims=("y", "x"), coords={"y": y, "x": x})

    # Grounded "wall" at upstream 5 columns (x < -8500 m)
    grounded = X < -8750.0
    floating = ~grounded

    # Apply NaN over grounded ice (matches REMA convention: shelf-only obs)
    h0_obs = np.where(floating, h0, np.nan)

    # 5-epoch steady stack
    times = pd.to_datetime(
        ["2019-01-01", "2019-07-01", "2020-01-01", "2020-07-01", "2021-01-01"]
    )
    layers = [
        xr.DataArray(h0_obs.copy(), dims=("y", "x"), coords={"y": y, "x": x}) for _ in times
    ]
    h_stack = xr.concat(layers, dim=pd.Index(times, name="time"))

    grounded_da = xr.DataArray(grounded, dims=("y", "x"), coords={"y": y, "x": x})
    floating_da = xr.DataArray(floating, dims=("y", "x"), coords={"y": y, "x": x})

    return h_stack, vx, vy, grounded_da, floating_da, x, y


def main():
    print("Building synthetic shelf...")
    h_stack, vx, vy, grounded, floating, x_coords, y_coords = _build_synthetic_shelf()
    print(f"  domain: {h_stack.sizes['y']} x {h_stack.sizes['x']} pixels @ 250 m")
    print(f"  floating fraction: {float(floating.mean()):.3f}")

    print("Backward-advecting every floating pixel to its GL crossing...")
    trajectories = backward_advect_pixels(
        x_coords=x_coords,
        y_coords=y_coords,
        vx=vx.values,
        vy=vy.values,
        grounded_mask=grounded.values,
        floating_mask=floating.values,
        dt_yr=0.05,
        max_tau_yr=120.0,
    )
    reason = trajectories["reason"]
    n_success = int((reason == 0).sum())
    n_total = int(reason.size)
    print(
        f"  trajectories: {n_success}/{n_total} success ({100*n_success/n_total:.1f}%)  "
        f"oob={int((reason == 1).sum())}  maxtau={int((reason == 2).sum())}  "
        f"non-floating={int((reason == 3).sum())}"
    )

    # Spot-check: analytic τ at x = +10 km with GL at x ≈ -8.75 km and v = 200 m/yr
    # should be (10000 - (-8750)) / 200 ≈ 93.75 yr. The integrator stops the first
    # substep crossing into grounded ice, so the recovered GL is within one cell.
    finite_tau = trajectories["tau"][np.isfinite(trajectories["tau"])]
    print(
        f"  tau: min={finite_tau.min():.2f}  median={np.median(finite_tau):.2f}  "
        f"max={finite_tau.max():.2f} yr"
    )

    # Build forcings on the same grid
    a_dot = xr.DataArray(
        np.zeros_like(vx.values),
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        name="a_dot",
    )
    d_fac = xr.DataArray(
        np.zeros_like(vx.values),
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        name="d_fac",
    )
    vdiv = divergence(vx, vy)  # zero for constant velocity → just a smoothness check

    sigma_per_epoch = np.full(h_stack.sizes["time"], 0.1, dtype=np.float64)  # 10 cm

    print("Assembling streamline training dataset...")
    dataset = build_streamline_dataset(
        h_stack=h_stack,
        trajectories=trajectories,
        sigma_per_epoch=sigma_per_epoch,
        a_dot=a_dot,
        vdiv=vdiv,
        d_fac=d_fac,
    )
    print(f"  rows: {len(dataset)} observations")

    print("Training streamline PINN (CPU, 30 epochs)...")
    result = fit_streamline_pinn(
        dataset,
        hidden_dim=128,
        n_hidden_layers=4,
        n_fourier_features=8,
        n_epochs=30,
        batch_size=2048,
        lr=2e-3,
        test_frac=0.1,
        device="cpu",
        verbose=True,
    )
    print(f"  final train_loss={result.train_losses[-1]:.4e}  test_loss={result.test_losses[-1]:.4e}")

    print("Predicting melt rate on Eulerian grid...")
    pred = predict_melt_grid(
        model=result.model,
        trajectories=trajectories,
        a_dot=a_dot,
        vdiv=vdiv,
        d_fac=d_fac,
        device="cpu",
    )

    m = pred.melt_rate.values
    valid = np.isfinite(m)
    print(
        f"  recovered m: median={float(np.nanmedian(m)):+.3f}  "
        f"IQR=[{float(np.nanpercentile(m, 25)):+.3f}, "
        f"{float(np.nanpercentile(m, 75)):+.3f}] m ice/yr  "
        f"({100*valid.mean():.1f}% finite)"
    )

    # Interior box — skip 10 pixels at each boundary (same as Lagrangian sanity)
    # and skip the upstream GL strip where τ ≈ 0 (autodiff at boundary).
    interior = m[20:-10, 20:-10]
    m_core = float(np.nanmedian(interior))
    print(f"  interior-box median m: {m_core:+.3f} m ice/yr  (analytic = -1.000)")

    tol = 0.10  # 10 cm/yr tolerance
    assert abs(m_core - (-1.0)) < tol, (
        f"Recovered m {m_core:+.3f} m/yr is more than {tol} m/yr off from analytic -1.0. "
        "Architecture, training, or autodiff residual is wrong."
    )
    print(f"  ✓ Interior melt rate matches analytic -1 m/yr within {tol} m/yr.")


if __name__ == "__main__":
    main()
