"""Smoke test for the GPU tilt-fit path with the searchsorted row-scaling fix.

Constructs a small synthetic DEM stack (~100k–1M pixels) with a known per-epoch
tilt + dh/dt + noise, then runs fit_tilt_stack with STEREO_MELT_BACKEND=cupy.
Verifies the cupy LSMR + IRLS reweighting completes without segfault or
ValueError. Times each scale so we can see if the row-scaling fix scales OK.

Run with:
    CUDA_VISIBLE_DEVICES=0 STEREO_MELT_BACKEND=cupy python scripts/test_gpu_tilt_fit.py
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import xarray as xr
import pandas as pd

from stereo_melt.coregister.tilt import fit_tilt_stack


def synthetic_stack(ny: int, nx: int, n_epochs: int, dx: float = 250.0, seed: int = 0):
    """Return a (time, y, x) xr.DataArray with a known mass-conserving signal."""
    rng = np.random.default_rng(seed)
    x_coord = np.arange(nx) * dx
    y_coord = (np.arange(ny)[::-1]) * dx  # y descending
    times = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(n_epochs) * 30, unit="D")
    # Background topography (a smooth ramp + perlin-ish noise)
    z_bg = 100 + 0.001 * x_coord[None, :] + 0.0005 * y_coord[:, None]
    z_bg = z_bg + 5 * rng.standard_normal((ny, nx))
    # Per-epoch tilt (αx, αy, αz)
    tilt_x = rng.uniform(-1e-4, 1e-4, size=n_epochs)
    tilt_y = rng.uniform(-1e-4, 1e-4, size=n_epochs)
    tilt_z = rng.uniform(-1.0, 1.0, size=n_epochs)
    # Background dh/dt (m/yr) — small drift
    dhdt = 0.01 * rng.standard_normal((ny, nx))
    t_years = (times - times[0]).total_seconds().to_numpy() / (86400 * 365.25)
    z = np.empty((n_epochs, ny, nx), dtype=np.float64)
    for i in range(n_epochs):
        plane = tilt_x[i] * x_coord[None, :] + tilt_y[i] * y_coord[:, None] + tilt_z[i]
        z[i] = z_bg + dhdt * t_years[i] + plane + 0.3 * rng.standard_normal((ny, nx))
    # Drop ~30% of cells to NaN (sparse coverage)
    mask = rng.random((n_epochs, ny, nx)) < 0.7
    z = np.where(mask, z, np.nan)
    da = xr.DataArray(
        z,
        dims=("time", "y", "x"),
        coords={"time": times, "y": y_coord, "x": x_coord},
    )
    return da


def run_test(ny: int, nx: int, n_epochs: int):
    print(f"\n--- ny={ny}, nx={nx}, n_epochs={n_epochs} ({ny*nx*n_epochs:,} cells) ---")
    da = synthetic_stack(ny, nx, n_epochs)
    print(f"  stack shape {da.shape}, finite frac = {np.isfinite(da.values).mean():.2f}")
    t0 = time.time()
    params, stack_corr = fit_tilt_stack(
        da,
        Eint=10.0, Edhdt=10.0, Ex=1e-3, Ey=1e-3, Ez=1.0,
        robust=True, robust_max_iter=3,
        solver="lsmr", lsmr_atol=1e-4, lsmr_maxiter=200,
    )
    dt = time.time() - t0
    print(f"  fit_tilt_stack returned in {dt:.1f}s")
    print(f"  params dims: {dict(params.dims)}")
    print(f"  alpha_z range: [{float(params['alpha_z'].min()):+.3f}, {float(params['alpha_z'].max()):+.3f}] m")
    return dt


if __name__ == "__main__":
    print(f"backend env: STEREO_MELT_BACKEND={os.environ.get('STEREO_MELT_BACKEND', '(unset)')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

    # Three sizes — small, medium, larger (still tiny vs the real problem)
    sizes = [
        (100, 100, 5),     # 50k cells; quick smoke test
        (300, 400, 15),    # 1.8M cells; comparable to a small basin
        (1000, 800, 25),   # 20M cells; comparable to mid-basin
    ]
    timings = []
    for ny, nx, n_epochs in sizes:
        try:
            dt = run_test(ny, nx, n_epochs)
            timings.append((ny, nx, n_epochs, dt, "OK"))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            timings.append((ny, nx, n_epochs, None, f"FAIL: {type(e).__name__}"))

    print("\n=== summary ===")
    for ny, nx, n, dt, status in timings:
        cells = ny * nx * n
        secs = f"{dt:.1f}s" if dt is not None else "—"
        print(f"  {ny:5d}x{nx:5d}x{n:3d} = {cells:>12,} cells  →  {secs:>8s}  [{status}]")
