# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Synthetic correctness gate for velocity_fusion.

Builds a known low-rank space-time velocity field (a mean flow plus a linear
acceleration pattern), punches random gaps + adds noise, and asserts that
:func:`fuse_velocity_field`:

1. recovers the truth at observed epochs (denoise + infill skill), and
2. interpolates a fully held-out epoch from its temporal neighbours.

Run::

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        tests/sanity_velocity_fusion.py
"""

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.dynamics.velocity_fusion import fuse_velocity_field, harmonic_infill_2d


def _build_truth(ny=30, nx=40, n_t=16):
    xx = (np.arange(nx)[None, :] / nx) * np.ones((ny, 1))
    yy = (np.arange(ny)[:, None] / ny) * np.ones((1, nx))
    mean_x = 100.0 + 60.0 * xx + 10.0 * yy            # mean flow pattern
    mean_y = -25.0 + 8.0 * yy
    accel_x = 8.0 * np.sin(np.pi * xx)                # acceleration EOF pattern
    accel_y = 2.0 * np.cos(np.pi * yy)
    times = pd.date_range("2015-02-15", periods=n_t, freq="QS")
    tyr = (times.year - 2015) + (times.dayofyear - 1) / 365.25
    tyr = tyr.to_numpy()
    vx = mean_x[None] + accel_x[None] * tyr[:, None, None]
    vy = mean_y[None] + accel_y[None] * tyr[:, None, None]
    return times, vx, vy


def main() -> int:
    rng = np.random.default_rng(0)
    times, vx_true, vy_true = _build_truth()
    n_t, ny, nx = vx_true.shape

    # Observed = truth + small noise + ~15% random gaps.
    vx_obs = vx_true + rng.normal(0, 0.4, vx_true.shape)
    vy_obs = vy_true + rng.normal(0, 0.4, vy_true.shape)
    holes = rng.random(vx_true.shape) < 0.15
    vx_obs[holes] = np.nan
    vy_obs[holes] = np.nan

    coords = {"time": times, "y": np.arange(ny), "x": np.arange(nx)}
    vx_da = xr.DataArray(vx_obs, dims=("time", "y", "x"), coords=coords)
    vy_da = xr.DataArray(vy_obs, dims=("time", "y", "x"), coords=coords)

    # --- harmonic infill leaves no NaN inside the (here full) domain ---
    domain = np.ones((ny, nx), dtype=bool)
    filled = harmonic_infill_2d(vx_obs[0], domain)
    assert np.isfinite(filled).all(), "harmonic infill left NaNs"
    print(f"[infill] epoch0 max|fill-truth| over gaps = "
          f"{np.nanmax(np.abs(filled - vx_true[0])):.3f} m/yr")

    # --- (1) reconstruct at observed epochs ---
    vx_f, vy_f, diag = fuse_velocity_field(
        vx_da, vy_da, times, n_modes=5, coverage_frac=0.5, seasonal=False, verbose=True
    )
    rmse_x = float(np.sqrt(np.nanmean((vx_f.values - vx_true) ** 2)))
    rmse_y = float(np.sqrt(np.nanmean((vy_f.values - vy_true) ** 2)))
    speed = np.hypot(vx_true, vy_true)
    rel = float(np.nanmedian(np.hypot(vx_f.values - vx_true, vy_f.values - vy_true) / speed))
    print(f"[recon] RMSE vx={rmse_x:.3f} vy={rmse_y:.3f} m/yr | median rel err={rel:.4%}")
    print(f"[eof]   {diag['k_retained']} modes, cum var {diag['cum_var']:.5f}, "
          f"per-mode {np.round(diag['var_explained'][:diag['k_retained']], 4)}")

    # --- (2) hold out a mid epoch; interpolate it from neighbours ---
    hold = n_t // 2
    keep = [i for i in range(n_t) if i != hold]
    vx_lo = vx_da.isel(time=keep)
    vy_lo = vy_da.isel(time=keep)
    vx_h, vy_h, _ = fuse_velocity_field(
        vx_lo, vy_lo, times, n_modes=5, coverage_frac=0.5, seasonal=False, verbose=False
    )
    held_rmse = float(np.sqrt(np.nanmean(
        (vx_h.values[hold] - vx_true[hold]) ** 2
        + (vy_h.values[hold] - vy_true[hold]) ** 2
    )))
    held_rel = float(np.nanmedian(
        np.hypot(vx_h.values[hold] - vx_true[hold], vy_h.values[hold] - vy_true[hold])
        / speed[hold]
    ))
    print(f"[LOO]   held-out epoch {hold}: RMSE={held_rmse:.3f} m/yr  rel={held_rel:.4%}")

    ok = (rel < 0.02) and (rmse_x < 1.5) and (held_rel < 0.05)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
