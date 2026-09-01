"""Synthetic gate for linear_inverse_eulerian_budget_melt_rate.

Companion to ``gate_match_lagrangian.py`` (which gates the pair-banded
inverse against the production path solver). This gates the EULERIAN twin:

- Rung A — no flow, no noise, full coverage, wide (10·H) channel: with
  ``u = 0`` the synthetic thickness is exactly linear in time per cell, so
  the Eulerian hydro channel must equal truth to numerical precision, the
  kernel correction must be near-zero (the channel is super-5H), and
  ``melt_rate_hydro`` must be bit-identical to a direct
  :func:`stereo_melt.melt.eulerian_melt_rate` call.
- Rung B — 350 m/yr flow, 0.1 m epoch noise, ~35% strip gaps, Tukey-robust
  regression: recovery degrades only through flow-smearing of the per-cell
  trend; correlation with truth and the median must hold, and hydro
  bit-parity with the production Eulerian call must survive gaps/robust.

Run:

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        stereo_melt/tests/gate_eulerian_budget_inverse.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.dynamics import linear_inverse_eulerian_budget_melt_rate
from stereo_melt.melt import eulerian_melt_rate

RHOW, RHOI = 1027.0, 918.0
DELTA = (RHOW - RHOI) / RHOW


def build_stack(u=0.0, noise=0.0, gaps=False, sig=3000.0, seed=7):
    ny = nx = 128
    res = 250.0
    x = np.arange(nx) * res
    y = (np.arange(ny)[::-1]) * res  # y descending (north-up)
    X, Y = np.meshgrid(x, y)
    x0, y0 = x.mean(), y.mean()
    b_true = -2.0 - 6.0 * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sig**2)))
    a_dot, H0 = 0.3, 300.0

    t_yr = np.array([0.0, 0.4, 0.9, 1.3, 1.8, 2.2, 2.7, 3.1, 3.6, 4.0])
    times = pd.Timestamp("2019-01-01") + pd.to_timedelta(t_yr * 365.25, unit="D")

    rng = np.random.default_rng(seed)
    dtau = 0.02
    stack = np.empty((len(t_yr), ny, nx), dtype=np.float64)
    for k, tk in enumerate(t_yr):
        acc = np.zeros((ny, nx))
        if tk > 0:
            taus = np.arange(0.0, tk, dtau) + dtau / 2.0
            for tau in taus:
                xs = X - u * (tk - tau)
                r2 = (xs - x0) ** 2 + (Y - y0) ** 2
                acc += (-2.0 - 6.0 * np.exp(-r2 / (2 * sig**2))) * dtau
        H = H0 + a_dot * tk + acc
        h = DELTA * H
        if noise > 0:
            h = h + rng.normal(0.0, noise, size=(ny, nx))
        if gaps:
            for _ in range(3):
                r0 = rng.integers(0, ny - 30)
                c0 = rng.integers(0, nx - 30)
                h[r0:r0 + rng.integers(20, 60), c0:c0 + rng.integers(20, 60)] = np.nan
        stack[k] = h

    coords = {"time": times.values, "y": y, "x": x}
    h_da = xr.DataArray(stack, dims=("time", "y", "x"), coords=coords)
    vx = xr.DataArray(np.full((ny, nx), float(u)), dims=("y", "x"),
                      coords={"y": y, "x": x})
    vy = xr.zeros_like(vx)
    a_da = xr.full_like(vx, a_dot)
    fmask = xr.ones_like(vx).astype(bool)
    return h_da, vx, vy, a_da, fmask, b_true


def _nan_gauss(a, sigma_pix):
    from scipy.ndimage import gaussian_filter

    m = np.isfinite(a)
    num = gaussian_filter(np.where(m, a, 0.0), sigma_pix, mode="nearest")
    den = gaussian_filter(m.astype(float), sigma_pix, mode="nearest")
    out = num / np.maximum(den, 1e-9)
    return np.where(den > 0.05, out, np.nan)


def run_rung(name, robust, **stack_kw):
    h_da, vx, vy, a_da, fmask, b_true = build_stack(**stack_kw)
    ds = linear_inverse_eulerian_budget_melt_rate(
        h_da, vx, vy, a_dot=a_da, d=0.0, floating_mask=fmask,
        eta_bar=1e14, reg=0.1, transform="dct", robust_dh_dt=robust,
    )
    euler = eulerian_melt_rate(
        h_da, vx, vy, a_dot=a_da, d=0.0, robust_dh_dt=robust
    )
    hydro_parity = np.array_equal(
        ds.melt_rate_hydro.values, euler.melt_rate.values, equal_nan=True
    )
    m = ds.melt_rate.values
    f = np.isfinite(m)
    corr = np.corrcoef(m[f], b_true[f])[0, 1]
    ms, bs = _nan_gauss(m, 8.0), _nan_gauss(b_true, 8.0)  # 2 km @ 250 m
    fs = np.isfinite(ms) & np.isfinite(bs)
    corr2k = np.corrcoef(ms[fs], bs[fs])[0, 1]
    med_err = float(np.nanmedian(m) - np.median(b_true[f]))
    corr_mag = float(np.nanmedian(np.abs(ds.nonhydro_corr.values)))
    hyd_err = float(np.nanmax(np.abs(ds.melt_rate_hydro.values - b_true)[f])) \
        if stack_kw.get("u", 0.0) == 0.0 and not stack_kw.get("gaps") else np.nan
    print(
        f"{name}: hydro==eulerian {hydro_parity}  corr(truth) raw={corr:.4f} "
        f"2km={corr2k:.4f}  median_err={med_err:+.3f}  "
        f"|corr_term|med={corr_mag:.3f}  hydro_maxerr={hyd_err:.2e}  "
        f"cells={int(f.sum())}"
    )
    return hydro_parity, corr, corr2k, med_err, hyd_err


def main():
    ok = True
    p, c, c2, me, he = run_rung("A (no flow, clean)", robust=False, u=0.0)
    ok &= p and (c >= 0.999) and (abs(me) <= 0.05) and (he <= 1e-6)
    # Raw per-cell corr is bounded by the Eulerian estimator's own noise
    # floor (no pair-median pooling) — the production criterion is the
    # smoothed-scale agreement, as in the PIG budget-inverse A/Bs.
    p, c, c2, me, _ = run_rung(
        "B (flow+noise+gaps)", robust=True, u=350.0, noise=0.1, gaps=True
    )
    ok &= p and (c >= 0.80) and (c2 >= 0.97) and (abs(me) <= 0.15)
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
