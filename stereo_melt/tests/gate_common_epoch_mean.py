"""Gate: the common-epoch stack mean (Shean-style temporal consistency).

A repeat-DEM stack samples each pixel at whatever epochs its strips supply,
so the plain time-mean is the field at a per-pixel mean epoch. On a changing
surface that injects a sampling artifact shaped like the strip footprints —
which the flux divergence then differentiates.

C1  UNBIASED UNDER RAGGED SAMPLING. A field with a known uniform trend,
    observed with a strip-shaped ragged sampling pattern: the plain mean is
    biased by rate * (t_bar - t0) with the artifact's spatial pattern, while
    the common-epoch mean recovers H(t0) to < 5 % of that bias.
C2  EXACT WHEN SAMPLING IS UNIFORM. With every pixel observed at every
    epoch the correction is identically zero (both means agree to 1e-9).
C3  NO VARIANCE PENALTY. On a noisy stack the corrected field is no rougher
    than the plain mean (the rate is smoothed, so the correction adds
    negligible short-scale variance, unlike a per-pixel refit).

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY stereo_melt/tests/gate_common_epoch_mean.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from stereo_melt.kinematics import common_epoch_mean, dh_dt  # noqa: E402

NY, NX, NT, RES = 60, 90, 24, 250.0
RATE = -3.0          # m/yr, uniform thinning
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def build(sampling, noise=0.0, seed=0):
    """Stack of a trending field, observed through `sampling` (nt, ny, nx)."""
    rng = np.random.default_rng(seed)
    y = np.arange(NY)[::-1] * RES
    x = np.arange(NX) * RES
    times = pd.to_datetime("2012-01-01") + pd.to_timedelta(
        np.arange(NT) * 365.25 / 2.0, unit="D")
    t_yr = (times - times[0]).total_seconds().values / (86400.0 * 365.25)
    base = 500.0 + 20.0 * np.cos(2 * np.pi * np.arange(NX) / 30.0)[None, :]
    cube = base[None] + RATE * t_yr[:, None, None]
    if noise:
        cube = cube + rng.normal(0.0, noise, cube.shape)
    cube = np.where(sampling, cube, np.nan)
    return xr.DataArray(cube, dims=("time", "y", "x"),
                        coords={"time": times, "y": y, "x": x}), t_yr


def ragged_sampling(seed=1):
    """Strip-like coverage, including a poorly-observed corner.

    Each epoch sees a contiguous band of columns, a few epochs are full
    scenes, and one band is observed ONLY in three early epochs — the real
    pattern that makes a per-pixel refit unstable, because its slope is
    estimated from a short baseline and then extrapolated to mid-window.
    """
    rng = np.random.default_rng(seed)
    s = np.zeros((NT, NY, NX), bool)
    for i in range(NT):
        c0 = rng.integers(0, NX // 2)
        s[i, :, c0:c0 + NX // 2] = True
    s |= (np.arange(NT)[:, None, None] % 6 == 0)          # a few full scenes
    sparse = slice(NX - 12, NX)
    s[:, :, sparse] = False
    s[:3, :, sparse] = True                               # 3 early epochs only
    return s


def main() -> int:
    print("C1  ragged sampling: plain mean is biased, common-epoch mean is not")
    samp = ragged_sampling()
    stack, t_yr = build(samp)
    reg = dh_dt(stack)
    ce = common_epoch_mean(stack, reg["slope"], sigma_px=2.0)
    t_mid = float(np.median((stack.time.values - stack.time.values.min())
                            / np.timedelta64(1, "s")))
    t_mid_yr = t_mid / (86400.0 * 365.25)
    truth = (500.0 + 20.0 * np.cos(2 * np.pi * np.arange(NX) / 30.0)[None, :]
             + RATE * t_mid_yr)
    plain = stack.mean("time", skipna=True).values
    ok = np.isfinite(plain) & np.isfinite(ce.values)
    e_plain = float(np.sqrt(np.mean((plain - truth)[ok] ** 2)))
    e_ce = float(np.sqrt(np.mean((ce.values - truth)[ok] ** 2)))
    check("common-epoch mean removes the sampling bias",
          e_ce < 0.05 * e_plain and e_plain > 1.0,
          f"rmse vs H(t_mid): plain {e_plain:.2f} m -> corrected {e_ce:.3f} m")

    print("C2  uniform sampling: the correction is identically zero")
    stack_u, _ = build(np.ones((NT, NY, NX), bool))
    reg_u = dh_dt(stack_u)
    ce_u = common_epoch_mean(stack_u, reg_u["slope"], sigma_px=2.0)
    d = float(np.nanmax(np.abs(ce_u.values - stack_u.mean("time").values)))
    check("no-op under uniform sampling", d < 1e-9, f"max|delta| {d:.1e}")

    print("C3  no short-scale variance penalty vs the plain mean")
    stack_n, _ = build(samp, noise=1.5, seed=7)
    reg_n = dh_dt(stack_n)
    ce_n = common_epoch_mean(stack_n, reg_n["slope"], sigma_px=2.0)
    per_pixel = (reg_n["intercept"] + reg_n["slope"] * t_mid).values

    def rough(a):
        b = np.where(np.isfinite(a), a, np.nan)
        lap = (b[:-2, 1:-1] + b[2:, 1:-1] + b[1:-1, :-2] + b[1:-1, 2:]
               - 4 * b[1:-1, 1:-1])
        return float(np.sqrt(np.nanmean(lap ** 2)))

    r_plain, r_ce, r_fit = (rough(stack_n.mean("time", skipna=True).values),
                            rough(ce_n.values), rough(per_pixel))
    check("corrected field is no rougher than the plain mean",
          r_ce < 1.15 * r_plain and r_fit > 1.5 * r_plain,
          f"roughness: plain {r_plain:.2f}  corrected {r_ce:.2f}  "
          f"per-pixel refit {r_fit:.2f} (the naive alternative)")

    print()
    if FAILS:
        print(f"GATE FAILED: {FAILS}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
