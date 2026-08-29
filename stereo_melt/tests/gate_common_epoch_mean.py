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
    epoch the correction is identically zero (both means agree to 1e-9),
    for evenly AND unevenly spaced epochs — the reference epoch is the
    mean sample time, which is what the plain mean already refers to.
C3  NO VARIANCE PENALTY. On a noisy stack the corrected field is no rougher
    than the plain mean (the rate is smoothed, so the correction adds
    negligible short-scale variance, unlike a per-pixel refit).
C4  NO COVERAGE LOSS. A block of pixels seen in fewer epochs than the
    regression's min_count (own slope undefined) beside well-sampled
    neighbours stays finite, borrows the neighbourhood's trend, and lands
    on H(t0); coverage is identical to the plain mean's.
C5  SOLVER PLUMBING. ``eulerian_melt_rate`` and ``restored_budget_melt_rate``
    default to the plain mean (attr ``common_epoch=0``); ``common_epoch=True``
    feeds the common-epoch mean to the divergence (attr 1) and changes the
    melt on a ragged stack — the switch is live and OFF by default.

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


def build(sampling, noise=0.0, seed=0, t_yr=None):
    """Stack of a trending field, observed through `sampling` (nt, ny, nx).

    ``t_yr`` gives the epochs in years from the first; default is NT epochs
    evenly spaced by half a year.
    """
    rng = np.random.default_rng(seed)
    y = np.arange(NY)[::-1] * RES
    x = np.arange(NX) * RES
    if t_yr is None:
        t_yr = np.arange(NT) * 0.5
    t_yr = np.asarray(t_yr, dtype=float)
    times = pd.to_datetime("2012-01-01") + pd.to_timedelta(
        np.round(t_yr * 365.25 * 86400.0), unit="s")
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


def truth_at(t_yr_ref):
    """The noise-free field at one epoch (years from the first sample)."""
    return (500.0 + 20.0 * np.cos(2 * np.pi * np.arange(NX) / 30.0)[None, :]
            + RATE * t_yr_ref)


def mean_epoch_s(stack):
    return float(np.mean((stack.time.values - stack.time.values.min())
                         / np.timedelta64(1, "s")))


def main() -> int:
    print("C1  ragged sampling: plain mean is biased, common-epoch mean is not")
    samp = ragged_sampling()
    stack, t_yr = build(samp)
    reg = dh_dt(stack)
    ce = common_epoch_mean(stack, reg["slope"], sigma_px=2.0)
    t_mid = mean_epoch_s(stack)
    t_mid_yr = t_mid / (86400.0 * 365.25)
    truth = truth_at(t_mid_yr)
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
    check("no-op under uniform, evenly spaced sampling", d < 1e-9, f"max|delta| {d:.1e}")
    t_uneven = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.5, 4.0, 6.5, 9.0, 11.0])
    stack_v, t_v = build(np.ones((len(t_uneven), NY, NX), bool), t_yr=t_uneven)
    reg_v = dh_dt(stack_v)
    ce_v = common_epoch_mean(stack_v, reg_v["slope"], sigma_px=2.0)
    d = float(np.nanmax(np.abs(ce_v.values - stack_v.mean("time").values)))
    skew = abs(RATE) * abs(np.mean(t_v) - np.median(t_v))
    check("no-op under uniform, UNEVENLY spaced sampling", d < 1e-9 and skew > 1.0,
          f"max|delta| {d:.1e} (median-referenced would be {skew:.2f} m)")

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

    print("C4  pixels below min_count beside well-sampled neighbours keep coverage")
    samp_h = np.ones((NT, NY, NX), bool)
    hole = (slice(20, 25), slice(40, 45))
    samp_h[:, hole[0], hole[1]] = False
    samp_h[:2, hole[0], hole[1]] = True                   # 2 early epochs < min_count
    stack_h, _ = build(samp_h)
    reg_h = dh_dt(stack_h, min_count=3)
    ce_h = common_epoch_mean(stack_h, reg_h["slope"], sigma_px=2.0)
    plain_h = stack_h.mean("time", skipna=True).values
    n_hole = int(np.isnan(reg_h["slope"].values[hole]).sum())
    check("the hole's own slope is undefined (the case under test)",
          n_hole == 25, f"{n_hole}/25 hole pixels have NaN slope")
    check("coverage identical to the plain mean",
          np.array_equal(np.isfinite(ce_h.values), np.isfinite(plain_h)),
          f"{int(np.isfinite(ce_h.values).sum())} vs {int(np.isfinite(plain_h).sum())} finite px")
    truth_h = truth_at(mean_epoch_s(stack_h) / (86400.0 * 365.25))
    e_plain_h = float(np.sqrt(np.mean((plain_h - truth_h)[hole] ** 2)))
    e_ce_h = float(np.sqrt(np.mean((ce_h.values - truth_h)[hole] ** 2)))
    check("hole borrows the neighbourhood trend and lands on H(t0)",
          e_plain_h > 1.0 and e_ce_h < 0.05 * e_plain_h,
          f"hole rmse vs H(t0): plain {e_plain_h:.2f} m -> corrected {e_ce_h:.3f} m")
    outside = np.ones((NY, NX), bool)
    outside[hole] = False
    d_out = float(np.nanmax(np.abs(ce_h.values - plain_h)[outside]))
    check("well-sampled neighbours are untouched", d_out < 1e-9, f"max|delta| {d_out:.1e}")

    print("C5  solver plumbing: common_epoch= is honoured, stamped, and DEFAULT OFF")
    from stereo_melt.dynamics.bridging_restoration import restored_budget_melt_rate
    from stereo_melt.freeboard import freeboard_to_thickness
    from stereo_melt.melt import eulerian_melt_rate
    vx = xr.DataArray(np.full((NY, NX), 300.0), dims=("y", "x"),
                      coords={"y": stack.y, "x": stack.x})
    vy = xr.zeros_like(vx)
    H_stack = freeboard_to_thickness(stack)
    plain_H = H_stack.mean("time", skipna=True)
    ce_H = common_epoch_mean(H_stack, dh_dt(H_stack)["slope"], sigma_px=2.0)
    solvers = (("eulerian_melt_rate", lambda **kw: eulerian_melt_rate(stack, vx, vy, **kw)),
               ("restored_budget_melt_rate",
                lambda **kw: restored_budget_melt_rate(stack, vx, vy, **kw)))
    for name, solve in solvers:
        off, on = solve(), solve(common_epoch=True)
        d_off = float(np.nanmax(np.abs(off.H_f_mean.values - plain_H.values)))
        d_on = float(np.nanmax(np.abs(on.H_f_mean.values - ce_H.values)))
        d_melt = float(np.nanmax(np.abs(on.melt_rate.values - off.melt_rate.values)))
        check(f"{name}: default is the plain mean, attr common_epoch=0",
              off.attrs.get("common_epoch") == 0 and d_off < 1e-9,
              f"attr {off.attrs.get('common_epoch')!r}  max|H_f_mean - plain| {d_off:.1e}")
        check(f"{name}: common_epoch=True uses the common-epoch mean, attr 1, melt changes",
              on.attrs.get("common_epoch") == 1 and d_on < 1e-9 and d_melt > 0.1,
              f"attr {on.attrs.get('common_epoch')!r}  max|H_f_mean - ce| {d_on:.1e}  "
              f"max|melt on - off| {d_melt:.2f} m/yr")

    print()
    if FAILS:
        print(f"GATE FAILED: {FAILS}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
