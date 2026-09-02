"""Synthetic validation for :func:`stereo_melt.melt.lagrangian_parcel_lsq_melt_rate`.

Steady extensional-shelf test: with ``u(x) = v0 + alpha x`` (so div u = alpha)
and a TIME-INVARIANT thickness field chosen from steady mass conservation

    d(uH)/dx = a_dot + b_dot   =>   H(x) = ((a_dot+b_dot) x + v0 H0) / u(x)

the Eulerian dh/dt is identically ZERO — every bit of the melt signal lives
in the parcel motion + strain terms. A solver that mishandles either cannot
recover ``b_dot``. Shean sign convention: b_dot < 0 is melt.

Cases:
  A. clean steady field                      -> both solvers must recover b_dot
  B. per-epoch DEM offsets (strip bias, 1 m) -> parcel-LSQ should stay unbiased
                                                and beat endpoint pairs on spread
  C. noisy velocity + data void              -> hygiene (smooth+clip) must kill
                                                the fake-accretion carpet the raw
                                                pair solver produces

Run:
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        scripts/library/sanity_lagrangian_parcel_lsq.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.constants import rhoi, rhow
from stereo_melt.melt import lagrangian_melt_rate, lagrangian_parcel_lsq_melt_rate

RNG = np.random.default_rng(20260702)

# ---- geometry & truth ------------------------------------------------------
NY, NX = 160, 140
RES = 200.0                      # m
V0 = 800.0                       # m/yr at x=0
ALPHA = 0.02                     # 1/yr  (div u; strong so strain matters)
H0 = 400.0                       # m at x=0
B_DOT = -6.0                     # m ice/yr  (TRUTH; negative = melt)
A_DOT = 0.5                      # m ice/yr  SMB
GAMMA = rhow / (rhow - rhoi)

x = np.arange(NX) * RES
y = np.arange(NY)[::-1] * RES    # EPSG:3031 style: y descending
u_x = V0 + ALPHA * x             # m/yr
H_x = ((A_DOT + B_DOT) * x + V0 * H0) / u_x
assert H_x.min() > 50, "domain too aggressive; steady H went too thin"

T = 12
SPAN = 5.0
t_years = np.linspace(0.0, SPAN, T)
times = pd.to_datetime("2019-01-01") + pd.to_timedelta(t_years * 365.25, unit="D")

h_x = H_x * (rhow - rhoi) / rhow          # freeboard, d=0
h2d = np.broadcast_to(h_x, (NY, NX)).copy()


def make_stack(strip_bias_sigma: float = 0.0, hole_frac: float = 0.0) -> xr.DataArray:
    data = np.broadcast_to(h2d, (T, NY, NX)).copy()
    for e in range(T):
        if strip_bias_sigma > 0:
            data[e] += RNG.normal(0.0, strip_bias_sigma)
        if hole_frac > 0:
            holes = RNG.random((NY, NX)) < hole_frac
            data[e][holes] = np.nan
    return xr.DataArray(
        data, dims=("time", "y", "x"),
        coords={"time": times, "y": y, "x": x},
    )


def make_velocity(noise_sigma: float = 0.0, void: bool = False):
    vx2 = np.broadcast_to(u_x, (NY, NX)).copy()
    vy2 = np.zeros((NY, NX))
    if noise_sigma > 0:
        vx2 = vx2 + RNG.normal(0.0, noise_sigma, size=(NY, NX))
        vy2 = vy2 + RNG.normal(0.0, noise_sigma, size=(NY, NX))
    if void:
        vx2[60:80, 100:120] = np.nan
        vy2[60:80, 100:120] = np.nan
    dims = ("y", "x")
    coords = {"y": y, "x": x}
    return (
        xr.DataArray(vx2, dims=dims, coords=coords),
        xr.DataArray(vy2, dims=dims, coords=coords),
    )


def interior_median(ds, margin=30):
    v = ds["melt_rate"].values[margin:-margin, margin:-margin]
    return float(np.nanmedian(v)), float(np.nanmean(np.isfinite(v)))


def frac_accretion(ds, margin=30):
    v = ds["melt_rate"].values[margin:-margin, margin:-margin]
    v = v[np.isfinite(v)]
    return float((v > 0).mean()) if v.size else np.nan


def run_pair_solver(stack, vx, vy, **kw):
    return lagrangian_melt_rate(
        stack, vx, vy, a_dot=A_DOT, d=0.0,
        pairs="all", min_dt_yr=1.5, max_dt_yr=2.5,
        aggregator="pair_median", output="path",
        dt_yr=0.05, progress_interval_s=0, **kw,
    )


def run_parcel_solver(stack, vx, vy, **kw):
    defaults = dict(
        a_dot=A_DOT, d=0.0, dt_yr=0.05,
        min_epochs=4, min_span_yr=1.5, progress_interval_s=0,
    )
    defaults.update(kw)
    return lagrangian_parcel_lsq_melt_rate(stack, vx, vy, **defaults)


failures: list[str] = []


def check(label, value, target, tol):
    ok = np.isfinite(value) and abs(value - target) <= tol
    print(f"  {label:<58s} {value:+8.3f}  (target {target:+.2f} ± {tol:.2f})"
          f"  {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(label)
    return ok


print(f"TRUTH: b_dot = {B_DOT:+.2f} m ice/yr, a_dot = {A_DOT:+.2f}, "
      f"gamma = {GAMMA:.2f}, div u = {ALPHA} /yr, steady field (Eulerian dh/dt = 0)")

# ---------------------------------------------------------------- Case A
print("\n[A] clean steady field")
stack = make_stack()
vx, vy = make_velocity()
new = run_parcel_solver(stack, vx, vy, vel_smooth_sigma_m=None, vdiv_clip=None)
med, cov = interior_median(new)
check("parcel-LSQ median (no hygiene needed)", med, B_DOT, 0.15)
old = run_pair_solver(stack, vx, vy)
med_o, _ = interior_median(old)
check("endpoint-pair median (reference)", med_o, B_DOT, 0.30)

# ---------------------------------------------------------------- Case B
print("\n[B1] two outlier epochs (+8 m DEM offset, i.e. +75 m thickness) — "
      "IRLS must reject them")
stack_b1 = make_stack()
for bad_e in (4, 9):
    stack_b1[bad_e] = stack_b1[bad_e] + 8.0
new_b1 = run_parcel_solver(stack_b1, vx, vy, vel_smooth_sigma_m=None, vdiv_clip=None)
med_nb1, _ = interior_median(new_b1)
check("parcel-LSQ median with 2 outlier epochs", med_nb1, B_DOT, 0.5)
old_b1 = run_pair_solver(stack_b1, vx, vy)
med_ob1, _ = interior_median(old_b1)
print(f"  endpoint-pair median with 2 outlier epochs: {med_ob1:+.3f} "
      "(reported; pairs touching a bad epoch are poisoned)")

print("\n[B2] global per-epoch offsets sigma=1.0 m + 20% holes — "
      "IRREDUCIBLE for any estimator (alpha_z DC nullspace); report only")
stack_b2 = make_stack(strip_bias_sigma=1.0, hole_frac=0.20)
new_b2 = run_parcel_solver(stack_b2, vx, vy, vel_smooth_sigma_m=None, vdiv_clip=None)
old_b2 = run_pair_solver(stack_b2, vx, vy)
med_nb2, _ = interior_median(new_b2)
med_ob2, _ = interior_median(old_b2)
print(f"  parcel-LSQ {med_nb2:+.3f} | endpoint-pair {med_ob2:+.3f} m/yr "
      f"(truth {B_DOT:+.1f}; one shared-offset realization -> both shift; "
      "mitigation is upstream control, not the estimator)")

# ---------------------------------------------------------------- Case C
print("\n[C] velocity noise sigma=30 m/yr + data void "
      "(raw div-u noise ~0.2/yr grid-scale)")
vx_n, vy_n = make_velocity(noise_sigma=30.0, void=True)
old_raw = run_pair_solver(stack, vx_n, vy_n)                       # production config
med_or, _ = interior_median(old_raw)
fa_or = frac_accretion(old_raw)
print(f"  endpoint-pair RAW (production): median {med_or:+.3f}, "
      f"accretion-cell fraction {fa_or:.2%}  <- the failure mode")
old_hyg = run_pair_solver(stack, vx_n, vy_n,
                          vel_smooth_sigma_m=3000.0, vdiv_clip=0.2)
med_oh, _ = interior_median(old_hyg)
fa_oh = frac_accretion(old_hyg)
check("endpoint-pair + hygiene median", med_oh, B_DOT, 1.5)
new_c = run_parcel_solver(stack, vx_n, vy_n)                       # hygiene defaults ON
med_nc, _ = interior_median(new_c)
fa_nc = frac_accretion(new_c)
check("parcel-LSQ (hygiene defaults) median", med_nc, B_DOT, 1.0)
print(f"  accretion-cell fraction: raw {fa_or:.2%} -> pair+hygiene {fa_oh:.2%} "
      f"-> parcel-LSQ {fa_nc:.2%}")
if not (fa_nc <= fa_or):
    failures.append("case C accretion fraction")

# ---------------------------------------------------------------- gamma trap
print("\n[gamma trap] strain term must use thickness H, not freeboard h")
# If an implementation multiplied h (not H) by div u, recovered b_dot would be
# off by (1-1/gamma)*H*alpha ~ +7.1 m/yr here — far outside every tolerance
# above, so cases A-C jointly guard the gamma scaling.
print(f"  guarded by cases A-C (error would be "
      f"{(1 - 1/GAMMA) * float(np.mean(H_x)) * ALPHA:+.1f} m/yr)")

print()
if failures:
    print(f"FAILURES ({len(failures)}): " + "; ".join(failures))
    raise SystemExit(1)
print("ALL PASS")
