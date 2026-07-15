"""Correctness test for the Shean ``output='origin'`` Lagrangian scheme.

Hand-built STEADY shelves (time-invariant surface) with closed-form melt.
Because the stack is steady (∂h/∂t = 0), mass conservation forces the basal
term to equal the flux divergence, so the Lagrangian solver must return

    b_dot = DH/Dt + H ∇·u  =  vx·dH/dx + H·∇·u            (Shean convention)

which is known analytically from the prescribed vx(x) and H(x). No forward
melt model is involved, so the recovery target is not circular.

Cases:
  A  uniform v, linear H            -> melt = -1 m/yr everywhere
  B  vx = v0 + s·x (∇·u = s), curved H(x)
                                    -> melt(x) = vx·dH/dx + H·s  (both signs)
  C  v = 0 (stagnant)               -> origin must DROP every parcel (all NaN)

We assert: (A) both modes hit -1; (B) origin recovers the closed form at
least as well as path, with NO grid-scale checkerboard and full interior
coverage (the whole point of the scheme), while path@stride2 checkers;
(C) drop-stuck fires.

Run::
    cd stereo_melt && /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u tests/sanity_lagrangian_origin.py
"""
import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.constants import rhoi, rhow
from stereo_melt.melt import lagrangian_melt_rate

RH = (rhow - rhoi) / rhow  # freeboard = RH * thickness  (firn d=0)
COMMON = dict(a_dot=0.0, dt_yr=0.05, pairs="all", min_dt_yr=0.5,
              max_dt_yr=1.5, progress_interval_s=0.0)


def steady_stack(H2d, x, y, n=8, t0="2013-01-01", dt_days=200):
    """A time-invariant freeboard stack from a thickness field (steady state)."""
    h0 = H2d * RH
    times = pd.to_datetime(t0) + pd.to_timedelta(np.arange(n) * dt_days, unit="D")
    layers = [xr.DataArray(h0.copy(), dims=("y", "x"), coords={"y": y, "x": x})
              for _ in times]
    return xr.concat(layers, dim=pd.Index(times, name="time"))


def checker_amp(a):
    """Grid-Nyquist (checkerboard) amplitude as a fraction of total |anomaly|.

    0 for a smooth field, ~1 for a perfect every-other-cell checker.
    """
    ny, nx = a.shape
    yy, xx = np.indices((ny, nx))
    sign = np.where((yy + xx) % 2 == 0, 1.0, -1.0)
    fin = np.isfinite(a)
    if fin.sum() == 0:
        return np.nan
    a0 = np.where(fin, a - np.nanmean(a), 0.0)
    return abs(float(np.sum(a0 * sign))) / max(float(np.sum(np.abs(a0) * fin)), 1e-9)


def recover(stack, vx, vy, *, output, seed_stride):
    r = lagrangian_melt_rate(stack, vx, vy, seed_stride=seed_stride,
                             output=output, **COMMON)
    return r.melt_rate.values, r.attrs.get("output")


print("=== Case A: uniform v, linear H  ->  melt = -1 m/yr ===")
xA = np.arange(-10000, 10000, 250.0)
yA = np.arange(10000, -10000, -250.0)
_, XA = np.meshgrid(yA, xA, indexing="ij")
HA = 300.0 - 0.005 * XA
vxA = xr.DataArray(np.full_like(HA, 200.0), dims=("y", "x"), coords={"y": yA, "x": xA})
vyA = xr.zeros_like(vxA)
stA = steady_stack(HA, xA, yA)
for mode in ("path", "origin"):
    rec, omode = recover(stA, vxA, vyA, output=mode, seed_stride=1)
    core = float(np.nanmean(rec[10:-10, 10:-10]))
    print(f"  {mode:6s} (attr output={omode}): interior mean = {core:+.3f}  (expect -1.000)")
    assert abs(core + 1.0) < 0.05, f"A {mode} core {core} far from -1"

print("\n=== Case B: vx=v0+s*x (div u = s), curved H(x)  ->  melt(x)=vx*dH/dx + H*s ===")
L, res = 40000.0, 250.0
xB = np.arange(-L / 2, L / 2, res)
yB = np.arange(L / 2, -L / 2, -res)
_, XB = np.meshgrid(yB, xB, indexing="ij")
v0, s = 300.0, 0.004
A, lam, H0 = 50.0, 40000.0, 400.0
HB = H0 + A * np.cos(2 * np.pi * XB / lam)
dHdx = -A * (2 * np.pi / lam) * np.sin(2 * np.pi * XB / lam)
vxB_field = v0 + s * XB
m_true = vxB_field * dHdx + HB * s          # Shean b_dot (negative = melt)
vxB = xr.DataArray(vxB_field, dims=("y", "x"), coords={"y": yB, "x": xB})
vyB = xr.zeros_like(vxB)
stB = steady_stack(HB, xB, yB)
print(f"  m_true: min={m_true.min():+.2f}  max={m_true.max():+.2f}  "
      f"mean={m_true.mean():+.3f} m/yr  (interior box for stats)")

by, bx = slice(6, -6), slice(28, -28)   # drop advection edges (flow is +x)
results = {}
for tag, mode, stride in [("path@stride2", "path", 2),
                          ("path@stride1", "path", 1),
                          ("origin", "origin", 1)]:
    rec, _ = recover(stB, vxB, vyB, output=mode, seed_stride=stride)
    v, t = rec[by, bx], m_true[by, bx]
    fin = np.isfinite(v)
    corr = float(np.corrcoef(v[fin], t[fin])[0, 1])
    mae = float(np.nanmedian(np.abs(v[fin] - t[fin])))
    meanrec, meantrue = float(np.nanmean(v)), float(np.nanmean(t))
    nanpct = 100.0 * float(np.mean(~fin))
    chk = checker_amp(rec[by, bx])
    results[tag] = dict(corr=corr, mae=mae, meanrec=meanrec, meantrue=meantrue,
                        nanpct=nanpct, chk=chk)
    print(f"  {tag:13s}: corr={corr:.3f}  medAE={mae:.3f}  "
          f"mean={meanrec:+.3f}(true {meantrue:+.3f})  nan%={nanpct:4.1f}  checker={chk:.3f}")

o, p2 = results["origin"], results["path@stride2"]
assert o["corr"] > 0.95, f"origin recovery corr too low: {o['corr']}"
assert o["mae"] < 0.5, f"origin median abs err too high: {o['mae']}"
assert abs(o["meanrec"] - o["meantrue"]) < 0.25, "origin mean melt off"
assert o["nanpct"] < 2.0, f"origin interior coverage incomplete: {o['nanpct']}%"
assert o["chk"] < 0.05, f"origin shows a checkerboard: {o['chk']}"
assert o["chk"] <= p2["chk"], "origin checker not below path@stride2"
assert o["corr"] >= p2["corr"] - 0.02, "origin recovery worse than path@stride2"
print(f"  -> origin checker {o['chk']:.3f} vs path@stride2 {p2['chk']:.3f}; "
      f"coverage {100-o['nanpct']:.0f}% vs {100-p2['nanpct']:.0f}%")

print("\n=== Case C: stagnant ice (v=0)  ->  origin drops every parcel ===")
vx0 = xr.zeros_like(vxB)
rec, _ = recover(stB, vx0, vx0, output="origin", seed_stride=1)
finite_pct = 100.0 * float(np.mean(np.isfinite(rec)))
print(f"  origin finite cells = {finite_pct:.1f}%  (expect ~0 — all parcels stuck)")
assert finite_pct < 1.0, f"drop-stuck failed: {finite_pct}% finite"

print("\nPASS: origin mode recovers the closed-form melt, is checkerboard-free with "
      "full interior coverage, path mode is unchanged, and drop-stuck fires.")
