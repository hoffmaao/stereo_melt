"""Gate: the LOCAL (per-geometry-bin, blended) monolithic bridging operator.

``budget_bridging_melt_rate(n_bins > 1)`` clusters the fit cells by
``(H, u_x, u_y[, eta])``, builds one transfer per bin, applies each to the
whole padded domain and blends the responses with partition-of-unity weights
(the ``BlendedStubblefieldForward`` construction). Rungs:

L1  REDUCTION. Uniform geometry: ``n_bins=4`` collapses to one bin and is
    bit-identical to the global multiplier (max |dm| < 1e-9 m/yr).
L2  IT VARIES. On a two-thickness shelf (400 m | 600 m, smooth step) the
    two-bin operator's responses to the same melt differ between halves by a
    real margin (guards against the bins silently sharing one multiplier).
L3  LOCAL RECOVERY. A stack generated through the two-bin forward is
    recovered by the two-bin inverse with < 1/3 the channel rmse of the
    single-``H_ref`` inverse (inverse crime by construction: this gates the
    plumbing and the adjoint-through-autograd, not the physics).
L4  ETA PER BIN. With an ``eta_field`` that is 3e13 on one half and 3e14 on
    the other, the bin geometry carries the two viscosities (not the median).

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY stereo_melt/tests/gate_budget_bridging_local.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")

from stereo_melt.constants import rhoi, rhow  # noqa: E402
from stereo_melt.dynamics.budget_bridging import (  # noqa: E402
    bridging_transfer_multiplier,
    budget_bridging_melt_rate,
)
from stereo_melt.kinematics import flux_divergence  # noqa: E402

NY, NX, RES = 64, 192, 250.0
U0 = 900.0
RHO = dict(rho_i=rhoi, rho_w=rhow)
C = (rhow - rhoi) / rhow
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def pad_apply(T, f):
    pad = np.concatenate([f, f[::-1]], 0)
    pad = np.concatenate([pad, pad[:, ::-1]], 1)
    return np.real(np.fft.ifft2(T * np.fft.fft2(pad)))[:NY, :NX]


def main() -> int:
    y = np.arange(NY)[::-1] * RES
    x = np.arange(NX) * RES
    xx = x[None, :] * np.ones((NY, 1))
    vx = xr.DataArray(np.full((NY, NX), U0), dims=("y", "x"), coords={"y": y, "x": x})
    vy = xr.zeros_like(vx)
    # melt: broad background + two narrow across-flow channels, one per half
    broad = -3.0 - 1.0 * np.sin(2 * np.pi * xx / (NX * RES))
    chan = (-8.0 * np.exp(-((xx - 0.25 * NX * RES) ** 2) / (2 * (2.0 * RES) ** 2))
            - 8.0 * np.exp(-((xx - 0.75 * NX * RES) ** 2) / (2 * (2.0 * RES) ** 2)))
    melt = broad + chan
    # two-thickness shelf with a smooth step at mid-domain
    Hf0 = 500.0 + 100.0 * np.tanh((xx - 0.5 * NX * RES) / 2000.0)
    left = xx < 0.5 * NX * RES

    def forward_stack(melt, Hf0, T_of_x, n_t=25, span=6.0):
        fdiv = flux_divergence(xr.DataArray(Hf0, dims=("y", "x"),
                                            coords={"y": y, "x": x}), vx, vy).values
        rate = T_of_x(melt) - np.nan_to_num(fdiv)
        t = np.linspace(-0.5 * span, 0.5 * span, n_t)
        times = (pd.to_datetime("2015-01-01")
                 + pd.to_timedelta((t - t[0]) * 365.25, unit="D"))
        return xr.DataArray(C * (Hf0[None] + rate[None] * t[:, None, None]),
                            dims=("time", "y", "x"),
                            coords={"time": times, "y": y, "x": x})

    kw = dict(eta_bar=1e14, lam=1e-5, iters=3000, converge_tol=1e-12, **RHO)

    print("L1  reduction on uniform geometry")
    T_u = bridging_transfer_multiplier(2 * NY, 2 * NX, RES, RES, H=500.0,
                                       ux_myr=U0, uy_myr=0.0, eta_bar=1e14)
    st_u = forward_stack(melt, np.full((NY, NX), 500.0), lambda f: pad_apply(T_u, f))
    m1 = budget_bridging_melt_rate(st_u, vx, vy, bridging=True, n_bins=1, **kw)
    m4 = budget_bridging_melt_rate(st_u, vx, vy, bridging=True, n_bins=4, **kw)
    d = float(np.nanmax(np.abs(m1.melt_rate.values - m4.melt_rate.values)))
    check("n_bins=4 == n_bins=1 on uniform geometry", d < 1e-9,
          f"max|dm| {d:.1e}  bins={m4.attrs['n_bins']}")

    print("L2  the bins differ")
    T_L = bridging_transfer_multiplier(2 * NY, 2 * NX, RES, RES, H=400.0,
                                       ux_myr=U0, uy_myr=0.0, eta_bar=1e14)
    T_R = bridging_transfer_multiplier(2 * NY, 2 * NX, RES, RES, H=600.0,
                                       ux_myr=U0, uy_myr=0.0, eta_bar=1e14)
    rL, rR = pad_apply(T_L, chan), pad_apply(T_R, chan)
    rel = float(np.sqrt(np.mean((rL - rR) ** 2)) / np.sqrt(np.mean(rL ** 2)))
    check("400 m vs 600 m responses differ", rel > 0.05, f"rel rms diff {rel:.3f}")

    print("L3  local recovery (two-bin forward, inverse crime)")
    # hard-split blended forward: left half sees T_L, right half T_R
    wL = np.where(left, 1.0, 0.0)

    def T_local(f):
        return wL * pad_apply(T_L, f) + (1 - wL) * pad_apply(T_R, f)

    st_2 = forward_stack(melt, Hf0, T_local)
    inv1 = budget_bridging_melt_rate(st_2, vx, vy, bridging=True, n_bins=1,
                                     **kw).melt_rate.values
    inv2 = budget_bridging_melt_rate(st_2, vx, vy, bridging=True, n_bins=2,
                                     blend_px=1.0, **kw)
    inv2v = inv2.melt_rate.values
    # score away from the step and the mirror edges
    sc = (np.abs(xx - 0.5 * NX * RES) > 3000.0) & (xx > 2000.0) & (xx < x[-1] - 2000.0)
    e1 = float(np.sqrt(np.nanmean((inv1 - melt)[sc] ** 2)))
    e2 = float(np.sqrt(np.nanmean((inv2v - melt)[sc] ** 2)))
    check("two-bin inverse beats single H_ref by 3x", e2 < e1 / 3.0,
          f"rmse two-bin {e2:.4f} vs single {e1:.4f} m/yr; "
          f"bins {inv2.attrs['bin_geometry']}")

    print("L4  eta per bin")
    eta_f = xr.DataArray(np.where(left, 3e13, 3e14), dims=("y", "x"),
                         coords={"y": y, "x": x})
    inv_e = budget_bridging_melt_rate(st_2, vx, vy, bridging=True, n_bins=2,
                                      blend_px=1.0, eta_field=eta_f, **kw)
    etas = sorted(float(row.split(",")[3]) for row in inv_e.attrs["bin_geometry"].split(";"))
    check("bins carry 3e13 and 3e14", len(etas) == 2 and abs(etas[0] / 3e13 - 1) < 0.05
          and abs(etas[1] / 3e14 - 1) < 0.05, f"{etas}")

    print()
    if FAILS:
        print(f"GATE FAILED: {FAILS}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
