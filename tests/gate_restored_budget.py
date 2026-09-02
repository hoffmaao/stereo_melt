"""Gate: the RESTORE-THEN-BUDGET solver (thickness-space deconvolution, then
the plain Eulerian budget).

R1  IN-BAND RECOVERY. On a stack built through the bridged forward model
    (uniform geometry, across-flow melt at 4H — inside the 2.5H band), the
    solver recovers the melt to < 5 % of the truth amplitude while the
    plain Eulerian misses 1 - T(4H) = 46 % of it.
R2  ACROSS-FLOW IDENTITY. For a field varying only ACROSS the flow (k ⊥ u)
    the flow-projection weight is zero, so the solver is EXACTLY the
    Eulerian budget (max |dm| < 1e-9) — the along-flow over-read of the
    monolithic operator cannot occur by construction.
R3  STEADY VELOCITY-CARRIED MELT IS EXACT. A k ⊥ u thickness structure in
    steady state with the melt balanced by the divergence of a varying
    velocity (dH/dt = 0): the filter has nothing to lift (weight 0 on
    k ⊥ u, no observed rate), and the budget on the restored thickness
    reproduces the melt exactly — the configuration the monolithic
    operator over-reads by 1/T.

The full-Stokes validation is the E2a ideal-surface suite (2026-08-23):
along-flow gain 0.959/nrmse 0.107 (= Eulerian; monolithic 1.54/0.575),
across-flow gain 0.51/corr 0.94 (Eulerian 0.07).

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_restored_budget.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from stereo_melt.constants import rhoi, rhow  # noqa: E402
from stereo_melt.dynamics.bridging_restoration import restored_budget_melt_rate  # noqa: E402
from stereo_melt.dynamics.budget_bridging import bridging_transfer_multiplier  # noqa: E402
from stereo_melt.kinematics import flux_divergence  # noqa: E402
from stereo_melt.melt import eulerian_melt_rate  # noqa: E402

NY, NX, RES = 64, 128, 250.0
H0, U0 = 500.0, 900.0
RHO = dict(rho_i=rhoi, rho_w=rhow)
C = (rhow - rhoi) / rhow
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def forward_stack(rate, Hf0, y, x, n_t=25, span=6.0):
    t = np.linspace(-0.5 * span, 0.5 * span, n_t)
    times = pd.to_datetime("2015-01-01") + pd.to_timedelta((t - t[0]) * 365.25, unit="D")
    return xr.DataArray(C * (Hf0[None] + rate[None] * t[:, None, None]),
                        dims=("time", "y", "x"), coords={"time": times, "y": y, "x": x})


def main() -> int:
    y = np.arange(NY)[::-1] * RES
    x = np.arange(NX) * RES
    xx, yy = np.meshgrid(x, y)
    da = lambda a: xr.DataArray(a, dims=("y", "x"), coords={"y": y, "x": x})  # noqa: E731
    vx, vy = da(np.full((NY, NX), U0)), da(np.zeros((NY, NX)))
    T = bridging_transfer_multiplier(2 * NY, 2 * NX, RES, RES, H=H0, ux_myr=U0,
                                     uy_myr=0.0, eta_bar=1e14)

    def apply_T(f):
        p = np.concatenate([f, f[::-1]], 0)
        p = np.concatenate([p, p[:, ::-1]], 1)
        return np.real(np.fft.ifft2(T * np.fft.fft2(p)))[:NY, :NX]

    inner = (xx > 4000) & (xx < x[-1] - 4000)

    print("R1  in-band recovery (4H across-flow cosine through the bridged forward)")
    melt = -5.0 * np.cos(2 * np.pi * xx / (4 * H0))
    st = forward_stack(apply_T(melt), np.full((NY, NX), H0), y, x)
    rb = restored_budget_melt_rate(st, vx, vy, **RHO).melt_rate.values
    eu = eulerian_melt_rate(st, vx, vy, **RHO).melt_rate.values
    e_rb = float(np.sqrt(np.nanmean((rb - melt)[inner] ** 2)) / 5.0)
    e_eu = float(np.sqrt(np.nanmean((eu - melt)[inner] ** 2)) / 5.0)
    check("restored budget recovers in-band melt", e_rb < 0.05 and e_eu > 0.35,
          f"rmse/amp: restored {e_rb:.3f}  Eulerian {e_eu:.3f} "
          f"(1 - T(4H) = 0.46 expected for the Eulerian)")

    print("R2  across-flow identity (k ⊥ u -> filter weight 0)")
    melt_y = -3.0 * np.cos(2 * np.pi * yy / (4 * H0))
    st2 = forward_stack(melt_y, np.full((NY, NX), H0), y, x)  # NO bridging applied
    rb2 = restored_budget_melt_rate(st2, vx, vy, **RHO).melt_rate.values
    eu2 = eulerian_melt_rate(st2, vx, vy, **RHO).melt_rate.values
    d = float(np.nanmax(np.abs(rb2 - eu2)))
    check("== Eulerian for k ⊥ u", d < 1e-9, f"max|dm| {d:.1e}")

    print("R3  the steady along-flow configuration is exact (velocity-carried melt)")
    # k ⊥ u thickness structure, velocity divergence carries the melt,
    # dH/dt = 0: nothing for the filter to lift (F = 1 on k ⊥ u; dHdt_obs = 0),
    # the budget on the restored thickness is the plain (exact) budget.
    H3 = np.full((NY, NX), H0) - 30.0 * np.cos(2 * np.pi * yy / (4 * H0))
    vx3 = da(U0 + 20.0 * np.sin(2 * np.pi * xx / (8 * H0)))
    m_true = np.nan_to_num(flux_divergence(da(H3), vx3, vy).values)
    st3 = forward_stack(np.zeros((NY, NX)), H3, y, x)
    rb3 = restored_budget_melt_rate(st3, vx3, vy, **RHO).melt_rate.values
    rel = float(np.sqrt(np.nanmean((rb3 - m_true)[inner] ** 2))
                / np.sqrt(np.mean(m_true[inner] ** 2)))
    mono_note = "the monolithic would lift this by 1/T"
    check("steady velocity-carried melt exact", rel < 0.02, f"rel {rel:.2e} ({mono_note})")

    print()
    if FAILS:
        print(f"GATE FAILED: {FAILS}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
