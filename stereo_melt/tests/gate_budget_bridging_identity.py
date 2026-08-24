"""Gate: the monolithic budget+bridging solver reduces to Shean Eq. 10.

The whole design rests on one claim — that setting the bridging operator to the
identity turns :func:`budget_bridging_melt_rate` back into
:func:`eulerian_melt_rate` **exactly**, for any positive weights, because the
collapsed per-pixel objective ``S_tt (D{m + a - div} - dHdt_obs)^2`` is then
minimised pixelwise at ``m = dHdt_obs + div - a``. If that fails, the operator
normalisation, the sign convention, or the moment collapse is wrong, and no
result from the bridging path can be believed.

Rungs
-----
G0  identity operator reproduces the Eulerian melt field (tight tolerance).
G1  same, with a non-trivial floating mask and ragged temporal sampling
    (NaNs per pixel) — the moment collapse must survive missing data.
G2  bridging ON is a genuine change (must NOT equal Eulerian) yet must keep the
    same spatial MEAN to within a few percent: D(0)=1 means the budget still
    fixes the level, which is the property the old high-passed operator lost.
G3  bridging ON recovers a known short-wavelength melt channel better than the
    hydrostatic Eulerian solver on a synthetic where the surface was generated
    THROUGH the bridging operator (an inverse-crime check of the plumbing, not
    of the physics — E2a/E1b is the non-crime test).
G4  the transfer itself equals the CLOSED-FORM flotation departure
    (1+d)B/(dB + R) at lambda = 2H/3H/4H, evaluated from a from-scratch
    transcription of Stubblefield Eq. 2.36-2.37 inside this file. No code path
    in the library can fake this rung: it is the one check the pre-2026-08-20
    operator (D = M_h/M_h(plateau), which silently dropped G_s) would have
    failed, and every other rung passed while it was wrong.
G5  the flux divergence belongs OUTSIDE the operator. On a synthetic built
    through the intended forward model with a non-uniform mean thickness (so
    div(H_f u) != 0), the shipped form recovers melt; putting the flux back
    inside the operator does not.

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY stereo_melt/tests/gate_budget_bridging_identity.py
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
from stereo_melt.melt import eulerian_melt_rate  # noqa: E402

NY, NX, RES = 96, 128, 250.0
H0, U0 = 500.0, 900.0
RHO = dict(rho_i=rhoi, rho_w=rhow)
C = (rhow - rhoi) / rhow           # freeboard / thickness


def _grid():
    y = np.arange(NY) * RES
    x = np.arange(NX) * RES
    return y, x


def synth_stack(melt_true, n_t=24, span_yr=6.0, ragged=False, seed=0):
    """Build a stack whose thickness trend is exactly the budget response.

    Uniform flow, zero SMB, uniform reference thickness -> flux divergence of
    the MEAN field is zero, so dH/dt = melt and the Eulerian answer is the
    truth. That isolates the collapse/normalisation, which is what this gates.
    """
    rng = np.random.default_rng(seed)
    y, x = _grid()
    t = np.linspace(0.0, span_yr, n_t)
    H = H0 + melt_true[None] * t[:, None, None]
    h = C * H                                   # d = 0, sea level 0
    if ragged:
        drop = rng.random(h.shape) < 0.35
        h = np.where(drop, np.nan, h)
    times = pd.to_datetime("2015-01-01") + pd.to_timedelta(t * 365.25, unit="D")
    stack = xr.DataArray(h, dims=("time", "y", "x"),
                         coords={"time": times, "y": y, "x": x})
    vx = xr.DataArray(np.full((NY, NX), U0), dims=("y", "x"),
                      coords={"y": y, "x": x})
    vy = xr.zeros_like(vx)
    return stack, vx, vy


def _stats(a, b, mask):
    d = (a - b)[mask]
    return float(np.nanmax(np.abs(d))), float(np.sqrt(np.nanmean(d ** 2)))


def main() -> int:
    y, x = _grid()
    xx = (np.arange(NX) * RES)[None, :] * np.ones((NY, 1))
    # a broad background plus a narrow across-flow channel (short lambda)
    broad = -3.0 - 1.5 * np.sin(2 * np.pi * xx / (NX * RES))
    chan = -8.0 * np.exp(-((xx - 0.45 * NX * RES) ** 2) / (2 * (2.5 * RES) ** 2))
    melt_true = broad + chan

    verdict = {}

    # ---------------- G0: identity == Eulerian ----------------
    stack, vx, vy = synth_stack(melt_true)
    eul = eulerian_melt_rate(stack, vx, vy, **RHO).melt_rate
    ident = budget_bridging_melt_rate(
        stack, vx, vy, bridging=False, lam=0.0, iters=1500, **RHO)
    fin = np.isfinite(eul.values) & np.isfinite(ident.melt_rate.values)
    mx, rms = _stats(ident.melt_rate.values, eul.values, fin)
    ok = mx < 5e-3
    verdict["G0"] = ok
    print(f"  gate[G0] identity vs Eulerian: max|Δ|={mx:.2e}  rms={rms:.2e} "
          f"m/yr  -> {'PASS' if ok else 'FAIL'}")

    # ---------------- G1: ragged sampling + mask ----------------
    stack_r, vx, vy = synth_stack(melt_true, n_t=40, ragged=True, seed=3)
    mask = xr.DataArray(np.ones((NY, NX), bool), dims=("y", "x"),
                        coords={"y": y, "x": x})
    mask.values[:8, :] = False
    eul_r = eulerian_melt_rate(stack_r, vx, vy, **RHO).melt_rate
    ident_r = budget_bridging_melt_rate(
        stack_r, vx, vy, bridging=False, floating_mask=mask, lam=0.0,
        iters=1500, **RHO)
    fin = (np.isfinite(eul_r.values) & np.isfinite(ident_r.melt_rate.values)
           & mask.values)
    mx1, rms1 = _stats(ident_r.melt_rate.values, eul_r.values, fin)
    ok = mx1 < 1e-2
    verdict["G1"] = ok
    print(f"  gate[G1] ragged+masked:        max|Δ|={mx1:.2e}  rms={rms1:.2e} "
          f"m/yr  -> {'PASS' if ok else 'FAIL'}")

    # ---- a stack built through the INTENDED forward model ----------------
    #   dH_f/dt = T{m + a} - div(H_f u)
    # Times are centred so the stack's time-mean thickness is exactly H_f0 and
    # the solver's own div(H_f u) is the one that went in. The earlier draft
    # built H(t) = H0 + T{m} t and called T{m} the observable, which is not a
    # mass budget: the time-mean thickness then varies in x, so a large flux
    # term the truth never contained leaked into every score.
    T_op = bridging_transfer_multiplier(2 * NY, 2 * NX, RES, RES, H=H0,
                                        ux_myr=U0, uy_myr=0.0, eta_bar=1e14)
    vx = xr.DataArray(np.full((NY, NX), U0), dims=("y", "x"),
                      coords={"y": y, "x": x})
    vy = xr.zeros_like(vx)

    def apply_T(f):
        pad = np.concatenate([f, f[::-1]], 0)
        pad = np.concatenate([pad, pad[:, ::-1]], 1)
        return np.real(np.fft.ifft2(T_op * np.fft.fft2(pad)))[:NY, :NX]

    def forward_stack(melt, Hf0, n_t=25, span=6.0):
        fdiv = flux_divergence(
            xr.DataArray(Hf0, dims=("y", "x"), coords={"y": y, "x": x}),
            vx, vy).values
        rate = apply_T(melt) - np.nan_to_num(fdiv)
        t = np.linspace(-0.5 * span, 0.5 * span, n_t)
        times = (pd.to_datetime("2015-01-01")
                 + pd.to_timedelta((t - t[0]) * 365.25, unit="D"))
        return xr.DataArray(
            C * (Hf0[None] + rate[None] * t[:, None, None]),
            dims=("time", "y", "x"),
            coords={"time": times, "y": y, "x": x})

    # ---------------- G3: the transfer earns its keep ----------------------
    # Uniform H_f0 so div(H_f u) = 0 and the ONLY difference between the two
    # solvers is the operator.
    stack_b = forward_stack(melt_true, np.full((NY, NX), H0))
    eul_b = eulerian_melt_rate(stack_b, vx, vy, **RHO).melt_rate.values
    inv_b = budget_bridging_melt_rate(
        stack_b, vx, vy, bridging=True, eta_bar=1e14, lam=1e-5, iters=4000,
        **RHO).melt_rate.values
    fin = np.isfinite(eul_b) & np.isfinite(inv_b)
    e_eul = float(np.sqrt(np.nanmean((eul_b - melt_true)[fin] ** 2)))
    e_inv = float(np.sqrt(np.nanmean((inv_b - melt_true)[fin] ** 2)))
    ok = e_inv < 0.5 * e_eul
    verdict["G3"] = ok
    print(f"  gate[G3] channel rmse: bridging {e_inv:.4f} vs hydrostatic "
          f"{e_eul:.4f} m/yr  -> {'PASS' if ok else 'FAIL'}")

    # ---------------- G2: the level survives ----------------
    # T(0) = 1 is what makes the melt MEAN identifiable -- the property the
    # high-passed anomaly operator threw away.
    m_true_mean = float(np.nanmean(melt_true[fin]))
    m_inv_mean = float(np.nanmean(inv_b[fin]))
    m_eul_mean = float(np.nanmean(eul_b[fin]))
    rel = abs(m_inv_mean - m_true_mean) / max(abs(m_true_mean), 1e-9)
    rel_eul = abs(m_eul_mean - m_true_mean) / max(abs(m_true_mean), 1e-9)
    ok = rel < rel_eul and rel < 0.02
    verdict["G2"] = ok
    print(f"  gate[G2] level: recovered mean {m_inv_mean:+.3f} vs truth "
          f"{m_true_mean:+.3f} ({100*rel:.2f}%) beats hydrostatic "
          f"{m_eul_mean:+.3f} ({100*rel_eul:.2f}%)  -> {'PASS' if ok else 'FAIL'}")

    # ---------------- G4: the transfer IS the flotation departure ----------
    # Independent transcription of Stubblefield Eq. 2.36-2.37 (R, B) and the
    # closed-form flotation departure T = (1+d) B / (d B + R), alpha = 0. This
    # touches no library code, so it cannot inherit a library bug.
    def _RB(kp):
        """R(k'), B(k') at non-dimensional wavenumber k' = k H, alpha = 0."""
        e2, e4 = np.exp(-2 * kp), np.exp(-4 * kp)
        e1, e3 = np.exp(-kp), np.exp(-3 * kp)
        den = kp * (1.0 - 2.0 * (1.0 + 2.0 * kp ** 2) * e2 + e4)
        R = (1.0 + 4.0 * kp * e2 - e4) / den
        B = 2.0 * ((kp + 1.0) * e1 + (kp - 1.0) * e3) / den
        return R, B

    delta = rhow / rhoi - 1.0
    # grid chosen so lambda = 2H, 3H, 4H land exactly on FFT bins
    gH, gdx, gnx = 500.0, 62.5, 96          # gnx*gdx = 6000 m = 12 H
    Tg = bridging_transfer_multiplier(4, gnx, gdx, gdx, H=gH, ux_myr=0.0,
                                      uy_myr=0.0, alpha_scale=0.0)
    print("  gate[G4] flotation departure vs closed form (alpha = 0):")
    g4 = [abs(Tg[0, 0] - 1.0) < 1e-12]
    print(f"      T(k=0) = {Tg[0, 0].real:.6f} (analytic limit 1)"
          f"  -> {'PASS' if g4[0] else 'FAIL'}")
    for j, lam_over_H in ((6, 2.0), (4, 3.0), (3, 4.0)):
        kp = 2.0 * np.pi / lam_over_H
        R, B = _RB(kp)
        ref = (1.0 + delta) * B / (delta * B + R)
        got = Tg[0, j]
        ok = abs(got.imag) < 1e-12 and abs(got.real - ref) < 2e-6
        g4.append(ok)
        print(f"      lambda = {lam_over_H:.0f}H: |T| = {abs(got):.6f}   "
              f"closed form {ref:.6f}   damping {100*(1-ref):4.1f}%  "
              f"-> {'PASS' if ok else 'FAIL'}")
    verdict["G4"] = all(g4)

    # ---------------- G5: the flux divergence lives OUTSIDE ----------------
    # H_f0 now carries structure at 4H, where |T| = 0.86, so div(H_f u) has
    # power in the band the operator actually filters. (A linear ramp would not
    # discriminate: its divergence is a constant, i.e. pure DC, and T(0) = 1.)
    Hf0 = (500.0 - 2.0e-3 * xx
           + 40.0 * np.sin(2 * np.pi * xx / (NX * RES))
           + 12.0 * np.sin(2 * np.pi * xx / (8 * RES)))
    stack5 = forward_stack(melt_true, Hf0)
    kw = dict(bridging=True, eta_bar=1e14, lam=1e-5, iters=4000, **RHO)
    err5 = {}
    for lab, fio in (("outside (shipped)", False), ("inside (pre-fix)", True)):
        mm = budget_bridging_melt_rate(stack5, vx, vy, flux_in_operator=fio,
                                       **kw).melt_rate.values
        good = np.isfinite(mm)
        err5[lab] = float(np.sqrt(np.nanmean((mm - melt_true)[good] ** 2)))
    ok = (err5["outside (shipped)"] < 0.25
          and err5["outside (shipped)"] < 0.5 * err5["inside (pre-fix)"])
    verdict["G5"] = ok
    print(f"  gate[G5] flux placement rmse: outside "
          f"{err5['outside (shipped)']:.4f} vs inside "
          f"{err5['inside (pre-fix)']:.4f} m/yr  "
          f"-> {'PASS' if ok else 'FAIL'}")

    allok = all(verdict.values())
    print(f"\nOVERALL GATE: {'PASS' if allok else 'FAIL'}  ({verdict})")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
