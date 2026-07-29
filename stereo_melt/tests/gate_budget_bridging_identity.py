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
    budget_bridging_melt_rate,
    normalized_bridging_multiplier,
)
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
        stack, vx, vy, bridging=False, lam=0.0, iters=1500, lr=0.05, **RHO)
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
        iters=1500, lr=0.05, **RHO)
    fin = (np.isfinite(eul_r.values) & np.isfinite(ident_r.melt_rate.values)
           & mask.values)
    mx1, rms1 = _stats(ident_r.melt_rate.values, eul_r.values, fin)
    ok = mx1 < 1e-2
    verdict["G1"] = ok
    print(f"  gate[G1] ragged+masked:        max|Δ|={mx1:.2e}  rms={rms1:.2e} "
          f"m/yr  -> {'PASS' if ok else 'FAIL'}")

    # ---- a surface generated THROUGH the operator: the self-consistent case.
    # G2 and G3 both score against it (applying the bridging inverse to a
    # hydrostatically-generated surface, as an earlier draft of G2 did, asks it
    # to deconvolve something that was never convolved -- it over-sharpens by
    # construction and the test says nothing).
    D = normalized_bridging_multiplier(2 * NY, 2 * NX, RES, RES, H=H0,
                                       ux_myr=U0, uy_myr=0.0, eta_bar=1e14)
    pad = np.concatenate([melt_true, melt_true[::-1]], 0)
    pad = np.concatenate([pad, pad[:, ::-1]], 1)
    bridged = np.real(np.fft.ifft2(D * np.fft.fft2(pad)))[:NY, :NX]
    t = np.linspace(0.0, 6.0, 24)
    H = H0 + bridged[None] * t[:, None, None]
    times = pd.to_datetime("2015-01-01") + pd.to_timedelta(t * 365.25, unit="D")
    stack_b = xr.DataArray(C * H, dims=("time", "y", "x"),
                           coords={"time": times, "y": y, "x": x})
    eul_b = eulerian_melt_rate(stack_b, vx, vy, **RHO).melt_rate.values
    inv_b = budget_bridging_melt_rate(
        stack_b, vx, vy, bridging=True, eta_bar=1e14, lam=1e-5, iters=4000,
        lr=0.05, **RHO).melt_rate.values
    fin = np.isfinite(eul_b) & np.isfinite(inv_b)
    e_eul = float(np.sqrt(np.nanmean((eul_b - melt_true)[fin] ** 2)))
    e_inv = float(np.sqrt(np.nanmean((inv_b - melt_true)[fin] ** 2)))
    ok = e_inv < e_eul
    verdict["G3"] = ok
    print(f"  gate[G3] channel rmse: bridging {e_inv:.3f} vs hydrostatic "
          f"{e_eul:.3f} m/yr  -> {'PASS' if ok else 'FAIL'}")

    # ---------------- G2: the level survives ----------------
    # D(0)=1 is what makes the melt MEAN identifiable -- the property the
    # high-passed anomaly operator threw away. On the self-consistent surface
    # the recovered mean must match the truth mean, while the hydrostatic
    # solver's does not have to.
    m_true_mean = float(np.nanmean(melt_true[fin]))
    m_inv_mean = float(np.nanmean(inv_b[fin]))
    m_eul_mean = float(np.nanmean(eul_b[fin]))
    rel = abs(m_inv_mean - m_true_mean) / max(abs(m_true_mean), 1e-9)
    rel_eul = abs(m_eul_mean - m_true_mean) / max(abs(m_true_mean), 1e-9)
    # Two-part claim: the level must beat the hydrostatic solver's, and stay
    # within 5%. It is not exact and should not be asserted to be -- a
    # mirror-padded deconvolution on a finite domain is not exactly
    # mean-preserving even with D(0)=1, so a few percent of edge leakage is
    # structural. The comparison against the hydrostatic level is the claim
    # that matters.
    ok = rel < rel_eul and rel < 0.05
    verdict["G2"] = ok
    print(f"  gate[G2] level: recovered mean {m_inv_mean:+.3f} vs truth "
          f"{m_true_mean:+.3f} ({100*rel:.1f}%) beats hydrostatic "
          f"{m_eul_mean:+.3f} ({100*rel_eul:.1f}%)  -> {'PASS' if ok else 'FAIL'}")

    allok = all(verdict.values())
    print(f"\nOVERALL GATE: {'PASS' if allok else 'FAIL'}  ({verdict})")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
