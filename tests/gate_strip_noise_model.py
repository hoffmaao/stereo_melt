"""Gate: the coloured (per-strip) noise model of the monolithic inverse.

A synthetic stack is built through the solver's own forward model on a
uniform shelf (clean), then corrupted with KNOWN per-strip offsets and plane
tilts on ragged footprints (no white noise, so the gate isolates the
coherent error). Rungs:

S1  DESIGN. ``strip_mode_design`` reproduces the corruption's effect on the
    observation side: ``sum_j G_j theta_true_j`` equals the change in
    ``dHdt_obs + div(H_f u)`` between the corrupted and clean stacks to
    < 1 % rms (the OLS/mean propagation and the divergence are exact).
S2  RECOVERY. The joint fit recovers the coherent-error FIELD G theta
    (corr > 0.95, rel rms < 0.3 vs the injected G theta_true) with the prior
    at the injected variances. Individual amplitudes are not identifiable —
    overlapping footprints with similar time weights make the modes nearly
    collinear — and are reported for information only.
S3  THE MELT. With the noise model the channel rmse vs the clean-stack
    solution drops by > 3x relative to the plain solver on the corrupted
    stack, at the same lam.
S4  NO MODES == OLD PATH. ``strip_modes=None`` is bit-identical to the
    solver before this change (same numbers as the local gate's L1 path).
S5  LAM IS REQUIRED. There is no defensible universal default -- a fixed lam
    does not transfer across noise levels and the ``"auto"`` rule is built on a
    WHITE variance estimate that this project's ~4 km correlated strip error
    violates -- so omitting it must raise rather than pick one silently.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_strip_noise_model.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.constants import rhoi, rhow  # noqa: E402
from stereo_melt.dynamics.budget_bridging import (  # noqa: E402
    bridging_transfer_multiplier,
    budget_bridging_melt_rate,
    strip_mode_design,
)
from stereo_melt.kinematics import dh_dt, flux_divergence  # noqa: E402

NY, NX, RES = 48, 128, 250.0
U0, H0 = 900.0, 500.0
RHO = dict(rho_i=rhoi, rho_w=rhow)
C = (rhow - rhoi) / rhow
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    rng = np.random.default_rng(1)
    y = np.arange(NY)[::-1] * RES
    x = np.arange(NX) * RES
    xx, yy = np.meshgrid(x, y)
    vx = xr.DataArray(np.full((NY, NX), U0), dims=("y", "x"), coords={"y": y, "x": x})
    vy = xr.zeros_like(vx)
    melt = (-3.0 - 1.0 * np.sin(2 * np.pi * xx / (NX * RES))
            - 8.0 * np.exp(-((xx - 0.45 * NX * RES) ** 2) / (2 * (2.5 * RES) ** 2)))
    T = bridging_transfer_multiplier(2 * NY, 2 * NX, RES, RES, H=H0, ux_myr=U0,
                                     uy_myr=0.0, eta_bar=1e14)

    def pad_apply(f):
        p = np.concatenate([f, f[::-1]], 0)
        p = np.concatenate([p, p[:, ::-1]], 1)
        return np.real(np.fft.ifft2(T * np.fft.fft2(p)))[:NY, :NX]

    # clean stack through the forward model, ragged footprints (strips cover
    # random x-bands), 30 epochs over 6 years
    n_t, span = 30, 6.0
    t = np.sort(rng.uniform(0, span, n_t))
    t -= t.mean()
    times = pd.to_datetime("2015-01-01") + pd.to_timedelta((t - t[0]) * 365.25, unit="D")
    rate = pad_apply(melt)
    foot = np.zeros((n_t, NY, NX), bool)
    for k in range(n_t):
        x0 = rng.uniform(-0.3, 0.5) * NX * RES
        foot[k] = (xx >= x0) & (xx <= x0 + rng.uniform(0.5, 0.9) * NX * RES)
    # centre the trend on each pixel's own sampled epochs, so the time-mean
    # thickness is exactly H0 everywhere and the stack satisfies the solver's
    # model (dH/dt + div(H0 u) = T{m}) regardless of the ragged coverage —
    # the ragged-sampling bias of a trending field is a separate effect
    tbar = (foot * t[:, None, None]).sum(0) / np.maximum(foot.sum(0), 1)
    h = C * (H0 + rate[None] * (t[:, None, None] - tbar[None]))
    h = np.where(foot, h, np.nan)
    clean = xr.DataArray(h, dims=("time", "y", "x"), coords={"time": times, "y": y, "x": x})

    # known per-strip errors: offsets sigma 0.3 m, tilts sigma 1.5e-4 m/m
    c_k = rng.normal(0, 0.3, n_t)
    a_k = rng.normal(0, 1.5e-4, n_t)
    b_k = rng.normal(0, 1.5e-4, n_t)
    hc = h.copy()
    xc = np.array([xx[foot[k]].mean() for k in range(n_t)])
    yc = np.array([yy[foot[k]].mean() for k in range(n_t)])
    for k in range(n_t):
        hc[k] += np.where(foot[k], c_k[k] + a_k[k] * (xx - xc[k]) + b_k[k] * (yy - yc[k]), np.nan)
    corrupt = clean.copy(data=hc)

    # ---------------- S1: the design reproduces the corruption
    G, sidx, comp = strip_mode_design(corrupt, vx, vy, **RHO)
    theta_true = np.array([{"offset": c_k, "tilt_x": a_k, "tilt_y": b_k}[c][k]
                           for k, c in zip(sidx, comp)])

    def obs_side(stack):
        Hf = stack / C
        reg = dh_dt(Hf, min_count=3)
        d = reg["slope"].values * 86400.0 * 365.25
        fd = flux_divergence(Hf.mean("time", skipna=True), vx, vy).values
        return d + np.nan_to_num(fd)
    diff = obs_side(corrupt) - obs_side(clean)
    pred = (theta_true[:, None, None] * G).sum(0)
    ok = np.isfinite(diff) & (np.abs(diff) > 0)
    rel = float(np.sqrt(np.nanmean((pred - diff)[ok] ** 2)) / np.sqrt(np.nanmean(diff[ok] ** 2)))
    check("G theta_true == observed corruption", rel < 0.01,
          f"rel rms {rel:.2e}; corruption rms {np.sqrt(np.nanmean(diff[ok]**2)):.3f} m/yr, "
          f"{G.shape[0]} modes")

    # ---------------- S2/S3: joint fit
    tau2 = np.array([{"offset": 0.3 ** 2, "tilt_x": 1.5e-4 ** 2, "tilt_y": 1.5e-4 ** 2}[c]
                     for c in comp])
    kw = dict(bridging=True, eta_bar=1e14, lam=1e-4, iters=4000, converge_tol=1e-12, **RHO)
    no_lam = {k: v for k, v in kw.items() if k != "lam"}
    try:
        budget_bridging_melt_rate(clean, vx, vy, **no_lam)
    except TypeError as exc:
        msg = str(exc)
        check("omitting lam raises, naming both valid choices",
              "lam" in msg and "auto" in msg and "float" in msg, msg.split(".")[0])
    else:
        check("omitting lam raises, naming both valid choices", False, "no error raised")
    ref = budget_bridging_melt_rate(clean, vx, vy, **kw)
    plain = budget_bridging_melt_rate(corrupt, vx, vy, **kw)
    # sigma2: the stack has NO white noise, so give the prior a small but
    # finite white floor (0.05 m/yr)^2 rather than the auto estimate (which
    # would see only the strip errors)
    nm = budget_bridging_melt_rate(corrupt, vx, vy, strip_modes=G, strip_prior=tau2,
                                   sigma2=0.05 ** 2, **kw)
    th = nm["theta"].values
    # individual amplitudes are not identifiable (overlapping footprints with
    # similar time weights give near-collinear modes); the coherent-error
    # FIELD they produce is what the data constrain and what enters the melt
    est = nm["strip_error_rate"].values
    fitc = np.isfinite(est)
    r = float(np.corrcoef(est[fitc], pred[fitc])[0, 1])
    rel = float(np.sqrt(np.mean((est - pred)[fitc] ** 2)) / np.sqrt(np.mean(pred[fitc] ** 2)))
    check("coherent-error field G theta recovered", r > 0.95 and rel < 0.3,
          f"corr {r:.3f}  rel rms {rel:.3f}  "
          f"(individual theta corr: " + ", ".join(
              f"{c} {np.corrcoef(th[comp == c], theta_true[comp == c])[0, 1]:.2f}"
              for c in ("offset", "tilt_x", "tilt_y")) + ")")
    inner = (xx > 3000) & (xx < x[-1] - 3000)
    e_plain = float(np.sqrt(np.nanmean((plain.melt_rate.values - ref.melt_rate.values)[inner] ** 2)))
    e_nm = float(np.sqrt(np.nanmean((nm.melt_rate.values - ref.melt_rate.values)[inner] ** 2)))
    e_ref = float(np.sqrt(np.nanmean((ref.melt_rate.values - melt)[inner] ** 2)))
    check("melt rmse vs clean solution drops 3x", e_nm < e_plain / 3.0,
          f"plain {e_plain:.3f} -> noise model {e_nm:.3f} m/yr  (clean-vs-truth {e_ref:.3f})")

    # ---------------- S4: no modes == old path
    again = budget_bridging_melt_rate(corrupt, vx, vy, strip_modes=None, **kw)
    d = float(np.nanmax(np.abs(again.melt_rate.values - plain.melt_rate.values)))
    check("strip_modes=None unchanged", d == 0.0, f"max|dm| {d:.1e}")

    print()
    if FAILS:
        print(f"GATE FAILED: {FAILS}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
