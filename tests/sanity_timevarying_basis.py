"""Synthetic validation of the reduced temporal-basis time-varying inverse.

Design note: ``literature/plan_timevarying_inverse.md`` §3.1, §5. Checks, in
order of increasing difficulty:

  A. adjoint identity  <G c, h> == <c, G† h>            (must hold to ~1e-9)
  B. forward_basis == general forward for a basis-representable m
  C. J=1 basis  ≈  cg_invert_stationary                 (degeneracy anchor)
  D. Gate B — recover a linear melt *trend* on clean, dense data
  E. Gate C-lite — survive 30% coverage gaps + noise (smoke test, not a gate)

Run::
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u tests/sanity_timevarying_basis.py
"""
import numpy as np

import xarray as xr

from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics.pseudospectral import (
    PerturbationForwardOp,
    _cumulative_integral,
    cg_invert_basis,
    cg_invert_stationary,
    inverse_time_varying,
    make_temporal_basis,
)

SECONDS_PER_YEAR = 86400.0 * 365.25
rng = np.random.default_rng(0)


def corr(a, b):
    a = a.ravel(); b = b.ravel()
    return float(np.corrcoef(a, b)[0, 1])


def demean(a):
    return a - a.mean()


# ---- grid + operator -------------------------------------------------------
nx = ny = 64
res = 300.0                      # m
H = 400.0                        # m ice thickness  (channel/feature scale ~ few H)
n_t = 16
t_secs = np.linspace(0.0, 3.0 * SECONDS_PER_YEAR, n_t)   # 3-yr window, ~1.7 t_r

op = PerturbationForwardOp(
    H=H, times=t_secs, nx=nx, ny=ny, dx=res, dy=res,
    eta_bar=1e14, alpha=0.0, gamma=0.0,
)
print(f"grid {ny}x{nx} @ {res:.0f} m  |  n_t={n_t}  |  t_r={op.tr/SECONDS_PER_YEAR:.2f} yr")

x = (np.arange(nx) - nx / 2) * res
y = (np.arange(ny) - ny / 2) * res
X, Y = np.meshgrid(x, y)


def gauss(cx, cy, sig):
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig ** 2))


# ===========================================================================
# A. adjoint identity
# ===========================================================================
print("\n=== A. adjoint identity ===")
phi_t, names = make_temporal_basis(t_secs, "trend")
A = op.build_basis_kernels(phi_t)
J = A.shape[1]
c_rand = rng.standard_normal((J, ny, nx))
h_rand = rng.standard_normal((n_t, ny, nx))
lhs = float(np.sum(op.forward_basis(c_rand, A) * h_rand))
rhs = float(np.sum(c_rand * op.adjoint_basis(h_rand, A)))
rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
print(f"  <Gc,h>={lhs:.6e}  <c,G'h>={rhs:.6e}  rel={rel:.2e}")
assert rel < 1e-8, f"adjoint identity failed: rel={rel}"

# ===========================================================================
# B. forward_basis == general forward for a basis-representable m
# ===========================================================================
print("\n=== B. forward_basis == general forward (basis-representable m) ===")
c0 = demean(2.0 * gauss(-1500, 0, 1600))          # mean-melt pattern (zero spatial mean)
c1 = demean(1.2 * gauss(+1800, 800, 1400))        # trend pattern (distinct location)
m_true = np.stack([c0 * phi_t[i, 0] + c1 * phi_t[i, 1] for i in range(n_t)])  # (n_t,ny,nx)
h_gen = op.forward(m_true)
h_basis = op.forward_basis(np.stack([c0, c1]), A)
d = float(np.max(np.abs(h_gen - h_basis)))
print(f"  max|forward - forward_basis| = {d:.2e} m")
assert d < 1e-9, f"forward_basis disagrees with general forward: {d}"

# ===========================================================================
# C. J=1 basis  ≈  cg_invert_stationary
# ===========================================================================
print("\n=== C. J=1 degeneracy vs cg_invert_stationary ===")
m_stat_true = demean(1.5 * gauss(0, 0, 1500))
h_stat = op.forward_stationary(m_stat_true, reference="anchor_first")
phi_c, _ = make_temporal_basis(t_secs, "const")
c_b, info_b = cg_invert_basis(op, h_stat, phi_c, tikhonov=1e-3, max_iter=300, tol=1e-10)
m_s, info_s = cg_invert_stationary(op, h_stat, tikhonov=1e-3, max_iter=300, tol=1e-10,
                                   reference="anchor_first")
print(f"  basis(J=1) vs stationary: corr={corr(c_b[0], m_s):.6f}  "
      f"iters {info_b['n_iter']}/{info_s['n_iter']}")
print(f"  each vs truth (AC): corr(basis)={corr(demean(c_b[0]), m_stat_true):.4f}  "
      f"corr(stat)={corr(demean(m_s), m_stat_true):.4f}")
assert corr(c_b[0], m_s) > 0.999, "J=1 basis should match cg_invert_stationary"
assert corr(demean(c_b[0]), m_stat_true) > 0.95, "J=1 basis should recover the AC pattern"

# ===========================================================================
# D. Gate B — recover a linear melt trend on clean, dense data
# ===========================================================================
print("\n=== D. Gate B: clean trend recovery (J=2) ===")
c_rec, info = cg_invert_basis(op, h_gen, phi_t, tikhonov=1e-3, max_iter=400, tol=1e-10)
r0 = corr(demean(c_rec[0]), c0)
r1 = corr(demean(c_rec[1]), c1)
print(f"  mode0 (mean-melt) corr={r0:.4f}   mode1 (trend) corr={r1:.4f}   "
      f"iters={info['n_iter']} converged={info['converged']}")
# per-pixel temporal slope recovery
m_rec = np.stack([c_rec[0] * phi_t[i, 0] + c_rec[1] * phi_t[i, 1] for i in range(n_t)])
tc = (t_secs - t_secs.mean())
slope_true = np.tensordot(tc, m_true - m_true.mean(0), axes=(0, 0)) / np.sum(tc ** 2)
slope_rec = np.tensordot(tc, m_rec - m_rec.mean(0), axes=(0, 0)) / np.sum(tc ** 2)
r_slope = corr(demean(slope_rec), demean(slope_true))
print(f"  per-pixel temporal-slope corr={r_slope:.4f}")
assert r0 > 0.9 and r1 > 0.9, f"clean trend recovery too weak: r0={r0}, r1={r1}"
assert r_slope > 0.9, f"temporal-slope recovery too weak: {r_slope}"

# ===========================================================================
# E. Gate C-lite — 30% coverage gaps + noise (smoke test)
# ===========================================================================
print("\n=== E. Coverage + noise robustness (Sobolev regularization engaged) ===")
# Two regime-targeted checks, each the median over 3 gap/noise realizations.
# Noise is SNR-controlled (fraction of the actual per-epoch surface signal) so
# the test is scale-invariant and honest about the regime, rather than pinning
# a magic absolute σ. Per-strip planar tilt / constant offsets are removed
# upstream by tilt_fit (a constant offset is pure-DC the kernel is blind to by
# construction), so what remains is high-freq coreg noise + contiguous gaps.
# Defaults tik=3e-2, L=4H are the noise-controlling values from the design
# note §3.4 sweep (≈0.98/0.90 recovery at SNR~4, no gaps).
hrms = float(np.median([h_gen[i].std() for i in range(1, n_t)]))
TIK, L = 3e-2, 4.0 * H


def _hole(arr, i):
    hh = int(ny * 0.55); ww = int(nx * 0.55)   # one contiguous hole ≈ 30% of tile
    y0 = rng.integers(0, ny - hh); x0 = rng.integers(0, nx - ww)
    arr[i, y0:y0 + hh, x0:x0 + ww] = np.nan


# E1 — the masked-CG coverage payoff (design note G1): 28% contiguous holes,
# noiseless → near-exact recovery. This is what passing NaN (not fillna(0)) to
# W buys; it is the single most important correctness property.
r0g, r1g = [], []
for _ in range(3):
    hg = h_gen.copy()
    for i in range(1, n_t):
        _hole(hg, i)
    cg, _ = cg_invert_basis(op, hg, phi_t, tikhonov=TIK, length_scale_m=L,
                            max_iter=300, tol=1e-8)
    r0g.append(corr(demean(cg[0]), c0)); r1g.append(corr(demean(cg[1]), c1))
print(f"  E1 gaps(28%), noiseless : mode0={np.median(r0g):.3f}  mode1={np.median(r1g):.3f}")
assert np.median(r0g) > 0.9 and np.median(r1g) > 0.85, "masked-CG must recover under gaps"

# E2 — realistic SNR: coreg noise ≈ 0.25×(signal rms) (SNR~4) + 28% gaps.
sig = 0.25 * hrms
r0n, r1n = [], []
for _ in range(3):
    hn = h_gen.copy()
    for i in range(1, n_t):
        hn[i] += rng.normal(0, sig, (ny, nx))
        _hole(hn, i)
    cn, _ = cg_invert_basis(op, hn, phi_t, tikhonov=TIK, length_scale_m=L,
                            max_iter=300, tol=1e-8)
    r0n.append(corr(demean(cn[0]), c0)); r1n.append(corr(demean(cn[1]), c1))
print(f"  E2 gaps + noise (SNR~{hrms/sig:.0f}): mode0={np.median(r0n):.3f}  "
      f"mode1={np.median(r1n):.3f}")
assert np.median(r0n) > 0.6 and np.median(r1n) > 0.45, \
    "trend must survive realistic SNR + gaps"

# ===========================================================================
# F. public xarray entry point — inverse_time_varying
# ===========================================================================
print("\n=== F. public inverse_time_varying (xarray) round-trip ===")
dt_sec = float(t_secs[1] - t_secs[0])
h_da = xr.DataArray(
    h_gen, dims=("time", "y", "x"),
    coords={"time": np.arange(n_t) * dt_sec, "y": y, "x": x},
)
melt = inverse_time_varying(h_da, H=H, temporal_basis="trend",
                            reg=1e-3, length_scale_m=0.0, max_iter=400, tol=1e-10)
assert melt.dims == ("time", "y", "x") and melt.shape == (n_t, ny, nx)
assert "Shean" in melt.attrs["units"] and melt.attrs["temporal_basis"] == "trend"
# Shean sign: recovered melt_rate ≈ -m_true (m_true is internal positive=melt).
cc = corr(demean(-melt.values), demean(m_true))
print(f"  dims={melt.dims}  basis={melt.attrs['temporal_basis']}  "
      f"cg_iter={melt.attrs['cg_iter']}  corr(-melt, m_true)={cc:.4f}")
assert cc > 0.95, f"public wrapper round-trip weak: {cc}"

# ===========================================================================
# G. DC recovery — the spatial-mean melt time series via mass balance
# ===========================================================================
print("\n=== G. per-mode DC splice recovers time-varying spatial-mean melt ===")
# Truth: nonzero-mean coefficient fields (internal positive=melt), so the
# spatial-mean melt m̄(t) = M0 + M1·φ1(t) is a real time-varying signal. The
# perturbation forward zeros k=0, so the DC signal must enter h the way it does
# in real data — via the hydrostatic mass balance dh̄/dt = -m̄/R (ȧ=0).
M0, M1 = 6.0, 3.0                                   # mean melt + its trend (m/yr)
R_hydro = rhow / (rhow - rhoi)
h_ac = op.forward_basis(np.stack([c0, c1]), A)      # AC surface response
Phi1 = _cumulative_integral(phi_t[:, 1], t_secs)    # ∫φ1 dτ  (s)
mbar_int = (M0 / SECONDS_PER_YEAR) * t_secs + (M1 / SECONDS_PER_YEAR) * Phi1  # ∫m̄ (m)
h_dc = -(1.0 / R_hydro) * mbar_int                  # uniform surface drop (m)
h_synth = h_ac + h_dc[:, None, None]
h_da_dc = xr.DataArray(
    h_synth, dims=("time", "y", "x"),
    coords={"time": np.arange(n_t) * dt_sec, "y": y, "x": x},
)
melt_dc = inverse_time_varying(h_da_dc, H=H, temporal_basis="trend", reg=1e-3,
                               length_scale_m=0.0, recover_dc=True, a_dot_dc=0.0,
                               max_iter=400, tol=1e-10)
assert melt_dc.attrs["dc_recovered"] == 1
mbar_true = M0 + M1 * phi_t[:, 1]                    # internal positive=melt
mbar_rec = -melt_dc.mean(dim=("y", "x")).values      # -(Shean) → internal
cc = corr(mbar_rec, mbar_true)
rel = float(np.max(np.abs(mbar_rec - mbar_true)) / np.max(np.abs(mbar_true)))
print(f"  m̄(t) truth range [{mbar_true.min():.2f},{mbar_true.max():.2f}] m/yr  "
      f"corr={cc:.4f}  max-rel-err={rel:.3f}")
assert cc > 0.99 and rel < 0.05, f"DC splice failed: corr={cc}, rel={rel}"
# without recover_dc the spatial mean should be ~0 (DC-blind), confirming the
# splice is what supplies the mean.
melt_nodc = inverse_time_varying(h_da_dc, H=H, temporal_basis="trend", reg=1e-3,
                                 length_scale_m=0.0, recover_dc=False,
                                 max_iter=400, tol=1e-10)
mbar_nodc = float(np.abs(-melt_nodc.mean(dim=("y", "x")).values).max())
print(f"  recover_dc=False → |m̄|max={mbar_nodc:.3f} m/yr (DC-blind, as expected)")
assert mbar_nodc < 0.5 * abs(M0), "without splice the mean should stay ~0"

print("\nPASS: reduced-basis time-varying inverse — adjoint exact (1e-15), forward "
      "consistent, J=1 ≡ stationary, clean trend recovered exactly, robust to "
      "coverage gaps, trend recovered at realistic SNR, public API round-trips, "
      "DC splice recovers the time-varying spatial-mean melt.")
