"""Sanity test for the time-dependent pseudo-spectral Lagrangian inverse.

Three checks:

1. **Adjoint correctness** --- ``<G m, h> = <m, G^T h>`` to ~1e-10 on a
   small grid with random inputs.
2. **Synthetic round-trip** --- prescribe ``m_true(t, y, x)`` with a
   real time-variation, forward to get ``h(t, y, x)``, run CG inverse,
   recover ``m_true`` within Tikhonov tolerance.
3. **End-to-end Lagrangian wrapper** --- prescribe ``m_true``, build a
   stationary-Eulerian forward stack with uniform advection, pass to
   :func:`pseudospectral_lagrangian_inverse`, recover.
"""

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics import (
    PerturbationForwardOp,
    cg_invert,
    forward,
    pseudospectral_lagrangian_inverse,
)

print("Pseudo-spectral Lagrangian module imports OK.\n")

# ----------------------------------------------------------------------
# Setup: small grid, short time series, paper-style parameters
# ----------------------------------------------------------------------
H = 500.0
eta_bar = 1e14
delta = rhow / rhoi - 1.0
tr_s = 2.0 * eta_bar / (rhoi * 9.81 * H)
SECONDS_PER_YEAR = 86400.0 * 365.25
tr_yr = tr_s / SECONDS_PER_YEAR
print(f"H={H} m  t_r={tr_yr:.2f} yr  delta={delta:.4f}")

L = 30_000.0
res = 500.0
x = np.arange(-L / 2, L / 2, res)
y = np.arange(L / 2, -L / 2, -res)
nx = len(x)
ny = len(y)
print(f"grid: ny={ny}  nx={nx}  res={res} m")


# ----------------------------------------------------------------------
# Test 1: adjoint correctness
# ----------------------------------------------------------------------
n_t = 6
times_s = np.linspace(0.0, tr_s, n_t)
times_dt = pd.to_datetime("2020-01-01") + pd.to_timedelta(times_s, unit="s")

op = PerturbationForwardOp(
    H=H, times=times_dt, nx=nx, ny=ny, dx=res, dy=res,
    eta_bar=eta_bar, alpha=0.0, gamma=0.0,
)

rng = np.random.default_rng(42)
m_test = rng.standard_normal((n_t, ny, nx))
h_test = rng.standard_normal((n_t, ny, nx))

Gm = op.forward(m_test)
GTh = op.adjoint(h_test)

lhs = float(np.sum(Gm * h_test))
rhs = float(np.sum(m_test * GTh))
rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
print(f"\n[adjoint] <Gm,h>={lhs:.6e}  <m,G^T h>={rhs:.6e}  rel_err={rel_err:.2e}")
assert rel_err < 1e-9, f"adjoint identity failed: rel_err={rel_err:.2e}"
print("  PASS: adjoint identity holds.")


# ----------------------------------------------------------------------
# Test 2: synthetic round-trip with time-varying m
# ----------------------------------------------------------------------
# m_true(t, x, y): a Gaussian in space whose amplitude grows linearly
# with time. The forward integrates this; CG should recover it.
sigma = (10.0 / 3.0) * H
m0 = 0.014 * H / tr_s * SECONDS_PER_YEAR  # ~5 m/yr at t_r
X, Y = np.meshgrid(x, y)
spatial = np.exp(-0.5 * (X**2 + Y**2) / sigma**2)
amp_t = np.linspace(0.5, 1.5, n_t)  # growing in time
m_true = m0 * amp_t[:, None, None] * spatial[None, :, :]
print(f"\n[round-trip] m_true peak={float(m_true.max()):.2f} m/yr; "
      f"amplitude varies {amp_t[0]:.1f}->{amp_t[-1]:.1f} over t_r")

h_synth = op.forward(m_true)
m_rec, info = cg_invert(op, h_synth, tikhonov=1e-3, max_iter=80, tol=1e-8)
print(f"  CG: {info['n_iter']} iters, ‖r‖/‖r_0‖="
      f"{info['residual_norm'] / max(info['residual_norm_initial'], 1e-30):.2e}, "
      f"converged={info['converged']}")

# Compare on the central window (FFT periodicity makes edges noisy).
# Note: the LAST epoch m(t_{N-1}) is structurally unobservable --- the
# time-causal forward kernel K_h(Δt=0) = -δB/μ · (e^0 - e^0) = 0
# identically, so m at the final observation time never enters h. Skip
# it in the round-trip check.
box = (np.abs(X) < L / 4) & (np.abs(Y) < L / 4)
errs = []
for ti in range(n_t - 1):
    diff = m_rec[ti][box] - m_true[ti][box]
    rmse = float(np.sqrt(np.mean(diff**2)))
    rel = rmse / m0
    errs.append(rel)
    print(f"  t={ti}: peak_recovered/peak_true="
          f"{m_rec[ti][box].max()/m_true[ti][box].max():.3f}  "
          f"central RMSE={rmse:.3f} m/yr ({100*rel:.1f}% of peak)")
print(f"  t={n_t - 1} skipped (structurally unobservable: K_h(0) ≡ 0)")
max_rel = max(errs)
assert max_rel < 0.10, f"round-trip RMSE too high: {max_rel:.3f}"
print(f"  PASS: round-trip recovery within 10% (max rel rmse {100*max_rel:.1f}%).")


# ----------------------------------------------------------------------
# Test 3: end-to-end via pseudospectral_lagrangian_inverse
# ----------------------------------------------------------------------
# Build a stationary-m Eulerian forward stack with uniform advection.
# The Lagrangian wrapper should advect h to a comoving frame and recover m.
m_da = xr.DataArray(
    spatial * m0,
    dims=("y", "x"),
    coords={"y": y, "x": x},
)
u_x = (sigma / 8.0) / tr_yr  # ~ slow uniform along-x flow
alpha_fwd = u_x * (1.0 / SECONDS_PER_YEAR) * tr_s / H
print(f"\n[end-to-end] uniform flow u_x={u_x:.1f} m/yr, alpha_fwd={alpha_fwd:.3f}")

# Use forward() with uniform alpha to get the Eulerian h_stack.
h_eulerian = forward(
    m_da, H=H, eta_bar=eta_bar, alpha=alpha_fwd, gamma=0.0,
    stationary=True, times=xr.DataArray(times_s, dims="t"),
)
h_stack = xr.DataArray(
    h_eulerian.values,
    dims=("time", "y", "x"),
    coords={"time": times_dt, "y": y, "x": x},
)
vx = xr.DataArray(np.full_like(X, u_x, dtype=np.float64), dims=("y", "x"), coords={"y": y, "x": x})
vy = xr.zeros_like(vx)

ds = pseudospectral_lagrangian_inverse(
    h_stack, vx, vy, H_ref=H, eta_bar=eta_bar, gamma=0.0,
    tikhonov=1e-3, max_iter=200, cg_tol=1e-8, dt_yr=0.05,
)
print(f"  CG: {ds.attrs['cg_iter']} iters, "
      f"converged={ds.attrs['cg_converged']}")
mr = ds.melt_rate.values  # Shean convention (wrapper negates Stubblefield m)
# Use a middle epoch (NOT the last --- structurally unobservable). For
# stationary m_true the inverse should give the same field at every
# constrained epoch; pick the median. Under the Shean public convention
# (negative = melt) the recovered Gaussian is the negative of the
# prescribed positive m_true, so the "peak" is m_rec.min().
mid_t = n_t // 2
peak_t = mr[mid_t][box].min()
peak_true_shean = -(spatial * m0)[box].max()
ratio = float(peak_t / peak_true_shean)
print(f"  mid-epoch (t={mid_t}) recovery: peak ratio = {ratio:.3f} (target ~1)")
assert 0.7 < ratio < 1.3, f"end-to-end peak ratio {ratio:.3f} outside [0.7, 1.3]"
print("  PASS: end-to-end Lagrangian inverse recovers stationary m.")

print("\nAll pseudo-spectral Lagrangian sanity checks passed.")
