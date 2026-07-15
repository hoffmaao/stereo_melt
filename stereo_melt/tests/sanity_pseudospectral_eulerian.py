"""Sanity test for the time-dependent pseudo-spectral Eulerian inverse.

Three checks:

1. **2-D advection adjoint** --- ``<G m, h> = <m, G^T h>`` with non-zero
   ``α_x`` *and* ``α_y`` to verify the 2-D-α extension to the kernel.
2. **Synthetic round-trip** --- prescribe ``m_true(t, y, x)`` with
   2-D advection in the background, recover via CG.
3. **End-to-end Eulerian wrapper** --- prescribe stationary
   ``m_true``, build a forward stack with uniform 2-D flow, pass to
   :func:`pseudospectral_eulerian_inverse`, recover.
"""

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics import (
    PerturbationForwardOp,
    cg_invert,
    forward,
    pseudospectral_eulerian_inverse,
)

print("Pseudo-spectral Eulerian module imports OK.\n")

# Setup
H = 500.0
eta_bar = 1e14
delta = rhow / rhoi - 1.0
tr_s = 2.0 * eta_bar / (rhoi * 9.81 * H)
SECONDS_PER_YEAR = 86400.0 * 365.25
tr_yr = tr_s / SECONDS_PER_YEAR
print(f"H={H} m  t_r={tr_yr:.2f} yr")

L = 30_000.0
res = 500.0
x = np.arange(-L / 2, L / 2, res)
y = np.arange(L / 2, -L / 2, -res)
nx = len(x)
ny = len(y)

n_t = 6
times_s = np.linspace(0.0, tr_s, n_t)
times_dt = pd.to_datetime("2020-01-01") + pd.to_timedelta(times_s, unit="s")


# ----------------------------------------------------------------------
# Test 1: adjoint correctness with non-trivial 2-D advection
# ----------------------------------------------------------------------
op_2d = PerturbationForwardOp(
    H=H, times=times_dt, nx=nx, ny=ny, dx=res, dy=res,
    eta_bar=eta_bar, alpha=0.3, alpha_y=0.4, gamma=0.05,
)
rng = np.random.default_rng(42)
m_test = rng.standard_normal((n_t, ny, nx))
h_test = rng.standard_normal((n_t, ny, nx))
Gm = op_2d.forward(m_test)
GTh = op_2d.adjoint(h_test)
lhs = float(np.sum(Gm * h_test))
rhs = float(np.sum(m_test * GTh))
rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
print(f"[adjoint, 2-D α] <Gm,h>={lhs:.6e}  <m,G^T h>={rhs:.6e}  rel_err={rel_err:.2e}")
assert rel_err < 1e-9, f"2-D adjoint identity failed: rel_err={rel_err:.2e}"
print("  PASS: 2-D adjoint identity holds.\n")


# ----------------------------------------------------------------------
# Test 2: synthetic round-trip (time-varying m, 2-D advection)
# ----------------------------------------------------------------------
sigma = (10.0 / 3.0) * H
m0 = 0.014 * H / tr_s * SECONDS_PER_YEAR
X, Y = np.meshgrid(x, y)
spatial = np.exp(-0.5 * (X**2 + Y**2) / sigma**2)
amp_t = np.linspace(0.5, 1.5, n_t)
m_true = m0 * amp_t[:, None, None] * spatial[None, :, :]

# Same operator, different random m.
h_synth = op_2d.forward(m_true)
m_rec, info = cg_invert(op_2d, h_synth, tikhonov=1e-3, max_iter=120, tol=1e-8)
print(f"[round-trip, 2-D α] CG: {info['n_iter']} iters  "
      f"‖r‖/‖r_0‖={info['residual_norm']/max(info['residual_norm_initial'],1e-30):.2e}  "
      f"converged={info['converged']}")

box = (np.abs(X) < L / 4) & (np.abs(Y) < L / 4)
errs = []
for ti in range(n_t - 1):  # skip last (K_h(0)≡0 unobservable)
    diff = m_rec[ti][box] - m_true[ti][box]
    rmse = float(np.sqrt(np.mean(diff**2)))
    rel = rmse / m0
    errs.append(rel)
    print(f"  t={ti}: peak_rec/peak_true="
          f"{m_rec[ti][box].max()/m_true[ti][box].max():.3f}  "
          f"RMSE={rmse:.3f} m/yr ({100*rel:.1f}% of peak)")
print(f"  t={n_t-1} skipped (structurally unobservable)")
max_rel = max(errs)
assert max_rel < 0.10, f"round-trip RMSE too high: {max_rel:.3f}"
print(f"  PASS: round-trip recovery within 10% (max rel rmse {100*max_rel:.1f}%).\n")


# ----------------------------------------------------------------------
# Test 3: end-to-end via pseudospectral_eulerian_inverse
# ----------------------------------------------------------------------
# Stationary m_true, uniform 2-D flow. The Eulerian wrapper should
# recover the field directly without a frame transformation.
m_da = xr.DataArray(
    spatial * m0,
    dims=("y", "x"),
    coords={"y": y, "x": x},
)
u_x = (sigma / 8.0) / tr_yr  # m/yr
u_y = (sigma / 12.0) / tr_yr
alpha_x_fwd = u_x * (1.0 / SECONDS_PER_YEAR) * tr_s / H
alpha_y_fwd = u_y * (1.0 / SECONDS_PER_YEAR) * tr_s / H
print(f"[end-to-end] u_x={u_x:.1f} m/yr  u_y={u_y:.1f} m/yr  "
      f"(α_x, α_y)=({alpha_x_fwd:.2f}, {alpha_y_fwd:.2f})")

h_eul = forward(
    m_da, H=H, eta_bar=eta_bar, alpha=alpha_x_fwd, alpha_y=alpha_y_fwd, gamma=0.0,
    stationary=True, times=xr.DataArray(times_s, dims="t"),
)
h_stack = xr.DataArray(
    h_eul.values, dims=("time", "y", "x"),
    coords={"time": times_dt, "y": y, "x": x},
)
vx = xr.DataArray(np.full_like(X, u_x, dtype=np.float64),
                  dims=("y", "x"), coords={"y": y, "x": x})
vy = xr.DataArray(np.full_like(X, u_y, dtype=np.float64),
                  dims=("y", "x"), coords={"y": y, "x": x})

ds = pseudospectral_eulerian_inverse(
    h_stack, vx, vy, H_ref=H, eta_bar=eta_bar, gamma=0.0,
    tikhonov=1e-3, max_iter=200, cg_tol=1e-8,
)
print(f"  CG: {ds.attrs['cg_iter']} iters, converged={ds.attrs['cg_converged']}, "
      f"derived (α_x, α_y)=({ds.attrs['alpha_x']:.3f}, {ds.attrs['alpha_y']:.3f})")
mr = ds.melt_rate.values  # Shean convention (wrapper negates Stubblefield m)
mid_t = n_t // 2
# Prescribed m is positive Gaussian (Stubblefield input to forward()); under
# the Shean public convention the recovered field is its negative, so the
# "peak" is m_rec.min() and the ground-truth peak is -peak_true.
peak_t = mr[mid_t][box].min()
peak_true_shean = -(spatial * m0)[box].max()
ratio = float(peak_t / peak_true_shean)
print(f"  mid-epoch (t={mid_t}) peak ratio = {ratio:.3f}")
assert 0.7 < ratio < 1.3, f"end-to-end peak ratio {ratio:.3f} outside [0.7, 1.3]"
print("  PASS: end-to-end Eulerian inverse recovers stationary m with 2-D advection.")

print("\nAll pseudo-spectral Eulerian sanity checks passed.")
