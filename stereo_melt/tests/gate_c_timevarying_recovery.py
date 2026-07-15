"""Gate C — the failable decision test for the time-varying Stubblefield inverse.

Design note ``literature/plan_timevarying_inverse.md`` §5. The A–G sanity checks
prove correctness on clean / mildly-degraded data; Gate C asks the *scientific*
question on data that resembles a real REMA stack — irregular sparse epochs,
contiguous coverage gaps, coreg noise, and **per-strip planar tilt residual**
(what survives ``tilt_fit``): does the inverse recover time-varying melt
*structure* (spatial pattern + trend + seasonal), and how does each corruption
axis degrade it?

The verdict is the PRINTED ladder (a finding, not an invariant). One hard assert
only: the clean floor must recover, validating the harness itself.

Metrics (all smoothed spatial correlations vs the known truth, 2 km):
  spatial : time-mean melt field  (recoverable by a stationary inverse too)
  trend   : per-pixel temporal slope  (stationary inverse: 0 DOF, cannot)
  season  : per-pixel seasonal amplitude  (stationary inverse: 0 DOF, cannot)
  mbar    : basin-mean melt-rate time series m̄(t)  (the DC / mass-balance mode)
  base    : stationary inverse's spatial corr  (the "recoverable-band" floor)

Run::
    cd stereo_melt && /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u tests/gate_c_timevarying_recovery.py
"""
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics import inverse_stationary
from stereo_melt.dynamics.pseudospectral import (
    PerturbationForwardOp,
    _cumulative_integral,
    inverse_time_varying,
    make_temporal_basis,
)

SECONDS_PER_YEAR = 86400.0 * 365.25
rng = np.random.default_rng(7)


def nan_gauss(a, s):
    m = np.isfinite(a)
    return gaussian_filter(np.where(m, a, 0.0), s) / np.maximum(
        gaussian_filter(m.astype(float), s), 1e-6)


def scorr(a, b, s):
    a2, b2 = nan_gauss(a, s), nan_gauss(b, s)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a2[m].ravel(), b2[m].ravel())[0, 1])


def slope(stack, t):
    tc = t - t.mean()
    return np.tensordot(tc, stack - stack.mean(0), axes=(0, 0)) / np.sum(tc ** 2)


def seas_amp(stack, phi):
    cs = np.tensordot(phi[:, 2], stack, axes=(0, 0))
    cc = np.tensordot(phi[:, 3], stack, axes=(0, 0))
    return np.sqrt(cs ** 2 + cc ** 2)


# --- grid, irregular sampling, transient truth --------------------------------
nx = ny = 64
res = 300.0
H = 450.0
n_t = 18
R_hydro = rhow / (rhow - rhoi)
sig_sm = 2000.0 / res

t_secs = np.sort(rng.uniform(0, 5 * SECONDS_PER_YEAR, n_t))
t_secs = t_secs - t_secs[0]
x = (np.arange(nx) - nx / 2) * res
y = (np.arange(ny) - ny / 2) * res
X, Y = np.meshgrid(x, y)
Xn, Yn = X / X.max(), Y / Y.max()


def g(cx, cy, sx, sy):
    return np.exp(-((X - cx) ** 2 / (2 * sx ** 2) + (Y - cy) ** 2 / (2 * sy ** 2)))


phi, names = make_temporal_basis(t_secs, "trend+annual")
# distinct spatial patterns per temporal mode (internal convention: += melt)
c_mean = 12.0 * g(0, 0, 2.0 * H, 8000)      # cross-channel melt ridge (~2H wide)
c_trend = 5.0 * g(2500, -1500, 4000, 4000)  # basin-scale thinning trend
c_sin = 3.0 * g(-3000, 2000, 2500, 2500)    # seasonal lobe
c_cos = 2.0 * g(1000, 3000, 2200, 2200)
coeff_true = [c_mean, c_trend, c_sin, c_cos]
m_true = sum(coeff_true[j][None] * phi[:, j][:, None, None] for j in range(4))

op = PerturbationForwardOp(H=H, times=t_secs, nx=nx, ny=ny, dx=res, dy=res, alpha=0.0)
h_ac = op.forward(m_true)                                     # AC (kernel zeros k=0)
mbar_t = sum(float(coeff_true[j].mean()) * phi[:, j] for j in range(4))  # m/yr
h_dc = -(1.0 / R_hydro) * _cumulative_integral(mbar_t / SECONDS_PER_YEAR, t_secs)
h_clean = h_ac + h_dc[:, None, None]
slope_true = slope(m_true, t_secs)
seas_true = seas_amp(m_true, phi)
mean_true = m_true.mean(0)

print(f"Gate C: {ny}x{nx}@{res:.0f}m  n_t={n_t} irregular / {t_secs[-1]/SECONDS_PER_YEAR:.1f}yr  "
      f"t_r={op.tr/SECONDS_PER_YEAR:.2f}yr  cond(phi)={np.linalg.cond(phi):.1f}")
print(f"  h rms={np.median([h_clean[i].std() for i in range(1,n_t)]):.2f} m  "
      f"channel melt≈{c_mean.max():.0f} m/yr  m̄(t)∈[{mbar_t.min():.1f},{mbar_t.max():.1f}] m/yr")


def corrupt(noise, gapfrac, tilt, seed):
    r = np.random.default_rng(seed)
    h = h_clean.copy()
    for i in range(1, n_t):
        if noise:
            h[i] = h[i] + r.normal(0, noise, (ny, nx))
        if tilt:
            h[i] = h[i] + r.normal(0, tilt) * Xn + r.normal(0, tilt) * Yn
        if gapfrac:
            gw = int(nx * gapfrac ** 0.5)
            y0 = r.integers(0, ny - gw); x0 = r.integers(0, nx - gw)
            h[i, y0:y0 + gw, x0:x0 + gw] = np.nan
    return h


def evaluate(h):
    h_da = xr.DataArray(h, dims=("time", "y", "x"),
                        coords={"time": t_secs, "y": y, "x": x})
    melt = inverse_time_varying(h_da, H=H, temporal_basis="trend+annual",
                                reg=3e-2, length_scale_m=2 * H, recover_dc=True,
                                max_iter=300, tol=1e-8)
    m_rec = -melt.values
    # stationary baseline: mean-fill gaps (the naive closed-form needs dense data)
    hf = h.copy()
    for i in range(n_t):
        fin = np.isfinite(hf[i])
        if fin.any() and (~fin).any():
            hf[i][~fin] = hf[i][fin].mean()
    base = inverse_stationary(
        xr.DataArray(hf, dims=("time", "y", "x"),
                     coords={"time": t_secs, "y": y, "x": x}),
        H=H, reg=3e-2, reference="first_epoch", recover_dc=False)
    m_base = -base.values
    return (
        scorr(m_rec.mean(0), mean_true, sig_sm),
        scorr(slope(m_rec, t_secs), slope_true, sig_sm),
        scorr(seas_amp(m_rec, phi), seas_true, sig_sm),
        float(np.corrcoef(m_rec.reshape(n_t, -1).mean(1), mbar_t)[0, 1]),
        scorr(m_base, mean_true, sig_sm),
    )


SCEN = [
    ("S0 clean",            dict(noise=0.0,  gapfrac=0.0,  tilt=0.0)),
    ("S1 +noise 0.3m",      dict(noise=0.3,  gapfrac=0.0,  tilt=0.0)),
    ("S2 +gaps 25%",        dict(noise=0.0,  gapfrac=0.25, tilt=0.0)),
    ("S3 noise+gaps",       dict(noise=0.3,  gapfrac=0.25, tilt=0.0)),
    ("S4 +tilt 0.1m",       dict(noise=0.3,  gapfrac=0.25, tilt=0.1)),
    ("S5 +tilt 0.3m",       dict(noise=0.3,  gapfrac=0.25, tilt=0.3)),
]
print("\n  scenario           spatial  trend   season  mbar    baseline")
floor = None
for name, kw in SCEN:
    rows = [evaluate(corrupt(seed=200 + s, **kw)) for s in range(2)]
    med = np.median(np.array(rows), axis=0)
    print(f"  {name:18s} {med[0]:6.3f}  {med[1]:6.3f}  {med[2]:6.3f}  "
          f"{med[3]:6.3f}  {med[4]:6.3f}")
    if name.startswith("S0"):
        floor = med

print("\n  Interpret (design note §5/§6):")
print("  - S0 must be ~1: validates the harness.")
print("  - spatial vs baseline: time-varying should at least match the stationary")
print("    inverse's spatial skill; trend/season are pure bonus (stationary=0 DOF).")
print("  - tilt rows (S4/S5): planar residual is low-k where the kernel is strong,")
print("    so it can leak into low-k melt — the expected soft spot.")
assert floor[0] > 0.9 and floor[1] > 0.8 and floor[3] > 0.9, \
    f"clean floor failed harness check: {floor}"
print(f"\n  FLOOR OK (S0 clean): spatial={floor[0]:.3f} trend={floor[1]:.3f} "
      f"season={floor[2]:.3f} mbar={floor[3]:.3f}")
