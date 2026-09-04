"""Gate: the Stubblefield 2023 linear perturbation forward model.

Reproduces the qualitative behavior of paper Figures 7-8:

- Wide Gaussian melt anomaly (sigma = 10/3 * H): surface h closely
  matches the perfect-flotation prediction -delta*s.
- Narrow Gaussian (sigma = 1/3 * H): surface expression is diminished
  and h != -delta*s (non-hydrostatic).
- Adding across-channel inflow (alpha > 0): surface expression is
  damped and asymmetric.

It also pins the k=0 (DC) behaviour of ``steady_state_kernel``, the
unrelaxed-mode diagnostic and its asymptote, and the agreement between
``forward(t -> infty)`` and ``steady_state`` including the DC mode.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_linear_perturbation.py
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.constants import rhoi, rhow  # noqa: E402
from stereo_melt.dynamics import steady_state  # noqa: E402

print("Linear perturbation module imports OK.")

# Paper's reference parameters (Stubblefield 2023, Section 4)
H = 500.0  # m
eta_bar = 1e14  # Pa s
delta = rhow / rhoi - 1.0
print(f"H={H} m, eta_bar={eta_bar:.1e} Pa s, delta={delta:.4f}")

# Paper's m0 = 0.014 (non-dimensional) corresponds to ~5 m/yr
# Convert to dimensional: m_dim = m_ndim * H / tr
tr = 2.0 * eta_bar / (rhoi * 9.81 * H)
m0_dim_m_per_s = 0.014 * H / tr
m0_dim_m_per_yr = m0_dim_m_per_s * 86400.0 * 365.25
print(f"tr = {tr / (86400*365.25):.1f} yr; m0_dim = {m0_dim_m_per_yr:.2f} m/yr")

# Domain: 60 km wide, 250 m grid. Wavenumbers resolve sigma in {10/3, 1/3}*H.
L = 60_000.0
res = 250.0
x = np.arange(-L / 2, L / 2, res)
y = np.arange(L / 2, -L / 2, -res)  # descending (EPSG:3031 convention)
X, Y = np.meshgrid(x, y)


def gaussian_melt(sigma_over_H: float, m0: float) -> xr.DataArray:
    sigma = sigma_over_H * H
    m_arr = m0 * np.exp(-0.5 * (X / sigma) ** 2)
    return xr.DataArray(m_arr, dims=("y", "x"), coords={"y": y, "x": x})


# ---- Wide channel (sigma = 10/3 H) — expect h ~ -delta * s ----
m_wide = gaussian_melt(10.0 / 3.0, m0_dim_m_per_yr)
h_wide, s_wide = steady_state(m_wide, H=H, eta_bar=eta_bar, alpha=0.0, gamma=0.0, return_basal=True)
center = (len(y) // 2, len(x) // 2)
h_c = float(h_wide.values[center])
s_c = float(s_wide.values[center])
print(f"\nWide channel (sigma = 10/3 H):")
print(f"  h_center  = {h_c:.3e} m")
print(f"  s_center  = {s_c:.3e} m")
print(f"  -delta*s  = {-delta * s_c:.3e} m")
print(f"  |h + delta*s| / |h| = {abs(h_c + delta * s_c) / abs(h_c):.3e}")

assert np.sign(h_c) == -np.sign(
    s_c
), f"h and s should have opposite signs for basal melt: h={h_c}, s={s_c}"
assert abs(h_c + delta * s_c) / abs(h_c) < 0.05, "wide channel should be near-hydrostatic"
print("  PASS: wide channel is near-hydrostatic (|h + delta*s|/|h| < 5%).")

# ---- Narrow channel (sigma = 1/3 H) — expect diminished h and h != -delta*s ----
m_narrow = gaussian_melt(1.0 / 3.0, m0_dim_m_per_yr)
h_narrow, s_narrow = steady_state(
    m_narrow, H=H, eta_bar=eta_bar, alpha=0.0, gamma=0.0, return_basal=True
)
h_n = float(h_narrow.values[center])
s_n = float(s_narrow.values[center])
hyd_n = -delta * s_n
rel_dev = abs(h_n - hyd_n) / abs(hyd_n)
print(f"\nNarrow channel (sigma = 1/3 H):")
print(f"  h_center = {h_n:.3e} m")
print(f"  s_center = {s_n:.3e} m")
print(f"  -delta*s = {hyd_n:.3e} m")
print(f"  h / (-delta*s) = {h_n / hyd_n:.3f}  (ideal hydrostatic = 1.0)")

assert np.sign(h_n) == -np.sign(s_n), "narrow channel: h and s opposite signs"
assert (
    rel_dev > 0.05
), f"narrow channel should break hydrostatic flotation but only deviated by {rel_dev:.3e}"
print(f"  PASS: narrow channel breaks hydrostatic flotation ({rel_dev * 100:.1f}% deviation).")

# ---- Effect of inflow alpha -> surface expression is damped ----
h_no_inflow = float(steady_state(m_narrow, H=H, eta_bar=eta_bar, alpha=0.0).values[center])
# alpha = 1/2 in Stubblefield corresponds to ubar_0 ~ 193 m/yr
h_inflow = float(steady_state(m_narrow, H=H, eta_bar=eta_bar, alpha=0.5).values[center])
print(f"\nNarrow channel + inflow:")
print(f"  |h_center| (alpha=0)   = {abs(h_no_inflow):.3e} m")
print(f"  |h_center| (alpha=0.5) = {abs(h_inflow):.3e} m")
print(f"  damping factor = {abs(h_inflow) / abs(h_no_inflow):.3f}")
assert abs(h_inflow) < abs(
    h_no_inflow
), "inflow should diminish the narrow-channel surface expression"
print("  PASS: inflow mutes narrow-channel surface topography.")

# ---- The k = 0 (DC) mode of the steady kernel ----
# A spatially uniform melt has no gradients, so it thins the shelf in exact
# hydrostatic flotation: h/s = -delta, and the flotation departure T = 1.
from stereo_melt.dynamics.linear_perturbation import LinearPerturbation

fb = 1.0 - rhoi / rhow
_zero = np.array([[0.0]])
for gam in (0.0, -0.02, 0.03):
    _m = LinearPerturbation(H=H, eta_bar=eta_bar, rho_i=rhoi, rho_w=rhow, g=9.81, gamma=gam)
    Gh0, Gs0 = _m.steady_state_kernel(_zero, _zero)
    Gh0, Gs0 = complex(Gh0[0, 0]), complex(Gs0[0, 0])
    T0 = Gh0 / (fb * (Gh0 - Gs0))
    print(f"\nDC kernel at gamma={gam:+.2f}: G_h(0)={Gh0.real:.6f}  G_s(0)={Gs0.real:.6f}  "
          f"h/s={(Gh0 / Gs0).real:+.6f}  T(0)={T0.real:.9f}")
    assert abs((Gh0 / Gs0).real + delta) < 1e-12, "DC mode must be exact flotation h/s = -delta"
    assert abs(T0.real - 1.0) < 1e-12 and abs(T0.imag) < 1e-12, "T(0) must be exactly 1"
    if gam == 0.0:
        assert abs(Gh0.real + 2.0) < 1e-12, f"G_h(0) at gamma=0 must be -2, got {Gh0.real}"
        assert abs(Gs0.real - 2.0 / delta) < 1e-9, f"G_s(0) at gamma=0 must be 2/delta"
print("  PASS: DC kernel is exact flotation, T(0) = 1, G_h(0) = -2 at gamma = 0.")

# ---- Which modes have no steady state (Re lambda_+ >= 0) ----
# The kernel does NOT screen for stability -- it returns the analytic
# continuation everywhere -- so the diagnostic is what has to be right. The
# criterion is Re(lambda_+) >= 0; asymptotically R -> 1/k', B -> 0, so
# Re(lambda_+) -> gamma - delta/k' and every k'H > delta/gamma is unrelaxed.
from stereo_melt.dynamics.linear_perturbation import _wavenumber_grids  # noqa: E402

_kx, _ky = _wavenumber_grids(128, 96, 250.0, 250.0)
_kp = np.hypot(np.asarray(_kx), np.asarray(_ky)) * H          # k' = k H
_dc = np.asarray(_kp) <= 0
_gamma_dc = delta / (2.0 * (delta + 1.0))                     # DC threshold

_m0 = LinearPerturbation(H=H, eta_bar=eta_bar, rho_i=rhoi, rho_w=rhow, g=9.81, gamma=0.0)
_u0 = np.asarray(_m0.unrelaxed_modes(_kx, _ky))
print(f"\nUnrelaxed modes: gamma=0 -> {_u0.sum()} of {_u0.size}; "
      f"cutoff wavelength {_m0.unrelaxed_cutoff_wavelength_m()}")
assert not _u0.any(), "gamma = 0 must leave every mode relaxed"
assert _m0.unrelaxed_cutoff_wavelength_m() == np.inf

# Below the DC threshold: a HIGH-k band only, cut where the asymptote says.
_gam = 0.02
assert _gam < _gamma_dc
_m1 = LinearPerturbation(H=H, eta_bar=eta_bar, rho_i=rhoi, rho_w=rhow, g=9.81, gamma=_gam)
_u1 = np.asarray(_m1.unrelaxed_modes(_kx, _ky))
_cut = _kp[_u1].min()
_asym = delta / _gam
print(f"  gamma={_gam}: {_u1.sum()} of {_u1.size} unrelaxed; first unrelaxed k'={_cut:.3f}, "
      f"asymptotic delta/gamma={_asym:.3f}; lambda_c={_m1.unrelaxed_cutoff_wavelength_m():.0f} m")
assert not _u1[_dc], "DC must still relax below the DC threshold"
# A genuine PARTIAL band: some modes in, some out (bool(...) so a numpy scalar
# is compared by value, not against the Python singleton).
assert bool(_u1.any()) and bool((~_u1).any()), "expected a partial (high-k) band"
# ...and the band is the HIGH-k tail, split at the ANALYTIC cut delta/gamma,
# which is computed from the model constants and never from the mask itself.
# A scattered mid-k mask would fail this even though it passed the old form.
_tol = 0.01
_far_above = _kp > _asym * (1.0 + _tol)
_far_below = _kp < _asym * (1.0 - _tol)
assert _far_above.any() and bool(_u1[_far_above].all()), \
    "every k' well above delta/gamma must be unrelaxed"
assert _far_below.any() and not bool(_u1[_far_below].any()), \
    "no k' well below delta/gamma may be unrelaxed"
assert abs(_cut / _asym - 1.0) < _tol, f"cutoff {_cut} vs asymptote {_asym}"
assert abs(_m1.unrelaxed_cutoff_wavelength_m() / (2 * np.pi * H * _gam / delta) - 1) < 1e-12

# At/above the DC threshold the DC bin joins, and then nothing relaxes.
_m2 = LinearPerturbation(H=H, eta_bar=eta_bar, rho_i=rhoi, rho_w=rhow, g=9.81,
                         gamma=_gamma_dc + 1e-6)
_u2 = np.asarray(_m2.unrelaxed_modes(_kx, _ky))
print(f"  gamma={_gamma_dc + 1e-6:.5f} (just past the DC threshold): "
      f"{_u2.sum()} of {_u2.size} unrelaxed, DC flagged={bool(_u2[_dc][0])}")
assert _u2[_dc].all(), "DC must be flagged once gamma >= delta/(2(delta+1))"
assert _u2.all(), "past the DC threshold every mode is unrelaxed"
print("  PASS: unrelaxed-mode diagnostic matches Re(lambda_+) >= 0 and its asymptote.")

# The kernel itself stays finite there -- it does not mask one bin while
# returning numbers for an equally unrelaxed band.
_gh_hot, _gs_hot = _m2.steady_state_kernel(_zero, _zero)
assert np.isfinite(complex(_gh_hot[0, 0])), "kernel must not NaN-mask only the DC bin"
assert np.isfinite(complex(_gs_hot[0, 0]))
_T_hot = complex(_gh_hot[0, 0]) / (fb * (complex(_gh_hot[0, 0]) - complex(_gs_hot[0, 0])))
assert abs(_T_hot.real - 1.0) < 1e-12, "T(0) = 1 holds for any lambda_0 != 0"
print("  PASS: kernel is unscreened and self-consistent past the threshold.")

# steady_state must not stay silent when it hands back a non-steady field.
import warnings as _warnings  # noqa: E402

with _warnings.catch_warnings(record=True) as _w:
    _warnings.simplefilter("always")
    steady_state(m_wide, H=H, eta_bar=eta_bar, gamma=_gam)
_msgs = [str(x.message) for x in _w if issubclass(x.category, RuntimeWarning)]
assert any("no steady state" in t for t in _msgs), f"expected a warning, got {_msgs}"
with _warnings.catch_warnings(record=True) as _w2:
    _warnings.simplefilter("always")
    steady_state(m_wide, H=H, eta_bar=eta_bar, gamma=0.0)
# (transfer_functions emits its own numpy divide warnings at k=0; only ours
# is about stability, so match on the message rather than the category.)
assert not [x for x in _w2 if "no steady state" in str(x.message)], \
    "gamma = 0 has no unrelaxed modes and must not warn"
print("  PASS: steady_state warns iff some mode has no steady state.")

# ---- Stationary forward at large t should match the steady state ----
from stereo_melt.dynamics import forward

# Long-wavelength relaxation time scale t_e = 2 tr (1 + 1/delta) ≈ 19 tr
# for delta ≈ 0.11, so a wide-channel steady state needs many t_e. The
# comparison covers the FULL field, DC included: a Gaussian on a finite tile
# carries a large mean, and that mode is exactly where the two paths used to
# disagree.
t_many = tr * 200.0
h_forward = forward(
    m_wide,
    H=H,
    eta_bar=eta_bar,
    stationary=True,
    times=np.array([t_many]),
    return_basal=False,
)
h_ss = steady_state(m_wide, H=H, eta_bar=eta_bar)
ratio = float(h_forward.isel(time=0).values[center]) / float(h_ss.values[center])
print(f"\nforward(t = 200 tr) / steady_state at wide-channel center = {ratio:.6f} (want ~1.0)")
assert abs(ratio - 1.0) < 0.02, f"forward at large t should match steady_state, got ratio={ratio}"
rel_field = float(np.max(np.abs(h_forward.isel(time=0).values - h_ss.values))
                  / np.max(np.abs(h_ss.values)))
print(f"  max|forward - steady_state| / max|steady_state| over the whole field = {rel_field:.2e}")
assert rel_field < 0.02, f"forward and steady_state disagree over the field by {rel_field:.3e}"
mean_ratio = float(h_forward.isel(time=0).values.mean()) / float(h_ss.values.mean())
print(f"  domain-mean ratio (the k = 0 mode alone) = {mean_ratio:.6f}")
assert abs(mean_ratio - 1.0) < 0.02, f"DC mode disagrees: mean ratio {mean_ratio}"
print("  PASS: forward(t -> infty) agrees with steady_state, DC mode included.")

print("\nGATE PASSED: all linear-perturbation checks.")
