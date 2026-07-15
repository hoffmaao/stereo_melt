"""Diagnose the closed-form Stubblefield inverse: sign + DC + spatial-pattern recovery.

Forward inputs use the **Stubblefield convention** (positive m = melt = column
loses mass), because the public ``forward()`` operator is in that convention.
``inverse_stationary`` flips its output to the **Shean public convention**
(positive = accretion, negative = melt), so each case is "consistent" iff
the recovered field is the **negative** of the forward field.

  case A: constant melt everywhere, m=+1 m/yr (Stubblefield)
          - tests sign convention (m>0 → h drops over time?)
          - tests DC recovery (under Shean the inverse should give
            spatially-uniform m_inv ≈ -1 m/yr)

  case B: localized Gaussian melt patch, +5 m/yr in a circle (Stubblefield)
          - tests spatial-pattern recovery
          - center of recovered patch should be -5 m/yr (Shean: negative=melt)

  case C: localized accretion patch, -5 m/yr (Stubblefield = +5 m/yr Shean)
          - mirror of B; recovered center should come back +ve (Shean: positive=accretion)

For each case we report:
  - h(t) shape vs expected (decrease for melt, increase for accretion)
  - inverse m: spatial mean, value at patch center, sign
  - whether DC offset is recoverable
  - ``ratio center/forward`` should be ≈ -1 for a sign-consistent solver

This isolates *the solver's behavior* from data quality (no observation
noise, no NaN holes, no advection, single basin geometry).
"""
from __future__ import annotations

import os
import sys

# PROJ_DATA fix per saved feedback (env-local proj.db).
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import numpy as np
import xarray as xr

from stereo_melt.dynamics.linear_perturbation import forward, inverse_stationary

# ---------------------------------------------------------------------------
# Domain: 256 × 256 cells at 100 m, 5 epochs at year-spaced times.
# Reference H = 500 m (Beardmore-ish).
# ---------------------------------------------------------------------------
NX = NY = 256
DX = DY = 100.0
H_REF = 500.0
ETA = 1e14

x = (np.arange(NX) - NX // 2) * DX
y = (np.arange(NY) - NY // 2) * DY
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx ** 2 + yy ** 2)

times = np.arange(5).astype("datetime64[Y]").astype("datetime64[s]") + np.timedelta64(0, "s")
print(f"Domain: {NX}x{NY} @ {DX} m, H_ref={H_REF} m, eta_bar={ETA}, n_epochs={len(times)}")
print(f"Times: {times}")
print()


def report(label, m_field, m_inv):
    """Report sign + amplitude + spatial-mean recovery."""
    m_inv_v = m_inv.values
    print(f"=== {label} ===")
    print(f"  forward m: spatial mean = {m_field.mean():+.3f} m/yr, "
          f"value at center = {m_field[NY//2, NX//2]:+.3f}")
    print(f"  inverse m: spatial mean = {m_inv_v.mean():+.4f} m/yr, "
          f"value at center = {m_inv_v[NY//2, NX//2]:+.4f}")
    print(f"  inverse m: min/max = {m_inv_v.min():+.3f} / {m_inv_v.max():+.3f}")
    print(f"  ratio center/forward = {m_inv_v[NY//2, NX//2] / max(abs(m_field[NY//2, NX//2]), 1e-9):+.3f}")
    print()


def run_case(label, m_2d):
    """Forward then inverse, return inverse m and the synthetic h_stack."""
    m_da = xr.DataArray(
        m_2d, dims=("y", "x"), coords={"y": y, "x": x}, name="m",
    )
    # Forward: stationary m, evaluated at our times.
    h = forward(
        m_da, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        times=xr.DataArray(times, dims="time"),
        stationary=True,
    )
    # Diagnostic: per-epoch min/max of h
    print(f"  forward h(t): per-epoch [min, max] in m:")
    for i, t in enumerate(times):
        v = h.isel(time=i).values
        print(f"    t={i}: [{v.min():+8.3f}, {v.max():+8.3f}]  mean={v.mean():+8.3f}")
    # Now invert. Use "time_mean" since that's what the wrapper uses.
    h_anom = h - h.mean("time")
    m_inv_fft = inverse_stationary(
        h_anom, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        reg=1e-3, reference="time_mean", transform="fft",
    )
    m_inv_dct = inverse_stationary(
        h_anom, H=H_REF, eta_bar=ETA, alpha=0.0, alpha_y=0.0, gamma=0.0,
        reg=1e-3, reference="time_mean", transform="dct",
    )
    report(f"{label} — FFT", m_2d, m_inv_fft)
    report(f"{label} — DCT", m_2d, m_inv_dct)
    return m_inv_fft, m_inv_dct, h


# Case A: spatially uniform m = +1 m/yr (positive = melt by Stubblefield;
# Shean inverse should give back m_inv ≈ -1)
print("============================================================")
print("Case A: spatially uniform m = +1 m/yr (pure DC mode)")
print("============================================================")
mA = np.ones((NY, NX), dtype=np.float64)
run_case("A const +1", mA)

# Case B: Gaussian melt blob centered at origin, peak +5 m/yr, sigma 1 km
print("============================================================")
print("Case B: localized Gaussian melt, +5 m/yr peak, sigma=1 km")
print("============================================================")
mB = 5.0 * np.exp(-0.5 * (r / 1000.0) ** 2)
run_case("B Gaussian +5", mB)

# Case C: localized accretion, -5 m/yr peak
print("============================================================")
print("Case C: localized Gaussian accretion, -5 m/yr peak, sigma=1 km")
print("============================================================")
mC = -5.0 * np.exp(-0.5 * (r / 1000.0) ** 2)
run_case("C Gaussian -5", mC)

# Case D: dipole — +5 to the right, -5 to the left
print("============================================================")
print("Case D: dipole (+5 right, -5 left), tests sign localization")
print("============================================================")
mD = (
    +5.0 * np.exp(-0.5 * ((xx - 5000.0) ** 2 + yy ** 2) / 1000.0 ** 2)
    + -5.0 * np.exp(-0.5 * ((xx + 5000.0) ** 2 + yy ** 2) / 1000.0 ** 2)
)
run_case("D Dipole", mD)
