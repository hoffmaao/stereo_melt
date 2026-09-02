"""Synthetic validation of the production Stubblefield inverse kernels.

Round-trips the faithful forward/inverse on a non-dimensional shelf and checks
that a sub-H channel (which the forward damps) is recovered exactly by the
Stubblefield inverse but UNDER-recovered by the hydrostatic m=-h/2.

Run::
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u tests/sanity_stubblefield_inverse.py
"""
import numpy as np

from stereo_melt.constants import rhoi, rhow
from stereo_melt.dynamics.stubblefield_inverse import (
    _steady_inverse_nd,
    stubblefield_forward_steady,
)

delta = rhow / rhoi - 1.0
Nx = 201
L = 40.0
x = np.linspace(-L, L, Nx)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, x, indexing="ij")

# broad melt + sub-H narrow channel (width 0.4 H < H -> non-hydrostatic)
m = -1.0 * np.exp(-(X**2 + Y**2) / (2 * 5.0**2)) - 2.5 * np.exp(-(X - 8.0) ** 2 / (2 * 0.4**2))

print("=== round-trip (alpha=0) ===")
h = stubblefield_forward_steady(m, dx, dx, delta, alphax=0.0, alphay=0.0)
m_rec = _steady_inverse_nd(h, dx, dx, delta, 0.0, 0.0, 0.0, tik=0.0)
m_hydro = -h / 2.0
corr = float(np.corrcoef(m_rec.ravel(), m.ravel())[0, 1])
print(f"  stubblefield: corr={corr:.5f}  max|Δ|={np.max(np.abs(m_rec - m)):.2e}")
assert corr > 0.999, f"round-trip corr too low: {corr}"

ch = np.unravel_index(np.argmin(m), m.shape)
print(f"  channel depth: true={m[ch]:+.3f}  stubblefield={m_rec[ch]:+.3f}  hydrostatic={m_hydro[ch]:+.3f}")
assert abs(m_rec[ch] - m[ch]) < 1e-3, "stubblefield did not recover the channel"
assert abs(m_hydro[ch] - m[ch]) > 0.1, "hydrostatic should under-recover the channel"

print("\n=== advection (alpha != 0) round-trips too ===")
ha = stubblefield_forward_steady(m, dx, dx, delta, alphax=0.3, alphay=-0.2)
m_a = _steady_inverse_nd(ha, dx, dx, delta, 0.3, -0.2, 0.0, tik=0.0)
corr_a = float(np.corrcoef(m_a.ravel(), m.ravel())[0, 1])
print(f"  alpha=(0.3,-0.2): corr={corr_a:.5f}  max|Δ|={np.max(np.abs(m_a - m)):.2e}")
assert corr_a > 0.999, f"advected round-trip corr too low: {corr_a}"

print("\nPASS: faithful forward/inverse round-trip (alpha=0 and advected); Stubblefield recovers the "
      "sub-H channel exactly where hydrostatic under-recovers it.")
