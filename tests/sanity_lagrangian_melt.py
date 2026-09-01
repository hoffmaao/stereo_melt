"""End-to-end synthetic sanity test for the Lagrangian melt solver.

Steady-state shelf (time-invariant surface), uniform velocity, linear
thickness gradient, zero SMB. Under these conditions the Eulerian and
Lagrangian solvers should agree — ∇·u = 0 so only the +DH/Dt term
contributes (Shean convention). Expected: melt_rate ~ -1 m/yr
(Shean convention: negative = melt, positive = accretion).
"""

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.constants import rhoi, rhow
from stereo_melt.melt import eulerian_melt_rate, lagrangian_melt_rate

print("All new modules import.")

# 20 km x 20 km domain, 250 m grid
x = np.arange(-10000, 10000, 250, dtype=float)
y = np.arange(10000, -10000, -250, dtype=float)
Y, X = np.meshgrid(y, x, indexing="ij")

# Linear thickness field H = 300 - 0.005 x (350 m at -10 km, 250 m at +10 km)
H0 = 300.0 - 0.005 * X
h0 = H0 * (rhow - rhoi) / rhow

# Constant velocity 200 m/yr in +x, zero in y
vx = xr.DataArray(np.full_like(H0, 200.0), dims=("y", "x"), coords={"y": y, "x": x})
vy = xr.DataArray(np.zeros_like(H0), dims=("y", "x"), coords={"y": y, "x": x})

# 5-epoch steady stack
times = pd.to_datetime(["2013-01-01", "2013-07-01", "2014-01-01", "2014-07-01", "2015-01-01"])
layers = [xr.DataArray(h0.copy(), dims=("y", "x"), coords={"y": y, "x": x}) for _ in times]
h_stack = xr.concat(layers, dim=pd.Index(times, name="time"))

euler = eulerian_melt_rate(h_stack, vx, vy, a_dot=0.0)
lagr = lagrangian_melt_rate(h_stack, vx, vy, a_dot=0.0, dt_yr=0.05, seed_stride=1)

print("Eulerian  mean melt_rate: {:.3f} m/yr".format(float(euler.melt_rate.mean())))
print("Lagrangian mean melt_rate: {:.3f} m/yr".format(float(lagr.melt_rate.mean(skipna=True))))

# Interior box (skip 10 pixels at each boundary where particles advect off-grid)
euler_arr = euler.melt_rate.values
lagr_arr = lagr.melt_rate.values
interior = (slice(10, -10), slice(10, -10))
euler_core = float(np.nanmean(euler_arr[interior]))
lagr_core = float(np.nanmean(lagr_arr[interior]))
print(f"Interior-box Eulerian  melt_rate: {euler_core:.3f} m/yr")
print(f"Interior-box Lagrangian melt_rate: {lagr_core:.3f} m/yr")

assert abs(euler_core - (-1.0)) < 0.01, f"Eulerian core far from -1.0 (Shean): {euler_core}"
assert abs(lagr_core - (-1.0)) < 0.05, f"Lagrangian core far from -1.0 (Shean): {lagr_core}"
print("Lagrangian core-region melt_rate matches Eulerian to within 0.05 m/yr.")

print(
    "Lagrangian dataset vars:",
    list(lagr.data_vars),
    "attrs keys:",
    sorted(lagr.attrs.keys()),
)
