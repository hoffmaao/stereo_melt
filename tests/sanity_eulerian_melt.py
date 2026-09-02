"""End-to-end synthetic sanity test for the Eulerian melt solver.

Steady-state shelf (dH/dt = 0), constant velocity, zero SMB, linear thickness
gradient. Expected: flux_div = vx * dH/dx = 200 * (-0.005) = -1 m/yr;
melt_rate = dHdt + flux_div - smb = 0 + (-1) - 0 = -1 m/yr (Shean
convention: negative = melt, positive = accretion).
"""

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.stack import target_grid, build_stack, save_stack, load_stack
from stereo_melt.freeboard import freeboard_to_thickness, thickness_to_freeboard
from stereo_melt.kinematics import (
    dh_dt,
    gradient,
    divergence,
    flux_divergence,
    FiniteDifferenceDivergence,
    SECONDS_PER_YEAR,
)
from stereo_melt.melt import eulerian_melt_rate
from stereo_melt.constants import rhow, rhoi

print("All new modules import.")

# Freeboard round trip
h, d = 50.0, 5.0
H = freeboard_to_thickness(h, d)
h_back = thickness_to_freeboard(H, d)
print(f"h={h}, d={d} -> H={H:.2f} -> h_back={h_back:.2f}")
assert abs(h_back - h) < 1e-9

# 20 km x 20 km domain, 250 m grid
x = np.arange(-10000, 10000, 250, dtype=float)
y = np.arange(10000, -10000, -250, dtype=float)
Y, X = np.meshgrid(y, x, indexing="ij")

# Linear thickness gradient: 350 m at x=-10km, 250 m at x=+10km
H0 = 300.0 - 0.005 * X
h0 = H0 * (rhow - rhoi) / rhow  # hydrostatic surface (d=0)

# Constant 200 m/yr outflow in +x
vx = xr.DataArray(np.full_like(H0, 200.0), dims=("y", "x"), coords={"y": y, "x": x})
vy = xr.DataArray(np.zeros_like(H0), dims=("y", "x"), coords={"y": y, "x": x})

# 5-epoch steady stack (dh/dt = 0)
times = pd.to_datetime(["2013-01-01", "2013-07-01", "2014-01-01", "2014-07-01", "2015-01-01"])
layers = [xr.DataArray(h0.copy(), dims=("y", "x"), coords={"y": y, "x": x}) for _ in times]
h_stack = xr.concat(layers, dim=pd.Index(times, name="time"))

result = eulerian_melt_rate(h_stack, vx, vy, a_dot=0.0)
print("Result vars:", list(result.data_vars))
print(f"Mean flux_div (m/yr): {float(result.flux_div.mean()):.3f}")
print(f"Mean melt_rate (m/yr): {float(result.melt_rate.mean()):.3f}")
print(f"dHdt max abs: {float(abs(result.dHdt).max()):.4g}")
print("Expected flux_div ~ -1.0 m/yr; melt_rate ~ -1.0 m/yr (Shean: negative = melt)")
