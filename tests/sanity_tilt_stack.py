"""Synthetic sanity test for the LSQ tilt-stack optimizer.

Start from a flat reference DEM. Add a known per-epoch planar tilt
(dx, dy, dz) to each layer and verify that:

1. The fitted tilt parameters recover the injected values.
2. The corrected stack collapses back to the flat reference (up to a
   global constant absorbed by the intercept block).

Also tests the T=2 degenerate case.
"""

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.coregister.tilt import apply_tilt, fit_tilt_stack

print("All new modules import.")

# 10 km x 10 km grid, 250 m spacing (EPSG:3031: x ascending, y descending)
x = np.arange(-5000, 5000, 250, dtype=float)
y = np.arange(5000, -5000, -250, dtype=float)
ny, nx = y.size, x.size
X, Y = np.meshgrid(x, y)

# Flat reference DEM
z_ref = np.full((ny, nx), 100.0)

# Inject known tilts at 4 epochs (slopes and offsets chosen to be distinct
# and to have non-degenerate signal — zero-mean across epochs so the
# per-pixel intercept can't absorb them globally).
times = pd.to_datetime(["2013-01-01", "2013-07-01", "2014-01-01", "2014-07-01"])
dx_true = np.array([1e-4, -1e-4, 5e-5, -5e-5])
dy_true = np.array([5e-5, 5e-5, -5e-5, -5e-5])
dz_true = np.array([0.5, -0.5, 0.3, -0.3])

xref_true = X.mean()
yref_true = Y.mean()

layers = []
for k in range(len(times)):
    tilt = dx_true[k] * (X - xref_true) + dy_true[k] * (Y - yref_true) + dz_true[k]
    layers.append(xr.DataArray(z_ref + tilt, dims=("y", "x"), coords={"y": y, "x": x}))
stack = xr.concat(layers, dim=pd.Index(times, name="time"))

# Whole-domain control (every pixel is a control surface for this synthetic)
mask = xr.DataArray(np.ones((ny, nx), dtype=bool), dims=("y", "x"), coords={"y": y, "x": x})

# Regularization:
#   Eint/Ez large  -> weak prior, intercepts/offsets free to fit
#   Edhdt small    -> strong prior of zero per-pixel trend (test has none)
#   Ex/Ey large    -> weak prior on tilt slopes (let solver find them)
params, stack_corrected = fit_tilt_stack(
    stack, control_mask=mask, Eint=1e6, Edhdt=1e-6, Ex=1.0, Ey=1.0, Ez=1e3
)

print("tilt_dx true: ", dx_true)
print("tilt_dx fit:  ", params.tilt_dx.values)
print("tilt_dy true: ", dy_true)
print("tilt_dy fit:  ", params.tilt_dy.values)
print("tilt_dz true: ", dz_true)
print("tilt_dz fit:  ", params.tilt_dz.values)

# Slope recovery should be very accurate; absolute offset can drift by a
# common mean (absorbed into intercept) so compare as differences.
assert np.allclose(
    params.tilt_dx.values, dx_true, atol=1e-6
), f"dx mismatch: {params.tilt_dx.values - dx_true}"
assert np.allclose(
    params.tilt_dy.values, dy_true, atol=1e-6
), f"dy mismatch: {params.tilt_dy.values - dy_true}"
dz_fit = params.tilt_dz.values
dz_res = (dz_fit - dz_fit.mean()) - (dz_true - dz_true.mean())
assert np.allclose(dz_res, 0.0, atol=1e-3), f"dz (mean-removed) mismatch: {dz_res}"

# Corrected stack should collapse to a flat value (up to a constant)
corr = stack_corrected.values
interior = (slice(None), slice(2, -2), slice(2, -2))
spread = float(corr[interior].std())
print(f"Corrected-stack std over interior box: {spread:.4e} m")
assert spread < 1e-3, f"Residual spread {spread} too large"

print("4-epoch tilt recovery passed.")

# ----- T = 2 minimal case -----
# For T=2 the solver can only recover the zero-mean (antisymmetric) part
# of each tilt component, because a constant-across-epochs tilt is
# exactly degenerate with a gradient in the per-pixel intercept. This
# is fine for correcting a stack to a common reference: the intercept
# absorbs any common-mode bias. Test with anti-symmetric tilts.
times2 = pd.to_datetime(["2013-01-01", "2013-07-01"])
dx2 = np.array([1e-4, -1e-4])
dy2 = np.array([5e-5, -5e-5])
dz2 = np.array([0.5, -0.5])
layers2 = []
for k in range(2):
    tilt_k = dx2[k] * (X - xref_true) + dy2[k] * (Y - yref_true) + dz2[k]
    layers2.append(xr.DataArray(z_ref + tilt_k, dims=("y", "x"), coords={"y": y, "x": x}))
stack2 = xr.concat(layers2, dim=pd.Index(times2, name="time"))
params2, _ = fit_tilt_stack(stack2, control_mask=mask, Eint=1e6, Edhdt=1e-6, Ex=1.0, Ey=1.0, Ez=1e3)
print("T=2 tilt_dx fit vs true:", params2.tilt_dx.values, "vs", dx2)
print("T=2 tilt_dy fit vs true:", params2.tilt_dy.values, "vs", dy2)
assert np.allclose(params2.tilt_dx.values, dx2, atol=1e-6)
assert np.allclose(params2.tilt_dy.values, dy2, atol=1e-6)
print("T=2 minimal case passed.")

# ----- apply_tilt helper -----
dem_single = layers[0]
dem_corrected = apply_tilt(
    dem_single, dx_true[0], dy_true[0], dz_true[0], xref=xref_true, yref=yref_true
)
assert np.allclose(dem_corrected.values, z_ref, atol=1e-9), "apply_tilt round-trip failed"
print("apply_tilt round-trip passed.")
