"""Gate: the LSQ tilt-stack optimizer, against a known injected tilt.

Start from a flat reference DEM. Add a known per-epoch planar tilt
(dx, dy, dz) to each layer and verify that:

1. The fitted tilt parameters recover the injected values.
2. The corrected stack collapses back to the flat reference (up to a
   global constant absorbed by the intercept block).

Also tests the T=2 degenerate case.

Runs at the library ``min_width`` default: this 10 km synthetic has a
spatial spread (``dist_ptp``) of ~4.7 km, while the basin drivers pass
``min_width=10000``. The reasoning behind the default lives in the
``min_width`` docstring of ``stereo_melt.coregister.tilt.fit_tilt_stack``.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_tilt_stack.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.backend import backend as _BACKEND  # noqa: E402
from stereo_melt.coregister.tilt import apply_tilt, fit_tilt_stack  # noqa: E402

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
# robust=False: the data are exact, so the IRLS residual scale collapses to
# 0 and the Tukey weights start rejecting ~28% of pixels as "outliers",
# biasing the slopes. Robustness is exercised on noisy stacks elsewhere.
params, stack_corrected = fit_tilt_stack(
    stack, control_mask=mask, Eint=1e6, Edhdt=1e-6, Ex=1.0, Ey=1.0, Ez=1e3,
    robust=False,
)

print("tilt_dx true: ", dx_true)
print("tilt_dx fit:  ", params.tilt_dx.values)
print("tilt_dy true: ", dy_true)
print("tilt_dy fit:  ", params.tilt_dy.values)
print("tilt_dz true: ", dz_true)
print("tilt_dz fit:  ", params.tilt_dz.values)

# Only the MEAN-REMOVED (epoch-to-epoch) part of each tilt component is
# identifiable, for the same reason the T=2 block below spells out: a
# constant-across-epochs tilt is exactly degenerate with a gradient in the
# per-pixel intercept, so the solver regularizes that common mode to zero
# (shifting dx_true by a constant moves the fit by exactly minus that
# constant). A second, weaker leak runs between the linear-in-time part of
# a slope and the per-pixel trend field, bounded by Edhdt. Neither reaches
# melt: both are absorbed by the static reference and cancel in any time
# difference. The epoch-to-epoch variation -- the part that does matter --
# comes back at ~1e-14, so this asserts it far tighter than the old
# absolute atol=1e-6 ever did.
def _mean_removed(fit, true, name, atol):
    res = (fit - fit.mean()) - (true - true.mean())
    assert np.allclose(res, 0.0, atol=atol), f"{name} (mean-removed) mismatch: {res}"


# The cupy path solves in float32 by design (tilt.py casts A to float32 on
# the device); slopes of 1e-4 come back ~5e-8 off there, which is float32
# precision through the LSQ's conditioning, not an error. Keep the float64
# assertion tight on the CPU path.
_slope_atol = 1e-6 if _BACKEND == "cupy" else 1e-9
_mean_removed(params.tilt_dx.values, dx_true, "dx", _slope_atol)
_mean_removed(params.tilt_dy.values, dy_true, "dy", _slope_atol)
_mean_removed(params.tilt_dz.values, dz_true, "dz", 1e-3)

# Corrected stack should collapse every epoch onto ONE common surface.
# It cannot collapse to a flat plane: the unidentifiable common-mode slope
# (above) survives as a static ramp that is identical at every epoch, and a
# static ramp is precisely what the reference DEM absorbs -- it cancels in
# any time difference, so it never reaches dh/dt or melt. The meaningful
# quantity is therefore the epoch-to-epoch scatter about the time mean,
# which lands at ~1e-5 m; asserting it at 1e-4 is two orders tighter than
# the old absolute 1e-3 in the dimension that actually matters.
corr = stack_corrected.values
interior = (slice(None), slice(2, -2), slice(2, -2))
c = corr[interior]
scatter = float((c - c.mean(axis=0)).std())
static = float(c.mean(axis=0).std())
print(f"Corrected-stack epoch-to-epoch scatter: {scatter:.4e} m")
print(f"Corrected-stack static (absorbed) ramp: {static:.4e} m")
assert scatter < 1e-4, f"Epoch-to-epoch scatter {scatter} too large"

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
params2, _ = fit_tilt_stack(stack2, control_mask=mask, Eint=1e6, Edhdt=1e-6,
                            Ex=1.0, Ey=1.0, Ez=1e3, robust=False)
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
