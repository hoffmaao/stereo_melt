"""Time-varying velocity invariant for ``lagrangian_melt_rate``.

A velocity field that is *constant in time* must produce exactly the same
melt-rate field through the new time-resolved advection path as through the
static path. This pins the time-varying code (introduced 2026-06-15 to feed
multi-year ITS_LIVE mosaics into PIG's Lagrangian melt) so a refactor can't
silently diverge the two paths. Run::

    STEREO_MELT_BACKEND=numpy \
        python -u tests/sanity_lagrangian_timevarying.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.melt import lagrangian_melt_rate

ny, nx, res = 40, 40, 100.0
# EPSG:3031 convention: x ascends, y descends.
x = np.arange(nx) * res
y = np.arange(ny - 1, -1, -1) * res
X, _ = np.meshgrid(x, y)

# 5 DEM epochs over ~3 yr; surface thins linearly so DH/Dt != 0.
times = pd.to_datetime(
    ["2019-01-01", "2019-09-01", "2020-06-01", "2021-02-01", "2021-10-01"]
)
t_years = np.array([(t - times[0]).total_seconds() / (86400 * 365.25) for t in times])
base = 60.0 + 5e-4 * X  # gentle along-flow surface gradient (m)
h = np.stack([base - 2.0 * ty for ty in t_years])  # 2 m/yr thinning
h_stack = xr.DataArray(
    h, dims=("time", "y", "x"), coords={"time": times, "y": y, "x": x}, name="h"
)

# Static velocity: 300 m/yr along +x.
vx2d = xr.DataArray(np.full((ny, nx), 300.0), dims=("y", "x"), coords={"y": y, "x": x})
vy2d = xr.zeros_like(vx2d)

# Time-varying stack of the SAME field at 3 mid-year timestamps.
vtimes = pd.to_datetime(["2019-07-02", "2020-07-02", "2021-07-02"])
vx3d = xr.concat([vx2d, vx2d, vx2d], dim="time").assign_coords(time=vtimes)
vy3d = xr.concat([vy2d, vy2d, vy2d], dim="time").assign_coords(time=vtimes)

kw = dict(dt_yr=0.05, seed_stride=1, pairs="all", min_dt_yr=0.1, progress_interval_s=0.0)
static = lagrangian_melt_rate(h_stack, vx2d, vy2d, **kw)
tv = lagrangian_melt_rate(h_stack, vx3d, vy3d, **kw)

assert int(tv.attrs["velocity_time_varying"]) == 1, tv.attrs
assert int(tv.attrs["velocity_n_epochs"]) == 3, tv.attrs
assert int(static.attrs["velocity_time_varying"]) == 0, static.attrs

a = static.melt_rate.values
b = tv.melt_rate.values
both = np.isfinite(a) & np.isfinite(b)
assert both.sum() > 0, "no overlapping finite cells"
# finite-footprint parity: the two paths must agree cell-for-cell.
assert np.array_equal(np.isfinite(a), np.isfinite(b)), "finite masks differ"
max_abs = float(np.nanmax(np.abs(a[both] - b[both])))
print(f"overlapping finite cells: {int(both.sum())}")
print(f"static  median melt_rate: {np.nanmedian(a):+.4f} m/yr")
print(f"tv-const median melt_rate: {np.nanmedian(b):+.4f} m/yr")
print(f"max |static - tv_const| = {max_abs:.3e} m/yr")
assert max_abs < 1e-6, f"time-constant tv path diverged from static: {max_abs}"
print("PASS: time-constant velocity reproduces the static-field melt rate.")

# --- NaN-robustness: a velocity gap must not crash, corrupt, or collapse ---
import warnings

vx_gap = vx3d.copy()
vy_gap = vy3d.copy()
vx_gap[:, :10, :10] = np.nan  # punch a NaN hole in a corner, all epochs
vy_gap[:, :10, :10] = np.nan
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    gap = lagrangian_melt_rate(h_stack, vx_gap, vy_gap, **kw)
cast_warns = [w for w in wlist if "cast" in str(w.message).lower()]
assert not cast_warns, f"NaN->int cast warning leaked: {[str(w.message) for w in cast_warns]}"

g = gap.melt_rate.values
nfin_clean = int(np.isfinite(b).sum())
nfin_gap = int(np.isfinite(g).sum())
print(f"finite cells: clean={nfin_clean}  with-gap={nfin_gap}")
assert nfin_gap > 0, "NaN velocity nuked the entire melt field"
assert nfin_gap >= 0.5 * nfin_clean, f"NaN gap collapsed coverage: {nfin_gap}/{nfin_clean}"
# Cells far from the gap (flow is +x, gap is the -x/-y corner) must stay finite.
far = np.zeros_like(g, dtype=bool)
far[20:, 20:] = True
both_far = far & np.isfinite(b)
frac_far_finite = float(np.isfinite(g[both_far]).mean())
print(f"far-from-gap cells still finite: {frac_far_finite:.3f}")
assert frac_far_finite > 0.95, "NaN spread into cells far from the gap"
print("PASS: NaN velocity handled without crash, corruption, or coverage collapse.")
