"""Gate: window-minimum floating-shelf extent (shelf_extent.min_shelf_extent).

Synthetic rungs (self-contained):
  1. calved region (reads ~0 m at >= min_hits epochs) is excluded;
  2. a single blunder low reading does NOT delete shelf (min_hits=2);
  3. grounded strip excluded via the static floating mask;
  4. Greene coastline constraint excluded where the observed extent says
     no ice;
  5. never-observed (all-NaN) pixels survive (no evidence != calved);
  6. bookkeeping attrs match the constructed truth.

Real-archive rungs (run only if data/Greene2022/icemask_composite.mat
exists):
  7. PIG-bbox extent is a proper nonempty subset of the grid;
  8. nesting: min extent over 2010-2024 is a subset of 2021-2024's.

Run:  $PY tests/gate_min_shelf_extent.py
"""
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.shelf_extent import min_shelf_extent  # noqa: E402

GREENE_MAT = pathlib.Path(
    "/wd2/projects/stereo_melt/data/Greene2022/icemask_composite.mat"
)

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- synthetic
ny, nx, nt = 20, 40, 6
x = np.arange(nx, dtype=float) * 250.0
y = -np.arange(ny, dtype=float) * 250.0  # descending, workspace convention
t = np.array([np.datetime64("2019-01-01") + np.timedelta64(180 * k, "D")
              for k in range(nt)])

z = np.full((nt, ny, nx), 30.0)
z[3:, :, 30:] = 0.5            # calved after epoch 3: 3 ocean hits
z[2, 10, 10] = 0.2             # single blunder low reading
z[:, 5, 5] = np.nan            # never observed
stack = xr.DataArray(z, dims=("time", "y", "x"),
                     coords={"time": t, "y": y, "x": x})

floating = np.ones((ny, nx), bool)
floating[:5, :] = False        # grounded strip
floating_da = xr.DataArray(floating, dims=("y", "x"),
                           coords={"y": y, "x": x})

greene = np.ones((ny, nx), bool)
greene[:, :4] = False          # coastline says no ice here
greene_da = xr.DataArray(greene, dims=("y", "x"),
                         coords={"y": y, "x": x},
                         attrs={"epochs_used": "synthetic"})

ext = min_shelf_extent(stack, floating_da, greene_extent=greene_da,
                       freeboard_max=5.0, min_hits=2)
m = ext.values

check("calved region excluded", not m[6:, 30:].any())
check("blunder pixel survives min_hits=2", bool(m[10, 10]))
check("grounded strip excluded", not m[:5, :].any())
check("greene constraint applied", not m[6:, :4].any())
check("all-NaN pixel survives", bool(m[5, 5]))
check("interior shelf retained", bool(m[6:, 4:30].all()))
truth_ocean = int(floating[:, 30:].sum())
check("attrs bookkeeping",
      ext.attrs["n_removed_ocean_test"] == truth_ocean
      and ext.attrs["n_removed_greene"] == int((floating[:, :4]).sum()),
      f"ocean {ext.attrs['n_removed_ocean_test']} (truth {truth_ocean}), "
      f"greene {ext.attrs['n_removed_greene']}")

# ------------------------------------------------------------- real archive
if GREENE_MAT.exists():
    from stereo_melt.io.greene import greene_min_extent_on_grid

    gx = np.arange(-1700e3, -1540e3, 250.0)
    gy = np.arange(-150e3, -360e3, -250.0)
    tmpl = xr.DataArray(np.zeros((gy.size, gx.size)), dims=("y", "x"),
                        coords={"y": gy, "x": gx})
    e_full = greene_min_extent_on_grid(tmpl, GREENE_MAT,
                                       t0="2010-01-01", t1="2024-01-10")
    e_late = greene_min_extent_on_grid(tmpl, GREENE_MAT,
                                       t0="2021-01-01", t1="2024-01-01")
    f_full = float(e_full.values.mean())
    f_late = float(e_late.values.mean())
    check("PIG extent nonempty proper subset",
          0.0 < f_full < 1.0, f"ice fraction {f_full:.3f}")
    check("nesting: 2010-2024 subset of 2021-2024",
          bool((~e_full.values | e_late.values).all())
          and f_full < f_late,
          f"{f_full:.3f} <= {f_late:.3f}")
    print(f"    epochs full window: {e_full.attrs['epochs_used']}")
    print(f"    epochs late window: {e_late.attrs['epochs_used']}")
else:
    print("[skip] real Greene archive not found; synthetic rungs only")

print()
if failures:
    print(f"GATE FAIL ({len(failures)}): {failures}")
    sys.exit(1)
print("GATE PASS")
