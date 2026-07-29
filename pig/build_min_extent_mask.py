"""Build the window-minimum floating-shelf extent mask for Pine Island.

Stage 4b (after ``pig.tilt_fit``, before/independent of ``pig.run_melt``).
Intersects three constraints on the stack grid via
:func:`stereo_melt.shelf_extent.min_shelf_extent`:

  1. BedMachine v3 floating ice (grounding-line side),
  2. Greene et al. 2022 annual observed ice extents over the stack window
     (covers the 2017-2020 tongue collapse, archive ends 2021.2),
  3. the stack's own per-epoch ocean test (surface < 5 m freeboard at
     >= 2 epochs — covers post-Greene calving at exactly our epochs).

Pixels the shelf lost mid-window otherwise enter the solvers as
ice-to-ocean dh/dt cliffs read as extreme melt.

Run:

    python -m pig.build_min_extent_mask [--res 250] [--tag is2ctempo]

Writes ``processed/pig_min_extent<suffix>_<start>_<end>.nc`` plus a QC
figure. ``pig.run_melt`` intersects its floating mask with this file
when present (``PIG_MIN_EXTENT=0`` opts out).
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _env_proj

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.io.greene import greene_min_extent_on_grid
from stereo_melt.shelf_extent import min_shelf_extent

from pig import config
from pig.run_melt import load_floating_mask, load_stack


def main(res_override: float | None = None, tag: str | None = None) -> None:
    config.ensure_output_dirs()
    if res_override is not None:
        stack_prefix = f"pig_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix, out_suffix = "pig_stack", ""
    if tag:
        stack_prefix += f"_{tag}"
        out_suffix += f"_{tag}"

    print(f"Loading stack ({stack_prefix})...")
    stack = load_stack(stack_prefix=stack_prefix)
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, "
          f"x={stack.sizes['x']}")

    print("Building floating-ice mask (BedMachine v3)...")
    floating = load_floating_mask(stack)

    print(f"Loading Greene 2022 window-minimum extent "
          f"({config.START_TIME}..{config.END_TIME})...")
    greene = greene_min_extent_on_grid(
        stack, config.GREENE_ICEMASK_MAT,
        t0=config.START_TIME, t1=config.END_TIME,
    )
    print(f"  epochs: {greene.attrs['epochs_used']}")

    print("Intersecting (min_shelf_extent, per-epoch ocean test)...")
    ext = min_shelf_extent(stack, floating, greene_extent=greene)
    res = float(abs(stack["x"].values[1] - stack["x"].values[0]))
    km2 = res * res / 1e6
    a = ext.attrs
    print(f"  floating {a['n_floating']} px ({a['n_floating'] * km2:.0f} km^2)")
    print(f"  - ocean test removed {a['n_removed_ocean_test']} px "
          f"({a['n_removed_ocean_test'] * km2:.0f} km^2)")
    print(f"  - greene removed     {a['n_removed_greene']} px "
          f"({a['n_removed_greene'] * km2:.0f} km^2)")
    n_final = int(ext.values.sum())
    print(f"  min extent {n_final} px ({n_final * km2:.0f} km^2)")

    out_nc = (
        config.PROCESSED_DIR
        / f"pig_min_extent{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    ds = xr.Dataset({"min_extent_mask": ext.astype(np.uint8)})
    ds["min_extent_mask"].attrs = dict(ext.attrs)
    ds.to_netcdf(out_nc)
    print(f"wrote {out_nc}")

    # QC map: what the min extent removes from the static floating mask.
    cls = np.zeros(ext.shape, dtype=np.int8)
    fl = floating.values.astype(bool)
    cls[fl & ~ext.values] = 1
    cls[ext.values] = 2
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.pcolormesh(
        stack["x"].values / 1e3, stack["y"].values / 1e3, cls,
        cmap=plt.matplotlib.colors.ListedColormap(
            ["0.92", "#c9432f", "#3465a8"]),
        vmin=-0.5, vmax=2.5, shading="auto",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(
        f"PIG window-minimum shelf extent {config.START_TIME}.."
        f"{config.END_TIME}\nblue kept {n_final * km2:.0f} km$^2$ · red "
        f"removed {(a['n_removed_ocean_test'] + a['n_removed_greene']) * km2:.0f} "
        f"km$^2$ (ocean test {a['n_removed_ocean_test'] * km2:.0f} + greene "
        f"{a['n_removed_greene'] * km2:.0f})"
    )
    out_png = config.FIGURES_DIR / f"min_extent_mask{out_suffix}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", type=float, default=None,
                        help="stack resolution variant (meters)")
    parser.add_argument("--tag", default=None,
                        help="experiment tag matching the stack build")
    args = parser.parse_args()
    main(res_override=args.res, tag=args.tag)
