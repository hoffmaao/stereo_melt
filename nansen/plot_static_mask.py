"""Standard QC plot: Nansen LS mask + per-pixel strip count.

Two-panel figure produced by
:func:`stereo_melt.coregister.tilt_qc.plot_static_mask_and_strip_count`:

  * left  — static-control (LS) mask fed to the per-epoch tilt LSQ
  * right — per-pixel REMA strip count across the stack window

Run:

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -m nansen.plot_static_mask
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Point pyproj at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

import xarray as xr

from stereo_melt.coregister.tilt import (
    build_static_area_polygon_mask,
    build_static_control_mask,
)
from stereo_melt.coregister.tilt_qc import plot_static_mask_and_strip_count

from nansen import config


def _pick_stack() -> Path:
    candidates = [
        config.PROCESSED_DIR / f"nansen_stack_tilt_corrected_{config.START_TIME}_{config.END_TIME}.nc",
        config.PROCESSED_DIR / f"nansen_stack_{config.START_TIME}_{config.END_TIME}.nc",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No stack found in {config.PROCESSED_DIR}. Tried: "
        + ", ".join(p.name for p in candidates)
    )


def _open_stack(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    if "__xarray_dataarray_variable__" in ds.data_vars:
        return ds["__xarray_dataarray_variable__"]
    return ds[list(ds.data_vars)[0]]


def main() -> None:
    config.ensure_output_dirs()

    stack_path = _pick_stack()
    print(f"Loading {stack_path.name}")
    stack = _open_stack(stack_path)
    print(
        f"  dims: time={stack.sizes['time']}, "
        f"y={stack.sizes['y']}, x={stack.sizes['x']}"
    )

    print("Building static-control (LS) mask...")
    polygon = build_static_area_polygon_mask(stack, config.BEDMACHINE_NC).values
    control = build_static_control_mask(stack, shapefile_mask=polygon).values
    print(f"  control: {int(control.sum()):,} cells ({100*float(control.mean()):.1f}%)")

    out_path = config.FIGURES_DIR / "static_mask_strip_count.png"
    plot_static_mask_and_strip_count(
        stack, control, out_path, title_prefix="Nansen",
        aoi_path=config.NANSEN_AOI_SHP,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
