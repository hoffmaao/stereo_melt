"""Standard QC plot: PIG LS mask + per-pixel strip count.

Two-panel figure produced by
:func:`stereo_melt.coregister.tilt_qc.plot_static_mask_and_strip_count`:

  * left  — static-control (LS) mask fed to the per-epoch tilt LSQ
  * right — per-pixel REMA strip count across the stack window

Run:

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -m pig.plot_static_mask
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

from pig import config


def _pick_stack(stack_prefix: str = "pig_stack") -> Path:
    # Prefer the config-matched stack; fall back to the most recent matching
    # stack on disk if the config window has been temporarily retargeted
    # (e.g. the 2018-only ATM integration window of May 2026).
    candidates = [
        config.PROCESSED_DIR / f"{stack_prefix}_tilt_corrected_{config.START_TIME}_{config.END_TIME}.nc",
        config.PROCESSED_DIR / f"{stack_prefix}_{config.START_TIME}_{config.END_TIME}.nc",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: any stack file matching the prefix
    tilt_corrected = sorted(config.PROCESSED_DIR.glob(f"{stack_prefix}_tilt_corrected_*.nc"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    if tilt_corrected:
        print(f"  (config window {config.START_TIME}_{config.END_TIME} has no stack; "
              f"falling back to most recent on disk)")
        return tilt_corrected[0]
    raw = sorted(config.PROCESSED_DIR.glob(f"{stack_prefix}_*.nc"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if raw:
        return raw[0]
    raise FileNotFoundError(
        f"No stack found in {config.PROCESSED_DIR} matching {stack_prefix}_*.nc."
    )


def _open_stack(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    if "__xarray_dataarray_variable__" in ds.data_vars:
        return ds["__xarray_dataarray_variable__"]
    return ds[list(ds.data_vars)[0]]


def main(res_override: float | None = None) -> None:
    config.ensure_output_dirs()

    if res_override is not None:
        stack_prefix = f"pig_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "pig_stack"
        out_suffix = ""

    stack_path = _pick_stack(stack_prefix=stack_prefix)
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

    out_path = config.FIGURES_DIR / f"static_mask_strip_count{out_suffix}.png"
    plot_static_mask_and_strip_count(
        stack, control, out_path, title_prefix="PIG",
        aoi_path=config.PIG_AOI_SHP,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help=(
            "Resolution variant (meters). When set, looks up "
            "pig_stack_<N>m*_<start>_<end>.nc and writes "
            "static_mask_strip_count_<N>m.png."
        ),
    )
    args = parser.parse_args()
    main(res_override=args.res)
