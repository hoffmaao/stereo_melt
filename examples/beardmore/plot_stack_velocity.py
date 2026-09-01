"""QC plot: per-epoch DEM panel grid with velocity arrows overlaid.

Run:

    python -m beardmore.plot_stack_velocity
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import xarray as xr

from stereo_melt.visualization import plot_dem_stack_with_velocity

from beardmore import config
from beardmore.run_melt import load_velocity_on_grid


def main() -> None:
    config.ensure_output_dirs()

    src = (
        config.PROCESSED_DIR
        / f"beardmore_stack_{config.START_TIME}_{config.END_TIME}.nc"
    )
    if not src.exists():
        raise SystemExit(f"Stack not found: {src}")
    print(f"Loading {src.name}...")
    ds = xr.open_dataset(src)
    payload = [v for v in ds.data_vars if v != "spatial_ref"]
    stack = ds[payload[0]]
    print(f"  dims: {dict(stack.sizes)}")

    print("Loading velocity on stack grid...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  velocity source: {vel_source}")

    out = config.FIGURES_DIR / "stack_with_velocity.png"
    print(f"Plotting -> {out}")
    plot_dem_stack_with_velocity(
        stack, vx, vy, out,
        title=f"Beardmore DEM stack — {config.START_TIME} to {config.END_TIME}",
        bad_epochs=getattr(config, "BAD_EPOCHS", ()),
        bad_strips=getattr(config, "BAD_STRIPS", ()),
    )
    print("done.")


if __name__ == "__main__":
    main()
