"""Re-run only the Lagrangian path-integration melt rate to verify velocity source.

Forces ``BEARDMORE_VELOCITY=measures`` so there is no ambiguity. Saves to a
``*_lagrangian_measures_verify.nc`` so we can byte-compare against the existing
``beardmore_melt_<window>.nc`` ``melt_rate_lagrangian`` field.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

os.environ["BEARDMORE_VELOCITY"] = "measures"  # force-MEaSUREs

if "STEREO_MELT_BACKEND" not in os.environ:
    try:
        import cupy as _cp  # noqa: F401
        os.environ["STEREO_MELT_BACKEND"] = "cupy"
    except ImportError:
        pass

import numpy as np
import xarray as xr

from stereo_melt.backend import backend as _BACKEND
from stereo_melt.melt import lagrangian_melt_rate

from beardmore import config
from beardmore.run_melt import (
    load_floating_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
)


def main() -> None:
    config.ensure_output_dirs()
    print(f"stereo_melt backend: {_BACKEND}")
    print(f"BEARDMORE_VELOCITY env: {os.environ.get('BEARDMORE_VELOCITY')}")

    print("Loading stack...")
    stack = load_stack()
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    floating = load_floating_mask(stack)
    print(f"  floating-ice fraction: {float(floating.mean()):.3f}")
    stack = stack.where(floating)

    print("Loading velocity (forced MEaSUREs)...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  vx range: {float(vx.min()):.1f} .. {float(vx.max()):.1f} m/yr")
    print(f"  vy range: {float(vy.min()):.1f} .. {float(vy.max()):.1f} m/yr")

    print("Loading SMB...")
    a_dot = load_smb_on_grid(stack)

    print("Running Lagrangian path-integration solver...")
    t0 = time.time()
    lagr = lagrangian_melt_rate(
        stack, vx, vy,
        a_dot=a_dot,
        dt_yr=0.05,
        seed_stride=2,
        pairs="all",
        min_dt_yr=2.0 / 12.0,
    )
    elapsed = time.time() - t0
    mr = lagr.melt_rate.where(floating)
    print(
        f"  elapsed={elapsed:.1f} s  "
        f"median={float(mr.median()):+.2f}  "
        f"IQR=[{float(mr.quantile(0.25)):+.2f}, {float(mr.quantile(0.75)):+.2f}]  "
        f"abs_max={float(np.abs(mr).max()):.1f} m ice/yr"
    )

    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_lagrangian_measures_verify_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Saving -> {out_nc}")
    ds_out = xr.Dataset(
        {
            "melt_rate_lagrangian": mr.astype("float32"),
            "floating_mask": floating.astype("uint8"),
        },
        attrs={
            "shelf": "Beardmore",
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "velocity_source": vel_source,
            "velocity_env": os.environ.get("BEARDMORE_VELOCITY", ""),
            "seed_stride": 2,
            "dt_yr": 0.05,
            "min_dt_yr": 2.0 / 12.0,
            "pairs": "all",
        },
    )
    ds_out.to_netcdf(out_nc)
    print(f"  wrote {out_nc}")


if __name__ == "__main__":
    main()
