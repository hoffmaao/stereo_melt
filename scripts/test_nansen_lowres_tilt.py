"""Low-resolution Nansen tilt_fit test — exercises the GPU code path quickly.

Loads the new wider-AOI Nansen stack, coarsens by N×, builds the static-control
mask, and runs fit_tilt_stack on cupy. Verifies the searchsorted row-scaling
fix without waiting for the full-resolution corrections + LSMR (which take ~30
min each).

Usage:
    CUDA_VISIBLE_DEVICES=0 STEREO_MELT_BACKEND=cupy \
      python scripts/test_nansen_lowres_tilt.py [coarsen_factor]
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

# Point pyproj at the env-local proj.db (base conda's is corrupt)
os.environ.setdefault("PROJ_DATA", "/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj")
os.environ.setdefault("PROJ_LIB", "/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj")

import numpy as np
import xarray as xr

from stereo_melt.coregister.tilt import (
    build_static_area_polygon_mask,
    build_static_control_mask,
    fit_tilt_stack,
)

ROOT = Path("/wd2/projects/stereo_melt")
STACK_NC = ROOT / "nansen/processed/nansen_stack_2019-01-01_2023-03-01.nc"
BEDMACHINE = ROOT / "data/bedmachine/BedMachineAntarctica-v3.nc"


def main(coarsen: int = 8) -> None:
    print(f"backend: STEREO_MELT_BACKEND={os.environ.get('STEREO_MELT_BACKEND', '(unset)')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

    print(f"\nLoading wider-AOI Nansen stack: {STACK_NC.name}")
    ds = xr.open_dataset(STACK_NC)
    # Pick the actual stack DataArray (not spatial_ref scalar)
    candidates = [v for v in ds.data_vars if "time" in ds[v].dims]
    if not candidates:
        raise SystemExit(f"no time-bearing data_var in {STACK_NC.name}")
    da = ds[candidates[0]]
    print(f"  variable: {candidates[0]}")
    print(f"  full dims: {dict(da.sizes)}")

    print(f"\nCoarsening by {coarsen}× in y and x...")
    coarse = da.coarsen(y=coarsen, x=coarsen, boundary="trim").mean()
    coarse = coarse.transpose("time", "y", "x")
    print(f"  coarsened dims: {dict(coarse.sizes)}  → {coarse.sizes['y']*coarse.sizes['x']*coarse.sizes['time']:,} cells")
    print(f"  finite frac: {float(np.isfinite(coarse.values).mean()):.3f}")

    print("\nBuilding polygon + static-control masks...")
    poly = build_static_area_polygon_mask(coarse, BEDMACHINE)
    print(f"  polygon: {int(poly.sum())}/{poly.size} ({100*float(poly.mean()):.1f}%)")
    static = build_static_control_mask(coarse, shapefile_mask=poly)
    print(f"  static control: {int(static.sum())}/{static.size} ({100*float(static.mean()):.1f}%)")

    print("\nRunning fit_tilt_stack on cupy backend...")
    t0 = time.time()
    params, stack_corr = fit_tilt_stack(
        coarse,
        control_mask=static,
        Eint=10.0, Edhdt=10.0, Ex=1e-3, Ey=1e-3, Ez=0.3,
        robust=True, robust_max_iter=5,
        solver="lsmr", lsmr_atol=1e-4, lsmr_maxiter=200,
        min_width=10000.0,
    )
    dt = time.time() - t0
    print(f"\nfit_tilt_stack returned in {dt:.1f}s")
    print(f"  params dims: {dict(params.sizes)}")
    print(f"  alpha_x range: [{float(params['alpha_x'].min()):+.2e}, {float(params['alpha_x'].max()):+.2e}] /m")
    print(f"  alpha_y range: [{float(params['alpha_y'].min()):+.2e}, {float(params['alpha_y'].max()):+.2e}] /m")
    print(f"  alpha_z range: [{float(params['alpha_z'].min()):+.3f}, {float(params['alpha_z'].max()):+.3f}] m")
    print(f"\nSUCCESS — GPU LSMR + IRLS path is working at coarsen={coarsen}×.")


if __name__ == "__main__":
    coarsen = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(coarsen=coarsen)
