"""Package a PIGREAL DEM-stack twin for the B-PINN (run in the stereo_melt env).

Writes ``results/bpinn/twin_<tag>[_<variant>].npz`` with the twin stack
converted to hydrostatic thickness, the time-mean twin velocity, the
prescribed truth melt field, and the Eulerian / Lagrangian benchmarks from the
production solvers on identical inputs, so the JAX environment needs nothing
but numpy.

The twin tier is a local dataset -- the Elmer runs, the synthetic stacks and the
``run_dem_stack_melt.py`` solver driver with its scoring helpers live only on the
analysis host -- so from a clean clone this script documents the validation
recipe rather than being runnable, and exits with that message when the driver is
absent.

    $PY elmer_synth/scripts/prep_bpinn_twin.py [tag] [pert] [variant] [axis]
    e.g. multixy_pigreal multixy_bmb tilt_corrected   (variant 'clean' = noise-free stack)

The truth axis comes from the run metadata -- 'xy' for a ``multicos`` melt, whose
components carry their own axes, else the run's ``melt.axis`` -- and is recorded
in the npz as ``truth_axis``. The optional 4th argument overrides it.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

_REPO = __import__("pathlib").Path(__file__).resolve().parents[3]
sys.path.insert(0, f"{_REPO}/examples")
sys.path.insert(0, f"{_REPO}/src")
from stereo_melt.freeboard import freeboard_to_thickness  # noqa: E402

_RDS_CANDIDATES = (
    f"{_REPO}/examples/elmer_synth/scripts/run_dem_stack_melt.py",
    "/wd2/projects/stereo_melt/examples/elmer_synth/scripts/run_dem_stack_melt.py",
)

SOLVERS = {"Eulerian": "eulerian", "Lagrangian path": "lagrangian"}


def _load_rds():
    """Load the twin solver driver, preferring the copy beside this script."""
    for path in _RDS_CANDIDATES:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("rds", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(
        "run_dem_stack_melt.py not found (looked in: " + ", ".join(_RDS_CANDIDATES) + "). "
        "Packaging a twin needs the elmer_synth twin tier on the analysis host -- the Elmer runs, "
        "the synthetic DEM stacks, and run_dem_stack_melt.py with its scoring and parameter helpers "
        "-- which is a local dataset that is not part of the repository. From a clean clone this "
        "script documents the validation recipe rather than being runnable."
    )


def main() -> int:
    rds = _load_rds()
    tag = sys.argv[1] if len(sys.argv) > 1 else "multixy_pigreal"
    pert = sys.argv[2] if len(sys.argv) > 2 else "multixy_bmb"
    variant = sys.argv[3] if len(sys.argv) > 3 else "tilt_corrected"
    axis_arg = sys.argv[4] if len(sys.argv) > 4 else None
    path = f"/wd2/projects/stereo_melt/examples/elmer_synth/data/processed/e2a_stack_200m_{tag}_{variant}.nc"
    out_tag = tag if variant == "tilt_corrected" else f"{tag}_{variant}"
    h = rds.load_stack(path)
    truth, meta = rds.load_run(pert, None)
    p = rds.p
    dens = dict(rho_i=p.RHO_I, rho_w=p.RHO_W)
    vx = truth.vx.mean("time")
    vy = truth.vy.mean("time")
    if not (vx.x.equals(h.x) and vx.y.equals(h.y)):
        vx, vy = vx.interp(x=h.x, y=h.y), vy.interp(x=h.x, y=h.y)
    a_dot = xr.zeros_like(vx)
    floating = xr.DataArray(np.ones((h.sizes["y"], h.sizes["x"]), bool),
                            coords={"y": h.y, "x": h.x}, dims=("y", "x"))
    print(f"stack {dict(h.sizes)} median surface {float(h.median()):.1f} m "
          f"(freeboard of H0={p.H0:.0f} m would be {p.H0 * (1 - p.RHO_I / p.RHO_W):.1f})")
    H_obs = freeboard_to_thickness(h, d=0.0, rho_w=p.RHO_W, rho_i=p.RHO_I)
    t = pd.to_datetime(h.time.values)
    t_yr = t.year + (t.dayofyear - 1) / 365.25
    melt_meta = meta["melt"]
    axis = axis_arg or ("xy" if melt_meta.get("kind") == "multicos" else melt_meta.get("axis", "xy"))
    print(f"truth axis {axis} (melt kind {melt_meta.get('kind')!r})")
    truth2d = rds.truth_field2d(h.x.values, h.y.values, melt_meta, axis)
    bench = {}
    for name, key in SOLVERS.items():
        try:
            m = rds._one_solver(name, h, vx, vy, a_dot, floating, dens)
        except Exception as exc:  # noqa: BLE001 - benchmark is optional
            print(f"  {name} benchmark failed: {exc!r}"[:200])
            continue
        bench[f"bench_{key}"] = np.asarray(m.values, float)
        e = bench[f"bench_{key}"] - truth2d
        fin = np.isfinite(e)
        print(f"  {name}: nrmse {np.sqrt(np.mean(e[fin] ** 2)) / np.sqrt(np.mean(truth2d[fin] ** 2)):.3f}  "
              f"corr {np.corrcoef(bench[f'bench_{key}'][fin], truth2d[fin])[0, 1]:.3f}  finite {fin.mean():.2f}")
    out = f"/wd2/projects/stereo_melt/examples/elmer_synth/results/bpinn/twin_{out_tag}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, H_obs=np.asarray(H_obs.values, np.float32), x=h.x.values, y=h.y.values,
                        t_yr=np.asarray(t_yr, float), vx=np.asarray(vx.values, float),
                        vy=np.asarray(vy.values, float), truth=truth2d, truth_axis=axis,
                        H0=float(p.H0), rho_i=p.RHO_I, rho_w=p.RHO_W, **bench)
    print(f"wrote {out} | obs finite {int(np.isfinite(H_obs.values).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
