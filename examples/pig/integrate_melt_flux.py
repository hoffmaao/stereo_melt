"""Integrate our production PIG melt field -> Gt/yr, vs Shean 2019's 82-93.

Applies the Shean-matching reporting treatment to the *existing* melt product
(no re-run): integrate over the floating shelf eroded by a grounding-line buffer
(stereo_melt.flux.grounding_buffer), with an optional dh/dt-quality keep-mask,
and bracket the heavy-tailed integral as robust(median x area) / clip|cap| / raw
(stereo_melt.flux.integrate_basal_flux).

This answers: with Shean's GL buffer + a coverage/rmse quality gate, does our
REAL-data integral land near 82-93 Gt/yr, or is it still noise-blown?

Env:
    PIG_GL_BUFFER_KM   grounding-line buffer width (default 2.0)
    PIG_MIN_COUNT      min Lagrangian epoch count to keep a pixel (default 10)
    PIG_MAX_RMSE       max Lagrangian rmse (m) to keep a pixel (default 25)

Run:
    python -m pig.integrate_melt_flux [melt_nc ...]
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path

import numpy as np
import xarray as xr

from stereo_melt.flux import grounding_buffer, integrate_basal_flux

from pig import config
from pig.compare_shean2019 import _rasterize_shelf_polygon
from pig.run_melt import load_grounded_mask

GL_BUFFER_M = float(os.environ.get("PIG_GL_BUFFER_KM", "2.0")) * 1000.0
MIN_COUNT = int(os.environ.get("PIG_MIN_COUNT", "10"))
MAX_RMSE = float(os.environ.get("PIG_MAX_RMSE", "25"))
CAP_MYR = 250.0

DEFAULTS = [
    "pig_melt_250m_is2ctempo_2010-01-01_2024-01-10.nc",
    "pig_melt_250m_is2ctempo_2018-10-01_2024-01-10.nc",
    "pig_melt_250m_is2ctempo_2011-02-04_2013-10-31.nc",  # W1, fair Shean overlap
]


def _row(label: str, b: dict) -> str:
    if not b.get("n"):
        return f"    {label:22s}  (empty domain)"
    return (
        f"    {label:22s} area={b['area_km2']:6.0f}km2  med={b['median_myr']:+5.1f} "
        f"mean={b['mean_myr']:+7.1f}  | Gt/yr robust={b['gt_robust']:6.1f} "
        f"clip|{CAP_MYR:.0f}|={b['gt_clip']:6.1f} raw={b['gt_raw']:6.1f}  "
        f"| top5%={b['frac_top5']*100:3.0f}% deepest={b['deepest_myr']:+5.0f}"
    )


def main(argv: list[str]) -> None:
    files = argv or DEFAULTS
    cache = {}
    for fname in files:
        p = Path(fname)
        if not p.is_absolute():
            p = config.RESULTS_DIR / fname
        if not p.exists():
            print(f"[skip] {p.name} (missing)")
            continue
        ds = xr.open_dataset(p)
        xs, ys = ds["x"].values, ds["y"].values
        res = abs(float(xs[1] - xs[0]))
        floating = ds["floating_mask"].values.astype(bool)

        key = (xs.size, ys.size, float(xs[0]), float(ys[0]))
        if key not in cache:
            like = ds["melt_rate_eulerian"]
            grounded = load_grounded_mask(like).values.astype(bool)
            buf_floating = grounding_buffer(floating, grounded, GL_BUFFER_M, res)
            poly = _rasterize_shelf_polygon(like)
            buf_poly = (grounding_buffer(poly.values, grounded, GL_BUFFER_M, res)
                        if poly is not None else None)
            cache[key] = (buf_floating, buf_poly)
        buf_floating, buf_poly = cache[key]

        # dh/dt quality keep-mask (coverage + Lagrangian rmse); applies to both
        # solvers as a coverage proxy + drops the worst-fit pixels.
        quality = None
        if "lagrangian_count" in ds and "lagrangian_rmse" in ds:
            cnt = ds["lagrangian_count"].values
            rms = ds["lagrangian_rmse"].values
            quality = (np.nan_to_num(cnt, nan=0) >= MIN_COUNT) & (np.nan_to_num(rms, nan=1e9) <= MAX_RMSE)

        print(f"\n=== {p.name} ===")
        print(f"    window {ds.attrs.get('window_start','?')}..{ds.attrs.get('window_end','?')}  "
              f"vel={ds.attrs.get('velocity_source','?')}")
        print(f"    GL buffer={GL_BUFFER_M/1e3:.1f}km  quality: count>={MIN_COUNT} & rmse<={MAX_RMSE}m")
        for dname, dmask in (("floating-buf", buf_floating), ("shelfpoly-buf", buf_poly)):
            if dmask is None:
                continue
            print(f"  [{dname}]")
            for solver, var in (("Eulerian", "melt_rate_eulerian"), ("Lagrangian", "melt_rate_lagrangian")):
                if var not in ds:
                    continue
                b_raw = integrate_basal_flux(ds[var], dmask, res, cap_myr=CAP_MYR)
                print(_row(f"{solver}", b_raw))
                if quality is not None:
                    b_q = integrate_basal_flux(ds[var], dmask, res, cap_myr=CAP_MYR, quality=quality)
                    print(_row(f"{solver}+quality", b_q))

    print(f"\n  Shean et al. 2019: 82-93 Gt/yr over the PIG shelf (his max channel ~250 m/yr).")
    print(f"  robust=median x area (tail-free floor); clip caps |melt|<{CAP_MYR:.0f}; raw=unguarded sum.")


if __name__ == "__main__":
    main(sys.argv[1:])
