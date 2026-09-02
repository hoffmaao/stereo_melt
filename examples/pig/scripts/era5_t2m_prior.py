"""ERA5 2 m air temperature over the PIG shelf -> the fluidity-inversion prior.

The inversion's prior fluidity is ``A0 = rate_factor(T0)`` with T0 CONSTANT
(2026-07-30 directive): the interior of PIG has no observational constraint
that would justify a spatially varying thermal prior, so the prior mean is a
single temperature and every spatial structure in the recovered fluidity is
paid for by the data. What this script fixes is the *value* — T0 was a
hardcoded 258 K with no provenance; here it is the window- and area-mean
surface temperature over the same shelf domain the inversion meshes.

Region and mask come from ``pig_eta_inv_inputs.npz``, so the temperature is
averaged over exactly the cells being inverted, in exactly the window the
thickness/velocity were sampled over.

Monthly means (not hourly): a constant prior needs a climatological mean, and
``reanalysis-era5-single-levels-monthly-means`` costs ~1/700 of the hourly
request the IBE cube uses.

Caveat, stated once: 2 m air temperature is the SURFACE boundary condition,
so it is colder than the depth-averaged ice column (the base sits near the
pressure-melting point). The prior is therefore stiffer than the true
depth-averaged rheology; that offset is close to spatially uniform and is
absorbed by a near-constant shift in theta, which the Whittle-Matern mass
term penalises but the velocity data can correct.

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY examples/pig/scripts/era5_t2m_prior.py [--overwrite]
"""
import argparse
import json
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _env_proj

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
BASIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from pyproj import Transformer  # noqa: E402

NPZ = f"{BASIN}/processed/pig_eta_inv_inputs.npz"
CACHE_DIR = f"{BASIN}/data/climate"
OUT_JSON = f"{BASIN}/processed/pig_eta_prior_T0.json"
PAD_DEG = 0.5


def months_between(t0: str, t1: str):
    """Whole (year, month) pairs covered by [t0, t1)."""
    y0, m0 = int(t0[:4]), int(t0[5:7])
    y1, m1 = int(t1[:4]), int(t1[5:7])
    out = []
    y, m = y0, m0
    while (y, m) < (y1, m1):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch(region, ym, path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        print(f"[era5] cache hit {path}")
        return path
    import cdsapi

    years = sorted({str(y) for y, _ in ym})
    months = sorted({f"{m:02d}" for _, m in ym})
    print(f"[era5] monthly-mean 2m_temperature\n"
          f"       region N={region[0]:.2f} W={region[1]:.2f} "
          f"S={region[2]:.2f} E={region[3]:.2f}\n"
          f"       years {years} months {months[0]}..{months[-1]}\n"
          f"       -> {path}", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": "2m_temperature",
            "year": years,
            "month": months,
            "time": "00:00",
            "area": list(region),
            "format": "netcdf",
        },
        path,
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    x, y, mask = d["x"], d["y"], d["mask"].astype(bool)
    t0, t1 = str(d["t0"]), str(d["t1"])
    Xg, Yg = np.meshgrid(x, y)
    xs, ys = Xg[mask], Yg[mask]
    print(f"[domain] {mask.sum()} shelf cells, window {t0}..{t1}", flush=True)

    tf = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(xs, ys)
    region = [
        float(np.ceil((lat.max() + PAD_DEG) * 4) / 4),
        float(np.floor((lon.min() - PAD_DEG) * 4) / 4),
        float(np.floor((lat.min() - PAD_DEG) * 4) / 4),
        float(np.ceil((lon.max() + PAD_DEG) * 4) / 4),
    ]

    ym = months_between(t0, t1)
    path = os.path.join(CACHE_DIR, f"era5_t2m_monthly_{t0}_{t1}.nc")
    fetch(region, ym, path, overwrite=args.overwrite)

    ds = xr.open_dataset(path)
    ren = {c: n for c, n in (("valid_time", "time"), ("lat", "latitude"),
                             ("lon", "longitude")) if c in ds.coords
           and n not in ds.coords}
    ds = ds.rename(ren) if ren else ds
    t2m = ds["t2m"]
    for extra in ("expver", "number"):
        if extra in t2m.dims:
            t2m = t2m.mean(extra)
    tv = t2m["time"].values.astype("datetime64[ns]")
    # floor to month starts: months_between is month-granular (partial first
    # month in, partial last month out), so the stamp filter must be too
    in_window = (tv >= np.datetime64(t0[:7])) & (tv < np.datetime64(t1[:7]))
    t2m = t2m.isel(time=np.flatnonzero(in_window))
    if t2m.sizes["time"] != len(ym):
        raise SystemExit(
            f"{path}: {t2m.sizes['time']} monthly means fall inside "
            f"[{t0[:7]}, {t1[:7]}) but the window spans {len(ym)} months")
    t2m_mean = t2m.mean("time")            # window-mean field (K)

    la = np.asarray(t2m_mean["latitude"].values, float)
    lo = np.asarray(t2m_mean["longitude"].values, float)
    vals = np.asarray(t2m_mean.transpose("latitude", "longitude").values, float)
    if la[0] > la[-1]:                     # ERA5 ships latitude descending
        la, vals = la[::-1], vals[::-1, :]

    # Sample the window-mean field at every inverted cell, then average. The
    # 250 m EPSG:3031 cells are equal-area, so a plain mean over the mask IS
    # the area-weighted mean over the inverted domain.
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator((la, lo), vals, bounds_error=False,
                                     fill_value=None)
    Tcell = interp(np.column_stack([lat, lon]))
    T0 = float(np.mean(Tcell))

    print(f"[t2m] {t2m.sizes['time']} monthly means over "
          f"{len(la)}x{len(lo)} ERA5 cells")
    print(f"[t2m] over the inverted domain: mean {T0:.2f} K "
          f"({T0 - 273.15:+.2f} C), "
          f"p10 {np.percentile(Tcell, 10):.2f} p90 "
          f"{np.percentile(Tcell, 90):.2f} K "
          f"(spatial spread {Tcell.max() - Tcell.min():.2f} K)")
    print(f"[t2m] seasonal range of the domain mean: "
          f"{float(t2m.mean(('latitude', 'longitude')).min()):.2f} .. "
          f"{float(t2m.mean(('latitude', 'longitude')).max()):.2f} K")

    rec = dict(T0_K=T0, source="ERA5 monthly-mean 2m_temperature",
               cache=path, region_NWSE=region, t0=t0, t1=t1,
               n_months=int(t2m.sizes["time"]), n_cells=int(mask.sum()),
               domain_npz=NPZ,
               spatial_spread_K=float(Tcell.max() - Tcell.min()))
    with open(OUT_JSON, "w") as f:
        json.dump(rec, f, indent=2)
    ds.close()
    print(f"wrote {OUT_JSON}\n  T0_K = {T0:.2f}   (was a hardcoded 258.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
