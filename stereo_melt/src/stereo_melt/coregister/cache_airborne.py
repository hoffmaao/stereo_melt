# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""IceBridge airborne lidar (ATM ILATM2 / LVIS ILVIS2) per-strip cache.

Mirrors :mod:`stereo_melt.coregister.cache_cs2` but for the NSIDC
IceBridge airborne products. NSIDC CMR provides discovery; the granule
download is Earthdata-authenticated HTTPS via ``~/.netrc``. Per-strip
filtered CSVs are written in the ASP-ready
``easting,northing,h_mean`` schema so
:func:`stereo_melt.coregister.asp.align_strip_with_asp` can consume
them as additional control sources alongside IS2 / CS2 / rock.

**Scope locks (Stage 1 of plumbing):**

- Products: ILATM2 (small-footprint ATM Icessn) + ILVIS2 (LVIS L2
  centroid). Both as v002 ASCII (CSV / TXT) by default.
- Filter regime: same as IS2/CS2 — BedMachine ``mask == 2`` (grounded
  ice) ∩ smoothed-MEaSUREs ``|v| < max_speed_myr`` m/yr.
- Per-strip temporal half-window: 90 days. IceBridge campaigns are
  October-November; choose the half-window to bracket the closest
  campaign, not to ensure overlap with every strip.

This module is the basin-agnostic library. Per-basin wrappers under
``<basin>/cache_atm.py`` and ``<basin>/cache_lvis.py`` pass their
config-supplied paths to :func:`run_cache_airborne_main`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
from pointCollection.data import data as PCData
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

from ..io.altimetry import from_atm, from_lvis
from ..io.nsidc_cmr import (
    download_nsidc_granule,
    make_session,
    query_nsidc_granules,
)


_DATE_RE = re.compile(r"_(\d{8})_")
_LVIS_NAME_RE = re.compile(r"ILVIS2_AQ(\d{4})_(\d{4})_")

# Descriptors for the two products. ``parse`` is the reader (returns a
# :class:`PCData`); ``date_from_name`` extracts the granule UTC date.
PRODUCT_INFO: dict[str, dict] = {
    "atm": {
        "short_name": "ILATM2",
        "label": "atm_ilatm2",
        "parse": from_atm,
        "name_to_date": lambda name: _ilatm2_name_to_date(name),
        "h5_glob": "ILATM2_*.csv",
    },
    "lvis": {
        "short_name": "ILVIS2",
        "label": "lvis_ilvis2",
        "parse": from_lvis,
        "name_to_date": lambda name: _ilvis2_name_to_date(name),
        "h5_glob": "ILVIS2_*.TXT",
    },
}


def _ilatm2_name_to_date(name: str) -> pd.Timestamp | None:
    m = _DATE_RE.search(name)
    if not m:
        return None
    return pd.Timestamp(m.group(1))


def _ilvis2_name_to_date(name: str) -> pd.Timestamp | None:
    m = _LVIS_NAME_RE.search(name)
    if not m:
        return None
    year = m.group(1)
    mmdd = m.group(2)
    return pd.Timestamp(f"{year}-{mmdd[:2]}-{mmdd[2:4]}")


# ---------------------------------------------------------------------------
# Filtering: BedMachine grounded mask + slow-velocity (mirrors cache_cs2)
# ---------------------------------------------------------------------------


def filter_grounded_slow_velocity(
    pc: PCData,
    bedmachine_path: Path,
    velocity_path: Path,
    *,
    grounded_mask_value: int = 2,
    max_speed_myr: float = 10.0,
    vel_smooth_sigma_m: float = 2_000.0,
    displacement_budget_m: float | None = None,
    delta_t_years: float | None = None,
) -> PCData:
    r"""Keep points over grounded ice ∩ slow-flowing.

    Two filtering modes (mutually exclusive — caller picks):

    - **Fixed speed threshold** (default): keep points with smoothed
      ``|v| < max_speed_myr``. Inherited from the IS2/CS2 catalogue
      regime where Δt is ~1 year and the implicit displacement budget
      is ``max_speed_myr * 1 yr ≈ 10 m``.
    - **Displacement-budget rule** (Shean 2019, ``|v|·|Δt| ≤ B``): pass
      ``displacement_budget_m`` and ``delta_t_years`` to switch to a
      time-aware threshold ``max_speed = B / Δt``. For airborne
      campaigns Δt is the granule-to-strip gap in years, so a ±100 d
      window admits ``|v| ≤ 10 m / (100/365 yr) ≈ 37 m/yr``. Mirrors
      the IS2 ATL06 filter in
      :func:`stereo_melt.coregister.reference.download_icesat2_data`.
    """
    if pc.size == 0:
        return pc

    east = np.asarray(pc.x, dtype=np.float64)
    north = np.asarray(pc.y, dtype=np.float64)
    pad = max(vel_smooth_sigma_m * 4, 5_000.0)
    bbox = (east.min() - pad, east.max() + pad, north.min() - pad, north.max() + pad)

    def _crop(ds, var):
        if ds["y"].values[0] > ds["y"].values[-1]:
            sub = ds[var].sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2]))
        else:
            sub = ds[var].sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3]))
        return sub.load()

    def _sample(arr, x_coords, y_coords, e, n):
        ix = np.searchsorted(x_coords, e).clip(0, len(x_coords) - 1)
        if y_coords[0] > y_coords[-1]:
            iy = (len(y_coords) - 1) - np.searchsorted(y_coords[::-1], n).clip(0, len(y_coords) - 1)
        else:
            iy = np.searchsorted(y_coords, n).clip(0, len(y_coords) - 1)
        return arr[iy, ix]

    vel_ds = xr.open_dataset(velocity_path)
    vx_sub = _crop(vel_ds, "VX")
    vy_sub = _crop(vel_ds, "VY")
    vx_arr = np.where(np.isfinite(vx_sub.values), vx_sub.values, 0.0)
    vy_arr = np.where(np.isfinite(vy_sub.values), vy_sub.values, 0.0)
    speed = np.hypot(vx_arr, vy_arr)
    dx = abs(float(vx_sub["x"].values[1] - vx_sub["x"].values[0]))
    dy = abs(float(vx_sub["y"].values[1] - vx_sub["y"].values[0]))
    sigma_pix = (vel_smooth_sigma_m / dy, vel_smooth_sigma_m / dx)
    speed_smooth = gaussian_filter(speed, sigma=sigma_pix, mode="nearest")
    pt_speed = _sample(speed_smooth, vx_sub["x"].values, vx_sub["y"].values, east, north)

    bm_ds = xr.open_dataset(bedmachine_path)
    bm_sub = _crop(bm_ds, "mask")
    pt_mask = _sample(bm_sub.values, bm_sub["x"].values, bm_sub["y"].values, east, north)

    # Resolve effective speed threshold.
    if displacement_budget_m is not None and delta_t_years is not None and delta_t_years > 0:
        eff_speed = float(displacement_budget_m) / float(delta_t_years)
        speed_label = (f"speed<{eff_speed:.1f}m/yr (budget={displacement_budget_m:.0f}m,"
                       f" Δt={delta_t_years:.2f}yr)")
    else:
        eff_speed = float(max_speed_myr)
        speed_label = f"speed<{eff_speed}m/yr"

    keep = (pt_mask == grounded_mask_value) & (pt_speed < eff_speed) & np.isfinite(pc.h)
    n_in = pc.size
    n_keep = int(keep.sum())
    print(f"  airborne grounded+slow-vel filter: {n_keep}/{n_in} kept "
          f"(grounded={int((pt_mask==grounded_mask_value).sum())}, "
          f"{speed_label}={int((pt_speed<eff_speed).sum())})")

    if n_keep == 0:
        return PCData().from_dict({"x": np.array([]), "y": np.array([]), "h": np.array([])})

    sub_dict = {f: np.asarray(getattr(pc, f))[keep] for f in pc.fields}
    return PCData().from_dict(sub_dict)


# ---------------------------------------------------------------------------
# Strip discovery + bbox utils (shared with cache_cs2 in spirit)
# ---------------------------------------------------------------------------


def _strip_bbox(strip_path: Path):
    import rasterio
    with rasterio.open(strip_path) as src:
        bx = src.bounds
    return box(bx.left, bx.bottom, bx.right, bx.top)


def _strip_bbox_lonlat(strip_path: Path, buffer_m: float = 25_000.0):
    r"""Return ((lat_lo, lat_hi), (lon_lo, lon_hi)) for a strip footprint
    after padding the EPSG:3031 bbox by ``buffer_m``. Edge-sampled to
    handle polar-projection distortion."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(strip_path) as src:
        b = src.bounds
    minx, miny, maxx, maxy = b.left - buffer_m, b.bottom - buffer_m, b.right + buffer_m, b.top + buffer_m
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    n = 25
    xs_top = np.linspace(minx, maxx, n)
    xs_bot = xs_top
    ys_left = np.linspace(miny, maxy, n)
    ys_right = ys_left
    ex = np.concatenate([xs_top, [maxx] * n, xs_bot[::-1], [minx] * n])
    ey = np.concatenate([[miny] * n, ys_right, [maxy] * n, ys_left[::-1]])
    lons, lats = transformer.transform(ex, ey)
    return (float(np.min(lats)), float(np.max(lats))), (float(np.min(lons)), float(np.max(lons)))


def _aoi_bbox_lonlat(aoi_polygon, buffer_m: float = 50_000.0):
    r"""AOI lat/lon bbox for a single CMR query covering the whole basin."""
    from pyproj import Transformer
    minx, miny, maxx, maxy = aoi_polygon.bounds
    minx -= buffer_m; maxx += buffer_m; miny -= buffer_m; maxy += buffer_m
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    n = 25
    xs_top = np.linspace(minx, maxx, n)
    ys_left = np.linspace(miny, maxy, n)
    ex = np.concatenate([xs_top, [maxx] * n, xs_top[::-1], [minx] * n])
    ey = np.concatenate([[miny] * n, ys_left, [maxy] * n, ys_left[::-1]])
    lons, lats = transformer.transform(ex, ey)
    return (float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats)))


def find_in_window_strips(
    strips_dir: Path,
    start_time: str,
    end_time: str,
    *,
    aoi_polygon=None,
) -> list[tuple[Path, pd.Timestamp]]:
    """Same convention as :mod:`cache_cs2.find_in_window_strips`."""
    strips_dir = Path(strips_dir)
    t_start = pd.Timestamp(start_time)
    t_end = pd.Timestamp(end_time)
    todo = []
    for p in sorted(strips_dir.glob("SETSM_*.tif")):
        if p.stem.endswith(("_matchtag", "_bitmask")):
            continue
        m = _DATE_RE.search(p.name)
        if not m:
            continue
        t = pd.Timestamp(m.group(1))
        if not (t_start <= t < t_end):
            continue
        sb = _strip_bbox(p)
        if aoi_polygon is not None and not sb.intersects(aoi_polygon):
            continue
        todo.append((p, t))
    return sorted(todo, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# CMR + download (shared granule store)
# ---------------------------------------------------------------------------


def discover_local_granules(
    granule_dir: Path,
    short_name: str,
    start_time: str,
    end_time: str,
) -> list[Path]:
    r"""Find airborne-lidar granules on disk in ``[start_time, end_time)``."""
    glob_pat = PRODUCT_INFO[short_name.lower().replace("ilatm2", "atm").replace("ilvis2", "lvis")]["h5_glob"] \
        if short_name.lower() in {"ilatm2", "ilvis2"} else f"{short_name}_*"
    if "atm" in short_name.lower():
        info = PRODUCT_INFO["atm"]
    elif "lvis" in short_name.lower():
        info = PRODUCT_INFO["lvis"]
    else:
        raise ValueError(f"unknown short_name {short_name}")
    granule_dir = Path(granule_dir)
    granule_dir.mkdir(parents=True, exist_ok=True)
    on_disk = sorted(granule_dir.glob(info["h5_glob"]))
    if not on_disk:
        return []
    t0 = pd.Timestamp(start_time)
    t1 = pd.Timestamp(end_time)
    keep = []
    for p in on_disk:
        t = info["name_to_date"](p.name)
        if t is None:
            continue
        if t0 <= t < t1:
            keep.append(p)
    return keep


def prefetch_airborne_granules(
    *,
    source: str,
    aoi_polygon,
    start_time: str,
    end_time: str,
    granule_dir: Path,
    workers: int = 4,
    aoi_buffer_m: float = 50_000.0,
    dry_run: bool = False,
    strip_dates: list[pd.Timestamp] | None = None,
    strip_match_window_days: int | None = None,
) -> dict:
    r"""Query NSIDC CMR for granules over the basin AOI + window, then
    download missing granules to the shared store.

    If ``strip_dates`` is supplied, the CMR result set is further
    filtered to granules whose acquisition date lies within
    ±``strip_match_window_days`` of any strip date — cuts download volume
    sharply for narrow per-strip windows (the typical case). Without
    that pair, every CMR-returned granule is downloaded (broad mode,
    useful for a one-time basin-wide warm-up of the shared store).

    Returns a counts dict: ``{"matched", "have", "downloaded", "bytes"}``.
    """
    if source not in PRODUCT_INFO:
        raise ValueError(f"source must be 'atm' or 'lvis', got {source!r}")
    info = PRODUCT_INFO[source]
    short_name = info["short_name"]

    granule_dir = Path(granule_dir)
    granule_dir.mkdir(parents=True, exist_ok=True)

    # Pad the basin bbox a bit — IceBridge tracks rarely fly directly over
    # the calving front, often slightly inland or further offshore.
    bbox = _aoi_bbox_lonlat(aoi_polygon, buffer_m=aoi_buffer_m)
    print(f"  CMR query: {short_name} bbox={bbox}  {start_time}..{end_time}")

    granules = query_nsidc_granules(
        short_name,
        bounding_box=bbox,
        temporal=(f"{start_time}T00:00:00Z", f"{end_time}T00:00:00Z"),
    )
    print(f"  CMR returned {len(granules)} {short_name} granules")

    if strip_dates and strip_match_window_days is not None:
        sorted_dates = sorted(pd.Timestamp(d) for d in strip_dates)
        half = pd.Timedelta(days=strip_match_window_days)
        kept = []
        for g in granules:
            t = info["name_to_date"](g["name"])
            if t is None:
                continue
            if any(abs(t - sd) <= half for sd in sorted_dates):
                kept.append(g)
        print(f"  per-strip ±{strip_match_window_days}d match: "
              f"{len(kept)}/{len(granules)} granules retained")
        granules = kept

    counts = {"matched": len(granules), "have": 0, "downloaded": 0, "bytes": 0}
    if not granules:
        return counts

    # Download (parallel). Earthdata HTTPS handles concurrent connections
    # but we keep workers low to avoid rate-limiting.
    def _one(g):
        name = g["name"]
        out = granule_dir / name
        if out.exists():
            return ("have", 0)
        if dry_run:
            return ("downloaded", 0)
        try:
            sess = make_session()
            download_nsidc_granule(g["url"], out, session=sess, overwrite=False)
            return ("downloaded", out.stat().st_size if out.exists() else 0)
        except Exception as exc:
            print(f"  !! download failed {name}: {exc}")
            return ("failed", 0)

    n_done = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{source}dl") as ex:
        futures = [ex.submit(_one, g) for g in granules]
        for fut in as_completed(futures):
            try:
                kind, sz = fut.result()
            except Exception as exc:
                print(f"  !! future raised: {exc}")
                continue
            if kind == "have":
                counts["have"] += 1
            elif kind == "downloaded":
                counts["downloaded"] += 1
                counts["bytes"] += sz
            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.monotonic() - t0
                print(f"  [{source}] {n_done}/{len(granules)} done "
                      f"({counts['downloaded']} new, {counts['bytes']/1e6:.1f} MB, "
                      f"{elapsed/60:.1f} min)", flush=True)

    print(f"  {source} prefetch done: have={counts['have']} "
          f"new={counts['downloaded']}  ({counts['bytes']/1e6:.1f} MB)")
    return counts


# ---------------------------------------------------------------------------
# Per-strip caching
# ---------------------------------------------------------------------------


def cache_one_strip_airborne(
    strip_path: Path,
    center_time: pd.Timestamp,
    *,
    source: str,
    cache_dir: Path,
    granule_dir: Path,
    bedmachine_path: Path,
    velocity_path: Path,
    time_window_days: int = 90,
    strip_crop_buffer_m: float = 5_000.0,
    overwrite: bool = False,
    displacement_budget_m: float | None = 10.0,
) -> str:
    r"""Per-strip filter + cache. Writes ``<cache_dir>/<source>_filtered_<dem_id>.csv``
    in the ASP-ready ``easting,northing,h_mean`` schema. Returns
    ``ok|skip|fail|empty``."""
    if source not in PRODUCT_INFO:
        raise ValueError(f"source must be 'atm' or 'lvis', got {source!r}")
    info = PRODUCT_INFO[source]

    dem_id = strip_path.stem
    expected = Path(cache_dir) / f"{source}_filtered_{dem_id}.csv"
    if expected.exists() and not overwrite:
        print(f"⏭  {dem_id}: {source} cache hit, skipping (use --overwrite)")
        return "skip"

    print(f"\n=== {center_time.date()}  {dem_id}  [{source}]  ===")
    t0 = (center_time - pd.Timedelta(days=time_window_days)).strftime("%Y-%m-%d")
    t1 = (center_time + pd.Timedelta(days=time_window_days)).strftime("%Y-%m-%d")
    aoi = _strip_bbox(strip_path)

    granule_dir = Path(granule_dir)
    granules = []
    for p in granule_dir.glob(info["h5_glob"]):
        t = info["name_to_date"](p.name)
        if t is None:
            continue
        if pd.Timestamp(t0) <= t < pd.Timestamp(t1):
            granules.append(p)
    granules = sorted(granules)
    if not granules:
        print(f"!! no {source} granules in ±{time_window_days}d for {dem_id}")
        return "fail"
    print(f"  {len(granules)} {source} granules in temporal window")

    parts: list[PCData] = []
    for g in granules:
        try:
            pc = info["parse"](g)
        except Exception as exc:
            print(f"  !! parse failed on {g.name}: {exc}")
            traceback.print_exc()
            continue
        if pc.size == 0:
            continue
        e = np.asarray(pc.x); n = np.asarray(pc.y)
        bx = aoi.bounds
        in_box = (
            (e >= bx[0] - strip_crop_buffer_m) & (e <= bx[2] + strip_crop_buffer_m)
            & (n >= bx[1] - strip_crop_buffer_m) & (n <= bx[3] + strip_crop_buffer_m)
        )
        if not in_box.any():
            continue
        sub = {f: np.asarray(getattr(pc, f))[in_box] for f in pc.fields}
        pc_box = PCData().from_dict(sub)
        # Per-granule Δt drives the displacement-budget filter (Shean 2019
        # |v|·|Δt| ≤ B). For airborne data each granule has a precise UTC
        # date, so Δt is the gap between strip and granule.
        granule_date = info["name_to_date"](g.name)
        if displacement_budget_m is not None and granule_date is not None:
            delta_t_years = abs((center_time - granule_date).total_seconds()) / (365.25 * 86400.0)
            pc_filt = filter_grounded_slow_velocity(
                pc_box, bedmachine_path, velocity_path,
                displacement_budget_m=displacement_budget_m,
                delta_t_years=delta_t_years,
            )
        else:
            pc_filt = filter_grounded_slow_velocity(
                pc_box, bedmachine_path, velocity_path,
            )
        if pc_filt.size > 0:
            parts.append(pc_filt)

    if not parts:
        print(f"!! {source} cache empty after filtering for {dem_id}")
        return "empty"

    # Concatenate and write the ASP-ready CSV.
    xs = np.concatenate([np.asarray(p.x) for p in parts])
    ys = np.concatenate([np.asarray(p.y) for p in parts])
    hs = np.concatenate([np.asarray(p.h) for p in parts])
    df = pd.DataFrame({"easting": xs, "northing": ys, "h_mean": hs})
    expected.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(expected, index=False)
    print(f"  ✅ wrote {expected.name}  (n={len(df)})")
    return "ok"


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _build_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dem-id", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--time-window", type=int, default=90,
                   help="±N day temporal half-window for airborne-lidar match")
    p.add_argument("--prefetch-only", action="store_true",
                   help="Run only the CMR + download step, then exit")
    p.add_argument("--no-prefetch", action="store_true",
                   help="Skip the CMR + download step; use only granules already on disk")
    p.add_argument("--prefetch-dry-run", action="store_true",
                   help="Query CMR but skip downloading missing granules")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel Earthdata downloads")
    p.add_argument("--displacement-budget-m", type=float, default=10.0,
                   help="Shean 2019 displacement-budget B for the per-granule velocity "
                        "filter (|v|·|Δt| ≤ B); set to 0 to disable")
    return p


def run_cache_airborne_main(
    *,
    source: str,
    aoi_polygon,
    strips_dir: Path,
    cache_dir: Path,
    granule_dir: Path,
    bedmachine_path: Path,
    velocity_path: Path,
    start_time: str,
    end_time: str,
    description: str | None = None,
    args: argparse.Namespace | None = None,
    fetch_window_env: tuple[str, str] | None = None,
) -> int:
    r"""Per-basin entry point. Wires CLI flags to CMR prefetch + per-strip
    filtering for one airborne-lidar product.

    Parameters mirror :func:`stereo_melt.coregister.cache_cs2.run_cache_cs2_main`,
    plus ``source`` ∈ {``"atm"``, ``"lvis"``}.
    """
    if source not in PRODUCT_INFO:
        raise ValueError(f"source must be 'atm' or 'lvis', got {source!r}")
    if description is None:
        description = (
            f"Cache IceBridge {PRODUCT_INFO[source]['short_name']} GCPs "
            f"for a basin's REMA strips."
        )
    if args is None:
        parser = _build_argparser(description)
        args = parser.parse_args()

    cache_dir = Path(cache_dir)
    granule_dir = Path(granule_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    granule_dir.mkdir(parents=True, exist_ok=True)

    if fetch_window_env is not None:
        start_env, end_env = fetch_window_env
        start_time = os.environ.get(start_env, start_time)
        end_time = os.environ.get(end_env, end_time)

    # IceBridge campaigns are seasonal (~Oct-Nov in Antarctica). For a
    # 2019-2024 strip window, we want CMR to also return the 2009-2018
    # campaigns since IS2-era strips often need ATM/LVIS from earlier
    # campaigns as the closest temporal match. Extend the CMR temporal
    # window backward to 2009 unless the caller's start_time is already
    # earlier.
    cmr_start = min(pd.Timestamp(start_time), pd.Timestamp("2009-01-01")).strftime("%Y-%m-%d")
    cmr_end = (pd.Timestamp(end_time) + pd.Timedelta(days=args.time_window)).strftime("%Y-%m-%d")

    print(f"AOI bounds (EPSG:3031): {aoi_polygon.bounds}")
    print(f"Strip window: {start_time} .. {end_time}")
    print(f"CMR query window: {cmr_start} .. {cmr_end}")

    todo = find_in_window_strips(strips_dir, start_time, end_time, aoi_polygon=aoi_polygon)
    print(f"Found {len(todo)} strips to consider for {source.upper()} caching.")
    if args.dem_id is not None:
        todo = [(p, t) for p, t in todo if p.stem == args.dem_id]
        if not todo:
            raise SystemExit(f"--dem-id {args.dem_id!r} not in window")
    if args.limit is not None:
        todo = todo[: args.limit]

    if not args.no_prefetch:
        prefetch_airborne_granules(
            source=source,
            aoi_polygon=aoi_polygon,
            start_time=cmr_start,
            end_time=cmr_end,
            granule_dir=granule_dir,
            workers=args.workers,
            dry_run=args.prefetch_dry_run,
            strip_dates=[t for _, t in todo],
            strip_match_window_days=args.time_window,
        )
        if args.prefetch_only:
            return 0

    n = len(todo)
    counts = {"ok": 0, "skip": 0, "fail": 0, "empty": 0}
    budget = args.displacement_budget_m if args.displacement_budget_m > 0 else None
    for i, (p, t) in enumerate(todo, start=1):
        result = cache_one_strip_airborne(
            p, t,
            source=source,
            cache_dir=cache_dir,
            granule_dir=granule_dir,
            bedmachine_path=bedmachine_path,
            velocity_path=velocity_path,
            time_window_days=args.time_window,
            overwrite=args.overwrite,
            displacement_budget_m=budget,
        )
        counts[result] += 1
        print(f"[{i}/{n}] {t.date()}  {p.stem}  -> {result.upper()}  "
              f"(ok={counts['ok']} skip={counts['skip']} "
              f"empty={counts['empty']} fail={counts['fail']})")

    print(f"\n=== {source.upper()} cache summary: {counts['ok']} new, "
          f"{counts['skip']} cached, {counts['empty']} empty, "
          f"{counts['fail']} failed (of {n}) ===")
    return 1 if counts["fail"] else 0
