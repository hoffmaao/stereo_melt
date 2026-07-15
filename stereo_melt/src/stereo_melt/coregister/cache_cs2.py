# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""CryoSat-2 SARIn L2 POCA cache: per-strip altimetric control points.

Stage 0 of the CS2 GCP integration plan (see
``project_cryosat2_gcp_plan.md``). Mirrors :mod:`stereo_melt.io.altimetry`
+ ``cache_icesat2``: discover in-window REMA strips, find CS2 SARIn
passes that overlap the strip footprint and date, filter to grounded
ice + low-slope, cache to HDF5 so per-basin ``align_strips`` drivers can
consume CS2 alongside (or instead of) IS2 without touching ESA again.

**Scope locks** (apply to every basin caller):

- Mode: SARIn only. LRM is for the interior plateau; SARIn covers ice
  margins + grounding zones, which is what we need.
- Product: L2 POCA (Baseline-D NetCDF). NO swath, NO LVIS, NO ATM, NO GLAS.
- Per-strip temporal half-window: 90 days (CS2's 369-day repeat means
  ±5 d would miss most strips). Tuneable via ``--time-window``.

**Granule download uses ESA EO Sign credentials from ``~/.netrc``** under
``machine science-pds.cryosat.esa.int``. Transport is HTTPS via the
JSON API behind the PHP front-end (see :mod:`stereo_melt.io.esa_cs2_https`).

This module is the basin-agnostic library. Per-basin wrappers under
``<basin>/cache_cryosat2.py`` pass their config-supplied paths to
:func:`run_cache_cs2_main` and are otherwise ~50-line glue.
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

import netCDF4 as ncf
import numpy as np
import pandas as pd
import xarray as xr
from pointCollection.data import data as PCData
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

from ..io.esa_cs2_https import EsaCryoSatHttpsClient

DATE_RE = re.compile(r"_(\d{8})_")
HDR_TIME_RE = re.compile(r"_(\d{8})T(\d{6})_(\d{8})T(\d{6})_")

# Default ~5 km buffer around the basin bbox to absorb SARIn cross-track
# look angle (~1.5 km footprint) plus orbit slop.
DEFAULT_STRIP_CROP_BUFFER_M = 5_000.0

# Remote path layout under the JSON-API ``Cry0Sat2_data/`` root.
ESA_SARIN_L2_REMOTE = "SIR_SIN_L2"


# ---------------------------------------------------------------------------
# Granule reading: CryoSat L2 -> PCData
# ---------------------------------------------------------------------------


def read_cs2_l2_sarin_granule(nc_path: Path) -> PCData | None:
    r"""Read one CS2 L2 SARIn POCA granule into a :class:`pointCollection.data`.

    Reads the Baseline-D ``SIR_SIN_2`` NetCDF directly via :mod:`netCDF4`,
    pulling the 20 Hz POCA variables we need: lat/lon (slope-corrected),
    surface height from retracker 1, time, surface-type mask, peakiness,
    and the two quality flags. We don't go through
    ``cryosat_toolkit.read_cryosat_L2`` because its Baseline-D path is a
    sea-ice reader (it requires ``freeboard_20_ku`` and other variables
    that don't exist in SARIn ice-sheet products).

    Returns ``None`` if the granule is not SARIn or has no usable POCA
    returns. Heights are WGS-84 ellipsoidal metres; horizontal coords
    are reprojected from POCA lat/lon to EPSG:3031.
    """
    fname = Path(nc_path).name
    if "SIR_SIN_2" not in fname:
        return None

    try:
        with ncf.Dataset(str(nc_path)) as f:
            lat = np.ma.filled(f.variables["lat_poca_20_ku"][:], np.nan).astype(np.float64)
            lon = np.ma.filled(f.variables["lon_poca_20_ku"][:], np.nan).astype(np.float64)
            h = np.ma.filled(f.variables["height_1_20_ku"][:], np.nan).astype(np.float64)
            # ESA retracker-1 range to surface — needed by the L1B LMG floor
            # test to anchor h_lmg = height_1 + (range_1 - range_lmg).
            rng = np.ma.filled(f.variables["range_1_20_ku"][:], np.nan).astype(np.float64)
            t_sec = np.ma.filled(f.variables["time_20_ku"][:], np.nan).astype(np.float64)
            surf = np.ma.filled(f.variables["surf_type_20_ku"][:], -1).astype(np.int8)
            peak = np.ma.filled(f.variables["peakiness_20_ku"][:], np.nan).astype(np.float64)
            qstatus = np.ma.filled(f.variables["flag_prod_status_20_ku"][:], 0).astype(np.int32)
            qretr = np.ma.filled(f.variables["retracker_1_quality_20_ku"][:], 0).astype(np.int32)
    except Exception as exc:
        print(f"  !! NetCDF read failed on {fname}: {exc}")
        return None

    keep = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h)
    if not keep.any():
        return None
    lat = lat[keep]; lon = lon[keep]; h = h[keep]; rng = rng[keep]
    t_sec = t_sec[keep]; surf = surf[keep]
    peak = peak[keep]; qstatus = qstatus[keep]; qretr = qretr[keep]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    east, north = transformer.transform(lon, lat)

    # time_20_ku is "seconds since 2000-01-01" (TAI). Stored as int64
    # ns-since-1970 so h5py can serialize it.
    t = (np.datetime64("2000-01-01T00:00:00", "ns")
         + (t_sec * 1e9).astype("int64").astype("timedelta64[ns]")
         ).astype(np.int64)

    out = {
        "x": np.asarray(east, dtype=np.float64),
        "y": np.asarray(north, dtype=np.float64),
        "h": np.asarray(h, dtype=np.float64),
        "range_esa": np.asarray(rng, dtype=np.float64),
        # Raw TAI seconds-since-2000 — the bit-identical clock the L1B
        # product carries, used to join POCA points to their waveforms.
        "t_tai": np.asarray(t_sec, dtype=np.float64),
        "t": t,
        "source": np.full(lat.size, "cs2_sarin_l2", dtype=object),
        "quality": qstatus,
        "retracker_quality": qretr,
        "peakiness": peak,
        "surface_type": surf,
    }
    return PCData().from_dict(out)


# ---------------------------------------------------------------------------
# Filtering: BedMachine grounded mask + slow-velocity filter
# ---------------------------------------------------------------------------


def filter_cs2_grounded_slow_velocity(
    pc: PCData,
    bedmachine_path: Path,
    velocity_path: Path,
    *,
    grounded_mask_value: int = 2,
    max_speed_myr: float = 10.0,
    vel_smooth_sigma_m: float = 2_000.0,
    rema_residual_gate_m: float | None = 100.0,
    dem_center_date=None,
    displacement_budget_m: float | None = None,
) -> PCData:
    r"""Keep CS2 returns over grounded ice (BedMachine ``mask == 2``)
    *and* slow-flowing (smoothed MEaSUREs |v| < ``max_speed_myr`` m/yr).

    Mirrors :func:`stereo_melt.coregister.reference._filter_is2_by_sampled_fields`
    so the CS2 GCP catalogue lives in exactly the same regime as the IS2
    catalogue: a CS2 point is retained iff IS2 *would have been* retained
    at that location.

    With ``rema_residual_gate_m`` set, also drops returns where
    ``|h_cs2 - h_BedMachine_surface| > gate`` (default 100 m, Lowery 2025
    PIG channel paper convention). BedMachine ``surface`` is sourced
    from the REMA mosaic and shares CS2's WGS84-ellipsoid vertical
    datum, so the residual is a direct difference without geoid offset.
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

    n_in = pc.size
    keep_grounded = (pt_mask == grounded_mask_value)
    keep_finite = np.isfinite(pc.h)
    # Temporal/velocity screen, two regimes:
    #   • displacement budget (Shean 2019, preferred): keep a point if its
    #     advection |v|·|Δt| over the DEM→overpass interval stays within
    #     ``displacement_budget_m`` — fast grounded ice is retained with a
    #     proportionally tighter time window instead of hard-cut. Mirrors the
    #     IS2 catalogue (reference.py) so both control sources share a regime.
    #   • hard slow cut (legacy/raw-CS2): |v| < ``max_speed_myr`` regardless Δt.
    use_budget = displacement_budget_m is not None and dem_center_date is not None
    if use_budget and "t" in pc.fields:
        _ts = pd.Timestamp(dem_center_date)
        if _ts.tz is not None:
            _ts = _ts.tz_convert(None)
        t_dem = np.datetime64(_ts, "ns")
        t_pt = np.asarray(pc.t, dtype="int64").astype("datetime64[ns]")
        dt_yr = (np.abs((t_pt - t_dem).astype("timedelta64[s]").astype(np.float64))
                 / (86400.0 * 365.25))
        adv = pt_speed * dt_yr  # metres of horizontal advection over |Δt|
        keep_temporal = adv <= displacement_budget_m
        _nt = int((keep_grounded & keep_finite & keep_temporal).sum())
        _md = (float(np.median(dt_yr[keep_grounded & keep_finite & keep_temporal])) * 365.25
               if _nt else float("nan"))
        vel_label = (f"|v|·|Δt|≤{displacement_budget_m:g}m={_nt}"
                     + (f" (med Δt={_md:.0f}d)" if _nt else ""))
    else:
        if use_budget:
            print("  ⚠  displacement-budget requested but pc has no per-point 't'; "
                  "falling back to hard velocity cut")
        keep_temporal = (pt_speed < max_speed_myr)
        vel_label = f"slow<{max_speed_myr:.0f}m/yr={int(keep_temporal.sum())}"
    keep = keep_grounded & keep_temporal & keep_finite

    # Optional REMA-residual gate (Lowery 2025 PIG channel paper, 100 m;
    # Zinck 2023 BURGEE Dotson, 30 m). BedMachine ``surface`` is sourced
    # from REMA and shares CS2's WGS84-ellipsoid datum, so |h_cs2 -
    # h_surface| is a direct difference. Catches gross retracker errors
    # (ocean leads, off-shelf returns, blunders) before they reach the
    # ICP plane.
    if rema_residual_gate_m is not None and rema_residual_gate_m > 0:
        surf_sub = _crop(bm_ds, "surface")
        pt_surf = _sample(
            surf_sub.values, surf_sub["x"].values, surf_sub["y"].values,
            east, north,
        )
        residual = np.abs(np.asarray(pc.h, dtype=np.float64) - pt_surf)
        keep_residual = np.isfinite(pt_surf) & (residual < rema_residual_gate_m)
        keep &= keep_residual
        res_label = (f", |Δh_surf|<{rema_residual_gate_m:.0f}m="
                     f"{int(keep_residual.sum())}")
    else:
        res_label = ""

    n_keep = int(keep.sum())
    print(f"  CS2 filter: {n_keep}/{n_in} kept "
          f"(grounded={int(keep_grounded.sum())}, "
          f"{vel_label}"
          f"{res_label})")

    if n_keep == 0:
        return PCData().from_dict({"x": np.array([]), "y": np.array([]), "h": np.array([])})

    sub_dict = {f: np.asarray(getattr(pc, f))[keep] for f in pc.fields}
    return PCData().from_dict(sub_dict)


def _check_cs2_distribution(
    pc: PCData,
    *,
    min_count: int = 80,
    min_spread_ns_m: float = 20_000.0,
    min_spread_ew_m: float = 5_000.0,
) -> tuple[bool, str]:
    r"""BURGEE-style geometric distribution test (Zinck 2023 §4.2).

    Require enough CS2 matchups, distributed widely enough, to constrain
    a per-strip ICP plane. BURGEE used this as a hard gate before
    attempting a per-strip plane-fit coregistration; ``pc_align`` is
    6-DoF point-to-plane ICP rather than a 3-DoF plane fit, but the
    same principle applies — a tight cluster of CS2 returns has no
    angular leverage on the ICP and gets dominated (and overweighted)
    by sample count vs. IS2.

    Defaults are downscaled from Zinck 2023 §4.2 (≥80 matchups, ≥60 km
    N-S, ≥10 km E-W). BURGEE's spread thresholds were tuned for their
    per-orbit strip composites; individual REMA stereo strips are
    ~30-50 km on a side, so we use ≥80 matchups, ≥20 km N-S, ≥5 km E-W
    (∼40% of a typical PIG strip extent). Override at the CLI for other
    basin geometries.

    Returns ``(passed, reason)``.
    """
    n = int(pc.size)
    if n < min_count:
        return False, f"only {n}<{min_count} matchups"
    spread_ew = float(np.asarray(pc.x).max() - np.asarray(pc.x).min())
    spread_ns = float(np.asarray(pc.y).max() - np.asarray(pc.y).min())
    if spread_ns < min_spread_ns_m:
        return False, (f"N-S spread {spread_ns/1e3:.1f} km < "
                       f"{min_spread_ns_m/1e3:.0f} km")
    if spread_ew < min_spread_ew_m:
        return False, (f"E-W spread {spread_ew/1e3:.1f} km < "
                       f"{min_spread_ew_m/1e3:.0f} km")
    return True, (f"n={n}, NS={spread_ns/1e3:.1f}km, "
                  f"EW={spread_ew/1e3:.1f}km")


# ---------------------------------------------------------------------------
# ESA Science Server HTTPS walker
# ---------------------------------------------------------------------------


def _strip_windows_to_months(strip_dates, time_window_days: int):
    """Sorted union of (year, month) tuples covering each strip date ± window."""
    months = set()
    for d in strip_dates:
        d = pd.Timestamp(d)
        t0 = d - pd.Timedelta(days=time_window_days)
        t1 = d + pd.Timedelta(days=time_window_days)
        cur = pd.Timestamp(t0.year, t0.month, 1)
        end = pd.Timestamp(t1.year, t1.month, 1)
        while cur <= end:
            months.add((cur.year, cur.month))
            cur = cur + pd.DateOffset(months=1)
    return sorted(months)


_HDR_FIELD_RE = {
    k: re.compile(rf'<{k}\s+unit="10-6 deg">([+-]?\d+)</{k}>')
    for k in ("Start_Lat", "Start_Long", "Stop_Lat", "Stop_Long")
}


def _parse_hdr_segment(hdr_bytes: bytes):
    text = hdr_bytes.decode("ascii", errors="replace")
    out = {}
    for k, regex in _HDR_FIELD_RE.items():
        m = regex.search(text)
        if not m:
            return None
        out[k] = int(m.group(1)) * 1e-6
    return out


def _wrap180(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _hdr_validity_start(hdr_name: str) -> pd.Timestamp | None:
    m = HDR_TIME_RE.search(hdr_name)
    if not m:
        return None
    return pd.Timestamp(f"{m.group(1)}T{m.group(2)}")


def _strip_bbox_lonlat(strip_path: Path, buffer_m: float = 25_000.0):
    r"""Return ((lat_lo, lat_hi), (lon_lo, lon_hi)) for a strip footprint
    after padding the EPSG:3031 bbox by ``buffer_m`` to absorb segment
    length, footprint, and orbit slop. Samples bbox edges (not just
    corners) before the polar→geographic transform."""
    import rasterio
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


def _segment_intersects_lonlat_bbox(seg, lat_range, lon_range):
    """Approximate test: does the great-circle segment (Start..Stop) bbox
    intersect ``(lat_range, lon_range)``? Date-line wraparounds are
    treated as intersections (cheaper than a precise test)."""
    seg_lats = sorted([seg["Start_Lat"], seg["Stop_Lat"]])
    if seg_lats[1] < lat_range[0] or seg_lats[0] > lat_range[1]:
        return False

    s_lon_a, s_lon_b = sorted([_wrap180(seg["Start_Long"]), _wrap180(seg["Stop_Long"])])
    a_lo, a_hi = _wrap180(lon_range[0]), _wrap180(lon_range[1])

    aoi_intervals = [(-180.0, a_hi), (a_lo, 180.0)] if a_lo > a_hi else [(a_lo, a_hi)]
    if s_lon_b - s_lon_a > 180.0:
        return True
    seg_intervals = [(s_lon_a, s_lon_b)]
    for s in seg_intervals:
        for a in aoi_intervals:
            if not (s[1] < a[0] or s[0] > a[1]):
                return True
    return False


def _process_month(
    client: EsaCryoSatHttpsClient,
    year: int,
    month: int,
    strip_meta: list[dict],
    half_window: pd.Timedelta,
    hdr_dir: Path,
    granule_dir: Path,
    dry_run: bool,
    heartbeat_every: int,
) -> dict:
    tag = f"{year}-{month:02d}"
    remote = f"{ESA_SARIN_L2_REMOTE}/{year}/{month:02d}"
    counts = {"hdr_have": 0, "hdr_dl": 0, "nc_have": 0, "nc_dl": 0, "nc_match": 0,
              "bytes_hdr": 0, "bytes_nc": 0}

    try:
        try:
            entries = client.list_dir(remote)
        except Exception as exc:
            print(f"  ⚠  [{tag}] list {remote}: {exc}", flush=True)
            return counts

        by_name = {e["name"]: e for e in entries}
        hdrs = sorted(n for n in by_name
                      if n.endswith(".HDR") and "SIR_SIN_2" in n)
        ncs = {n for n in by_name if n.endswith(".nc") and "SIR_SIN_2" in n}

        local_hdr_dir = hdr_dir / f"{year}" / f"{month:02d}"
        local_hdr_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.monotonic()
        t_last = t_start
        n_dl_since_hb = 0
        for i, hdr_name in enumerate(hdrs, start=1):
            local = local_hdr_dir / hdr_name
            if local.exists():
                counts["hdr_have"] += 1
            else:
                try:
                    counts["bytes_hdr"] += client.download(
                        by_name[hdr_name]["path"], local,
                    )
                    counts["hdr_dl"] += 1
                    n_dl_since_hb += 1
                except Exception as exc:
                    print(f"    !! [{tag}] HDR fetch failed {hdr_name}: {exc}",
                          flush=True)
                    continue

            if heartbeat_every and i % heartbeat_every == 0:
                now = time.monotonic()
                rate = n_dl_since_hb / max(now - t_last, 1e-6)
                print(
                    f"  [{tag}] HDR {i}/{len(hdrs)}  "
                    f"have={counts['hdr_have']} dl={counts['hdr_dl']}  "
                    f"{counts['bytes_hdr']/1e6:.1f} MB  "
                    f"({rate:.1f} dl/s, "
                    f"{(now-t_start)/60:.1f} min elapsed)",
                    flush=True,
                )
                t_last = now
                n_dl_since_hb = 0

        wanted_nc = []
        for hdr_name in hdrs:
            local = local_hdr_dir / hdr_name
            if not local.exists():
                continue
            seg = _parse_hdr_segment(local.read_bytes())
            if seg is None:
                continue
            t_seg = _hdr_validity_start(hdr_name)
            if t_seg is None:
                continue
            for sm in strip_meta:
                if abs(t_seg - sm["date"]) > half_window:
                    continue
                if _segment_intersects_lonlat_bbox(seg, sm["lat"], sm["lon"]):
                    nc_name = hdr_name[:-4] + ".nc"
                    if nc_name in ncs:
                        wanted_nc.append(nc_name)
                    break

        for nc_name in wanted_nc:
            local = granule_dir / nc_name
            if local.exists():
                counts["nc_have"] += 1
                continue
            if dry_run:
                counts["nc_dl"] += 1
                continue
            try:
                counts["bytes_nc"] += client.download(
                    by_name[nc_name]["path"], local,
                )
                counts["nc_dl"] += 1
            except Exception as exc:
                print(f"    !! [{tag}] .nc fetch failed {nc_name}: {exc}",
                      flush=True)
                continue
        counts["nc_match"] = len(wanted_nc)

        print(
            f"  [{tag}] DONE: HDR {counts['hdr_have']} cached + "
            f"{counts['hdr_dl']} new ({len(hdrs)} total); "
            f"matched {len(wanted_nc)} → .nc {counts['nc_have']} cached + "
            f"{counts['nc_dl']} new",
            flush=True,
        )
    except Exception:
        print(f"  !! [{tag}] worker crashed:", flush=True)
        traceback.print_exc()

    return counts


def prefetch_cs2_sarin_granules(
    strips: list[tuple[Path, pd.Timestamp]],
    granule_dir: Path,
    hdr_dir: Path,
    *,
    time_window_days: int = 90,
    strip_buffer_m: float = 25_000.0,
    dry_run: bool = False,
    workers: int = 4,
    heartbeat_every: int = 200,
) -> dict:
    r"""Walk the ESA Science Server, prefetch CS2 SARIn POCA granules whose
    ~20-sec segment plausibly overlaps any in-window strip footprint.

    Two-stage download: HDRs first (~33 KB each, paired 1:1 with .nc),
    parse for segment endpoint lat/lon, then download only matched .nc
    files. HDRs are kept on disk so re-runs are cheap.
    """
    if not strips:
        print("  prefetch: no strips supplied; nothing to do")
        return {}

    granule_dir.mkdir(parents=True, exist_ok=True)
    hdr_dir.mkdir(parents=True, exist_ok=True)

    strip_meta = []
    for p, t in strips:
        try:
            lat_r, lon_r = _strip_bbox_lonlat(p, buffer_m=strip_buffer_m)
        except Exception as exc:
            print(f"  ⚠  could not bbox {p.name}: {exc}")
            continue
        strip_meta.append({"path": p, "date": pd.Timestamp(t), "lat": lat_r, "lon": lon_r})

    print(f"  prefetch: {len(strip_meta)} strips, ±{time_window_days} d window, "
          f"strip bbox buffer {strip_buffer_m/1e3:.0f} km", flush=True)

    months = _strip_windows_to_months([s["date"] for s in strip_meta], time_window_days)
    print(f"  prefetch covers {len(months)} (year, month) buckets: "
          f"{months[0][0]}-{months[0][1]:02d} → {months[-1][0]}-{months[-1][1]:02d}; "
          f"workers={workers}, heartbeat every {heartbeat_every} HDRs",
          flush=True)

    half_window = pd.Timedelta(days=time_window_days)
    totals = {"hdr_have": 0, "hdr_dl": 0, "nc_have": 0, "nc_dl": 0, "nc_match": 0,
              "bytes_hdr": 0, "bytes_nc": 0}

    client = EsaCryoSatHttpsClient(timeout=60)
    n_workers = max(1, min(workers, len(months)))
    t_pool_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="cs2hdr") as ex:
        future_to_tag = {
            ex.submit(
                _process_month,
                client, year, month, strip_meta, half_window,
                hdr_dir, granule_dir, dry_run, heartbeat_every,
            ): f"{year}-{month:02d}"
            for (year, month) in months
        }
        n_done = 0
        for fut in as_completed(future_to_tag):
            tag = future_to_tag[fut]
            try:
                c = fut.result()
            except Exception as exc:
                print(f"  !! [{tag}] future raised: {exc}", flush=True)
                continue
            for k, v in c.items():
                totals[k] += v
            n_done += 1
            print(f"  [pool] {n_done}/{len(months)} months done "
                  f"({(time.monotonic()-t_pool_start)/60:.1f} min elapsed)",
                  flush=True)

    print("\n  prefetch summary:")
    print(f"    HDRs:  {totals['hdr_have']} cached + {totals['hdr_dl']} new "
          f"({totals['bytes_hdr']/1e6:.1f} MB downloaded)")
    print(f"    .nc:   {totals['nc_have']} cached + {totals['nc_dl']} new "
          f"({totals['bytes_nc']/1e6:.1f} MB downloaded), "
          f"{totals['nc_match']} matched AOI")
    return totals


def discover_cs2_sarin_granules(
    aoi_polygon,
    start_time: str,
    end_time: str,
    cache_dir: Path,
) -> list[Path]:
    r"""Find CS2 SARIn granules on disk in ``[start_time, end_time)``.
    Spatial filtering happens elsewhere (per-strip HDR bbox at prefetch
    + post-read crop in :func:`cache_one_strip`); ``aoi_polygon`` is
    accepted for interface stability."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    on_disk = sorted(cache_dir.glob("CS_*SIR_SIN_2*.nc"))
    if not on_disk:
        print(f"  ⚠  no CS2 granules on disk under {cache_dir}")
        return []

    t0 = pd.Timestamp(start_time)
    t1 = pd.Timestamp(end_time)
    fname_re = re.compile(r"_(\d{8})T\d{6}_")
    keep = []
    for p in on_disk:
        m = fname_re.search(p.name)
        if not m:
            continue
        t = pd.Timestamp(m.group(1))
        if t0 <= t < t1:
            keep.append(p)
    print(f"  found {len(keep)}/{len(on_disk)} CS2 granules in window {start_time}..{end_time}")
    return keep


# ---------------------------------------------------------------------------
# Per-strip caching
# ---------------------------------------------------------------------------


def _strip_bbox(strip_path: Path):
    import rasterio
    with rasterio.open(strip_path) as src:
        bx = src.bounds
    return box(bx.left, bx.bottom, bx.right, bx.top)


def _date_from_path(p: Path) -> pd.Timestamp | None:
    m = DATE_RE.search(p.name)
    return pd.Timestamp(m.group(1)) if m else None


def find_in_window_strips(
    strips_dir: Path,
    start_time: str,
    end_time: str,
    *,
    aoi_polygon=None,
) -> list[tuple[Path, pd.Timestamp]]:
    r"""Discover REMA strips in ``[start_time, end_time)`` whose bbox
    intersects ``aoi_polygon`` (if supplied). Drops PGC quality
    companions (``*_matchtag.tif`` / ``*_bitmask.tif``)."""
    strips_dir = Path(strips_dir)
    t_start = pd.Timestamp(start_time)
    t_end = pd.Timestamp(end_time)
    todo = []
    for p in sorted(strips_dir.glob("SETSM_*.tif")):
        if p.stem.endswith(("_matchtag", "_bitmask")):
            continue
        t = _date_from_path(p)
        if t is None or not (t_start <= t < t_end):
            continue
        sb = _strip_bbox(p)
        if aoi_polygon is not None and not sb.intersects(aoi_polygon):
            continue
        todo.append((p, t))
    return sorted(todo, key=lambda x: x[1])


def cache_one_strip(
    strip_path: Path,
    center_time: pd.Timestamp,
    *,
    cache_dir: Path,
    granule_dir: Path,
    bedmachine_path: Path,
    velocity_path: Path,
    time_window_days: int = 90,
    strip_crop_buffer_m: float = DEFAULT_STRIP_CROP_BUFFER_M,
    overwrite: bool = False,
    rema_residual_gate_m: float | None = 100.0,
    dist_min_count: int = 80,
    dist_min_spread_ns_m: float = 20_000.0,
    dist_min_spread_ew_m: float = 5_000.0,
) -> str:
    """Populate the CS2 cache for one strip.

    Returns one of ``ok|skip|fail|empty|sparse``:

    - ``ok``: filtered cache written.
    - ``skip``: cache hit, no work done (use ``overwrite=True`` to refresh).
    - ``fail``: no CS2 granules in temporal window.
    - ``empty``: granules present but all returns filtered out
      (grounded mask, slow-velocity, or residual gate).
    - ``sparse``: returns survive filtering but the BURGEE distribution
      test fails — too few matchups or too tight a cluster to constrain
      an ICP plane; cache is NOT written (Zinck 2023 §4.2). Setting
      ``dist_min_count <= 0`` disables this test.
    """
    dem_id = strip_path.stem
    expected = cache_dir / f"cs2_filtered_{dem_id}.h5"
    if expected.exists() and not overwrite:
        print(f"⏭  {dem_id}: CS2 cache hit, skipping (use --overwrite)")
        return "skip"

    print(f"\n=== {center_time.date()}  {dem_id} ===")

    t0 = (center_time - pd.Timedelta(days=time_window_days)).strftime("%Y-%m-%d")
    t1 = (center_time + pd.Timedelta(days=time_window_days)).strftime("%Y-%m-%d")
    aoi = _strip_bbox(strip_path)

    granules = discover_cs2_sarin_granules(
        aoi_polygon=aoi, start_time=t0, end_time=t1, cache_dir=granule_dir,
    )
    if not granules:
        print(f"!! no CS2 granules in window for {dem_id}; skipping")
        return "fail"

    parts: list[PCData] = []
    for nc in granules:
        try:
            pc = read_cs2_l2_sarin_granule(nc)
        except Exception as exc:
            print(f"  !! read failed on {nc.name}: {exc}")
            traceback.print_exc()
            continue
        if pc is None or pc.size == 0:
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
        pc_filt = filter_cs2_grounded_slow_velocity(
            pc_box, bedmachine_path, velocity_path,
            rema_residual_gate_m=rema_residual_gate_m,
        )
        if pc_filt.size > 0:
            parts.append(pc_filt)

    if not parts:
        print(f"!! CS2 cache empty after filtering for {dem_id}")
        return "empty"

    all_fields = sorted({f for p in parts for f in p.fields})
    merged = {
        f: np.concatenate([np.asarray(getattr(p, f)) for p in parts if f in p.fields])
        for f in all_fields
    }
    out = PCData().from_dict(merged)

    # BURGEE per-strip distribution test (Zinck 2023 §4.2). Disabled when
    # dist_min_count <= 0.
    if dist_min_count > 0:
        passed, reason = _check_cs2_distribution(
            out,
            min_count=dist_min_count,
            min_spread_ns_m=dist_min_spread_ns_m,
            min_spread_ew_m=dist_min_spread_ew_m,
        )
        if not passed:
            print(f"  ⚠  CS2 distribution test FAIL ({reason}); "
                  f"not caching for {dem_id}")
            return "sparse"
        print(f"  ✓ CS2 distribution test pass ({reason})")

    out.to_h5(str(expected))
    print(f"  ✅ wrote {expected.name}  (n={out.size})")
    return "ok"


# ---------------------------------------------------------------------------
# Top-level driver — basin wrappers call this from their main()
# ---------------------------------------------------------------------------


def _build_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dem-id", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--time-window", type=int, default=90,
                   help="±N day temporal half-window for CS2-strip matching")
    p.add_argument("--strip-buffer-km", type=float, default=25.0,
                   help="Buffer (km) padded around each strip bbox for HDR-segment match")
    p.add_argument("--prefetch-only", action="store_true",
                   help="Run only the HTTPS prefetch, then exit (no per-strip caching)")
    p.add_argument("--no-prefetch", action="store_true",
                   help="Skip the HTTPS prefetch and use only granules already on disk")
    p.add_argument("--prefetch-dry-run", action="store_true",
                   help="List what would be downloaded without contacting the server")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel ESA-server connections — one per month-bucket worker")
    p.add_argument("--heartbeat-every", type=int, default=200,
                   help="Print HDR progress every N files within a month (0 to disable)")
    p.add_argument("--rema-residual-gate-m", type=float, default=100.0,
                   help="Per-point gate on |h_cs2 - h_BedMachine_surface| (m). "
                        "Lowery 2025 PIG channel paper uses 100 m; Zinck 2023 "
                        "BURGEE uses 30 m. Set to 0 to disable.")
    p.add_argument("--cs2-min-matchups", type=int, default=80,
                   help="BURGEE distribution test: minimum CS2 matchups per "
                        "strip. Strips below threshold are not cached. Set "
                        "to 0 to disable.")
    p.add_argument("--cs2-min-spread-ns-km", type=float, default=20.0,
                   help="BURGEE distribution test: minimum N-S CS2 spread (km). "
                        "Default 20 km is downscaled from Zinck 2023's 60 km "
                        "(BURGEE used per-orbit strip composites; our REMA "
                        "strips are 30-50 km on a side, so 20 km ~ 40%% of "
                        "strip extent).")
    p.add_argument("--cs2-min-spread-ew-km", type=float, default=5.0,
                   help="BURGEE distribution test: minimum E-W CS2 spread (km). "
                        "Default 5 km downscaled from Zinck 2023's 10 km, same "
                        "rationale as --cs2-min-spread-ns-km.")
    return p


def run_cache_cs2_main(
    *,
    aoi_polygon,
    strips_dir: Path,
    cache_dir: Path,
    granule_dir: Path,
    hdr_dir: Path,
    bedmachine_path: Path,
    velocity_path: Path,
    start_time: str,
    end_time: str,
    description: str = "Cache CS2 SARIn POCA control points for a basin's REMA strips.",
    args: argparse.Namespace | None = None,
    fetch_window_env: tuple[str, str] | None = None,
    on_aoi_loaded: Callable[[], None] | None = None,
) -> int:
    r"""Per-basin entry point. Wires CLI flags to prefetch + per-strip caching.

    Parameters
    ----------
    aoi_polygon : shapely.geometry.Polygon
        Basin AOI in EPSG:3031, used to filter which REMA strips are
        considered. Loading is the caller's responsibility (use whatever
        the basin's other drivers use).
    strips_dir : Path
        REMA strips directory (shared across basins).
    cache_dir : Path
        Per-basin CS2 cache output dir, e.g. ``<basin>/data/ASP/cs2_data/``.
    granule_dir, hdr_dir : Path
        Shared CS2 storage. Conventionally ``<MAIN_DIR>/data/CS2/granules``
        and ``.../data/CS2/hdrs``. Granules and HDRs are downloaded once
        for the whole project; per-basin caches are filtered subsets.
    bedmachine_path, velocity_path : Path
        Filtering inputs (BedMachine grounded mask + MEaSUREs phase map).
    start_time, end_time : str
        ``YYYY-MM-DD`` window. Override with ``fetch_window_env`` to read
        ``BASIN_FETCH_START`` / ``BASIN_FETCH_END`` env vars (mirrors
        ``<basin>.fetch_strips``).
    args : argparse.Namespace, optional
        Pre-parsed CLI args; if ``None``, parses ``sys.argv``.
    fetch_window_env : tuple[str, str], optional
        ``(start_env_var, end_env_var)`` pair, e.g.
        ``("NANSEN_FETCH_START", "NANSEN_FETCH_END")`` to allow a basin
        driver to extend its caching window beyond the run window.
    """
    if args is None:
        parser = _build_argparser(description)
        args = parser.parse_args()

    cache_dir = Path(cache_dir)
    granule_dir = Path(granule_dir)
    hdr_dir = Path(hdr_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    granule_dir.mkdir(parents=True, exist_ok=True)
    hdr_dir.mkdir(parents=True, exist_ok=True)

    if fetch_window_env is not None:
        start_env, end_env = fetch_window_env
        start_time = os.environ.get(start_env, start_time)
        end_time = os.environ.get(end_env, end_time)

    print(f"AOI bounds (EPSG:3031): {aoi_polygon.bounds}")
    if on_aoi_loaded is not None:
        on_aoi_loaded()

    todo = find_in_window_strips(
        strips_dir, start_time, end_time, aoi_polygon=aoi_polygon,
    )
    print(f"Window {start_time} .. {end_time}")
    print(f"Found {len(todo)} strips to consider for CS2 caching.")

    if args.dem_id is not None:
        todo = [(p, t) for p, t in todo if p.stem == args.dem_id]
        if not todo:
            raise SystemExit(f"--dem-id {args.dem_id!r} not in window")
    if args.limit is not None:
        todo = todo[: args.limit]

    if not args.no_prefetch:
        prefetch_cs2_sarin_granules(
            todo, granule_dir, hdr_dir,
            time_window_days=args.time_window,
            strip_buffer_m=args.strip_buffer_km * 1_000.0,
            dry_run=args.prefetch_dry_run,
            workers=args.workers,
            heartbeat_every=args.heartbeat_every,
        )
        if args.prefetch_only:
            return 0

    n = len(todo)
    counts = {"ok": 0, "skip": 0, "fail": 0, "empty": 0, "sparse": 0}
    residual_gate = (args.rema_residual_gate_m
                     if args.rema_residual_gate_m > 0 else None)
    for i, (p, t) in enumerate(todo, start=1):
        result = cache_one_strip(
            p, t,
            cache_dir=cache_dir,
            granule_dir=granule_dir,
            bedmachine_path=bedmachine_path,
            velocity_path=velocity_path,
            time_window_days=args.time_window,
            overwrite=args.overwrite,
            rema_residual_gate_m=residual_gate,
            dist_min_count=args.cs2_min_matchups,
            dist_min_spread_ns_m=args.cs2_min_spread_ns_km * 1_000.0,
            dist_min_spread_ew_m=args.cs2_min_spread_ew_km * 1_000.0,
        )
        counts[result] += 1
        print(f"[{i}/{n}] {t.date()}  {p.stem}  -> {result.upper()}  "
              f"(ok={counts['ok']} skip={counts['skip']} "
              f"sparse={counts['sparse']} empty={counts['empty']} "
              f"fail={counts['fail']})")

    print(f"\n=== CS2 cache summary: {counts['ok']} new, {counts['skip']} cached, "
          f"{counts['sparse']} sparse (distribution test fail), "
          f"{counts['empty']} empty (filter fail), "
          f"{counts['fail']} failed (no granules) of {n} ===")
    return 1 if counts["fail"] else 0
