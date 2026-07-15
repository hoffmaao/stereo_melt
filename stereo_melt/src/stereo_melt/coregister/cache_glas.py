# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""ICESat-1 GLAS (GLAH12) per-strip control-point cache.

Mirrors :mod:`stereo_melt.coregister.cache_airborne` but for the NSIDC
GLAH12 laser-altimetry product — the only laser control that can reach
the earliest REMA strips (2009 → Oct 2010; CryoTEMPO starts 2010-07,
IS2 2018-10). Per-strip filtered CSVs are written in the ASP-ready
``easting,northing,h_mean`` schema so
:func:`stereo_melt.coregister.asp.align_strip_with_asp` can consume
them alongside IS2 / CS2 / ATM / LVIS / rock.

**GLAH12-specific mechanics** (vs the airborne module):

- Granule filenames carry orbit/track counters, not calendar dates, so
  the prefetch step filters on CMR's granule ``begin_time`` metadata and
  the per-strip matching uses a **local granule time index**
  (``glah12_index.csv`` in the granule store) built by reading
  ``Data_40HZ/DS_UTCTime_40`` from each file once.
- The reader (:func:`stereo_melt.io.altimetry.from_glah12`) handles the
  TOPEX/Poseidon→WGS84 ellipsoid conversion (~70 cm), the saturation
  range correction, and the elev_use/sat_corr/numPk quality gates.
- Default temporal half-window is ±365 days (Shean ±1 yr control cap):
  ICESat campaigns are ~month-long snapshots 2-3× per year, so a ±90 d
  airborne-style window would strand most strips; the displacement-
  budget velocity filter (|v|·|Δt| ≤ B) tightens the admitted ground
  speed as Δt grows, exactly as for airborne granules.

Filter regime is identical to IS2/CS2/ATM: BedMachine ``mask == 2``
(grounded ice) ∩ displacement budget on smoothed MEaSUREs speed.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pointCollection.data import data as PCData

from ..io.altimetry import from_glah12, glah12_time_range
from ..io.nsidc_cmr import (
    download_nsidc_granule,
    make_session,
    query_nsidc_granules,
)
from .cache_airborne import (
    _aoi_bbox_lonlat,
    _strip_bbox,
    filter_grounded_slow_velocity,
    find_in_window_strips,
)

GLAH12_SHORT_NAME = "GLAH12"
GLAH12_VERSION = "034"
# ICESat-1 laser campaigns: L1A 2003-02-20 → L2F ends 2009-10-11.
GLAS_MISSION_START = pd.Timestamp("2003-02-01")
GLAS_MISSION_END = pd.Timestamp("2009-10-12")
GRANULE_INDEX_NAME = "glah12_index.csv"


# ---------------------------------------------------------------------------
# Granule time index
# ---------------------------------------------------------------------------


def build_granule_index(granule_dir: Path, verbose: bool = True) -> pd.DataFrame:
    r"""Return (and incrementally maintain) the granule→time-range index.

    Scans ``granule_dir`` for ``GLAH12_*.H5`` files, reads
    ``DS_UTCTime_40`` for any file not yet indexed, and persists the
    result to ``glah12_index.csv``. Rows for files no longer on disk are
    dropped. Returns a DataFrame with columns ``name, t0, t1``.
    """
    granule_dir = Path(granule_dir)
    granule_dir.mkdir(parents=True, exist_ok=True)
    idx_path = granule_dir / GRANULE_INDEX_NAME

    known: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    if idx_path.exists():
        try:
            df = pd.read_csv(idx_path, parse_dates=["t0", "t1"])
            known = {
                str(r["name"]): (pd.Timestamp(r["t0"]), pd.Timestamp(r["t1"]))
                for _, r in df.iterrows()
            }
        except Exception as exc:
            print(f"  !! unreadable {idx_path.name} ({exc}); rebuilding from scratch")
            known = {}

    on_disk = sorted(
        set(granule_dir.glob("GLAH12_*.H5")) | set(granule_dir.glob("GLAH12_*.h5"))
    )
    names_on_disk = {p.name for p in on_disk}
    new = [p for p in on_disk if p.name not in known]
    n_bad = 0
    for i, p in enumerate(new, start=1):
        try:
            t0, t1 = glah12_time_range(p)
        except Exception as exc:
            n_bad += 1
            print(f"  !! index skip {p.name}: {exc}")
            continue
        known[p.name] = (t0, t1)
        if verbose and i % 100 == 0:
            print(f"  granule index: {i}/{len(new)} new files read", flush=True)

    known = {n: tt for n, tt in known.items() if n in names_on_disk}
    out = pd.DataFrame(
        {
            "name": list(known.keys()),
            "t0": [tt[0] for tt in known.values()],
            "t1": [tt[1] for tt in known.values()],
        }
    ).sort_values("t0").reset_index(drop=True)
    out.to_csv(idx_path, index=False)
    if verbose and (new or n_bad):
        print(f"  granule index: {len(out)} granules "
              f"({len(new) - n_bad} newly read, {n_bad} unreadable)")
    return out


# ---------------------------------------------------------------------------
# CMR prefetch
# ---------------------------------------------------------------------------


def prefetch_glah12_granules(
    *,
    aoi_polygon,
    granule_dir: Path,
    strip_dates: list[pd.Timestamp],
    strip_match_window_days: int = 365,
    workers: int = 4,
    aoi_buffer_m: float = 50_000.0,
    dry_run: bool = False,
) -> dict:
    r"""Query CMR for GLAH12 granules over the basin AOI whose acquisition
    time lies within ±``strip_match_window_days`` of any strip date, then
    download missing ones to the shared granule store.

    Returns a counts dict ``{"matched", "have", "downloaded", "bytes"}``.
    """
    granule_dir = Path(granule_dir)
    granule_dir.mkdir(parents=True, exist_ok=True)

    counts = {"matched": 0, "have": 0, "downloaded": 0, "bytes": 0}
    if not strip_dates:
        print("  no strips → no GLAH12 prefetch")
        return counts

    half = pd.Timedelta(days=strip_match_window_days)
    t_lo = max(min(strip_dates) - half, GLAS_MISSION_START)
    t_hi = min(max(strip_dates) + half, GLAS_MISSION_END)
    if t_lo >= t_hi:
        print(f"  strip window ±{strip_match_window_days}d misses the GLAS "
              f"mission ({GLAS_MISSION_START.date()} → {GLAS_MISSION_END.date()}); "
              f"nothing to prefetch")
        return counts

    bbox = _aoi_bbox_lonlat(aoi_polygon, buffer_m=aoi_buffer_m)
    temporal = (
        t_lo.strftime("%Y-%m-%dT00:00:00Z"),
        t_hi.strftime("%Y-%m-%dT00:00:00Z"),
    )
    print(f"  CMR query: {GLAH12_SHORT_NAME} v{GLAH12_VERSION} bbox={bbox}  "
          f"{temporal[0]} .. {temporal[1]}")
    granules = query_nsidc_granules(
        GLAH12_SHORT_NAME,
        version=GLAH12_VERSION,
        bounding_box=bbox,
        temporal=temporal,
    )
    if not granules:
        # Version string mismatches (e.g. a future re-release) shouldn't
        # silently produce an empty control base.
        print(f"  CMR returned 0 granules for v{GLAH12_VERSION}; retrying "
              f"without a version filter")
        granules = query_nsidc_granules(
            GLAH12_SHORT_NAME, bounding_box=bbox, temporal=temporal,
        )
    print(f"  CMR returned {len(granules)} GLAH12 granules")

    # Match on CMR's begin_time (filenames carry no calendar date).
    sorted_dates = sorted(pd.Timestamp(d) for d in strip_dates)
    kept = []
    for g in granules:
        bt = g.get("begin_time")
        if not bt:
            kept.append(g)  # no time metadata — keep, index sorts it out
            continue
        t = pd.Timestamp(bt).tz_localize(None)
        if any(abs(t - sd) <= half for sd in sorted_dates):
            kept.append(g)
    print(f"  per-strip ±{strip_match_window_days}d match: "
          f"{len(kept)}/{len(granules)} granules retained")
    granules = kept
    counts["matched"] = len(granules)
    if not granules:
        return counts

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
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="glasdl") as ex:
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
                print(f"  [glas] {n_done}/{len(granules)} done "
                      f"({counts['downloaded']} new, {counts['bytes']/1e6:.1f} MB, "
                      f"{elapsed/60:.1f} min)", flush=True)

    print(f"  glas prefetch done: have={counts['have']} "
          f"new={counts['downloaded']}  ({counts['bytes']/1e6:.1f} MB)")
    return counts


# ---------------------------------------------------------------------------
# Per-strip caching
# ---------------------------------------------------------------------------


def cache_one_strip_glas(
    strip_path: Path,
    center_time: pd.Timestamp,
    *,
    cache_dir: Path,
    granule_dir: Path,
    bedmachine_path: Path,
    velocity_path: Path,
    time_window_days: int = 365,
    strip_crop_buffer_m: float = 5_000.0,
    overwrite: bool = False,
    displacement_budget_m: float | None = 10.0,
    granule_index: pd.DataFrame | None = None,
) -> str:
    r"""Per-strip filter + cache. Writes
    ``<cache_dir>/glas_filtered_<dem_id>.csv`` in the ASP-ready
    ``easting,northing,h_mean`` schema. Returns ``ok|skip|fail|empty``.
    """
    dem_id = strip_path.stem
    expected = Path(cache_dir) / f"glas_filtered_{dem_id}.csv"
    if expected.exists() and not overwrite:
        print(f"⏭  {dem_id}: glas cache hit, skipping (use --overwrite)")
        return "skip"

    print(f"\n=== {center_time.date()}  {dem_id}  [glas]  ===")
    if granule_index is None:
        granule_index = build_granule_index(granule_dir, verbose=False)
    if len(granule_index) == 0:
        print("!! empty GLAH12 granule store — run the prefetch step first")
        return "fail"

    win_lo = center_time - pd.Timedelta(days=time_window_days)
    win_hi = center_time + pd.Timedelta(days=time_window_days)
    sel = granule_index[
        (pd.to_datetime(granule_index["t1"]) >= win_lo)
        & (pd.to_datetime(granule_index["t0"]) <= win_hi)
    ]
    if len(sel) == 0:
        print(f"!! no GLAH12 granules in ±{time_window_days}d for {dem_id}")
        return "fail"
    print(f"  {len(sel)} GLAH12 granules in temporal window")

    aoi = _strip_bbox(strip_path)
    bx = aoi.bounds

    parts: list[PCData] = []
    for _, row in sel.iterrows():
        gpath = Path(granule_dir) / str(row["name"])
        if not gpath.exists():
            continue
        try:
            pc = from_glah12(gpath)
        except Exception as exc:
            print(f"  !! parse failed on {gpath.name}: {exc}")
            traceback.print_exc()
            continue
        if pc.size == 0:
            continue
        e = np.asarray(pc.x); n = np.asarray(pc.y)
        in_box = (
            (e >= bx[0] - strip_crop_buffer_m) & (e <= bx[2] + strip_crop_buffer_m)
            & (n >= bx[1] - strip_crop_buffer_m) & (n <= bx[3] + strip_crop_buffer_m)
        )
        if not in_box.any():
            continue
        sub = {f: np.asarray(getattr(pc, f))[in_box] for f in pc.fields}
        pc_box = PCData().from_dict(sub)
        # Shean 2019 displacement budget |v|·|Δt| ≤ B, with Δt the
        # strip↔granule gap (granule midpoint; campaigns are ~1 month so
        # the midpoint is representative to ~2 weeks).
        g_mid = pd.Timestamp(row["t0"]) + (pd.Timestamp(row["t1"]) - pd.Timestamp(row["t0"])) / 2
        if displacement_budget_m is not None:
            delta_t_years = abs((center_time - g_mid).total_seconds()) / (365.25 * 86400.0)
            delta_t_years = max(delta_t_years, 1.0 / 365.25)
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
        print(f"!! glas cache empty after filtering for {dem_id}")
        return "empty"

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
    p.add_argument("--time-window", type=int, default=365,
                   help="±N day temporal half-window for GLAS↔strip match "
                        "(default 365 = the Shean ±1 yr control cap)")
    p.add_argument("--prefetch-only", action="store_true",
                   help="Run only the CMR + download step, then exit")
    p.add_argument("--no-prefetch", action="store_true",
                   help="Skip the CMR + download step; use only granules already on disk")
    p.add_argument("--prefetch-dry-run", action="store_true",
                   help="Query CMR but skip downloading missing granules")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel Earthdata downloads")
    p.add_argument("--displacement-budget-m", type=float, default=10.0,
                   help="Shean 2019 displacement-budget B for the per-granule "
                        "velocity filter (|v|·|Δt| ≤ B); set to 0 to disable")
    return p


def run_cache_glas_main(
    *,
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
    r"""Per-basin entry point. Wires CLI flags to the CMR prefetch + the
    per-strip GLAH12 filtering. Mirrors
    :func:`stereo_melt.coregister.cache_airborne.run_cache_airborne_main`.
    """
    if description is None:
        description = "Cache ICESat-1 GLAS (GLAH12) GCPs for a basin's REMA strips."
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

    print(f"AOI bounds (EPSG:3031): {aoi_polygon.bounds}")
    print(f"Strip window: {start_time} .. {end_time}")

    todo = find_in_window_strips(strips_dir, start_time, end_time, aoi_polygon=aoi_polygon)
    print(f"Found {len(todo)} strips to consider for GLAS caching.")
    if args.dem_id is not None:
        todo = [(p, t) for p, t in todo if p.stem == args.dem_id]
        if not todo:
            raise SystemExit(f"--dem-id {args.dem_id!r} not in window")
    if args.limit is not None:
        todo = todo[: args.limit]

    if not args.no_prefetch:
        prefetch_glah12_granules(
            aoi_polygon=aoi_polygon,
            granule_dir=granule_dir,
            strip_dates=[t for _, t in todo],
            strip_match_window_days=args.time_window,
            workers=args.workers,
            dry_run=args.prefetch_dry_run,
        )
        if args.prefetch_only:
            return 0

    granule_index = build_granule_index(granule_dir)

    n = len(todo)
    counts = {"ok": 0, "skip": 0, "fail": 0, "empty": 0}
    budget = args.displacement_budget_m if args.displacement_budget_m > 0 else None
    for i, (p, t) in enumerate(todo, start=1):
        result = cache_one_strip_glas(
            p, t,
            cache_dir=cache_dir,
            granule_dir=granule_dir,
            bedmachine_path=bedmachine_path,
            velocity_path=velocity_path,
            time_window_days=args.time_window,
            overwrite=args.overwrite,
            displacement_budget_m=budget,
            granule_index=granule_index,
        )
        counts[result] += 1
        print(f"[{i}/{n}] {t.date()}  {p.stem}  -> {result.upper()}  "
              f"(ok={counts['ok']} skip={counts['skip']} "
              f"empty={counts['empty']} fail={counts['fail']})")

    print(f"\n=== GLAS cache summary: {counts['ok']} new, "
          f"{counts['skip']} cached, {counts['empty']} empty, "
          f"{counts['fail']} failed (of {n}) ===")
    return 1 if counts["fail"] else 0
