# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""ESA CryoTEMPO Land Ice (TDP_LI) cache: per-strip altimetric control points.

This is the CryoSat-2 control base we adopt **instead of** re-retracking
raw L2 (see :mod:`stereo_melt.coregister.cryotempo` and the retracker
bake-off: a single-waveform leading-edge retracker gave 7-11 m scatter vs
IS2 on clean grounded ice, while CryoTEMPO's LMC retracker + interferometric
POCA gives 0.82 m MAD with ``uncertainty<1 m``). It slots into the existing
ASP pipeline exactly where :mod:`stereo_melt.coregister.cache_cs2` did:
per-strip filtered control written to HDF5 so ``align_strips`` consumes it
through the unchanged ``_cs2_h5_to_csv`` path.

We reuse the ``cache_cs2`` machinery wholesale — strip discovery, the
BedMachine-grounded ∩ slow-velocity ∩ REMA-residual filter, and the BURGEE
distribution test — so a CryoTEMPO control point lives in exactly the same
regime as an IS2 or raw-CS2 control point. Two pieces are new:

1. **Prefetch.** TDP_LI granules carry start/stop time + cycle + orbit in
   their filename but **no lat/lon and no HDR companions**, so they cannot
   be spatially pre-filtered from the filename alone. Instead we reuse the
   raw L2 SARIn **HDRs already on disk** (downloaded by ``cache_cs2``) as a
   spatial index: an HDR carries the pass's segment lat/lon, and the TDP_LI
   granule for the same orbit segment shares that pass's time span. So we
   find raw-L2 passes that match a strip spatially+temporally and download
   only the TDP_LI granules whose time span overlaps one of them. (This is
   the path the 2026-06-09 validation used; see ``/tmp/validate_cryotempo.py``.)

2. **Uncertainty gate.** CryoTEMPO ships a per-point ``uncertainty`` (m);
   gating to ``< uncertainty_max`` (default 3 m) is the coverage-maximizing
   cut. The tight ``<1 m`` cut (validated 2026-06-09: ~0.82 m MAD vs IS2,
   roughly half the ~1.1 m ungated scatter) was the original default, but it
   is the *binding* constraint on strip coverage — on fast basins like PIG it
   thins many ≥80-point strips below the BURGEE distribution threshold and
   drops them entirely (2026-06-17 diagnosis: CryoTEMPO control was a strict
   subset of raw-CS2, 353 vs 701 strips, ~210 dropped purely by this cut).
   Loosening to ``<3 m`` recovers those strips at ~1.1 m per-point precision —
   far better than the no-control alternative for the pre-IS2 era that has no
   IS2 anchor. The h5 **retains** per-point ``uncertainty``, so a tighter
   downstream re-filter — or future per-point inverse-variance ICP weighting —
   stays a cheap post-step and the cache remains a strict superset of the
   ``<1 m`` set (loosened 2026-06-18). The matching align-time quality lever is
   ``ICP_SIGMA_PER_SOURCE_M['cryotempo']`` (control_source.py), which should
   rise 0.82 → ~1.1 to reflect the looser cut's per-point scatter.

Output: ``cryotempo_filtered_<dem_id>.h5`` under a per-basin
``cryotempo_data/`` cache (parallel to ``cs2_data/`` so the raw-L2 control
is preserved for A/B comparison). The h5 carries ``x, y, h`` plus
``backscatter`` (the Schröder et al. 2019 penetration-correction input, which
CryoTEMPO does **not** apply) and ``uncertainty`` (per-point ICP weighting).

Per-basin wrappers under ``<basin>/cache_cryotempo.py`` pass their
config-supplied paths to :func:`run_cache_cryotempo_main` and are otherwise
~50-line glue.
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
from pointCollection.data import data as PCData

from ..io.esa_cs2_https import EsaCryoSatHttpsClient
from .cryotempo import read_cryotempo_li_granule
from .cache_cs2 import (
    DEFAULT_STRIP_CROP_BUFFER_M,
    _check_cs2_distribution,
    _parse_hdr_segment,
    _segment_intersects_lonlat_bbox,
    _strip_bbox,
    _strip_bbox_lonlat,
    _strip_windows_to_months,
    filter_cs2_grounded_slow_velocity,
    find_in_window_strips,
)

# Remote layout: ``TEMPO_POCA_LI`` (top level = latest baseline) → year →
# month → region. Swath/EOLIS lives under ``TEMPO_SWATH_*`` and is NOT POCA.
TEMPO_POCA_LI_REMOTE = "TEMPO_POCA_LI"

# Two ``YYYYmmddTHHMMSS`` tokens appear in both raw-L2 HDR names and TDP_LI
# granule names (start, stop). One regex parses the time span of either.
_TS_RE = re.compile(r"(\d{8}T\d{6})")

# Tolerance when overlap-matching a TDP_LI granule's [start, stop] against a
# raw-L2 pass span. TDP_LI and L2 segment the same acquisition slightly
# differently; a few seconds of slop absorbs the boundary mismatch.
_SPAN_PAD = pd.Timedelta(seconds=10)


def _name_span(name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Parse ``(start, stop)`` timestamps from an HDR / TDP_LI filename."""
    ts = _TS_RE.findall(name)
    if len(ts) < 2:
        return None
    return pd.Timestamp(ts[0]), pd.Timestamp(ts[1])


def _spans_overlap(a: tuple[pd.Timestamp, pd.Timestamp],
                   b: tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    return (a[0] <= b[1] + _SPAN_PAD) and (a[1] >= b[0] - _SPAN_PAD)


# ---------------------------------------------------------------------------
# Filtering: reuse the CS2 grounded ∩ slow ∩ residual filter + uncertainty gate
# ---------------------------------------------------------------------------


def filter_cryotempo_control(
    pc: PCData,
    bedmachine_path: Path,
    velocity_path: Path,
    *,
    uncertainty_max: float | None = 3.0,
    grounded_mask_value: int = 2,
    max_speed_myr: float = 10.0,
    vel_smooth_sigma_m: float = 2_000.0,
    rema_residual_gate_m: float | None = 100.0,
    dem_center_date=None,
    displacement_budget_m: float | None = None,
) -> PCData:
    r"""Filter CryoTEMPO control to the IS2/CS2 regime + an uncertainty gate.

    Delegates the grounded (BedMachine ``mask==2``) ∩ slow-velocity
    (smoothed MEaSUREs ``|v|<max_speed_myr``) ∩ REMA-residual screen to
    :func:`stereo_melt.coregister.cache_cs2.filter_cs2_grounded_slow_velocity`
    — identical to the raw-CS2 catalogue so the two control sources are
    directly comparable — then drops points with per-point
    ``uncertainty >= uncertainty_max`` (default 3 m, the coverage-maximizing
    cut; ``<1 m`` is the tight ~0.82 m-MAD selector but drops many strips
    below the BURGEE threshold — see module docstring). Set
    ``uncertainty_max=None`` to disable.
    """
    filt = filter_cs2_grounded_slow_velocity(
        pc, bedmachine_path, velocity_path,
        grounded_mask_value=grounded_mask_value,
        max_speed_myr=max_speed_myr,
        vel_smooth_sigma_m=vel_smooth_sigma_m,
        rema_residual_gate_m=rema_residual_gate_m,
        dem_center_date=dem_center_date,
        displacement_budget_m=displacement_budget_m,
    )
    if filt.size == 0 or uncertainty_max is None:
        return filt
    if "uncertainty" not in filt.fields:
        print("  ⚠  CryoTEMPO filter: no 'uncertainty' field; skipping unc gate")
        return filt
    unc = np.asarray(filt.uncertainty, dtype=np.float64)
    keep = np.isfinite(unc) & (unc < uncertainty_max)
    n_keep = int(keep.sum())
    print(f"  CryoTEMPO uncertainty gate: {n_keep}/{filt.size} kept "
          f"(unc<{uncertainty_max:.1f} m)")
    if n_keep == 0:
        return PCData().from_dict({"x": np.array([]), "y": np.array([]),
                                   "h": np.array([])})
    sub = {f: np.asarray(getattr(filt, f))[keep] for f in filt.fields}
    return PCData().from_dict(sub)


# ---------------------------------------------------------------------------
# Prefetch: raw-L2 HDR spatial index → time-match TDP_LI granules
# ---------------------------------------------------------------------------


def _relevant_l2_spans_for_month(
    strip_meta: list[dict],
    year: int,
    month: int,
    hdr_dir: Path,
    half_window: pd.Timedelta,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    r"""Time spans of raw-L2 SARIn passes (from local HDRs) that match any
    strip spatially **and** temporally in the given month.

    Uses the HDRs already downloaded by ``cache_cs2`` as a spatial index.
    Returns an empty list (and the caller skips the month) if no HDRs are
    on disk — log warns so the gap is visible.
    """
    local = hdr_dir / f"{year}" / f"{month:02d}"
    if not local.is_dir():
        return []
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    n_hdr = 0
    for hdr in sorted(local.glob("*.HDR")):
        if "SIR_SIN_2" not in hdr.name:
            continue
        n_hdr += 1
        sp = _name_span(hdr.name)
        if sp is None:
            continue
        seg = _parse_hdr_segment(hdr.read_bytes())
        if seg is None:
            continue
        for sm in strip_meta:
            if abs(sp[0] - sm["date"]) > half_window:
                continue
            if _segment_intersects_lonlat_bbox(seg, sm["lat"], sm["lon"]):
                spans.append(sp)
                break
    if n_hdr == 0:
        print(f"  ⚠  [{year}-{month:02d}] no raw-L2 HDRs on disk under {local}; "
              f"cannot spatially index TDP_LI for this month. Run "
              f"`cache_cryosat2 --prefetch-only` to populate HDRs first.",
              flush=True)
    return spans


def _prefetch_tempo_month(
    client: EsaCryoSatHttpsClient,
    year: int,
    month: int,
    strip_meta: list[dict],
    half_window: pd.Timedelta,
    hdr_dir: Path,
    granule_dir: Path,
    region: str,
    dry_run: bool,
) -> dict:
    tag = f"{year}-{month:02d}"
    counts = {"nc_have": 0, "nc_dl": 0, "nc_match": 0, "bytes_nc": 0}

    spans = _relevant_l2_spans_for_month(strip_meta, year, month, hdr_dir,
                                         half_window)
    if not spans:
        print(f"  [{tag}] no spatially+temporally matched L2 passes; skip",
              flush=True)
        return counts

    remote = f"{TEMPO_POCA_LI_REMOTE}/{year}/{month:02d}/{region}"
    try:
        entries = client.list_dir(remote)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠  [{tag}] list {remote}: {exc}", flush=True)
        return counts

    granules = [e for e in entries
                if e["name"].endswith(".nc") and "TDP_LI" in e["name"]]
    wanted = []
    for e in granules:
        sp = _name_span(e["name"])
        if sp is None:
            continue
        if any(_spans_overlap(sp, ls) for ls in spans):
            wanted.append(e)
    counts["nc_match"] = len(wanted)

    for e in wanted:
        local = granule_dir / e["name"]
        if local.exists():
            counts["nc_have"] += 1
            continue
        if dry_run:
            counts["nc_dl"] += 1
            continue
        try:
            counts["bytes_nc"] += client.download(e["path"], local)
            counts["nc_dl"] += 1
        except Exception as exc:  # noqa: BLE001
            print(f"    !! [{tag}] TDP_LI fetch failed {e['name']}: {exc}",
                  flush=True)
            continue

    print(f"  [{tag}] DONE: {len(spans)} matched L2 passes, "
          f"{len(granules)} TDP_LI listed, {len(wanted)} overlap → "
          f".nc {counts['nc_have']} cached + {counts['nc_dl']} new",
          flush=True)
    return counts


def prefetch_cryotempo_granules(
    strips: list[tuple[Path, pd.Timestamp]],
    granule_dir: Path,
    hdr_dir: Path,
    *,
    time_window_days: int = 90,
    strip_buffer_m: float = 25_000.0,
    region: str = "ANTARC",
    dry_run: bool = False,
    workers: int = 4,
) -> dict:
    r"""Walk ``TEMPO_POCA_LI`` and prefetch the TDP_LI granules whose orbit
    segment overlaps an in-window strip footprint.

    Spatial matching is delegated to the raw-L2 SARIn HDRs already on disk
    (downloaded by ``cache_cs2``): see :func:`_relevant_l2_spans_for_month`.
    Granules are ~140-460 KB each, so unlike the raw-L2 prefetch there is
    no HDR-first stage — the listing + time-overlap match is cheap and the
    download is the only network cost.
    """
    if not strips:
        print("  prefetch: no strips supplied; nothing to do")
        return {}

    granule_dir.mkdir(parents=True, exist_ok=True)

    strip_meta: list[dict] = []
    for p, t in strips:
        try:
            lat_r, lon_r = _strip_bbox_lonlat(p, buffer_m=strip_buffer_m)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠  could not bbox {p.name}: {exc}")
            continue
        strip_meta.append({"path": p, "date": pd.Timestamp(t),
                           "lat": lat_r, "lon": lon_r})

    print(f"  prefetch: {len(strip_meta)} strips, ±{time_window_days} d window, "
          f"strip bbox buffer {strip_buffer_m/1e3:.0f} km, region={region}",
          flush=True)

    months = _strip_windows_to_months([s["date"] for s in strip_meta],
                                      time_window_days)
    print(f"  prefetch covers {len(months)} (year, month) buckets: "
          f"{months[0][0]}-{months[0][1]:02d} → "
          f"{months[-1][0]}-{months[-1][1]:02d}; workers={workers}",
          flush=True)

    half_window = pd.Timedelta(days=time_window_days)
    totals = {"nc_have": 0, "nc_dl": 0, "nc_match": 0, "bytes_nc": 0}

    client = EsaCryoSatHttpsClient(timeout=120)
    n_workers = max(1, min(workers, len(months)))
    t_pool_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=n_workers,
                            thread_name_prefix="tempo") as ex:
        future_to_tag = {
            ex.submit(
                _prefetch_tempo_month,
                client, year, month, strip_meta, half_window,
                hdr_dir, granule_dir, region, dry_run,
            ): f"{year}-{month:02d}"
            for (year, month) in months
        }
        n_done = 0
        for fut in as_completed(future_to_tag):
            tag = future_to_tag[fut]
            try:
                c = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  !! [{tag}] future raised: {exc}", flush=True)
                continue
            for k, v in c.items():
                totals[k] += v
            n_done += 1
            print(f"  [pool] {n_done}/{len(months)} months done "
                  f"({(time.monotonic()-t_pool_start)/60:.1f} min elapsed)",
                  flush=True)

    print("\n  prefetch summary:")
    print(f"    TDP_LI: {totals['nc_have']} cached + {totals['nc_dl']} new "
          f"({totals['bytes_nc']/1e6:.1f} MB downloaded), "
          f"{totals['nc_match']} matched AOI passes")
    return totals


# ---------------------------------------------------------------------------
# Discovery + per-strip caching
# ---------------------------------------------------------------------------


def discover_cryotempo_granules(
    start_time: str,
    end_time: str,
    granule_dir: Path,
) -> list[Path]:
    r"""Find TDP_LI granules on disk whose start time is in ``[start, end)``.

    Spatial filtering is deferred to the per-strip crop + grounded filter
    in :func:`cache_one_strip`.
    """
    granule_dir = Path(granule_dir)
    granule_dir.mkdir(parents=True, exist_ok=True)
    on_disk = sorted(granule_dir.glob("CS_*TDP_LI*.nc"))
    if not on_disk:
        print(f"  ⚠  no CryoTEMPO TDP_LI granules on disk under {granule_dir}")
        return []

    t0 = pd.Timestamp(start_time)
    t1 = pd.Timestamp(end_time)
    keep = []
    for p in on_disk:
        sp = _name_span(p.name)
        if sp is None:
            continue
        if t0 <= sp[0] < t1:
            keep.append(p)
    print(f"  found {len(keep)}/{len(on_disk)} TDP_LI granules in window "
          f"{start_time}..{end_time}")
    return keep


def cache_one_strip(
    strip_path: Path,
    center_time: pd.Timestamp,
    *,
    cache_dir: Path,
    granule_dir: Path,
    bedmachine_path: Path,
    velocity_path: Path,
    time_window_days: int = 90,
    displacement_budget_m: float | None = None,
    strip_crop_buffer_m: float = DEFAULT_STRIP_CROP_BUFFER_M,
    overwrite: bool = False,
    uncertainty_max: float | None = 3.0,
    rema_residual_gate_m: float | None = 100.0,
    dist_min_count: int = 80,
    dist_min_spread_ns_m: float = 20_000.0,
    dist_min_spread_ew_m: float = 5_000.0,
) -> str:
    """Populate the CryoTEMPO cache for one strip.

    Returns one of ``ok|skip|fail|empty|sparse`` (same semantics as
    :func:`stereo_melt.coregister.cache_cs2.cache_one_strip`). Output is
    ``cryotempo_filtered_<dem_id>.h5``.
    """
    dem_id = strip_path.stem
    expected = cache_dir / f"cryotempo_filtered_{dem_id}.h5"
    if expected.exists() and not overwrite:
        print(f"⏭  {dem_id}: CryoTEMPO cache hit, skipping (use --overwrite)")
        return "skip"

    print(f"\n=== {center_time.date()}  {dem_id} ===")

    t0 = (center_time - pd.Timedelta(days=time_window_days)).strftime("%Y-%m-%d")
    t1 = (center_time + pd.Timedelta(days=time_window_days)).strftime("%Y-%m-%d")
    aoi = _strip_bbox(strip_path)

    granules = discover_cryotempo_granules(t0, t1, granule_dir)
    if not granules:
        print(f"!! no TDP_LI granules in window for {dem_id}; skipping")
        return "fail"

    parts: list[PCData] = []
    for nc in granules:
        try:
            pc = read_cryotempo_li_granule(nc)
        except Exception as exc:  # noqa: BLE001
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
        pc_filt = filter_cryotempo_control(
            pc_box, bedmachine_path, velocity_path,
            uncertainty_max=uncertainty_max,
            rema_residual_gate_m=rema_residual_gate_m,
            dem_center_date=center_time,
            displacement_budget_m=displacement_budget_m,
        )
        if pc_filt.size > 0:
            parts.append(pc_filt)

    if not parts:
        print(f"!! CryoTEMPO cache empty after filtering for {dem_id}")
        return "empty"

    all_fields = sorted({f for p in parts for f in p.fields})
    merged = {
        f: np.concatenate([np.asarray(getattr(p, f)) for p in parts if f in p.fields])
        for f in all_fields
    }
    out = PCData().from_dict(merged)

    if dist_min_count > 0:
        passed, reason = _check_cs2_distribution(
            out,
            min_count=dist_min_count,
            min_spread_ns_m=dist_min_spread_ns_m,
            min_spread_ew_m=dist_min_spread_ew_m,
        )
        if not passed:
            print(f"  ⚠  CryoTEMPO distribution test FAIL ({reason}); "
                  f"not caching for {dem_id}")
            return "sparse"
        print(f"  ✓ CryoTEMPO distribution test pass ({reason})")

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
    p.add_argument("--time-window", type=int, default=None,
                   help="±N day temporal half-window for TDP_LI-strip matching "
                        "(default: basin config value, else 90)")
    p.add_argument("--displacement-budget-m", type=float, default=None,
                   help="Shean advection budget (m): keep a CryoTEMPO point where "
                        "|v|·|Δt| ≤ this, retaining fast grounded ice with a "
                        "velocity-tightened window instead of a hard slow cut. "
                        "Default: basin config (None ⇒ legacy hard slow<10 m/yr cut).")
    p.add_argument("--strip-buffer-km", type=float, default=25.0,
                   help="Buffer (km) padded around each strip bbox for the "
                        "raw-L2 HDR segment match")
    p.add_argument("--region", type=str, default="ANTARC",
                   choices=("ANTARC", "GREENL"),
                   help="CryoTEMPO polar region subtree")
    p.add_argument("--prefetch-only", action="store_true",
                   help="Run only the HTTPS prefetch, then exit")
    p.add_argument("--no-prefetch", action="store_true",
                   help="Skip the HTTPS prefetch and use only granules on disk")
    p.add_argument("--prefetch-dry-run", action="store_true",
                   help="List what would be downloaded without downloading")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel ESA-server connections (one per month bucket)")
    p.add_argument("--uncertainty-max", type=float, default=3.0,
                   help="Drop CryoTEMPO points with per-point uncertainty ≥ this "
                        "(m). Default 3 m maximizes strip coverage; <1 m is the "
                        "tight ~0.82 m-MAD cut but drops many strips below the "
                        "BURGEE distribution threshold. The h5 retains per-point "
                        "uncertainty for downstream re-filter/weighting. 0 to disable.")
    p.add_argument("--rema-residual-gate-m", type=float, default=100.0,
                   help="Gate on |h - h_BedMachine_surface| (m). Lowery 2025 "
                        "uses 100 m. Set 0 to disable.")
    p.add_argument("--cs2-min-matchups", type=int, default=80,
                   help="BURGEE distribution test: min matchups per strip. "
                        "Set 0 to disable.")
    p.add_argument("--cs2-min-spread-ns-km", type=float, default=20.0,
                   help="BURGEE distribution test: min N-S spread (km).")
    p.add_argument("--cs2-min-spread-ew-km", type=float, default=5.0,
                   help="BURGEE distribution test: min E-W spread (km).")
    return p


def run_cache_cryotempo_main(
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
    time_window_days: int = 90,
    displacement_budget_m: float | None = None,
    description: str = "Cache CryoTEMPO Land Ice control points for a basin's REMA strips.",
    args: argparse.Namespace | None = None,
    fetch_window_env: tuple[str, str] | None = None,
    on_aoi_loaded: Callable[[], None] | None = None,
) -> int:
    r"""Per-basin entry point. Wires CLI flags to prefetch + per-strip caching.

    Mirrors :func:`stereo_melt.coregister.cache_cs2.run_cache_cs2_main`.

    Parameters
    ----------
    cache_dir : Path
        Per-basin CryoTEMPO cache output dir, e.g.
        ``<basin>/data/ASP/cryotempo_data/`` (parallel to ``cs2_data/``).
    granule_dir : Path
        Shared TDP_LI granule store, e.g. ``<MAIN_DIR>/data/CS2/tempo_li``.
    hdr_dir : Path
        Shared raw-L2 SARIn HDR store (``<MAIN_DIR>/data/CS2/hdrs``) — reused
        here as the spatial index, NOT re-downloaded.
    """
    if args is None:
        parser = _build_argparser(description)
        args = parser.parse_args()

    cache_dir = Path(cache_dir)
    granule_dir = Path(granule_dir)
    hdr_dir = Path(hdr_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    granule_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Found {len(todo)} strips to consider for CryoTEMPO caching.")

    # CLI flags override the basin-supplied defaults when explicitly set.
    eff_time_window = args.time_window if args.time_window is not None else time_window_days
    eff_budget = (args.displacement_budget_m if args.displacement_budget_m is not None
                  else displacement_budget_m)
    print("CryoTEMPO linking: "
          f"±{eff_time_window} d window, "
          + (f"displacement budget |v|·|Δt|≤{eff_budget:g} m (fast grounded ice kept, "
             f"velocity-tightened window)" if eff_budget is not None
             else "hard slow<10 m/yr cut (legacy)"))

    if args.dem_id is not None:
        todo = [(p, t) for p, t in todo if p.stem == args.dem_id]
        if not todo:
            raise SystemExit(f"--dem-id {args.dem_id!r} not in window")
    if args.limit is not None:
        todo = todo[: args.limit]

    if not args.no_prefetch:
        prefetch_cryotempo_granules(
            todo, granule_dir, hdr_dir,
            time_window_days=eff_time_window,
            strip_buffer_m=args.strip_buffer_km * 1_000.0,
            region=args.region,
            dry_run=args.prefetch_dry_run,
            workers=args.workers,
        )
        if args.prefetch_only:
            return 0

    n = len(todo)
    counts = {"ok": 0, "skip": 0, "fail": 0, "empty": 0, "sparse": 0}
    residual_gate = (args.rema_residual_gate_m
                     if args.rema_residual_gate_m > 0 else None)
    unc_max = args.uncertainty_max if args.uncertainty_max > 0 else None
    for i, (p, t) in enumerate(todo, start=1):
        result = cache_one_strip(
            p, t,
            cache_dir=cache_dir,
            granule_dir=granule_dir,
            bedmachine_path=bedmachine_path,
            velocity_path=velocity_path,
            time_window_days=eff_time_window,
            displacement_budget_m=eff_budget,
            overwrite=args.overwrite,
            uncertainty_max=unc_max,
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

    print(f"\n=== CryoTEMPO cache summary: {counts['ok']} new, "
          f"{counts['skip']} cached, {counts['sparse']} sparse, "
          f"{counts['empty']} empty, {counts['fail']} no-granules of {n} ===")
    return 1 if counts["fail"] else 0
