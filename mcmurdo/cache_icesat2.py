"""Pre-cache ICESat-2 ATL06 control points for every McMurdo strip.

Run once before ``mcmurdo.align_strips``. Each in-window strip gets a
filtered IS2 cache at::

    <ASP_ROOT>/icesat2_data/icesat2_filtered_<dem_id>.h5

where ``ASP_ROOT`` is the per-basin ASP working tree
(``mcmurdo/data/ASP/``). ``.csv`` legacy caches from before the
HDF5 migration are still honoured as cache hits.

``align_strips`` then reads from this cache and never touches sliderule
itself, so a transient IS2 download failure can no longer take a strip
out of the align batch.

Run:

    python -m mcmurdo.cache_icesat2                  # all in-window strips
    python -m mcmurdo.cache_icesat2 --overwrite      # refresh every strip
    python -m mcmurdo.cache_icesat2 --dem-id <id>    # single strip
    python -m mcmurdo.cache_icesat2 --limit 3        # smoke test first 3
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

# Point pyproj at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

import fiona
import pandas as pd
from shapely.geometry import shape

from stereo_melt.coregister.reference import download_icesat2_data

from mcmurdo import config

DATE_RE = re.compile(r"_(\d{8})_")


def _date_from_path(p: Path) -> pd.Timestamp | None:
    m = DATE_RE.search(p.name)
    return pd.Timestamp(m.group(1)) if m else None


def _strip_bbox(strip_path: Path):
    import rasterio
    from shapely.geometry import box

    with rasterio.open(strip_path) as src:
        bx = src.bounds
    return box(bx.left, bx.bottom, bx.right, bx.top)


def find_in_window_strips(
    aoi_polygon=None,
    grounded_polygon=None,
) -> list[tuple[Path, pd.Timestamp]]:
    """Return all in-window strips that intersect the AOI ∩ grounded ice."""
    strips_dir = Path(config.STRIPS_DIR)
    t_start = pd.Timestamp(config.START_TIME)
    t_end = pd.Timestamp(config.END_TIME)

    todo: list[tuple[Path, pd.Timestamp]] = []
    for p in sorted(strips_dir.glob("SETSM_*.tif")):
        if p.stem.endswith(("_matchtag", "_bitmask")):
            continue
        t = _date_from_path(p)
        if t is None or not (t_start <= t < t_end):
            continue
        sb = _strip_bbox(p)
        if aoi_polygon is not None and not sb.intersects(aoi_polygon):
            continue
        if grounded_polygon is not None and not sb.intersects(grounded_polygon):
            continue
        todo.append((p, t))
    return sorted(todo, key=lambda x: x[1])


def _load_aoi() -> "shape":
    with fiona.open(config.MCMURDO_AOI_SHP) as src:
        return shape(next(iter(src))["geometry"])


def cache_one(
    strip_path: Path,
    center_time: pd.Timestamp,
    cache_dir: Path,
    overwrite: bool = False,
) -> str:
    """Populate the IS2 cache for one strip; return ``ok|skip|fail``."""
    dem_id = strip_path.stem
    h5_path = cache_dir / f"icesat2_filtered_{dem_id}.h5"
    csv_path = cache_dir / f"icesat2_filtered_{dem_id}.csv"  # legacy
    if (h5_path.exists() or csv_path.exists()) and not overwrite:
        which = "h5" if h5_path.exists() else "csv (legacy)"
        print(f"⏭  {dem_id}: cache hit ({which}), skipping (use --overwrite to refresh)")
        return "skip"

    print(f"\n=== {center_time.date()}  {dem_id} ===")
    try:
        out = download_icesat2_data(
            strip_boundary=_strip_bbox(strip_path),
            grounded_shapefile=str(config.GROUNDING_LINE_SHP),
            dem_center_date=center_time.strftime("%Y-%m-%d"),
            output_dir=str(cache_dir),
            time_window=config.IS2_TIME_WINDOW_DAYS,
            velocity_path=str(config.MEASURES_PHASE_NC),
            bedmachine_path=str(config.BEDMACHINE_NC),
            max_speed_myr=config.IS2_MAX_SPEED_MYR,
            displacement_budget_m=getattr(config, "IS2_DISPLACEMENT_BUDGET_M", None),
            dem_id=dem_id,
        )
    except Exception as exc:
        print(f"!! IS2 cache FAILED for {dem_id}: {exc}")
        traceback.print_exc()
        return "fail"
    if not out:
        print(f"!! IS2 cache produced no rows for {dem_id} (empty after filter)")
        return "fail"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N strips (smoke test).")
    parser.add_argument("--dem-id", type=str, default=None,
                        help="Process only the strip with this dem_id.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Refresh cache entries even if they already exist.")
    args = parser.parse_args()

    config.ensure_output_dirs()

    asp_root = Path(config.ASP_ROOT)
    cache_dir = asp_root / "icesat2_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    aoi_polygon = _load_aoi()
    print(f"AOI polygon bounds (EPSG:3031): {aoi_polygon.bounds}")

    import geopandas as gpd
    grounded_polygon = gpd.read_file(config.GROUNDING_LINE_SHP).iloc[0].geometry

    todo = find_in_window_strips(
        aoi_polygon=aoi_polygon,
        grounded_polygon=grounded_polygon,
    )
    print(f"Window {config.START_TIME} .. {config.END_TIME}")
    print(f"Found {len(todo)} strips to consider for IS2 caching.")

    if args.dem_id is not None:
        todo = [(p, t) for p, t in todo if p.stem == args.dem_id]
        if not todo:
            raise SystemExit(f"--dem-id {args.dem_id!r} not in window")
    if args.limit is not None:
        todo = todo[: args.limit]

    n = len(todo)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    for i, (p, t) in enumerate(todo, start=1):
        result = cache_one(p, t, cache_dir, overwrite=args.overwrite)
        counts[result] += 1
        print(f"[{i}/{n}] {t.date()}  {p.stem}  -> {result.upper()}  "
              f"(ok={counts['ok']} skip={counts['skip']} fail={counts['fail']})")

    print(f"\n=== IS2 cache summary: {counts['ok']} new, "
          f"{counts['skip']} already cached, {counts['fail']} failed "
          f"(of {n} strips) ===")
    if counts["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
