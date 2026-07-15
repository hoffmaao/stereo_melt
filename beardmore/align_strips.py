"""Batch ASP coregistration of downloaded REMA strips for Beardmore.

Stage 1 of the Beardmore pipeline. Iterates over ``config.STRIPS_DIR``,
filters to strips in ``[START_TIME, END_TIME)`` whose ASP output
(``*-trans_reference-DEM.tif`` in ``STRIP_ALIGNED_DIR``) is missing,
and runs :func:`stereo_melt.coregister.asp.align_strip` on each to
download IS2 control + extract rock elevations + run ``pc_align``.

Run:

    python -m beardmore.align_strips                 # all unaligned strips
    python -m beardmore.align_strips --limit 1       # one strip (smoke test)
    python -m beardmore.align_strips --dem-id <id>   # specific strip by id
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
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import fiona
import pandas as pd
from shapely.geometry import shape

from stereo_melt.coregister.asp import align_strip

from beardmore import config

DATE_RE = re.compile(r"_(\d{8})_")


def _date_from_path(p: Path) -> pd.Timestamp | None:
    m = DATE_RE.search(p.name)
    return pd.Timestamp(m.group(1)) if m else None


def _strip_bbox(strip_path: Path):
    """Return strip footprint as a shapely box in EPSG:3031."""
    import rasterio
    from shapely.geometry import box

    with rasterio.open(strip_path) as src:
        bx = src.bounds
    return box(bx.left, bx.bottom, bx.right, bx.top)


def find_unaligned_strips(
    aoi_polygon=None,
    grounded_polygon=None,
    aligned_dir: Path | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> tuple[list[tuple[Path, pd.Timestamp]], list[tuple[Path, pd.Timestamp]], list[tuple[Path, pd.Timestamp]]]:
    """Return ``(todo, outside_aoi, no_grounded)`` for in-window unaligned strips.

    Strips need to (a) bbox-intersect ``aoi_polygon`` (otherwise they're
    not actually for our basin -- pdemtools' bbox search can be loose),
    and (b) bbox-intersect ``grounded_polygon`` (otherwise they have no
    grounded ice and can't be coregistered against IS2 control).

    ``aligned_dir`` overrides ``config.STRIP_ALIGNED_DIR`` so a
    non-default ASP run (e.g. ``--asp-suffix=_cs2``) can detect its own
    completed strips without re-doing them.

    ``start_time`` / ``end_time`` (``YYYY-MM-DD``) override the basin's
    config window, used to era-split the IS2-era (post-Oct 2018,
    ``--control is2+cs2``) from the pre-IS2 era (``--control cs2``)
    so each writes into its own ASP root.
    """
    strips_dir = Path(config.STRIPS_DIR)
    aligned_dir = Path(aligned_dir) if aligned_dir is not None else Path(config.STRIP_ALIGNED_DIR)
    t_start = pd.Timestamp(start_time) if start_time is not None else pd.Timestamp(config.START_TIME)
    t_end = pd.Timestamp(end_time) if end_time is not None else pd.Timestamp(config.END_TIME)

    todo: list[tuple[Path, pd.Timestamp]] = []
    outside_aoi: list[tuple[Path, pd.Timestamp]] = []
    no_grounded: list[tuple[Path, pd.Timestamp]] = []
    for p in sorted(strips_dir.glob("SETSM_*.tif")):
        # Skip PGC quality companion files — the glob also matches
        # *_matchtag.tif and *_bitmask.tif, which are binary masks and
        # would be nonsense to feed to pc_align.
        if p.stem.endswith(("_matchtag", "_bitmask")):
            continue
        t = _date_from_path(p)
        if t is None or not (t_start <= t < t_end):
            continue
        out = aligned_dir / f"{p.stem}-trans_reference-DEM.tif"
        if out.exists():
            continue
        # Sentinel left by stereo_melt.coregister.asp._quarantine_alignment
        # when a previous run produced |Δ| > max_displacement; skip so we
        # don't burn 30-60 min retrying a known-bad strip every batch.
        if (aligned_dir / f"{p.stem}.bad_align").exists():
            continue
        sb = _strip_bbox(p)
        if aoi_polygon is not None and not sb.intersects(aoi_polygon):
            outside_aoi.append((p, t))
            continue
        if grounded_polygon is not None and not sb.intersects(grounded_polygon):
            no_grounded.append((p, t))
            continue
        todo.append((p, t))
    return (
        sorted(todo, key=lambda x: x[1]),
        sorted(outside_aoi, key=lambda x: x[1]),
        sorted(no_grounded, key=lambda x: x[1]),
    )


def _load_aoi() -> "shape":
    # Strip selection uses the wider analysis-grid AOI (same one
    # fetch_strips uses), not the narrow grounding-zone polygon. Narrow
    # AOI was over-filtering: ~80% of fetched strips fell outside it,
    # leaving them un-aligned. Per-point IS2/CS2/GCP queries are already
    # strip-extent-only inside the lib (asp.align_strip + cache_*).
    with fiona.open(config.BEARDMORE_STACK_AOI_SHP) as src:
        return shape(next(iter(src))["geometry"])


def align_one(
    strip_path: Path,
    center_time: pd.Timestamp,
    asp_root: Path,
    aoi_polygon=None,
    use_is2: bool = True,
    cs2_h5_path: Path | None = None,
) -> bool:
    """Run :func:`align_strip` on a single strip; return True on success.

    The strip is fed raw to ASP ``pc_align``. Tide and IBE corrections
    are deferred to the post-coregistration analysis-grid stage (see
    :mod:`stereo_melt.corrections.post_coreg`); pc_align is anchored on
    grounded IS2 ATL06 (and optionally CS2 SARIn POCA) + rock-outcrop
    GCPs which are tide-free by construction, so no pre-ASP time-varying
    correction is needed.
    """
    print(f"\n=== {center_time.date()}  {strip_path.name} ===")
    try:
        align_strip(
            file_path=str(strip_path),
            corrections=None,
            x_min=None, x_max=None, y_min=None, y_max=None,
            mosaic_dir=str(config.MOSAIC_DIR),
            rock_shapefile=str(config.ROCK_POLYGONS_SHP),
            output_dir=str(asp_root),
            model=config.TIDE_MODEL,
            center_time=center_time,
            aoi_polygon=aoi_polygon,
            use_is2=use_is2,
            cs2_h5_path=str(cs2_h5_path) if cs2_h5_path is not None else None,
        )
    except Exception as exc:
        print(f"!! FAILED for {strip_path.name}: {exc}")
        traceback.print_exc()
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N strips (smoke test).")
    parser.add_argument("--dem-id", type=str, default=None,
                        help="Process only the strip with this dem_id.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of strips to align in parallel "
                             "(default 1 = serial).")
    parser.add_argument("--control", type=str, default="is2",
                        choices=("is2", "cs2", "is2+cs2"),
                        help="Altimetric control source(s) for pc_align: "
                             "is2=IS2 ATL06 only (default); cs2=CS2 SARIn POCA "
                             "only; is2+cs2=both concatenated.")
    parser.add_argument("--asp-suffix", type=str, default="",
                        help="Suffix for the ASP output dir name (results land "
                             "at <STRIPS_DIR>/ASP<suffix>/). Use to keep CS2-only "
                             "and combined runs separate from the canonical IS2-only "
                             "results. Caches (icesat2_data/, cs2_data/) are still "
                             "read from the canonical <STRIPS_DIR>/ASP/ tree.")
    parser.add_argument("--start", type=str, default=None,
                        help="YYYY-MM-DD start (overrides config.START_TIME). "
                             "Use with --end to era-split alignment runs (pre-Oct "
                             "2018 with --control cs2, post with --control is2+cs2).")
    parser.add_argument("--end", type=str, default=None,
                        help="YYYY-MM-DD end (overrides config.END_TIME). See --start.")
    args = parser.parse_args()

    config.ensure_output_dirs()

    canonical_asp = Path(config.STRIPS_DIR) / "ASP"
    asp_root = Path(config.STRIPS_DIR) / f"ASP{args.asp_suffix}"
    asp_root.mkdir(parents=True, exist_ok=True)
    (asp_root / "asp_aligned").mkdir(parents=True, exist_ok=True)

    # When running into a suffixed ASP dir, link the IS2 cache from the
    # canonical tree so align_strip's lookup at <output_dir>/icesat2_data
    # still finds it. CS2 h5 paths are passed in absolute, no symlink needed.
    if args.asp_suffix:
        is2_link = asp_root / "icesat2_data"
        if not is2_link.exists():
            os.symlink(canonical_asp / "icesat2_data", is2_link)

    use_is2 = "is2" in args.control
    use_cs2 = "cs2" in args.control
    cs2_cache_dir = canonical_asp / "cs2_data" if use_cs2 else None
    print(f"Control source: {args.control}  "
          f"(use_is2={use_is2}, use_cs2={use_cs2})")
    print(f"ASP output: {asp_root}")

    aoi_polygon = _load_aoi()
    print(f"AOI polygon bounds (EPSG:3031): {aoi_polygon.bounds}")

    import geopandas as gpd
    grounded_polygon = gpd.read_file(config.GROUNDING_LINE_SHP).iloc[0].geometry

    todo, outside, no_grounded = find_unaligned_strips(
        aoi_polygon=aoi_polygon,
        grounded_polygon=grounded_polygon,
        aligned_dir=asp_root / "asp_aligned",
        start_time=args.start,
        end_time=args.end,
    )
    eff_start = args.start or config.START_TIME
    eff_end = args.end or config.END_TIME
    print(f"Window {eff_start} .. {eff_end}")
    print(f"Found {len(todo)} strips needing ASP coregistration (AOI ∩ grounded).")
    for p, t in todo:
        print(f"  {t.date()}  {p.stem}")
    if outside:
        print(f"\nSkipping {len(outside)} downloaded strips outside AOI:")
        for p, t in outside:
            print(f"  {t.date()}  {p.stem}")
    if no_grounded:
        print(f"\nSkipping {len(no_grounded)} strips that don't reach grounded ice (no IS2 control possible):")
        for p, t in no_grounded:
            print(f"  {t.date()}  {p.stem}")

    if args.dem_id is not None:
        todo = [(p, t) for p, t in todo if p.stem == args.dem_id]
        if not todo:
            raise SystemExit(f"--dem-id {args.dem_id!r} not in unaligned set")
    if args.limit is not None:
        todo = todo[: args.limit]

    def _cs2_h5_for(stem: str) -> Path | None:
        if cs2_cache_dir is None:
            return None
        path = cs2_cache_dir / f"cs2_filtered_{stem}.h5"
        if not path.exists():
            print(f"  ⚠  CS2 cache miss for {stem}; skipping (run cache_cryosat2 first)")
            return None
        return path

    n_total = len(todo)
    if args.parallel > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"\nParallel align with {args.parallel} workers over {n_total} strips...")
        n_ok = 0
        n_done = 0
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(
                    align_one, p, t, asp_root,
                    aoi_polygon=aoi_polygon,
                    use_is2=use_is2,
                    cs2_h5_path=_cs2_h5_for(p.stem),
                ): (p, t)
                for p, t in todo
            }
            for fut in as_completed(futures):
                p, t = futures[fut]
                n_done += 1
                try:
                    ok = fut.result()
                except Exception as exc:
                    print(f"!! worker raised for {p.name}: {exc}")
                    ok = False
                if ok:
                    n_ok += 1
                tag = "OK" if ok else "FAIL"
                print(f"[{n_done}/{n_total}] {t.date()}  {p.stem}  -> {tag}  "
                      f"(running tally: {n_ok} ok)")
    else:
        n_ok = 0
        for i, (p, t) in enumerate(todo, start=1):
            print(f"\n[{i}/{n_total}] starting...")
            if align_one(
                p, t, asp_root,
                aoi_polygon=aoi_polygon,
                use_is2=use_is2,
                cs2_h5_path=_cs2_h5_for(p.stem),
            ):
                n_ok += 1

    print(f"\n=== batch summary: {n_ok}/{n_total} succeeded ===")


if __name__ == "__main__":
    main()
