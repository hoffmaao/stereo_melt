# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""Shared batch-alignment driver behind every basin's ``align_strips``.

Before 2026-07-10 each basin package carried its own ~300-500 line
``align_strips.py``; seven copies had drifted apart (PIG grew
``--control none`` / ``--use-lvis`` / inverse-variance balancing,
beardmore_shelf grew the GLAS channel, and a control-coverage-filter
ordering bug was fixed in one copy but latent in the others). This module
is the single implementation — the union of those feature sets — and the
basin files are now thin wrappers passing config-supplied paths, exactly
like the ``run_cache_*_main`` pattern in :mod:`cache_airborne` /
:mod:`cache_glas` / :mod:`cache_cs2`.

Control-source model
--------------------
``--control`` picks the **spaceborne** base: ``is2`` (default), ``cs2``,
``is2+cs2``, or ``none``. ``--use-atm`` / ``--use-lvis`` / ``--use-glas``
additively attach the cached airborne / ICESat-1 channels. ``glas`` is
also accepted as a ``--control`` choice (sugar for ``none`` +
``--use-glas`` — the 2009 → Oct-2010 REMA back-extension era has no
spaceborne altimetry at all). Rock-outcrop GCPs are always included when
available.

When IS2 is not in the mix, strips with no control cache hit at all are
dropped up front as an explicit SKIP class — **before** ``--dem-id`` /
``--limit`` — so a pilot ``--limit N`` picks N strips that actually have
control (the 2026-07-10 beardmore_shelf pilot bug).

Basin wrapper contract (all config-derived, no basin branching here):
``aoi_shp, grounding_line_shp, strips_dir, canonical_asp_root,
mosaic_dir, rock_shapefile, tide_model, start_time, end_time`` plus the
basin's module name for actionable cache-miss hints.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd

DATE_RE = re.compile(r"_(\d{8})_")

# Conservative planning number for the disk-budget printout: slim-format
# aligned output plus transient pc_align workspace. Actual steady-state
# cost is ~0.2-0.5 GB/strip since the 2026-06-12 slim change.
PER_STRIP_GB = 3.5
MIN_FREE_GB_FLOOR = 100.0

CONTROL_CHOICES = ("is2", "cs2", "is2+cs2", "glas", "none")


def _disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


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
    strips_dir: Path,
    aligned_dir: Path,
    start_time: str,
    end_time: str,
    *,
    aoi_polygon=None,
    grounded_polygon=None,
):
    """Return ``(todo, outside_aoi, no_grounded)`` for in-window unaligned strips.

    Strips need to (a) bbox-intersect ``aoi_polygon`` (pdemtools' bbox
    search can be loose), and (b) bbox-intersect ``grounded_polygon``
    (otherwise no grounded altimetric control is possible). Strips whose
    ``*-trans_reference-DEM.tif`` already exists in ``aligned_dir``, or
    that carry a ``.bad_align`` quarantine sentinel from a previous
    diverged run, are excluded from ``todo``.
    """
    strips_dir = Path(strips_dir)
    aligned_dir = Path(aligned_dir)
    t_start = pd.Timestamp(start_time)
    t_end = pd.Timestamp(end_time)

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
        if (aligned_dir / f"{p.stem}-trans_reference-DEM.tif").exists():
            continue
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


def align_one(
    strip_path: Path,
    center_time: pd.Timestamp,
    asp_root: Path,
    *,
    mosaic_dir: Path,
    rock_shapefile: Path,
    tide_model: str,
    aoi_polygon=None,
    use_is2: bool = True,
    cs2_h5_path: Path | None = None,
    cs2_source_label: str = "cs2",
    atm_csv_path: Path | None = None,
    lvis_csv_path: Path | None = None,
    glas_csv_path: Path | None = None,
    cap_per_source: int | None = None,
    inverse_variance_balance: bool = False,
    inv_var_base_count: int = 5000,
) -> bool:
    """Run :func:`stereo_melt.coregister.asp.align_strip` on one strip.

    The strip is fed raw to ASP ``pc_align``; tide and IBE corrections
    are deferred to the post-coregistration analysis-grid stage. The
    control cloud combines grounded IS2 / CS2-or-CryoTEMPO / ATM / LVIS /
    GLAS altimetry (each optional) with rock-outcrop GCPs, all tide-free
    by construction. Returns True on success.
    """
    from stereo_melt.coregister.asp import align_strip

    print(f"\n=== {center_time.date()}  {strip_path.name} ===")
    try:
        align_strip(
            file_path=str(strip_path),
            corrections=None,
            x_min=None, x_max=None, y_min=None, y_max=None,
            mosaic_dir=str(mosaic_dir),
            rock_shapefile=str(rock_shapefile),
            output_dir=str(asp_root),
            model=tide_model,
            center_time=center_time,
            aoi_polygon=aoi_polygon,
            use_is2=use_is2,
            cs2_h5_path=str(cs2_h5_path) if cs2_h5_path is not None else None,
            cs2_source_label=cs2_source_label,
            atm_csv_path=str(atm_csv_path) if atm_csv_path is not None else None,
            lvis_csv_path=str(lvis_csv_path) if lvis_csv_path is not None else None,
            glas_csv_path=str(glas_csv_path) if glas_csv_path is not None else None,
            cap_per_source=cap_per_source,
            inverse_variance_balance=inverse_variance_balance,
            inv_var_base_count=inv_var_base_count,
        )
    except Exception as exc:
        print(f"!! FAILED for {strip_path.name}: {exc}")
        traceback.print_exc()
        return False
    return True


def _build_argparser(description: str, basin: str,
                     default_cs2_source: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N strips (smoke test).")
    p.add_argument("--dem-id", type=str, default=None,
                   help="Process only the strip with this dem_id.")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of strips to align in parallel (default 1 = serial).")
    p.add_argument("--control", type=str, default="is2", choices=CONTROL_CHOICES,
                   help="Spaceborne altimetric control for pc_align: is2=IS2 "
                        "ATL06 (default); cs2=CS2 SARIn POCA; is2+cs2=both; "
                        "none=no spaceborne control (combine with --use-atm/"
                        "--use-lvis/--use-glas); glas=shorthand for "
                        "'none + --use-glas' (the 2009 → 2010-10 ICESat-1 era). "
                        "Airborne/GLAS channels attach via --use-* flags and "
                        "combine into the same reference cloud.")
    p.add_argument("--cs2-source", type=str, default=default_cs2_source,
                   choices=("raw", "cryotempo"),
                   help="CS2 cache feeding pc_align when --control includes cs2: "
                        "cryotempo=ESA CryoTEMPO Land Ice LMC "
                        f"(cryotempo_data/, run {basin}.cache_cryotempo; production "
                        "default — 0.82 m MAD vs IS2), raw=ESA L2 SARIn POCA "
                        f"(cs2_data/, run {basin}.cache_cryosat2). The label is "
                        "recorded in the sources sidecar so tilt_fit Ez matches. "
                        f"Default: {default_cs2_source}.")
    p.add_argument("--use-atm", action="store_true",
                   help=f"Include cached IceBridge ATM GCPs from ASP/atm_data/ "
                        f"(run {basin}.cache_atm first).")
    p.add_argument("--use-lvis", action="store_true",
                   help=f"Include cached IceBridge LVIS GCPs from ASP/lvis_data/ "
                        f"(run {basin}.cache_lvis first).")
    p.add_argument("--use-glas", action="store_true",
                   help=f"Include cached ICESat-1 GLAS (GLAH12) GCPs from "
                        f"ASP/glas_data/ (run {basin}.cache_glas first).")
    p.add_argument("--cap-per-source", type=int, default=None,
                   help="Cap each control source at N rows before pc_align "
                        "(random subsample, fixed seed). Equal-weight balancing.")
    p.add_argument("--inv-variance-balance", action="store_true",
                   help="Balance per-source row counts ∝ 1/σ² from "
                        "ICP_SIGMA_PER_SOURCE_M (icepack-style observation "
                        "weighting). Mutually exclusive with --cap-per-source.")
    p.add_argument("--inv-var-base-count", type=int, default=5000,
                   help="Row target for the most-precise source under "
                        "--inv-variance-balance (default 5000).")
    p.add_argument("--asp-suffix", type=str, default="",
                   help="Suffix for the ASP output root (results land at "
                        "<BASIN>/data/ASP<suffix>/). Keeps variant runs separate "
                        "from the canonical root; control caches are still read "
                        "from the canonical ASP/ tree.")
    p.add_argument("--start", type=str, default=None,
                   help="YYYY-MM-DD start (overrides the basin config window). "
                        "Use with --end to era-split alignment runs.")
    p.add_argument("--end", type=str, default=None,
                   help="YYYY-MM-DD end (overrides the basin config window).")
    return p


def run_align_strips_main(
    *,
    basin: str,
    aoi_shp: Path,
    grounding_line_shp: Path,
    strips_dir: Path,
    canonical_asp_root: Path,
    mosaic_dir: Path,
    rock_shapefile: Path,
    tide_model: str,
    start_time: str,
    end_time: str,
    ensure_output_dirs=None,
    default_cs2_source: str = "cryotempo",
    description: str | None = None,
    args: argparse.Namespace | None = None,
) -> int:
    r"""Per-basin entry point for batch ASP coregistration.

    Parameters are the basin-config surface; ``basin`` is the package
    name (e.g. ``"pig"``) used only in help/hint text. Returns a process
    exit code (0 ok, 1 = nothing succeeded out of a non-empty batch).
    """
    if description is None:
        description = f"Batch ASP coregistration of downloaded REMA strips ({basin})."
    if args is None:
        parser = _build_argparser(description, basin, default_cs2_source)
        args = parser.parse_args()

    for label, path in (("AOI shapefile", aoi_shp),
                        ("grounding-line shapefile", grounding_line_shp),
                        ("strips dir", strips_dir)):
        if not Path(path).exists():
            raise SystemExit(
                f"❌ {label} not found at {path} — check {basin}/config.py."
            )

    if ensure_output_dirs is not None:
        ensure_output_dirs()

    # ``--control glas`` is sugar: no spaceborne base + the GLAS channel.
    control = args.control
    use_glas = bool(getattr(args, "use_glas", False)) or control == "glas"
    if control == "glas":
        control = "none"
    use_is2 = "is2" in control
    use_cs2 = "cs2" in control
    if control == "none" and not (args.use_atm or args.use_lvis or use_glas):
        raise SystemExit(
            "--control none requires at least one of --use-atm / --use-lvis / "
            "--use-glas (otherwise pc_align has no altimetric control)."
        )

    # Per-basin ASP root (feedback_per_basin_asp_root). The canonical
    # (unsuffixed) root holds the control caches; suffixed variant roots
    # sit alongside it under <BASIN>/data/.
    canonical_asp = Path(canonical_asp_root)
    asp_root = canonical_asp.with_name(f"ASP{args.asp_suffix}") if args.asp_suffix \
        else canonical_asp
    asp_root.mkdir(parents=True, exist_ok=True)
    (asp_root / "asp_aligned").mkdir(parents=True, exist_ok=True)

    # Suffixed roots read the IS2 cache through a symlink so align_strip's
    # <output_dir>/icesat2_data lookup still resolves.
    if args.asp_suffix:
        is2_link = asp_root / "icesat2_data"
        if not is2_link.exists() and (canonical_asp / "icesat2_data").exists():
            os.symlink(canonical_asp / "icesat2_data", is2_link)

    if use_cs2 and args.cs2_source == "cryotempo":
        cs2_cache_dir = canonical_asp / "cryotempo_data"
        cs2_filename_tmpl = "cryotempo_filtered_{stem}.h5"
        cs2_source_label = "cryotempo"
        cs2_driver_hint = f"{basin}.cache_cryotempo"
    elif use_cs2:
        cs2_cache_dir = canonical_asp / "cs2_data"
        cs2_filename_tmpl = "cs2_filtered_{stem}.h5"
        cs2_source_label = "cs2"
        cs2_driver_hint = f"{basin}.cache_cryosat2"
    else:
        cs2_cache_dir = None
        cs2_filename_tmpl = "cs2_filtered_{stem}.h5"
        cs2_source_label = "cs2"
        cs2_driver_hint = ""
    atm_cache_dir = (canonical_asp / "atm_data") if args.use_atm else None
    lvis_cache_dir = (canonical_asp / "lvis_data") if args.use_lvis else None
    glas_cache_dir = (canonical_asp / "glas_data") if use_glas else None

    sources_label = "+".join(filter(None, [
        control if control != "none" else None,
        "atm" if args.use_atm else None,
        "lvis" if args.use_lvis else None,
        "glas" if use_glas else None,
    ])) or "none"
    print(f"Control sources: {sources_label} "
          f"(use_is2={use_is2}, use_cs2={use_cs2}"
          + (f" [cs2_source={args.cs2_source}→label '{cs2_source_label}']" if use_cs2 else "")
          + f", use_atm={args.use_atm}, use_lvis={args.use_lvis}, use_glas={use_glas})")
    print(f"ASP output: {asp_root}")

    import fiona
    from shapely.geometry import shape
    with fiona.open(str(aoi_shp)) as src:
        aoi_polygon = shape(next(iter(src))["geometry"])
    print(f"AOI polygon bounds (EPSG:3031): {aoi_polygon.bounds}")

    import geopandas as gpd
    grounded_polygon = gpd.read_file(str(grounding_line_shp)).iloc[0].geometry

    eff_start = args.start or start_time
    eff_end = args.end or end_time
    todo, outside, no_grounded = find_unaligned_strips(
        strips_dir,
        asp_root / "asp_aligned",
        eff_start,
        eff_end,
        aoi_polygon=aoi_polygon,
        grounded_polygon=grounded_polygon,
    )
    print(f"Window {eff_start} .. {eff_end}")
    print(f"Found {len(todo)} strips needing ASP coregistration (AOI ∩ grounded).")
    for p, t in todo:
        print(f"  {t.date()}  {p.stem}")
    if outside:
        print(f"\nSkipping {len(outside)} downloaded strips outside AOI:")
        for p, t in outside:
            print(f"  {t.date()}  {p.stem}")
    if no_grounded:
        print(f"\nSkipping {len(no_grounded)} strips that don't reach grounded ice "
              f"(no grounded altimetric control possible):")
        for p, t in no_grounded:
            print(f"  {t.date()}  {p.stem}")

    def _cs2_h5_for(stem: str) -> Path | None:
        if cs2_cache_dir is None:
            return None
        path = cs2_cache_dir / cs2_filename_tmpl.format(stem=stem)
        if not path.exists():
            print(f"  ⚠  CS2 ({args.cs2_source}) cache miss for {stem}; "
                  f"skipping the cs2 channel (run {cs2_driver_hint} first)")
            return None
        return path

    def _atm_csv_for(stem: str) -> Path | None:
        if atm_cache_dir is None:
            return None
        path = atm_cache_dir / f"atm_filtered_{stem}.csv"
        return path if path.exists() else None

    def _lvis_csv_for(stem: str) -> Path | None:
        if lvis_cache_dir is None:
            return None
        path = lvis_cache_dir / f"lvis_filtered_{stem}.csv"
        return path if path.exists() else None

    def _glas_csv_for(stem: str) -> Path | None:
        if glas_cache_dir is None:
            return None
        path = glas_cache_dir / f"glas_filtered_{stem}.csv"
        return path if path.exists() else None

    # Without IS2 in the mix, a strip whose every selected cache misses
    # would reach align_strip with zero altimetric control and fail after
    # paying the DEM-load cost. Drop those up front as an explicit SKIP
    # class — and do it BEFORE --dem-id/--limit so a pilot `--limit N`
    # picks N strips that actually have control (2026-07-10 pilot bug:
    # the old per-basin copies filtered after --limit).
    n_skipped_no_control = 0
    if not use_is2:
        def _has_any_control(stem: str) -> bool:
            # Silent existence checks — the printing helpers above would
            # emit one warning per miss across the whole strip list.
            if cs2_cache_dir is not None and \
                    (cs2_cache_dir / cs2_filename_tmpl.format(stem=stem)).exists():
                return True
            if atm_cache_dir is not None and \
                    (atm_cache_dir / f"atm_filtered_{stem}.csv").exists():
                return True
            if lvis_cache_dir is not None and \
                    (lvis_cache_dir / f"lvis_filtered_{stem}.csv").exists():
                return True
            if glas_cache_dir is not None and \
                    (glas_cache_dir / f"glas_filtered_{stem}.csv").exists():
                return True
            return False

        kept, skipped = [], []
        for p, t in todo:
            (kept if _has_any_control(p.stem) else skipped).append((p, t))
        n_skipped_no_control = len(skipped)
        if skipped:
            print(f"\nSKIP {len(skipped)} strips with no altimetric control "
                  f"(no {sources_label} cache hit — expected for this era, "
                  f"not a failure):")
            for p, t in skipped[:20]:
                print(f"  {t.date()}  {p.stem}")
            if len(skipped) > 20:
                print(f"  ... ({len(skipped) - 20} more)")
        todo = kept
        print(f"After control-coverage filter: {len(todo)} strips to align.")

    if args.dem_id is not None:
        todo = [(p, t) for p, t in todo if p.stem == args.dem_id]
        if not todo:
            raise SystemExit(f"--dem-id {args.dem_id!r} not in unaligned set")
    if args.limit is not None:
        todo = todo[: args.limit]

    n_total = len(todo)
    free_gb = _disk_free_gb(asp_root)
    projected_gb = n_total * PER_STRIP_GB
    print(f"\nDisk budget: {free_gb:.0f} GB free at {asp_root}; "
          f"projected need {projected_gb:.0f} GB "
          f"({n_total} strips × {PER_STRIP_GB} GB/strip).")
    if free_gb - projected_gb < MIN_FREE_GB_FLOOR:
        print(f"  ⚠  margin would dip below {MIN_FREE_GB_FLOOR:.0f} GB; "
              f"loop will abort early if free space hits the floor mid-run.")

    common = dict(
        mosaic_dir=mosaic_dir,
        rock_shapefile=rock_shapefile,
        tide_model=tide_model,
        aoi_polygon=aoi_polygon,
        use_is2=use_is2,
        cs2_source_label=cs2_source_label,
        cap_per_source=args.cap_per_source,
        inverse_variance_balance=args.inv_variance_balance,
        inv_var_base_count=args.inv_var_base_count,
    )

    if args.parallel > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"\nParallel align with {args.parallel} workers over {n_total} strips...")
        n_ok = 0
        n_done = 0
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(
                    align_one, p, t, asp_root,
                    cs2_h5_path=_cs2_h5_for(p.stem),
                    atm_csv_path=_atm_csv_for(p.stem),
                    lvis_csv_path=_lvis_csv_for(p.stem),
                    glas_csv_path=_glas_csv_for(p.stem),
                    **common,
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
            free_gb = _disk_free_gb(asp_root)
            if free_gb < MIN_FREE_GB_FLOOR:
                print(f"\n!! free space {free_gb:.0f} GB < floor "
                      f"{MIN_FREE_GB_FLOOR:.0f} GB; aborting batch with "
                      f"{n_total - i + 1} strips remaining.")
                break
            print(f"\n[{i}/{n_total}] starting (free: {free_gb:.0f} GB)...")
            if align_one(
                p, t, asp_root,
                cs2_h5_path=_cs2_h5_for(p.stem),
                atm_csv_path=_atm_csv_for(p.stem),
                lvis_csv_path=_lvis_csv_for(p.stem),
                glas_csv_path=_glas_csv_for(p.stem),
                **common,
            ):
                n_ok += 1

    print(f"\n=== batch summary: {n_ok}/{n_total} succeeded"
          + (f" ({n_skipped_no_control} more skipped upfront: no altimetric control)"
             if n_skipped_no_control else "")
          + " ===")
    if n_total > 0 and n_ok == 0:
        return 1
    return 0
