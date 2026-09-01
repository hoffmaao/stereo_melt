"""Dense-tie re-alignment FROM RAW strips — geometry rescue + balanced datum.

Unlike :mod:`pig.align_strips_densetie` (which re-ties only the strips that
already one-stage-aligned, in ``config.STRIP_SOURCES``), this driver discovers
**every raw REMA strip** in the window ∩ AOI ∩ grounded — including the ~150
pre-IS2 strips that one-stage altimetry ``pc_align`` *failed to converge on*
(sparse-altimetry divergence). It then fixes, in a single pass, the three
problems diagnosed 2026-06-18:

  1. **~9.4× coreg-noise ramp** — Stage 1 aligns the raw strip to the dense
     static-masked REMA reference (surface-to-surface geometry, altimetry-FREE),
     so sparse/noisy altimetry never sets the strip's tilt/rotation.
  2. **13:1 CryoTEMPO-over-IS2 overweight** — that was a *pc_align* problem;
     densetie-from-raw sidesteps it because Stage 1 uses the dense REMA surface,
     not altimetry. Stage 2 ties the vertical datum with a plain robust *median*
     of (control − aligned) over ALL altimetry points (Shean coregistration): no
     good CryoTEMPO is discarded, and the median is IS2-dominated by sheer count
     where IS2 is present, CryoTEMPO-set where it's the only source. The
     IS2-vs-CryoTEMPO precision weighting lives downstream in the tilt-fit LSQ
     (per-source Ez — IS2 0.1 m, CryoTEMPO 1.0 m), the optimisation we actually
     minimise (Shean's regularised stack fit). Rock is excluded: after Stage 1 the
     strip already matches REMA over rock, so rock residuals are ≈0 and would only
     dilute the epoch datum signal.
  3. **~150 dropped pre-IS2 strips** — Stage 1 needs no per-strip altimetry, so
     the strips that diverged on sparse altimetry are recovered as long as they
     overlap the static REMA surface (the AOI∩grounded filter ensures they do).

Discovery is resumable for free: :func:`pig.align_strips.find_unaligned_strips`
pointed at this (initially empty) root treats *every* raw strip as "to do", and
skips ones already written here (or quarantined ``.bad_align``) on a re-run.

Output → ``config.ASP_DENSETIE_FROMRAW_ROOT`` (a NEW root; the is2ctempo /
ctempo / ASP_densetie baselines are left intact for the A/B). No baseline
symlink fallback: the whole point is geometry-from-REMA, and the recovered
FAILs have no baseline DEM to fall back to anyway.

Run:
    python -m pig.align_strips_densetie_fromraw --dry-run          # discover, no ASP
    python -m pig.align_strips_densetie_fromraw --parallel 8       # full re-align
    python -m pig.align_strips_densetie_fromraw --dem-id <stem>    # one strip
    python -m pig.align_strips_densetie_fromraw --limit 2 --parallel 2   # smoke
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Point pyproj/GDAL at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from stereo_melt.coregister.asp import (
    _cs2_h5_to_csv,
    _read_translation_magnitude,
    align_strip_two_stage,
    build_control_csv,
)
from stereo_melt.coregister.control_source import write_sources_sidecar

from pig import config

FROMRAW_ROOT = config.ASP_DENSETIE_FROMRAW_ROOT
_SUFFIX = "-trans_reference-DEM.tif"
_SIDECAR = ".sources.json"  # per-strip GCP record; tilt_fit reads it for per-epoch Ez

# Per-strip altimetry caches all live under the canonical (unsuffixed) ASP root,
# keyed by strip stem — the same files align_strip / pig.align_strips consume.
_IS2_CACHE = config.ASP_ROOT / "icesat2_data"
_CT_CACHE = config.ASP_ROOT / "cryotempo_data"
_ATM_CACHE = config.ASP_ROOT / "atm_data"
_LVIS_CACHE = config.ASP_ROOT / "lvis_data"

# Dense-tie outputs (ASP point cloud deleted post-point2dem since 2026-06-12,
# see reference_disk_budget) — the 2 m DEM + geodiff + reference CSVs measured
# ~0.57 GB/strip on the ASP_densetie root (307 GB / 540), so budget ~0.6.
PER_STRIP_GB = 0.6
MIN_FREE_GB_FLOOR = 100.0


def _is2_for(stem: str) -> Path | None:
    h5 = _IS2_CACHE / f"icesat2_filtered_{stem}.h5"
    csv = _IS2_CACHE / f"icesat2_filtered_{stem}.csv"
    return h5 if h5.exists() else (csv if csv.exists() else None)


def _ct_h5_for(stem: str) -> Path | None:
    p = _CT_CACHE / f"cryotempo_filtered_{stem}.h5"
    return p if p.exists() else None


def _atm_for(stem: str) -> Path | None:
    p = _ATM_CACHE / f"atm_filtered_{stem}.csv"
    return p if p.exists() else None


def _lvis_for(stem: str) -> Path | None:
    p = _LVIS_CACHE / f"lvis_filtered_{stem}.csv"
    return p if p.exists() else None


def _classify_failure(msg: str) -> str:
    if "no transformed cloud" in msg or "insufficient static" in msg:
        return "no_static"   # pure shelf / no static overlap with REMA
    if "exceeds cap" in msg or "|Δ|" in msg:
        return "diverged"    # Stage-1 surface-to-surface tie diverged past cap
    return "error"


def _build_datum_csv(stem: str):
    """Build the altimetry-only datum CSV for the Stage-2 Δz tie — Shean-style.

    Combines whichever of {IS2, CryoTEMPO, ATM, LVIS} are cached for this strip,
    keeping EVERY point (no subsampling). Stage 2 takes a plain robust *median* of
    (control − aligned), which is IS2-dominated by sheer count where IS2 is present
    and CryoTEMPO-set where it's the only source — so no good CryoTEMPO is dropped.
    The IS2-vs-CryoTEMPO precision weighting is NOT applied here; it lives in the
    optimisation we minimise (the tilt-fit LSQ, per-source Ez), where Shean's
    regularised fit carries observation uncertainty. NO rock (the Stage-1 REMA tie
    already fixes the static surface). Returns ``(datum_csv | None, sources_used)``.
    """
    refdir = FROMRAW_ROOT / "reference_files"
    refdir.mkdir(parents=True, exist_ok=True)
    is2 = _is2_for(stem)
    ct_h5 = _ct_h5_for(stem)
    atm = _atm_for(stem)
    lvis = _lvis_for(stem)

    ct_csv = None
    if ct_h5 is not None:
        ct_csv = refdir / f"cryotempo_{stem}.csv"
        _cs2_h5_to_csv(str(ct_h5), str(ct_csv))  # PointCollection h5 → easting,northing,h_mean

    datum_csv = refdir / f"datum_{stem}.csv"
    sources_used = build_control_csv(
        str(datum_csv),
        rock_csv=None,                       # altimetry-only — Stage 1 owns rock/geometry
        icesat2_csv=str(is2) if is2 else None,
        cs2_csv=str(ct_csv) if ct_csv else None,
        cs2_source_label="cryotempo",
        atm_csv=str(atm) if atm else None,
        lvis_csv=str(lvis) if lvis else None,
        inverse_variance_balance=False,      # keep ALL control (Shean robust median); weight in tilt-fit Ez
        verbose=True,
    )
    if not sources_used:
        return None, []
    return datum_csv, sources_used


def _align_one(item: dict, ref: str, max_disp: float, keep_pc: bool,
               force: bool):
    """Dense-tie one raw strip into the from-raw root; return (outcome, stem).

    outcome ∈ {dense, dense_nodatum, skip, no_static, diverged, error}.
    ``dense_nodatum`` = Stage 1 succeeded but no altimetry was cached, so the
    strip kept the REMA (multi-year-mean) datum — flagged so downstream can
    decide whether to trust its absolute height.
    """
    stem = item["stem"]
    out_dem = FROMRAW_ROOT / "asp_aligned" / f"{stem}{_SUFFIX}"
    if out_dem.exists() and not force:
        return ("skip", stem)
    print(f"\n=== densetie-fromraw: {stem} ===", flush=True)
    try:
        datum_csv, sources_used = _build_datum_csv(stem)
    except Exception as exc:  # noqa: BLE001 — datum build is best-effort; fall back to REMA datum
        print(f"   datum-csv build failed ({exc!r}); proceeding REMA-datum-only", flush=True)
        datum_csv, sources_used = None, []
    try:
        align_strip_two_stage(
            file_path=str(item["raw"]),
            reference_dem=ref,
            datum_csv=str(datum_csv) if datum_csv else None,
            output_dir=str(FROMRAW_ROOT),
            max_displacement=max_disp,
            keep_point_cloud=keep_pc,
            verbose=True,
        )
    except RuntimeError as exc:
        reason = _classify_failure(str(exc))
        print(f"   -> {reason}: {exc}", flush=True)
        return (reason, stem)
    except Exception as exc:  # noqa: BLE001 — batch harness: log, continue
        print(f"   -> error: {exc!r}", flush=True)
        return ("error", stem)
    if sources_used:
        # Record the GCP sidecar so tilt_fit picks the per-epoch Ez tier matched
        # to the altimeter precision (is2/atm/lvis tight, cryotempo looser).
        try:
            write_sources_sidecar(str(FROMRAW_ROOT), stem, sources_used)
        except OSError:
            pass
        return ("dense", stem)
    return ("dense_nodatum", stem)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem-id", default=None, help="Dense-tie a single raw strip by stem.")
    ap.add_argument("--parallel", type=int, default=1, help="ASP workers (default 1).")
    ap.add_argument("--max-displacement", type=float, default=100.0,
                    help="pc_align cap, m (Shean default 100).")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N strips (smoke).")
    ap.add_argument("--keep-point-cloud", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if a from-raw DEM exists for the strip.")
    ap.add_argument("--start", default=None,
                    help="Window start override YYYY-MM-DD (default config.START_TIME).")
    ap.add_argument("--end", default=None,
                    help="Window end override YYYY-MM-DD (default config.END_TIME).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Discover raw strips + report control coverage, then exit.")
    args = ap.parse_args()

    ref = str(config.REMA_STATIC_REFERENCE_TIF)
    if not os.path.exists(ref):
        raise SystemExit(
            f"Static reference missing: {ref}\n"
            f"Build it first:  python -m pig.build_rema_static_reference")

    import geopandas as gpd

    from pig.align_strips import _load_aoi, find_unaligned_strips

    aoi = _load_aoi()
    grounded = gpd.read_file(config.GROUNDING_LINE_SHP).iloc[0].geometry

    (FROMRAW_ROOT / "asp_aligned").mkdir(parents=True, exist_ok=True)

    # Empty/partial from-raw root → find_unaligned_strips returns every raw strip
    # in window∩AOI∩grounded not yet tied here (and skips .bad_align sentinels).
    todo, outside, no_grounded = find_unaligned_strips(
        aoi_polygon=aoi,
        grounded_polygon=grounded,
        aligned_dir=FROMRAW_ROOT / "asp_aligned",
        start_time=args.start,
        end_time=args.end,
    )
    eff_start = args.start or config.START_TIME
    eff_end = args.end or config.END_TIME
    print(f"Reference: {ref}")
    print(f"Window {eff_start} .. {eff_end}")
    print(f"Discovered {len(todo)} raw strips to dense-tie (AOI∩grounded, not yet in "
          f"{FROMRAW_ROOT.name}); skipped {len(outside)} outside-AOI, "
          f"{len(no_grounded)} with no grounded ice.")

    if args.dem_id:
        todo = [(p, t) for p, t in todo if p.stem == args.dem_id]
        if not todo:
            raise SystemExit(f"--dem-id {args.dem_id!r} not in the discovered raw set")
    if args.limit is not None:
        todo = todo[: args.limit]

    targets = [{"stem": p.stem, "raw": p, "t": t} for p, t in todo]

    # Control-coverage report: which strips can be datum-tied vs REMA-datum-only.
    cov = {"is2": 0, "cryotempo": 0, "atm": 0, "lvis": 0, "none": 0}
    for it in targets:
        s = it["stem"]
        any_src = False
        if _is2_for(s):
            cov["is2"] += 1; any_src = True
        if _ct_h5_for(s):
            cov["cryotempo"] += 1; any_src = True
        if _atm_for(s):
            cov["atm"] += 1; any_src = True
        if _lvis_for(s):
            cov["lvis"] += 1; any_src = True
        if not any_src:
            cov["none"] += 1
    print(f"  control coverage of {len(targets)} strips: is2={cov['is2']} "
          f"cryotempo={cov['cryotempo']} atm={cov['atm']} lvis={cov['lvis']}; "
          f"{cov['none']} have NO altimetry cache (→ REMA-datum-only).")

    if args.dry_run:
        for it in targets[:40]:
            print(f"  {it['t'].date()}  {it['stem']}")
        if len(targets) > 40:
            print(f"  ... ({len(targets) - 40} more)")
        return

    FROMRAW_ROOT.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(FROMRAW_ROOT).free / 1e9
    projected = len(targets) * PER_STRIP_GB
    print(f"\nDisk: {free_gb:.0f} GB free at {FROMRAW_ROOT}; "
          f"projected ~{projected:.0f} GB ({len(targets)}×{PER_STRIP_GB} GB).")
    if free_gb - projected < MIN_FREE_GB_FLOOR:
        raise SystemExit(
            f"Would dip below {MIN_FREE_GB_FLOOR:.0f} GB free floor — free disk first.")

    outcomes: list[tuple[str, str]] = []
    if args.parallel > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"\nDense-tie-from-raw {len(targets)} strips with {args.parallel} workers...")
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futs = {
                ex.submit(_align_one, it, ref, args.max_displacement,
                          args.keep_point_cloud, args.force): it
                for it in targets
            }
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    outcome, stem = fut.result()
                except Exception as exc:  # noqa: BLE001
                    outcome, stem = "error", futs[fut]["stem"]
                    print(f"!! worker raised for {stem}: {exc!r}", flush=True)
                outcomes.append((outcome, stem))
                print(f"[{done}/{len(targets)}] {stem} -> {outcome}", flush=True)
    else:
        for i, it in enumerate(targets, 1):
            free_gb = shutil.disk_usage(FROMRAW_ROOT).free / 1e9
            if free_gb < MIN_FREE_GB_FLOOR:
                print(f"\n!! free {free_gb:.0f} GB < floor {MIN_FREE_GB_FLOOR:.0f} GB; "
                      f"aborting with {len(targets) - i + 1} strips remaining.")
                break
            outcome, stem = _align_one(it, ref, args.max_displacement,
                                       args.keep_point_cloud, args.force)
            outcomes.append((outcome, stem))
            print(f"[{i}/{len(targets)}] {stem} -> {outcome}", flush=True)

    # --- summary ---
    counts: dict[str, int] = {}
    for o, _ in outcomes:
        counts[o] = counts.get(o, 0) + 1
    print("\n" + "=" * 72)
    print("DENSETIE-FROM-RAW SUMMARY")
    print("=" * 72)
    for k in ("dense", "dense_nodatum", "skip", "no_static", "diverged", "error"):
        if counts.get(k):
            print(f"  {k:14s}: {counts[k]}")
    n_ok = counts.get("dense", 0) + counts.get("dense_nodatum", 0)
    print(f"\n  -> aligned {n_ok}/{len(targets)} "
          f"({counts.get('dense', 0)} with altimetry datum tie, "
          f"{counts.get('dense_nodatum', 0)} REMA-datum-only); "
          f"{counts.get('no_static', 0) + counts.get('diverged', 0)} un-tieable; "
          f"{counts.get('error', 0)} errored.")

    # |Δ| screen — large translation on the dense tie ⇒ inspect before stacking.
    aln = FROMRAW_ROOT / "asp_aligned"
    big = []
    for outcome, stem in outcomes:
        if not outcome.startswith("dense"):
            continue
        dmag = _read_translation_magnitude(str(aln / stem))
        if dmag is not None and dmag > args.max_displacement:
            big.append((stem, dmag))
    if big:
        print(f"\n  ⚠ {len(big)} dense-tied strips have |Δ| > {args.max_displacement:.0f} m "
              f"(inspect; should be < cap):")
        for stem, dmag in sorted(big, key=lambda x: -x[1])[:20]:
            print(f"      |Δ|={dmag:10.1f} m   {stem}")

    print(f"\nFrom-raw dense-tied DEMs are in {aln}/ (<stem>{_SUFFIX}). Next (one root = both eras):\n"
          f"  python -m pig.build_stack --is2-asp densetie_fromraw "
          f"--pre-is2-asp densetie_fromraw --tag densetie_fromraw\n"
          f"  → tilt_fit → run_melt; A/B vs is2ctempo + the old ASP_densetie.")


if __name__ == "__main__":
    main()
