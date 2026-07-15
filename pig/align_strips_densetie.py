"""Step 5 of ``literature/plan_alignment.md``: full dense-tie re-alignment.

Re-align **every production-aligned strip** (the IS2-era ``is2cs2atmlvis`` +
pre-IS2 ``cs2atmlvis`` roots in ``config.STRIP_SOURCES``) with the Option-B
two-stage dense tie — Stage 1 ``pc_align`` strip → the static-masked REMA
reference (``config.REMA_STATIC_REFERENCE_TIF``), Stage 2 a bulk Δz to the
strip's existing altimetry control CSV — into a NEW per-basin root
``config.ASP_DENSETIE_ROOT`` (``pig/data/ASP_densetie/``). The baseline roots
are left untouched, so the densetie product is a clean A/B against the current
``is2ctempo`` melt rather than a destructive overwrite.

The prototype gate (``pig.prototype_two_stage``) passed, so this scales the
same per-strip tie (:func:`stereo_melt.coregister.asp.align_strip_two_stage`)
to the full record. The decisive test remains the downstream
build_stack → tilt_fit → run_melt rebuild on this root (plan step 6).

**Coverage parity (for a fair A/B).** A strip with no static-surface overlap
(pure shelf) can't be densely tied — Stage 1 raises and the plan says to *fall
back to the existing altimetry-only alignment*. Rather than drop those strips
(which would give the densetie stack less coverage than the baseline and
confound the A/B), we **symlink the strip's baseline aligned DEM** into the
densetie root. The result: dense-tied geometry where static overlap allows it,
the baseline alignment everywhere else, same strip set as the baseline.

Run:
    python -m pig.align_strips_densetie --dry-run        # list targets, no ASP
    python -m pig.align_strips_densetie --parallel 8     # full re-align
    python -m pig.align_strips_densetie --dem-id <stem>  # one strip
    python -m pig.align_strips_densetie --no-fallback    # skip the symlink fallback
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

import numpy as np

from stereo_melt.coregister.asp import _read_translation_magnitude, align_strip_two_stage

from pig import config

RESCUE_ROOT = None  # (unused; kept names parallel to rescue_bad_aligns for grep)
DENSETIE_ROOT = config.ASP_DENSETIE_ROOT
_SUFFIX = "-trans_reference-DEM.tif"
_SIDECAR = ".sources.json"  # per-strip GCP record; tilt_fit reads it for per-epoch Ez

# Densetie outputs are slim (ASP cloud deleted post-point2dem since 2026-06-12,
# see reference_disk_budget): ~0.2-0.3 GB/strip. Symlinked fallbacks cost 0.
PER_STRIP_GB = 0.3
MIN_FREE_GB_FLOOR = 100.0


def _aligned_strips(sources) -> list[dict]:
    """One record per unique production-aligned strip with a raw DEM on disk:
    ``{stem, raw, csv, base_dem, side, root, label}``. Earlier era roots in
    ``sources`` win on duplicate stems (a stem should appear in only one era
    anyway, split by date — but the ctempo roots' date spans overlap, so the
    IS2-era source listed first wins, matching build_stack's stem dedup)."""
    seen: dict[str, dict] = {}
    for aligned_dir, label in sources:
        aligned_dir = Path(aligned_dir)
        root = aligned_dir.parent
        for dem in sorted(aligned_dir.glob(f"*{_SUFFIX}")):
            stem = dem.name[: -len(_SUFFIX)]
            if stem in seen:
                continue
            raw = Path(config.STRIPS_DIR) / f"{stem}.tif"
            if not raw.exists():
                continue  # can't re-align from scratch without the raw strip
            csv = root / "reference_files" / f"combined_reference_{stem}.csv"
            side = aligned_dir / f"{stem}{_SIDECAR}"
            seen[stem] = {
                "stem": stem, "raw": raw,
                "csv": csv if csv.exists() else None,
                "base_dem": dem,
                "side": side if side.exists() else None,
                "root": root, "label": label,
            }
    return list(seen.values())


def _classify_failure(msg: str) -> str:
    if "no transformed cloud" in msg or "insufficient static" in msg:
        return "no_static"   # pure shelf — expected, fall back to baseline
    if "exceeds cap" in msg or "|Δ|" in msg:
        return "diverged"    # dense tie diverged past cap
    return "error"


def _fallback_to_baseline(item: dict, out_dem: Path) -> None:
    """Symlink the strip's baseline aligned DEM (+ sidecar) into the densetie
    root so coverage matches the baseline even when the dense tie can't run."""
    if out_dem.exists() or out_dem.is_symlink():
        out_dem.unlink()  # clear any partial/truncated write before linking
    os.symlink(str(item["base_dem"].resolve()), str(out_dem))
    if item["side"] is not None:
        link = out_dem.parent / f"{item['stem']}{_SIDECAR}"
        if not (link.exists() or link.is_symlink()):
            os.symlink(str(item["side"].resolve()), str(link))


def _align_one(item: dict, ref: str, max_disp: float, keep_pc: bool,
               fallback: bool, force: bool = False):
    """Two-stage align one strip into the densetie root; return (outcome, stem).

    outcome ∈ {dense, skip, fallback, no_static, diverged, error}. With
    ``fallback`` the failure classes (no_static/diverged/error) also symlink the
    baseline DEM and the returned outcome is ``fallback`` (the failure reason is
    printed)."""
    stem = item["stem"]
    out_dem = DENSETIE_ROOT / "asp_aligned" / f"{stem}{_SUFFIX}"
    if (out_dem.exists() or out_dem.is_symlink()) and not force:
        return ("skip", stem)
    print(f"\n=== densetie: {stem}  [{item['label']}] ===", flush=True)
    try:
        align_strip_two_stage(
            file_path=str(item["raw"]),
            reference_dem=ref,
            datum_csv=str(item["csv"]) if item["csv"] else None,
            output_dir=str(DENSETIE_ROOT),
            max_displacement=max_disp,
            keep_point_cloud=keep_pc,
            verbose=True,
        )
    except RuntimeError as exc:
        reason = _classify_failure(str(exc))
        print(f"   -> {reason}: {exc}", flush=True)
        if fallback:
            _fallback_to_baseline(item, out_dem)
            return ("fallback", stem)
        return (reason, stem)
    except Exception as exc:  # noqa: BLE001 — batch harness: log, fall back, continue
        print(f"   -> error: {exc!r}", flush=True)
        if fallback:
            _fallback_to_baseline(item, out_dem)
            return ("fallback", stem)
        return ("error", stem)
    # success — copy the GCP sidecar so the densetie root is self-describing
    if item["side"] is not None:
        try:
            shutil.copy2(item["side"], DENSETIE_ROOT / "asp_aligned" / f"{stem}{_SIDECAR}")
        except OSError:
            pass
    return ("dense", stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem-id", type=str, default=None, help="Re-align a single strip by stem.")
    ap.add_argument("--is2-asp", type=str, default="is2ctempoatmlvis",
                    help="IS2-era source root ASP_<value>/asp_aligned to re-align "
                         "(default the is2ctempo production root). Mirrors "
                         "build_stack --is2-asp so the densetie set == the A/B baseline set.")
    ap.add_argument("--pre-is2-asp", type=str, default="ctempoatmlvis",
                    help="pre-IS2 source root ASP_<value>/asp_aligned (default the "
                         "ctempo production root). Mirrors build_stack --pre-is2-asp.")
    ap.add_argument("--parallel", type=int, default=1, help="ASP workers (default 1).")
    ap.add_argument("--max-displacement", type=float, default=100.0,
                    help="pc_align cap, m (Shean default 100).")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N strips (smoke).")
    ap.add_argument("--keep-point-cloud", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run even if a densetie DEM exists.")
    ap.add_argument("--no-fallback", action="store_true",
                    help="Do NOT symlink the baseline DEM for un-tieable strips "
                         "(densetie stack will then have less coverage than baseline).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List discovered production-aligned strips and exit.")
    args = ap.parse_args()

    ref = str(config.REMA_STATIC_REFERENCE_TIF)
    if not os.path.exists(ref):
        raise SystemExit(
            f"Static reference missing: {ref}\n"
            f"Build it first:  python -m pig.build_rema_static_reference"
        )

    # Re-align the CURRENT production strip set. config.STRIP_SOURCES is the
    # stale raw-CS2 baseline (cs2atmlvis now empty); the live is2ctempo product
    # is built by build_stack from ASP_is2ctempoatmlvis + ASP_ctempoatmlvis via
    # its --is2-asp/--pre-is2-asp overrides, so we mirror those here. The
    # baseline DEMs we discover (and fall back to) are then exactly the
    # is2ctempo alignment the densetie product is A/B'd against.
    sources = [
        (config.BASIN_DIR / "data" / f"ASP_{args.is2_asp.lstrip('_')}" / "asp_aligned",
         args.is2_asp.lstrip("_")),
        (config.BASIN_DIR / "data" / f"ASP_{args.pre_is2_asp.lstrip('_')}" / "asp_aligned",
         args.pre_is2_asp.lstrip("_")),
    ]
    print("Source roots (mirrors build_stack --is2-asp/--pre-is2-asp):")
    for d, lbl in sources:
        exists = "" if d.exists() else "  ⚠ MISSING"
        print(f"  [{lbl:18s}] {d}{exists}")

    targets = _aligned_strips(sources)
    if args.dem_id:
        targets = [t for t in targets if t["stem"] == args.dem_id]
        if not targets:
            raise SystemExit(f"--dem-id {args.dem_id!r} is not a production-aligned strip.")
    targets.sort(key=lambda t: t["stem"])

    by_label: dict[str, int] = {}
    n_csv = 0
    for t in targets:
        by_label[t["label"]] = by_label.get(t["label"], 0) + 1
        n_csv += t["csv"] is not None
    print(f"{len(targets)} production-aligned strips with a raw DEM on disk:")
    for lbl, n in sorted(by_label.items()):
        print(f"  {lbl:16s}: {n}")
    print(f"  ({n_csv} have a control CSV for the Stage-2 Δz; "
          f"{len(targets) - n_csv} will keep the REMA datum)")
    if args.limit is not None:
        targets = targets[: args.limit]
        print(f"  (--limit {args.limit} → {len(targets)} strips this run)")
    if args.dry_run:
        return

    DENSETIE_ROOT.mkdir(parents=True, exist_ok=True)
    (DENSETIE_ROOT / "asp_aligned").mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(DENSETIE_ROOT).free / 1e9
    projected = len(targets) * PER_STRIP_GB
    print(f"\nDisk: {free_gb:.0f} GB free at {DENSETIE_ROOT}; "
          f"projected dense-tie need ~{projected:.0f} GB ({len(targets)}×{PER_STRIP_GB} GB, "
          f"fallbacks are symlinks @ 0 GB).")
    if free_gb - projected < MIN_FREE_GB_FLOOR:
        raise SystemExit(f"Would dip below {MIN_FREE_GB_FLOOR:.0f} GB free floor — free disk first.")

    fallback = not args.no_fallback
    outcomes: list[tuple[str, str]] = []
    if args.parallel > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"\nRe-aligning {len(targets)} strips with {args.parallel} workers "
              f"(fallback={'on' if fallback else 'OFF'})...")
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futs = {
                ex.submit(_align_one, t, ref, args.max_displacement,
                          args.keep_point_cloud, fallback, args.force): t
                for t in targets
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
        for i, t in enumerate(targets, 1):
            outcome, stem = _align_one(t, ref, args.max_displacement,
                                       args.keep_point_cloud, fallback, args.force)
            outcomes.append((outcome, stem))
            print(f"[{i}/{len(targets)}] {stem} -> {outcome}", flush=True)

    # --- summary ---
    counts: dict[str, int] = {}
    for o, _ in outcomes:
        counts[o] = counts.get(o, 0) + 1
    print("\n" + "=" * 72)
    print("DENSETIE RE-ALIGN SUMMARY")
    print("=" * 72)
    for k in ("dense", "fallback", "skip", "no_static", "diverged", "error"):
        if counts.get(k):
            print(f"  {k:10s}: {counts[k]}")
    n_dense = counts.get("dense", 0)
    print(f"\n  -> dense-tied {n_dense} / {len(targets)} "
          f"({100 * n_dense / max(1, len(targets)):.0f}%); "
          f"fallback-symlinked {counts.get('fallback', 0)}; "
          f"already-done {counts.get('skip', 0)}.")

    # New |Δ| of the dense-tied strips — a quick screen for any that wandered
    # (large |Δ| under the dense tie ⇒ inspect before trusting in the stack).
    aln = DENSETIE_ROOT / "asp_aligned"
    big = []
    for outcome, stem in outcomes:
        if outcome != "dense":
            continue
        d = _read_translation_magnitude(str(aln / stem))
        if d is not None and d > args.max_displacement:
            big.append((stem, d))
    if big:
        print(f"\n  ⚠ {len(big)} dense-tied strips have |Δ| > {args.max_displacement:.0f} m "
              f"(inspect; should be < cap):")
        for stem, d in sorted(big, key=lambda x: -x[1])[:20]:
            print(f"      |Δ|={d:10.1f} m   {stem}")

    print(f"\nDensetie aligned DEMs are in {aln}/ (<stem>{_SUFFIX}). Next: "
          f"build_stack → tilt_fit → run_melt on this root (--tag densetie), A/B vs is2ctempo.")


if __name__ == "__main__":
    main()
