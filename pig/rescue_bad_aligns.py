"""Rescue quarantined (`.bad_align`) strips via the two-stage dense-REMA tie.

Strips get a `.bad_align` sentinel and are thrown out of the pipeline when the
sparse-control `pc_align` diverges past `--max-displacement` (100 m): the
quarantine `_REASON.txt` reads "pc_align did not converge". The usual cause is
that the **sparse altimetry control** (a few grounded IS2/CS2/ATM/LVIS points)
couldn't constrain the 6-DOF ICP, so it wandered to a bad local minimum.

The Option-B two-stage tie (`align_strip_two_stage`) doesn't use altimetry for
geometry at all — Stage 1 ties the strip to the **dense static REMA reference**
(hundreds of thousands of points spanning rock + stable grounded ice), which is
far better-conditioned. So a quarantined strip that *has static-surface overlap*
can re-converge and be recovered; the altimetry control is only used in Stage 2
for the absolute datum (and skipped if too sparse).

This driver re-aligns every quarantined strip into ``ASP_densetie_rescue/`` and
auto-sorts the outcome:

  * **rescued**     — Stage 1 converged within the cap; a clean aligned DEM now
                      exists, ready to fold back into the stack (separate step).
  * **still_bad**   — even the dense tie diverges past the cap ⇒ the DEM is
                      genuinely broken (wrong CRS / corrupt / mostly-nodata).
  * **no_static**   — the strip is pure shelf with no static overlap ⇒ the dense
                      tie has nothing to anchor to; not rescuable by this method.

It reports each strip's original vs new |Δ| and the held-out altimetry-control
residual for the rescued ones. Reference is PGC REMA only (never Shean's grids).

Run:
    python -m pig.rescue_bad_aligns --dry-run        # list targets, no ASP
    python -m pig.rescue_bad_aligns --parallel 3     # rescue all
    python -m pig.rescue_bad_aligns --dem-id <stem>  # one strip
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Point pyproj/GDAL at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import numpy as np
import pandas as pd

from stereo_melt.coregister.asp import _read_translation_magnitude, align_strip_two_stage

from pig import config
from pig.prototype_two_stage import _control_residual_stats

# Production roots that hold the quarantine (.bad_align sentinels + bad_align/).
ROOTS = [config.ASP_IS2CS2ATMLVIS_ROOT, config.ASP_CS2ATMLVIS_ROOT]
RESCUE_ROOT = config.BASIN_DIR / "data" / "ASP_densetie_rescue"

_SUFFIX = "-trans_reference-DEM.tif"
_DELTA_RE = re.compile(r"\|Δ\|=([0-9.eE+]+)")


def _quarantined() -> list[dict]:
    """One record per unique quarantined strip with a raw DEM on disk:
    ``{stem, raw, csv, orig_delta, root}``."""
    seen: dict[str, dict] = {}
    for root in ROOTS:
        for sent in sorted((root / "asp_aligned").glob("*.bad_align")):
            stem = sent.name[: -len(".bad_align")]
            if stem in seen:
                continue
            raw = Path(config.STRIPS_DIR) / f"{stem}.tif"
            if not raw.exists():
                continue
            # original |Δ| from the quarantine reason (sparse-control divergence)
            orig = np.nan
            reason = root / "bad_align" / stem / "_REASON.txt"
            if reason.exists():
                m = _DELTA_RE.search(reason.read_text())
                if m:
                    orig = float(m.group(1))
            # datum CSV for Stage 2 (either root); may be absent → REMA datum kept
            csv = None
            for r in ROOTS:
                c = r / "reference_files" / f"combined_reference_{stem}.csv"
                if c.exists():
                    csv = c
                    break
            seen[stem] = {
                "stem": stem, "raw": raw, "csv": csv,
                "orig_delta": orig, "root": root.name,
            }
    return list(seen.values())


def _classify_failure(msg: str) -> str:
    if "no transformed cloud" in msg or "insufficient static" in msg:
        return "no_static"
    if "exceeds cap" in msg or "|Δ|" in msg:
        return "still_bad"
    return "error"


def _rescue_one(item: dict, ref: str, max_disp: float, keep_pc: bool, force: bool = False):
    """Re-align one quarantined strip; return (outcome, stem)."""
    stem = item["stem"]
    out_dem = RESCUE_ROOT / "asp_aligned" / f"{stem}{_SUFFIX}"
    if out_dem.exists() and not force:
        return ("rescued", stem)
    print(f"\n=== rescue: {stem}  (orig |Δ|={item['orig_delta']:.1f} m) ===", flush=True)
    try:
        align_strip_two_stage(
            file_path=str(item["raw"]),
            reference_dem=ref,
            datum_csv=str(item["csv"]) if item["csv"] else None,
            output_dir=str(RESCUE_ROOT),
            max_displacement=max_disp,
            keep_point_cloud=keep_pc,
            verbose=True,
        )
    except RuntimeError as exc:
        outcome = _classify_failure(str(exc))
        print(f"   -> {outcome}: {exc}", flush=True)
        return (outcome, stem)
    except Exception as exc:  # noqa: BLE001
        print(f"   -> error: {exc!r}", flush=True)
        return ("error", stem)
    return ("rescued", stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dem-id", type=str, default=None, help="Rescue a single strip by stem.")
    ap.add_argument("--parallel", type=int, default=1, help="ASP workers (default 1).")
    ap.add_argument("--max-displacement", type=float, default=100.0,
                    help="pc_align cap, m (Shean default 100).")
    ap.add_argument("--keep-point-cloud", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run even if a rescue DEM exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List quarantined targets (sorted by original |Δ|) and exit.")
    ap.add_argument("--metrics-only", action="store_true",
                    help="Skip alignment; just score existing rescue DEMs.")
    args = ap.parse_args()

    ref = str(config.REMA_STATIC_REFERENCE_TIF)
    if not os.path.exists(ref):
        raise SystemExit(
            f"Static reference missing: {ref}\n"
            f"Build it first:  python -m pig.build_rema_static_reference"
        )

    targets = _quarantined()
    if args.dem_id:
        targets = [t for t in targets if t["stem"] == args.dem_id]
        if not targets:
            raise SystemExit(f"--dem-id {args.dem_id!r} is not a quarantined strip.")
    targets.sort(key=lambda t: (np.isnan(t["orig_delta"]), t["orig_delta"]))

    print(f"{len(targets)} quarantined strips with raw DEM on disk "
          f"(sorted by original sparse-control |Δ|):")
    n_csv = sum(1 for t in targets if t["csv"] is not None)
    print(f"  ({n_csv} have a control CSV for the Stage-2 datum tie; "
          f"{len(targets) - n_csv} will keep the REMA datum)")
    for t in targets:
        od = f"{t['orig_delta']:.0f}" if np.isfinite(t["orig_delta"]) else "?"
        print(f"  orig|Δ|={od:>12} m   {t['stem']}   [{t['root']}]")
    if args.dry_run:
        return

    RESCUE_ROOT.mkdir(parents=True, exist_ok=True)
    (RESCUE_ROOT / "asp_aligned").mkdir(parents=True, exist_ok=True)

    outcomes: list[tuple[str, str]] = []
    if not args.metrics_only:
        if args.parallel > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            print(f"\nRe-aligning {len(targets)} strips with {args.parallel} workers...")
            with ProcessPoolExecutor(max_workers=args.parallel) as ex:
                futs = {
                    ex.submit(_rescue_one, t, ref, args.max_displacement,
                              args.keep_point_cloud, args.force): t
                    for t in targets
                }
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    outcome, stem = fut.result()
                    outcomes.append((outcome, stem))
                    print(f"[{done}/{len(targets)}] {stem} -> {outcome}", flush=True)
        else:
            for i, t in enumerate(targets, 1):
                outcome, stem = _rescue_one(t, ref, args.max_displacement,
                                            args.keep_point_cloud, args.force)
                outcomes.append((outcome, stem))
                print(f"[{i}/{len(targets)}] {stem} -> {outcome}", flush=True)
    else:
        for t in targets:
            out_dem = RESCUE_ROOT / "asp_aligned" / f"{t['stem']}{_SUFFIX}"
            outcomes.append(("rescued" if out_dem.exists() else "missing", t["stem"]))

    # --- summary + per-strip metrics for the rescued set ---
    by_stem = {t["stem"]: t for t in targets}
    counts: dict[str, int] = {}
    for o, _ in outcomes:
        counts[o] = counts.get(o, 0) + 1
    print("\n" + "=" * 72)
    print("RESCUE SUMMARY")
    print("=" * 72)
    for k in ("rescued", "still_bad", "no_static", "error", "missing"):
        if counts.get(k):
            print(f"  {k:10s}: {counts[k]}")
    n_resc = counts.get("rescued", 0)
    print(f"\n  -> recovered {n_resc} / {len(targets)} quarantined strips "
          f"({100 * n_resc / max(1, len(targets)):.0f}%)")

    rows = []
    for outcome, stem in outcomes:
        if outcome != "rescued":
            continue
        t = by_stem[stem]
        aln_dir = RESCUE_ROOT / "asp_aligned"
        new_delta = _read_translation_magnitude(str(aln_dir / stem))
        dem = aln_dir / f"{stem}{_SUFFIX}"
        ctrl = (_control_residual_stats(str(dem), str(t["csv"]))
                if t["csv"] is not None else {"n": 0, "median": np.nan, "nmad": np.nan})
        rows.append({
            "stem": stem,
            "orig_delta": t["orig_delta"],
            "new_delta": new_delta if new_delta is not None else np.nan,
            "ctrl_n": ctrl["n"],
            "ctrl_med": ctrl["median"],
            "ctrl_nmad": ctrl["nmad"],
        })
    if rows:
        df = pd.DataFrame(rows).sort_values("orig_delta")
        print("\n" + "-" * 72)
        print("RESCUED strips: original sparse |Δ| -> new dense |Δ| (m), held-out control residual")
        print("-" * 72)
        with pd.option_context("display.float_format", lambda v: f"{v:11.3f}",
                               "display.width", 220, "display.max_columns", 20):
            print(df.to_string(index=False))
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = config.RESULTS_DIR / "rescue_bad_aligns_metrics.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")
        print(f"\nRescued aligned DEMs are in {RESCUE_ROOT / 'asp_aligned'}/ "
              f"(<stem>{_SUFFIX}). Folding them into the production stack is a "
              f"separate, deliberate step.")


if __name__ == "__main__":
    main()
