"""Shean-style "nocorr" ingestion of never-coregistered Beardmore_Shelf strips.

Shean 2019 kept every DEM his ICP could not anchor (``stack_proc.sh``:
a-priori geolocation + one class-mean bias via ``stack_nocorr_adjust.py``,
then the joint tilt LSQ with a loosened Ez=1.0 prior, dropping only DEMs
the LSQ could not adjust). This driver is the ingestion step: selected
strips are copied into ``data/ASP_nocorr/asp_aligned/`` with ``--z-offset``
added and a ``sources=["nocorr"]`` sidecar for the Ez tier.

Nocorr classes (07-11 full-record census):

- **fully-floating** (default): in-window AOI strips that do not touch
  grounded ice — no control is ever possible (IS2-era: 101; pre-IS2: 211;
  GLAS-era: 6).
- ``--include-align-failed``: grounded-touching strips whose pc_align was
  ATTEMPTED and died without a DEM (log exists, no ``.bad_align``) — the
  "no point to minimize" near-floating class (IS2-era: 8; pre-IS2: 8;
  GLAS-era: 11).
- ``--include-no-control``: grounded-touching strips never attempted (no
  control cache hit, upfront-SKIPPED by the align driver; pre-IS2: 95;
  GLAS-era: 12).
- ``--readmit-underdetermined``: QC-quarantined strips whose only sin was
  a thin control count (``screen==QUARANTINE`` with ``n_ctl < 50`` but
  small before/after medians, read from ``logs/qc_<slug>_align.csv``) —
  their weak ICP solution is discarded and the RAW strip re-enters as
  nocorr, Shean-faithfully. Known-bad a-priori (|before_med| >
  ``--max-known-before``, default 15 m — the SETSM blunder class) and
  misconverged strips stay dead. Align-time translation-cap quarantines
  are NOT re-admitted (their huge ICP translations mark suspect a-priori).

**Per-era class bias** (measured 07-11, diff = control − DEM): run once per
era with the era window in the environment and the era's offset —

    BEARDMORE_SHELF_START=2009-01-01 BEARDMORE_SHELF_END=2010-10-13 \
        ... ingest_nocorr --z-offset -2.25 --include-align-failed \
        --include-no-control --readmit-underdetermined       # GLAS era
    BEARDMORE_SHELF_START=2010-10-13 BEARDMORE_SHELF_END=2019-01-01 \
        ... ingest_nocorr --z-offset -0.80 ...                # pre-IS2
    BEARDMORE_SHELF_START=2019-01-01 BEARDMORE_SHELF_END=2024-01-10 \
        ... ingest_nocorr --z-offset -2.34 --include-align-failed  # IS2

Downstream: ``BEARDMORE_SHELF_SOURCES=<mode> build_stack`` fuses the roots,
then ``tilt_fit`` with the shelf-inclusive domain + dh/dt smoothness
(BEARDMORE_SHELF_TILT_DOMAIN=full, BEARDMORE_SHELF_TILT_DHDT_SMOOTH=1.0)
sets each nocorr epoch's αz by cross-epoch self-consistency — the
static-domain tilt CANNOT correct them (07-11 A/B: αz collapsed to the
prior and every band degraded).

Run (disconnect-safe):

    cd /wd2/projects/stereo_melt
    nohup /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u \
        -m beardmore_shelf.ingest_nocorr \
        > beardmore_shelf/logs/ingest_nocorr.log 2>&1 &
"""
from __future__ import annotations

import stereo_melt.envsetup  # noqa: F401  (PROJ_DATA fix, before geo imports)

import argparse
import time
import traceback
from pathlib import Path

import geopandas as gpd
import pandas as pd

from stereo_melt.coregister.asp import ingest_strip_nocorr

from beardmore_shelf import config
from beardmore_shelf.cache_icesat2 import (
    _date_from_path,
    _load_aoi,
    find_in_window_strips,
)

# Trans roots where a grounded-touching strip could have been aligned or
# attempted. ASP_nocorr is deliberately absent: ingest idempotency is
# handled per strip in main() via the existing-DEM skip.
TRANS_ROOTS = ("ASP", "ASP_ctempoatm", "ASP_glas")
# QC screen tables (qc_glas_align.py --root <name>) consulted by
# --readmit-underdetermined.
QC_TABLES = (("glas", "ASP_glas"), ("ctempoatm", "ASP_ctempoatm"))


def _trans_dirs():
    return [Path(config.BASIN_DIR) / "data" / r / "asp_aligned" for r in TRANS_ROOTS]


def find_nocorr_strips(include_align_failed: bool = False,
                       include_no_control: bool = False):
    """Nocorr-candidate strips in the (env-overridable) config window.

    Returns ``[(path, timestamp, klass), ...]`` where klass is one of
    ``floating`` / ``align_failed`` / ``no_control``.
    """
    aoi = _load_aoi()
    grounded = gpd.read_file(config.GROUNDING_LINE_SHP).iloc[0].geometry
    in_aoi = find_in_window_strips(aoi_polygon=aoi)
    grounded_ok = {
        p.stem for p, _ in find_in_window_strips(
            aoi_polygon=aoi, grounded_polygon=grounded
        )
    }
    out = [(p, t, "floating") for p, t in in_aoi if p.stem not in grounded_ok]

    if include_align_failed or include_no_control:
        dirs = _trans_dirs()
        for p, t in in_aoi:
            if p.stem not in grounded_ok:
                continue
            if any((d / f"{p.stem}-trans_reference-DEM.tif").exists() for d in dirs):
                continue  # aligned somewhere: trans, not nocorr
            if any((d / f"{p.stem}.bad_align").exists() for d in dirs):
                continue  # quarantined: --readmit-underdetermined territory
            attempted = any(
                next(iter(d.glob(f"{p.stem}-log-pc_align-*.txt")), None) is not None
                for d in dirs
            )
            if attempted and include_align_failed:
                out.append((p, t, "align_failed"))
            elif not attempted and include_no_control:
                out.append((p, t, "no_control"))
    return out


def find_readmit_strips(max_known_before: float, min_ctl: int = 50,
                        max_after: float = 2.0):
    """Under-determined QC quarantines eligible for nocorr re-admission."""
    strips_dir = Path(config.STRIPS_DIR)
    t_start = pd.Timestamp(config.START_TIME)
    t_end = pd.Timestamp(config.END_TIME)
    out = []
    for slug, root in QC_TABLES:
        csvp = Path(config.BASIN_DIR) / "logs" / f"qc_{slug}_align.csv"
        if not csvp.exists():
            continue
        df = pd.read_csv(csvp)
        if "screen" not in df.columns:
            continue
        sel = df[
            (df["screen"] == "QUARANTINE")
            & (df["n_ctl"] < min_ctl)
            & (df["before_med"].abs() <= max_known_before)
            & (df["after_med"].abs() <= max_after)
        ]
        for sid in sel["strip"]:
            p = strips_dir / f"{sid}.tif"
            if not p.exists():
                continue
            t = _date_from_path(p)
            if t is None or not (t_start <= t < t_end):
                continue
            out.append((p, t, f"readmit[{root}]"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--z-offset", type=float, default=config.NOCORR_Z_OFFSET_M,
                   help=f"class-mean vertical bias to ADD, m "
                        f"(default {config.NOCORR_Z_OFFSET_M}; use the ERA "
                        f"constant — glas −2.25 / preis2 −0.80 / is2 −2.34)")
    p.add_argument("--include-align-failed", action="store_true",
                   help="also ingest grounded-touching strips whose pc_align "
                        "was attempted and produced no DEM")
    p.add_argument("--include-no-control", action="store_true",
                   help="also ingest grounded-touching strips never attempted "
                        "(no control cache hit)")
    p.add_argument("--readmit-underdetermined", action="store_true",
                   help="re-ingest QC-quarantined strips whose only failure "
                        "was n_ctl<50 with small before/after medians")
    p.add_argument("--max-known-before", type=float, default=15.0,
                   help="re-admission cap on |before_med| (blunder screen), m")
    p.add_argument("--limit", type=int, default=None,
                   help="ingest at most N strips (smoke test)")
    p.add_argument("--dem-id", type=str, default=None,
                   help="ingest only this strip")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    todo = find_nocorr_strips(
        include_align_failed=args.include_align_failed,
        include_no_control=args.include_no_control,
    )
    if args.readmit_underdetermined:
        have = {p.stem for p, _, _ in todo}
        todo += [r for r in find_readmit_strips(args.max_known_before)
                 if r[0].stem not in have]
    todo.sort(key=lambda x: x[1])

    print(f"window {config.START_TIME} .. {config.END_TIME}   "
          f"z-offset {args.z_offset:+.2f} m")
    by_klass: dict[str, int] = {}
    for _, _, k in todo:
        by_klass[k] = by_klass.get(k, 0) + 1
    print(f"nocorr candidates: {len(todo)}  by class: {by_klass}")
    if args.dem_id is not None:
        todo = [(p, t, k) for p, t, k in todo if p.stem == args.dem_id]
        if not todo:
            raise SystemExit(f"--dem-id {args.dem_id!r} not in the nocorr set")
    if args.limit is not None:
        todo = todo[: args.limit]

    counts = {"ok": 0, "skip": 0, "fail": 0}
    for i, (path, t, klass) in enumerate(todo, start=1):
        try:
            before = (config.ASP_NOCORR_ALIGNED_DIR
                      / f"{path.stem}-trans_reference-DEM.tif").exists()
            ingest_strip_nocorr(
                str(path), str(config.ASP_NOCORR_ROOT), args.z_offset,
                overwrite=args.overwrite,
            )
            counts["skip" if (before and not args.overwrite) else "ok"] += 1
        except Exception as exc:
            print(f"!! FAILED for {path.name}: {exc}")
            traceback.print_exc()
            counts["fail"] += 1
        print(f"[{i}/{len(todo)}] {t.date()}  {path.stem}  [{klass}]  "
              f"(ok={counts['ok']} skip={counts['skip']} fail={counts['fail']})")

    print(f"\n=== nocorr ingest summary: {counts['ok']} new, "
          f"{counts['skip']} cached, {counts['fail']} failed "
          f"(of {len(todo)}) in {(time.time() - t0) / 60:.1f} min ===")
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
