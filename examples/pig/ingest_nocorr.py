"""Shean-style "nocorr" ingestion of never-coregistered PIG strips.

Test of 2026-09-06 ("include more DEMs over the shelf"): the REMA s2s041 index
holds 373 in-window strips that touch the Pine Island shelf but were never
aligned, because the strip search (:mod:`stereo_melt.io.rema`) admits only
strips that intersect a rock or slow-ice control polygon. Shean 2019 kept such
DEMs ("nocorr", ``stack_proc.sh``): a-priori geolocation + ONE class-mean
vertical bias (``stack_nocorr_adjust.py``) + the joint tilt LSQ with a loosened
Ez = 1.0 prior, dropping only DEMs the LSQ could not adjust. This driver is the
ingestion step of that recipe for PIG.

Selection (:func:`select_nocorr_candidates`): index strips in [START, END)
whose footprint intersects the stack AOI, that are in NO production align root
(``config.STRIP_SOURCES`` granule stems), not in ``BAD_STRIPS``, that overlap
the Pine Island shelf polygon by at least ``--min-shelf-km2`` (default 200),
and whose raw ``<dem_id>.tif`` is under ``config.STRIPS_DIR``. Written to
``results/pig_nocorr_selection.csv``.

Class bias (``--estimate-bias``): median over the production-aligned strips of
each era of the per-strip median (control − raw DEM) at that strip's
``reference_files/combined_reference_<dem_id>.csv`` points — what ``pc_align``
had to remove vertically (Shean: "DEMs are above points", −3.1 m for his
2010–2016 WorldView DEMs). Written to ``results/pig_nocorr_class_bias.csv``.
Eras: IS2 (acqdate ≥ 2018-10-14, the ATL06 start) and pre-IS2.

Ingest (``--z-offset-pre-is2 X --z-offset-is2 Y``): each selected strip is
copied into ``data/ASP_nocorr/asp_aligned/<dem_id>-trans_reference-DEM.tif``
with the era offset ADDED and a ``sources=["nocorr"]`` sidecar
(:func:`stereo_melt.coregister.asp.ingest_strip_nocorr`). Idempotent.

Downstream: ``PIG_SOURCES=nocorr pig.build_stack --res 250 --tag <tag>`` fuses
the root (last in precedence), then ``PIG_SOURCES=nocorr PIG_TILT_DOMAIN=full
PIG_TILT_DHDT_SMOOTH=1.0 pig.tilt_fit`` — the static-domain tilt gives
control-free epochs zero observation rows (beardmore_shelf 2026-07-11 A/B).
The PGC quality screen (matchtag 1, bitmask keep {0, 2}) needs nothing here:
``build_stack`` resolves ``<dem_id>_matchtag.tif``/``_bitmask.tif`` from
``STRIPS_DIR`` for EVERY root in ``STRIP_SOURCES``, this one included, and
applies it at fuse time.

What actually went wrong on 2026-09-12, so a follow-up does not chase the wrong
cause: the joint tilt LSQ could not adjust a large minority of the nocorr
epochs — about 24 of 124 came out with residual NMAD > 5 m against the stack's
temporal median, the worst 60–200 m — and they stayed in the fuse. Shean's
recipe drops exactly those (``stack_filter.py remove_nocorr``); that post-fit
screen is NOT shipped here, and without it the stack carries epochs the LSQ
left unconstrained. Compounding it, the class bias is one constant per era, and
the 2010–2013 strips are under-corrected by several metres against it.

The outcome was negative and the path is NOT adopted: the screened variant came
out about 6 % noisier than the production stack on common pixels. It stays
opt-in behind ``PIG_SOURCES=nocorr`` for follow-up work only.

Run (from ``examples/``)::

    $PY -m pig.ingest_nocorr --estimate-bias [--n-per-era 80]
    $PY -m pig.ingest_nocorr --z-offset-pre-is2 <X> --z-offset-is2 <Y>
"""
from __future__ import annotations

import stereo_melt.envsetup  # noqa: F401  (PROJ_DATA fix, before geo imports)

import argparse
import os
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from stereo_melt.coregister.asp import ingest_strip_nocorr

from pig import config

IS2_START = pd.Timestamp("2018-10-14", tz="UTC")
ERA_ROOTS = (
    ("pre_is2", config.ASP_CTEMPOATMLVIS_ROOT),
    ("is2", config.ASP_IS2CTEMPOATMLVIS_ROOT),
)


def _union(geoms):
    return geoms.union_all() if hasattr(geoms, "union_all") else geoms.unary_union


def _raw_dem_index() -> dict[str, Path]:
    """Map raw strip stem -> path for every ``*.tif`` under STRIPS_DIR."""
    out: dict[str, Path] = {}
    for root, _dirs, files in os.walk(config.STRIPS_DIR):
        for f in files:
            if f.endswith(".tif") and "SETSM" in f and not any(
                k in f for k in ("matchtag", "bitmask", "ortho")
            ):
                out.setdefault(f[:-4], Path(root) / f)
    return out


def _aligned_stems() -> set[str]:
    stems: set[str] = set()
    for aligned_dir, variant in config.STRIP_SOURCES:
        # PIG_SOURCES=nocorr appends our own output root to STRIP_SOURCES, so skip
        # it here or a re-run (e.g. --overwrite with a revised bias) selects nothing.
        if variant == "nocorr" or Path(aligned_dir) == config.ASP_NOCORR_ALIGNED_DIR:
            continue
        for p in Path(aligned_dir).glob("*-trans_reference-DEM.tif"):
            stems.add(p.name.replace("-trans_reference-DEM.tif", ""))
    return stems


def select_nocorr_candidates(min_shelf_km2: float = 200.0) -> pd.DataFrame:
    idx = gpd.read_parquet(config.STRIP_INDEX_SHP)
    idx["dem_id"] = idx["dem_id"].astype(str)
    aoi = _union(gpd.read_file(config.PIG_AOI_SHP).to_crs(idx.crs).geometry)
    shelf = gpd.read_file(config.ICESHELF_FEATURE_SHP)
    shelf = shelf[shelf["NAME"] == config.ICESHELF_FEATURE_NAME].to_crs(idx.crs)
    shelf_geom = _union(shelf.geometry)
    t = pd.to_datetime(idx["acqdate1"], utc=True)
    win = (t >= pd.Timestamp(config.START_TIME, tz="UTC")) & (
        t < pd.Timestamp(config.END_TIME, tz="UTC")
    )
    cand = idx[win & idx.intersects(aoi)].copy()
    aligned = _aligned_stems()
    bad = set(config.BAD_STRIPS)
    cand = cand[~cand.dem_id.isin(aligned) & ~cand.dem_id.isin(bad)]
    cand = cand[cand.intersects(shelf_geom)].copy()
    m = cand.to_crs("EPSG:3031")
    shelf_m = _union(shelf.to_crs("EPSG:3031").geometry)
    cand["shelf_km2"] = m.geometry.intersection(shelf_m).area.values / 1e6
    cand = cand[cand.shelf_km2 >= min_shelf_km2]
    raw = _raw_dem_index()
    cand["raw_path"] = cand.dem_id.map(lambda d: str(raw.get(d, "")))
    cand["is2era"] = pd.to_datetime(cand["acqdate1"], utc=True) >= IS2_START
    n_missing = int((cand.raw_path == "").sum())
    cand = cand[cand.raw_path != ""]
    out = pd.DataFrame({
        "dem_id": cand.dem_id.values,
        "acqdate1": pd.to_datetime(cand["acqdate1"], utc=True).dt.strftime("%Y-%m-%d").values,
        "shelf_km2": np.round(cand.shelf_km2.values, 1),
        "is2era": cand.is2era.values,
        "raw_path": cand.raw_path.values,
    }).sort_values("acqdate1").reset_index(drop=True)
    print(f"nocorr candidates: {len(out)} strips (shelf overlap >= {min_shelf_km2:g} km2; "
          f"{n_missing} more have no raw DEM on disk); IS2-era {int(out.is2era.sum())}, "
          f"pre-IS2 {int((~out.is2era).sum())}; total shelf overlap "
          f"{out.shelf_km2.sum():.0f} strip-km2")
    return out


def _sample_dem(path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        vals = np.array([v[0] for v in src.sample(zip(x, y))], dtype=float)
        nodata = src.nodata
    if nodata is not None:
        vals[vals == nodata] = np.nan
    vals[np.abs(vals) > 1e4] = np.nan
    return vals


def estimate_class_bias(n_per_era: int = 80, seed: int = 0, min_pts: int = 100) -> pd.DataFrame:
    """Median (control - raw DEM) per era over the production-aligned strips."""
    raw = _raw_dem_index()
    rng = np.random.default_rng(seed)
    rows = []
    for era, root in ERA_ROOTS:
        csvs = sorted(Path(root, "reference_files").glob("combined_reference_*.csv"))
        csvs = [c for c in csvs if c.name[len("combined_reference_"):-4] in raw]
        if len(csvs) > n_per_era:
            csvs = [csvs[i] for i in sorted(rng.choice(len(csvs), n_per_era, replace=False))]
        print(f"[{era}] sampling {len(csvs)} aligned strips against their raw DEMs ...", flush=True)
        t0 = time.time()
        for k, c in enumerate(csvs, 1):
            dem_id = c.name[len("combined_reference_"):-4]
            ctl = pd.read_csv(c)
            if len(ctl) < min_pts:
                continue
            dem = _sample_dem(raw[dem_id], ctl["easting"].values, ctl["northing"].values)
            diff = ctl["h_mean"].values - dem
            diff = diff[np.isfinite(diff)]
            if diff.size < min_pts:
                continue
            rows.append({"era": era, "dem_id": dem_id, "year": int(dem_id.split("_")[3][:4]),
                         "n_pts": int(diff.size), "median_ctl_minus_dem_m": float(np.median(diff)),
                         "nmad_m": float(1.4826 * np.median(np.abs(diff - np.median(diff))))})
            if k % 20 == 0:
                print(f"   {k}/{len(csvs)}  {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "pig_nocorr_class_bias.csv"
    df.to_csv(out, index=False)
    print(f"\nper-strip medians written to {out}")
    for era, g in df.groupby("era"):
        v = g.median_ctl_minus_dem_m
        print(f"  {era:8s} n={len(g):3d}  class bias (median of medians) {v.median():+.2f} m  "
              f"IQR [{v.quantile(.25):+.2f}, {v.quantile(.75):+.2f}]  by year: "
              f"{g.groupby('year').median_ctl_minus_dem_m.median().round(2).to_dict()}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimate-bias", action="store_true")
    ap.add_argument("--n-per-era", type=int, default=80)
    ap.add_argument("--min-shelf-km2", type=float, default=200.0)
    ap.add_argument("--z-offset-pre-is2", type=float, default=None,
                    help="class bias ADDED to pre-IS2 nocorr strips (m)")
    ap.add_argument("--z-offset-is2", type=float, default=None,
                    help="class bias ADDED to IS2-era nocorr strips (m)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    config.ensure_output_dirs()

    if args.estimate_bias:
        estimate_class_bias(n_per_era=args.n_per_era)
        return 0

    sel = select_nocorr_candidates(args.min_shelf_km2)
    sel_path = config.RESULTS_DIR / "pig_nocorr_selection.csv"
    sel.drop(columns=["raw_path"]).to_csv(sel_path, index=False)
    print(f"selection written to {sel_path}")
    if args.z_offset_pre_is2 is None or args.z_offset_is2 is None:
        print("no --z-offset-pre-is2 / --z-offset-is2 given: selection only, nothing ingested")
        return 0

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    for i, r in enumerate(sel.itertuples(), start=1):
        z = args.z_offset_is2 if r.is2era else args.z_offset_pre_is2
        try:
            before = (config.ASP_NOCORR_ALIGNED_DIR
                      / f"{Path(r.raw_path).stem}-trans_reference-DEM.tif").exists()
            ingest_strip_nocorr(r.raw_path, str(config.ASP_NOCORR_ROOT), z, overwrite=args.overwrite)
            counts["skip" if (before and not args.overwrite) else "ok"] += 1
        except Exception as exc:
            print(f"!! FAILED for {r.dem_id}: {exc}", flush=True)
            traceback.print_exc()
            counts["fail"] += 1
        if i % 10 == 0:
            print(f"   {i}/{len(sel)}  (ok={counts['ok']} skip={counts['skip']} "
                  f"fail={counts['fail']})  {time.time() - t0:.0f} s", flush=True)
    print(f"\n=== nocorr ingest summary: {counts['ok']} new, {counts['skip']} cached, "
          f"{counts['fail']} failed (of {len(sel)}) into {config.ASP_NOCORR_ALIGNED_DIR} "
          f"in {time.time() - t0:.0f} s ===")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
