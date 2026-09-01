"""
Classify every REMA strip in the shared `data/REMA/strips/SETSM_*.tif` pile
by which basin AOI(s) its footprint intersects.

Active basins (Beardmore, Nansen, PIG) define the "must keep" set: a strip is
keep-able if its footprint intersects any active AOI, since strips are filed
by image-pair ID and shared across whichever basins overlap them.

A strip is *deletable* iff:
  - it intersects only D+C and/or McMurdo (basins we haven't started), OR
  - it intersects no basin AOI at all (downloaded for some reason but unused).

Outputs: tallies, sizes, and a deletable-strip list.
"""

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

STRIPS_DIR = Path("/wd2/projects/stereo_melt/data/REMA/strips")
PARQUET = Path("/wd2/projects/stereo_melt/data/shapefiles/REMA_Strip_Index_s2s041.parquet")

# (label, path, active?) — use the wider STACK_AOI polygons, which is what
# `fetch_strips` actually uses (the narrow per-basin AOIs are for melt-rate
# interpretation / viz, not for strip selection).
AOIS = [
    ("beardmore", "/wd2/projects/stereo_melt/data/shapefiles/beardmore_stack_extent.shp", True),
    ("nansen",    "/wd2/projects/stereo_melt/data/shapefiles/nansen_stack_extent.shp",    True),
    ("pig",       "/wd2/projects/stereo_melt/data/shapefiles/pig_stack_extent.shp",       True),
    ("dotson_crosson", "/wd2/projects/stereo_melt/data/shapefiles/dotson_crosson_stack.shp", False),
    ("mcmurdo",   "/wd2/projects/stereo_melt/data/shapefiles/mcmurdo_stack_extent.shp",   False),
]


_REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
def main():
    # 1. List on-disk strips and sizes.
    print("Inventorying shared strips dir...")
    strip_paths = sorted(STRIPS_DIR.glob("SETSM_*.tif"))
    # Filter to base DEMs (drop _matchtag/_bitmask which we treat as siblings)
    base = [p for p in strip_paths if not any(s in p.name for s in ("_matchtag", "_bitmask"))]
    print(f"  {len(strip_paths):,} total .tif (incl. matchtag/bitmask)")
    print(f"  {len(base):,} base SETSM strips")

    size_by_id = {}
    sibling_size_by_id = {}
    for p in strip_paths:
        # dem_id = filename without suffix and without _matchtag/_bitmask
        stem = p.stem
        for suf in ("_matchtag", "_bitmask"):
            if stem.endswith(suf):
                dem_id = stem[: -len(suf)]
                sibling_size_by_id[dem_id] = sibling_size_by_id.get(dem_id, 0) + p.stat().st_size
                break
        else:
            dem_id = stem
            size_by_id[dem_id] = size_by_id.get(dem_id, 0) + p.stat().st_size

    # 2. Load strip-index parquet (this has per-strip footprints).
    print(f"Loading strip-index parquet ({PARQUET.stat().st_size/1e6:.0f} MB)...")
    idx = gpd.read_parquet(PARQUET)
    print(f"  columns: {list(idx.columns)[:20]}")
    print(f"  CRS: {idx.crs}")

    # Identify the dem_id column. The parquet has `dem_id` which carries the
    # full SETSM_s2s041_..._lsf_seg<N> stem matching the on-disk filename.
    id_col = None
    for cand in ("dem_id", "stripname", "name", "filename"):
        if cand in idx.columns:
            id_col = cand
            break
    if id_col is None:
        # Fall back: try to find any string column that contains SETSM_
        for c in idx.columns:
            try:
                if idx[c].dtype == object and idx[c].astype(str).str.contains("SETSM_", regex=False).any():
                    id_col = c
                    break
            except Exception:
                pass
    print(f"  using ID column: {id_col!r}")
    assert id_col is not None, "Couldn't find a dem-id column in the parquet"

    # 3. Project AOIs to the index CRS, build active and inactive unary unions.
    aoi_geoms = {}
    for label, path, _ in AOIS:
        g = gpd.read_file(path)
        if g.crs != idx.crs:
            g = g.to_crs(idx.crs)
        aoi_geoms[label] = g.union_all() if hasattr(g, "union_all") else g.unary_union
        print(f"  AOI {label:18s} loaded, area={g.area.sum()/1e6:.0f} km^2")

    # 4. Restrict the index to strips currently on disk.
    on_disk = set(size_by_id.keys())
    # Strip-index dem-id can carry trailing version/suffix; match on prefix.
    # Fast path: filter rows whose id_col exact-matches a disk file.
    idx_disk = idx[idx[id_col].astype(str).isin(on_disk)].copy()
    print(f"\n{len(idx_disk):,} of {len(on_disk):,} on-disk strips matched in parquet by exact id")
    if len(idx_disk) < 0.5 * len(on_disk):
        # Try a prefix match — some entries may have extra suffixes
        print("  (poor exact-match rate; trying prefix match)")
        all_idx_ids = idx[id_col].astype(str).tolist()
        prefix_set = set()
        for did in on_disk:
            for ii in all_idx_ids:
                if ii.startswith(did) or did.startswith(ii):
                    prefix_set.add(ii)
                    break
        idx_disk = idx[idx[id_col].astype(str).isin(prefix_set)].copy()
        print(f"  prefix match: {len(idx_disk):,} rows")

    # 5. Spatial classification (vectorized via GeoSeries.intersects).
    print("\nClassifying strip footprints against basin AOIs...")
    classes = {label: set() for label, _, _ in AOIS}
    geoms = gpd.GeoSeries(idx_disk.geometry.values, index=idx_disk[id_col].astype(str).values, crs=idx.crs)
    valid_mask = ~(geoms.is_empty | geoms.isna())
    valid_geoms = geoms[valid_mask]
    no_aoi = list(geoms.index[~valid_mask])
    for label, _, _ in AOIS:
        hits_mask = valid_geoms.intersects(aoi_geoms[label])
        classes[label] = set(valid_geoms.index[hits_mask])
    union_hit = set().union(*classes.values())
    no_aoi += [did for did in valid_geoms.index if did not in union_hit]

    print("\n=== Per-basin strip counts (on-disk strips intersecting each AOI) ===")
    for label, _, active in AOIS:
        marker = "ACTIVE" if active else "candidate-for-cleanup"
        print(f"  {label:18s} {marker:22s}  {len(set(classes[label])):5,d} strips")
    print(f"  no AOI hit             (downloaded but no AOI)  {len(set(no_aoi)):5,d} strips")

    # 6. Active set is the union of active-basin strip IDs.
    active_set = set()
    for label, _, active in AOIS:
        if active:
            active_set.update(classes[label])
    inactive_only_set = (set(classes["dotson_crosson"]) | set(classes["mcmurdo"])) - active_set
    no_aoi_set = set(no_aoi) - active_set

    deletable = inactive_only_set | no_aoi_set
    print(f"\n=== Summary ===")
    print(f"  Active set (Beardmore∪Nansen∪PIG):  {len(active_set):5,d} strips (KEEP)")
    print(f"  D+C/McMurdo only:                   {len(inactive_only_set):5,d} strips")
    print(f"  No-AOI orphans:                     {len(no_aoi_set):5,d} strips")
    print(f"  -> Deletable total:                 {len(deletable):5,d} strips")

    # 7. Size accounting.
    main_size = sum(size_by_id.get(did, 0) for did in deletable)
    sib_size = sum(sibling_size_by_id.get(did, 0) for did in deletable)
    print(f"\n  Deletable base .tif size:           {main_size/1e9:7.1f} GB")
    print(f"  Deletable matchtag+bitmask size:    {sib_size/1e9:7.1f} GB")
    print(f"  TOTAL freeable:                     {(main_size+sib_size)/1e9:7.1f} GB")

    # 8. Write deletable list for review.
    out = Path(f"{_REPO}/scripts/deletable_strips.txt")
    with out.open("w") as f:
        for did in sorted(deletable):
            f.write(f"{did}\n")
    print(f"\n  Wrote deletable IDs to {out}")


if __name__ == "__main__":
    main()
