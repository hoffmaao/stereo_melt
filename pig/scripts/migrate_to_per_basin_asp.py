"""Move PIG-AOI aligned strips out of shared ASP roots into pig/data/ASP/.

Run once. Idempotent — re-running after a partial move is safe.

Source dirs scanned:
  /wd2/projects/stereo_melt/data/REMA/strips/ASP/asp_aligned/        (old shared)
  /wd2/projects/stereo_melt/data/REMA/strips/ASP_is2cs2/asp_aligned/ (variant-split)
  /wd2/projects/stereo_melt/data/REMA/strips/ASP_cs2/asp_aligned/    (variant-split)

Target dir:
  /wd2/projects/stereo_melt/pig/data/ASP/asp_aligned/

For each ``-trans_reference-DEM.tif`` whose bbox intersects the PIG AOI bbox,
move every file with the same stem (DEM, point-cloud raster, transform/log/csv).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import rasterio

from pig import config
from pig.build_stack import _aoi_bounds_3031, _stack_aoi_path

SOURCE_ROOTS = [
    Path("/wd2/projects/stereo_melt/data/REMA/strips/ASP/asp_aligned"),
    Path("/wd2/projects/stereo_melt/data/REMA/strips/ASP_is2cs2/asp_aligned"),
    Path("/wd2/projects/stereo_melt/data/REMA/strips/ASP_cs2/asp_aligned"),
]
TARGET = config.BASIN_DIR / "data" / "ASP" / "asp_aligned"

DEM_SUFFIX = "-trans_reference-DEM.tif"


def overlaps_aoi(p: Path, bbox: tuple[float, float, float, float]) -> bool:
    a_xmin, a_ymin, a_xmax, a_ymax = bbox
    with rasterio.open(p) as src:
        bx_min, by_min, bx_max, by_max = src.bounds
    return not (
        bx_max < a_xmin or bx_min > a_xmax or by_max < a_ymin or by_min > a_ymax
    )


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    bbox = _aoi_bounds_3031(_stack_aoi_path())
    print(f"PIG AOI bbox: {bbox}")
    print(f"Target: {TARGET}")

    grand_total_dems = 0
    grand_total_files = 0
    grand_total_bytes = 0
    for src_root in SOURCE_ROOTS:
        if not src_root.exists():
            print(f"\n[skip] {src_root} does not exist")
            continue
        dems = list(src_root.glob(f"*{DEM_SUFFIX}"))
        pig_dems = [p for p in dems if overlaps_aoi(p, bbox)]
        print(f"\n[{src_root.name}] {len(pig_dems)}/{len(dems)} PIG-AOI DEMs to migrate")

        for dem in pig_dems:
            stem = dem.name[: -len(DEM_SUFFIX)]
            companions = list(src_root.glob(f"{stem}*"))
            n_files = len(companions)
            total_bytes = sum(c.stat().st_size for c in companions)
            for src in companions:
                dst = TARGET / src.name
                if dst.exists():
                    print(f"    [exists] {dst.name}")
                    continue
                shutil.move(str(src), str(dst))
            print(
                f"  {stem}  →  {n_files} files, {total_bytes / 1e9:.2f} GB"
            )
            grand_total_dems += 1
            grand_total_files += n_files
            grand_total_bytes += total_bytes

    print(
        f"\nMigrated: {grand_total_dems} DEMs, "
        f"{grand_total_files} files total, "
        f"{grand_total_bytes / 1e9:.1f} GB (inode-rename, no copy)"
    )
    print(f"\nNext step: update pig/config.STRIP_SOURCES to point at {TARGET}")


if __name__ == "__main__":
    main()
