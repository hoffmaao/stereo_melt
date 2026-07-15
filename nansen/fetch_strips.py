"""Search the REMA strip index for the Nansen AOI + study window and download.

Stage 0 of the Nansen pipeline. Mirrors :mod:`beardmore.fetch_strips`.
Renders an AOI / strip-coverage QC figure to ``nansen/figures/`` before
downloading so the user can verify the polygon being queried.

Run:

    python -m nansen.fetch_strips
"""

from __future__ import annotations

import os

import fiona
import pandas as pd
from shapely.geometry import shape

# Point pyproj at the env-local proj.db; the conda base proj.db is corrupt.
os.environ.setdefault("PROJ_DATA", "/home/hoffmaao/miniconda3/envs/stereo_melt/share/proj")

from stereo_melt.io.rema import (
    download_rema_quality_files,
    download_rema_strips,
    list_rema_strips,
    plot_aoi_strip_coverage,
)

from nansen import config


def main() -> None:
    config.ensure_output_dirs()

    # Env overrides let us fetch a different time window than the basin's
    # primary processing config (e.g. pre-IS2 strip backfill while the
    # IS2-era stack stays at config.START_TIME/END_TIME).
    start = os.environ.get("NANSEN_FETCH_START", config.START_TIME)
    end = os.environ.get("NANSEN_FETCH_END", config.END_TIME)

    # Use the wider stack/tilt extent for fetching when defined — otherwise
    # we'd miss the strips needed to populate the wider analysis grid.
    aoi_path = config.NANSEN_AOI_SHP
    with fiona.open(aoi_path) as src:
        polygon = shape(next(iter(src))["geometry"])

    from pathlib import Path
    print(f"AOI: {config.SHELF} ({Path(aoi_path).name}), bounds={polygon.bounds}, "
          f"area={polygon.area / 1e6:.1f} km^2")
    print(f"Window: [{start}, {end})")

    coverage_png = config.FIGURES_DIR / f"aoi_strip_coverage_{start}_{end}.png"
    n_in_index = plot_aoi_strip_coverage(
        polygon=polygon,
        strip_index_fpath=config.STRIP_INDEX_SHP,
        output_path=coverage_png,
        date_range=(start, end),
        title=f"{config.SHELF} AOI strip coverage [{start}, {end})",
    )
    print(f"  wrote {coverage_png}  ({n_in_index} strips intersect AOI in window)")

    # list_rema_strips wants a shapely Polygon, not a bounds tuple.
    # Nansen's wider stack AOI is a single rectangular Polygon so this
    # is a direct pass-through; if a future basin AOI is a MultiPolygon,
    # wrap with shapely.geometry.box(*polygon.bounds).
    strips = list_rema_strips(
        polygon=polygon,
        strip_index_fpath=config.STRIP_INDEX_SHP,
        date_range=(start, end),
    )
    if strips is None or len(strips) == 0:
        print("No strips found by pdemtools.search.")
        return

    times = pd.to_datetime(strips["pdt_time1"])
    print(f"\nFound {len(strips)} strips. Per-year counts:")
    print(times.dt.year.value_counts().sort_index().to_string())

    strips_dir = config.STRIPS_DIR
    existing = {p.stem for p in strips_dir.glob("*.tif")}
    new_mask = ~strips["dem_id"].isin(existing)
    print(f"\n{int(new_mask.sum())} of {len(strips)} are not yet on disk; downloading missing ones.")

    download_rema_strips(strips, strips_dir)

    print("\nFetching quality companion files (matchtag, bitmask)...")
    download_rema_quality_files(strips, strips_dir)


if __name__ == "__main__":
    main()
