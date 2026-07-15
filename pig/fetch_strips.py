"""Search the REMA strip index for the PIG AOI + study window and download.

Stage 0 of the Pine Island pipeline. Uses ``pdemtools`` (via
``stereo_melt.io.rema``) to query the s2s041 strip index for strips that
overlap ``config.PIG_AOI_SHP`` between ``config.START_TIME`` and
``config.END_TIME``, then downloads any missing ones to
``config.STRIPS_DIR``.

Run:

    python -m pig.fetch_strips
"""

from __future__ import annotations

import os
from pathlib import Path

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

from pig import config


def main() -> None:
    config.ensure_output_dirs()

    # Env-var overrides for one-shot pulls that span a different window
    # than the basin's primary processing config.
    start = os.environ.get("PIG_FETCH_START", config.START_TIME)
    end = os.environ.get("PIG_FETCH_END", config.END_TIME)

    # Prefer the wider stack/tilt extent shapefile when present so fetch
    # picks up strips covering the full target grid (matches the AOI used
    # by build_stack and the melt inversion). Falls back to the narrow
    # PIG_AOI_SHP if the wider one isn't defined.
    aoi_path = config.PIG_AOI_SHP
    with fiona.open(aoi_path) as src:
        polygon = shape(next(iter(src))["geometry"])

    print(f"AOI: {config.SHELF}  ({Path(aoi_path).name})  "
          f"bounds={polygon.bounds}, area={polygon.area / 1e6:.1f} km^2")
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

    strips = list_rema_strips(
        polygon=polygon,
        strip_index_fpath=config.STRIP_INDEX_SHP,
        date_range=(start, end),
    )
    if strips is None or len(strips) == 0:
        print("No strips found.")
        return

    times = pd.to_datetime(strips["pdt_time1"])
    print(f"\nFound {len(strips)} strips. Per-year counts:")
    print(times.dt.year.value_counts().sort_index().to_string())

    strips_dir = Path(config.STRIPS_DIR)
    existing = {p.stem for p in strips_dir.glob("*.tif")}
    new_mask = ~strips["dem_id"].isin(existing)
    print(f"\n{int(new_mask.sum())} of {len(strips)} are not yet on disk; downloading missing ones.")

    download_rema_strips(strips, strips_dir)

    print("\nFetching quality companion files (matchtag, bitmask)...")
    download_rema_quality_files(strips, strips_dir)


if __name__ == "__main__":
    main()
