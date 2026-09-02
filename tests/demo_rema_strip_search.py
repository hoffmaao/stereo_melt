"""Demo: search the REMA strip index for a given AOI and time window.

Exercises the library's REMA strip-search API on the Nansen Ice Shelf
(2022-2023) as a standalone integration test — does not depend on any
Beardmore-specific data or config.

Requires:
- REMA_Strip_Index_s2s041.parquet on disk (pan-Antarctic)
- IceShelf_Antarctica_v02.shp on disk (to look up the Nansen polygon)
"""

from pathlib import Path

import fiona
from shapely.geometry import shape

from stereo_melt.io.rema import list_rema_strips

STRIP_INDEX = Path("/wd2/projects/stereo_melt/data/shapefiles/REMA_Strip_Index_s2s041.parquet")
ICE_SHELF_SHP = Path("/wd2/projects/stereo_melt/data/shapefiles/IceShelf_Antarctica_v02.shp")


def load_shelf_polygon(name: str):
    """Return the shapely polygon for an ice shelf by NAME attribute."""
    with fiona.open(ICE_SHELF_SHP) as src:
        for feat in src:
            if str(feat["properties"].get("NAME", "")).strip().lower() == name.lower():
                return shape(feat["geometry"])
    raise KeyError(f"No shelf named {name!r} in {ICE_SHELF_SHP}")


def main():
    print(f"Loading Nansen AOI from {ICE_SHELF_SHP.name}...")
    polygon = load_shelf_polygon("Nansen")
    print(f"  bounds (EPSG:3031): {polygon.bounds}")
    print(f"  area: {polygon.area / 1e6:.1f} km^2")

    print("\nSearching REMA strip index for 2022-01-01 to 2024-01-01...")
    strips = list_rema_strips(
        polygon=polygon,
        strip_index_fpath=STRIP_INDEX,
        date_range=("2022-01-01", "2024-01-01"),
    )

    if strips is None or len(strips) == 0:
        print("No strips found.")
        return

    print(f"\nFound {len(strips)} strip(s). First five:")
    cols_to_show = [c for c in ("dem_id", "pdt_time1", "pdt_time2", "sensor1", "sensor2") if c in strips.columns]
    print(strips[cols_to_show].head().to_string(index=False))

    print("\nColumns available in result dataframe:")
    print("  " + ", ".join(strips.columns.tolist()))


if __name__ == "__main__":
    main()
