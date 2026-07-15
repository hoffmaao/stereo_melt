"""Pre-cache CryoSat-2 L2 SARIn POCA control points for the Beardmore stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_cs2`. See that module
for scope locks, server-layout notes, and the granule-then-perstrip
pipeline. Run:

    python -m beardmore.cache_cryosat2                       # all in-window strips
    python -m beardmore.cache_cryosat2 --overwrite            # refresh every cache entry
    python -m beardmore.cache_cryosat2 --dem-id <id>          # one strip
    python -m beardmore.cache_cryosat2 --limit 3              # smoke test first 3
    python -m beardmore.cache_cryosat2 --time-window 60       # ±N day temporal match

Beardmore's CS2 cache lives under the *shared* ``data/REMA/strips/ASP/``
root (alongside the IS2 cache from when the strips were aligned). When
Beardmore is migrated to a per-basin ASP root (see
``feedback_per_basin_asp_root.md``), the cache_dir below will move with it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Point pyproj at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import fiona
from shapely.geometry import shape

from stereo_melt.coregister import cache_cs2

from beardmore import config


def _load_aoi():
    # Strip selection uses the wider analysis-grid AOI (same one
    # fetch_strips and align_strips use). Per-strip CS2 query in
    # cache_cs2.cache_one_strip uses the strip's own bbox via
    # `_strip_bbox(strip_path)`, so this AOI only filters which
    # strips are considered, never the per-shot footprint.
    with fiona.open(config.BEARDMORE_STACK_AOI_SHP) as src:
        return shape(next(iter(src))["geometry"])


def main() -> None:
    config.ensure_output_dirs()

    # Beardmore predates the per-basin ASP migration, so its CS2 cache
    # currently lives under the shared REMA strips/ASP/ tree alongside
    # the IS2 cache from the strips' original alignment. Move with the
    # migration when it lands.
    cache_dir = Path(config.STRIPS_DIR) / "ASP" / "cs2_data"
    granule_dir = config.MAIN_DIR / "data" / "CS2" / "granules"
    hdr_dir = config.MAIN_DIR / "data" / "CS2" / "hdrs"

    aoi = _load_aoi()
    rc = cache_cs2.run_cache_cs2_main(
        aoi_polygon=aoi,
        strips_dir=Path(config.STRIPS_DIR),
        cache_dir=cache_dir,
        granule_dir=granule_dir,
        hdr_dir=hdr_dir,
        bedmachine_path=config.BEDMACHINE_NC,
        velocity_path=config.MEASURES_PHASE_NC,
        start_time=config.START_TIME,
        end_time=config.END_TIME,
        fetch_window_env=("BEARDMORE_FETCH_START", "BEARDMORE_FETCH_END"),
        description="Cache CS2 SARIn POCA control points for the Beardmore stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
