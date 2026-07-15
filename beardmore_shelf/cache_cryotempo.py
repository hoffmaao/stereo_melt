"""Pre-cache ESA CryoTEMPO Land Ice (TDP_LI) control points for the
Beardmore-shelf stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_cryotempo`, mirroring
``pig.cache_cryotempo``. CryoTEMPO is the CS2 control base we adopt instead
of re-retracking raw L2 (validated 0.82 m MAD vs IS2 at PIG). The TDP_LI
prefetch reuses the raw-L2 SARIn HDRs already on disk
(``<MAIN_DIR>/data/CS2/hdrs``, complete for 2019-01..2024-03 from the 2026-04
Beardmore CS2 pull) as a spatial index. Run:

    python -m beardmore_shelf.cache_cryotempo                 # all in-window strips
    python -m beardmore_shelf.cache_cryotempo --overwrite     # refresh every entry
    python -m beardmore_shelf.cache_cryotempo --dem-id <id>   # one strip
    python -m beardmore_shelf.cache_cryotempo --limit 3       # smoke test
    python -m beardmore_shelf.cache_cryotempo --prefetch-only # just pull granules

Per-strip CryoTEMPO filtered caches land under the canonical per-basin ASP
root (``beardmore_shelf/data/ASP/cryotempo_data/``); suffixed align runs
read them from there. The shared TDP_LI granule store is
``<MAIN_DIR>/data/CS2/tempo_li/``.

Coverage caveat for this sector: the HDR index is SARIn-mode only. The TAM
front / grounding zone sits inside the SARIn ring, but AOI portions in
SAR-mode territory will show thin or zero per-strip counts — read the
per-strip summary this prints before judging the alignment recipe.
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

from stereo_melt.coregister import cache_cryotempo

from beardmore_shelf import config


def _load_aoi():
    # Mirror align_strips: select strips by the stack extent so everything
    # build_stack reprojects has CryoTEMPO control available.
    with fiona.open(config.BEARDMORE_SHELF_AOI_SHP) as src:
        return shape(next(iter(src))["geometry"])


def main() -> None:
    config.ensure_output_dirs()

    cache_dir = config.ASP_ROOT / "cryotempo_data"
    granule_dir = config.MAIN_DIR / "data" / "CS2" / "tempo_li"
    hdr_dir = config.MAIN_DIR / "data" / "CS2" / "hdrs"

    aoi = _load_aoi()
    rc = cache_cryotempo.run_cache_cryotempo_main(
        aoi_polygon=aoi,
        strips_dir=Path(config.STRIPS_DIR),
        cache_dir=cache_dir,
        granule_dir=granule_dir,
        hdr_dir=hdr_dir,
        bedmachine_path=config.BEDMACHINE_NC,
        velocity_path=config.MEASURES_PHASE_NC,
        start_time=config.START_TIME,
        end_time=config.END_TIME,
        time_window_days=config.CRYOTEMPO_TIME_WINDOW_DAYS,
        displacement_budget_m=config.CRYOTEMPO_DISPLACEMENT_BUDGET_M,
        fetch_window_env=("BEARDMORE_SHELF_FETCH_START", "BEARDMORE_SHELF_FETCH_END"),
        description=(
            "Cache CryoTEMPO Land Ice control points for the Beardmore-shelf stack."
        ),
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
