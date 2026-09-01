"""Pre-cache ESA CryoTEMPO Land Ice (TDP_LI) control points for the PIG stack.

Thin wrapper over :mod:`stereo_melt.coregister.cache_cryotempo`. CryoTEMPO
is the CS2 control base we adopt instead of re-retracking raw L2 (see that
module + ``stereo_melt.coregister.cryotempo``). The TDP_LI prefetch reuses
the raw-L2 SARIn HDRs already on disk (``<MAIN_DIR>/data/CS2/hdrs``) as a
spatial index — run ``pig.cache_cryosat2 --prefetch-only`` first if any
window-month lacks HDRs. Run:

    python -m pig.cache_cryotempo                       # all in-window strips
    python -m pig.cache_cryotempo --overwrite           # refresh every cache entry
    python -m pig.cache_cryotempo --dem-id <id>         # one strip
    python -m pig.cache_cryotempo --limit 3             # smoke test first 3
    python -m pig.cache_cryotempo --prefetch-only       # just pull granules

Per-strip CryoTEMPO filtered caches land under the per-basin ASP root
(``pig/data/ASP/cryotempo_data/``), parallel to ``cs2_data/`` so the raw-L2
control is preserved for A/B. The shared TDP_LI granule store is
``<MAIN_DIR>/data/CS2/tempo_li/``.
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

from pig import config


def _load_aoi():
    # Mirror cache_cryosat2 / align_strips: select strips by the wider stack
    # extent so the band build_stack reprojects has CryoTEMPO control.
    aoi_path = config.PIG_AOI_SHP
    with fiona.open(aoi_path) as src:
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
        fetch_window_env=("PIG_FETCH_START", "PIG_FETCH_END"),
        description="Cache CryoTEMPO Land Ice control points for the Pine Island stack.",
    )
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
