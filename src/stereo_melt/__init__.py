"""stereo_melt — basal melt rate beneath Antarctic ice shelves from repeat stereo DEMs."""

import os
import sys

# Work around a common conda-env quirk where pyproj finds the base conda's
# proj.db (which has stale DATABASE.LAYOUT.VERSION metadata and breaks
# EPSG code lookups) before the environment's own proj.db. If PROJ_DATA
# is unset and the active env ships a proj/ directory, point PROJ_DATA at
# the env-local proj.db so EPSG lookups succeed. No-op when PROJ_DATA is
# already set or when no env-local proj dir exists.
if "PROJ_DATA" not in os.environ and "PROJ_LIB" not in os.environ:
    _env_proj = os.path.join(sys.prefix, "share", "proj")
    if os.path.isfile(os.path.join(_env_proj, "proj.db")):
        os.environ["PROJ_DATA"] = _env_proj

__version__ = "0.1.0"
