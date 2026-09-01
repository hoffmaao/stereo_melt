# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""Import-time environment fixes shared by every basin driver.

Import this module FIRST in basin entry-point scripts (before anything
that pulls in pyproj)::

    import stereo_melt.envsetup  # noqa: F401

Currently it points ``PROJ_DATA`` / ``PROJ_LIB`` at the conda env's own
``share/proj`` when a ``proj.db`` exists there — the base conda install's
proj.db is corrupt (feedback_proj_data_env_fix) and pyproj must never
fall back to it. Idempotent; respects values already set explicitly.
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)
