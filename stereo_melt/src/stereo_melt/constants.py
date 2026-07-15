# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Physical constants used throughout stereo_melt.

Densities follow the Shean 2019 / Chartrand 2024 convention for Antarctic
ice shelves. They are exposed as module-level floats so callers can
override on a per-call basis (e.g. temperature-dependent ice density).
"""

#: Seawater density, kg m\ :sup:`-3`.
rhow = 1027.0

#: Pure-ice density, kg m\ :sup:`-3`.
rhoi = 918.0
