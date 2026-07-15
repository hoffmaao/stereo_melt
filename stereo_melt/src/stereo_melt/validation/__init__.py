# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Validation harnesses for altimetric GCP sources used by ASP coregistration.

The first member is :mod:`cs2_vs_is2` — collocates CryoSat-2 SARIn POCA
returns against ICESat-2 ATL06 over the basin window, stratifies the
residuals, and decides whether CS2 is safe to use as a second GCP source.
The same shape works for any future single-beam source (LVIS, ATM, GLAS).
"""
