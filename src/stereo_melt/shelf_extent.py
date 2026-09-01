# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Window-minimum floating-shelf extent.

Melt products integrated over a multi-year window are only meaningful on
ice that was present and afloat for the WHOLE window: a pixel the shelf
lost to calving mid-window (e.g. the Pine Island 2017-2020 tongue
collapse) swings from tens of meters of freeboard to ~0, and every
dh/dt-consuming solver reads that ice-to-ocean cliff as extreme melt.
The static BedMachine floating mask cannot see this — it is a single
snapshot.

:func:`min_shelf_extent` intersects three independent constraints:

1. **the static floating mask** (BedMachine ``mask == 3``) — keeps the
   grounding-line side of the domain honest;
2. **observed ice extents over the window** (Greene et al. 2022 annual
   coastlines via :func:`stereo_melt.io.greene.greene_min_extent_on_grid`)
   — covers calving from the window start through the archive end
   (2021.2);
3. **a per-epoch ocean test on the stack itself** — a pixel whose
   corrected surface height reads below ``freeboard_max`` at
   ``min_hits`` or more epochs was open water / mélange at some point in
   the window. This covers the post-Greene era at exactly the analysed
   epochs (and needs at least two hits so a single blunder strip cannot
   delete shelf).

The result is a drop-in replacement for the ``floating_mask`` argument
every solver already takes.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def min_shelf_extent(
    stack: xr.DataArray,
    floating_mask: xr.DataArray,
    *,
    greene_extent: xr.DataArray | None = None,
    freeboard_max: float = 5.0,
    min_hits: int = 2,
) -> xr.DataArray:
    r"""Boolean mask of ice present and afloat for the whole stack window.

    Parameters
    ----------
    stack : xarray.DataArray, dims ``(time, y, x)``
        The (tilt-corrected) DEM stack over the analysis window; heights
        are meters above the tidal/MDT-corrected sea surface, so they
        read as freeboard on the shelf.
    floating_mask : xarray.DataArray of bool, dims ``(y, x)``
        Static floating-ice mask (BedMachine ``mask == 3``) on the stack
        grid.
    greene_extent : xarray.DataArray of bool, optional
        Window-minimum observed ice extent from
        :func:`stereo_melt.io.greene.greene_min_extent_on_grid`. Omit to
        skip the coastline constraint (e.g. synthetic stacks).
    freeboard_max : float
        Heights below this (m) count as open water. Ice-shelf freeboard
        is >~15 m everywhere real; ocean/mélange reads ~0-3 m.
    min_hits : int
        Number of epochs a pixel must read as ocean before it is
        excluded. Two by default, so one blunder DEM cannot delete
        shelf.

    Returns
    -------
    xarray.DataArray of bool, dims ``(y, x)``
        True where every constraint holds. ``attrs`` record the
        parameters and how many pixels each constraint removed from the
        static floating mask.
    """
    if "time" not in stack.dims:
        raise ValueError("stack must have a time dimension")
    z = stack.transpose("time", "y", "x").values
    floating = np.asarray(
        floating_mask.transpose("y", "x").values, dtype=bool
    )

    with np.errstate(invalid="ignore"):
        ocean_hits = np.nansum(
            np.isfinite(z) & (z < float(freeboard_max)), axis=0
        )
    ever_ocean = ocean_hits >= int(min_hits)

    extent = floating & ~ever_ocean
    n_ocean = int((floating & ever_ocean).sum())
    n_greene = 0
    if greene_extent is not None:
        greene = np.asarray(
            greene_extent.transpose("y", "x").values, dtype=bool
        )
        n_greene = int((extent & ~greene).sum())
        extent &= greene

    out = xr.DataArray(
        extent,
        dims=("y", "x"),
        coords={"y": stack["y"].values, "x": stack["x"].values},
        name="min_extent_mask",
        attrs={
            "long_name": "window-minimum floating-shelf extent",
            "freeboard_max_m": float(freeboard_max),
            "min_hits": int(min_hits),
            "n_epochs": int(z.shape[0]),
            "n_floating": int(floating.sum()),
            "n_removed_ocean_test": n_ocean,
            "n_removed_greene": n_greene,
            "greene": (
                greene_extent.attrs.get("epochs_used", "yes")
                if greene_extent is not None else "not applied"
            ),
        },
    )
    return out
