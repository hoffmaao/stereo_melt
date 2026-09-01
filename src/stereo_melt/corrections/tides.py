# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Tidal correction via :mod:`pyTMD` (Sutterley).

Two modes:

* :func:`apply_tidal_correction` evaluates a tide model at one
  ``center_time`` across many ``(x, y)`` points -- the usual case for a
  single DEM strip captured at a single moment.
* :func:`apply_tidal_correction_drift` evaluates the model at
  per-point times -- needed when correcting a cloud of altimetry control
  points (e.g. ICESat-2 ATL06 within a ±10 day window of a strip), where
  each point has its own UTC timestamp.

Both routes are basin-agnostic: the caller passes ``model`` and
``model_directory``, so the same orchestrator works for Antarctic
(``CATS2008``), Arctic (``Arc5km2018``, ``AOTIM-5-2018``), or global
(``TPXO9-atlas-v5``, ``EOT20``, ``FES2014``) basins. No model name is
hardcoded.
"""

from __future__ import annotations

import numpy as np
from pyTMD.compute import tide_elevations

# pyTMD's `crs.from_input` calls `pyproj.CRS.from_user_input(EPSG)`. With an
# integer EPSG that path hits proj.db; this env's proj.db is incomplete (see
# memory `feedback_proj_data_env_fix.md`) so all three of pyTMD's fallbacks
# fail and the bare `raise pyproj.exceptions.CRSError` surfaces as a TypeError
# under modern pyproj. Passing a PROJ4 string lets pyproj parse it directly
# without touching the db.
_EPSG_3031_PROJ4 = (
    "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 "
    "+datum=WGS84 +units=m +no_defs +type=crs"
)


def apply_tidal_correction(
    xs,
    ys,
    center_time,
    model: str = "CATS2008",
    model_directory=None,
):
    r"""Return tidal elevations at ``(xs, ys)`` for a single ``center_time``.

    Use this when every input point shares the same evaluation time
    (e.g. a single DEM strip).

    Parameters
    ----------
    xs, ys : numpy.ndarray
        Pixel coordinates in EPSG:3031 (meters).
    center_time : str or numpy.datetime64
        Evaluation time (UTC). Anything ``numpy.datetime64`` accepts.
    model : str
        pyTMD model name. Defaults to ``"CATS2008"`` for Antarctic use;
        pass ``"Arc5km2018"``, ``"AOTIM-5-2018"``, ``"TPXO9-atlas-v5"``,
        ``"EOT20"``, ``"FES2014"``, etc. for non-Antarctic basins.
    model_directory : str or pathlib.Path, optional
        Directory holding the model files. Required when pyTMD's default
        environment-based discovery is not configured. The CATS2008
        directory shipped with this project is what
        ``config.TIDE_MODEL_DIR`` points at.

    Returns
    -------
    numpy.ndarray
        Tidal elevation (meters) at the input coordinates. Sign matches
        pyTMD's convention: a positive value means the sea surface is
        above the mean. To remove the tidal signal from a DEM, subtract
        this from the DEM elevation.
    """
    tide_dates = np.datetime64(center_time)
    delta_time = (tide_dates - np.datetime64("2000-01-01T00:00:00")) / np.timedelta64(1, "s")

    kwargs = {
        "MODEL": model,
        "EPOCH": (2000, 1, 1, 0, 0, 0),
        "EPSG": _EPSG_3031_PROJ4,
        "TYPE": "drift",
    }
    if model_directory is not None:
        kwargs["DIRECTORY"] = str(model_directory)

    # pyTMD wants flat 1-D arrays in 'drift' mode; the time array must
    # match the (xs, ys) length so we broadcast the single center_time.
    xs_flat = np.asarray(xs).ravel()
    ys_flat = np.asarray(ys).ravel()
    delta_arr = np.full(xs_flat.shape, float(delta_time))

    tide_h = tide_elevations(xs_flat, ys_flat, delta_arr, **kwargs)

    finite = np.isfinite(np.asarray(tide_h))
    if finite.any():
        vals = np.asarray(tide_h)[finite]
        print(
            f"   🌊 tide({model}) @ {center_time}: "
            f"min={vals.min():.3f} max={vals.max():.3f} m  "
            f"(over {finite.sum()}/{xs_flat.size} ocean pts)"
        )
    else:
        print(f"   🌊 tide({model}) @ {center_time}: all points outside model coverage")
    return np.asarray(tide_h).reshape(np.asarray(xs).shape)


def apply_tidal_correction_drift(
    xs,
    ys,
    times,
    model: str = "CATS2008",
    model_directory=None,
):
    r"""Return per-point tidal elevations for a cloud of points each with its own time.

    Use this for tide-correcting altimetry control (ICESat-2 ATL06,
    OIB ATM, CryoSat-2) that span a ±10 day window before they're
    handed to ``pc_align`` against a contemporaneous strip — every
    point is evaluated at its native UTC timestamp instead of a shared
    ``center_time``.

    Parameters
    ----------
    xs, ys : numpy.ndarray
        Point coordinates in EPSG:3031 (meters), shape ``(N,)``.
    times : array-like of datetime-like, shape ``(N,)``
        UTC timestamp per point. Anything ``pandas.to_datetime`` would
        accept, but a ``numpy.datetime64[ns]`` array is the cleanest
        input.
    model, model_directory : see :func:`apply_tidal_correction`.

    Returns
    -------
    numpy.ndarray, shape ``(N,)``
        Tidal elevation per point (meters). Subtract from the point
        elevations to remove the tidal signal.
    """
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()
    times_dt64 = np.asarray(times, dtype="datetime64[s]").ravel()
    if not (xs.shape == ys.shape == times_dt64.shape):
        raise ValueError(
            f"xs/ys/times shape mismatch: {xs.shape} / {ys.shape} / {times_dt64.shape}"
        )

    delta_time = (times_dt64 - np.datetime64("2000-01-01T00:00:00")) / np.timedelta64(1, "s")
    delta_time = delta_time.astype("float64")

    kwargs = {
        "MODEL": model,
        "EPOCH": (2000, 1, 1, 0, 0, 0),
        "EPSG": _EPSG_3031_PROJ4,
        "TYPE": "drift",
    }
    if model_directory is not None:
        kwargs["DIRECTORY"] = str(model_directory)

    tide_h = tide_elevations(xs, ys, delta_time, **kwargs)
    finite = np.isfinite(np.asarray(tide_h))
    if finite.any():
        vals = np.asarray(tide_h)[finite]
        print(
            f"   🌊 tide({model}) drift over {len(xs)} pts: "
            f"min={vals.min():.3f} max={vals.max():.3f} m  "
            f"(over {finite.sum()}/{xs.size} ocean pts)"
        )
    else:
        print(f"   🌊 tide({model}) drift: all {xs.size} points outside model coverage")
    return np.asarray(tide_h)
