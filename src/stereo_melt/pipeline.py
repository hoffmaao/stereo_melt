# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Stack-level correction orchestrators (post-coregistration).

ASP ``pc_align`` is referenced against grounded IS2 ATL06 + rock-outcrop
GCPs (tide-free by construction), so pre-ASP tide / IBE corrections
provide no numerical benefit and are deliberately omitted -- DEM strips
go to ``pc_align`` raw. All geophysical corrections that operate on
stacked DEMs live here:

* :func:`apply_mdt_to_stack` removes the steady-state mean dynamic
  topography (DTU22) from floating-ice pixels. MDT is zeroed over
  grounded ice (BedMachine ``mask == 2``).
* :func:`apply_geoid_to_stack` removes the WGS84-ellipsoid →
  orthometric reference difference (Shean 2019 Eq. 1) at every pixel.

Both static-field corrections run **before the per-epoch tilt fit**
(tide -> IBE -> MDT -> geoid -> tilt fit). With static fields removed
first, the tilt LSQ's per-pixel intercept block carries only residual
elevation around the orthometric / MSL surface, so the
``Ez`` Tikhonov prior on per-epoch :math:`\alpha_z` is correctly
calibrated to per-strip coregistration drift rather than to the
absolute geoid offset.

Tide and IBE are applied at the 25 m analysis-grid stage (see
:mod:`stereo_melt.corrections.post_coreg`), gated by a 3 km
``uniform_filter``-feathered floating-ice mask.

Firn air content (FAC) is *not* a vertical bias correction at all; it
converts surface elevation into ice-equivalent thickness via the
hydrostatic relation. That conversion lives in
:mod:`stereo_melt.freeboard` and runs at melt-rate-inversion time.

The IBE step relies on a basin-level NetCDF cube of ERA5 surface
pressure produced by :func:`populate_climate_cache`; that pull is
explicit and one-shot per basin so CDS contact and queue waits are
visible, not buried inside per-strip processing.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401 - registers .rio accessor
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import shape

from .corrections.geoid import apply_geoid_correction_xr
from .io.mdt import load_dtu10_mdt


# ---------------------------------------------------------------------------
# Pre-ASP per-strip corrections: deliberately empty.
#
# pc_align is referenced against grounded IS2 ATL06 + rock-outcrop GCPs
# (tide-free by construction). Pre-ASP tide / IBE corrections provide no
# numerical benefit at that anchor. Tide and IBE are applied post-coreg
# at the 25 m analysis grid in :mod:`stereo_melt.corrections.post_coreg`.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pre-tilt static-field corrections (MDT, geoid)
# ---------------------------------------------------------------------------
#
# Order: MDT first, then geoid (commutative: both static and subtractive).
# ---------------------------------------------------------------------------

def apply_mdt_to_stack(
    stack,
    *,
    mdt_path,
    bedmachine_mask_path,
    grounded_mask_value: int = 2,
):
    r"""Subtract DTU22 MDT from floating-ice pixels of a stack.

    MDT (mean dynamic topography) is the steady-state sea-surface
    departure from the geoid. Removing it expresses floating-ice
    freeboard relative to true mean sea level rather than the geoid.

    MDT is zeroed over grounded ice (BedMachine ``mask == 2``)
    because the steady ocean-surface offset has no physical meaning
    there; without this mask the grounded-ice footprint would receive
    a spurious ~1 m vertical shift.

    DTU22 MDT only covers ``lat ≥ -79°S``. Basins poleward of that
    boundary (e.g. Beardmore at ~-84°S) cannot be MDT-corrected --
    skip this step entirely for those basins.

    Parameters
    ----------
    stack : xarray.DataArray
        Stack with dims ``(time, y, x)`` in EPSG:3031.
    mdt_path : str or pathlib.Path
        DTU10/DTU22 MDT in GRAVSOFT XYZ format.
    bedmachine_mask_path : str or pathlib.Path
        BedMachine NetCDF used to zero MDT over grounded ice.
    grounded_mask_value : int
        BedMachine v3 mask integer for grounded ice (default ``2``).

    Returns
    -------
    xarray.DataArray
        Stack with MDT subtracted on floating pixels and the
        ``corrections`` attr appended.
    """
    if "x" not in stack.coords or "y" not in stack.coords:
        raise ValueError("stack must have x/y coords (EPSG:3031).")
    if stack.rio.crs is None:
        stack = stack.rio.write_crs("EPSG:3031", inplace=False)

    print("\n🌊 Applying MDT to stack (floating-only)")

    mdt_grid = _interp_mdt_to_stack_grid(stack, mdt_path)
    grounded = _grounded_mask_for_stack(
        stack, bedmachine_mask_path, grounded_mask_value
    )
    mdt_grid = np.where(grounded, 0.0, mdt_grid)

    print(
        f"   📊 MDT range on stack grid: "
        f"min={np.nanmin(mdt_grid):+.3f} max={np.nanmax(mdt_grid):+.3f} m  "
        f"(zeroed over {int(grounded.sum())} grounded-ice pixels)"
    )

    corrected = stack - xr.DataArray(
        mdt_grid, dims=("y", "x"), coords={"y": stack["y"], "x": stack["x"]}
    )
    corrected.attrs.update(stack.attrs)
    prior = stack.attrs.get("corrections", "")
    corrected.attrs["corrections"] = (prior + "; mdt").strip("; ")
    return corrected


def apply_geoid_to_stack(stack, *, bedmachine_path):
    r"""Apply geoid undulation to a stack (Shean 2019 Eq. 1).

    Removes the WGS84-ellipsoid → orthometric reference difference at
    every pixel. The geoid is a static spatial field, so this step
    runs before the per-epoch tilt LSQ to leave :math:`\alpha_z` to absorb
    only per-strip coregistration drift.

    Parameters
    ----------
    stack : xarray.DataArray
        Stack with dims ``(time, y, x)`` in EPSG:3031.
    bedmachine_path : str or pathlib.Path
        BedMachine NetCDF providing the geoid undulation field.

    Returns
    -------
    xarray.DataArray
        Geoid-corrected stack with the ``corrections`` attr appended.
    """
    if "x" not in stack.coords or "y" not in stack.coords:
        raise ValueError("stack must have x/y coords (EPSG:3031).")
    if stack.rio.crs is None:
        stack = stack.rio.write_crs("EPSG:3031", inplace=False)

    print("\n🌍 Applying geoid to stack")
    corrected = apply_geoid_correction_xr(stack, str(bedmachine_path))
    corrected.attrs.update(stack.attrs)
    prior = stack.attrs.get("corrections", "")
    corrected.attrs["corrections"] = (prior + "; geoid").strip("; ")
    return corrected


def _interp_mdt_to_stack_grid(stack: xr.DataArray, mdt_path) -> np.ndarray:
    """Bilinear-interpolate MDT (lat/lon grid) onto the stack's (y, x) grid."""
    mdt_data = load_dtu10_mdt(str(mdt_path))
    lats = np.asarray(mdt_data["lat"], dtype=np.float64)
    lons = np.asarray(mdt_data["lon"], dtype=np.float64)
    field = np.asarray(mdt_data["mdt"], dtype=np.float64)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        field = field[::-1, :]

    interp = RegularGridInterpolator(
        (lats, lons), field, bounds_error=False, fill_value=np.nan
    )

    xs1d = stack["x"].values
    ys1d = stack["y"].values
    xs2d, ys2d = np.meshgrid(xs1d, ys1d)
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    target_lons, target_lats = transformer.transform(xs2d, ys2d)
    if lons.max() > 180.0:
        target_lons = np.where(target_lons < 0.0, target_lons + 360.0, target_lons)
    out = interp(np.column_stack([target_lats.ravel(), target_lons.ravel()]))
    out = out.reshape(target_lats.shape)
    n_missing = int(np.isnan(out).sum())
    if n_missing == out.size:
        raise ValueError(
            "MDT grid does not cover any of the stack pixels "
            f"(stack lat range {target_lats.min():.2f}..{target_lats.max():.2f}, "
            f"MDT lat range {lats.min():.2f}..{lats.max():.2f}). "
            "DTU22 stops at -79°S; pass mdt_path=None to skip MDT for "
            "basins poleward of that boundary."
        )
    if n_missing:
        frac = n_missing / out.size
        warnings.warn(
            f"MDT coverage is incomplete on stack grid: {n_missing}/{out.size} "
            f"pixels ({100 * frac:.1f}%) outside DTU22. Those cells will "
            "remain NaN in the final stack.",
            stacklevel=2,
        )
    return out


def _grounded_mask_for_stack(
    stack: xr.DataArray, bedmachine_path, grounded_mask_value: int
) -> np.ndarray:
    """Return a (y, x) bool mask of grounded-ice pixels for the stack grid."""
    bm = xr.open_dataset(str(bedmachine_path))
    mask = bm["mask"]
    bm_x = mask["x"].values
    bm_y = mask["y"].values
    if bm_y[0] > bm_y[-1]:
        bm_y = bm_y[::-1]
        mask_arr = mask.values[::-1, :]
    else:
        mask_arr = mask.values
    interp = RegularGridInterpolator(
        (bm_y, bm_x), mask_arr.astype(np.float64),
        bounds_error=False, fill_value=0, method="nearest",
    )
    xs2d, ys2d = np.meshgrid(stack["x"].values, stack["y"].values)
    nearest = interp(np.column_stack([ys2d.ravel(), xs2d.ravel()]))
    bm.close()
    return (nearest.reshape(xs2d.shape).astype(int) == grounded_mask_value)


# ---------------------------------------------------------------------------
# One-shot ERA5 cube population (called by the basin's cache_climate driver)
# ---------------------------------------------------------------------------

def populate_climate_cache(config_module, *, cadence_hours: int = 1, overwrite: bool = False):
    r"""Pull ERA5 surface pressure for a basin via the ARCO-Zarr point endpoint.

    Convenience entry point for the basin-level ``cache_climate``
    driver. Reads ``START_TIME``, ``END_TIME``, the AOI shapefile, and
    ``ERA5_CACHE_NC`` from the supplied config module and forwards to
    :func:`stereo_melt.corrections.ibe.bulk_fetch_era5_pressure_timeseries`
    using the AOI centroid as the query point.

    Why the centroid: within-basin spatial variation of ERA5 surface
    pressure is ~0.1-0.3 hPa across 50 km = 1-3 mm of IBE response,
    well below the DEM noise floor. The CDS-Beta ``timeseries`` endpoint
    is purpose-built for this access pattern and returns sub-minute,
    avoiding the queue-bound bbox-gridded path.

    Parameters
    ----------
    config_module : module
        A basin config module exposing ``START_TIME``, ``END_TIME``,
        ``ERA5_CACHE_NC``, and an AOI attribute (one of
        ``BEARDMORE_AOI_SHP``, ``NANSEN_AOI_SHP``, or any
        ``*_AOI_SHP`` symbol).
    cadence_hours : int
        Reserved for backward compatibility; the ARCO endpoint is
        natively hourly.
    overwrite : bool
        If True, refresh the cache even if it already exists.

    Returns
    -------
    pathlib.Path
        Path to the written NetCDF cube (1D-time, point timeseries).
    """
    from .corrections.ibe import bulk_fetch_era5_pressure_timeseries

    aoi_path = _find_aoi_path(config_module)
    lat, lon = _aoi_centroid_latlon(aoi_path)
    print(
        f"🌐 Climate cache population for {getattr(config_module, 'SHELF', '?')}:"
        f"  window {config_module.START_TIME} .. {config_module.END_TIME}\n"
        f"   AOI centroid: lat={lat:.4f}, lon={lon:.4f}"
    )
    return bulk_fetch_era5_pressure_timeseries(
        start_time=config_module.START_TIME,
        end_time=config_module.END_TIME,
        latitude=lat,
        longitude=lon,
        output_path=config_module.ERA5_CACHE_NC,
        overwrite=overwrite,
    )


def _find_aoi_path(config_module) -> Path:
    """Locate a ``*_AOI_SHP`` attribute on the basin config module."""
    candidates = [
        name for name in dir(config_module)
        if name.endswith("_AOI_SHP") and not name.startswith("_")
    ]
    if not candidates:
        raise AttributeError(
            f"{config_module.__name__} has no *_AOI_SHP attribute; "
            "expected e.g. BEARDMORE_AOI_SHP or NANSEN_AOI_SHP."
        )
    if len(candidates) > 1:
        # Prefer the narrow per-basin AOI over any extended/stack variant.
        narrow = [c for c in candidates if "STACK" not in c and "EXTENT" not in c]
        if narrow:
            candidates = narrow
    return Path(getattr(config_module, candidates[0]))


def _aoi_bbox_latlon(aoi_path: Path, pad_deg: float = 1.0):
    """Return ``[north, west, south, east]`` for the AOI shapefile."""
    with fiona.open(aoi_path) as src:
        geom = shape(next(iter(src))["geometry"])
    minx, miny, maxx, maxy = geom.bounds
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    corners_x = [minx, minx, maxx, maxx]
    corners_y = [miny, maxy, miny, maxy]
    lons, lats = transformer.transform(corners_x, corners_y)
    north = float(np.max(lats)) + pad_deg
    south = float(np.min(lats)) - pad_deg
    west = float(np.min(lons)) - pad_deg
    east = float(np.max(lons)) + pad_deg
    # ERA5 expects N >= S and W <= E (no antimeridian handling required for
    # Antarctic AOIs that don't straddle 180°; basins that do will need a
    # pre-processing split).
    return [north, west, south, east]


def _aoi_centroid_latlon(aoi_path: Path):
    """Return ``(lat, lon)`` of the AOI centroid in EPSG:4326."""
    with fiona.open(aoi_path) as src:
        geom = shape(next(iter(src))["geometry"])
    cx, cy = geom.centroid.x, geom.centroid.y
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx, cy)
    return float(lat), float(lon)
