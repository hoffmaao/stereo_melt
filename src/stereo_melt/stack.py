# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Build a repeat-DEM stack on a common EPSG:3031 target grid.

Consumes a list of corrected DEM GeoTIFFs and their epoch timestamps,
reprojects each onto a shared target grid, and returns a single
``xarray.DataArray`` with dims ``(time, y, x)``. The stack is the input
surface for the mass-budget stages in :mod:`stereo_melt.freeboard`,
:mod:`stereo_melt.kinematics`, and :mod:`stereo_melt.melt`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers the .rio accessor on xarray objects
import xarray as xr
from rasterio.enums import Resampling

__all__ = [
    "target_grid",
    "build_stack",
    "save_stack",
    "load_stack",
    "load_basin_stack",
]


def target_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    res: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return 1-D ``(x, y)`` coord arrays for a target EPSG:3031 grid.

    The :math:`x` coordinate ascends and :math:`y` descends (image /
    north-up), matching the default rioxarray orientation for EPSG:3031
    rasters.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        Extent of the target grid in meters (EPSG:3031).
    res : float
        Grid spacing in meters.

    Returns
    -------
    tuple of numpy.ndarray
        The ``(x, y)`` coordinate arrays.
    """
    x = np.arange(x_min, x_max, res, dtype="float64")
    y = np.arange(y_max, y_min, -res, dtype="float64")
    return x, y


def _open_and_reproject(
    path,
    tx: np.ndarray,
    ty: np.ndarray,
    resampling: Resampling = Resampling.bilinear,
    crs: str = "EPSG:3031",
    src_crs_override: str | None = None,
    matchtag_path=None,
    bitmask_path=None,
    matchtag_keep=(1,),
    bitmask_keep=(0, 2),
) -> xr.DataArray:
    r"""Open a DEM and reproject-match it onto the target ``(tx, ty)`` grid.

    ``src_crs_override`` lets the caller force the source CRS, which is
    useful when the raster ships with a generic ``LOCAL_CS`` tag (as
    ASP ``pc_align`` outputs do) that PROJ can't reproject from.

    If ``matchtag_path`` and/or ``bitmask_path`` are provided, the PGC
    quality filter (matchtag in ``matchtag_keep`` AND bitmask in
    ``bitmask_keep``, mirroring ``altimetryFit/pgc_2m_dem.py``) is
    applied on the target grid after reprojection. Companion rasters
    are reprojected with nearest-neighbour resampling so flag values
    are preserved.
    """
    dem = rioxarray.open_rasterio(path, masked=True)
    if "band" in dem.dims:
        dem = dem.squeeze("band", drop=True)
    if src_crs_override is not None:
        dem = dem.rio.write_crs(src_crs_override)
    tmpl = xr.DataArray(
        np.zeros((len(ty), len(tx)), dtype="float32"),
        coords={"y": ty, "x": tx},
        dims=("y", "x"),
    ).rio.write_crs(crs)
    dem_target = dem.rio.reproject_match(tmpl, resampling=resampling)

    keep = None
    if matchtag_path is not None:
        mt = rioxarray.open_rasterio(matchtag_path, masked=False)
        if "band" in mt.dims:
            mt = mt.squeeze("band", drop=True)
        if src_crs_override is not None:
            mt = mt.rio.write_crs(src_crs_override)
        mt_t = mt.rio.reproject_match(tmpl, resampling=Resampling.nearest)
        mt_keep = np.isin(mt_t.values, list(matchtag_keep))
        keep = mt_keep if keep is None else (keep & mt_keep)
    if bitmask_path is not None:
        bm = rioxarray.open_rasterio(bitmask_path, masked=False)
        if "band" in bm.dims:
            bm = bm.squeeze("band", drop=True)
        if src_crs_override is not None:
            bm = bm.rio.write_crs(src_crs_override)
        bm_t = bm.rio.reproject_match(tmpl, resampling=Resampling.nearest)
        bm_keep = np.isin(bm_t.values, list(bitmask_keep))
        keep = bm_keep if keep is None else (keep & bm_keep)
    if keep is not None:
        vals = dem_target.values.astype("float32")
        vals[~keep] = np.nan
        dem_target = dem_target.copy(data=vals)
    return dem_target


def build_stack(
    paths: Iterable[str | Path],
    times: Iterable,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    res: float,
    resampling: Resampling = Resampling.bilinear,
    crs: str = "EPSG:3031",
    src_crs_override: str | None = None,
    quality_files: list | None = None,
    matchtag_keep=(1,),
    bitmask_keep=(0, 2),
) -> xr.DataArray:
    r"""Return a ``(time, y, x)`` stack from corrected DEM GeoTIFFs.

    Each input DEM is reprojected onto the shared target grid defined by
    ``(x_min, x_max, y_min, y_max, res)`` and concatenated along a new
    ``time`` dimension sorted in ascending order.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        DEM GeoTIFFs to stack.
    times : iterable of datetime-like
        Epoch timestamp for each path.
    x_min, x_max, y_min, y_max, res : float
        Target-grid extent and resolution in meters.
    resampling : rasterio.enums.Resampling
        Resampling method for the reprojection. Defaults to bilinear.
    crs : str
        Target CRS. Defaults to EPSG:3031.
    src_crs_override : str, optional
        Force source CRS on every input raster before reprojecting.
        Useful when inputs ship with a generic ``LOCAL_CS`` tag (e.g.
        ASP ``pc_align`` outputs) that PROJ cannot reproject from.
    quality_files : list of (matchtag_path, bitmask_path) or None
        Optional per-strip quality companion files. If supplied, must
        have the same length as ``paths``; each entry is a tuple of
        ``(matchtag_path, bitmask_path)`` (either may be None to skip
        one filter). When provided, the PGC quality filter is applied
        on the target grid after reprojection. Mirrors
        ``altimetryFit/pgc_2m_dem.py``.
    matchtag_keep, bitmask_keep : tuple of int
        Whitelist values for the quality filter. Defaults: matchtag
        ``{1}`` (stereo matched), bitmask ``{0, 2}`` (clean + water,
        excluding edges and clouds).

    Returns
    -------
    xarray.DataArray
        Dims ``(time, y, x)`` with the rio CRS written.
    """
    paths = list(paths)
    times = list(times)
    if len(paths) != len(times):
        raise ValueError("paths and times must have the same length")
    if not paths:
        raise ValueError("at least one DEM path is required")
    if quality_files is not None and len(quality_files) != len(paths):
        raise ValueError("quality_files must match paths length")

    tx, ty = target_grid(x_min, x_max, y_min, y_max, res)
    n = len(paths)
    print(
        f"  build_stack: reprojecting {n} strips onto "
        f"{len(ty)}x{len(tx)} grid (res={res} m)...",
        flush=True,
    )
    t0 = time.time()
    print_every = max(1, n // 20)
    layers = []
    for i, p in enumerate(paths):
        mt_path, bm_path = (None, None)
        if quality_files is not None and quality_files[i] is not None:
            mt_path, bm_path = quality_files[i]
        layers.append(_open_and_reproject(
            p, tx, ty, resampling, crs, src_crs_override,
            matchtag_path=mt_path, bitmask_path=bm_path,
            matchtag_keep=matchtag_keep, bitmask_keep=bitmask_keep,
        ))
        if (i + 1) % print_every == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - (i + 1)) / max(rate, 1e-6)
            print(
                f"    reproject: {i + 1}/{n} strips  "
                f"elapsed={elapsed:.1f}s  rate={rate:.2f} strips/s  eta={eta:.0f}s",
                flush=True,
            )
    ts = pd.to_datetime(times)
    stack = xr.concat(layers, dim=pd.Index(ts, name="time"))
    stack = stack.sortby("time")
    stack.rio.write_crs(crs, inplace=True)
    return stack


def save_stack(
    stack: xr.DataArray,
    path,
    *,
    compress: bool = True,
    complevel: int = 4,
) -> None:
    r"""Persist a stack to NetCDF at ``path``.

    By default the payload variable is written with zlib compression and
    per-epoch chunking ``(1, ny, nx)``. Repeat-DEM stacks are
    NaN-dominated -- each epoch is a single strip footprint on the full
    target grid -- so zlib level 4 typically shrinks the file several-fold
    with no reader changes: NetCDF4 decompresses transparently, so
    :func:`load_stack` and :func:`load_basin_stack` are unaffected. Pass
    ``compress=False`` to fall back to the legacy uncompressed write.
    """
    to_write = stack.copy(deep=False)
    encoding = None
    if compress:
        # A stack loaded from a contiguous (uncompressed) file carries
        # ``contiguous=True`` in the payload var's read-time encoding,
        # which collides with ``chunksizes`` on write. Drop it (the
        # shallow copy keeps this off the caller's object). 1-D coords
        # round-trip fine as contiguous, so they are left untouched.
        to_write.encoding = {}
        name = (
            to_write.name
            if to_write.name is not None
            else "__xarray_dataarray_variable__"
        )
        chunksizes = tuple(
            1 if dim == "time" else size
            for dim, size in zip(to_write.dims, to_write.shape)
        )
        encoding = {
            name: {"zlib": True, "complevel": complevel, "chunksizes": chunksizes}
        }
    to_write.to_netcdf(path, encoding=encoding)


def load_stack(path) -> xr.DataArray:
    r"""Load a previously-saved stack from NetCDF at ``path``.

    Returns the single payload DataArray, excluding the ``spatial_ref``
    CRS grid-mapping variable that :func:`save_stack` writes alongside it.
    ``xr.open_dataarray`` cannot be used here precisely because that
    second variable is present; this mirrors the payload selection in
    :func:`load_basin_stack`.
    """
    ds = xr.open_dataset(path)
    payload = [v for v in ds.data_vars if v != "spatial_ref"]
    if len(payload) != 1:
        raise ValueError(f"Expected one stack variable in {path}, got {payload}")
    return ds[payload[0]]


def load_basin_stack(
    processed_dir: Path,
    basin_prefix: str,
    start_time: str,
    end_time: str,
    *,
    prefer_tilt_corrected: bool = True,
    bad_epochs: tuple[str, ...] = (),
    bad_strips: tuple[str, ...] = (),
) -> tuple[xr.DataArray, Path]:
    r"""Resolve and load a basin's DEM stack with ``BAD_EPOCHS`` applied.

    Replaces the per-driver loader logic that diverged across
    ``run_melt``, ``tilt_fit``, ``run_pseudospectral``, and
    ``run_stationary`` (three of which bypassed the bad-epoch filter).

    Search order, given ``basin_prefix = "beardmore_stack"``:
      1. ``{prefix}_tilt_corrected_{start}_{end}.nc`` (window-matched, tilt-corrected)
      2. ``{prefix}_{start}_{end}.nc`` (window-matched, raw)
      3. any ``{prefix}_tilt_corrected_*.nc`` (sorted)
      4. any other ``{prefix}_*.nc`` (sorted, raw only)

    With ``prefer_tilt_corrected=False`` the tilt-corrected candidates
    are skipped entirely (used by ``tilt_fit`` itself, since it is the
    stage that produces the tilt-corrected file).

    Parameters
    ----------
    processed_dir : pathlib.Path
        Directory containing the basin's stack NetCDF files.
    basin_prefix : str
        Filename stem before the time window (e.g. ``"beardmore_stack"``).
    start_time, end_time : str
        ISO date strings used in the window-matched filename.
    prefer_tilt_corrected : bool
        If True (default), prefer ``*_tilt_corrected_*.nc`` over the raw
        stack. If False, only consider raw stacks.
    bad_epochs : tuple of str
        Date strings for epochs to drop after loading. Matched against
        the stack's ``time`` coordinate at day resolution. Drops *every*
        slice sharing a flagged day -- use ``bad_strips`` to spare clean
        same-day siblings.
    bad_strips : tuple of str
        SETSM strip ids (``dem_id``) to drop after loading -- the
        strip-level analogue of ``bad_epochs`` that does not discard
        clean same-day siblings. Requires the stack to carry a ``dem_id``
        coordinate (written by ``build_stack`` since 2026-06-20); on
        older date-only stacks it warns and is skipped. Applied in
        addition to ``bad_epochs``.

    Returns
    -------
    (xarray.DataArray, pathlib.Path)
        The stack DataArray (single payload variable, ``spatial_ref``
        excluded) and the path of the file actually loaded.
    """
    processed_dir = Path(processed_dir)
    candidates: list[Path] = []

    if prefer_tilt_corrected:
        tc_match = processed_dir / f"{basin_prefix}_tilt_corrected_{start_time}_{end_time}.nc"
        if tc_match.exists():
            candidates.append(tc_match)
    raw_match = processed_dir / f"{basin_prefix}_{start_time}_{end_time}.nc"
    if raw_match.exists():
        candidates.append(raw_match)

    if not candidates:
        if prefer_tilt_corrected:
            candidates.extend(
                sorted(processed_dir.glob(f"{basin_prefix}_tilt_corrected_*.nc"))
            )
        candidates.extend(
            p
            for p in sorted(processed_dir.glob(f"{basin_prefix}_*.nc"))
            if "tilt_corrected" not in p.name
        )

    if not candidates:
        raise SystemExit(
            f"No {basin_prefix}_*.nc in {processed_dir}. "
            f"Run build_stack first."
        )

    chosen = candidates[0]
    is_tc = "tilt_corrected" in chosen.name
    if prefer_tilt_corrected and not is_tc:
        print(f"  ⚠️ no tilt-corrected stack found; using raw {chosen.name}")
    elif is_tc:
        print(f"  loading tilt-corrected stack: {chosen.name}")
    else:
        print(f"  loading raw stack: {chosen.name}")

    ds = xr.open_dataset(chosen)
    payload = [v for v in ds.data_vars if v != "spatial_ref"]
    if len(payload) != 1:
        raise ValueError(f"Expected one stack variable in {chosen}, got {payload}")
    stack = ds[payload[0]]

    if bad_epochs:
        bad_dt = pd.to_datetime(list(bad_epochs)).normalize()
        epoch_dt = pd.to_datetime(stack["time"].values).normalize()
        keep = ~np.isin(epoch_dt, bad_dt)
        n_dropped = int((~keep).sum())
        if n_dropped:
            dropped = [str(d)[:10] for d, k in zip(epoch_dt, keep) if not k]
            print(f"  dropping {n_dropped} bad epoch(s): {dropped}")
            if "dem_id" in stack.coords:
                print(
                    "  ↳ date-keyed drop on a strip-addressable stack removes "
                    "ALL slices on each flagged day; prefer bad_strips (per-DEM) "
                    "to spare clean same-day siblings."
                )
        stack = stack.isel(time=np.where(keep)[0])

    if bad_strips:
        bad_set = set(bad_strips)
        if "dem_id" in stack.coords:
            slice_ids = np.asarray(stack["dem_id"].values, dtype=str)
            keep = ~np.isin(slice_ids, list(bad_set))
            n_dropped = int((~keep).sum())
            if n_dropped:
                dropped = [s for s, k in zip(slice_ids, keep) if not k]
                print(f"  dropping {n_dropped} bad strip(s): {dropped}")
            stack = stack.isel(time=np.where(keep)[0])
            missing = bad_set - set(slice_ids.tolist())
            if missing:
                shown = sorted(missing)[:5]
                print(
                    f"  ⚠️ {len(missing)} BAD_STRIPS id(s) not present in stack "
                    f"(already absent): {shown}"
                    + (" ..." if len(missing) > 5 else "")
                )
        else:
            print(
                f"  ⚠️ bad_strips given ({len(bad_set)}) but stack carries no "
                "'dem_id' coord; rebuild with build_stack to enable strip-level "
                "dropping. Skipping strip filter."
            )

    # Honor the requested [start_time, end_time) window (half-open, matching
    # build_stack's filter) even when filename fallback loaded a wider stack,
    # e.g. an era-sliced solve on a full-record tilt-corrected file. Exact
    # window-matched loads already satisfy the filter, so this is a no-op
    # for them.
    t = pd.to_datetime(stack["time"].values)
    keep = (t >= pd.Timestamp(start_time)) & (t < pd.Timestamp(end_time))
    n_outside = int((~keep).sum())
    if n_outside:
        if not keep.any():
            raise SystemExit(
                f"Requested window [{start_time}, {end_time}) contains none "
                f"of the {t.size} epochs in {chosen.name}."
            )
        print(
            f"  windowing to [{start_time}, {end_time}): dropping "
            f"{n_outside} epoch(s) outside the window "
            f"({int(keep.sum())} remain)"
        )
        stack = stack.isel(time=np.where(keep)[0])

    return stack, chosen
