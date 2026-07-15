# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Inverse Barometer Effect correction.

Removes the oceanographic IBE bias from elevations over floating ice,
where a 1 hPa decrease in surface pressure raises the sea surface --
and the ice riding on it -- by 1 cm. The sign convention follows
Padman et al. 2003: the IBE-induced surface anomaly relative to
a 1013.25 hPa reference is

.. math::

    h_\mathrm{IBE} = -(P - P_\mathrm{ref}) / (\rho_w \, g)

so the corrected elevation is ``h_corrected = h_measured - h_IBE``,
i.e. ``h_measured + (P - P_ref) / (\rho_w \, g)``. Numerically with
``\rho_w g \approx 100`` Pa/cm, this is one centimeter per hPa.

Two-step workflow:

1. :func:`bulk_fetch_era5_pressure_window` is called *once per basin*
   from the explicit ``cache_climate`` driver. It pulls the entire
   basin time/space window from the Copernicus CDS into a NetCDF cube
   on disk.
2. :func:`apply_ibe_correction_from_cache` and
   :func:`apply_ibe_correction_drift_from_cache` mirror the
   tide-correction module's two modes -- single ``center_time`` for a
   full DEM strip, per-point times for an altimetry control cloud --
   and interpolate from the cube on disk. No CDS contact at strip- or
   stack-time.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import cdsapi
import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

# 1 hPa = 1 cm of IBE response. Stored as meters per hPa so the unit
# arithmetic is explicit downstream (the old per-strip implementation
# silently subtracted a centimeter-scale correction from a meter-scale
# DEM, which is 100x too small).
_METERS_PER_HPA = 0.01
_REFERENCE_PRESSURE_HPA = 1013.25


# ---------------------------------------------------------------------------
# Bulk CDS pull (run once per basin via the cache_climate driver)
# ---------------------------------------------------------------------------

def bulk_fetch_era5_pressure_window(
    start_time,
    end_time,
    region,
    output_path,
    cadence_hours: int = 1,
    overwrite: bool = False,
):
    r"""Pull ERA5 surface pressure for an entire basin window in one CDS request.

    Writes a NetCDF cube of shape ``(time, latitude, longitude)`` with
    surface pressure in hPa. The cube also carries auxiliary
    ``(x, y)`` coordinates in EPSG:3031 so downstream code can
    interpolate without re-projecting.

    Intended to be called *once* per basin, from an explicit driver
    (e.g. ``python -m beardmore.cache_climate``) so that CDS contact
    is visible and queue waits don't hide inside per-strip processing.

    Parameters
    ----------
    start_time, end_time : str or datetime-like
        Inclusive bounds on the time axis. Strings are parsed with
        ``numpy.datetime64``.
    region : sequence of float
        ERA5 query window ``[north, west, south, east]`` in degrees.
    output_path : str or pathlib.Path
        Destination NetCDF path. The basin's ``ERA5_CACHE_NC`` is the
        canonical home -- the filename encodes the window so changing
        ``START_TIME``/``END_TIME`` produces a different cache file
        and never silently uses a stale one.
    cadence_hours : int
        Hourly stride for the time axis. ``1`` (default) is hourly; use
        ``3``/``6`` to shrink the cube if the window is large or the
        basin is high-latitude with wide longitude span.
    overwrite : bool
        If False (default), skip the CDS request when ``output_path``
        already exists. Set True to force a refresh.

    Returns
    -------
    pathlib.Path
        ``output_path`` (for chaining).
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"⏭  ERA5 cache already at {output_path}; skipping CDS pull.")
        return output_path

    start = np.datetime64(start_time)
    end = np.datetime64(end_time)
    if end <= start:
        raise ValueError(f"end_time {end_time!r} must be after start_time {start_time!r}")
    if cadence_hours < 1 or 24 % cadence_hours:
        raise ValueError(
            f"cadence_hours must divide 24 evenly; got {cadence_hours}"
        )

    years = list(_years_between(start, end))
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    times = [f"{h:02d}:00" for h in range(0, 24, cadence_hours)]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"📡 Bulk ERA5 surface_pressure pull (chunked by year):\n"
        f"   window  : {start_time} .. {end_time}\n"
        f"   region  : N={region[0]}, W={region[1]}, S={region[2]}, E={region[3]}\n"
        f"   cadence : {cadence_hours}h ({len(times)} hours/day)\n"
        f"   years   : {years}\n"
        f"   target  : {output_path}"
    )

    # CDS-Beta enforces a per-request cost limit, so split into one
    # request per year and concatenate. Each year-cube lives next to
    # the final cube as a temp file until concat succeeds.
    client = cdsapi.Client()
    year_paths = []
    try:
        for y in years:
            ypath = output_path.with_name(f"{output_path.stem}.year{y}.nc")
            year_paths.append(ypath)
            if ypath.exists():
                print(f"   ⏭  year {y} already cached at {ypath.name}")
                continue
            print(f"   ⏳ requesting year {y} → {ypath.name}")
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    # MSL (mean sea level pressure), not surface pressure.
                    # ERA5 'sp' over Antarctica reflects the model topography
                    # (an 0.25° cell often averages mountains and ice shelf,
                    # so SP runs ~100 hPa below 1013 hPa reference and IBE
                    # blows up to +1 m). 'msl' is corrected to sea level and
                    # is the right input for the floating-ice IBE response.
                    "variable": "mean_sea_level_pressure",
                    "product_type": "reanalysis",
                    "year": [str(y)],
                    "month": months,
                    "day": days,
                    "time": times,
                    "area": list(region),
                    "format": "netcdf",
                },
                str(ypath),
            )

        print(f"   🔗 concatenating {len(year_paths)} year-cubes → {output_path}")
        # Per-year files arrive with either 'time' or 'valid_time' coord
        # depending on CDS-Beta version; xarray.open_mfdataset handles
        # alignment when a consistent dim name exists, so normalise here.
        per_year = []
        for yp in year_paths:
            ds = xr.open_dataset(yp).load()
            if "valid_time" in ds.coords and "time" not in ds.coords:
                ds = ds.rename({"valid_time": "time"})
            per_year.append(ds)
        merged = xr.concat(per_year, dim="time").sortby("time")
        merged.to_netcdf(output_path, mode="w")
        for ds in per_year:
            ds.close()
    finally:
        # Keep year files; they are useful when extending the window
        # later (next basin or year). Delete only on explicit overwrite.
        if overwrite:
            for yp in year_paths:
                if yp.exists():
                    yp.unlink()

    print(f"📥 ERA5 cube written to {output_path}")

    # Trim to the exact window and tag with EPSG:3031 (x, y) coords.
    _postprocess_pressure_cube(output_path, start, end)
    return output_path


def _years_between(start: np.datetime64, end: np.datetime64) -> Sequence[int]:
    y0 = int(str(start)[:4])
    y1 = int(str(end)[:4])
    return list(range(y0, y1 + 1))


# ---------------------------------------------------------------------------
# ARCO-Zarr point-timeseries pull (preferred over gridded for IBE)
# ---------------------------------------------------------------------------

def bulk_fetch_era5_pressure_timeseries(
    start_time,
    end_time,
    latitude: float,
    longitude: float,
    output_path,
    overwrite: bool = False,
):
    r"""Pull ERA5 surface pressure as a single-point time series via ARCO-Zarr.

    Hits the CDS-Beta ``reanalysis-era5-single-levels-timeseries`` endpoint
    backed by the Analysis Ready Cloud Optimised (Zarr) archive. The
    endpoint is purpose-built for this access pattern -- one variable,
    one location, long time window -- and bypasses the bbox-gridded
    queue used by :func:`bulk_fetch_era5_pressure_window` (which trips
    CDS-Beta cost limits for multi-year windows).

    Within-basin spatial variation of surface pressure is ~0.1-0.3 hPa
    over 50 km = 1-3 mm of IBE response, which is well below the DEM
    noise floor. Querying at the basin centroid is therefore physically
    sufficient. Use the AOI centroid (in EPSG:4326) for ``latitude``
    and ``longitude``.

    Parameters
    ----------
    start_time, end_time : str or datetime-like
        Inclusive bounds. ``start_time``/``end_time`` should be the
        same calendar bounds you'd give the gridded fetcher.
    latitude, longitude : float
        Query location in EPSG:4326. ERA5 snaps to the nearest 0.25°
        grid point.
    output_path : str or pathlib.Path
        Destination NetCDF (the ARCO endpoint ships a zip; we unpack
        and rewrite at this path).
    overwrite : bool
        If False (default), skip when ``output_path`` already exists.

    Returns
    -------
    pathlib.Path
        ``output_path``.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"⏭  ERA5 point cache already at {output_path}; skipping.")
        return output_path

    start = np.datetime64(start_time)
    end = np.datetime64(end_time)
    if end <= start:
        raise ValueError(f"end_time {end_time!r} must be after start_time {start_time!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The ARCO endpoint takes a single date range string, not a
    # year/month/day cross product, so the request payload is tiny
    # regardless of window length.
    date_range = f"{str(start)[:10]}/{str(end)[:10]}"

    print(
        f"📡 ERA5 point timeseries pull (ARCO-Zarr):\n"
        f"   window  : {date_range}\n"
        f"   point   : lat={latitude:.4f}, lon={longitude:.4f}\n"
        f"   target  : {output_path}"
    )

    import tempfile, zipfile, shutil
    client = cdsapi.Client()
    with tempfile.TemporaryDirectory() as tmpd:
        zip_path = Path(tmpd) / "arco.zip"
        client.retrieve(
            "reanalysis-era5-single-levels-timeseries",
            {
                # MSL (mean sea level pressure), not surface pressure.
                # ERA5 'sp' at the basin centroid reflects the model
                # topography (an 0.25° cell averages mountains + shelf in
                # Antarctica, so SP runs ~100 hPa below 1013 hPa reference
                # and IBE blows up to +1 m). 'msl' is corrected to sea
                # level and is the right input for the floating-ice IBE
                # response. Keep the variable name 'msl' on disk; the
                # post-process + interpolator have a rename hook to map
                # it onto the legacy 'surface_pressure' internal label
                # so consumers don't need to change.
                "variable": ["mean_sea_level_pressure"],
                "location": {"latitude": float(latitude), "longitude": float(longitude)},
                "date": [date_range],
                "data_format": "netcdf",
            },
            str(zip_path),
        )
        # The endpoint ships a zip with one .nc inside.
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if n.endswith(".nc")]
            if not members:
                raise RuntimeError(f"ARCO archive {zip_path} contained no .nc")
            zf.extract(members[0], tmpd)
            shutil.move(str(Path(tmpd) / members[0]), output_path)

    print(f"📥 ERA5 point cube written to {output_path}")
    return output_path


def _postprocess_pressure_cube(path: Path, start: np.datetime64, end: np.datetime64):
    """Slice cube to ``[start, end]`` and attach EPSG:3031 (x, y) coords."""
    with xr.open_dataset(path) as ds:
        ds = ds.load()

    # ERA5 NetCDF historically used 'time' but newer downloads ship 'valid_time'.
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"
    ds = ds.sel({time_dim: slice(start, end)})

    # CDS 'mean_sea_level_pressure' ships as 'msl'; legacy 'surface_pressure'
    # came back as 'sp'. Both get normalized to the internal name
    # 'surface_pressure' (a misnomer for MSL data, kept for backward
    # compatibility). New caches are MSL; older caches were SP.
    if "msl" in ds and "surface_pressure" not in ds:
        ds = ds.rename({"msl": "surface_pressure"})
    elif "sp" in ds and "surface_pressure" not in ds:
        ds = ds.rename({"sp": "surface_pressure"})

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lons2d, lats2d = np.meshgrid(ds[lon_name].values, ds[lat_name].values)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    xs2d, ys2d = transformer.transform(lons2d, lats2d)
    ds = ds.assign_coords(
        {
            "x": ((lat_name, lon_name), xs2d),
            "y": ((lat_name, lon_name), ys2d),
        }
    )

    ds.to_netcdf(path, mode="w")


# ---------------------------------------------------------------------------
# Cache-backed application (no CDS contact at run time)
# ---------------------------------------------------------------------------

def apply_ibe_correction_from_cache(
    z,
    xs,
    ys,
    center_time,
    cache_path,
    pressure_var: str = "surface_pressure",
    bedmachine_path=None,
):
    r"""Apply IBE to a DEM (single ``center_time``) using a precomputed cube.

    Use this for a DEM strip captured at one moment in time -- the
    natural pre-ASP correction step. For a cloud of altimetry points
    each with its own UTC timestamp, use
    :func:`apply_ibe_correction_drift_from_cache` (mirrors the
    tides module's drift mode).

    Parameters
    ----------
    z : numpy.ndarray
        Input elevations (meters). Shape arbitrary; ``xs`` and ``ys``
        must broadcast against it.
    xs, ys : numpy.ndarray
        EPSG:3031 coordinates matching ``z``.
    center_time : str or numpy.datetime64
        UTC evaluation time.
    cache_path : str or pathlib.Path
        NetCDF cube written by :func:`bulk_fetch_era5_pressure_window`.
    pressure_var : str
        Variable name in the cube (default ``"surface_pressure"``).

    Returns
    -------
    numpy.ndarray
        IBE-corrected elevations on the same shape as ``z``.
    """
    interp = _load_pressure_interpolator(cache_path, pressure_var)
    t_axis = interp._t_axis  # type: ignore[attr-defined]
    lat_axis = interp._lat_axis  # type: ignore[attr-defined]
    lon_axis = interp._lon_axis  # type: ignore[attr-defined]

    t = np.datetime64(center_time)
    t_secs = _to_seconds_since_epoch(t)

    lats, lons = _epsg3031_to_latlon(np.asarray(xs).ravel(), np.asarray(ys).ravel())
    pts = np.column_stack(
        [
            np.full(lats.shape, t_secs, dtype=np.float64),
            lats,
            lons,
        ]
    )
    pressure_hpa = interp(pts).reshape(np.asarray(xs).shape)
    h_ibe = -(pressure_hpa - _REFERENCE_PRESSURE_HPA) * _METERS_PER_HPA

    if bedmachine_path is not None:
        from ..io.bedmachine import interp_bedmachine_mask_at, MASK_FLOATING_ICE
        floating = interp_bedmachine_mask_at(
            bedmachine_path, xs, ys, mask_value=MASK_FLOATING_ICE,
        )
        h_ibe = np.where(floating, h_ibe, 0.0)
    else:
        floating = None

    finite = np.isfinite(h_ibe)
    if finite.any():
        v = h_ibe[finite]
        n_floating = int(floating.sum()) if floating is not None else h_ibe.size
        print(
            f"   🌬  IBE @ {center_time}: min={v.min():+.4f} max={v.max():+.4f} m  "
            f"({n_floating}/{h_ibe.size} floating-ice pts; rest zeroed)"
        )
    else:
        print(f"   🌬  IBE @ {center_time}: all points outside cube extent")

    return z - h_ibe


def apply_ibe_correction_drift_from_cache(
    z,
    xs,
    ys,
    times,
    cache_path,
    pressure_var: str = "surface_pressure",
    bedmachine_path=None,
):
    r"""Apply IBE per-point with each point at its own UTC timestamp.

    Use this for tide-style 'drift' corrections of an altimetry control
    cloud (ICESat-2 ATL06, CryoSat-2, OIB ATM) where every point has its
    own observation time. Same cube, different access pattern from
    :func:`apply_ibe_correction_from_cache`.

    Note that current IS2 control in this project is restricted to
    grounded ice, where IBE does not apply (rigid bed, no pressure
    response). This function is provided for the planned extension to
    floating-ice control points and CS-2 / OIB altimetry, where the
    sea surface (and the ice on it) bobs with synoptic pressure.

    Parameters
    ----------
    z : numpy.ndarray, shape ``(N,)``
        Per-point elevations.
    xs, ys : numpy.ndarray, shape ``(N,)``
        EPSG:3031 coordinates per point.
    times : array-like of datetime-like, shape ``(N,)``
        Per-point UTC timestamp.
    cache_path : str or pathlib.Path
        NetCDF cube from :func:`bulk_fetch_era5_pressure_window`.
    pressure_var : str
        Variable name in the cube.

    Returns
    -------
    numpy.ndarray, shape ``(N,)``
        IBE-corrected elevations.
    """
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()
    z = np.asarray(z).ravel()
    times_dt64 = np.asarray(times, dtype="datetime64[s]").ravel()
    if not (xs.shape == ys.shape == z.shape == times_dt64.shape):
        raise ValueError(
            f"xs/ys/z/times shape mismatch: {xs.shape} / {ys.shape} / "
            f"{z.shape} / {times_dt64.shape}"
        )

    interp = _load_pressure_interpolator(cache_path, pressure_var)
    t_secs = (times_dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    t_secs = t_secs.astype("float64")

    lats, lons = _epsg3031_to_latlon(xs, ys)
    pts = np.column_stack([t_secs, lats, lons])
    pressure_hpa = interp(pts)
    h_ibe = -(pressure_hpa - _REFERENCE_PRESSURE_HPA) * _METERS_PER_HPA

    if bedmachine_path is not None:
        from ..io.bedmachine import interp_bedmachine_mask_at, MASK_FLOATING_ICE
        floating = interp_bedmachine_mask_at(
            bedmachine_path, xs, ys, mask_value=MASK_FLOATING_ICE,
        )
        h_ibe = np.where(floating, h_ibe, 0.0)
        n_floating = int(floating.sum())
    else:
        n_floating = len(z)

    finite = np.isfinite(h_ibe)
    if finite.any():
        v = h_ibe[finite]
        print(
            f"   🌬  IBE drift over {len(z)} pts: min={v.min():+.4f} max={v.max():+.4f} m  "
            f"({n_floating}/{len(z)} floating; {int(finite.sum())}/{len(z)} in cube extent)"
        )
    else:
        print(f"   🌬  IBE drift: all {len(z)} points outside cube extent")

    return z - h_ibe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pressure_interpolator(cache_path, pressure_var: str):
    """Return a callable interpolator over the cube's pressure (hPa).

    Supports two cube layouts:

    * **Gridded**: dims ``(time, lat, lon)`` from the legacy
      :func:`bulk_fetch_era5_pressure_window` path. Trilinear in time,
      latitude, longitude.
    * **Point timeseries**: dim ``(time,)`` with scalar latitude /
      longitude coords from :func:`bulk_fetch_era5_pressure_timeseries`
      (ARCO-Zarr ``reanalysis-era5-single-levels-timeseries`` endpoint).
      Linear in time; lat/lon arguments are ignored because the basin
      is small relative to ERA5 synoptic scales (within-basin spatial
      variation of surface pressure is ~0.1-0.3 hPa = 1-3 mm IBE, well
      below DEM noise floor).

    Returned callable accepts ``pts`` of shape ``(N, 3)`` columns
    ``(t_seconds, lat, lon)`` and returns ``(N,)`` pressures in hPa.
    Out-of-range times return :data:`_REFERENCE_PRESSURE_HPA` (no-op IBE).
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"ERA5 cache not found at {cache_path}. Run the basin's "
            "cache_climate driver (e.g. `python -m beardmore.cache_climate`) "
            "to populate it."
        )

    ds = xr.open_dataset(cache_path)
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    if pressure_var not in ds:
        # Fall back to ERA5's compact CDS shortnames. New caches store
        # MSL ('msl'); legacy caches stored SP ('sp'). Either way we end
        # up with pressure in Pa and proceed identically downstream.
        if "msl" in ds:
            pressure_var = "msl"
        elif "sp" in ds:
            pressure_var = "sp"
        else:
            raise KeyError(
                f"pressure variable {pressure_var!r} not found in {cache_path}; "
                f"available: {list(ds.data_vars)}"
            )

    pressure_pa = ds[pressure_var].values
    pressure_hpa = pressure_pa / 100.0

    t_seconds = (
        ds[time_dim].values.astype("datetime64[s]")
        - np.datetime64("1970-01-01T00:00:00")
    ) / np.timedelta64(1, "s")
    t_seconds = t_seconds.astype("float64")

    is_point = pressure_hpa.ndim == 1
    if is_point:
        # ARCO point timeseries: pressure has only the time axis.
        # Build a 1D interpolator that ignores (lat, lon).
        from scipy.interpolate import interp1d
        time_interp = interp1d(
            t_seconds,
            pressure_hpa,
            bounds_error=False,
            fill_value=_REFERENCE_PRESSURE_HPA,
        )
        lat0 = float(ds[lat_name].values)
        lon0 = float(ds[lon_name].values)

        def interp(pts):
            arr = np.asarray(pts)
            if arr.ndim == 1:
                arr = arr[None, :]
            return time_interp(arr[:, 0])

        interp._t_axis = t_seconds
        interp._lat_axis = np.array([lat0])
        interp._lon_axis = np.array([lon0])
        interp._mode = "point"
    else:
        lats = ds[lat_name].values.astype("float64")
        lons = ds[lon_name].values.astype("float64")
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            pressure_hpa = pressure_hpa[:, ::-1, :]
        interp = RegularGridInterpolator(
            (t_seconds, lats, lons),
            pressure_hpa,
            bounds_error=False,
            fill_value=_REFERENCE_PRESSURE_HPA,
        )
        interp._t_axis = t_seconds
        interp._lat_axis = lats
        interp._lon_axis = lons
        interp._mode = "gridded"
    ds.close()
    return interp


def _to_seconds_since_epoch(t: np.datetime64) -> float:
    return float(
        (t.astype("datetime64[s]") - np.datetime64("1970-01-01T00:00:00"))
        / np.timedelta64(1, "s")
    )


def _epsg3031_to_latlon(xs: np.ndarray, ys: np.ndarray):
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)
