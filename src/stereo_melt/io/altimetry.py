# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Unified altimetry / GCP framework for ASP coregistration.

Each provider (ICESat-2 ATL06, CryoSat-2 L2 SARIn POCA, REMA-mosaic
rock outcrops, IceBridge ATM/LVIS, ICESat-1 GLAS GLAH12) returns its
data as a :class:`pointCollection.data` instance with a common minimal
schema:

- ``x``, ``y`` — EPSG:3031 meters (polar stereographic south, lat_ts=-71)
- ``h``       — height above WGS84 ellipsoid, meters
- ``t``       — UTC timestamp (numpy datetime64[ns]) where available
- ``source``  — short string label, free-form (e.g. ``"is2"``, ``"cs2_sarin_l2"``,
  ``"rock_rema_mosaic"``); used by the ASP emitter for provenance only

Providers are kept import-light so the module can be imported without
pulling sliderule / read-cryosat-2 unless a function that needs them is
actually called.

The unified ASP emitter :func:`to_asp_gcp_csv` concatenates multiple
``pointCollection.data`` instances and writes the
``easting,northing,h_mean`` CSV that
:func:`stereo_melt.coregister.asp.align_strip_with_asp` already
consumes — so this module slots in upstream of the existing ASP entry
point without changing the boundary contract.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pointCollection.data import data as PCData

__all__ = [
    "PCData",
    "from_is2_csv",
    "from_rock_csv",
    "from_atm",
    "from_lvis",
    "from_glah12",
    "glah12_time_range",
    "to_asp_gcp_csv",
]


def _ensure_xyh(d: dict) -> dict:
    r"""Validate that an altimetry dict has the minimum (x, y, h) keys.

    Raises ``KeyError`` if any are missing. Returns ``d`` unchanged so
    this can be used as a guard in `from_*` constructors.
    """
    for k in ("x", "y", "h"):
        if k not in d:
            raise KeyError(f"altimetry record missing required field {k!r}; got {list(d)}")
    return d


def _df_to_pc(df: pd.DataFrame, source: str, time_col: str | None = None) -> PCData:
    r"""Convert a DataFrame with ``easting,northing,h_mean`` columns to PCData.

    Adds a ``source`` field of constant value ``source`` for provenance.
    If ``time_col`` is supplied and present, copies it as ``t``.
    """
    n = len(df)
    out = {
        "x": df["easting"].to_numpy(np.float64),
        "y": df["northing"].to_numpy(np.float64),
        "h": df["h_mean"].to_numpy(np.float64),
        "source": np.full(n, source, dtype=object),
    }
    if time_col and time_col in df.columns:
        out["t"] = pd.to_datetime(df[time_col]).to_numpy()
    _ensure_xyh(out)
    return PCData().from_dict(out)


def from_is2_csv(csv_path: str | os.PathLike) -> PCData:
    r"""Load an existing IS2 ATL06 control CSV (sliderule output) as PCData.

    The CSV is expected to be the per-strip cache produced by
    :func:`stereo_melt.coregister.reference.download_icesat2_data`,
    schema ``time, easting, northing, h_mean``. Time column is optional;
    older caches without it still load.
    """
    df = pd.read_csv(csv_path)
    return _df_to_pc(df, source="is2_atl06", time_col="time")


def from_rock_csv(csv_path: str | os.PathLike) -> PCData:
    r"""Load an existing REMA-mosaic rock-outcrop CSV as PCData.

    Two on-disk schemas are supported (the per-strip cache pipeline
    has used both at different times):

    - ``easting, northing, h_mean`` (extract_rock_elevations)
    - ``x, y, elevation, tile``    (extract_rock_elevations_from_mosaics)

    Either form is harmonized to the unified
    ``(x, y, h, source="rock_rema_mosaic")`` schema.
    """
    df = pd.read_csv(csv_path)
    if {"easting", "northing", "h_mean"}.issubset(df.columns):
        return _df_to_pc(df, source="rock_rema_mosaic")
    if {"x", "y", "elevation"}.issubset(df.columns):
        out = {
            "x": df["x"].to_numpy(np.float64),
            "y": df["y"].to_numpy(np.float64),
            "h": df["elevation"].to_numpy(np.float64),
            "source": np.full(len(df), "rock_rema_mosaic", dtype=object),
        }
        return PCData().from_dict(out)
    raise ValueError(
        f"rock CSV {csv_path} has unrecognized schema; got columns {list(df.columns)}"
    )


# IceBridge airborne lidar product readers.
#
# Antarctic IceBridge campaigns (Operation IceBridge, 2009-2019) flew the
# small-footprint ATM and large-footprint LVIS lidars as primary
# instruments. NSIDC distributes the L2 products at:
#
#   ILATM2 v002 — ATM L2 Icessn Elevation, Slope, and Roughness (CSV)
#   ILVIS2 v002 — LVIS L2 Geolocated Surface Elevation (TXT, whitespace)
#
# Older versions shipped HDF5; v002 (current) ships ASCII. We auto-detect
# by suffix.

# ATM v002 column header (after stripping "# "): the 11-column smoothed-
# nadir schema used since 2014. The leading column is UTC seconds-of-day.
ATM_V002_COLS = (
    "UTC_Seconds_Of_Day",
    "Latitude",
    "Longitude",
    "WGS84_Ellipsoid_Height",
    "South_to_North_Slope",
    "West_to_East_Slope",
    "RMS_Fit_cm",
    "N_ATM_Used",
    "N_ATM_Removed",
    "Block_Distance_From_Aircraft_m",
    "Track_Identifier",
)

# LVIS v002 columns. The reader is whitespace-tolerant and finds the
# header line via "# LVIS_LFID" or similar.
LVIS_V002_COLS = (
    "LVIS_LFID", "SHOTNUMBER", "TIME",
    "LONGITUDE_CENTROID", "LATITUDE_CENTROID", "ELEVATION_CENTROID",
    "LONGITUDE_LOW", "LATITUDE_LOW", "ELEVATION_LOW",
    "LONGITUDE_HIGH", "LATITUDE_HIGH", "ELEVATION_HIGH",
)


def _read_atm_csv(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Parse an ILATM2 v002 CSV and return ``(lat, lon, elev, sec_of_day)``.

    The file format is a leading ``#``-prefixed header block followed by
    comma-separated numerical rows. We use :func:`numpy.loadtxt` with
    ``comments="#"`` to skip the header.
    """
    arr = np.loadtxt(str(path), delimiter=",", comments="#")
    # Some granules have a single row collapsed to 1-D; force 2-D.
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 4:
        raise ValueError(
            f"ILATM2 CSV {path}: expected ≥4 columns, got {arr.shape[1]}"
        )
    sec_of_day = arr[:, 0]
    lat = arr[:, 1]
    lon = arr[:, 2]
    elev = arr[:, 3]
    return lat, lon, elev, sec_of_day


def _read_lvis_txt(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Parse an ILVIS2 v002 TXT and return ``(lat, lon, elev, sec_of_day)``.

    Whitespace-separated, ``#``-prefixed header. Reads the
    ``CENTROID`` columns (the standard choice for ice-sheet GCPs) and
    ``TIME`` as seconds-of-day.
    """
    arr = np.loadtxt(str(path), comments="#")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 6:
        raise ValueError(
            f"ILVIS2 TXT {path}: expected ≥6 columns, got {arr.shape[1]}"
        )
    sec_of_day = arr[:, 2]
    lon = arr[:, 3]   # LONGITUDE_CENTROID
    lat = arr[:, 4]   # LATITUDE_CENTROID
    elev = arr[:, 5]  # ELEVATION_CENTROID
    return lat, lon, elev, sec_of_day


_DATE_FROM_NAME_ATM = re.compile(r"ILATM2_(\d{8})_")
_DATE_FROM_NAME_LVIS = re.compile(r"ILVIS2_AQ(\d{4})_(\d{4})_")


def _granule_date_atm(name: str) -> np.datetime64 | None:
    m = _DATE_FROM_NAME_ATM.search(name)
    if not m:
        return None
    return np.datetime64(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}", "ns")


def _granule_date_lvis(name: str) -> np.datetime64 | None:
    """ILVIS2 names look like ``ILVIS2_AQ2009_1020_R1408_055102.TXT``
    (AQ2009 = austral campaign 2009, 1020 = MMDD)."""
    m = _DATE_FROM_NAME_LVIS.search(name)
    if not m:
        return None
    year = m.group(1)
    mmdd = m.group(2)
    return np.datetime64(f"{year}-{mmdd[:2]}-{mmdd[2:4]}", "ns")


def from_atm(path: str | os.PathLike) -> PCData:
    r"""Load an IceBridge ILATM2 granule as :class:`PCData`.

    Auto-detects the on-disk format by suffix:

    - ``.csv`` (NSIDC ILATM2 v002, current) — comma-separated, 11 cols
    - ``.h5`` / ``.he5`` — older HDF5 distribution

    Coordinates are reprojected from geographic (WGS84) to EPSG:3031;
    longitudes in the 0..360 convention are normalized to -180..180.
    Time is reconstructed by combining the granule-date in the filename
    with ``UTC_Seconds_Of_Day``.

    No filtering (grounded mask / slow-velocity) is applied here — that
    is the caller's responsibility (see
    :mod:`stereo_melt.coregister.cache_airborne`).
    """
    from pyproj import Transformer

    p = os.fspath(path)
    suffix = os.path.splitext(p)[1].lower()
    if suffix in (".csv", ".txt"):
        lat, lon, elev, sod = _read_atm_csv(p)
    elif suffix in (".h5", ".he5", ".hdf5"):
        # Older ATM L2 HDF5 distributions; try the most-common dataset names.
        import h5py
        with h5py.File(p, "r") as f:
            def _ds(names):
                for n in names:
                    if n in f:
                        return np.asarray(f[n][()], dtype=np.float64).ravel()
                return None
            lat = _ds(("latitude", "Latitude"))
            lon = _ds(("longitude", "Longitude"))
            elev = _ds(("elevation", "Elevation"))
            if lat is None or lon is None or elev is None:
                raise ValueError(
                    f"ILATM2 HDF5 {p}: could not locate latitude/longitude/elevation"
                )
            sod = np.full(lat.size, np.nan)
    else:
        raise ValueError(f"unsupported ILATM2 suffix {suffix!r} on {p}")

    finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(elev)
    lat = lat[finite]; lon = lon[finite]; elev = elev[finite]; sod = sod[finite]
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    east, north = transformer.transform(lon, lat)

    out = {
        "x": np.asarray(east, dtype=np.float64),
        "y": np.asarray(north, dtype=np.float64),
        "h": np.asarray(elev, dtype=np.float64),
        "source": np.full(lat.size, "atm_ilatm2", dtype=object),
    }
    # Reconstruct absolute time = granule_date (00:00 UTC) + sec_of_day.
    base = _granule_date_atm(os.path.basename(p))
    if base is not None and np.isfinite(sod).any():
        t_ns = base.astype("int64") + (sod * 1e9).astype("int64")
        out["t"] = t_ns
    return PCData().from_dict(out)


def from_lvis(path: str | os.PathLike) -> PCData:
    r"""Load an IceBridge ILVIS2 granule as :class:`PCData`.

    Auto-detects the on-disk format by suffix:

    - ``.txt`` / ``.TXT`` (NSIDC ILVIS2 v002, current) — whitespace-
      separated, 12 cols with ``LATITUDE_CENTROID`` / ``LONGITUDE_CENTROID``
      / ``ELEVATION_CENTROID``
    - ``.h5`` / ``.he5`` — older HDF5 distribution
    """
    from pyproj import Transformer

    p = os.fspath(path)
    suffix = os.path.splitext(p)[1].lower()
    if suffix in (".txt", ".csv"):
        lat, lon, elev, sod = _read_lvis_txt(p)
    elif suffix in (".h5", ".he5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            def _ds(names):
                for n in names:
                    if n in f:
                        return np.asarray(f[n][()], dtype=np.float64).ravel()
                return None
            lat = _ds(("LATITUDE_CENTROID", "latitude_centroid", "LATITUDE_T", "latitude"))
            lon = _ds(("LONGITUDE_CENTROID", "longitude_centroid", "LONGITUDE_T", "longitude"))
            elev = _ds(("ELEVATION_CENTROID", "elevation_centroid", "ELEVATION_T", "elevation"))
            if lat is None or lon is None or elev is None:
                raise ValueError(
                    f"ILVIS2 HDF5 {p}: could not locate "
                    f"LATITUDE_CENTROID/LONGITUDE_CENTROID/ELEVATION_CENTROID"
                )
            sod = np.full(lat.size, np.nan)
    else:
        raise ValueError(f"unsupported ILVIS2 suffix {suffix!r} on {p}")

    finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(elev)
    lat = lat[finite]; lon = lon[finite]; elev = elev[finite]; sod = sod[finite]
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    east, north = transformer.transform(lon, lat)

    out = {
        "x": np.asarray(east, dtype=np.float64),
        "y": np.asarray(north, dtype=np.float64),
        "h": np.asarray(elev, dtype=np.float64),
        "source": np.full(lat.size, "lvis_ilvis2", dtype=object),
    }
    base = _granule_date_lvis(os.path.basename(p))
    if base is not None and np.isfinite(sod).any():
        t_ns = base.astype("int64") + (sod * 1e9).astype("int64")
        out["t"] = t_ns
    return PCData().from_dict(out)


# ICESat-1 GLAS (GLAH12) reader.
#
# GLAH12 v034 — GLAS/ICESat L2 Antarctic and Greenland Ice Sheet
# Altimetry (HDF5, NSIDC). Two well-known gotchas both handled here:
#
# 1. **Ellipsoid**: ``d_elev`` is referenced to the TOPEX/Poseidon
#    ellipsoid, ~70 cm BELOW WGS84 at these latitudes. The per-point
#    ``d_deltaEllip`` field is defined as (elevation w.r.t. T/P) minus
#    (elevation w.r.t. WGS84), so ``h_wgs84 = d_elev - d_deltaEllip``.
# 2. **Saturation**: bright flat ice saturates the 1064 nm detector;
#    ``d_satElevCorr`` is the additive range correction. Convention
#    (Schröder et al. 2019; NSIDC user guide): apply the correction when
#    it is valid, reject shots with ``sat_corr_flg > 2`` (not
#    correctable).
#
# Additional quality gates: ``elev_use_flg == 0`` (elevation usable) and
# ``i_numPk == 1`` (single waveform peak — multi-peak returns over ice
# usually mean clouds/forward scattering). Invalid doubles are DBL_MAX.

_GLAH12_FILL_THRESHOLD = 1.0e30
_GLAS_J2000_EPOCH = np.datetime64("2000-01-01T12:00:00", "ns")


def glah12_time_range(path: str | os.PathLike) -> tuple[pd.Timestamp, pd.Timestamp]:
    r"""Return ``(t_min, t_max)`` of a GLAH12 granule as UTC Timestamps.

    Reads ``Data_40HZ/DS_UTCTime_40`` (seconds since the J2000 epoch,
    2000-01-01T12:00:00 UTC). GLAH12 filenames carry orbit/track counters,
    not calendar dates, so callers that need granule↔strip Δt matching
    must read the time from inside the file (see
    :mod:`stereo_melt.coregister.cache_glas`'s granule index).
    """
    import h5py
    with h5py.File(os.fspath(path), "r") as f:
        tt = np.asarray(f["Data_40HZ/DS_UTCTime_40"][()], dtype=np.float64).ravel()
    tt = tt[np.isfinite(tt) & (np.abs(tt) < _GLAH12_FILL_THRESHOLD)]
    if tt.size == 0:
        raise ValueError(f"GLAH12 {path}: no valid DS_UTCTime_40 values")
    t0 = pd.Timestamp(_GLAS_J2000_EPOCH + np.timedelta64(int(tt.min() * 1e9), "ns"))
    t1 = pd.Timestamp(_GLAS_J2000_EPOCH + np.timedelta64(int(tt.max() * 1e9), "ns"))
    return t0, t1


def from_glah12(
    path: str | os.PathLike,
    *,
    max_abs_elev_m: float = 5000.0,
) -> PCData:
    r"""Load an ICESat-1 GLAS GLAH12 granule as :class:`PCData`.

    Applies the saturation correction, converts the TOPEX/Poseidon-
    referenced elevations to WGS84 (see the block comment above), gates
    on ``elev_use_flg == 0``, ``sat_corr_flg <= 2`` and ``i_numPk == 1``,
    and reprojects to EPSG:3031. No geographic filtering (grounded mask /
    slow-velocity) is applied here — that is the caller's responsibility
    (see :mod:`stereo_melt.coregister.cache_glas`).
    """
    import h5py
    from pyproj import Transformer

    p = os.fspath(path)
    with h5py.File(p, "r") as f:
        def _ds(name, dtype=np.float64):
            if name in f:
                return np.asarray(f[name][()], dtype=dtype).ravel()
            return None

        lat = _ds("Data_40HZ/Geolocation/d_lat")
        lon = _ds("Data_40HZ/Geolocation/d_lon")
        elev = _ds("Data_40HZ/Elevation_Surfaces/d_elev")
        if lat is None or lon is None or elev is None:
            raise ValueError(
                f"GLAH12 {p}: missing Data_40HZ d_lat/d_lon/d_elev datasets"
            )
        sat_corr = _ds("Data_40HZ/Elevation_Corrections/d_satElevCorr")
        delta_ellip = _ds("Data_40HZ/Geophysical/d_deltaEllip")
        use_flg = _ds("Data_40HZ/Quality/elev_use_flg")
        sat_flg = _ds("Data_40HZ/Quality/sat_corr_flg")
        num_pk = _ds("Data_40HZ/Waveform/i_numPk")
        tsec = _ds("Data_40HZ/DS_UTCTime_40")

    def _valid(a):
        return np.isfinite(a) & (np.abs(a) < _GLAH12_FILL_THRESHOLD)

    if delta_ellip is None:
        raise ValueError(
            f"GLAH12 {p}: missing Data_40HZ/Geophysical/d_deltaEllip — refusing "
            f"to emit TOPEX/Poseidon-referenced heights as WGS84 (~70 cm bias)."
        )

    keep = _valid(lat) & _valid(lon) & _valid(elev)
    keep &= np.abs(elev) < max_abs_elev_m
    keep &= _valid(delta_ellip) & (np.abs(delta_ellip) < 10.0)
    if use_flg is not None:
        keep &= use_flg == 0
    if sat_flg is not None:
        keep &= sat_flg <= 2
    if num_pk is not None:
        keep &= num_pk == 1

    # Saturation correction is additive where valid; for unsaturated shots
    # (flags 0/1) the field is 0 or fill, so the guard keeps it a no-op.
    h = elev.copy()
    if sat_corr is not None:
        corr = np.where(_valid(sat_corr) & (np.abs(sat_corr) < 10.0), sat_corr, 0.0)
        h = h + corr
    h = np.where(_valid(delta_ellip), h - delta_ellip, np.nan)
    keep &= np.isfinite(h)

    lat = lat[keep]; lon = lon[keep]; h = h[keep]
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    east, north = transformer.transform(lon, lat)

    out = {
        "x": np.asarray(east, dtype=np.float64),
        "y": np.asarray(north, dtype=np.float64),
        "h": np.asarray(h, dtype=np.float64),
        "source": np.full(lat.size, "glas_glah12", dtype=object),
    }
    if tsec is not None:
        ts = tsec[keep]
        good_t = _valid(ts)
        if good_t.any():
            t_ns = _GLAS_J2000_EPOCH.astype("int64") + np.where(
                good_t, (ts * 1e9), np.nan
            )
            # NaT for invalid times, ns since epoch otherwise.
            t_arr = np.full(ts.size, np.datetime64("NaT"), dtype="datetime64[ns]")
            t_arr[good_t] = t_ns[good_t].astype("int64").view("datetime64[ns]")
            out["t"] = t_arr
    return PCData().from_dict(out)


def to_asp_gcp_csv(
    parts: Sequence[PCData],
    output_csv: str | os.PathLike,
    write_provenance: bool = True,
) -> str:
    r"""Concatenate altimetry sources and write the ASP-format GCP CSV.

    The CSV schema is the contract :func:`align_strip_with_asp` already
    consumes: ``easting, northing, h_mean`` (one row per control point).
    All input ``parts`` must carry the unified ``(x, y, h)`` fields;
    optional fields (``source``, ``t``, ...) are passed through to a
    side-car CSV at ``<output_csv>.provenance.csv`` when
    ``write_provenance=True``.

    Parameters
    ----------
    parts : sequence of :class:`pointCollection.data`
        One per provider. Empty parts are skipped silently. NaN-h rows
        are dropped (ASP rejects them on its own, but dropping here keeps
        the side-car aligned).
    output_csv : path
        Destination for the ASP CSV.
    write_provenance : bool
        If True (default), also writes ``<output_csv>.provenance.csv``
        with extra columns (``source``, ``t``) for downstream diagnostics.
    """
    rows_xyh: list[np.ndarray] = []
    rows_src: list[np.ndarray] = []
    rows_t: list[np.ndarray] = []
    for pc in parts:
        if pc is None or pc.size == 0:
            continue
        for k in ("x", "y", "h"):
            if k not in pc.fields:
                raise KeyError(
                    f"provider missing required field {k!r}; got fields {pc.fields}"
                )
        x = np.asarray(pc.x, dtype=np.float64).ravel()
        y = np.asarray(pc.y, dtype=np.float64).ravel()
        h = np.asarray(pc.h, dtype=np.float64).ravel()
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(h)
        if not keep.any():
            continue
        rows_xyh.append(np.column_stack([x[keep], y[keep], h[keep]]))
        if "source" in pc.fields:
            rows_src.append(np.asarray(pc.source).ravel()[keep])
        else:
            rows_src.append(np.full(int(keep.sum()), "unknown", dtype=object))
        if "t" in pc.fields:
            rows_t.append(np.asarray(pc.t).ravel()[keep])
        else:
            rows_t.append(np.full(int(keep.sum()), np.datetime64("NaT"), dtype="datetime64[ns]"))

    if not rows_xyh:
        raise ValueError("to_asp_gcp_csv: no finite (x, y, h) rows across providers")

    xyh = np.concatenate(rows_xyh, axis=0)
    src = np.concatenate(rows_src, axis=0)
    t = np.concatenate(rows_t, axis=0)

    df = pd.DataFrame({
        "easting":  xyh[:, 0],
        "northing": xyh[:, 1],
        "h_mean":   xyh[:, 2],
    })
    df.to_csv(output_csv, index=False)

    if write_provenance:
        prov_path = str(output_csv) + ".provenance.csv"
        prov = df.copy()
        prov["source"] = src
        prov["t"] = t
        prov.to_csv(prov_path, index=False)

    return str(output_csv)


# ---------------------------------------------------------------------------
# Compressed HDF5 cache for ground-control point clouds.
# ---------------------------------------------------------------------------
# The IS2 ATL06 filtered caches used to live as text CSV at
# ``<asp_root>/icesat2_data/icesat2_filtered_<dem_id>.csv`` (4-5 MB per
# strip, 237 strips on Beardmore alone -> ~8 GB of redundant ASCII).
# Re-encoded as gzip+shuffle HDF5 with float64 datasets, the same data
# fits in ~600 KB per strip (~7x compression) and reads back in
# milliseconds rather than the ~hundreds of ms pandas spends parsing.
# CS2 caches are already H5 via PointCollection so this helper is also
# the reference encoding for any new control-point sidecar we cache.


_CONTROL_H5_SCHEMA = "stereo_melt_control_v1"


def write_control_h5(
    df: "pd.DataFrame",
    path: "str | os.PathLike",
    *,
    compression: str = "gzip",
    compression_opts: int = 4,
    shuffle: bool = True,
) -> str:
    r"""Write a control-point DataFrame to compressed HDF5.

    Required columns: ``easting``, ``northing``, ``h_mean``. Optional
    column: ``time`` (parsed to nanosecond-precision int64 and stored
    as ``/time_ns``). All other columns are dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        Control points. Must carry ``easting``, ``northing``, ``h_mean``.
    path : str or PathLike
        Destination ``.h5``.
    compression, compression_opts, shuffle : HDF5 dataset filters.
        Defaults (gzip level 4 + shuffle) hit a ~7x ratio on
        IS2-like float64 data with sub-second per-strip write time.

    Returns
    -------
    str
        ``str(path)``.
    """
    import h5py
    e = df["easting"].to_numpy(dtype=np.float64)
    n = df["northing"].to_numpy(dtype=np.float64)
    h = df["h_mean"].to_numpy(dtype=np.float64)
    n_rows = len(e)
    chunk = (min(n_rows, 8192),) if n_rows > 0 else None

    has_time = "time" in df.columns
    if has_time:
        # IS2 cache writes nanosecond-precision UTC timestamps; preserve
        # full precision by storing as int64 ns since the Unix epoch.
        t_ns = pd.to_datetime(df["time"]).to_numpy(dtype="datetime64[ns]").view(np.int64)

    kw = {"compression": compression, "compression_opts": compression_opts,
          "shuffle": shuffle}
    if chunk is not None:
        kw["chunks"] = chunk
    with h5py.File(str(path), "w") as f:
        f.attrs["schema"] = _CONTROL_H5_SCHEMA
        f.create_dataset("easting",  data=e, **kw)
        f.create_dataset("northing", data=n, **kw)
        f.create_dataset("h_mean",   data=h, **kw)
        if has_time:
            f.create_dataset("time_ns", data=t_ns, **kw)
    return str(path)


def read_control_h5(path: "str | os.PathLike") -> "pd.DataFrame":
    r"""Read a control-point HDF5 file written by :func:`write_control_h5`.

    Returns a DataFrame with columns ``easting``, ``northing``,
    ``h_mean`` and (if present) ``time``. Time is restored to
    nanosecond-precision ``datetime64[ns]``.
    """
    import h5py
    with h5py.File(str(path), "r") as f:
        out = {
            "easting":  f["easting"][...].astype(np.float64),
            "northing": f["northing"][...].astype(np.float64),
            "h_mean":   f["h_mean"][...].astype(np.float64),
        }
        if "time_ns" in f:
            out["time"] = pd.to_datetime(f["time_ns"][...].astype(np.int64), unit="ns")
    return pd.DataFrame(out)


def read_control_any(path: "str | os.PathLike") -> "pd.DataFrame":
    r"""Read a control-point cache from either ``.h5`` (preferred) or
    ``.csv`` (legacy). The sibling-file fallback lets the pipeline keep
    consuming CSV caches written before the HDF5 migration while the
    writer always emits HDF5.
    """
    p = str(path)
    if p.endswith(".h5") and os.path.exists(p):
        return read_control_h5(p)
    if p.endswith(".csv") and os.path.exists(p):
        return pd.read_csv(p)
    # Allow callers to pass either suffix and we figure it out.
    base, _ = os.path.splitext(p)
    for ext in (".h5", ".csv"):
        alt = base + ext
        if os.path.exists(alt):
            if ext == ".h5":
                return read_control_h5(alt)
            return pd.read_csv(alt)
    raise FileNotFoundError(f"No control cache found at {p} (.h5 or .csv)")
