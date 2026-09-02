# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""RACMO2.4p1 SMB loader and regridder.

Source: Zenodo 19255213 (van Dalum et al. 2026) — monthly RACMO2.4p1 data for
Antarctica at 11 km resolution, 1979-01 through 2025-12. Paper:
van Dalum et al. 2025, The Cryosphere 19, 4061-4090 (doi:10.5194/tc-19-4061-2025).

Primary variable: ``smbgl`` — surface mass balance over glaciated surfaces in
kg m^-2 per month (= mm water equivalent per month). Companion ANT11_masks.nc
provides IceMask (fractional), Icesheet_Only (grounded), and Area (m^2).

Grid is rotated-pole (CF ``grid_mapping=rotated_pole``); reprojection to
EPSG:3031 is done via pyproj using the file's 2-D lat/lon coordinates.

Typical usage::

    from stereo_melt.io.smb import download_racmo, smb_over_window

    smb_path, _ = download_racmo("/wd2/projects/stereo_melt/data/RACMO/")
    smb_m_ie = smb_over_window(
        smb_path, target_x, target_y,
        start="2013-01-01", end="2015-01-01",
    )  # meters of ice equivalent over the window on the EPSG:3031 target grid
"""

from pathlib import Path

import numpy as np
import requests
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import griddata
from tqdm import tqdm

from ..constants import rhoi

RACMO_SMB_URL = (
    "https://zenodo.org/api/records/19255213/files/"
    "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202512.nc/content"
)
RACMO_MASK_URL = "https://zenodo.org/api/records/19255213/files/ANT11_masks.nc/content"
SMB_FILENAME = "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202512.nc"
MASK_FILENAME = "ANT11_masks.nc"


def download_racmo(dest_dir, overwrite=False):
    """Download RACMO2.4p1 SMB + mask NetCDFs to ``dest_dir``.

    Parameters
    ----------
    dest_dir : str or Path
        Destination directory; created if missing.
    overwrite : bool
        If False (default), skip files already present on disk.

    Returns
    -------
    tuple[Path, Path]
        (smb_path, mask_path).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    smb_path = dest_dir / SMB_FILENAME
    mask_path = dest_dir / MASK_FILENAME
    for url, path in [(RACMO_SMB_URL, smb_path), (RACMO_MASK_URL, mask_path)]:
        if path.exists() and not overwrite:
            mb = path.stat().st_size / 1e6
            print(f"✅ {path.name} already present ({mb:.0f} MB)")
            continue
        _stream_download(url, path)
    return smb_path, mask_path


def _stream_download(url, path):
    """Stream a URL to ``path`` with a progress bar."""
    print(f"📡 Downloading {url} → {path}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with (
            open(path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def load_racmo_smb(smb_path):
    """Open the RACMO2.4p1 monthly SMB NetCDF as an xarray DataArray.

    Returns the ``smbgl`` field with dims (time, rlat, rlon) and 2-D lat/lon
    coordinates attached for later reprojection.
    """
    ds = xr.open_dataset(smb_path, decode_times=True)
    if "smbgl" not in ds.variables:
        raise KeyError(f"Expected variable 'smbgl' in {smb_path}; found {list(ds.data_vars)}")
    smb = ds["smbgl"]
    # RACMO files typically ship 2-D lat/lon as separate variables; keep them
    # as coordinates on the returned DataArray so the regridder can find them.
    for name in ("lat", "lon"):
        if name not in smb.coords and name in ds.variables:
            smb = smb.assign_coords({name: ds[name]})
    return smb


def load_racmo_mask(mask_path):
    """Open the ANT11 mask NetCDF. Returns the full dataset.

    Useful variables: ``IceMask`` (fractional coverage 0-1), ``Icesheet_Only``
    (grounded-ice flag), ``Area`` (m^2 per cell).
    """
    return xr.open_dataset(mask_path)


def integrate_smb_window(smb_da, start, end):
    """Sum monthly SMB values over [start, end] inclusive.

    RACMO monthly files report accumulated totals for each month (kg m^-2 per
    month). Summing over a date window gives cumulative SMB in kg m^-2.

    Parameters
    ----------
    smb_da : xr.DataArray
        Output of :func:`load_racmo_smb`, dims (time, rlat, rlon).
    start, end : str, datetime, or pandas Timestamp
        Window bounds, interpreted by :meth:`xarray.DataArray.sel`.

    Returns
    -------
    xr.DataArray
        Cumulative SMB over the window, dims (rlat, rlon), units kg m^-2.
    """
    sel = smb_da.sel(time=slice(start, end))
    if sel.time.size == 0:
        raise ValueError(
            f"No RACMO monthly SMB samples found in [{start}, {end}]."
            f" Data spans {smb_da.time.min().values} to {smb_da.time.max().values}."
        )
    return sel.sum("time", keep_attrs=True)


def regrid_smb_to_xy(smb_window, target_x, target_y, method="linear"):
    """Reproject a rotated-pole SMB field to EPSG:3031 target points.

    Builds EPSG:3031 (x, y) from the 2-D lat/lon coords via pyproj, then uses
    :func:`scipy.interpolate.griddata` to sample at (target_x, target_y).

    Parameters
    ----------
    smb_window : xr.DataArray
        Output of :func:`integrate_smb_window`, dims (rlat, rlon). Must carry
        2-D lat/lon coords.
    target_x, target_y : np.ndarray
        Target grid coordinates in EPSG:3031 (meters). Either 1-D (meshgridded
        internally) or 2-D arrays of matching shape.
    method : str
        ``"linear"`` (default) or ``"nearest"``.

    Returns
    -------
    np.ndarray
        SMB at target points with the shape of the (x, y) meshgrid.
    """
    if "lat" not in smb_window.coords or "lon" not in smb_window.coords:
        raise KeyError("smb_window must carry lat/lon coords from load_racmo_smb")

    lat = np.asarray(smb_window["lat"].values)
    lon = np.asarray(smb_window["lon"].values)
    if lat.ndim == 1 and lon.ndim == 1:
        # Some files ship 1-D lat/lon along rlat/rlon; promote via meshgrid.
        lon, lat = np.meshgrid(lon, lat)
    elif lat.ndim != 2 or lon.ndim != 2:
        raise ValueError(f"Unexpected lat/lon dims: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")

    # Project source cell centers to EPSG:3031
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    src_x, src_y = tf.transform(lon, lat)

    # Flatten and drop NaNs / off-domain cells
    values = np.asarray(smb_window.values, dtype=float)
    points = np.column_stack([src_x.ravel(), src_y.ravel()])
    vals = values.ravel()
    ok = np.isfinite(vals) & np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points, vals = points[ok], vals[ok]

    tx = np.asarray(target_x)
    ty = np.asarray(target_y)
    if tx.ndim == 1 and ty.ndim == 1:
        tx, ty = np.meshgrid(tx, ty)

    return griddata(points, vals, (tx, ty), method=method)


def smb_to_m_ice_equivalent(smb_kg_m2):
    """Convert SMB in kg m^-2 (= mm water equivalent) to meters of ice equivalent.

    Derivation: 1 kg m^-2 of mass = 1 mm of pure water. To express that same
    mass as thickness of ice at density ``rhoi``, divide by ``rhoi`` (kg m^-3),
    giving meters of ice.

    The current ``rhoi`` in stereo_melt.constants is 918 kg m^-3, so one
    kilogram per square meter equals ~1.09 mm of ice.
    """
    return np.asarray(smb_kg_m2) / rhoi


def smb_over_window(smb_path, target_x, target_y, start, end, method="linear"):
    """End-to-end: load, integrate over [start, end], regrid, and convert.

    Returns SMB in meters of ice equivalent, accumulated over the window, at
    (target_x, target_y) in EPSG:3031.
    """
    smb = load_racmo_smb(smb_path)
    windowed = integrate_smb_window(smb, start, end)
    grid_kg = regrid_smb_to_xy(windowed, target_x, target_y, method=method)
    return smb_to_m_ice_equivalent(grid_kg)
