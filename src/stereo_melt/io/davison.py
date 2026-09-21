"""Davison et al. 2023 ice-shelf melt-rate dataset loaders.

Davison B. J. et al., "Annual mass budget of Antarctic ice shelves from
1997 to 2021", Science Advances 9, eadi0186 (2023). DOI 10.1126/sciadv.adi0186.
Data archived at Zenodo 10.5281/zenodo.8052519, on disk at
``/wd2/projects/stereo_melt/data/Davison2023/``.

Two products:

- ``basal_melt/basal_melt_map_racmo_firn_air_corrected.tif`` — pan-Antarctic
  1 km gridded basal melt rate (m ice/yr, positive = melt) on EPSG:3031.
  Static composite, RACMO firn-air corrected.
- ``basal_melt/timeseries/csv_outputs/<Shelf>-timeseries.csv`` — per-shelf
  monthly + yearly time series with errors. 55 named shelves; Beardmore is
  not in this list (Davison's mask catalog is Greene 2022 named shelves).

**Sign convention divergence.** Davison publishes positive = melt. This
package's ``melt_rate`` carries basal
mass balance with positive = accretion, negative = melt. To compare,
**negate** Davison values (or our ``melt_rate``) so both are on the
same convention. Loaders here keep Davison's native sign and document
it; conversion is the caller's responsibility.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr  # noqa: F401  (registers .rio accessor on xarray)
import xarray as xr

DAVISON_ROOT = Path("/wd2/projects/stereo_melt/data/Davison2023")
DAVISON_GRIDDED_TIF = DAVISON_ROOT / "data" / "basal_melt" / "basal_melt_map_racmo_firn_air_corrected.tif"
DAVISON_CSV_DIR = DAVISON_ROOT / "data" / "basal_melt" / "timeseries" / "csv_outputs"


def load_davison_gridded_on_grid(template: xr.DataArray) -> xr.DataArray:
    r"""Reproject the Davison 1 km gridded basal-melt map onto ``template`` (y, x).

    Uses a bilinear `interp_like` (both arrays are EPSG:3031), so the
    output shares the template's coordinates exactly. NaN-fills the
    ``-3.4e38`` Davison no-data sentinel before interpolation so it
    does not bleed into valid cells.

    Parameters
    ----------
    template : xr.DataArray
        Any of our basin DataArrays with ``(y, x)`` coords on EPSG:3031.

    Returns
    -------
    xr.DataArray
        Davison melt rate (m ice/yr, positive = melt) on the template grid.
    """
    if not DAVISON_GRIDDED_TIF.exists():
        raise FileNotFoundError(
            f"Davison gridded TIFF not at {DAVISON_GRIDDED_TIF}. "
            "Did the Zenodo unzip step finish?"
        )
    da = rxr.open_rasterio(str(DAVISON_GRIDDED_TIF)).squeeze(drop=True)
    nodata = da.rio.nodata
    da = da.where(da != nodata) if nodata is not None else da
    da = da.where(np.isfinite(da))
    out = da.interp(
        x=template["x"], y=template["y"], method="linear",
    )
    out.name = "davison_basal_melt"
    out.attrs = {
        "units": "m ice yr^-1; positive = melt",
        "source": "Davison 2023 (Sci Adv adi0186), Zenodo 8052519",
        "product": "basal_melt_map_racmo_firn_air_corrected.tif",
        "regridding": "bilinear interp on EPSG:3031 from 1 km native",
    }
    return out


def load_davison_shelf_timeseries(shelf_name: str) -> pd.DataFrame:
    r"""Read the Davison per-shelf monthly+yearly basal melt CSV.

    Parameters
    ----------
    shelf_name : str
        Name as it appears in the CSV filename, e.g. ``"Nansen"``,
        ``"Pine_Island"``, ``"Crosson"``, ``"Dotson"``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``dates_datetime``, ``dates_decimal``, ``bm_monthly``,
        ``bm_monthly_errors``, ``bm_yearly``, ``bm_yearly_errors``,
        ``area_km2``. ``dates_datetime`` is parsed to pandas Timestamps.
    """
    csv = DAVISON_CSV_DIR / f"{shelf_name}-timeseries.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"No Davison CSV for {shelf_name!r}. "
            f"Look at {DAVISON_CSV_DIR} for the available list of 55 shelves."
        )
    df = pd.read_csv(csv, index_col=0)
    df["dates_datetime"] = pd.to_datetime(df["dates_datetime"])
    return df


def davison_window_mean(
    shelf_name: str, start: str, end: str,
) -> tuple[float, float, int]:
    r"""Return ``(mean, mean_error, n_months)`` of Davison ``bm_yearly`` over
    ``[start, end)`` for a given shelf, or ``(nan, nan, 0)`` if the window
    falls outside Davison's coverage. Errors are mean-of-monthly errors;
    treat as a noisy bound. Sign convention is Davison-native
    (positive = melt); use :func:`davison_window_mean_shean` for the
    negated package-convention value."""
    df = load_davison_shelf_timeseries(shelf_name)
    mask = (df["dates_datetime"] >= start) & (df["dates_datetime"] < end)
    sub = df[mask]
    if sub.empty:
        return float("nan"), float("nan"), 0
    return (
        float(sub["bm_yearly"].mean(skipna=True)),
        float(sub["bm_yearly_errors"].mean(skipna=True)),
        int(sub["bm_yearly"].notna().sum()),
    )


def load_davison_gridded_in_shean(template: xr.DataArray) -> xr.DataArray:
    r"""Davison gridded TIFF on ``template``, in Shean public convention.

    Convenience wrapper over :func:`load_davison_gridded_on_grid`: the
    Davison product publishes positive = melt; we negate so the result
    aligns with the package-wide Shean convention (negative = melt,
    positive = accretion) used by every ``melt_rate`` output in this
    library. Use this helper when comparing Davison to our solver
    outputs; use :func:`load_davison_gridded_on_grid` only if you need
    Davison's native sign for some external integration.
    """
    native = load_davison_gridded_on_grid(template)
    out = -native
    out.attrs = {
        **native.attrs,
        "units": "m ice yr^-1; Shean convention: negative = melt "
                 "(Davison.tif negated from native positive=melt)",
        "convention": "shean",
    }
    out.name = "davison_basal_melt_shean"
    return out


def davison_window_mean_shean(
    shelf_name: str, start: str, end: str,
) -> tuple[float, float, int]:
    r"""Davison per-shelf window mean in Shean public convention.

    See :func:`davison_window_mean` for parameters. The Davison CSV is
    positive = melt; this wrapper returns ``(-mean, error, n_months)``
    so the mean aligns with the Shean public convention used elsewhere
    in this package. Error is unchanged (a magnitude).
    """
    mean_native, err, n = davison_window_mean(shelf_name, start, end)
    if not (mean_native == mean_native):  # NaN check without numpy import dep
        return mean_native, err, n
    return -mean_native, err, n
