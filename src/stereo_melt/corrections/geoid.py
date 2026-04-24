"""Geoid correction: convert WGS84-ellipsoid REMA DEMs to orthometric (geoid-referenced)
heights using BedMachine's geoid undulation field.

Primary path is pdemtools' ``dem.pdt.geoid_correct(geoid)`` for xarray DataArrays.
A numpy path is provided for the flat-array pipeline flow.

Sign convention (Shean 2019 Eq. 1, and pdemtools):
    h_orthometric = h_ellipsoid - N,  where N is the geoid undulation
(positive-up height of geoid above WGS84 ellipsoid). BedMachine stores N
in the ``geoid`` variable.
"""

import xarray as xr
import rioxarray  # noqa: F401 - registers .rio accessor
import pdemtools  # noqa: F401 - registers .pdt accessor via DemAccessor

__all__ = [
    "load_bedmachine_geoid",
    "apply_geoid_correction_xr",
    "apply_geoid_correction_np",
]


def load_bedmachine_geoid(bm_path: str) -> xr.DataArray:
    """Load the BedMachine geoid field as an EPSG:3031-georeferenced DataArray."""
    ds = xr.open_dataset(bm_path)
    geoid = ds["geoid"]
    return geoid.rio.write_crs("EPSG:3031", inplace=False)


def apply_geoid_correction_xr(dem: xr.DataArray, bm_path: str) -> xr.DataArray:
    """Apply geoid correction to a DEM DataArray via pdemtools.

    Parameters
    ----------
    dem : xr.DataArray
        Input DEM as ellipsoidal height (REMA is WGS84). Must carry rio CRS.
    bm_path : str
        Path to BedMachine NetCDF.

    Returns
    -------
    xr.DataArray
        Orthometric (geoid-referenced) DEM.
    """
    geoid = load_bedmachine_geoid(bm_path)
    return dem.pdt.geoid_correct(geoid)


def apply_geoid_correction_np(dem_flat, xs, ys, bedmachine_data):
    """Apply geoid correction to a flat numpy DEM using the BedMachine geoid.

    This is the numpy equivalent of :func:`apply_geoid_correction_xr`, used by
    the legacy flat-array correction path. Prefer the xarray version for new
    code.

    Parameters
    ----------
    dem_flat : np.ndarray
        Flat DEM values (height above WGS84 ellipsoid).
    xs, ys : np.ndarray
        Flat coordinate arrays in EPSG:3031.
    bedmachine_data : dict
        Result of :func:`stereo_melt.io.bedmachine.load_bedmachine`.

    Returns
    -------
    np.ndarray
        Orthometric (geoid-subtracted) DEM values.
    """
    from scipy.interpolate import RegularGridInterpolator

    geoid_interp = RegularGridInterpolator(
        (bedmachine_data["y"], bedmachine_data["x"]),
        bedmachine_data["geoid"],
        bounds_error=False,
        fill_value=0,
    )
    geoid_N = geoid_interp((ys, xs))
    return dem_flat - geoid_N
