"""Tidal correction via pyTMD."""

import numpy as np
from pyproj import Proj, transform
from pyTMD.compute import tide_elevations


def apply_tidal_correction(xs, ys, center_time, model="CATS2008"):
    """Apply tidal correction to the DEM strip using pyTMD."""
    print("🌊 Applying Tidal Correction with pyTMD...")

    # Convert polar stereographic (EPSG:3031) to geographic (EPSG:4326)
    proj_polar = Proj("epsg:3031")
    proj_geo = Proj(proj="latlong", datum="WGS84")
    lon, lat = transform(proj_polar, proj_geo, xs, ys)

    # Convert time to seconds since 2000-01-01T00:00:00
    tide_dates = np.datetime64(center_time)
    delta_time = (tide_dates - np.datetime64("2000-01-01T00:00:00")) / np.timedelta64(1, "s")

    # Compute tidal elevations
    tide_h = tide_elevations(
        lon, lat, delta_time,
        MODEL=model, EPOCH=(2000, 1, 1, 0, 0, 0),
        EPSG=3031, TYPE="grid"
    )

    print(f"   📊 Tidal correction applied: min={np.nanmin(tide_h):.2f}, max={np.nanmax(tide_h):.2f}")

    return tide_h
