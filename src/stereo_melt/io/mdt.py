"""Mean Dynamic Topography loader (DTU22 ASCII/GRAVSOFT XYZ format)."""

import numpy as np
import pandas as pd


def load_gravsoft_xyz(file_path):
    """
    Load GRAVSOFT grid data in ASCII XYZ format.

    Parameters:
    - file_path (str): Path to the ASCII XYZ file.

    Returns:
    - dict: A dictionary with 'lon', 'lat', and 'mdt' arrays.
    """
    try:
        # Read the ASCII file into a DataFrame
        data = pd.read_csv(
            file_path,
            delim_whitespace=True,
            header=None,
            names=["lon", "lat", "mdt", "err"]
        )
        # Extract unique latitude and longitude
        lons = np.unique(data["lon"])
        lats = np.unique(data["lat"])

        # Reshape MDT values to a 2D grid
        mdt = data["mdt"].values.reshape(len(lats), len(lons))

        return {
            "lon": lons,
            "lat": lats,
            "mdt": mdt,
        }
    except Exception as e:
        raise ValueError(f"Error loading GRAVSOFT grid file: {e}")


def load_dtu10_mdt(file_path):
    """
    Load DTU10 Mean Dynamic Topography (MDT) data.

    Parameters:
    - file_path: Path to the DTU10 MDT NetCDF file.
    """
    return load_gravsoft_xyz(file_path)
