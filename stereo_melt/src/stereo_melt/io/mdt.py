# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Mean Dynamic Topography loader (DTU22 ASCII/GRAVSOFT XYZ format)."""

import numpy as np
import pandas as pd


def load_gravsoft_xyz(file_path):
    r"""Load a GRAVSOFT ASCII XYZ grid and return a gridded dict.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the whitespace-delimited ``lon lat mdt err`` file.

    Returns
    -------
    dict
        ``{"lon": 1-D array, "lat": 1-D array, "mdt": 2-D array}`` where
        ``mdt[i, j]`` corresponds to ``lat[i], lon[j]``.
    """
    try:
        data = pd.read_csv(
            file_path,
            delim_whitespace=True,
            header=None,
            names=["lon", "lat", "mdt", "err"],
        )
        lons = np.unique(data["lon"])
        lats = np.unique(data["lat"])
        mdt = data["mdt"].values.reshape(len(lats), len(lons))
        return {"lon": lons, "lat": lats, "mdt": mdt}
    except Exception as e:
        raise ValueError(f"Error loading GRAVSOFT grid file: {e}")


def load_dtu10_mdt(file_path):
    r"""Load DTU10 Mean Dynamic Topography in GRAVSOFT XYZ format.

    Thin wrapper around :func:`load_gravsoft_xyz`; see that function for
    the returned dict layout.
    """
    return load_gravsoft_xyz(file_path)
