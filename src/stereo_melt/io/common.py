# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Generic file loaders used across the pipeline."""

import numpy as np
import rasterio
import scipy.io


def load_mat_file(file_path):
    r"""Return the contents of a MATLAB ``.mat`` file as a dict."""
    return scipy.io.loadmat(file_path)


def load_geotiff(file_path):
    r"""Load a single-band GeoTIFF and return ``(data, metadata)``.

    Pixels with values below ``-1000`` are replaced with NaN to drop the
    conventional nodata sentinels used by REMA and BedMachine products.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the GeoTIFF.

    Returns
    -------
    tuple
        ``(numpy.ndarray, dict)`` — the first band's data and the rasterio
        metadata dictionary.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        data[data < -1e3] = np.nan
        metadata = src.meta
    return data, metadata
