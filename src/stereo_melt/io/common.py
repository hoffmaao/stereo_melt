"""Generic file loaders."""

import numpy as np
import rasterio
import scipy.io


def load_mat_file(file_path):
    """Load a .mat file and return its content."""
    data = scipy.io.loadmat(file_path)
    return data


def load_geotiff(file_path):
    """Load a GeoTIFF file and return its metadata and data."""
    with rasterio.open(file_path) as src:
        data = src.read(1)
        data[data < -1e3] = np.nan  # Apply the NaN mask
        metadata = src.meta
    return data, metadata
