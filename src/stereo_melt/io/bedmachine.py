"""BedMachine Antarctica loader (bed, thickness, surface, geoid, firn, mask)."""

import h5py


def load_bedmachine(file_path):
    """
    Load BedMachine data from an HDF5 file.

    Parameters:
    - file_path: Path to the HDF5 file.

    Returns:
    - dict: A dictionary containing relevant datasets.
    """
    with h5py.File(file_path, 'r') as f:
        bedmachine_data = {
            "x": f["x"][:],
            "y": f["y"][:],
            "geoid": f["geoid"][:],
            "bed": f["bed"][:],
            "thickness": f["thickness"][:],
            "surface": f["surface"][:],
            "source": f["source"][:],
            "errbed": f["errbed"][:],
            "mask": f["mask"][:],
            "firn": f["firn"][:],
        }
    return bedmachine_data
