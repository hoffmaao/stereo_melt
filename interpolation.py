from scipy.interpolate import griddata
import numpy as np

def interpolate_data(grid_x, grid_y, data_x, data_y, data_values, method="linear"):
    """
    Interpolate data onto a regular grid.

    Parameters:
    - grid_x, grid_y: The target grid coordinates (meshgrid format).
    - data_x, data_y: Original data point coordinates.
    - data_values: Original data values at those coordinates.
    - method: Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
    - interpolated_data: Data interpolated onto the grid.
    """
    points = np.array([data_x, data_y]).T  # Combine data_x and data_y into pairs
    grid_points = np.array([grid_x.ravel(), grid_y.ravel()]).T
    interpolated_data = griddata(points, data_values, grid_points, method=method)

    return interpolated_data.reshape(grid_x.shape)