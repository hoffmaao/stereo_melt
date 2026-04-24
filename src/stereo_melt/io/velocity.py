"""Velocity ingestion: MEaSUREs annual interpolation, gap-filling, stack building."""

import os

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter

from .masks import create_polyshape


def measuresann_interp(variable, data_dir, xi, yi, method="linear", inpaint_nans=False):
    """
    Interpolate MEaSUREs dataset for the given variable.

    Parameters:
    - variable (str): 'velocity', 'speed', 'error', or 'count'.
    - xi, yi (array-like): Coordinates in polar stereographic meters.
    - data_dir (str): Directory containing MEaSUREs .nc files.
    - method (str): Interpolation method ('linear', 'nearest'). Default is 'linear'.
    - inpaint_nans (bool): Whether to fill missing values using griddata. Default is False.

    Returns:
    - (vx, vy, t) for 'velocity' or 'error', or (v, t) for 'speed' or 'count'.
    """
    # Supported variables and corresponding NetCDF variable names
    variable_mapping = {
        "velocity": ("vx", "vy"),
        "error": ("vx_error", "vy_error"),
    }

    # Check if the variable is valid
    if variable.lower() not in variable_mapping:
        raise ValueError(f"Invalid variable '{variable}'. Choose from {list(variable_mapping.keys())}.")

    # List all NetCDF files in the data directory
    file_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")])
    if not file_list:
        raise FileNotFoundError(f"No NetCDF files found in {data_dir}.")

    # Determine the variable names to read
    vars_to_read = variable_mapping[variable.lower()]
    is_velocity = len(vars_to_read) == 2  # True for 'velocity' and 'error'

    # Initialize storage for interpolated values
    values_x, values_y = [], []
    t_values = np.arange(2018, 2021)

    for file in file_list:
        # Open the NetCDF file
        with xr.open_dataset(file) as ds:
            # Get spatial and temporal dimensions
            x = ds["x"].values
            y = ds["y"].values

            # Read the variables
            data_x = ds[vars_to_read[0]].values
            if is_velocity:
                data_y = ds[vars_to_read[1]].values

            # Interpolate using RegularGridInterpolator
            interpolator_x = RegularGridInterpolator((y, x), data_x, method=method, bounds_error=False, fill_value=np.nan)
            interpolated_x = interpolator_x(np.column_stack((yi.ravel(), xi.ravel()))).reshape(xi.shape)

            if is_velocity:
                interpolator_y = RegularGridInterpolator((y, x), data_y, method=method, bounds_error=False, fill_value=np.nan)
                interpolated_y = interpolator_y(np.column_stack((yi.ravel(), xi.ravel()))).reshape(xi.shape)

            # Handle missing values if requested
            if inpaint_nans:
                interpolated_x = _inpaint_nans(interpolated_x)
                if is_velocity:
                    interpolated_y = _inpaint_nans(interpolated_y)

            # Store interpolated values
            values_x.append(interpolated_x)
            if is_velocity:
                values_y.append(interpolated_y)

    # Combine results into numpy arrays
    values_x = np.stack(values_x, axis=-1)
    if is_velocity:
        values_y = np.stack(values_y, axis=-1)

    # Return results based on the variable type
    t_values = np.array(t_values).flatten()
    if variable.lower() == "speed":
        return np.sqrt(values_x**2 + values_y**2), t_values
    elif variable.lower() in ["velocity", "error"]:
        return values_x, values_y, t_values
    elif variable.lower() == "count":
        return values_x, t_values


def _inpaint_nans(data):
    """
    Fill NaN values in a 2D or 3D array using griddata interpolation.

    Parameters:
    - data (ndarray): Input array with NaN values.

    Returns:
    - ndarray: Array with NaNs filled.
    """
    coords = np.array(np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"))
    valid_mask = ~np.isnan(data)

    # Interpolate for each time step if 3D
    if data.ndim == 3:
        for i in range(data.shape[2]):
            time_slice = data[:, :, i]
            data[:, :, i] = griddata(
                coords[:, valid_mask[:, :, i]].T,
                time_slice[valid_mask[:, :, i]],
                coords.reshape(2, -1).T,
                method="linear",
            ).reshape(data.shape[:2])
    else:
        data = griddata(
            coords[:, valid_mask].T,
            data[valid_mask],
            coords.reshape(2, -1).T,
            method="linear",
        ).reshape(data.shape)

    return data


# TODO add ability to include other velocity products.

def create_velocity_stack(x, y, vel_dir, MYrs, std_thr=2.5, QAnn=None, QYrs=None):
    """
    Create a velocity stack using MEaSUREs annual velocities and optionally additional velocity fields (QAnn).

    Parameters:
    - x, y (1D arrays): Original grid coordinates.
    - MYrs (1D array): Years corresponding to MEaSUREs annual velocities.
    - QAnn (dict, optional): Dictionary containing additional velocity fields (vx, vy). Defaults to None.
    - QYrs (1D array, optional): Years corresponding to QAnn velocity fields. Defaults to None.

    Returns:
    - VelAnn (dict): Dictionary containing stacked and filtered velocity data.
    """
    # Define new grid and map info
    XB = [min(x) - 10000, max(x) + 10000]  # Extend bounds by 10 km
    YB = [min(y) - 10000, max(y) + 10000]
    dx = abs(x[1] - x[0])  # Grid resolution
    x_new = np.arange(XB[0], XB[1] + dx, dx)
    y_new = np.arange(YB[0], YB[1] + dx, dx)

    # Create grid
    xg, yg = np.meshgrid(x_new, y_new)

    # Interpolate MEaSUREs annual velocities
    MeasAnn_vx, MeasAnn_vy, MeasAnn_t = measuresann_interp("velocity", vel_dir, xg, yg, method="linear", inpaint_nans=True)

    # Initialize VelAnn dictionary
    VelAnn = {
        "x": x_new,
        "y": y_new,
        "vx": [],
        "vy": [],
        "Years": [],
    }

    # Match MEaSUREs data to MYrs
    Mapidx = np.argmin(np.abs(MeasAnn_t - np.array(MYrs)[:, None]), axis=1)
    print(MeasAnn_vx)
    print(Mapidx)
    MeasAnn_vx = MeasAnn_vx[:, :, Mapidx]
    MeasAnn_vy = MeasAnn_vy[:, :, Mapidx]
    VelAnn["Years"].extend(MYrs)

    # Stack additional QAnn velocity data if provided
    if QAnn and QYrs:
        VelAnn["vx"] = np.concatenate((MeasAnn_vx, QAnn["vx"]), axis=-1)
        VelAnn["vy"] = np.concatenate((MeasAnn_vy, QAnn["vy"]), axis=-1)
        VelAnn["Years"].extend(QYrs)
    else:
        VelAnn["vx"] = MeasAnn_vx
        VelAnn["vy"] = MeasAnn_vy

    # Calculate statistics
    VelAnn["vx_mean"] = np.nanmean(VelAnn["vx"], axis=-1)
    VelAnn["vy_mean"] = np.nanmean(VelAnn["vy"], axis=-1)
    VelAnn["vx_med"] = np.nanmedian(VelAnn["vx"], axis=-1)
    VelAnn["vy_med"] = np.nanmedian(VelAnn["vy"], axis=-1)
    VelAnn["vx_std"] = np.nanstd(VelAnn["vx"], axis=-1)
    VelAnn["vy_std"] = np.nanstd(VelAnn["vy"], axis=-1)

    # Filter and smooth velocities
    for ii in range(VelAnn["vx"].shape[-1]):
        # Process vx
        vx_tmp = VelAnn["vx"][:, :, ii]
        mask_vx = (vx_tmp > VelAnn["vx_mean"] + std_thr * VelAnn["vx_std"]) | \
                  (vx_tmp < VelAnn["vx_mean"] - std_thr * VelAnn["vx_std"])
        vx_tmp[mask_vx | np.isnan(vx_tmp)] = VelAnn["vx_mean"][mask_vx | np.isnan(vx_tmp)]
        VelAnn["vx"][:, :, ii] = gaussian_filter(vx_tmp, sigma=30)

        # Process vy
        vy_tmp = VelAnn["vy"][:, :, ii]
        mask_vy = (vy_tmp > VelAnn["vy_mean"] + std_thr * VelAnn["vy_std"]) | \
                  (vy_tmp < VelAnn["vy_mean"] - std_thr * VelAnn["vy_std"])
        vy_tmp[mask_vy | np.isnan(vy_tmp)] = VelAnn["vy_mean"][mask_vy | np.isnan(vy_tmp)]
        VelAnn["vy"][:, :, ii] = gaussian_filter(vy_tmp, sigma=30)

    # Calculate velocity magnitude
    VelAnn["vMag"] = np.sqrt(VelAnn["vx"]**2 + VelAnn["vy"]**2)
    VelAnn["vMag_mean"] = np.nanmean(VelAnn["vMag"], axis=-1)
    VelAnn["vMag_med"] = np.nanmedian(VelAnn["vMag"], axis=-1)

    # Gradient and divergence calculations
    dx = np.diff(x_new)[0]
    dy = np.diff(y_new)[0]
    VelAnn["grad_vx"] = np.gradient(VelAnn["vx"], dx, axis=0)
    VelAnn["grad_vy"] = np.gradient(VelAnn["vy"], dy, axis=1)
    VelAnn["vdiv_stack"] = VelAnn["grad_vx"] + VelAnn["grad_vy"]

    return VelAnn


def compute_gradients_and_masks(VelAnn, Corrections):
    """
    Compute velocity gradients and create masks.

    Parameters:
    - VelAnn: Stacked velocity data.
    - Corrections: Dictionary containing surface height and mask data.

    Returns:
    - VelAnn: Updated velocity data with gradients and velocity magnitude.
    """
    dx = np.diff(VelAnn['x'])[0]
    dy = np.diff(VelAnn['y'])[0]

    # Calculate velocity gradients
    VelAnn['grad_vx'], _ = np.gradient(VelAnn['vx'], dx, axis=0)
    _, VelAnn['grad_vy'] = np.gradient(VelAnn['vy'], dy, axis=1)
    VelAnn['vdiv_stack'] = VelAnn['grad_vx'] + VelAnn['grad_vy']

    # Velocity magnitude
    VelAnn['vMag'] = np.sqrt(VelAnn['vx']**2 + VelAnn['vy']**2)
    VelAnn['vMag_mean'] = np.nanmean(VelAnn['vMag'], axis=-1)
    VelAnn['vMag_med'] = np.nanmedian(VelAnn['vMag'], axis=-1)

    # Create masks
    Corrections['rockPoly'] = create_polyshape(Corrections['mask'] == 1, Corrections['x'], Corrections['y'])
    return VelAnn
