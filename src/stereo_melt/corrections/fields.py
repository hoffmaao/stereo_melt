# corrections.py (continued)
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import zoom
from ..geospatial_operations import project_to_latlon, project_to_epsg3031


class Corrections:
    def __init__(self, x, y):
        """
        Initialize Corrections object with grid coordinates.
        """
        self.x = x
        self.y = y
        self.geoid = None
        self.bed_h = None
        self.thickness = None
        self.surface_h = None
        self.bed_source = None
        self.bed_err = None
        self.mask = None
        self.firn = None
        self.z = None  # Elevation above ellipsoid
        self.mdt = None  # Mean dynamic topography
        self.corrected_heights = None  # MDT-corrected height


    def interpolate_bedmachine(self, bm_data, grid_x, grid_y):
        """
        Interpolate BedMachine data onto the survey grid using RegularGridInterpolator.

        Parameters:
        - bm_data: Dictionary containing BedMachine data (x, y, geoid, bed_z, etc.).
        - grid_x, grid_y: Grid coordinates for the survey area.
        """
        # Print data shapes for debugging
        print(f"x shape: {bm_data['x'].shape}")
        print(f"y shape: {bm_data['y'].shape}")
        print(f"geoid shape: {bm_data['geoid'].shape}")

        # Ensure bm_data['x'] and bm_data['y'] are 1D arrays
        x_coords = bm_data["x"]
        y_coords = bm_data["y"]

        # Create interpolators for each field
        geoid_interpolator = RegularGridInterpolator((y_coords, x_coords), bm_data["geoid"], bounds_error=False, fill_value=np.nan)
        bed_h_interpolator = RegularGridInterpolator((y_coords, x_coords), bm_data["bed"], bounds_error=False, fill_value=np.nan)
        thickness_interpolator = RegularGridInterpolator((y_coords, x_coords), bm_data["thickness"], bounds_error=False, fill_value=np.nan)
        surface_h_interpolator = RegularGridInterpolator((y_coords, x_coords), bm_data["surface"], bounds_error=False, fill_value=np.nan)
        bed_source_interpolator = RegularGridInterpolator((y_coords, x_coords), bm_data["source"], bounds_error=False, fill_value=np.nan)
        bed_err_interpolator = RegularGridInterpolator((y_coords, x_coords), bm_data["errbed"], bounds_error=False, fill_value=np.nan)
        mask_interpolator = RegularGridInterpolator((y_coords, x_coords), bm_data["mask"], bounds_error=False, fill_value=np.nan)
        firn_interpolator = RegularGridInterpolator((y_coords, x_coords), bm_data["firn"], bounds_error=False, fill_value=np.nan)

        # Create survey points for interpolation
        survey_points = np.column_stack((grid_y.ravel(), grid_x.ravel()))

        # Interpolate data
        self.geoid = geoid_interpolator(survey_points).reshape(grid_x.shape)
        self.bed = bed_h_interpolator(survey_points).reshape(grid_x.shape)
        self.thickness = thickness_interpolator(survey_points).reshape(grid_x.shape)
        self.surface_h = surface_h_interpolator(survey_points).reshape(grid_x.shape)
        self.bed_source = bed_source_interpolator(survey_points).reshape(grid_x.shape)
        self.bed_err = bed_err_interpolator(survey_points).reshape(grid_x.shape)
        self.mask = mask_interpolator(survey_points).reshape(grid_x.shape)
        self.firn = firn_interpolator(survey_points).reshape(grid_x.shape)

        # Update full column thickness and surface height
        self.thickness += self.firn
        self.surface_h += self.firn

        # Calculate elevation above ellipsoid
        self.z = self.surface_h + self.geoid


    def mask_labels(self):
        """
        Provide mask labels for reference.
        """
        return {
            0: "ocean",
            1: "ice-free land",
            2: "grounded ice",
            3: "floating ice",
            4: "Lake Vostok"
        }

    def dtu22_mdt_func(lat, lon, dtu22_data):
        """
        Interpolate DTU22 MDT values onto a given latitude/longitude grid.

        Parameters:
        - lat (2D array): Array of latitude values where MDT is needed.
        - lon (2D array): Array of longitude values where MDT is needed.
        - dtu22_data (dict): Dictionary containing 'lat', 'lon', and 'mdt' arrays.

        Returns:
        - interpolated_mdt (2D array): Interpolated MDT values on the provided grid.
        """
        # Flatten the source data for interpolation
        source_points = np.column_stack((dtu22_data["lon"].ravel(), dtu22_data["lat"].ravel()))
        source_values = dtu22_data["mdt"].ravel()

        # Flatten target grid for interpolation
        target_points = np.column_stack((lon.ravel(), lat.ravel()))

        # Perform interpolation
        interpolated_mdt = griddata(
            points=source_points,
            values=source_values,
            xi=target_points,
            method="linear",
        )

        # Reshape the output to match the target grid shape
        return interpolated_mdt.reshape(lat.shape)

    def interpolate_mdt(self, dtu22_data):
        """
        Interpolate DTU10 MDT values onto the grid (self.grid_x, self.grid_y).
        """
        x_resized = zoom(self.x,0.5)
        y_resized = zoom(self.y,0.5)
        # Create meshgrid
        X, Y = np.meshgrid(x_resized, y_resized)

        # Convert to Latitude and Longitude
        lat, lon = project_to_latlon(X, Y)

        source_lon, source_lat = np.meshgrid(dtu22_data["lon"], dtu22_data["lat"], indexing="xy")

        source_points = np.column_stack((source_lon.ravel(), source_lat.ravel()))
        source_values = dtu22_data["mdt"].ravel()

        # Flatten target grid for interpolation
        target_points = np.column_stack((lon.ravel(), lat.ravel()))

        # Perform interpolation
        MDT = griddata(
            points=source_points,
            values=source_values,
            xi=target_points,
            method="linear",
        ).reshape(lat.shape)

        # Interpolate MDT back to the original grid
        grid_x, grid_y = np.meshgrid(self.x, self.y)
        MDT_interp = griddata((X.ravel(), Y.ravel()), MDT.ravel(), (grid_x, grid_y), method="linear")

        # Handle missing values
        MDT_interp[np.isnan(MDT_interp)] = np.nanmean(MDT_interp)

        # Apply mask to set grounded ice (mask == 2) MDT to 0
        MDT_interp[self.mask == 2] = 0

        # Update corrections dictionary
        self.mdt_correction = MDT_interp
        self.z += self.mdt_correction
