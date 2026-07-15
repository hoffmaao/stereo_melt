# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Correction field container.

The :class:`Corrections` class holds the interpolated BedMachine and
MDT fields on a shared survey grid and provides helpers to populate
them from their respective source datasets.
"""

import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import zoom

from ..geospatial_operations import project_to_latlon, project_to_epsg3031


class Corrections:
    r"""Bundle of interpolated correction fields on a shared ``(x, y)`` grid.

    Attributes
    ----------
    x, y : numpy.ndarray
        1-D survey-grid coordinates.
    geoid, bed_h, thickness, surface_h, bed_source, bed_err, mask, firn
        BedMachine fields interpolated onto the survey grid by
        :meth:`interpolate_bedmachine`.
    z : numpy.ndarray
        Elevation above the WGS84 ellipsoid, derived from surface and
        geoid.
    mdt, mdt_correction : numpy.ndarray
        Mean dynamic topography field and its interpolated correction
        (populated by :meth:`interpolate_mdt`).
    """

    def __init__(self, x, y):
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
        self.z = None  # elevation above ellipsoid
        self.mdt = None  # mean dynamic topography
        self.corrected_heights = None  # MDT-corrected height

    def interpolate_bedmachine(self, bm_data, grid_x, grid_y):
        r"""Interpolate BedMachine fields onto the survey grid.

        Parameters
        ----------
        bm_data : dict
            BedMachine dict from
            :func:`stereo_melt.io.bedmachine.load_bedmachine`.
        grid_x, grid_y : numpy.ndarray
            2-D survey-grid coordinates.
        """
        print(f"x shape: {bm_data['x'].shape}")
        print(f"y shape: {bm_data['y'].shape}")
        print(f"geoid shape: {bm_data['geoid'].shape}")

        x_coords = bm_data["x"]
        y_coords = bm_data["y"]

        def _interp(field):
            return RegularGridInterpolator(
                (y_coords, x_coords), field, bounds_error=False, fill_value=np.nan
            )

        geoid_interpolator = _interp(bm_data["geoid"])
        bed_h_interpolator = _interp(bm_data["bed"])
        thickness_interpolator = _interp(bm_data["thickness"])
        surface_h_interpolator = _interp(bm_data["surface"])
        bed_source_interpolator = _interp(bm_data["source"])
        bed_err_interpolator = _interp(bm_data["errbed"])
        mask_interpolator = _interp(bm_data["mask"])
        firn_interpolator = _interp(bm_data["firn"])

        survey_points = np.column_stack((grid_y.ravel(), grid_x.ravel()))

        self.geoid = geoid_interpolator(survey_points).reshape(grid_x.shape)
        self.bed = bed_h_interpolator(survey_points).reshape(grid_x.shape)
        self.thickness = thickness_interpolator(survey_points).reshape(grid_x.shape)
        self.surface_h = surface_h_interpolator(survey_points).reshape(grid_x.shape)
        self.bed_source = bed_source_interpolator(survey_points).reshape(grid_x.shape)
        self.bed_err = bed_err_interpolator(survey_points).reshape(grid_x.shape)
        self.mask = mask_interpolator(survey_points).reshape(grid_x.shape)
        self.firn = firn_interpolator(survey_points).reshape(grid_x.shape)

        # Inflate to full ice column (firn-corrected) and lift to ellipsoid
        self.thickness += self.firn
        self.surface_h += self.firn
        self.z = self.surface_h + self.geoid

    def mask_labels(self):
        r"""Return the BedMachine ``mask`` integer code legend."""
        return {
            0: "ocean",
            1: "ice-free land",
            2: "grounded ice",
            3: "floating ice",
            4: "Lake Vostok",
        }

    def dtu22_mdt_func(lat, lon, dtu22_data):
        r"""Interpolate DTU22 MDT values onto a ``(lat, lon)`` grid.

        Parameters
        ----------
        lat, lon : numpy.ndarray
            2-D latitude/longitude grids where MDT is required.
        dtu22_data : dict
            Dict with ``lat``, ``lon``, and ``mdt`` arrays.

        Returns
        -------
        numpy.ndarray
            MDT values on the ``(lat, lon)`` grid.
        """
        source_points = np.column_stack((dtu22_data["lon"].ravel(), dtu22_data["lat"].ravel()))
        source_values = dtu22_data["mdt"].ravel()

        target_points = np.column_stack((lon.ravel(), lat.ravel()))

        interpolated_mdt = griddata(
            points=source_points,
            values=source_values,
            xi=target_points,
            method="linear",
        )

        return interpolated_mdt.reshape(lat.shape)

    def interpolate_mdt(self, dtu22_data):
        r"""Interpolate DTU10 MDT onto the survey grid and apply to :attr:`z`.

        Parameters
        ----------
        dtu22_data : dict
            Dict with ``lat``, ``lon``, and ``mdt`` arrays.
        """
        x_resized = zoom(self.x, 0.5)
        y_resized = zoom(self.y, 0.5)
        X, Y = np.meshgrid(x_resized, y_resized)

        lat, lon = project_to_latlon(X, Y)

        source_lon, source_lat = np.meshgrid(dtu22_data["lon"], dtu22_data["lat"], indexing="xy")

        source_points = np.column_stack((source_lon.ravel(), source_lat.ravel()))
        source_values = dtu22_data["mdt"].ravel()

        target_points = np.column_stack((lon.ravel(), lat.ravel()))

        MDT = griddata(
            points=source_points,
            values=source_values,
            xi=target_points,
            method="linear",
        ).reshape(lat.shape)

        grid_x, grid_y = np.meshgrid(self.x, self.y)
        MDT_interp = griddata(
            (X.ravel(), Y.ravel()), MDT.ravel(), (grid_x, grid_y), method="linear"
        )

        MDT_interp[np.isnan(MDT_interp)] = np.nanmean(MDT_interp)

        # Grounded ice (mask == 2) has no MDT correction
        MDT_interp[self.mask == 2] = 0

        self.mdt_correction = MDT_interp
        self.z += self.mdt_correction
