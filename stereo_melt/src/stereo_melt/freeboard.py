# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Hydrostatic freeboard ⇄ ice thickness conversions.

For a freely-floating ice shelf in hydrostatic equilibrium with surface
elevation :math:`h`, firn air content :math:`d`, and densities
:math:`\rho_w, \rho_i`, the ice-equivalent thickness :math:`H_f` is

.. math::
    H_f = (h - d)\,\frac{\rho_w}{\rho_w - \rho_i}

This is Shean 2019 Eq. 8. The inverse mapping is

.. math::
    h = H_f\,\frac{\rho_w - \rho_i}{\rho_w} + d

For a grounded-to-floating transition, the hydrostatic assumption fails
and an :math:`\alpha` ramp over the grounding zone must be applied
upstream (see ``corrections/gz_ramp.py``).
"""

from __future__ import annotations

from .constants import rhoi, rhow

__all__ = ["freeboard_to_thickness", "thickness_to_freeboard"]


def freeboard_to_thickness(h, d=0.0, rho_w: float = rhow, rho_i: float = rhoi):
    r"""Return the ice-equivalent thickness of a freely-floating shelf.

    .. math::
        H_f = (h - d)\,\frac{\rho_w}{\rho_w - \rho_i}

    Parameters
    ----------
    h
        Geoid-referenced surface elevation in meters. Scalar, numpy array,
        or xarray DataArray.
    d
        Firn air content in meters, broadcastable against ``h``. Defaults
        to 0 (pure ice).
    rho_w, rho_i
        Seawater and ice densities in kg m\ :sup:`-3`.

    Returns
    -------
    Same type as ``h``
        Ice-equivalent thickness in meters.
    """
    return (h - d) * rho_w / (rho_w - rho_i)


def thickness_to_freeboard(H_f, d=0.0, rho_w: float = rhow, rho_i: float = rhoi):
    r"""Return the surface elevation of a freely-floating shelf.

    Inverse of :func:`freeboard_to_thickness`:

    .. math::
        h = H_f\,\frac{\rho_w - \rho_i}{\rho_w} + d

    Parameters
    ----------
    H_f
        Ice-equivalent thickness in meters.
    d
        Firn air content in meters. Defaults to 0.
    rho_w, rho_i
        Seawater and ice densities in kg m\ :sup:`-3`.

    Returns
    -------
    Same type as ``H_f``
        Geoid-referenced surface elevation in meters.
    """
    return H_f * (rho_w - rho_i) / rho_w + d
