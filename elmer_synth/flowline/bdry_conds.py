#-------------------------------------------------------------------------------
# This file contains functions that:
# (1) define the boundaries of the mesh (ice-air, ice-water, inflow, front)
# (2) mark the boundaries of the mesh.
# Forked from vendor/linear-shelf-melt/nonlinear-model/bdry_conds.py.
# E1b deltas: locators are DISJOINT (a facet belongs to exactly one set) so the
# open outflow front receives water pressure, the inflow wall receives the
# velocity Dirichlet, and the kinematic displacement BCs act only on interior
# top/bottom surface facets.
#-------------------------------------------------------------------------------
import numpy as np
from dolfinx.mesh import locate_entities, meshtags
from params import L, z_max

TOL = 1.0   # m; drops wall-adjacent corner facets from the surface/water sets


Z_TOL = 40.0  # m; < cell height (100 m) so wall facets stay excluded


def WaterBoundary(x):
    # ice-water pressure boundary: basal surface + front face below waterline.
    # Inflow-wall FACETS are excluded (velocity Dirichlet lives there), but the
    # wall-adjacent BASAL facet must keep its water pressure: with it dropped,
    # a 100 m strip at the inflow corner loses ~4.5 MPa of buoyant support and
    # collapses at ~1000 m/yr, wrecking the mesh in one step (07-13 debug).
    # The z<Z_TOL branch readmits facets lying on the (near-z=0) base while
    # still rejecting wall facets, whose off-corner vertex has z >= 100 m.
    return np.logical_and(np.less(x[1], z_max),
                          np.logical_or(np.greater(x[0], -L/2.0 + TOL),
                                        np.less(x[1], Z_TOL)))

def BottomSurface(x):
    # basal surface only (kinematic displacement BC); excludes both walls
    return np.logical_and(np.less(x[1], z_max),
                          np.logical_and(np.greater(x[0], -L/2.0 + TOL),
                                         np.less(x[0], L/2.0 - TOL)))

def TopBoundary(x):
    # ice-air surface (kinematic displacement BC); excludes both walls
    return np.logical_and(np.greater(x[1], z_max),
                          np.logical_and(np.greater(x[0], -L/2.0 + TOL),
                                         np.less(x[0], L/2.0 - TOL)))

def LeftBoundary(x):
    # inflow wall
    return np.isclose(x[0], -L/2.0)

def RightBoundary(x):
    # front face above the waterline (traction-free)
    return np.logical_and(np.isclose(x[0], L/2.0), np.greater(x[1], z_max))

#-------------------------------------------------------------------------------

def mark_boundary(domain):
    # Assign markers to each boundary segment.
    #
    # Boundary marker numbering convention (upstream-compatible):
    # 1 - inflow wall, 2 - front above waterline, 3 - ice-water, 4 - top

    boundaries = [(3, WaterBoundary), (4, TopBoundary), (1, LeftBoundary), (2, RightBoundary)]
    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    return facet_tag
