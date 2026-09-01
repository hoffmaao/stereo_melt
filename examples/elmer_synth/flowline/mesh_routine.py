#-------------------------------------------------------------------------------------
# These functions are used to:
# (1) move the mesh at each timestep according the solution and forcings (move_mesh),
# (2) retrieve numpy arrays of the upper and lower surface elevations (get_surfaces)
# (3) retrieve per-column surface/basal velocities (get_surface_fields)
# Forked from vendor/linear-shelf-melt/nonlinear-model/mesh_routine.py.
# E1b deltas: displacement Dirichlet acts on the interior top/bottom surface
# facets only; the inflow wall is clamped (disp = 0, constant feed geometry);
# the open front is Laplace-extended. Mesh motion stays vertical-only, so the
# domain is an Eulerian window: ice flows through fixed [-L/2, L/2].
#-------------------------------------------------------------------------------------

import numpy as np
from bdry_conds import BottomSurface, LeftBoundary, TopBoundary
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from params import dt
from petsc4py.PETSc import ScalarType
from ufl import TestFunction, TrialFunction, dx, grad, inner


def move_mesh(w,domain,t,smb_h,smb_s):
    # this function computes the surface displacements and moves the mesh
    # by solving Laplace's equation for a smooth displacement function
    # defined for all mesh vertices

    V = FunctionSpace(domain, ("CG", 1))
    fdim = domain.topology.dim-1

    facets_b = locate_entities_boundary(domain, fdim, BottomSurface)
    facets_t = locate_entities_boundary(domain, fdim, TopBoundary)
    facets_l = locate_entities_boundary(domain, fdim, LeftBoundary)

    # Kinematic surface displacement, built per vertex column in numpy with an
    # UPWIND slope (u > 0 everywhere): dz/dt = w - u*z_x + forcing. The
    # previous centered/projected slope made the explicit update FTCS —
    # unconditionally unstable under through-flow advection; the front columns
    # (u ~ 1400 m/yr, local CFL 1.3 at dt=0.1 yr) blew up in ~10 steps
    # (07-13 debug). First-order upwind is monotone for CFL = u*dt/dx < 1
    # (dt = 0.05 yr -> CFL ~ 0.7 at the front). Its numerical diffusion
    # kappa ~ u*dx/2 smears transients but perturbs the quasi-steady
    # melt-carved topography by only ~1/Pe ~ 2% (Pe = u*sigma/kappa ~ 55).
    u_f = Function(V); w_f = Function(V)
    u_f.interpolate(Expression(w.sub(0).sub(0), V.element.interpolation_points()))
    w_f.interpolate(Expression(w.sub(0).sub(1), V.element.interpolation_points()))
    uv = u_f.x.array; wv = w_f.x.array

    Xd = V.tabulate_dof_coordinates()
    xd = Xd[:,0]; zd = Xd[:,1]
    x_u, col = np.unique(np.round(xd, 6), return_inverse=True)
    ncol = x_u.size
    jt = np.zeros(ncol, dtype=np.int64); jb = np.zeros(ncol, dtype=np.int64)
    for i in range(ncol):
        cols = np.where(col == i)[0]
        jt[i] = cols[np.argmax(zd[cols])]
        jb[i] = cols[np.argmin(zd[cols])]

    h = zd[jt]; s = zd[jb]
    hx = np.zeros(ncol); sx = np.zeros(ncol)
    hx[1:] = np.diff(h)/np.diff(x_u)   # upwind: ice always arrives from -x
    sx[1:] = np.diff(s)/np.diff(x_u)

    disp_top = dt*(wv[jt] - uv[jt]*hx + smb_h(x_u, t))
    disp_bot = dt*(wv[jb] - uv[jb]*sx + smb_s(x_u, t))

    disp_h_fcn = Function(V)
    disp_s_fcn = Function(V)
    disp_h_fcn.x.array[jt] = disp_top
    disp_s_fcn.x.array[jb] = disp_bot
    dofs_b = locate_dofs_topological(V, domain.topology.dim-1, facets_b)
    dofs_t = locate_dofs_topological(V, domain.topology.dim-1, facets_t)
    dofs_l = locate_dofs_topological(V, domain.topology.dim-1, facets_l)

    # displacement BCs: kinematic on interior top/bottom surfaces, zero on the
    # inflow wall (constant-geometry ice feed); front face Laplace-extended
    bc1 = dirichletbc(disp_s_fcn, dofs_b)
    bc2 = dirichletbc(disp_h_fcn, dofs_t)
    bc3 = dirichletbc(ScalarType(0.0), dofs_l, V)

    bcs = [bc1,bc2,bc3]

    # # solve Laplace's equation for a smooth displacement field on all vertices,
    # # given the boundary displacement disp_bdry
    disp = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(disp), grad(v))*dx
    f = Constant(domain, ScalarType(0.0))
    L = f*v*dx

    problem = LinearProblem(a,L, bcs=bcs)
    sol = problem.solve()

    disp_vv = sol.x.array

    X = domain.geometry.x

    X[:,1] += disp_vv

    return domain


def get_surfaces(domain):
# retrieve numpy arrays of the upper and lower surface elevations
    X = domain.geometry.x
    x = np.sort(X[:,0])
    z = X[:,1][np.argsort(X[:,0])]
    x_u = np.unique(X[:,0])
    h = np.zeros(x_u.size)      # upper surface elevation
    s = np.zeros(x_u.size)      # lower surface elevation

    for i in range(h.size):
        h[i] = np.max(z[np.where(np.isclose(x_u[i],x))])
        s[i] = np.min(z[np.where(np.isclose(x_u[i],x))])

    return h,s,x_u


def get_surface_fields(sol,domain):
# retrieve per-column horizontal/vertical velocities at the upper and lower
# surfaces (CG1 interpolation, then top/bottom of each vertex column)
    V = FunctionSpace(domain, ("CG", 1))
    u_f = Function(V)
    w_f = Function(V)
    u_f.interpolate(Expression(sol.sub(0).sub(0), V.element.interpolation_points()))
    w_f.interpolate(Expression(sol.sub(0).sub(1), V.element.interpolation_points()))
    u = u_f.x.array
    w = w_f.x.array
    X = V.tabulate_dof_coordinates()
    x = X[:,0]
    z = X[:,1]
    x_u = np.unique(np.round(x, 6))
    u_top = np.zeros(x_u.size); w_top = np.zeros(x_u.size)
    u_bot = np.zeros(x_u.size); w_bot = np.zeros(x_u.size)
    for i in range(x_u.size):
        cols = np.where(np.isclose(x, x_u[i]))[0]
        jt = cols[np.argmax(z[cols])]
        jb = cols[np.argmin(z[cols])]
        u_top[i], w_top[i] = u[jt], w[jt]
        u_bot[i], w_bot[i] = u[jb], w[jb]
    return u_top,w_top,u_bot,w_bot,x_u
