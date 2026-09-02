# This file contains the functions needed for solving the nonlinear Stokes problem.
# Forked from vendor/linear-shelf-melt/nonlinear-model/stokes.py. E1b deltas:
# inflow Dirichlet u_x = u0 on the left wall; the right wall is an OPEN front
# (water pressure below the waterline via ds(3), traction-free above) with no
# velocity constraint; hardness carries the spatial prefactor hook bfac(x).
import os

from bdry_conds import LeftBoundary, mark_boundary
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc,
                         locate_dofs_topological)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.log import LogLevel, set_log_level
from dolfinx.mesh import locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from params import B, bfac, dt, eps_v, g, rho_i, rho_w, rm2, sea_level, u0
from petsc4py import PETSc
from ufl import (FacetNormal, FiniteElement, Measure, MixedElement,
                 SpatialCoordinate, TestFunctions, div, dx, grad, inner, split,
                 sym)


def eta(u, x):
      # ice viscosity with spatially varying prefactor
      return 0.5*B*bfac(x)*((inner(sym(grad(u)),sym(grad(u)))+eps_v)**(rm2/2.0))

def weak_form(u,p,v,q,f,g_base,ds,nu,x):
    # Weak form residual of the ice-shelf problem
    F = 2*eta(u,x)*inner(sym(grad(u)),sym(grad(v)))*dx
    F += (- div(v)*p + q*div(u))*dx - inner(f, v)*dx
    # Water-pressure follower load: p_w after boundary motion is
    # g_base - rho_w*g*dt*u_z (pressure tracks the VERTICAL motion of the
    # boundary point). The vendored (u.n) form is equivalent on the base
    # (n ~ -z) but adds a spurious ~13% backpressure ~ u_x on the vertical
    # front; -u[1] is correct on both.
    F += (g_base - rho_w*g*dt*u[1])*inner(v,nu)*ds(3)
    return F

def stokes_solve(domain, w_init=None):
        # Stokes solver for the ice-shelf problem using Taylor-Hood elements.
        # w_init: previous step's solution — warm-starts Newton (mesh topology
        # is fixed, so the dof layout is identical across steps).

        # Define function spaces
        P1 = FiniteElement('P',domain.ufl_cell(),1)     # Pressure p
        P2 = FiniteElement('P',domain.ufl_cell(),2)     # Velocity u
        element = MixedElement([[P2,P2],P1])
        W = FunctionSpace(domain,element)  # Function space for (u,p)

        #---------------------Define variational problem------------------------
        w = Function(W)
        if w_init is not None and w_init.x.array.size == w.x.array.size:
            w.x.array[:] = w_init.x.array
        (u,p) = split(w)
        (v,q) = TestFunctions(W)

        # Neumann condition at ice-water boundary
        x = SpatialCoordinate(domain)
        g_0 = rho_w*g*(sea_level-x[1])
        g_base = 0.5*(g_0+abs(g_0))

        # Body force
        f = Constant(domain,PETSc.ScalarType((0,-rho_i*g)))

        # Outward-pointing unit normal to the boundary
        nu = FacetNormal(domain)

        # Mark bounadries of mesh and define a measure for integration
        facet_tag = mark_boundary(domain)
        ds = Measure('ds', domain=domain, subdomain_data=facet_tag)

        # Inflow condition: full plug flow (u_x, u_z) = (u0, 0) on the left
        # wall — consistent with the clamped feed geometry in move_mesh. The
        # front is left unconstrained (open boundary under ds(3) pressure).
        facets_1 = locate_entities_boundary(domain, domain.topology.dim-1, LeftBoundary)
        dofs_1x = locate_dofs_topological(W.sub(0).sub(0), domain.topology.dim-1, facets_1)
        dofs_1z = locate_dofs_topological(W.sub(0).sub(1), domain.topology.dim-1, facets_1)
        bc1 = dirichletbc(PETSc.ScalarType(u0), dofs_1x,W.sub(0).sub(0))
        bc2 = dirichletbc(PETSc.ScalarType(0.0), dofs_1z,W.sub(0).sub(1))
        bcs = [bc1,bc2]

        # Define weak form
        F = weak_form(u,p,v,q,f,g_base,ds,nu,x)

        # Solve for (u,p)
        problem = NonlinearProblem(F, w, bcs=bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        # Warm-started steps begin with a small initial residual, so the
        # default rtol=1e-9 demands an absolute residual below the LU roundoff
        # floor and Newton stalls at max_it. atol=1.0 matches the absolute
        # level cold-started solves converge to (~0.1-17 vs force scale 1e10).
        solver.atol = 1.0
        solver.rtol = 1e-9
        solver.max_it = int(os.environ.get("E1B_NEWTON_MAXIT", "600"))
        solver.error_on_nonconvergence = False

        # Full Newton on the regularized power law (rm2 != 0) falls into a
        # period-2 limit cycle on warm-started steps once the mesh has moved
        # (r oscillates 2.4e5 <-> 3.2e5 forever); relaxation 0.5 converges
        # the same steps in ~30 iterations. n = 1 is linear, so full Newton
        # is exact in one step — keep it undamped. The ladder halves the
        # relaxation on any residual failure before giving up.
        relax0 = float(os.environ.get(
            "E1B_NEWTON_RELAX", "1.0" if rm2 == 0.0 else "0.5"))
        w0 = w.x.array.copy()
        set_log_level(LogLevel.WARNING)
        for relax in (relax0, 0.5*relax0, 0.25*relax0):
            w.x.array[:] = w0
            solver.relaxation_parameter = relax
            n, converged = solver.solve(w)
            if converged:
                break
        assert(converged)

        return w
