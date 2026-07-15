#------------------------------------------------------------------------------------
# This program solves a nonlinear Stokes problem describing an ice-shelf flowline
# WITH through-flow (E1b): ice enters at x=-L/2 at speed u0, exits through an open
# front at x=+L/2, and transits an ocean-fixed basal melt anomaly. Forked from
# vendor/linear-shelf-melt/nonlinear-model/main.py; deltas: through-flow BCs live
# in stokes/bdry_conds/mesh_routine, per-step progress prints, periodic
# surface-velocity capture for melt-solver evaluation.
#------------------------------------------------------------------------------------
import time

import numpy as np
from dolfinx.mesh import create_rectangle
from mesh_routine import get_surface_fields, get_surfaces, move_mesh
from mpi4py import MPI
from params import H, L, Nx, Nz, nt, save_every, t_f, yr
from stokes import stokes_solve


def solve(a,m):

    # generate mesh
    p0 = [-L/2.0,0.0]
    p1 = [L/2.0,H]
    domain = create_rectangle(MPI.COMM_WORLD,[p0,p1], [Nx, Nz])

    # Define arrays for saving surfaces and surface velocities
    h_i,s_i,x = get_surfaces(domain)
    nx = x.size
    h = np.zeros((nx,nt))
    s = np.zeros((nx,nt))
    n_saves = int(np.ceil(nt/save_every))
    u_top = np.zeros((nx,n_saves)); w_top = np.zeros((nx,n_saves))
    u_bot = np.zeros((nx,n_saves)); w_bot = np.zeros((nx,n_saves))
    t_saves = np.zeros(n_saves)

    t = np.linspace(0,t_f, nt)
    t0_wall = time.time()
    ks = 0
    sol = None
    # Begin time stepping
    for i in range(nt):

        t_i = t[i]

        # Solve the Stokes problem for w = (u,p), warm-started from last step
        sol = stokes_solve(domain, sol)

        # Capture surface/basal velocities every save_every steps
        if i % save_every == 0:
            ut,wt,ub,wb,_ = get_surface_fields(sol,domain)
            u_top[:,ks] = ut; w_top[:,ks] = wt
            u_bot[:,ks] = ub; w_bot[:,ks] = wb
            t_saves[ks] = t_i
            ks += 1

        # Move the mesh
        domain = move_mesh(sol,domain,t_i,a,m)

        h_i,s_i,x = get_surfaces(domain)

        h[:,i] = h_i
        s[:,i] = s_i

        if (i+1) % 50 == 0 or i == 0:
            print(f"step {i+1}/{nt}  t={t_i/yr:.1f} yr  "
                  f"H_center={h_i[nx//2]-s_i[nx//2]:.1f} m  "
                  f"elapsed={(time.time()-t0_wall)/60.0:.1f} min", flush=True)

    return h,s,x,u_top,w_top,u_bot,w_bot,t,t_saves[:ks]
