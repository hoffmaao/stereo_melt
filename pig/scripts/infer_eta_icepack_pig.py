"""PIG fluidity inversion — the single-field mismip_time-dependent-da
procedure on the real shelf, yielding the eta_field for the fused inverse.

Chain: exported (H, u, outline) from the tilt-corrected 250 m stack
(pig/scripts/export_eta_inversion_inputs.py) -> gmsh mesh of the shelf
polygon -> icepack IceShelf (default Glen n=3 viscosity; PIG-density gravity
clone) -> single log-control theta with A_eff = A0*exp(theta),
A0 = icepack.rate_factor(T0) -> Whittle-Matern prior (TDDA prior.py) ->
bounded scipy L-BFGS-B on the pyadjoint tape with quadratic-barrier recovery.

v1 boundary treatment: Dirichlet u = u_obs on the ENTIRE outline (grounding
line, margins, AND front), so the terminus term is inactive and no boundary
classification is needed; interior misfit constrains the viscosity. Revisit
with a proper calving-front id for the stress-condition version.

Outputs:
  pig/processed/pig_eta_field_250m.npz — theta, A (MPa^-3 yr^-1), and the
    Newtonian-equivalent  eta_bar = 1/2 A^(-1/3) eps_e^(-2/3)  [Pa s] on the
    stack grid (strain rate from the MAP model velocity, floored), ready for
    variational_melt_rate(eta_field=...).
  pig/figures/pig_eta_inversion_qc.png — |u_obs|, misfit, theta, log10 eta.

Run:
    elmer_synth/scripts/icepack_python.sh pig/scripts/infer_eta_icepack_pig.py \
        [--lc 1250] [--max-iters 50] [--gamma 1.0] [--sigma_u 20]
"""
import argparse
import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

import firedrake
from firedrake import (
    Constant, Function, FunctionSpace, SpatialCoordinate, VectorFunctionSpace,
    assemble, dx, grad, inner, sym, tr as trace,
)
import firedrake.adjoint as fda
import icepack
from icepack.constants import gravity as G_IC, year as YR_IC
from scipy.optimize import minimize as scipy_minimize

TDDA = "/wd2/projects/mismip_time-dependent-da"
sys.path.insert(0, TDDA)
from prior import regularization_form  # noqa: E402

REPO = "/wd2/projects/stereo_melt"
NPZ = f"{REPO}/pig/processed/pig_eta_inv_inputs.npz"
MSH = f"{REPO}/pig/processed/pig_shelf.msh"
OUT = f"{REPO}/pig/processed/pig_eta_field_250m.npz"
FIG = f"{REPO}/pig/figures/pig_eta_inversion_qc.png"
DEG = 2
T0_K = 258.0                 # A0 = rate_factor(T0); theta absorbs the offset
EPS_MIN = 1.0e-4             # 1/yr strain-rate floor for the eta map


def make_pig_gravity(rho_i_kg, rho_w_kg):
    rho = (rho_i_kg / YR_IC ** 2 * 1.0e-6) * (1.0 - rho_i_kg / rho_w_kg)

    def pig_gravity(**kw):
        u, h = kw["velocity"], kw["thickness"]
        return -0.5 * rho * G_IC * inner(grad(h ** 2), u)

    def pig_terminus(**kw):          # inactive under all-Dirichlet; clone anyway
        u, h = kw["velocity"], kw["thickness"]
        nu = firedrake.FacetNormal(u.function_space().mesh())
        return 0.5 * rho * G_IC * h ** 2 * inner(u, nu)

    return pig_gravity, pig_terminus


def build_mesh(poly, lc):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("pig_shelf")
    ring = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    d = np.hypot(*np.diff(np.vstack([ring, ring[:1]]), axis=0).T)
    keep = np.ones(len(ring), bool)
    keep[1:] = d[:-1] > 1.0          # drop consecutive duplicates
    ring = ring[keep]
    pts = [gmsh.model.geo.addPoint(float(px), float(py), 0.0, lc)
           for px, py in ring]
    lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)])
             for i in range(len(pts))]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, lines, 1)
    gmsh.model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(MSH)
    n_tri = len(gmsh.model.mesh.getElementsByType(2)[0])
    gmsh.finalize()
    print(f"[mesh] {len(ring)} boundary pts, lc {lc:.0f} m, "
          f"{n_tri} triangles -> {MSH}", flush=True)
    return firedrake.Mesh(MSH)


def sample_to(Q, x, y, arr, mesh, deg):
    interp = RegularGridInterpolator((y, x), arr, bounds_error=False,
                                     fill_value=None)
    Vc = VectorFunctionSpace(mesh, "CG", deg)
    X = Function(Vc)
    X.interpolate(SpatialCoordinate(mesh))
    pts = X.dat.data_ro
    f = Function(Q)
    f.dat.data[:] = interp(np.column_stack([pts[:, 1], pts[:, 0]]))
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lc", type=float, default=1250.0, help="mesh size (m)")
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--sigma_u", type=float, default=20.0,
                    help="velocity misfit weight (m/yr)")
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    x, y = d["x"], d["y"]
    H_g, ux_g, uy_g, mask_g = d["H"], d["ux"], d["uy"], d["mask"].astype(bool)
    rho_i, rho_w = float(d["RHO_I"]), float(d["RHO_W"])
    print(f"[data] grid {H_g.shape}, mask {mask_g.sum()} cells, "
          f"rho {rho_i:.0f}/{rho_w:.0f}, vel {d['vsrc']}", flush=True)

    mesh = build_mesh(np.asarray(d["poly"], float), args.lc)
    Q = FunctionSpace(mesh, "CG", DEG)
    V = VectorFunctionSpace(mesh, "CG", DEG)
    Q1 = FunctionSpace(mesh, "CG", 1)

    h = sample_to(Q, x, y, H_g, mesh, DEG)
    ux_f = sample_to(Q, x, y, ux_g, mesh, DEG)
    uy_f = sample_to(Q, x, y, uy_g, mesh, DEG)
    u_obs = Function(V)
    u_obs.dat.data[:, 0] = ux_f.dat.data_ro
    u_obs.dat.data[:, 1] = uy_f.dat.data_ro
    area = float(assemble(Constant(1.0) * dx(mesh)))
    print(f"[fields] H [{h.dat.data_ro.min():.0f}, {h.dat.data_ro.max():.0f}] m"
          f"  area {area / 1e6:.0f} km^2", flush=True)

    A0 = float(icepack.rate_factor(Constant(T0_K)))
    print(f"[prior] A0 = rate_factor({T0_K:.0f} K) = {A0:.4e} MPa^-3 yr^-1",
          flush=True)

    pig_gravity, pig_terminus = make_pig_gravity(rho_i, rho_w)
    model = icepack.models.IceShelf(gravity=pig_gravity, terminus=pig_terminus)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters={
            "snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_max_it": 200, "snes_rtol": 1e-8,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        })

    theta = Function(Q1, name="theta_A").assign(0.0)
    sigma_u = Constant(args.sigma_u)
    inv_area = Constant(1.0 / area)

    # Tape once; ReducedFunctional replays inside the TDDA loop.
    fda.continue_annotation()
    A_eff = Function(Q1, name="A_eff")
    A_eff.interpolate(Constant(A0) * firedrake.exp(theta))
    u_c = solver.diagnostic_solve(
        velocity=Function(V).assign(u_obs), thickness=h, fluidity=A_eff)
    du = u_c - u_obs
    J0 = assemble(0.5 * inv_area
                  * ((du[0] / sigma_u) ** 2 + (du[1] / sigma_u) ** 2) * dx
                  + regularization_form(theta, args.gamma, area))
    Jhat = fda.ReducedFunctional(J0, fda.Control(theta))
    fda.pause_annotation()

    den = float(assemble(inner(u_obs, u_obs) * dx))
    rel0 = float(np.sqrt(float(assemble(inner(du, du) * dx)) / den))
    print(f"[preflight] theta=0 (A(-15C) uniform): rel vel misfit "
          f"{100 * rel0:.1f}%", flush=True)

    n_dof = Q1.dim()
    theta_work = Function(Q1)
    best = {"J": float("inf"), "x": theta.dat.data_ro.copy()}
    nev = {"n": 0, "fail": 0}

    def _recovery(xv):
        dxv = xv - best["x"]
        pen = 10.0 * abs(best["J"]) + 1.0
        return best["J"] + 0.5 * pen * float(dxv @ dxv), pen * dxv

    def objective_and_gradient(xv):
        theta_work.dat.data[:] = xv
        try:
            Jval = float(Jhat(theta_work))
            try:
                dJ = Jhat.derivative(options={"riesz_representation": None})
            except TypeError:
                dJ = Jhat.derivative()
            g = np.asarray(dJ.dat.data_ro).copy()
        except Exception as e:
            nev["fail"] += 1
            print(f"    eval FAILED ({e.__class__.__name__}: {e})", flush=True)
            return _recovery(xv)
        nev["n"] += 1
        if Jval < best["J"]:
            best["J"], best["x"] = Jval, xv.copy()
        if nev["n"] % 2 == 1:
            print(f"    eval {nev['n']:3d}: J {Jval:.6e}  |g| "
                  f"{np.linalg.norm(g):.3e}  |theta| "
                  f"{np.max(np.abs(xv)):.2f}", flush=True)
        return Jval, g

    print(f"[invert] {n_dof} DOFs, gamma {args.gamma}, sigma_u {args.sigma_u} "
          f"m/yr, bounds +/-5, max {args.max_iters} iters", flush=True)
    res = scipy_minimize(
        objective_and_gradient, theta.dat.data_ro.copy(), method="L-BFGS-B",
        jac=True, bounds=[(-5.0, 5.0)] * n_dof,
        options={"maxiter": args.max_iters, "gtol": 1e-7, "maxcor": 30,
                 "ftol": 1e-12})
    theta.dat.data[:] = best["x"]
    print(f"[invert] {res.message}  iters {res.nit} evals {nev['n']} "
          f"fails {nev['fail']}  J {best['J']:.6e}", flush=True)

    # MAP state + Newtonian-equivalent viscosity from the MAP strain rate
    A_map = Function(Q1, name="A_map")
    A_map.interpolate(Constant(A0) * firedrake.exp(theta))
    u_map = solver.diagnostic_solve(
        velocity=Function(V).assign(u_obs), thickness=h, fluidity=A_map)
    du = u_map - u_obs
    rel1 = float(np.sqrt(float(assemble(inner(du, du) * dx)) / den))
    print(f"[map] rel vel misfit {100 * rel1:.1f}% (from {100 * rel0:.1f}%)",
          flush=True)

    eps = sym(grad(u_map))
    eps_e = firedrake.sqrt((inner(eps, eps) + trace(eps) ** 2) / 2
                           + Constant(EPS_MIN ** 2))
    eta_q1 = Function(Q1, name="eta_bar_mpaa")
    eta_q1.interpolate(0.5 * A_map ** (-1.0 / 3.0) * eps_e ** (-2.0 / 3.0))

    # Resample Q1 fields to the 250 m stack grid (linear griddata; NaN off-mesh)
    Vc1 = VectorFunctionSpace(mesh, "CG", 1)
    X1 = Function(Vc1)
    X1.interpolate(SpatialCoordinate(mesh))
    pts = X1.dat.data_ro
    Xg, Yg = np.meshgrid(x, y)
    gp = np.column_stack([Xg.ravel(), Yg.ravel()])

    def to_grid(f):
        v = griddata(pts, np.asarray(f.dat.data_ro), gp,
                     method="linear").reshape(Xg.shape)
        return np.where(mask_g, v, np.nan)

    theta_g = to_grid(theta)
    A_g = to_grid(A_map)
    eta_mpaa_g = to_grid(eta_q1)
    eta_pas_g = eta_mpaa_g * 1.0e6 * YR_IC          # MPa a -> Pa s
    umod_x = to_grid(Function(Q1).interpolate(u_map[0]))
    umod_y = to_grid(Function(Q1).interpolate(u_map[1]))

    np.savez(OUT, x=x, y=y, theta=theta_g, A_mpa3yr=A_g,
             eta_mpaa=eta_mpaa_g, eta_pas=eta_pas_g,
             u_model_x=umod_x, u_model_y=umod_y, mask=mask_g.astype(np.uint8),
             A0=A0, T0_K=T0_K, gamma=args.gamma, sigma_u=args.sigma_u,
             lc=args.lc, rel_misfit0=rel0, rel_misfit_map=rel1,
             eps_min=EPS_MIN, rho_i=rho_i, rho_w=rho_w)
    fin = np.isfinite(eta_pas_g)
    print(f"wrote {OUT}")
    print(f"  eta_pas: p10 {np.nanpercentile(eta_pas_g, 10):.2e}  median "
          f"{np.nanmedian(eta_pas_g):.2e}  p90 "
          f"{np.nanpercentile(eta_pas_g, 90):.2e} Pa s  ({fin.sum()} cells)",
          flush=True)

    # ---- QC figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp_obs = np.hypot(ux_g, uy_g)
    dsp = np.hypot(umod_x - ux_g, umod_y - uy_g)
    fig, axs = plt.subplots(1, 4, figsize=(19, 5), sharex=True, sharey=True)
    panels = [
        (np.where(mask_g, sp_obs, np.nan), "|u_obs| (m/yr)", "viridis", None),
        (dsp, "|u_map - u_obs| (m/yr)", "magma", None),
        (theta_g, "theta = ln(A/A0)", "RdBu_r", 3.0),
        (np.log10(eta_pas_g), "log10 eta_bar (Pa s)", "viridis", None),
    ]
    for ax, (fld, ttl, cmap, vl) in zip(axs, panels):
        kw = dict(cmap=cmap, shading="auto")
        if vl:
            kw.update(vmin=-vl, vmax=vl)
        hh = ax.pcolormesh(x / 1e3, y / 1e3, fld, **kw)
        ax.set_title(ttl, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        fig.colorbar(hh, ax=ax, shrink=0.8)
    axs[0].set_ylabel("y (km)")
    fig.suptitle(
        f"PIG fluidity inversion (single-field TDDA) — rel misfit "
        f"{100 * rel0:.1f}% -> {100 * rel1:.1f}%,  gamma {args.gamma}, "
        f"sigma_u {args.sigma_u} m/yr, lc {args.lc:.0f} m", fontsize=12)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    print(f"wrote {FIG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
