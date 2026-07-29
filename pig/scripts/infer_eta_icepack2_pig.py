"""PIG fluidity inversion v2 — icepack2 DUAL form + window-consistent inputs.

Differences from v1 (infer_eta_icepack_pig.py, primal icepack):
  * DUAL (mixed velocity/membrane-stress) formulation from icepack2: the
    problem stays well-posed toward zero thickness and tolerates the
    discontinuities of a real calving front; the front is the NATURAL
    boundary of the shelf balance form (0.5*rho*g*h^2*div(u) term), so front
    segments are simply NOT pinned — no all-Dirichlet dodge, no terminus
    term needed on a pure shelf.
  * Boundary ids from the exporter's BedMachine classification: 1 =
    grounding line / margins (Dirichlet u = u_obs), 2 = calving front
    (natural).
  * Inputs are TIME-CONSISTENT (H, s, u, and the front geometry all from the
    same window; exporter default 2021-2024) so momentum-balance
    inconsistencies from mixing eras do not masquerade as fluidity.
  * n-continuation (n: 1 -> 3, warm-started) for the nonlinear dual solve,
    per the icepack2 test suite.

Inversion machinery unchanged: single-field TDDA (theta, A = A0*exp(theta),
Whittle-Matern prior via mismip_time-dependent-da prior.py, bounded
L-BFGS-B on the pyadjoint tape, quadratic-barrier recovery).

eta for the melt operator comes straight from the DUAL pair: with
eps = A*M_e^(n-1)*M, the Newtonian-equivalent viscosity is
eta_bar = 1/(2*A*M_e^(n-1)) — no velocity differentiation needed.

Run:
    elmer_synth/scripts/icepack_python.sh pig/scripts/infer_eta_icepack2_pig.py \
        [--lc 1250] [--max-iters 50] [--gamma 1.0] [--sigma_u 20]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

import firedrake
from firedrake import (
    Constant, DirichletBC, Function, FunctionSpace, NonlinearVariationalProblem,
    NonlinearVariationalSolver, SpatialCoordinate, TensorFunctionSpace,
    VectorFunctionSpace, assemble, derivative, div, dx, grad, inner, sym,
    tr as trace,
)
import firedrake.adjoint as fda
import icepack                                  # v1: rate_factor only
from icepack2.constants import (
    gravity as G2, year as YR2, glen_flow_law as N_GLEN,
)
from icepack2.model.minimization import viscous_power
from scipy.optimize import minimize as scipy_minimize

TDDA = "/wd2/projects/mismip_time-dependent-da"
sys.path.insert(0, TDDA)
from prior import L_REG, regularization_form  # noqa: E402

REPO = "/wd2/projects/stereo_melt"
NPZ = f"{REPO}/pig/processed/pig_eta_inv_inputs.npz"
MSH = f"{REPO}/pig/processed/pig_shelf_dual.msh"
OUT = f"{REPO}/pig/processed/pig_eta_field_250m_dual.npz"
FIG = f"{REPO}/pig/figures/pig_eta_inversion_dual_qc.png"
T0_K = 258.0
M_E_MIN = 1.0e-3          # MPa; membrane-stress floor for the eta map


def build_mesh(polys, vclasses, lc, msh_path):
    """Multi-loop polygon mesh: one plane surface per shelf component,
    physical line groups 1=dirichlet, 2=front shared across all loops."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("pig_shelf_dual")
    surfs, dir_lines, frt_lines = [], [], []
    n_pts = 0
    for poly, vclass in zip(polys, vclasses):
        poly = np.asarray(poly, float)
        vclass = np.asarray(vclass)
        ring = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
        vc = vclass[: len(ring)]
        d = np.hypot(*np.diff(np.vstack([ring, ring[:1]]), axis=0).T)
        keep = np.ones(len(ring), bool)
        keep[1:] = d[:-1] > 1.0
        ring, vc = ring[keep], vc[keep]
        pts = [gmsh.model.geo.addPoint(float(px), float(py), 0.0, lc)
               for px, py in ring]
        n_pts += len(pts)
        lines = []
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            ln = gmsh.model.geo.addLine(pts[i], pts[j])
            lines.append(ln)
            if vc[i] == 2 and vc[j] == 2:
                frt_lines.append(ln)
            else:
                dir_lines.append(ln)
        loop = gmsh.model.geo.addCurveLoop(lines)
        surfs.append(gmsh.model.geo.addPlaneSurface([loop]))
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, dir_lines, 1)
    if frt_lines:
        gmsh.model.addPhysicalGroup(1, frt_lines, 2)
    gmsh.model.addPhysicalGroup(2, surfs, 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(msh_path)
    n_tri = len(gmsh.model.mesh.getElementsByType(2)[0])
    gmsh.finalize()
    print(f"[mesh] {len(surfs)} shelf component(s), {n_pts} boundary pts "
          f"({len(frt_lines)} front segs), lc {lc:.0f} m, {n_tri} triangles "
          f"-> {msh_path}", flush=True)
    return firedrake.Mesh(msh_path)


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
    ap.add_argument("--lc", type=float, default=1250.0)
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--sigma_u", type=float, default=20.0)
    ap.add_argument("--bound", type=float, default=0.0,
                    help="|theta| box bound for L-BFGS-B; <=0 = unbounded "
                         "(default — the Whittle-Matern prior is the "
                         "regularizer, 2026-07-27 directive)")
    ap.add_argument("--theta0-npz", default=None,
                    help="warm-start theta from a previous output npz "
                         "(theta on the input grid)")
    ap.add_argument("--out-tag", default="",
                    help="suffix for OUT/FIG/MSH so variant runs (e.g. the "
                         "whole-embayment domain) don't clobber the canonical "
                         "outputs")
    args = ap.parse_args()

    tag = f"_{args.out_tag.strip('_')}" if args.out_tag else ""
    out_npz = OUT.replace(".npz", f"{tag}.npz")
    out_fig = FIG.replace(".png", f"{tag}.png")
    msh_path = MSH.replace(".msh", f"{tag}.msh")

    d = np.load(NPZ, allow_pickle=True)
    x, y = d["x"], d["y"]
    H_g = d["H"]          # (surface not needed: hydrostatic shelf balance)
    ux_g, uy_g, mask_g = d["ux"], d["uy"], d["mask"].astype(bool)
    polys, vclasses = list(d["polys"]), list(d["vclasses"])
    rho_i, rho_w = float(d["RHO_I"]), float(d["RHO_W"])
    print(f"[data] window {d['t0']}..{d['t1']} ({int(d['n_epochs'])} epochs), "
          f"mask {mask_g.sum()} cells ({len(polys)} shelf components), "
          f"vel {d['vsrc']}", flush=True)

    mesh = build_mesh(polys, vclasses, args.lc, msh_path)
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    S = TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
    Z = V * S
    z = Function(Z)

    h = sample_to(Q, x, y, H_g, mesh, 1)
    ux_f = sample_to(Q, x, y, ux_g, mesh, 1)
    uy_f = sample_to(Q, x, y, uy_g, mesh, 1)
    u_obs = Function(V)
    u_obs.dat.data[:, 0] = ux_f.dat.data_ro
    u_obs.dat.data[:, 1] = uy_f.dat.data_ro
    area = float(assemble(Constant(1.0) * dx(mesh)))
    print(f"[fields] H [{h.dat.data_ro.min():.0f}, "
          f"{h.dat.data_ro.max():.0f}] m  area {area / 1e6:.0f} km^2",
          flush=True)

    A0 = float(icepack.rate_factor(Constant(T0_K)))
    print(f"[prior] A0 = rate_factor({T0_K:.0f} K) = {A0:.4e} MPa^-3 yr^-1",
          flush=True)

    # PIG-density shelf balance (icepack2's uses module constants 917/1024)
    rho_pig = (rho_i / YR2 ** 2 * 1.0e-6) * (1.0 - rho_i / rho_w)

    def pig_shelf_balance(**kw):
        u, M, hh = kw["velocity"], kw["membrane_stress"], kw["thickness"]
        eps = sym(grad(u))
        return (-hh * inner(M, eps) + 0.5 * rho_pig * G2 * hh ** 2
                * div(u)) * dx

    n_c = Constant(1.0)
    u_split, M_split = firedrake.split(z)
    fields = {"velocity": u_split, "membrane_stress": M_split, "thickness": h}
    theta = Function(Q, name="theta_A").assign(0.0)
    if args.theta0_npz:
        d0 = np.load(args.theta0_npz)
        th0 = np.asarray(d0["theta"], float)
        bad = ~np.isfinite(th0)
        if bad.any():
            # nearest-fill outside the product mask so mesh sampling stays
            # finite near the boundary
            Yg0, Xg0 = np.meshgrid(d0["y"], d0["x"], indexing="ij")
            th0[bad] = griddata(
                np.column_stack([Yg0[~bad], Xg0[~bad]]), th0[~bad],
                np.column_stack([Yg0[bad], Xg0[bad]]), method="nearest")
        theta.assign(sample_to(Q, d0["x"], d0["y"], th0, mesh, 1))
        if args.bound > 0:
            np.clip(theta.dat.data, -args.bound, args.bound,
                    out=theta.dat.data)
        print(f"[warm] theta0 <- {args.theta0_npz}  max|theta0| "
              f"{np.max(np.abs(theta.dat.data_ro)):.2f}", flush=True)
    A_eff = Function(Q, name="A_eff")
    A_eff.interpolate(Constant(A0) * firedrake.exp(theta))
    rheology = {"flow_law_exponent": n_c, "flow_law_coefficient": A_eff}
    L = (viscous_power(**fields, **rheology)
         + pig_shelf_balance(**fields))
    F = derivative(L, z)
    bcs = [DirichletBC(Z.sub(0), u_obs, (1,))]
    pparams = {"form_compiler_parameters": {"quadrature_degree": 8}}
    sparams = {"solver_parameters": {
        "snes_type": "newtonls", "snes_max_it": 200,
        "snes_linesearch_type": "nleqerr",
        "ksp_type": "gmres", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }}
    problem = NonlinearVariationalProblem(F, z, bcs, **pparams)
    solver = NonlinearVariationalSolver(problem, **sparams)

    # Untaped n-continuation at the prior (n: 1 -> 3, warm-started): gives the
    # preflight misfit AND the initial state the taped n=3 solve starts from.
    z.sub(0).assign(u_obs)
    for expn in np.linspace(1.0, N_GLEN, 5):
        n_c.assign(float(expn))
        solver.solve()
    u_pre = z.subfunctions[0]
    den = float(assemble(inner(u_obs, u_obs) * dx))
    rel0 = float(np.sqrt(
        float(assemble(inner(u_pre - u_obs, u_pre - u_obs) * dx)) / den))
    start = "theta0 (warm)" if args.theta0_npz else "A0 uniform"
    print(f"[preflight] dual solve at {start} (n-continuation): rel vel "
          f"misfit {100 * rel0:.1f}%", flush=True)

    # Taped n=3 solve from the continuation state
    sigma_u = Constant(args.sigma_u)
    inv_area = Constant(1.0 / area)
    fda.continue_annotation()
    A_eff.interpolate(Constant(A0) * firedrake.exp(theta))
    solver.solve()
    du = firedrake.split(z)[0] - u_obs
    J0 = assemble(0.5 * inv_area
                  * ((du[0] / sigma_u) ** 2 + (du[1] / sigma_u) ** 2) * dx
                  + regularization_form(theta, args.gamma, area))
    Jhat = fda.ReducedFunctional(J0, fda.Control(theta))
    fda.pause_annotation()

    n_dof = Q.dim()
    theta_work = Function(Q)
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
            print(f"    eval FAILED ({e.__class__.__name__})", flush=True)
            return _recovery(xv)
        nev["n"] += 1
        if Jval < best["J"]:
            best["J"], best["x"] = Jval, xv.copy()
        if nev["n"] % 2 == 1:
            print(f"    eval {nev['n']:3d}: J {Jval:.6e}  |g| "
                  f"{np.linalg.norm(g):.3e}  |theta| "
                  f"{np.max(np.abs(xv)):.2f}", flush=True)
        return Jval, g

    bnds = ([(-args.bound, args.bound)] * n_dof if args.bound > 0 else None)
    btxt = f"+/-{args.bound:g}" if args.bound > 0 else "none (prior-only)"
    print(f"[invert] DUAL, {n_dof} DOFs, gamma {args.gamma}, sigma_u "
          f"{args.sigma_u} m/yr, bounds {btxt}, max "
          f"{args.max_iters} iters", flush=True)
    res = scipy_minimize(
        objective_and_gradient, theta.dat.data_ro.copy(), method="L-BFGS-B",
        jac=True, bounds=bnds,
        options={"maxiter": args.max_iters, "gtol": 1e-7, "maxcor": 30,
                 "ftol": 1e-12})
    theta.dat.data[:] = best["x"]
    print(f"[invert] {res.message}  iters {res.nit} evals {nev['n']} "
          f"fails {nev['fail']}  J {best['J']:.6e}", flush=True)

    # MAP state: untaped continuation re-solve at the MAP fluidity
    A_map = Function(Q, name="A_map")
    A_map.interpolate(Constant(A0) * firedrake.exp(theta))
    A_eff.assign(A_map)
    z.sub(0).assign(u_obs)
    z.sub(1).dat.data[:] = 0.0
    for expn in np.linspace(1.0, N_GLEN, 5):
        n_c.assign(float(expn))
        solver.solve()
    u_map, M_map = z.subfunctions
    rel1 = float(np.sqrt(
        float(assemble(inner(u_map - u_obs, u_map - u_obs) * dx)) / den))
    print(f"[map] rel vel misfit {100 * rel1:.1f}% (from {100 * rel0:.1f}%)",
          flush=True)

    # Objective split at the MAP, for the L-curve (mismip_time-dependent-da
    # convention): J = J_misfit + J_reg, and the gamma-INDEPENDENT solution
    # seminorm J_reg/gamma is the L-curve's regularization axis.
    dmap = u_map - u_obs
    J_misfit = float(assemble(
        0.5 * inv_area * ((dmap[0] / sigma_u) ** 2
                          + (dmap[1] / sigma_u) ** 2) * dx))
    J_reg = float(assemble(regularization_form(theta, args.gamma, area)))
    seminorm = J_reg / float(args.gamma) if args.gamma else float("nan")
    print(f"[map] J_misfit {J_misfit:.6e}  J_reg {J_reg:.6e}  "
          f"seminorm(J_reg/gamma) {seminorm:.6e}", flush=True)

    # Newtonian-equivalent viscosity from the DUAL pair:
    # eps = A*M_e^(n-1)*M  =>  eta_bar = 1/(2*A*M_e^(n-1)); M_e floored.
    Me2 = (inner(M_map, M_map) - trace(M_map) ** 2 / 3.0) / 2.0
    Me = firedrake.sqrt(Me2 + Constant(M_E_MIN ** 2))
    eta_q1 = Function(Q, name="eta_bar_mpaa")
    eta_q1.interpolate(1.0 / (2.0 * A_map * Me ** (N_GLEN - 1.0)))

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
    eta_pas_g = eta_mpaa_g * 1.0e6 * YR2
    umod_x = to_grid(Function(Q).interpolate(u_map[0]))
    umod_y = to_grid(Function(Q).interpolate(u_map[1]))

    np.savez(out_npz, x=x, y=y, theta=theta_g, A_mpa3yr=A_g,
             eta_mpaa=eta_mpaa_g, eta_pas=eta_pas_g,
             u_model_x=umod_x, u_model_y=umod_y, mask=mask_g.astype(np.uint8),
             A0=A0, T0_K=T0_K, gamma=args.gamma, sigma_u=args.sigma_u,
             lc=args.lc, rel_misfit0=rel0, rel_misfit_map=rel1,
             bound=args.bound, theta0_src=str(args.theta0_npz or ""),
             n_components=len(polys),
             m_e_min=M_E_MIN, rho_i=rho_i, rho_w=rho_w,
             t0=str(d["t0"]), t1=str(d["t1"]), form="icepack2-dual")
    print(f"wrote {out_npz}")
    print(f"  eta_pas: p10 {np.nanpercentile(eta_pas_g, 10):.2e}  median "
          f"{np.nanmedian(eta_pas_g):.2e}  p90 "
          f"{np.nanpercentile(eta_pas_g, 90):.2e} Pa s", flush=True)

    # summary.json — the L-curve reader's input (one per sweep point)
    summary = {
        "gamma": float(args.gamma), "sigma_u": float(args.sigma_u),
        "L_reg": float(L_REG), "bound": float(args.bound),
        "J_total": float(best["J"]), "J_misfit": J_misfit, "J_reg": J_reg,
        "seminorm": seminorm,
        "rel_misfit0": rel0, "rel_misfit_map": rel1,
        "n_iters": int(res.nit), "n_evals": int(nev["n"]),
        "n_failures": int(nev["fail"]), "opt_message": str(res.message),
        "grad_norm_final": float(np.linalg.norm(
            np.asarray(Jhat.derivative().dat.data_ro))),
        "max_abs_theta": float(np.max(np.abs(theta.dat.data_ro))),
        "n_dof": int(n_dof), "area_km2": area / 1e6,
        "n_components": len(polys), "lc": float(args.lc),
        "eta_p10": float(np.nanpercentile(eta_pas_g, 10)),
        "eta_median": float(np.nanmedian(eta_pas_g)),
        "eta_p90": float(np.nanpercentile(eta_pas_g, 90)),
        "t0": str(d["t0"]), "t1": str(d["t1"]),
        "theta0_src": str(args.theta0_npz or ""), "out_npz": out_npz,
    }
    out_json = out_npz.replace(".npz", "_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}", flush=True)

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
        f"PIG fluidity inversion — icepack2 DUAL, window {d['t0']}..{d['t1']}"
        f" — rel misfit {100 * rel0:.1f}% -> {100 * rel1:.1f}%,  gamma "
        f"{args.gamma}, sigma_u {args.sigma_u} m/yr", fontsize=12)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    print(f"wrote {out_fig}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
