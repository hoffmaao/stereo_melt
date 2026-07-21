"""Level-B keystone: differentiable variational melt inverse on an E2a twin.

Fits basal melt m(x,y) to the OBSERVED surface anomaly dZs = zs - zs_control by
pushing m through the linearized Stubblefield forward operator as a fixed
differentiable torch layer and minimizing
  || Forward(m) - dZs_obs ||^2_mask + lam * ||m||^2
over a dense grid or a SIREN coordinate network via Adam. The forward operator
and the fit live in the library (``stereo_melt.dynamics.stubblefield_forward``);
this script is the E2a-specific driver: it loads a run, differences against the
control, fits, and scores the recovered across-flow melt against truth.

Why this matters (vs Level A): Level A deconvolved the Eulerian melt OUTPUT with
the transfer T, which over-lifts pure across-flow ridges (cosy 1.07 -> 1.54)
because the Eulerian already recovers those via u*dH/dx (dynamics, not the
surface transfer). Fitting the SURFACE through the forward operator has no such
double count: the across-flow dynamics live in G_h itself, so the fit recovers
cosy with NO angular weight -- the residual is a forward-model calibration
(eta_bar), not a double-count artifact.

  PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
  STEREO_MELT_BACKEND=numpy $PY elmer_synth/scripts/level_b_variational_inverse.py \
      --pert cosy_L3H_a008 --rep grid
"""
import argparse
import math
import sys

import numpy as np

sys.path.insert(0, "/wd2/projects/stereo_melt/elmer_synth")
sys.path.insert(0, "/wd2/projects/stereo_melt/elmer_synth/scripts")
sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")
# Library first (pulls in pandas via xarray), then torch: torch loads a newer
# libstdc++ that shadows the system one pandas' C-extensions link against.
from stereo_melt.dynamics.stubblefield_forward import (  # noqa: E402
    StubblefieldForward,
    stubblefield_forward_multiplier,
    variational_melt_inverse,
)

import e2a.params as p  # noqa: E402
from score_e2a import cosine_fit, load_run, profile_1d, science_window  # noqa: E402

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pert", default="cosy_L3H_a008")
    ap.add_argument("--control", default="control_200")
    ap.add_argument("--rep", choices=["grid", "siren"], default="grid")
    ap.add_argument("--alpha-scale", type=float, default=0.34)
    ap.add_argument("--eta-bar", type=float, default=1e14)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--self-test", action="store_true",
                    help="fit the model's OWN synthetic forward (must -> 1.0)")
    args = ap.parse_args()

    pert, meta = load_run(args.pert, None)
    ctrl, _ = load_run(args.control, None)
    mm = meta["melt"]
    x = pert.x.values
    y = pert.y.values
    ny, nx = pert.sizes["y"], pert.sizes["x"]
    dx, dy = float(abs(x[1] - x[0])), float(abs(y[1] - y[0]))
    dzs = (pert.zs.isel(time=-1).values - ctrl.zs.isel(time=-1).values)
    u = float(np.nanmean(pert.vx.values))
    v = float(np.nanmean(pert.vy.values))
    X, Y = np.meshgrid(x, y)
    sci = science_window(pert)
    ylo, yhi = min(sci["y0"], sci["y1"]), max(sci["y0"], sci["y1"])
    mask = ((X >= sci["x0"]) & (X <= sci["x1"]) & (Y >= ylo) & (Y <= yhi)
            & np.isfinite(dzs))

    if args.self_test:
        # Replace the Elmer surface with the model's OWN forward of the true
        # cosy melt: a correct inverse must recover ratio 1.0 (machinery check,
        # isolates forward-model error from optimization/plumbing error).
        M_h = stubblefield_forward_multiplier(
            2 * ny, 2 * nx, dx, dy, p.H0, u, v,
            eta_bar=args.eta_bar, alpha_scale=args.alpha_scale,
            rho_i=p.RHO_I, rho_w=p.RHO_W, g=p.G)
        fwd = StubblefieldForward(M_h, ny, nx)
        k0 = 2.0 * math.pi / mm["wavelength"]
        m_true = mm["amp"] * np.cos(k0 * Y)
        with torch.no_grad():
            dzs = fwd(torch.from_numpy(m_true)).cpu().numpy()
        print("  [self-test] fitting the model's own synthetic forward")

    print(f"== Level-B variational inverse: {args.pert} (rep={args.rep}, "
          f"u={u:.0f} m/yr, lam={args.lam:g}, eta_bar={args.eta_bar:g}) ==")
    print(f"  observed |dZs| max {np.nanmax(np.abs(dzs)):.3f} m")
    res = variational_melt_inverse(
        dzs.astype(np.float64), mask, dx, dy, p.H0, u, v,
        rep=args.rep, eta_bar=args.eta_bar, alpha_scale=args.alpha_scale,
        lam=args.lam, iters=args.iters, lr=args.lr,
        rho_i=p.RHO_I, rho_w=p.RHO_W, g=p.G,
        log_every=max(1, args.iters // 5))
    m_rec = res.melt

    # score: cosine-fit the recovered melt across-flow (y), ratio to truth amp.
    import xarray as xr
    m_da = xr.DataArray(m_rec, dims=("y", "x"), coords={"y": y, "x": x})
    k = 2.0 * math.pi / mm["wavelength"]
    coord, prof = profile_1d(m_da, "y", sci)
    fit_c = cosine_fit(prof, coord, k)
    ratio = fit_c["amp"] / mm["amp"]
    shift = float(np.angle(np.exp(1j * (fit_c["phase"] - np.pi)))) / k
    print(f"\n  recovered across-flow cosine amp {fit_c['amp']:.4f} "
          f"(truth {mm['amp']:.4f})  ratio {ratio:.3f}  r2 {fit_c['r2']:.3f}  "
          f"shift {shift:+.0f} m")
    print(f"  >>> Level A +invert gave 1.538 here; forward-fit ratio = "
          f"{ratio:.3f} <<<")


if __name__ == "__main__":
    main()
