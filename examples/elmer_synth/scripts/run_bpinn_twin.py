"""Fit the space-time B-PINN to a packaged DEM-stack twin and score it (jaxmelt env).

    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -u \
        elmer_synth/scripts/run_bpinn_twin.py [--tag multixy_pigreal] [--steps 20000] \
        [--ensemble 4] [--hmc 0] [--sigma-h 2] [--sigma-r 1] [--col-slices 24] \
        [--transfer] [--eta 1e14]

Scores the MAP / posterior mean against the prescribed truth (nrmse, corr, bias,
2-sigma coverage) next to the Eulerian / Lagrangian benchmarks packaged by
prep_bpinn_twin.py, over the full domain and again in an interior window
(``--edge-km``), and reports the amplitude at the truth's channel wavelengths
(1.0 / 1.5 km for the multixy twins), measured along the axis the prescribed
melt varies in as recorded in the npz (``truth_axis``: x for the ``multicos``
xy twins, y for a y-only twin).

The twin itself is a local dataset written by ``prep_bpinn_twin.py`` into
``<checkout>/examples/elmer_synth/results/bpinn`` -- the first directory searched
here, so the two drivers always name the same file; the analysis host's absolute
path is kept as a second entry for twins packaged before that. From a clean clone
this script documents the recipe below rather than being runnable, and exits with
that message when the twin is absent.

The recorded twin result -- corr 0.231 without the transfer, 0.693 with it
(nrmse 0.995 -> 0.764; measured 2026-09-13 on the packaging in this tree) -- is
the noise-free tier. Package it with ``prep_bpinn_twin.py multixy_pigreal
multixy_bmb clean`` (writes ``twin_multixy_pigreal_clean.npz``) and score it
once each way:

    run_bpinn_twin.py --tag multixy_pigreal_clean --transfer --sigma-h 0.5 \
        --no-planes --sigma-r 1 --col-slices 24 --steps 10000 --ensemble 1

The recorded number is the single-member ``B-PINN MAP`` row of the full-domain
block, hence ``--ensemble 1``. The default ``multixy_pigreal`` tag is the
tilt-corrected, error-injected tier, which is noise-limited here (corr about 0).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO = __import__("pathlib").Path(__file__).resolve().parents[3]
sys.path.insert(0, f"{_REPO}/src")
from stereo_melt.dynamics.bpinn import BPINNConfig, fit_bpinn, prepare_bpinn_data  # noqa: E402

_TWIN_DIRS = tuple(dict.fromkeys((
    f"{_REPO}/examples/elmer_synth/results/bpinn",
    "/wd2/projects/stereo_melt/examples/elmer_synth/results/bpinn",
)))


def _load_twin(tag):
    """``(npz, results_dir)`` for ``twin_<tag>.npz``, preferring this checkout's.

    This checkout's directory is where ``prep_bpinn_twin.py`` writes, so the
    recipe's two halves resolve to the same file; the analysis-host path is the
    fallback for twins packaged before that. Scores and figures are written back
    beside the twin that was scored.
    """
    for d in _TWIN_DIRS:
        path = f"{d}/twin_{tag}.npz"
        if os.path.exists(path):
            return np.load(path), d
    raise SystemExit(
        f"twin_{tag}.npz not found (looked in: " + ", ".join(_TWIN_DIRS) + "). "
        "Scoring the B-PINN needs a packaged twin from the elmer_synth twin tier on the "
        "analysis host -- the Elmer runs and the synthetic DEM stacks it is built from are a "
        "local dataset that is not part of the repository. From a clean clone this script "
        "documents the validation recipe rather than being runnable; on the analysis host, "
        "package the twin first with `prep_bpinn_twin.py multixy_pigreal multixy_bmb clean`."
    )


def _amp_at(m, x, y, domain, lam, axis="xy"):
    """Amplitude of the domain-masked field at wavelength ``lam`` (m).

    Measured along the axis the prescribed melt varies in, as recorded in the
    npz: a ``y`` twin is averaged over x and projected onto y, anything else
    over y and onto x. ``xy`` (the ``multicos`` twins, which superpose
    components on both axes) keeps the x projection, where the 1.0 / 1.5 km
    channels the caller asks about live.
    """
    fin = np.isfinite(m) & domain
    prof = np.nanmean(np.where(fin, m, np.nan), axis=(1 if axis == "y" else 0))
    prof = prof - np.nanmean(prof)
    coord = (y if axis == "y" else x)
    ok = np.isfinite(prof)
    cc = coord - coord[0]
    return 2 * np.abs(np.mean(prof[ok] * np.exp(-1j * 2 * np.pi / lam * cc[ok])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="multixy_pigreal")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--hmc", type=int, default=0)
    ap.add_argument("--hmc-joint", action="store_true")
    ap.add_argument("--sigma-h", type=float, default=2.0)
    ap.add_argument("--sigma-r", type=float, default=1.0,
                    help="physics residual sd (m/yr); the steady twins were validated at 1 with 24 slices")
    ap.add_argument("--col-slices", type=int, default=24,
                    help="nominal physics observations = domain cells x this; the steady twins were validated "
                         "at --sigma-r 1 --col-slices 24, while the library defaults (20 / 4) are the pair that "
                         "survives a real stack")
    ap.add_argument("--nu", type=float, default=4.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--melt-scales", default="1,2,4,8")
    ap.add_argument("--edge-km", type=float, default=3.0,
                    help="also score an interior window this far from every edge (Elmer inflow and calving-front boundary layers dominate full-domain scores)")
    ap.add_argument("--melt-t-scales", default="",
                    help="time bands (yr) for a time-dependent melt b(x,y,t), e.g. 0.5,1,2,4; empty = steady melt; the map is the window mean")
    ap.add_argument("--xy-scales", default="0.5,1,2,4,8")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--eta", type=float, default=1e14)
    ap.add_argument("--alpha", type=float, default=0.34)
    ap.add_argument("--bg-sigma-H", type=float, default=3.0)
    ap.add_argument("--n-bins", type=int, default=1,
                    help="local transfer: bins on (H, ux, uy) with partition-of-unity blending; 1 = one reference geometry")
    ap.add_argument("--blend-px", type=float, default=8.0, help="Gaussian blend width of the bin weights (px)")
    ap.add_argument("--batch-epochs", type=int, default=16)
    ap.add_argument("--no-planes", action="store_true")
    ap.add_argument("--H-scale", type=float, default=20.0)
    ap.add_argument("--no-base-field", action="store_true")
    args = ap.parse_args()

    z, R = _load_twin(args.tag)
    H_obs, x, y, t_yr, vx, vy, truth = (z[k] for k in ("H_obs", "x", "y", "t_yr", "vx", "vy", "truth"))
    print(f"twin {args.tag}: stack {H_obs.shape}, dx {x[1] - x[0]:.0f} m, {t_yr.min():.2f}..{t_yr.max():.2f}, "
          f"finite {np.isfinite(H_obs).mean():.2f}, |v| {np.hypot(vx, vy).mean():.0f} m/yr")
    data = prepare_bpinn_data(H_obs, x, y, t_yr, vx, vy, rho_i=float(z["rho_i"]), rho_w=float(z["rho_w"]))
    cfg = BPINNConfig(n_steps=args.steps, ensemble=args.ensemble, hmc_samples=args.hmc, hmc_joint=args.hmc_joint,
                      sigma_h_m=args.sigma_h, sigma_r_myr=args.sigma_r, nu=(None if args.nu <= 0 else args.nu),
                      n_col_slices=args.col_slices, hidden=args.hidden, layers=args.layers,
                      melt_scales_km=tuple(float(s) for s in args.melt_scales.split(",")),
                      melt_t_scales_yr=(tuple(float(s) for s in args.melt_t_scales.split(","))
                                        if args.melt_t_scales else None),
                      xy_scales_km=tuple(float(s) for s in args.xy_scales.split(",")),
                      transfer=args.transfer, eta_bar=args.eta, alpha_scale=args.alpha, n_bins=args.n_bins, blend_px=args.blend_px,
                      transfer_bg_sigma_H=args.bg_sigma_H, batch_epochs=args.batch_epochs,
                      epoch_planes=not args.no_planes, H_scale_m=args.H_scale,
                      base_field=not args.no_base_field)
    t0 = time.time()
    res = fit_bpinn(data, cfg)
    print(f"fit time {time.time() - t0:.0f} s")
    tb = np.asarray(res.extras["transfer_bins"], float).reshape(-1, 3)
    print(f"transfer operator: {res.extras['n_bins_effective']} bin(s) built of {cfg.n_bins} requested"
          + "".join(f"\n  bin {i}: H {b[0]:.0f} m, u ({b[1]:+.0f}, {b[2]:+.0f}) m/yr"
                    for i, b in enumerate(tb)))

    def score(m, name, sd=None, region=None):
        fin = np.isfinite(m) & np.isfinite(truth) & data.domain
        if region is not None:
            fin &= region
        e = m[fin] - truth[fin]
        line = (f"  {name:22s} nrmse {np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(truth[fin] ** 2)):.3f}  "
                f"corr {np.corrcoef(m[fin], truth[fin])[0, 1]:.3f}  bias {e.mean():+.2f} m/yr  rms {np.sqrt(np.mean(e ** 2)):.2f}")
        if sd is not None and np.any(sd > 0):
            zz = e / np.maximum(sd[fin], 1e-6)
            line += f"  | 2σ coverage {np.mean(np.abs(zz) < 2) * 100:.0f}%  z-std {zz.std():.2f}  median σ {np.median(sd[fin]):.2f}"
        print(line)

    print("=== scores vs truth (domain = observed pixels) ===")
    for k in ("bench_eulerian", "bench_lagrangian"):
        if k in z:
            score(z[k], k.replace("bench_", "") + " (production)")
    score(res.map_melt, "B-PINN MAP")
    if res.samples.shape[0] > 1:
        score(res.melt_mean, "B-PINN posterior mean", res.melt_sd)
    if args.edge_km > 0:
        X, Y = np.meshgrid(x, y)
        e_m = 1000.0 * args.edge_km
        inner = ((X >= x.min() + e_m) & (X <= x.max() - e_m)
                 & (Y >= y.min() + e_m) & (Y <= y.max() - e_m))
        print(f"=== interior window ({args.edge_km:g} km from every edge) ===")
        for k in ("bench_eulerian", "bench_lagrangian"):
            if k in z:
                score(z[k], k.replace("bench_", "") + " (production)", region=inner)
        score(res.map_melt, "B-PINN MAP", region=inner)
        if res.samples.shape[0] > 1:
            score(res.melt_mean, "B-PINN posterior mean", res.melt_sd, region=inner)
    truth_axis = str(z["truth_axis"]) if "truth_axis" in z else "xy"
    amp_axis = "y" if truth_axis == "y" else "x"
    print(f"amplitude along {amp_axis} (the truth's axis, {truth_axis!r}) "
          f"at λ=1.0 km / 1.5 km:")
    for name, m in (("truth", truth), ("Eulerian", z.get("bench_eulerian")),
                    ("Lagrangian", z.get("bench_lagrangian")), ("B-PINN mean", res.melt_mean)):
        if m is not None:
            print(f"  {name:12s} {_amp_at(m, x, y, data.domain, 1000, truth_axis):.1f} / "
                  f"{_amp_at(m, x, y, data.domain, 1500, truth_axis):.1f}")
    np.savez_compressed(f"{R}/bpinn_{args.tag}{args.out_suffix}.npz", mean=res.melt_mean, sd=res.melt_sd,
                        samples=res.samples, map=res.map_melt, obs_rms=res.obs_rms_m, loss=res.loss_history,
                        transfer_bins=tb, n_bins_effective=res.extras["n_bins_effective"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("truth", truth), ("Eulerian (production)", z.get("bench_eulerian")),
              ("Lagrangian (production)", z.get("bench_lagrangian")), ("B-PINN posterior mean", res.melt_mean),
              ("B-PINN posterior σ", res.melt_sd), ("B-PINN mean − truth", res.melt_mean - truth)]
    fig, axs = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)
    for ax, (name, m) in zip(axs.ravel(), panels):
        if m is None:
            ax.set_axis_off()
            continue
        is_sd = "σ" in name
        im = ax.imshow(np.where(data.domain, m, np.nan), origin="upper",
                       extent=[x[0] / 1e3, x[-1] / 1e3, y[-1] / 1e3, y[0] / 1e3],
                       cmap="viridis" if is_sd else "RdBu_r", vmin=0 if is_sd else -25, vmax=None if is_sd else 25)
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"PIGREAL twin {args.tag}: prescribed melt (Shean sign) vs solvers; B-PINN {res.samples.shape[0]} "
                 f"samples, obs rms {res.obs_rms_m:.2f} m", fontsize=11)
    out = f"{R}/bpinn_{args.tag}{args.out_suffix}.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
