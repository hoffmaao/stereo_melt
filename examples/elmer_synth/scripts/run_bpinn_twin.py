"""Fit the space-time B-PINN to a packaged DEM-stack twin and score it (jaxmelt env).

    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -u \
        elmer_synth/scripts/run_bpinn_twin.py [--tag multixy_pigreal] [--steps 20000] \
        [--ensemble 4] [--hmc 0] [--sigma-h 2] [--sigma-r 1] [--transfer] [--eta 1e14]

Scores the MAP / posterior mean against the prescribed truth (nrmse, corr, bias,
2-sigma coverage) next to the Eulerian / Lagrangian benchmarks packaged by
prep_bpinn_twin.py, and reports the along-flow amplitude at the truth's channel
wavelengths (1.0 / 1.5 km for the multixy twins).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/wd2/projects/stereo_melt/src")
from stereo_melt.dynamics.bpinn import BPINNConfig, fit_bpinn, prepare_bpinn_data  # noqa: E402

R = "/wd2/projects/stereo_melt/examples/elmer_synth/results/bpinn"


def _amp_at(m, x, domain, lam):
    """Along-x amplitude of the domain-masked field at wavelength ``lam`` (m)."""
    fin = np.isfinite(m) & domain
    row = np.nanmean(np.where(fin, m, np.nan), axis=0)
    row = row - np.nanmean(row)
    ok = np.isfinite(row)
    xx = x - x[0]
    return 2 * np.abs(np.mean(row[ok] * np.exp(-1j * 2 * np.pi / lam * xx[ok])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="multixy_pigreal")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--hmc", type=int, default=0)
    ap.add_argument("--hmc-joint", action="store_true")
    ap.add_argument("--sigma-h", type=float, default=2.0)
    ap.add_argument("--sigma-r", type=float, default=1.0)
    ap.add_argument("--nu", type=float, default=4.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--melt-scales", default="1,2,4,8")
    ap.add_argument("--xy-scales", default="0.5,1,2,4,8")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--eta", type=float, default=1e14)
    ap.add_argument("--alpha", type=float, default=0.34)
    ap.add_argument("--bg-sigma-H", type=float, default=3.0)
    ap.add_argument("--batch-epochs", type=int, default=16)
    ap.add_argument("--no-planes", action="store_true")
    ap.add_argument("--H-scale", type=float, default=20.0)
    ap.add_argument("--no-base-field", action="store_true")
    args = ap.parse_args()

    z = np.load(f"{R}/twin_{args.tag}.npz")
    H_obs, x, y, t_yr, vx, vy, truth = (z[k] for k in ("H_obs", "x", "y", "t_yr", "vx", "vy", "truth"))
    print(f"twin {args.tag}: stack {H_obs.shape}, dx {x[1] - x[0]:.0f} m, {t_yr.min():.2f}..{t_yr.max():.2f}, "
          f"finite {np.isfinite(H_obs).mean():.2f}, |v| {np.hypot(vx, vy).mean():.0f} m/yr")
    data = prepare_bpinn_data(H_obs, x, y, t_yr, vx, vy, rho_i=float(z["rho_i"]), rho_w=float(z["rho_w"]))
    cfg = BPINNConfig(n_steps=args.steps, ensemble=args.ensemble, hmc_samples=args.hmc, hmc_joint=args.hmc_joint,
                      sigma_h_m=args.sigma_h, sigma_r_myr=args.sigma_r, nu=(None if args.nu <= 0 else args.nu),
                      hidden=args.hidden, layers=args.layers,
                      melt_scales_km=tuple(float(s) for s in args.melt_scales.split(",")),
                      xy_scales_km=tuple(float(s) for s in args.xy_scales.split(",")),
                      transfer=args.transfer, eta_bar=args.eta, alpha_scale=args.alpha,
                      transfer_bg_sigma_H=args.bg_sigma_H, batch_epochs=args.batch_epochs,
                      epoch_planes=not args.no_planes, H_scale_m=args.H_scale,
                      base_field=not args.no_base_field)
    t0 = time.time()
    res = fit_bpinn(data, cfg)
    print(f"fit time {time.time() - t0:.0f} s")

    def score(m, name, sd=None):
        fin = np.isfinite(m) & np.isfinite(truth) & data.domain
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
    print("along-flow amplitude at λ=1.0 km / 1.5 km (truth 15 / 10 m/yr):")
    for name, m in (("Eulerian", z.get("bench_eulerian")), ("Lagrangian", z.get("bench_lagrangian")),
                    ("B-PINN mean", res.melt_mean)):
        if m is not None:
            print(f"  {name:12s} {_amp_at(m, x, data.domain, 1000):.1f} / {_amp_at(m, x, data.domain, 1500):.1f}")
    np.savez_compressed(f"{R}/bpinn_{args.tag}{args.out_suffix}.npz", mean=res.melt_mean, sd=res.melt_sd,
                        samples=res.samples, map=res.map_melt, obs_rms=res.obs_rms_m, loss=res.loss_history)

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
