"""Fit the B-PINN to the packaged PIG trunk window (jaxmelt env, GPU).

    CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=<pip nvidia libs>:$LD_LIBRARY_PATH \
      XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -u pig/run_bpinn.py [--tag is2ctempo_sheltilt_qcey]
      [--half all|A|B] [--steps 10000] [--ensemble 1] [--eta 1e13] [--no-transfer]

``--half A|B`` fits alternate epochs (sorted by time) for the half-stack noise
floor, the same split ``run_noise_floor`` uses. Scores the MAP / posterior mean
against the production Eulerian, Lagrangian and monolithic v2 on the window
(like-for-like fluxes on the common pixels, correlations, accretion area) and
writes ``results/bpinn/bpinn_trunk_<tag>_<half><suffix>.{npz,png}``.

Read the two diagnostics printed per fit before believing anything: the
surrogate's fitted ``time trend`` must be of the order of the observed thinning
(about -3 to -6 m/yr median on the trunk; 0 means the fit collapsed to the
static base field and the melt is just the steady budget of the median map),
and ``--no-transfer`` must change the answer. The 2026-09-13 defaults
(``--sigma-r 20 --col-slices 4``) pass both; ``--sigma-r 5 --col-slices 24``
collapsed. Uncertainty: ``--ensemble N`` (epoch bootstrap) gives a σ map that
under-reads the independent half-stack σ by about 1.3-1.8x; the halves are the
honest per-pixel σ.

Every trunk number recorded so far -- the fluxes, the correlations and that σ
ratio -- predates the transfer-orientation fixes and is a diagnostic only. The
transfer-on runs in particular used an operator whose along/across damping was
swapped at 2H, so repeat them with the corrected operator before quoting any of
it. Validated numbers come from the DEM-stack twin, where the truth is known:
``examples/elmer_synth/scripts/run_bpinn_twin.py``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from stereo_melt.dynamics.bpinn import BPINNConfig, fit_bpinn, prepare_bpinn_data  # noqa: E402

RHO_I_FLUX = 918.0
R = Path(__file__).resolve().parent / "results" / "bpinn"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="is2ctempo_sheltilt_qcey")
    ap.add_argument("--half", default="all", choices=["all", "A", "B"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--hmc", type=int, default=0)
    ap.add_argument("--eta", type=float, default=1e13)
    ap.add_argument("--alpha", type=float, default=0.34)
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--n-bins", type=int, default=1,
                    help="local transfer: bins on (H, ux, uy) with partition-of-unity blending; 1 = one reference geometry")
    ap.add_argument("--blend-px", type=float, default=8.0, help="Gaussian blend width of the bin weights (px)")
    ap.add_argument("--no-planes", action="store_true")
    ap.add_argument("--sigma-h", type=float, default=None, help="obs scale (thickness m); default = measured NMAD")
    ap.add_argument("--sigma-r", type=float, default=20.0,
                    help="physics-residual tolerance (m/yr); 5 over-weights the physics on PIG and the surrogate collapses to a static field")
    ap.add_argument("--b-scale", type=float, default=50.0, help="melt-network output scale (m/yr); PIG trunk melt is O(50)")
    ap.add_argument("--col-slices", type=int, default=4, help="nominal physics observations = domain cells x this (4 validated on PIG; 24 collapses)")
    ap.add_argument("--batch-epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--melt-scales", default="1,2,4,8")
    ap.add_argument("--xy-scales", default="0.5,1,2,4,8")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--H-scale", type=float, default=100.0)
    ap.add_argument("--no-base-field", action="store_true")
    args = ap.parse_args()

    z = np.load(R / f"trunk_{args.tag}.npz")
    H, x, y, t = z["H_obs"].astype(np.float32), z["x"], z["y"], z["t_yr"]
    vx, vy = z["vx"], z["vy"]
    vt = z["vt_yr"] if z["vt_yr"].size else None
    dom = z["domain"].astype(bool)
    order = np.argsort(t)
    if args.half != "all":
        keep = order[0::2] if args.half == "A" else order[1::2]
        H, t = H[keep], t[keep]
    sig = args.sigma_h if args.sigma_h else float(z["sigma_H_nmad"])
    print(f"PIG trunk {args.tag} half={args.half}: stack {H.shape}, finite {np.isfinite(H).mean():.2f}, "
          f"sigma_h {sig:.1f} m, velocity {'time-varying' if vt is not None else 'static'}, transfer {not args.no_transfer}")
    data = prepare_bpinn_data(H, x, y, t, vx, vy, a_dot=z["a_dot"], domain=dom,
                              rho_i=float(z["rho_i"]), rho_w=float(z["rho_w"]), vt_yr=vt)
    cfg = BPINNConfig(n_steps=args.steps, ensemble=args.ensemble, hmc_samples=args.hmc, sigma_h_m=sig,
                      sigma_r_myr=args.sigma_r, transfer=not args.no_transfer, eta_bar=args.eta, alpha_scale=args.alpha, n_bins=args.n_bins, blend_px=args.blend_px,
                      batch_epochs=args.batch_epochs, epoch_planes=not args.no_planes, H_scale_m=args.H_scale, base_field=not args.no_base_field, b_scale_myr=args.b_scale, n_col_slices=args.col_slices, lr=args.lr,
                      melt_scales_km=tuple(float(s) for s in args.melt_scales.split(",")),
                      xy_scales_km=tuple(float(s) for s in args.xy_scales.split(",")))
    t0 = time.time()
    res = fit_bpinn(data, cfg)
    print(f"fit time {time.time() - t0:.0f} s")
    tb = np.asarray(res.extras["transfer_bins"], float).reshape(-1, 3)
    print(f"transfer operator: {res.extras['n_bins_effective']} bin(s) built of {cfg.n_bins} requested"
          + "".join(f"\n  bin {i}: H {b[0]:.0f} m, u ({b[1]:+.0f}, {b[2]:+.0f}) m/yr"
                    for i, b in enumerate(tb)))

    fields = {"B-PINN": res.melt_mean, "Eulerian": z["bench_eulerian"], "Lagrangian": z["bench_lagrangian"],
              "monolithic v2": z["bench_monolithic_v2"], "restored local+Helm": z["bench_restored_local_helm"]}
    common = dom.copy()
    for v in fields.values():
        common &= np.isfinite(v)
    def flux(m):
        return float(-np.nansum(np.where(common, m, 0.0)) * 250 * 250 * RHO_I_FLUX / 1e12)
    print(f"=== window scores on {common.sum()} common pixels ===")
    for k, v in fields.items():
        print(f"  {k:22s} flux {flux(v):6.1f} Gt/yr | median {np.nanmedian(v[common]):+6.1f} | accretion>+2 {(v[common] > 2).mean()*100:4.1f}% | "
              f"corr vs Eulerian {np.corrcoef(v[common], fields['Eulerian'][common])[0,1]:.3f} | vs mono {np.corrcoef(v[common], fields['monolithic v2'][common])[0,1]:.3f}")
    if res.samples.shape[0] > 1:
        print(f"  posterior sd: median {np.nanmedian(res.melt_sd[common]):.1f} m/yr; |mean|>2sd on {(np.abs(res.melt_mean[common]) > 2*res.melt_sd[common]).mean()*100:.0f}% of pixels")
    stem = R / f"bpinn_trunk_{args.tag}_{args.half}{args.out_suffix}"
    np.savez_compressed(str(stem) + ".npz", mean=res.melt_mean, sd=res.melt_sd, samples=res.samples, map=res.map_melt,
                        obs_rms=res.obs_rms_m, loss=res.loss_history, common=common,
                        transfer_bins=tb, n_bins_effective=res.extras["n_bins_effective"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    try:
        from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
        cmap, norm = melt_cmap(), melt_norm(vmax=300)
    except Exception:
        cmap, norm = "RdBu_r", None
    ext = [x[0] / 1e3, x[-1] / 1e3, y[-1] / 1e3, y[0] / 1e3]
    panels = [("B-PINN posterior mean" if res.samples.shape[0] > 1 else "B-PINN MAP", res.melt_mean),
              ("Eulerian (production)", z["bench_eulerian"]), ("Lagrangian (production)", z["bench_lagrangian"]),
              ("monolithic v2 (production)", z["bench_monolithic_v2"]),
              ("B-PINN − Eulerian", res.melt_mean - z["bench_eulerian"]), ("B-PINN − monolithic v2", res.melt_mean - z["bench_monolithic_v2"])]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    im = imd = None
    for ax, (name, m) in zip(axs.ravel(), panels):
        mm = np.where(common, m, np.nan)
        if name.startswith("B-PINN −"):
            imd = ax.imshow(mm, origin="upper", extent=ext, cmap="RdBu_r", vmin=-100, vmax=100)
            ax.set_title(f"{name}   rms {np.sqrt(np.nanmean(mm**2)):.0f} m/yr", fontsize=10)
        else:
            im = ax.imshow(mm, origin="upper", extent=ext, cmap=cmap, norm=norm) if norm is not None else ax.imshow(mm, origin="upper", extent=ext, cmap="RdBu_r", vmin=-100, vmax=100)
            ax.set_title(f"{name}\n{flux(m):.1f} Gt/yr like-for-like, accretion {(m[common] > 2).mean()*100:.0f}%", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    # 4 melt panels + 2 difference panels do not partition by row: monolithic v2 sits at
    # axs[1, 0], so each colorbar spans its own panels rather than a whole row.
    melt_axes = axs[0].tolist() + [axs[1, 0]]
    diff_axes = [axs[1, 1], axs[1, 2]]
    try:
        add_melt_colorbar(fig, im, ax=melt_axes, shrink=0.8, pad=0.01, label="ḃ (m ice a⁻¹), negative = melt")
    except Exception:
        fig.colorbar(im, ax=melt_axes, shrink=0.8)
    fig.colorbar(imd, ax=diff_axes, shrink=0.8, pad=0.01, label="Δ (m a⁻¹, ±100)")
    fig.suptitle(f"PIG trunk {args.tag} ({args.half}): B-PINN {'with' if not args.no_transfer else 'without'} bridging transfer "
                 f"(η {args.eta:.0e}), {res.samples.shape[0]} sample(s), surrogate obs rms {res.obs_rms_m:.1f} m", fontsize=11)
    fig.savefig(str(stem) + ".png", dpi=150)
    print("wrote", str(stem) + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
