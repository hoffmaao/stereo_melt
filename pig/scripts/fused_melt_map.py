"""Apply the fused (budget ⊕ dfNN) melt inverse to the PIG stack + map it.

Standalone diagnostic (the fused solver is not yet wired into pig/run_melt.py).
Reuses pig.run_melt's loaders for the stack / velocity / floating mask / SMB /
firn, then runs four reconstructions on the SAME inputs:

  Eulerian (hydrostatic) · budget lin-inv eulerian (→ the fused level m_prior) ·
  variational (DC-blind, high-pass) · fused (budget-anchored variational).

No truth exists for PIG, so this is a self-consistency / behavior map, not a
score: does the fusion carry the budget's absolute level while sharpening
channel structure, or (as on the Elmer localized patch) inject a spurious
large-scale mode?

2026-07-25: the fused leg now runs the BAND-PASS projection contract
(anchor_lp_sigma_H=4, short cut 2.5H): the budget prior owns the level, the
long wavelengths, and the sub-transfer-floor scales; the fit shapes only the
usable middle band and its residual is blinded to match. The kept-band
fit_var_explained is the no-truth channel detector (E2a: ~0.7 = real channel
signal, ~0.1 = correction is noise, trust the prior). The 07-24 run (no
contract) had fitVE -5.28 with the fit swamped by ~50 m RMS of large-scale
reference topography a degree-1 background cannot represent -- exactly the
band now blinded.

Env: PIG_FUSED_ITERS (default 4000; set ~400 for a plumbing smoke),
     PIG_VELOCITY (default measures; production PIG uses fused velocity).

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    PIG_FUSED_ITERS=400 $PY <this>.py           # smoke
    $PY <this>.py                                # full
"""
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _env_proj

REPO = "/wd2/projects/stereo_melt"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "stereo_melt", "src"))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pig.run_melt as R  # noqa: E402  (library-before-torch; brings loaders)
from pig import config  # noqa: E402
from stereo_melt.colormaps import (  # noqa: E402
    add_melt_colorbar, melt_cmap, melt_norm,
)
from stereo_melt.melt import eulerian_melt_rate  # noqa: E402
from stereo_melt.dynamics import linear_inverse_eulerian_budget_melt_rate  # noqa: E402
from stereo_melt.io.bedmachine import load_firn_on_grid  # noqa: E402

STACK_PREFIX = "pig_stack_250m_is2ctempo"
RHO = dict(rho_i=918.0, rho_w=1027.0)
ITERS = int(os.environ.get("PIG_FUSED_ITERS", "4000"))
TAG = os.environ.get("PIG_FUSED_TAG", "").strip()
TAG = (f"_{TAG}" if TAG and not TAG.startswith("_") else TAG)
OUT_FIG = f"/wd2/projects/stereo_melt/pig/figures/pig_fused_melt_map{TAG}.png"
OUT_NC = f"/wd2/projects/stereo_melt/pig/results/pig_fused_melt_map{TAG}.nc"


PANELS = ["Eulerian", "budget lininv eul", "variational", "fused"]
LABEL = {"Eulerian": "Eulerian (hydrostatic)",
         "budget lininv eul": "budget lin-inv (m_prior / level)",
         "variational": "variational (DC-blind)",
         "fused": "fused (budget ⊕ dfNN)"}


def _2d(m):
    return m.mean("time") if "time" in m.dims else m


def render_map(fields, xk, yk, out_fig, subtitle):
    """Draw the 4-panel figure from `fields` (name -> 2-D DataArray).

    Split out of :func:`main` so the figure can be re-rendered from the saved
    NetCDF (see ``pig/scripts/replot_melt_figures.py``) without re-running the
    solvers -- a colormap change should not cost an hour of torch fitting.

    All four panels share ONE LADDIE symmetric-log scale. The old figure had to
    give the variational panel its own linear range (its amplitude is an order
    of magnitude below the melt trio, so a shared +/-60 linear stretch rendered
    it blank); the log decades carry both amplitudes at once, so the panels are
    now directly comparable by eye.
    """
    mcmap = melt_cmap()
    mnorm = melt_norm(vmax=100.0)
    # Crop to the union footprint of the panels: the stack grid is the AOI, and
    # the shelf fills well under half of it, so an uncropped axis spends most of
    # its pixels on masked gray.
    cover = np.zeros((yk.size, xk.size), bool)
    for n in PANELS:
        cover |= np.isfinite(np.asarray(fields[n].values))
    jj, ii = np.where(cover)
    padx = 0.02 * (xk.max() - xk.min())
    pady = 0.02 * (yk.max() - yk.min())
    xlim = (xk[ii.min()] - padx, xk[ii.max()] + padx)
    ylim = (yk[jj.min()] - pady, yk[jj.max()] + pady)

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 11), sharex=True, sharey=True)
    for ax, n in zip(axs.ravel(), PANELS):
        fld = np.asarray(fields[n].values)
        im = ax.pcolormesh(xk, yk, fld, cmap=mcmap, norm=mnorm, shading="auto")
        ax.set_xlim(*xlim)
        ax.set_ylim(*sorted(ylim))
        va = fields[n].attrs.get("fit_var_explained")
        ttl = LABEL[n] + (f"\nvar_expl {float(va):.2f}" if va is not None else "")
        ax.set_title(ttl, fontsize=11)
        ax.set_aspect("equal")
        ax.set_facecolor("0.94")
        add_melt_colorbar(fig, im, ax=ax, shrink=0.85)
    for ax in axs[:, 0]:
        ax.set_ylabel("y (km, EPSG:3031)")
    for ax in axs[1, :]:
        ax.set_xlabel("x (km, EPSG:3031)")
    fig.suptitle(subtitle, fontsize=13)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_fig}", flush=True)


def main() -> int:
    print(f"[load] stack {STACK_PREFIX} ...", flush=True)
    stack = R.load_stack(stack_prefix=STACK_PREFIX)
    floating = R.load_floating_mask(stack)
    # Same geometry as the production run_melt products (2026-07-28): without
    # this the rig re-admitted the calved sector, so every leg was fitting an
    # ice-to-ocean cliff the shelf does not have.
    floating = R.apply_min_extent(floating, mask_suffix="_250m_is2ctempo")
    stack = stack.where(floating)
    vx, vy, vsrc = R.load_velocity_on_grid(stack)
    a_dot = R.load_smb_on_grid(stack)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)
    print(f"[load] time={stack.sizes['time']} y={stack.sizes['y']} "
          f"x={stack.sizes['x']}  vel='{vsrc}'  iters={ITERS}", flush=True)

    # DIAGNOSTIC: OLS dh/dt (robust_dh_dt=False) — the robust per-pixel
    # regression over 513 epochs is ~10x slower and this is a behavior map, not
    # the production melt. Both linear solvers use the same choice for a fair
    # cross-solver comparison. Set PIG_ROBUST_DHDT=1 for the production estimate.
    robust = os.environ.get("PIG_ROBUST_DHDT", "0") == "1"
    fields = {}
    print(f"[1/4] Eulerian (hydrostatic, robust_dh_dt={robust}) ...", flush=True)
    fields["Eulerian"] = _2d(eulerian_melt_rate(
        stack, vx, vy, a_dot=a_dot, d=firn, robust_dh_dt=robust, **RHO).melt_rate)

    print("[2/4] budget lin-inv eulerian (-> m_prior) ...", flush=True)
    budget = _2d(linear_inverse_eulerian_budget_melt_rate(
        stack, vx, vy, a_dot=a_dot, d=firn, floating_mask=floating,
        eta_bar=1e14, robust_dh_dt=robust, progress=True, **RHO).melt_rate)
    fields["budget lininv eul"] = budget

    from stereo_melt.dynamics.stubblefield_forward import variational_melt_rate

    print("[3/4] variational (DC-blind, high-pass) ...", flush=True)
    fields["variational"] = _2d(variational_melt_rate(
        stack, vx, vy, floating_mask=floating, d=firn, rep="grid",
        eta_bar=1e13, alpha_scale=0.34, lam=1e-4, iters=ITERS, lr=3e-3,
        sigma_hp_H=5.0, log_every=500, n_bins=8, blend_km=4.0, **RHO).melt_rate)

    # Momentum-balance effective viscosity (single-field TDDA inversion,
    # pig/scripts/infer_eta_icepack_pig.py): per-bin eta replaces the hand-set
    # scalar wherever the field exists; the scalar stays as fallback.
    #
    # 2026-07-28 PRODUCTION ADOPTION: the source is now the v4 DUAL-form
    # whole-shelf inversion (infer_eta_icepack2_pig.py, unbounded, min-extent
    # domain, 2 lobes) -- MAP rel vel misfit 1.9% against the v1 single-field
    # primal's 5.6%, and a median eta 1.33e14 vs the primal's 2.47e14 Pa s.
    # PIG_ETA_NPZ overrides (set it to pig_eta_field_250m.npz for the v1 A/B).
    eta_da = None
    ETA_NPZ = os.environ.get("PIG_ETA_NPZ") or os.path.join(
        REPO, "pig", "processed", "pig_eta_field_250m_dual_embayment.npz")
    if os.path.exists(ETA_NPZ):
        ed = np.load(ETA_NPZ)
        print(f"[eta] source {os.path.basename(ETA_NPZ)}", flush=True)
        eta_da = xr.DataArray(
            ed["eta_pas"], dims=("y", "x"),
            coords={"y": stack.y.values, "x": stack.x.values})
        print(f"[eta] momentum-balance eta_field: p10 "
              f"{np.nanpercentile(ed['eta_pas'], 10):.2e}  median "
              f"{np.nanmedian(ed['eta_pas']):.2e}  p90 "
              f"{np.nanpercentile(ed['eta_pas'], 90):.2e} Pa s "
              f"(MAP rel misfit {float(ed['rel_misfit_map']):.3f})", flush=True)

    print("[4/4] fused (budget-anchored variational) ...", flush=True)
    ds_fused = variational_melt_rate(
        stack, vx, vy, floating_mask=floating, d=firn, m_prior=budget,
        anchor_lp_sigma_H=4.0, bg_degree=1, eta_bar=1e13, alpha_scale=0.34,
        lam=1e-2, iters=ITERS, lr=3e-3, log_every=500, n_bins=8, blend_km=4.0,
        eta_field=eta_da, **RHO)
    fields["fused"] = _2d(ds_fused.melt_rate)
    fields["fused"].attrs.update(ds_fused.attrs)  # persist kept-band fit diagnostics
    print(f"    fused var_explained="
          f"{float(ds_fused.attrs.get('fit_var_explained', np.nan)):.3f}",
          flush=True)

    # ---- save fields ----
    os.makedirs(os.path.dirname(OUT_NC), exist_ok=True)
    xr.Dataset({k.replace(" ", "_"): v for k, v in fields.items()}).to_netcdf(OUT_NC)
    print(f"wrote {OUT_NC}", flush=True)

    # ---- map ----
    dhdt_lbl = "robust dh/dt" if robust else "OLS dh/dt (diagnostic)"
    render_map(
        fields, stack.x.values / 1e3, stack.y.values / 1e3, OUT_FIG,
        f"PIG basal melt — four reconstructions on the same 250 m stack "
        f"({stack.sizes['time']} epochs, {vsrc.split('(')[0].strip()}, "
        f"{dhdt_lbl})")

    print("\n  solver               median  p10    p90   (m/yr, floating)")
    for n in PANELS:
        v = np.asarray(fields[n].values)
        v = v[np.isfinite(v)]
        print(f"  {n:<20s} {np.median(v):>6.2f} {np.percentile(v, 10):>6.2f} "
              f"{np.percentile(v, 90):>6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
