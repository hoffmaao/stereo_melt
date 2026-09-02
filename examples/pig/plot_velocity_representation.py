"""Paper figure: velocity representation is first-order for trajectory solvers.

Four panels on the PIG 250 m is2ctempo grid, 2010-2024 window:

  (a) production path-solver melt, TIME-VARYING quarterly velocity
  (b) same solver/config, velocity collapsed to the TIME-MEAN
  (c) difference (b - a): where the mean-velocity approximation lands
  (d) the mechanism: temporal sigma of quarterly speed (nonstationarity),
      with mean-speed contours

Inputs (already on disk, 2026-07-02/03):
  results/pig_melt_250m_is2ctempo_parcellsq_*.nc      (tv production baseline)
  results/pig_diag_path_meanvel_250m_is2ctempo_*.nc   (mean-velocity rerun)
  ASE quarterly mosaics via PIG_VELOCITY=ase-quarterly (37 quarters, NaN-aware)

Flux numbers quoted from pig/logs/diag_path_meanvel.log (identical GL-2km
clip gate): tv 88.6 Gt/yr vs mean-vel 125.2 Gt/yr on the tv gate (+41%).

Run:
    cd /wd2/projects/stereo_melt/examples
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u \
        -m pig.plot_velocity_representation
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

os.environ["PIG_VELOCITY"] = "ase-quarterly"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pig import config
from pig.run_melt import load_grounded_mask, load_velocity_on_grid

TV_NC = (
    config.RESULTS_DIR
    / "pig_melt_250m_is2ctempo_parcellsq_2010-01-01_2024-01-10.nc"
)
MV_NC = (
    config.RESULTS_DIR
    / "pig_diag_path_meanvel_250m_is2ctempo_2010-01-01_2024-01-10.nc"
)
FLUX_TV, FLUX_MV = 88.6, 125.2  # Gt/yr, identical GL-2km clip gate

MELT_LIM = 30.0   # m ice/yr, symmetric; saturates the GL channel on purpose
DIFF_LIM = 15.0
MIN_QUARTERS = 8  # cells need >= this many finite quarters for sigma


def km(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, float) / 1e3


def main() -> None:
    tv_ds = xr.open_dataset(TV_NC)
    mv_ds = xr.open_dataset(MV_NC)
    tv = tv_ds.melt_rate_lagrangian
    mv = mv_ds.melt_rate.interp_like(tv, method="nearest")
    flo = np.asarray(tv_ds.floating_mask.values, bool)
    x, y = tv["x"].values, tv["y"].values
    extent = [km(x)[0], km(x)[-1], km(y)[-1], km(y)[0]]

    print("Loading quarterly velocity (NaN-aware)...")
    vx, vy, src = load_velocity_on_grid(tv)
    if "time" not in vx.dims or vx.sizes["time"] < 2:
        raise SystemExit(f"expected quarterly stack, got {src}")
    speed = np.hypot(vx.values, vy.values)  # (t, y, x)
    n_fin = np.isfinite(speed).sum(axis=0)
    with np.errstate(invalid="ignore"):
        sig = np.nanstd(speed, axis=0)
        mean_speed = np.nanmean(speed, axis=0)
    sig = np.where(n_fin >= MIN_QUARTERS, sig, np.nan)
    print(f"  {vx.sizes['time']} quarters ({src})")

    a = np.where(flo, np.asarray(tv.values, float), np.nan)
    b = np.where(flo, np.asarray(mv.values, float), np.nan)
    d = b - a
    both = np.isfinite(a) & np.isfinite(b)
    med_d = np.median(d[both])
    print(f"  common cells {both.sum():,}; median diff {med_d:+.2f} m/yr")

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.6), constrained_layout=True)
    (axa, axb), (axc, axd) = axes

    ima = axa.imshow(a, cmap="RdBu_r", vmin=-MELT_LIM, vmax=MELT_LIM,
                     extent=extent, interpolation="nearest")
    axb.imshow(b, cmap="RdBu_r", vmin=-MELT_LIM, vmax=MELT_LIM,
               extent=extent, interpolation="nearest")
    imc = axc.imshow(d, cmap="PuOr", vmin=-DIFF_LIM, vmax=DIFF_LIM,
                     extent=extent, interpolation="nearest")
    grounded = np.asarray(load_grounded_mask(tv).values, bool)
    ice = flo | grounded
    imd = axd.imshow(np.where(ice, sig, np.nan),
                     cmap="viridis", vmin=0,
                     vmax=float(np.ceil(np.nanpercentile(sig[flo], 98) / 25) * 25),
                     extent=extent, interpolation="nearest")
    cs = axd.contour(km(x), km(y), np.where(ice, mean_speed, np.nan),
                     levels=[1000, 2000, 3000],
                     colors="w", linewidths=0.6, alpha=0.85)
    axd.clabel(cs, fmt="%d", fontsize=6)

    for ax in axes.ravel():
        ax.contour(km(x), km(y), flo.astype(float), levels=[0.5],
                   colors="0.35", linewidths=0.5)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
    for ax in (axa, axb):
        ax.set_xticklabels([])
    for ax in (axb, axd):
        ax.set_yticklabels([])
    axc.set_xlabel("x (km, EPSG:3031)", fontsize=9)
    axd.set_xlabel("x (km, EPSG:3031)", fontsize=9)
    axa.set_ylabel("y (km)", fontsize=9)
    axc.set_ylabel("y (km)", fontsize=9)

    axa.set_title("(a) Path solver, time-varying quarterly velocity", fontsize=10)
    axb.set_title("(b) Path solver, time-mean velocity", fontsize=10)
    axc.set_title("(c) Difference (b − a)", fontsize=10)
    axd.set_title("(d) Quarterly speed σ (velocity nonstationarity)", fontsize=10)

    kw = dict(fontsize=8.5, ha="left", va="top",
              bbox=dict(fc="white", ec="0.6", alpha=0.85, boxstyle="round,pad=0.25"))
    axa.text(0.02, 0.98, f"GL-2 km clip flux {FLUX_TV:.1f} Gt yr$^{{-1}}$",
             transform=axa.transAxes, **kw)
    axb.text(0.02, 0.98,
             f"{FLUX_MV:.1f} Gt yr$^{{-1}}$ on the matched gate "
             f"(+{100 * (FLUX_MV / FLUX_TV - 1):.0f}%)",
             transform=axb.transAxes, **kw)
    axc.text(0.02, 0.98,
             f"median Δ {med_d:+.2f} m yr$^{{-1}}$; corr raw 0.58 / 2 km 0.75",
             transform=axc.transAxes, **kw)
    axd.text(0.02, 0.98, "37 quarterly mosaics 2015Q1–2024Q1 (Joughin v05)\n"
             "contours: mean speed (m yr$^{-1}$)",
             transform=axd.transAxes, **kw)

    cba = fig.colorbar(ima, ax=[axa, axb], shrink=0.75, pad=0.01)
    cba.set_label("basal mass balance (m ice yr$^{-1}$; negative = melt)", fontsize=9)
    cbc = fig.colorbar(imc, ax=axc, shrink=0.85, pad=0.01)
    cbc.set_label("Δ melt rate (m ice yr$^{-1}$)", fontsize=9)
    cbd = fig.colorbar(imd, ax=axd, shrink=0.85, pad=0.01)
    cbd.set_label("σ$_t$(speed) (m yr$^{-1}$)", fontsize=9)

    out = config.FIGURES_DIR / "fig_velocity_representation"
    fig.savefig(f"{out}.png", dpi=300)
    fig.savefig(f"{out}.pdf")
    plt.close(fig)
    print(f"wrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
