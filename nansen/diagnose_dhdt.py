"""Diagnose the per-pixel dh/dt that drives the Nansen Eulerian sign flip.

Davison 2023 reports +1-3 m ice/yr basal melt over Nansen
(Davison-native positive = melt), which translates to -1 to -3 m/yr in
the Shean public convention used by this package; our Eulerian +
Lagrangian + dh/dt-DCT solvers reported the equivalent of strong basal
accretion (Davison-convention -4 m ice/yr ≡ Shean +4 m/yr, *opposite*
sign from Davison). After hydrostatic conversion
(rho_w / (rho_w - rho_i) ~ 9.3), even ~+0.4 m/yr of fake freeboard
rise produces an artifact of magnitude ~4 m/yr in basal balance.

This script answers: is there a positive dh/dt artifact in the
tilt-corrected stack, and is it real (basin-scale climate signal) or a
tilt-fit residual?

Outputs:

  - ``figures/diagnose_dhdt_maps.png`` — 2x3 grid of dh/dt fields:
        raw OLS dh/dt        / tilt-corrected OLS dh/dt    / LSQ dhdt (joint)
        floating-only        / static-control-only         / floating - static

  - ``figures/diagnose_dhdt_hist.png`` — histograms of dh/dt over
        floating  vs  static-control  for each of the three estimates,
        annotated with mean / median / IQR / what they imply for
        b_dot via -dh/dt * (rho_w / (rho_w - rho_i)).

  - per-epoch median freeboard over floating + static control (vs time)
        showing whether the per-epoch z_pix carries a coherent drift.

If the static-control dh/dt is centered on zero AND the floating dh/dt
is biased high, the bias is physical (or unmodelled). If the static-
control dh/dt is also biased, the tilt-fit didn't lock the absolute
elevation reference.

Run::

    python -m nansen.diagnose_dhdt
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.coregister.tilt import build_static_area_polygon_mask
from stereo_melt.constants import rhoi, rhow
from stereo_melt.kinematics import SECONDS_PER_YEAR, dh_dt
from stereo_melt.stack import load_basin_stack

from nansen import config
from nansen.run_melt import load_floating_mask

R_HYDRO = rhow / (rhow - rhoi)  # ~9.41 -- freeboard rate -> ice-equiv thickness rate


def _stats(name: str, vals: np.ndarray) -> dict:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "implied_b": -float(np.mean(v)) * R_HYDRO,  # m ice/yr melt implied by mean
    }


def _stat_line(s: dict) -> str:
    if s["n"] == 0:
        return f"{s['name']}: no data"
    return (
        f"{s['name']:<28s}  n={s['n']:>7d}  "
        f"mean={s['mean']:+.3f}  med={s['median']:+.3f}  "
        f"IQR=[{s['q25']:+.3f}, {s['q75']:+.3f}]  "
        f"=> implied -dh/dt * R = {s['implied_b']:+.2f} m ice/yr"
    )


def _imshow(ax, arr, *, x, y, vmin, vmax, cmap="RdBu"):
    return ax.imshow(
        arr,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="upper" if y[0] > y[-1] else "lower",
        cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="nearest",
    )


def main() -> None:
    print(f"Hydrostatic gain rho_w/(rho_w-rho_i) = {R_HYDRO:.3f}")

    # ---- load stacks (raw + tilt-corrected) ----
    print("\nLoading raw + tilt-corrected stacks...")
    raw, raw_path = load_basin_stack(
        config.PROCESSED_DIR, "nansen_stack",
        config.START_TIME, config.END_TIME,
        prefer_tilt_corrected=False,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    print(f"  raw: {raw_path.name}  dims={dict(raw.sizes)}")

    tc, tc_path = load_basin_stack(
        config.PROCESSED_DIR, "nansen_stack",
        config.START_TIME, config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    print(f"  tc : {tc_path.name}  dims={dict(tc.sizes)}")

    # ---- masks ----
    print("\nLoading floating + static-control masks...")
    floating = load_floating_mask(tc).values.astype(bool)
    static = build_static_area_polygon_mask(tc, config.BEDMACHINE_NC).values.astype(bool)
    print(f"  floating frac: {floating.mean():.3f}")
    print(f"  static  frac: {static.mean():.3f}")

    # ---- per-pixel OLS dh/dt of raw + tc stacks (m/yr) ----
    print("\nComputing per-pixel OLS dh/dt for raw and tilt-corrected stacks...")
    dh_raw = dh_dt(raw)["slope"].values * SECONDS_PER_YEAR
    dh_tc = dh_dt(tc)["slope"].values * SECONDS_PER_YEAR
    print(f"  raw dh/dt range (finite): {np.nanmin(dh_raw):+.3f} .. {np.nanmax(dh_raw):+.3f} m/yr")
    print(f"  tc  dh/dt range (finite): {np.nanmin(dh_tc):+.3f} .. {np.nanmax(dh_tc):+.3f} m/yr")

    # ---- joint-LSQ dhdt from tilt-fit params (m/day -> m/yr) ----
    params_path = (
        config.PROCESSED_DIR
        / f"nansen_tilt_params_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nReading joint-LSQ dhdt from {params_path.name}...")
    params = xr.open_dataset(params_path)
    dh_lsq = params["dhdt"].values * 365.25  # m/day -> m/yr
    print(f"  lsq dh/dt range (finite): {np.nanmin(dh_lsq):+.3f} .. {np.nanmax(dh_lsq):+.3f} m/yr")

    # ---- summary stats over floating + static-control ----
    print("\n--- per-pixel OLS, FLOATING ---")
    s_raw_f = _stats("raw OLS / floating", dh_raw[floating])
    s_tc_f = _stats("tilt-corr OLS / floating", dh_tc[floating])
    s_lsq_f = _stats("joint-LSQ / floating", dh_lsq[floating])
    print(_stat_line(s_raw_f))
    print(_stat_line(s_tc_f))
    print(_stat_line(s_lsq_f))

    print("\n--- per-pixel OLS, STATIC CONTROL (rock + grounded) ---")
    s_raw_s = _stats("raw OLS / static", dh_raw[static])
    s_tc_s = _stats("tilt-corr OLS / static", dh_tc[static])
    s_lsq_s = _stats("joint-LSQ / static", dh_lsq[static])
    print(_stat_line(s_raw_s))
    print(_stat_line(s_tc_s))
    print(_stat_line(s_lsq_s))

    # ---- per-epoch median z over floating + static ----
    print("\nPer-epoch median freeboard offsets...")
    times = pd.to_datetime(tc["time"].values)
    z_tc = tc.values
    floating_med = np.array([
        float(np.nanmedian(z_tc[k][floating])) for k in range(z_tc.shape[0])
    ])
    static_med = np.array([
        float(np.nanmedian(z_tc[k][static])) for k in range(z_tc.shape[0])
    ])
    floating_anom = floating_med - np.nanmean(floating_med)
    static_anom = static_med - np.nanmean(static_med)

    # ----- map figure -----
    print("\nWriting figures...")
    x = tc["x"].values
    y = tc["y"].values
    vmax_map = 0.5  # m/yr scale for dh/dt maps
    fig, axes = plt.subplots(2, 3, figsize=(3 * 4.6, 2 * 5.6), constrained_layout=True)
    panels = [
        ("raw OLS dh/dt", dh_raw, dh_raw),
        ("tilt-corrected OLS dh/dt", dh_tc, dh_tc),
        ("joint-LSQ dh/dt (tilt-fit)", dh_lsq, dh_lsq),
    ]
    for col, (title, full_arr, masked_arr) in enumerate(panels):
        # Top row: full grid masked to floating+static union for visibility
        full = full_arr.copy()
        view_mask = floating | static
        full[~view_mask] = np.nan
        im = _imshow(axes[0, col], full, x=x, y=y, vmin=-vmax_map, vmax=vmax_map)
        axes[0, col].set_title(
            f"{title}\nfloating + static union", fontsize=10,
        )
        fig.colorbar(im, ax=axes[0, col], fraction=0.045, label="m/yr (freeboard)")
        axes[0, col].set_xlabel("x (m)")

        # Bottom row: floating-only minus mean, sharper scale
        f_only = full_arr.copy()
        f_only[~floating] = np.nan
        f_mean = float(np.nanmean(f_only))
        im2 = _imshow(
            axes[1, col], f_only - f_mean, x=x, y=y,
            vmin=-vmax_map, vmax=vmax_map,
        )
        axes[1, col].set_title(
            f"{title}\nfloating only, demeaned (mean={f_mean:+.3f} m/yr)",
            fontsize=10,
        )
        fig.colorbar(im2, ax=axes[1, col], fraction=0.045, label="m/yr (freeboard)")
        axes[1, col].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")

    fig.suptitle(
        f"Nansen dh/dt diagnosis  ({config.START_TIME} → {config.END_TIME})\n"
        f"Hydrostatic gain  rho_w/(rho_w-rho_i) = {R_HYDRO:.2f}: "
        f"each +0.1 m/yr freeboard rise -> +{0.1 * R_HYDRO:.2f} m ice/yr "
        f"misattributed to basal accretion",
        fontsize=12,
    )
    out_map = config.FIGURES_DIR / "diagnose_dhdt_maps.png"
    fig.savefig(out_map, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_map}")

    # ----- histograms + per-epoch series -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    bins = np.linspace(-2, 2, 81)
    for ax, (title, dh) in zip(
        axes.flat[:3],
        [
            ("raw OLS dh/dt (m/yr)", dh_raw),
            ("tilt-corrected OLS dh/dt (m/yr)", dh_tc),
            ("joint-LSQ dh/dt (m/yr)", dh_lsq),
        ],
    ):
        f_vals = dh[floating]
        s_vals = dh[static]
        ax.hist(
            f_vals[np.isfinite(f_vals)], bins=bins, alpha=0.5,
            color="C0", label=f"floating (med={np.nanmedian(f_vals):+.2f})",
            density=True,
        )
        ax.hist(
            s_vals[np.isfinite(s_vals)], bins=bins, alpha=0.5,
            color="C1", label=f"static (med={np.nanmedian(s_vals):+.2f})",
            density=True,
        )
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("dh/dt (m/yr)")
        ax.set_ylabel("pdf")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(times, floating_anom, "o-", color="C0", label="floating median anomaly")
    ax.plot(times, static_anom, "s-", color="C1", label="static median anomaly")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("median freeboard anomaly (m)")
    ax.set_xlabel("epoch")
    ax.set_title("per-epoch median z relative to time-mean (tilt-corrected stack)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Nansen dh/dt distribution  "
        f"({config.START_TIME} → {config.END_TIME})  "
        f"floating vs static-control  (mean stat -> b_dot via -dh/dt * "
        f"{R_HYDRO:.2f})",
        fontsize=12,
    )
    out_hist = config.FIGURES_DIR / "diagnose_dhdt_hist.png"
    fig.savefig(out_hist, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_hist}")

    print("\nVERDICT (read the means above):")
    print(
        "  - if static_mean ≈ 0 and floating_mean is biased ⇒ physical / unmodelled"
    )
    print(
        "  - if static_mean is biased the same direction ⇒ tilt-fit didn't lock z"
    )


if __name__ == "__main__":
    main()
