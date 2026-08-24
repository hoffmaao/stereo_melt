"""Amplitude/flux gate at PIG-like speeds: does path deposition dilute melt,
or does seed attribution inflate it?

Context (2026-07-03, PIG 250 m A/B): the budget-corrected pair-banded inverse
(seed-attributed) matched the path solver's MEDIANS per coverage bin but showed
a ~1.6x anomaly stretch (deep side deeper) and +32 Gt/yr on identical gates;
the Eulerian arbiter (116.9 Gt/yr) sided with the inverse. Two readings:

  E1  seed attribution + single-trajectory sampling AMPLIFIES noise where fan
      counts are low -> LIN flux is inflated; REF (path) is right.
  E2  path deposition smears each pair estimate 6-10 km along-flow (|v|~4 km/yr
      x 1.5-2.5 yr) and the two-level median of smeared fields DILUTES real
      channelized melt -> REF flux is biased low; LIN amplitude is right.

The original gate (gate_match_lagrangian.py) cannot see this: it scored 5 km
correlations only, at <=1.85 km/yr. This gate rebuilds the budget-exact
synthetic with PIG-like speeds (~4.6 km/yr max), deep compact melt (-45 m/yr,
sigma 3 km), and scores AMPLITUDE TRANSFER (LSQ slope of estimate anomaly on
truth anomaly), conditional bias per truth-depth bin, and domain clip-flux vs
TRUTH, per corruption rung:

    G0 clean/full   G1 +strip coverage   G2 +0.3 m noise   G3 +0.10 m tilt

Interpretation key: REF beta<1 or flux deficit ON G0 (no noise!) => E2 is
structural. LIN beta~1 on G0 but >1 / flux excess appearing on G2-G3 => E1.

Run:
    cd /wd2/projects/stereo_melt/stereo_melt
    python -u tests/gate_flux_amplitude.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gate_match_lagrangian import (  # noqa: E402
    GEN_DT_YR, RES, SECONDS_PER_YEAR,
    add_noise, add_per_strip_tilt, add_strip_coverage,
    budget_exact_stack, build_epochs, nan_gauss, scorr,
)

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm  # noqa: E402
from stereo_melt.constants import rhoi, rhow  # noqa: E402
from stereo_melt.dynamics import linear_inverse_budget_melt_rate  # noqa: E402
from stereo_melt.melt import eulerian_melt_rate, lagrangian_melt_rate  # noqa: E402

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

LX = 64_000.0
LY = 48_000.0
T_WINDOW_YR = 6.0
N_EPOCHS = 16
VEL_SCALE = 2.5  # PIG trunk ~4 km/yr
EDGE_PX = 12     # score inset: outflow/edge artefacts excluded


def build_fields_fast():
    """PIG-like variant of gate_match_lagrangian.build_fields: 2.5x speeds,
    thicker inflow, and a deep compact melt channel (-45 m/yr, sigma 3 km)."""
    x = np.arange(0.0, LX, RES)
    y = np.arange(LY, 0.0, -RES)
    X, Y = np.meshgrid(x, y)
    yc = LY / 2.0
    xi = (X - x.min()) / LX

    def g2(x0, y0, sx, sy):
        return np.exp(-0.5 * (((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2))

    m_true = (
        -2.0
        - 30.0 * g2(14_000, yc, 4_000, 3_500)           # deep GL melt
        - 45.0 * g2(30_000, yc + 4_000, 3_000, 3_000)   # deep compact channel
        - 10.0 * g2(44_000, yc + 9_000, 6_000, 2_800)   # secondary lobe
        + 2.0 * g2(52_000, yc - 12_000, 5_000, 4_000)   # accretion patch
    )
    a_dot = 0.35 + 0.25 * (1.0 - xi) + 0.10 * np.cos(2.0 * np.pi * Y / 32_000.0)
    H0 = 950.0 - 350.0 * xi + 40.0 * np.sin(2.0 * np.pi * Y / 24_000.0)

    jet = 0.75 + 0.45 * np.exp(-0.5 * ((Y - yc) / 10_000.0) ** 2)
    vx = (250.0 + 1_100.0 * xi) * jet
    vx = vx + 500.0 * np.exp(-0.5 * ((X - 30_000.0) / 9_000.0) ** 2) \
        * np.exp(-0.5 * ((Y - (yc + 7_000.0)) / 6_000.0) ** 2)
    vy = 150.0 * ((Y - yc) / (LY / 2.0)) * (0.3 + 0.7 * xi)
    vx = VEL_SCALE * vx
    vy = VEL_SCALE * vy

    coords = {"y": y, "x": x}
    da = lambda a, n: xr.DataArray(a, dims=("y", "x"), coords=coords, name=n)  # noqa: E731
    return (
        da(m_true, "m_true"), da(a_dot, "a_dot"), da(H0, "H0"),
        da(vx, "vx"), da(vy, "vy"),
    )


def add_strip_blunders(h_stack, n_bad=3, amp_m=1.5, seed=77):
    """Coherent per-strip blunders: ``n_bad`` epochs get a constant bias +
    mild plane over a random rectangle (~a badly coregistered strip)."""
    rng = np.random.default_rng(seed)
    out = h_stack.values.copy()
    nt, ny, nx = out.shape
    bad = rng.choice(nt, size=n_bad, replace=False)
    for k in bad:
        h = rng.integers(ny // 3, ny)
        w = rng.integers(nx // 3, nx)
        yy = rng.integers(0, ny - h + 1)
        xx = rng.integers(0, nx - w + 1)
        sgn = rng.choice([-1.0, 1.0])
        plane = np.linspace(-0.3, 0.3, w)[None, :] * rng.choice([-1.0, 1.0])
        out[k, yy:yy + h, xx:xx + w] += sgn * amp_m + plane
    return h_stack.copy(data=out)


def corrupt(h_clean, rung):
    h = h_clean
    if rung in ("G1", "G2", "G3", "G4"):
        h = add_strip_coverage(h, 0.35, seed=11)
    if rung in ("G2", "G3", "G4"):
        h = add_noise(h, 0.3, seed=101)
    if rung in ("G3", "G4"):
        h = add_per_strip_tilt(h, 0.10, seed=23)
    if rung == "G4":
        h = add_strip_blunders(h, n_bad=3, amp_m=1.5, seed=77)
    return h


def amplitude_row(name, v, truth, inset, res_m):
    """Amplitude transfer + flux vs truth on the joint-finite inset mask."""
    v = np.asarray(v, float)
    t = np.asarray(truth, float)
    f = np.isfinite(v) & np.isfinite(t) & inset
    n = int(f.sum())
    if n < 500:
        return None
    va, ta = v[f], t[f]
    dv, dt_ = va - np.median(va), ta - np.median(ta)
    beta_raw = float((dv * dt_).sum() / (dt_ ** 2).sum())
    v2 = nan_gauss(np.where(f, v, np.nan), 2000.0 / res_m)
    t2 = nan_gauss(np.where(f, t, np.nan), 2000.0 / res_m)
    f2 = np.isfinite(v2) & np.isfinite(t2)
    dv2 = v2[f2] - np.median(v2[f2])
    dt2 = t2[f2] - np.median(t2[f2])
    beta_2k = float((dv2 * dt2).sum() / (dt2 ** 2).sum())
    cell = res_m * res_m * rhoi / 1e12
    flux = float(np.sum(-np.clip(va, -250, 250)) * cell)
    flux_t = float(np.sum(-np.clip(ta, -250, 250)) * cell)
    # conditional bias per truth-depth bin (median of estimate - truth)
    cond = {}
    for lo, hi in ((-200, -30), (-30, -15), (-15, -5), (-5, 0)):
        m = f & (t >= lo) & (t < hi)
        cond[f"{lo}..{hi}"] = float(np.median(v[m] - t[m])) if m.sum() > 200 else np.nan
    return {
        "name": name, "n": n,
        "median": float(np.median(va)), "q25": float(np.percentile(va, 25)),
        "beta_raw": beta_raw, "beta_2k": beta_2k,
        "flux": flux, "flux_truth": flux_t,
        "c2k": scorr(np.where(f, v, np.nan), np.where(f, t, np.nan), 2000.0 / res_m),
        "cond": cond,
    }


def main():
    fields = build_fields_fast()
    m_true, a_dot, H0, vx, vy = fields
    t_years = build_epochs(seed=5)
    print(
        f"grid {m_true.shape[1]}x{m_true.shape[0]} @ {RES:.0f} m, "
        f"{N_EPOCHS} epochs over {T_WINDOW_YR} yr, max|v|="
        f"{float(np.hypot(vx, vy).max()):.0f} m/yr"
    )
    t0 = time.time()
    h_clean = budget_exact_stack(m_true, a_dot, H0, vx, vy, t_years, dt_yr=GEN_DT_YR)
    print(f"budget-exact stack built [{time.time() - t0:.0f}s]")

    ny, nx = m_true.shape
    inset = np.zeros((ny, nx), bool)
    inset[EDGE_PX:-EDGE_PX, EDGE_PX:-EDGE_PX] = True
    res_m = RES
    floating = xr.DataArray(
        np.ones(m_true.shape, dtype=bool), dims=("y", "x"),
        coords={"y": m_true["y"].values, "x": m_true["x"].values},
    )

    keep_maps = {}
    for rung in ("G0", "G1", "G2", "G3", "G4"):
        h = corrupt(h_clean, rung)
        print(f"\n===== {rung} =====")
        rows = []

        t0 = time.time()
        lagr = lagrangian_melt_rate(
            h, vx, vy, a_dot=a_dot, d=0.0, dt_yr=0.05,
            seed_stride=1, output="path", aggregator="pair_median",
            pairs="all", min_dt_yr=1.5, max_dt_yr=2.5,
            progress_interval_s=0.0,
        ).melt_rate
        rows.append(amplitude_row("path(REF)", lagr.values, m_true.values, inset, res_m))
        print(f"  path solver done [{time.time() - t0:.0f}s]")

        t0 = time.time()
        lin = linear_inverse_budget_melt_rate(
            h, vx, vy, a_dot=a_dot, d=0.0, floating_mask=floating,
            eta_bar=1e14, reg=1e-1, transform="dct",
            min_pair_dt_yr=1.5, max_pair_dt_yr=2.5, dt_yr=0.05,
            progress=False,
        )
        rows.append(amplitude_row("lin_budget", lin.melt_rate.values, m_true.values, inset, res_m))
        rows.append(amplitude_row("lin_hydro", lin.melt_rate_hydro.values, m_true.values, inset, res_m))
        lin_cf = linear_inverse_budget_melt_rate(
            h, vx, vy, a_dot=a_dot, d=0.0, floating_mask=floating,
            eta_bar=1e14, reg=1e-1, transform="dct",
            min_pair_dt_yr=1.5, max_pair_dt_yr=2.5, dt_yr=0.05,
            corr_aggregate="fan-median", progress=False,
        )
        rows.append(amplitude_row("lin_corrfm", lin_cf.melt_rate.values, m_true.values, inset, res_m))
        lin_ir = linear_inverse_budget_melt_rate(
            h, vx, vy, a_dot=a_dot, d=0.0, floating_mask=floating,
            eta_bar=1e14, reg=1e-1, transform="dct",
            min_pair_dt_yr=1.5, max_pair_dt_yr=2.5, dt_yr=0.05,
            estimator="irls", progress=False,
        )
        rows.append(amplitude_row("lin_irls", lin_ir.melt_rate.values, m_true.values, inset, res_m))
        lin_pa = linear_inverse_budget_melt_rate(
            h, vx, vy, a_dot=a_dot, d=0.0, floating_mask=floating,
            eta_bar=1e14, reg=1e-1, transform="dct",
            min_pair_dt_yr=1.5, max_pair_dt_yr=2.5, dt_yr=0.05,
            attribution="path", progress=False,
        )
        rows.append(amplitude_row("lin_path", lin_pa.melt_rate.values, m_true.values, inset, res_m))
        print(f"  lin_budget x3 modes done ({lin.attrs['n_starts']} fans, "
              f"{lin.attrs['n_pairs']} pairs; irls sigma="
              f"{lin_ir.attrs.get('irls_sigma_m_yr', float('nan')):.2f} m/yr, "
              f"mean_w={lin_ir.attrs.get('irls_mean_weight', float('nan')):.2f}) "
              f"[{time.time() - t0:.0f}s]")

        eul = eulerian_melt_rate(h, vx, vy, a_dot=a_dot, d=0.0, robust_dh_dt=True).melt_rate
        rows.append(amplitude_row("eulerian", eul.values, m_true.values, inset, res_m))

        if rung == "G4":
            keep_maps = {"truth": m_true.values, "path(REF)": lagr.values,
                         "lin_budget": lin.melt_rate.values,
                         "lin_irls": lin_ir.melt_rate.values}

        print(f"  {'name':<11} {'median':>7} {'q25':>8} {'beta_raw':>8} {'beta_2k':>8} "
              f"{'flux':>7} {'fluxT':>7} {'c2k':>6}   cond bias (est-truth) deep..shallow")
        for r in rows:
            if r is None:
                print("  (insufficient cells)")
                continue
            cond = "  ".join(f"{k}:{v:+.1f}" if np.isfinite(v) else f"{k}:--"
                             for k, v in r["cond"].items())
            print(f"  {r['name']:<11} {r['median']:>7.2f} {r['q25']:>8.2f} "
                  f"{r['beta_raw']:>8.3f} {r['beta_2k']:>8.3f} {r['flux']:>7.2f} "
                  f"{r['flux_truth']:>7.2f} {r['c2k']:>6.3f}   {cond}")

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), constrained_layout=True)
    for ax, (name, v) in zip(axes, keep_maps.items()):
        im = ax.imshow(v, cmap=melt_cmap(), norm=melt_norm(vmax=50.0),
                       interpolation="nearest")
        ax.set_title(f"{name} (G4: +3 blunder strips)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        add_melt_colorbar(fig, im, ax=ax, shrink=0.75)
    out = OUT_DIR / "gate_flux_amplitude_G4.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
