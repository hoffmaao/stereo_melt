"""L-curve knee selection for the PIG fluidity-inversion prior strength gamma.

Single-control port of mismip_time-dependent-da/scripts/lcurve_knee.py. That
script sweeps two controls (theta_C, theta_A) and votes a modal knee across
rows; the PIG dual inversion has ONE control (theta_A = ln A/A0), so the sweep
is 1-D and the knee is read straight off the single curve.

Procedure, unchanged from the reference:
  * L-curve axes are (log10 J_misfit, log10 seminorm), where the seminorm is
    the gamma-INDEPENDENT solution roughness

        seminorm = J_reg / gamma = 0.5/area * int[theta^2 + L_reg^2 |grad theta|^2] dx

    so the vertical axis measures the solution itself, not the penalty put on
    it, and points at different gamma are directly comparable.
  * both axes are normalized to [0, 1] over the sweep so neither dominates the
    curvature;
  * the knee is the INTERIOR point of maximum discrete (Menger) curvature.

Run:
    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY examples/pig/scripts/lcurve_eta_knee.py [--glob '...*_lcurve_g*_summary.json']
"""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GLOB = f"{BASIN}/processed/pig_eta_field_250m_dual_lcurve_g*_summary.json"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"


def menger(p1, p2, p3):
    """Discrete curvature of the circle through three 2-D points."""
    a = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1])
                  - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    d = np.hypot(*(p2 - p1)) * np.hypot(*(p3 - p2)) * np.hypot(*(p3 - p1))
    return 0.0 if d == 0 else 4 * a / d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out", default=f"{BASIN}/figures/pig_eta_lcurve.png")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(args.glob):
        with open(path) as f:
            s = json.load(f)
        if s.get("J_misfit") is None or not s.get("seminorm"):
            continue
        rows.append(s)
    if len(rows) < 3:
        print(f"need >=3 sweep points, found {len(rows)} at {args.glob}")
        return 1
    rows.sort(key=lambda s: s["gamma"])

    g = np.array([s["gamma"] for s in rows])
    Jm = np.array([s["J_misfit"] for s in rows])
    semi = np.array([s["seminorm"] for s in rows])
    mis = np.array([100 * s["rel_misfit_map"] for s in rows])
    etam = np.array([s["eta_median"] for s in rows])
    nit = np.array([s["n_iters"] for s in rows])
    gnorm = np.array([s.get("grad_norm_final", np.nan) for s in rows])

    def norm_log(v):
        lv = np.log10(v)
        return (lv - lv.min()) / (lv.max() - lv.min() + 1e-30)

    x, yv = norm_log(Jm), norm_log(semi)
    pts = np.column_stack([x, yv])
    curv = np.array([menger(pts[i - 1], pts[i], pts[i + 1])
                     for i in range(1, len(pts) - 1)])
    k = 1 + int(np.argmax(curv))
    g_star = g[k]

    print("=== L-curve, prior strength gamma (single control theta_A) ===")
    print(f"  {len(rows)} points from {os.path.dirname(args.glob)}")
    print(f"  {'gamma':>8} {'J_misfit':>11} {'seminorm':>11} {'misfit%':>8} "
          f"{'eta_med':>10} {'iters':>6} {'|g|':>9} {'curv':>7}")
    for i, s in enumerate(rows):
        c = f"{curv[i - 1]:7.3f}" if 0 < i < len(rows) - 1 else "      -"
        star = "  <== KNEE" if i == k else ""
        print(f"  {g[i]:8.3g} {Jm[i]:11.4e} {semi[i]:11.4e} {mis[i]:8.2f} "
              f"{etam[i]:10.2e} {nit[i]:6d} {gnorm[i]:9.2e} {c}{star}")
    if k in (1, len(rows) - 2):
        print("  WARNING: knee is adjacent to a sweep endpoint — widen the "
              "gamma range to confirm the corner is bracketed.")
    print(f"  SUGGESTED gamma* = {g_star:g}  "
          f"(misfit {mis[k]:.2f}%, eta median {etam[k]:.2e} Pa s)")

    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)

    ax = axs[0]
    ax.plot(x, yv, "-o", color=BLUE, lw=1.8, ms=6, zorder=2)
    ax.plot(x[k], yv[k], "o", ms=15, mfc="none", mec=ORANGE, mew=2.6, zorder=3)
    for i in range(len(rows)):
        ax.annotate(f"γ={g[i]:g}", (x[i], yv[i]), textcoords="offset points",
                    xytext=(9, 7), fontsize=9,
                    color=ORANGE if i == k else INK2)
    ax.set_xlabel("normalized log₁₀ J_misfit  →  worse data fit",
                  fontsize=9.5, color=INK2)
    ax.set_ylabel("normalized log₁₀ seminorm (J_reg/γ)  →  rougher θ",
                  fontsize=9.5, color=INK2)
    ax.set_title(f"L-curve — knee at γ* = {g_star:g} (max Menger curvature)",
                 fontsize=11.5, color=INK)

    ax = axs[1]
    ax.semilogx(g, mis, "-o", color=BLUE, lw=1.8, ms=5.5,
                label="MAP velocity misfit (%)")
    ax.axvline(g_star, color=ORANGE, ls="--", lw=1.6)
    ax.text(g_star, ax.get_ylim()[1], f" γ* = {g_star:g}", color=ORANGE,
            fontsize=9.5, va="top")
    ax.set_xlabel("prior strength γ", fontsize=9.5, color=INK2)
    ax.set_ylabel("rel. velocity misfit (%)", fontsize=9.5, color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax.twinx()
    ax2.semilogx(g, etam, "-s", color="#1a7f5a", lw=1.6, ms=5,
                 label="median η̄")
    ax2.set_yscale("log")
    ax2.set_ylabel("median η̄ (Pa s)", fontsize=9.5, color="#1a7f5a")
    ax2.tick_params(axis="y", labelcolor="#1a7f5a")
    ax.set_title("What the choice costs: fit vs recovered viscosity",
                 fontsize=11.5, color=INK)

    for a in (axs[0], axs[1]):
        a.grid(alpha=0.25, lw=0.5)
        a.set_axisbelow(True)
        a.tick_params(labelsize=8.5, colors=INK2)
        for s_ in ("top",):
            a.spines[s_].set_visible(False)
        for s_ in ("left", "bottom", "right"):
            a.spines[s_].set_color("0.8")

    s0 = rows[0]
    fig.suptitle(
        f"PIG fluidity inversion — prior-strength L-curve "
        f"(mismip_time-dependent-da procedure)\n"
        f"{s0['n_components']} component(s), {s0['area_km2']:.0f} km², "
        f"L_reg {s0['L_reg']:.0f} m, σᵤ {s0['sigma_u']:g} m/yr, "
        f"cold start per point, {s0['n_iters']}-iteration budget",
        fontsize=12.5, color=INK)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"  saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
