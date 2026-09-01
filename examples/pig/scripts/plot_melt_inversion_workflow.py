"""Build pig/figures/melt_inversion_workflow.png — the melt-rate inversion
procedure, one readable card per step, plus the high-pass explainer.

This is the *inversion* half of the chain (PIPELINE.md stage 7 onward); the
upstream acquisition/coregistration/tilt half is `build_pipeline_chart.py`.
It exists because the two solver families answer DIFFERENT questions and are
routinely confused: the mass-budget family conserves mass and owns the absolute
melt level, while the Stubblefield/dfNN family is DC-blind by construction and
only sharpens channel-scale pattern. The high-pass step is where that blindness
is introduced, so it gets its own panel with the real transfer curve.

Numbers are from the 2026-07-28 production run
(`pig_melt_250m_is2ctempo_minext_2010-01-01_2024-01-10.nc`, 513 epochs,
MEaSUREs velocity, min-extent mask) and the v4-eta fused run.

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY pig/scripts/plot_melt_inversion_workflow.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = "/wd2/projects/stereo_melt/examples/pig/figures/melt_inversion_workflow.png"

# dataviz-skill reference palette, categorical slots in fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"

# Geometry the high-pass is actually parameterised by (production PIG run).
H_REF = 473.5        # m, reference ice thickness
RES = 250.0          # m, grid posting
SIGMA_HP_H = 5.0     # the inherited `sigma_hp_H` (units of H)
SIGMA_M = SIGMA_HP_H * H_REF          # Gaussian sigma in metres
KNEE_KM = 2 * np.pi * H_REF / 1e3     # transfer knee, lambda ~ 2*pi*H
PEAK_KM = 3 * H_REF / 1e3             # transfer peak, lambda ~ 3H


def card(ax, x, y, w, h, title, body, color, *, lw=1.6, fs_t=10.5, fs_b=8.6,
         face=None, title_color=None):
    """One rounded step card: colored rule + title + wrapped body text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
        linewidth=lw, edgecolor=color,
        facecolor=face if face else "white", zorder=2))
    ax.text(x + 1.0, y + h - 1.5, title, fontsize=fs_t, fontweight="bold",
            color=title_color or color, va="top", ha="left", zorder=3)
    if body:
        ax.text(x + 1.0, y + h - 4.0, body, fontsize=fs_b, color=INK2,
                va="top", ha="left", linespacing=1.45, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=13, linewidth=lw,
                                 color=color, zorder=1,
                                 shrinkA=0, shrinkB=0))


def build_flow(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 132)
    ax.axis("off")

    # ---------------- inputs ----------------
    ax.text(0, 129.5, "INPUTS  ·  everything below runs on one 250 m EPSG:3031 grid",
            fontsize=9.5, fontweight="bold", color=MUTED, va="top")
    inputs = [
        ("tilt-corrected DEM stack", "513 epochs, 2010–2024\n(time, y, x) cube"),
        ("velocity  u", "time-varying;\nMEaSUREs / fused"),
        ("SMB  ȧ  +  firn  d", "RACMO2.4p1 +\nBedMachine FAC"),
        ("floating ∩ min-extent", "3 km feathered;\ncalved sector removed"),
    ]
    for i, (t, b) in enumerate(inputs):
        card(ax, i * 25.2, 114, 23.2, 12, t, b, MUTED,
             lw=1.1, fs_t=8.8, fs_b=7.6, title_color=INK)

    # ---------------- shared kinematics ----------------
    arrow(ax, 50, 114, 50, 106.5)
    card(ax, 6, 93.5, 88, 12,
         "1 · Shared kinematics — freeboard → thickness → budget terms",
         "H_f = ρ_w/(ρ_w−ρ_i) · (z_s − d)   ·   ∂H_f/∂t  (per-pixel OLS or robust regression)   ·   "
         "∇·(H_f u)\nEvery solver below consumes these same three fields, so any difference "
         "between products is the INVERSE, not the inputs.",
         INK, lw=1.3, fs_t=9.6, fs_b=8.2, title_color=INK)

    # ---------------- the split ----------------
    arrow(ax, 27, 93.5, 27, 86)
    arrow(ax, 73, 93.5, 73, 86)

    # ============ LEFT: mass-budget family ============
    ax.add_patch(FancyBboxPatch(
        (1.5, 26.5), 46, 59, boxstyle="round,pad=0.5,rounding_size=1.5",
        linewidth=0, facecolor=BLUE, alpha=0.055, zorder=0))
    ax.text(4, 83.5, "A · MASS-BUDGET FAMILY", fontsize=10.5,
            fontweight="bold", color=BLUE, va="top")
    ax.text(4, 80.6, "conserves mass → owns the ABSOLUTE LEVEL (m ice/yr).\n"
                     "These are the melt products.",
            fontsize=8.3, color=INK2, va="top", linespacing=1.4)

    card(ax, 3, 65.5, 43, 10.5, "2a · Eulerian  (Shean Eq. 10)",
         "melt = ∂H_f/∂t + ∇·(H_f u) − ȧ\nfixed frame · median −3.93 m/yr · 63.0 Gt/yr",
         BLUE, fs_t=9.4)
    arrow(ax, 24.5, 65.5, 24.5, 61.5, BLUE)
    card(ax, 3, 50.5, 43, 10.5, "2b · Lagrangian path  (PRODUCTION)",
         "same budget integrated along flow paths;\nΔt floor 1.5 yr · median −6.10 · 65.6 Gt/yr",
         BLUE, fs_t=9.4)
    arrow(ax, 24.5, 50.5, 24.5, 46.5, BLUE)
    card(ax, 3, 35.5, 43, 10.5, "2c · Budget linear inverse",
         "linearised twin of the path solver, pair-banded\n+ kernel correction · median −5.36",
         BLUE, fs_t=9.4)
    arrow(ax, 24.5, 35.5, 24.5, 31.5, BLUE)
    ax.text(24.5, 30.2, "→  m_prior : the level the fusion is anchored to",
            fontsize=8.4, color=BLUE, ha="center", va="top", style="italic")

    # ============ RIGHT: spectral / dfNN family ============
    ax.add_patch(FancyBboxPatch(
        (52.5, 26.5), 46, 59, boxstyle="round,pad=0.5,rounding_size=1.5",
        linewidth=0, facecolor=ORANGE, alpha=0.055, zorder=0))
    ax.text(55, 83.5, "B · SPECTRAL / dfNN FAMILY", fontsize=10.5,
            fontweight="bold", color=ORANGE, va="top")
    ax.text(55, 80.6, "DC-blind by construction → a channel-scale PATTERN\n"
                      "correction. NOT a mass-budget melt; not interchangeable.",
            fontsize=8.3, color=INK2, va="top", linespacing=1.4)

    card(ax, 54, 65.5, 43, 10.5, "3a · Remove the reference state  ⟵ HIGH-PASS",
         "δz_s = z̄_s − G_σ * z̄_s,  σ = sigma_hp_H · H_ref\n"
         "5H at H=473 m → σ = 2.37 km  (see panel at right)",
         ORANGE, fs_t=9.0)
    arrow(ax, 75.5, 65.5, 75.5, 61.5, ORANGE)
    card(ax, 54, 50.5, 43, 10.5, "3b · Forward operator  m → δz_s",
         "bridging transfer T(k): flat (hydrostatic) at long λ,\ndamped below ~10 km · 8 geometry bins (H, |u|, η̄)",
         ORANGE, fs_t=9.4)
    arrow(ax, 65, 50.5, 65, 46.5, ORANGE)
    arrow(ax, 86, 50.5, 86, 46.5, ORANGE)
    card(ax, 54, 35.5, 20.5, 10.5, "4a · Direct",
         "divide by T(k)\n(Stubblefield)\nmedian +0.65", ORANGE,
         fs_t=9.0, fs_b=7.8)
    card(ax, 76.5, 35.5, 20.5, 10.5, "4b · dfNN",
         "FIT T(k) forward\n(torch, 4000 it)\nvar_expl 0.476", ORANGE,
         fs_t=9.0, fs_b=7.8)
    arrow(ax, 75.5, 35.5, 75.5, 31.5, ORANGE)
    ax.text(75.5, 30.2, "→  pattern only: zero mean by construction",
            fontsize=8.4, color=ORANGE, ha="center", va="top", style="italic")

    # ============ FUSION ============
    arrow(ax, 24.5, 27.5, 24.5, 22.5, BLUE)
    arrow(ax, 75.5, 27.5, 75.5, 22.5, ORANGE)
    card(ax, 6, 8.5, 88, 13.5,
         "5 · Fused inverse   =   budget LEVEL  ⊕  dfNN PATTERN",
         "Band-pass contract: the prior owns the level, the long wavelengths AND the "
         "sub-transfer-floor short scales;\nthe fit shapes only the usable middle band and "
         "its residual is blinded to match. The kept-band var_explained is then\n"
         "the no-truth channel detector — E2a calibration ≈0.7 = real channel, ≈0.1 = the "
         "correction is noise, trust the prior.",
         AQUA, lw=1.8, fs_t=10.5, fs_b=8.3,
         face="#f4fbf8", title_color=AQUA)
    ax.text(50, 6.4, "PIG, v4 η field:  fused median −5.04 m/yr (budget −4.91 → level held)  "
                     "but kept-band var_expl −2.73  ⇒  pattern NOT supported by the data here",
            fontsize=8.6, color=INK, ha="center", va="top", fontweight="bold")


def build_highpass(ax):
    """The Gaussian high-pass transfer: what the step keeps and what it deletes."""
    lam = np.logspace(np.log10(0.4), np.log10(120), 900)      # km
    k = 2 * np.pi / (lam * 1e3)
    gain = 1 - np.exp(-(k * SIGMA_M) ** 2 / 2)                # retained fraction

    lam50 = float(lam[np.argmin(np.abs(gain - 0.5))])

    # A HIGH-pass keeps SHORT wavelengths (left, gain -> 1) and deletes LONG
    # ones (right, gain -> 0). Shade and label accordingly.
    ax.axvspan(0.4, lam50, color=ORANGE, alpha=0.07, lw=0)
    ax.axvspan(lam50, 120, color=BLUE, alpha=0.07, lw=0)
    ax.plot(lam, gain, color=ORANGE, lw=2.0, zorder=3)

    ax.axvline(lam50, color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.text(lam50 * 0.93, 0.50, f"50% at λ={lam50:.0f} km", fontsize=8.2,
            color=INK2, rotation=90, va="center", ha="right")
    for x in (PEAK_KM, KNEE_KM):
        ax.axvline(x, color=AQUA, lw=1.0, ls=":", zorder=2)
    ax.annotate(f"T(k) peak {PEAK_KM:.1f} km,\nknee 2πH = {KNEE_KM:.1f} km\n"
                "|M_h| flat above ~10 km;\nrolls off below",
                xy=(KNEE_KM, 0.995), xytext=(4.6, 0.80),
                fontsize=7.9, color=AQUA, va="top", ha="left",
                arrowprops=dict(arrowstyle="-", color=AQUA, lw=0.8,
                                shrinkA=2, shrinkB=2))

    for lk, dx, dy in ((5, 8, -13), (10, 8, -13), (20, 8, 9), (40, -8, 12)):
        g = 1 - np.exp(-((2 * np.pi / (lk * 1e3)) * SIGMA_M) ** 2 / 2)
        ax.plot([lk], [g], "o", ms=7, color=ORANGE, mec="white", mew=1.6,
                zorder=4)
        ax.annotate(f"{lk} km → {g:.2f}", (lk, g), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.0, color=INK,
                    ha="right" if dx < 0 else "left")

    ax.set_xscale("log")
    ax.set_xlim(0.6, 120)
    ax.set_ylim(0, 1.06)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("wavelength λ (km)", fontsize=9, color=INK2)
    ax.set_ylabel("fraction of the signal kept", fontsize=9, color=INK2)
    ax.set_title("What the high-pass does   ·   σ = 5H = 2.37 km at H_ref = 473 m",
                 fontsize=10.5, color=INK, pad=8)
    ax.text(1.35, 0.30, "KEPT\nthe dfNN fits\nthis band", fontsize=8.8,
            color=ORANGE, ha="center", va="center", fontweight="bold")
    ax.text(58, 0.62, "DELETED\nthe budget solver\nowns this band",
            fontsize=8.8, color=BLUE, ha="center", va="center",
            fontweight="bold")
    ax.grid(alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.8")


NOTES = [
    ("What high-pass filtering MEANS here",
     "Literally one line of code: smooth the time-median surface with a Gaussian "
     "of width σ, subtract that smooth version, keep the residual — an unsharp "
     "mask. σ is set in units of ICE THICKNESS (sigma_hp_H · H_ref), not in km, "
     "because the physics that motivates it scales with H."),
    ("Why the step exists — and what is NOT a reason (measured 2026-07-28)",
     "NOT operator conditioning. |M_h(k)| was evaluated on the production "
     "geometry: it is FLAT at ~0.297 for every λ ≳ 10 km (0.2947 at 20 km, "
     "0.2966 at 40 km) and rolls off only at SHORT λ (0.104 at 3H = 1.4 km). "
     "The one degenerate mode is the exact k = 0 bin, which linear_perturbation "
     "zeroes BY HAND so perturbations carry no DC offset. So the operator is "
     "well-conditioned across the whole band the high-pass deletes — 'DC-blind' "
     "is literal: the spatial MEAN, and nothing else.\n"
     "① The real reason: the long wavelengths of the DATA aren't melt. The raw "
     "surface carries the shelf's reference profile, residual coregistration / "
     "tilt planes, geoid/MDT error, firn — tens of metres against a "
     "metre-scale channel signal.\n"
     "② Division of labour. The mass-budget solvers own the level and the large "
     "scales already, correctly and by construction."),
    ("What it COSTS — the open issue",
     "It is a band cut applied to the DATA: melt genuinely living at λ > ~13 km "
     "is deleted from the target before fitting and can never be recovered — "
     "and per the measurement above, that is a band the operator could have "
     "inverted perfectly well. σ = 5H is INHERITED from legacy code and never "
     "validated; at PIG's H it throws away 76% of λ=20 km and 93% of λ=40 km. "
     "It must NOT be tuned on var_explained: shrinking the passband "
     "mechanically raises variance explained because there is less structure "
     "left to explain."),
    ("The newer alternative (what the fused leg uses)",
     "`bg_degree` fits a polynomial background INSIDE the model and projects it "
     "out of the residual each step (variable projection), so the raw surface is "
     "the correct input — it removes the reference state WITHOUT cutting a "
     "wavelength band. Combined with the long-side anchor pin (4H) and the "
     "short cut (2.5H) this becomes the band-pass contract in step 5."),
]


def build_notes(ax):
    """Wrap by hand: matplotlib's `wrap=True` measures against the FIGURE, not
    the axes, so long paragraphs run past the right edge."""
    import textwrap

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    head_colors = [ORANGE, ORANGE, INK, AQUA]
    LINE = 0.0335          # vertical step per wrapped line, axes fraction
    y = 1.0
    for (head, body), hc in zip(NOTES, head_colors):
        ax.text(0, y, head, fontsize=9.6, fontweight="bold", color=hc,
                va="top")
        y -= 0.052
        lines = []
        for para in body.split("\n"):
            lines.extend(textwrap.wrap(para, width=118) or [""])
        ax.text(0, y, "\n".join(lines), fontsize=8.3, color=INK2, va="top",
                linespacing=1.5)
        y -= LINE * len(lines) + 0.040


def main() -> int:
    fig = plt.figure(figsize=(21.5, 13.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.34, 1.0],
                          height_ratios=[1.0, 1.02],
                          left=0.028, right=0.982, top=0.915, bottom=0.028,
                          wspace=0.10, hspace=0.16)
    ax_flow = fig.add_subplot(gs[:, 0])
    ax_flow.set_facecolor(SURFACE)
    build_flow(ax_flow)

    ax_hp = fig.add_subplot(gs[0, 1])
    ax_hp.set_facecolor(SURFACE)
    build_highpass(ax_hp)

    ax_notes = fig.add_subplot(gs[1, 1])
    build_notes(ax_notes)

    fig.suptitle("The melt-rate inversion procedure — two solver families, and "
                 "where high-pass filtering enters",
                 fontsize=15.5, color=INK, y=0.968)
    fig.text(0.5, 0.938,
             "PIG 250 m · 513 epochs 2010–2024 · min-extent mask · numbers from the "
             "2026-07-28 production run and the v4-η fused run · sign convention: "
             "negative = melt",
             fontsize=9.5, color=MUTED, ha="center")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
