"""Summary figure for a PIG icepack2 dual-form fluidity inversion.

Six panels: theta (with bound-pin markers), log10 eta_bar, MAP velocity
misfit, L-BFGS-B convergence parsed from the run log, movement vs a
reference run, and the eta_bar distribution against the legacy scalar
values the blended melt operator used to assume.

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY pig/scripts/plot_eta_inversion.py \
        --npz pig/processed/pig_eta_field_250m_dual.npz \
        --log pig/logs/pig_eta_v3_chain.log \
        --ref-npz pig/processed/pig_eta_field_250m_dual_20260727_iter200.npz \
        --ref-label "iter 200 (old domain)" --label "v3 min-extent" \
        --out pig/figures/pig_eta_inversion_v3_summary.png
"""
import argparse
import os
import re
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = _env_proj

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots
INK, INK2 = "#0b0b0b", "#52514e"
LEGACY_ETA = 1.0e13                          # hand-set scalar in the operator


def parse_j(path):
    """(eval index, J) pairs from a run log; [] if absent/unreadable."""
    ev, jj = [], []
    if not path or not os.path.exists(path):
        return np.array(ev), np.array(jj)
    for line in open(path):
        m = re.search(r"eval\s+(\d+): J ([0-9.eE+-]+)", line)
        if m:
            ev.append(int(m.group(1)))
            jj.append(float(m.group(2)))
    return np.array(ev), np.array(jj)


def style_axes(ax):
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--ref-npz", default=None)
    ap.add_argument("--label", default="this run")
    ap.add_argument("--ref-label", default="reference")
    ap.add_argument("--inputs-npz", default=None,
                    help="exporter npz for u_obs (default: alongside npz)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    x, y = d["x"], d["y"]
    xk, yk = x / 1e3, y / 1e3
    mask = d["mask"].astype(bool)
    th = np.where(mask, d["theta"], np.nan)
    eta = np.where(mask, d["eta_pas"], np.nan)
    bound = float(d["bound"]) if "bound" in d.files else 0.0
    ncomp = int(d["n_components"]) if "n_components" in d.files else 1
    rel0, rel1 = float(d["rel_misfit0"]), float(d["rel_misfit_map"])

    inp = args.inputs_npz or os.path.join(
        os.path.dirname(args.npz), "pig_eta_inv_inputs.npz")
    dsp = None
    if os.path.exists(inp):
        di = np.load(inp, allow_pickle=True)
        if di["ux"].shape == mask.shape:
            dsp = np.where(mask, np.hypot(d["u_model_x"] - di["ux"],
                                          d["u_model_y"] - di["uy"]), np.nan)

    ref = np.load(args.ref_npz, allow_pickle=True) if args.ref_npz else None

    fig, axs = plt.subplots(2, 3, figsize=(19, 10.5), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03)
    fin = np.isfinite(th)
    yy, xx = np.nonzero(fin)
    pad = 6.0
    xlim = (xk[xx].min() - pad, xk[xx].max() + pad)
    ylim = (yk[yy].min() - pad, yk[yy].max() + pad)

    def draw(ax, fld, cmap, ttl, vmin=None, vmax=None, cb_label=""):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.94")
        h = ax.pcolormesh(xk, yk, fld, cmap=cm, vmin=vmin, vmax=vmax,
                          shading="auto")
        ax.set_title(ttl, fontsize=10.5, color=INK)
        cb = fig.colorbar(h, ax=ax, shrink=0.75, pad=0.02)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        cb.outline.set_visible(False)
        if cb_label:
            cb.set_label(cb_label, fontsize=8.5, color=INK2)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x (km)", fontsize=8.5, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
        for s in ax.spines.values():
            s.set_color("0.8")
        return h

    # (a) theta
    tlim = max(3.0, float(np.nanpercentile(np.abs(th), 99.9)))
    ax = axs[0, 0]
    draw(ax, th, "RdBu_r", f"θ = ln(A/A₀) — {args.label}",
         vmin=-tlim, vmax=tlim, cb_label="softer →")
    ax.set_ylabel("y (km)", fontsize=8.5, color=INK2)
    if bound > 0:
        pins = fin & (np.abs(th) >= bound - 0.05)
        py, px = np.nonzero(pins)
        if py.size:
            ax.plot(xk[px], yk[py], ".", color=INK, ms=3.5)
        ax.text(0.02, 0.02, f"bounds ±{bound:g}: {py.size} px pinned",
                transform=ax.transAxes, fontsize=8, color=INK2)
    else:
        ax.text(0.02, 0.02,
                f"unbounded (prior-only) · max|θ| {np.nanmax(np.abs(th)):.2f}",
                transform=ax.transAxes, fontsize=8, color=INK2)

    # (b) eta
    p10, p50, p90 = (np.nanpercentile(eta, q) for q in (10, 50, 90))
    draw(axs[0, 1], np.log10(eta), "viridis",
         f"log₁₀ η̄  —  p10 {p10:.1e} · median {p50:.1e} · p90 {p90:.1e} Pa s",
         vmin=12.0, vmax=15.5, cb_label="log₁₀ Pa s")

    # (c) misfit
    if dsp is not None:
        draw(axs[0, 2], dsp, "magma",
             f"MAP |u − u_obs|  —  rel misfit {100 * rel1:.1f}%",
             vmin=0, vmax=120, cb_label="m/yr (capped)")
    else:
        axs[0, 2].axis("off")

    # (d) convergence
    ax = axs[1, 0]
    ev, jj = parse_j(args.log)
    if ev.size:
        ax.semilogy(ev, jj, color=BLUE, lw=2)
        ax.set_xlabel("L-BFGS-B evaluation", fontsize=9, color=INK2)
        ax.set_ylabel("J (data + regularization)", fontsize=9, color=INK2)
        ax.set_title(f"Convergence — J {jj[-1]:.3f}, misfit "
                     f"{100 * rel0:.1f}% → {100 * rel1:.1f}%",
                     fontsize=10.5, color=INK)
        style_axes(ax)
    else:
        ax.axis("off")

    # (e) movement vs reference
    ax = axs[1, 1]
    if ref is not None and ref["theta"].shape == th.shape:
        dth = th - np.where(ref["mask"].astype(bool), ref["theta"], np.nan)
        lim = float(np.nanpercentile(np.abs(dth), 99.5)) or 1.0
        draw(ax, dth, "RdBu_r", f"Δθ vs {args.ref_label}",
             vmin=-lim, vmax=lim, cb_label="Δθ")
        ax.set_ylabel("y (km)", fontsize=8.5, color=INK2)
    else:
        ax.axis("off")

    # (f) eta distribution
    ax = axs[1, 2]
    bins = np.arange(11.4, 16.2, 0.08)
    series = [(eta, BLUE, args.label)]
    if ref is not None:
        series.insert(0, (np.where(ref["mask"].astype(bool), ref["eta_pas"],
                                   np.nan), ORANGE, args.ref_label))
    for e, c, lab in series:
        v = np.log10(e[np.isfinite(e)])
        ax.hist(v, bins=bins, histtype="step", lw=2, color=c,
                weights=np.full(v.size, 100.0 / v.size), label=lab)
    ax.axvline(np.log10(LEGACY_ETA), color="0.55", ls="--", lw=1.2)
    ax.text(np.log10(LEGACY_ETA) - 0.03, ax.get_ylim()[1] * 0.55,
            "legacy hand-set 1e13 ", fontsize=8, color=INK2, va="top",
            ha="right", rotation=90)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_xlabel("log₁₀ η̄ (Pa s)", fontsize=9, color=INK2)
    ax.set_ylabel("% of shelf cells", fontsize=9, color=INK2)
    ax.set_title("η̄ distribution", fontsize=10.5, color=INK)
    style_axes(ax)

    area = float(mask.sum()) * abs((x[1] - x[0]) * (y[1] - y[0])) / 1e6
    fig.suptitle(
        f"PIG fluidity inversion, icepack2 dual form — {args.label} — "
        f"{ncomp} component(s), {area:.0f} km², window {d['t0']}..{d['t1']} "
        f"(γ={float(d['gamma']):g}, σᵤ={float(d['sigma_u']):g} m/yr) — "
        f"rel vel misfit {100 * rel0:.1f}% → {100 * rel1:.1f}%",
        fontsize=13, color=INK)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
