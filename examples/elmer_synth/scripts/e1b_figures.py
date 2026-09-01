"""Figures for the E1b through-flow experiment (melt run vs m0=0 control):
steady-state profiles, differenced anomaly response, and a Hovmoller of
dH(x,t). Annotation values are computed from the data. Outputs PNGs to
elmer_synth/figures/. The outermost grid column at each wall is trimmed from
line plots (documented open-front notch; common-mode with the control)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

R = "/wd2/projects/stereo_melt/examples/elmer_synth/results/"
FIG = "/wd2/projects/stereo_melt/examples/elmer_synth/figures/"
os.makedirs(FIG, exist_ok=True)

# palette (dataviz reference, light surface)
BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF, DIVMID = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK2,
})

yr = 3.154e7
rho_i, rho_w = 917.0, 1020.0
dm = np.load(R + "e1b_throughflow_result.npz")
dc = np.load(R + "e1b_throughflow_control_m0.npz")
x = dm["x"]
t = dm["t"]
m0, stdev, xc, u0, t_r, H0 = (float(dm[k]) for k in
                              ("m0", "stdev", "xc", "u0", "t_r", "H"))
sl = H0 * rho_i / rho_w

cut = slice(1, -1)  # trim the wall/front columns from line plots
xk = x[cut] / 1e3
hm, sm = dm["h"][cut, -1], dm["s"][cut, -1]
hc, sc = dc["h"][cut, -1], dc["s"][cut, -1]
Hm, Hc = hm - sm, hc - sc
ub_m = 0.5 * (dm["u_top"][cut, -1] + dm["u_bot"][cut, -1]) * yr
ub_c = 0.5 * (dc["u_top"][cut, -1] + dc["u_bot"][cut, -1]) * yr
m_x = m0 * yr * np.exp(-(((x[cut] - xc) / stdev) ** 2))

dH, dh, ds = Hm - Hc, hm - hc, sm - sc
plateau = dH[(x[cut] > 0) & (x[cut] < 15e3)].mean()
j_sc = np.argmin(dH)
x_sc, v_sc = x[cut][j_sc], dH[j_sc]
pred = -np.trapz(m_x / ub_m, x[cut])
recov = 100 * (plateau - pred) / abs(pred)

band = dict(color=DIVMID, alpha=1.0, zorder=0)
bx = [(xc - 2 * stdev) / 1e3, (xc + 2 * stdev) / 1e3]


def at(xq, arr):
    return arr[np.searchsorted(x[cut], xq)]


# --- Fig 1: steady-state profiles -----------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True,
                        gridspec_kw={"height_ratios": [2.2, 1.4, 1.0]})
a = axs[0]
a.axvspan(*bx, **band)
a.axhline(sl, color=MUTED, lw=1.0, ls=(0, (4, 3)))
a.text(-19.5, sl + 10, "sea level", color=MUTED, fontsize=8.5)
lm, = a.plot(xk, hm, color=BLUE, lw=2.0, label="melt run")
a.plot(xk, sm, color=BLUE, lw=2.0)
lc, = a.plot(xk, hc, color=AQUA, lw=2.0, label="control (m₀=0)")
a.plot(xk, sc, color=AQUA, lw=2.0)
a.text(0.5, at(0.5e3, sm) + 12, "melt run (base)", color=INK2, fontsize=8.5)
a.text(3.0, at(3.0e3, sc) - 24, "control (base)", color=INK2, fontsize=8.5)
a.text(xc / 1e3, 250, "melt anomaly\n(±2σ)", color=INK2,
       fontsize=8.5, ha="center")
a.legend(handles=[lm, lc], loc="center right", fontsize=8.5)
a.set_ylabel("elevation z (m)")
a.set_title("E1b steady state after 200 yr — through-flow flowline "
            f"(u₀=600 m/yr, α={u0*t_r/H0:.2f})")

b = axs[1]
b.axvspan(*bx, **band)
b.plot(xk, ub_m, color=BLUE, lw=2.0, label="melt run")
b.plot(xk, ub_c, color=AQUA, lw=2.0, label="control (m₀=0)")
b.text(13, at(13e3, ub_m) - 62, "melt run", color=INK2, fontsize=8.5)
b.text(13, at(13e3, ub_c) + 28, "control", color=INK2, fontsize=8.5)
b.legend(loc="upper left", fontsize=8.5)
b.set_ylabel("depth-mean speed\n$\\bar{u}$ (m/yr)")

c = axs[2]
c.axvspan(*bx, **band)
c.fill_between(xk, 0, m_x, color=DIVMID, edgecolor=MUTED, lw=1.4)
c.plot(xk, m_x, color=MUTED, lw=1.6)
c.text(-4.5, 3.2, "prescribed melt m(x)\npeak 5 m/yr (ocean-fixed)",
       color=INK2, fontsize=8.5, ha="left")
c.set_ylabel("m (m/yr)")
c.set_xlabel("x (km)")
c.set_xlim(-20, 20)
fig.align_ylabels(axs)
fig.tight_layout()
fig.savefig(FIG + "e1b_steady_profiles.png", dpi=160)
plt.close(fig)

# --- Fig 2: differenced anomaly response ----------------------------------
fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                           gridspec_kw={"height_ratios": [2.4, 1.0]})
a.axvspan(*bx, **band)
a.axhline(0, color=AXIS, lw=1.0)
a.axhline(pred, color=MUTED, lw=1.2, ls=(0, (4, 3)))
a.text(-19.5, pred - 1.7, f"pure-advection transit integral {pred:+.1f} m",
       color=INK2, fontsize=8.5)
a.plot(xk, dH, color=BLUE, lw=2.2, label="ΔH (thickness)")
a.plot(xk, dh, color=AQUA, lw=1.8, label="Δh (surface)")
a.plot(xk, ds, color=YELLOW, lw=1.8, label="Δs (base)")
a.text(17.2, at(17.2e3, dH) - 2.2, "ΔH", color=INK2, fontsize=9)
a.text(17.2, at(17.2e3, ds) + 1.0, "Δs", color=INK2, fontsize=9)
a.text(17.2, at(17.2e3, dh) - 2.4, "Δh", color=INK2, fontsize=9)
a.annotate(f"scoop {v_sc:+.1f} m at {(x_sc-xc)/(u0*t_r):+.1f} u₀t$_r$\n"
           "downstream of the anomaly",
           xy=(x_sc / 1e3, v_sc), xytext=(-17.0, -12.5), color=INK,
           fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=1.0))
a.text(7.0, plateau + 1.2,
       f"plateau {plateau:+.1f} m ({recov:+.0f}% dynamic recovery)",
       color=INK, fontsize=9, ha="center")
a.legend(loc="upper left", fontsize=8.5)
a.set_ylim(-23.5, 17.5)
a.set_ylabel("melt − control at t = 200 yr (m)")
a.set_title("E1b anomaly response: exact differencing vs the m₀=0 control")

ratio = np.full_like(dH, np.nan)
ok = (np.abs(dH) > 2.0) & (x[cut] < 17e3)  # settled response, pre-front-layer
ratio[ok] = dh[ok] / dH[ok]
b.axvspan(*bx, **band)
b.axhline(1 - rho_i / rho_w, color=MUTED, lw=1.2, ls=(0, (4, 3)))
b.text(-19.5, 1 - rho_i / rho_w + 0.0004,
       f"hydrostatic 1−ρ$_i$/ρ$_w$ = {1-rho_i/rho_w:.4f}",
       color=INK2, fontsize=8.5)
b.plot(xk, ratio, color=BLUE, lw=2.0)
b.set_ylim(0.0955, 0.1045)
b.set_ylabel("Δh / ΔH")
b.set_xlabel("x (km)")
b.set_xlim(-20, 20)
fig.align_ylabels((a, b))
fig.tight_layout()
fig.savefig(FIG + "e1b_anomaly_response.png", dpi=160)
plt.close(fig)

# --- Fig 3: Hovmoller of dH(x,t) -------------------------------------------
stride = 5
dHxt = ((dm["h"] - dm["s"]) - (dc["h"] - dc["s"]))[:, ::stride]
tk = t[::stride] / yr

# diverging: red (thinning) <- neutral gray 0 -> blue (thickening)
cmap = LinearSegmentedColormap.from_list("divRB", [
    (0.00, "#8f2726"), (0.20, RED), (0.42, "#f2b8b7"),
    (0.50, DIVMID),
    (0.58, "#b7d3f6"), (0.80, BLUE), (1.00, "#104281")])
lim = np.ceil(np.abs(dHxt).max())
norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

fig, a = plt.subplots(figsize=(7.2, 4.6))
pm = a.pcolormesh(x / 1e3, tk, dHxt.T, cmap=cmap, norm=norm, shading="auto",
                  rasterized=True)
cb = fig.colorbar(pm, ax=a, pad=0.015)
cb.set_label("ΔH = H(melt) − H(control)  (m)", color=INK2)
cb.outline.set_edgecolor(AXIS)
t_exit = (x[-1] - xc) / u0 / yr
a.plot([xc / 1e3, x[-1] / 1e3], [0, t_exit], color=INK, lw=1.2,
       ls=(0, (5, 4)))
a.text(2.5, 26, "u₀ transit slope", color=INK, fontsize=8.5,
       rotation=32, rotation_mode="anchor")
a.axvline(xc / 1e3, color=INK2, lw=1.0, ls=(0, (2, 3)))
a.text(xc / 1e3 - 0.5, 183, "anomaly center x$_c$", color=INK2,
       fontsize=8.5, ha="right")
a.grid(False)
a.set_xlabel("x (km)")
a.set_ylabel("t (yr)")
a.set_title("E1b thickness response ΔH(x,t): carving, advection, "
            "steady state")
fig.tight_layout()
fig.savefig(FIG + "e1b_hovmoller.png", dpi=160)
plt.close(fig)

print("wrote:", ", ".join(sorted(os.listdir(FIG))))
print(f"(annotations: scoop {v_sc:+.2f} m @ {(x_sc-xc)/(u0*t_r):+.2f} u0*t_r, "
      f"plateau {plateau:+.2f} m, pred {pred:+.2f} m, recovery {recov:+.0f}%)")
