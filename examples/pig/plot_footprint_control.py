#!/usr/bin/env python
"""Footprint-controlled Eulerian melt across PIG calving windows.

The per-window "own-footprint" basin median conflates melt change with valid-area
change: the high-melt outer tongue calved away, so later windows average over a
different (smaller, inner) footprint. Re-taking every window's median over a FIXED
pixel set separates real local melt change from footprint change.

Result: the apparent "steady -2.8 then halve post-2020" is mostly footprint. On a
fixed mask, W1~=W2~=W4 and W3 (2017-2020) is an elevated-melt outlier.
Shean convention: negative = melt.
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/wd2/projects/stereo_melt/examples/pig/results/"
OUT = "/wd2/projects/stereo_melt/examples/pig/figures/melt_footprint_control_250m_is2ctempo.png"
ACELL = 0.0625
V = "melt_rate_eulerian"

WINS = [
    ("W1\n2011–13\npre-B31",  "pig_melt_250m_is2ctempo_2011-02-04_2013-10-31.nc"),
    ("W2\n2013–17\nB31→B44",  "pig_melt_250m_is2ctempo_2013-11-01_2017-08-31.nc"),
    ("W3\n2017–20\nB44→B49",  "pig_melt_250m_is2ctempo_2017-09-01_2020-02-08.nc"),
    ("W4\n2020–24\npost-B49", "pig_melt_250m_is2ctempo_2020-02-09_2024-01-10.nc"),
]
labels = [w[0] for w in WINS]
fields = [xr.open_dataset(BASE + w[1])[V].values.astype("float64") for w in WINS]
fin = [np.isfinite(a) for a in fields]
common = fin[0] & fin[1] & fin[2] & fin[3]
w4mask = fin[3]

def med(a, m):
    v = a[m]; v = v[np.isfinite(v)]
    return np.median(v) if v.size else np.nan

own = [med(fields[i], fin[i]) for i in range(4)]
com = [med(fields[i], common) for i in range(4)]
w4f = [med(fields[i], w4mask) for i in range(4)]
x = np.arange(4)

fig, ax = plt.subplots(figsize=(9.5, 6.2))
ax.axhline(0, color="0.6", lw=0.8)
series = [
    (own, "own footprint (as-reported)", "#1f4e9c", "o", "-", 9),
    (w4f, f"fixed: W4 footprint ({w4mask.sum()*ACELL:.0f} km²)", "#2a9d3a", "^", "--", 9),
    (com, f"fixed: shared-by-all-4 ({common.sum()*ACELL:.0f} km²)", "#777777", "s", ":", 8),
]
for vals, lab, col, mk, ls, ms in series:
    ax.plot(x, vals, ls, color=col, lw=1.8, marker=mk, ms=ms, label=lab, zorder=3)
    for xi, v in zip(x, vals):
        dy = 12 if col == "#1f4e9c" else (-16 if col == "#2a9d3a" else 14)
        ax.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8.5, color=col, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Eulerian basin melt median  (m ice / yr)\n← melt        accretion →", fontsize=10)
ax.set_ylim(-3.3, 0.6)
ax.set_title("PIG melt across calving windows — footprint-controlled (Eulerian)\n"
             "blue = own (changing) footprint; green/gray = identical pixels every window",
             fontsize=11.5)
ax.legend(loc="lower left", framealpha=0.93, fontsize=9)
ax.grid(axis="y", alpha=0.25)
ax.annotate("own-footprint shows\n'steady then halve'", (0.5, own[0]), xytext=(0.5, -0.35),
            ha="center", fontsize=8.5, color="#1f4e9c",
            arrowprops=dict(arrowstyle="->", color="#1f4e9c", alpha=0.6))
ax.annotate("on fixed pixels: W3 is the\nmelt high, W1≈W2≈W4", (2.0, w4f[2]),
            xytext=(2.55, -2.55), ha="center", fontsize=8.5, color="#2a9d3a",
            arrowprops=dict(arrowstyle="->", color="#2a9d3a", alpha=0.6))

fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
print(f"\n{'window':10s} {'own':>7s} {'W4-fp':>7s} {'common':>7s}")
for i, lab in enumerate(labels):
    print(f"{lab.replace(chr(10),' '):20s} {own[i]:7.2f} {w4f[i]:7.2f} {com[i]:7.2f}")
