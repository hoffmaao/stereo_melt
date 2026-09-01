#!/usr/bin/env python
"""Cross-window melt comparison for PIG split on major calving events.

Windows split on bergs B-31 (2013-11), B-44 (2017-09), B-49 (2020-02):
    W1 pre-B31   2011-02-04 .. 2013-10-31
    W2 B31->B44  2013-11-01 .. 2017-08-31
    W3 B44->B49  2017-09-01 .. 2020-02-08
    W4 post-B49  2020-02-09 .. 2024-01-10

Eulerian (full-field) and Lagrangian (path-integral) only; the linear-inverse
solver collapses to a single grid-filling value (diagnostic-only) and is excluded.
Shean convention: negative = melt, positive = accretion.
"""
import datetime as dt
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RESULTS = "/wd2/projects/stereo_melt/examples/pig/results"
OUT = "/wd2/projects/stereo_melt/examples/pig/figures/melt_calving_windows_250m_is2ctempo.png"
ACELL = 0.0625  # km^2 per 250 m cell

WINDOWS = [
    ("W1\npre-B31",  "2011-02-04", "2013-10-31", "pig_melt_250m_is2ctempo_2011-02-04_2013-10-31.nc"),
    ("W2\nB31→B44", "2013-11-01", "2017-08-31", "pig_melt_250m_is2ctempo_2013-11-01_2017-08-31.nc"),
    ("W3\nB44→B49", "2017-09-01", "2020-02-08", "pig_melt_250m_is2ctempo_2017-09-01_2020-02-08.nc"),
    ("W4\npost-B49", "2020-02-09", "2024-01-10", "pig_melt_250m_is2ctempo_2020-02-09_2024-01-10.nc"),
]
CALVINGS = [("B-31", "2013-11-01"), ("B-44", "2017-09-01"), ("B-49", "2020-02-09")]
IS2_LAUNCH = "2018-09-15"
VEL_START = "2015-02-14"

def d(s):
    return dt.datetime.strptime(s, "%Y-%m-%d")

def stats(ds, var):
    if var not in ds:
        return None
    a = ds[var].values.astype("float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    med = float(np.median(a))
    p25, p75 = (float(v) for v in np.percentile(a, [25, 75]))
    return dict(med=med, p25=p25, p75=p75, n=a.size, area=a.size * ACELL)

rows = []
for label, t0, t1, fn in WINDOWS:
    ds = xr.open_dataset(f"{RESULTS}/{fn}")
    mid = d(t0) + (d(t1) - d(t0)) / 2
    rows.append(dict(label=label, t0=d(t0), t1=d(t1), mid=mid,
                     eul=stats(ds, "melt_rate_eulerian"),
                     lag=stats(ds, "melt_rate_lagrangian")))

fig, (axM, axA) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                               gridspec_kw=dict(height_ratios=[2.4, 1.0], hspace=0.08))

# ---- regime shading: pre-IS2 control era ----
axM.axvspan(d("2010-06-01"), d(IS2_LAUNCH), color="#d9534f", alpha=0.06, zorder=0)
axM.text(d("2014-02-01"), 9.0, "pre–ICESat-2 control\n(CryoTEMPO/CS2 only)",
         ha="center", va="top", fontsize=8.5, color="#a33", style="italic")
axM.text(d("2021-05-01"), 9.0, "ICESat-2 era control",
         ha="center", va="top", fontsize=8.5, color="#357", style="italic")

# ---- calving + epoch markers ----
for ax in (axM, axA):
    ax.axhline(0, color="0.5", lw=0.8, zorder=1)
    for name, ds_ in CALVINGS:
        ax.axvline(d(ds_), color="k", ls="--", lw=1.1, alpha=0.55, zorder=1)
    ax.axvline(d(IS2_LAUNCH), color="#357", ls=":", lw=1.3, alpha=0.8, zorder=1)
    ax.axvline(d(VEL_START), color="#2a8", ls=":", lw=1.3, alpha=0.8, zorder=1)
for name, ds_ in CALVINGS:
    axM.text(d(ds_), -27, name, rotation=90, va="bottom", ha="right",
             fontsize=8, alpha=0.7)
axM.text(d(IS2_LAUNCH), 7.0, "IS-2", rotation=90, va="top", ha="right", fontsize=7.5, color="#357")
axM.text(d(VEL_START), 7.0, "vel start", rotation=90, va="top", ha="right", fontsize=7.5, color="#2a8")

COL = {"eul": "#1f4e9c", "lag": "#c8500a"}
OFF = {"eul": -28, "lag": 28}  # x-offset in days so the two solvers don't overplot

for key, name in (("eul", "Eulerian (full-field)"), ("lag", "Lagrangian (path-integral)")):
    xs, meds, los, his, spanlo, spanhi = [], [], [], [], [], []
    for r in rows:
        s = r[key]
        if s is None:
            continue
        x = r["mid"] + dt.timedelta(days=OFF[key])
        xs.append(x); meds.append(s["med"]); los.append(s["med"] - s["p25"]); his.append(s["p75"] - s["med"])
        spanlo.append(r["t0"]); spanhi.append(r["t1"])
        # window temporal span as a faint horizontal bar at the median
        axM.plot([r["t0"], r["t1"]], [s["med"], s["med"]], color=COL[key], lw=0.8, alpha=0.35, zorder=2)
        # annotate median value; offset horizontally per solver so close medians don't collide
        dxpt, ha = (-11, "right") if key == "eul" else (11, "left")
        axM.annotate(f"{s['med']:+.2f}", (x, s["med"]), textcoords="offset points",
                     xytext=(dxpt, 0), va="center", ha=ha, fontsize=8.5,
                     color=COL[key], fontweight="bold")
    axM.errorbar(xs, meds, yerr=[los, his], fmt="o", ms=8, color=COL[key],
                 ecolor=COL[key], elinewidth=1.6, capsize=4, label=name, zorder=4)

axM.set_ylim(-30, 11)
axM.set_ylabel("basin melt rate  (m ice / yr)\n← melt        accretion →", fontsize=10)
axM.legend(loc="lower left", framealpha=0.92, fontsize=9)
axM.set_title("Pine Island basin melt across major calving windows  —  250 m, IS2+CryoTEMPO control\n"
              "markers = median; whiskers = IQR (p25–p75); faint bar = window time span", fontsize=11)
axM.grid(axis="y", alpha=0.25)

# ---- valid-area panel ----
W = 95  # bar half-width in days
for i, key in enumerate(("eul", "lag")):
    for r in rows:
        s = r[key]
        if s is None:
            continue
        x = mdates.date2num(r["mid"]) + (i - 0.5) * 2 * W
        axA.bar(x, s["area"], width=2 * W * 0.9, color=COL[key], alpha=0.75,
                edgecolor="k", linewidth=0.4)
        axA.text(x, s["area"], f"{s['area']:.0f}", ha="center", va="bottom", fontsize=7.5, color=COL[key])
axA.set_ylabel("valid floating\narea (km²)", fontsize=10)
axA.set_ylim(0, 5200)
axA.grid(axis="y", alpha=0.25)
axA.xaxis.set_major_locator(mdates.YearLocator())
axA.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axA.set_xlim(d("2010-06-01"), d("2024-07-01"))
fig.autofmt_xdate(rotation=0, ha="center")

# tidy legend note for area panel
axA.text(0.985, 0.93, "W4 Lagrangian = 92 km² (≈ 2% of shelf → unconstrained)",
         transform=axA.transAxes, fontsize=8, color="#c8500a", va="top", ha="right")

fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)

# ---- console summary table ----
print(f"\n{'window':12s} {'span(yr)':>8s} | {'Eul med':>8s} {'Eul IQR':>14s} {'Eul km2':>8s} | "
      f"{'Lag med':>8s} {'Lag IQR':>14s} {'Lag km2':>8s}")
for r in rows:
    e, l = r["eul"], r["lag"]
    span = (r["t1"] - r["t0"]).days / 365.25
    print(f"{r['label'].replace(chr(10),' '):12s} {span:8.2f} | "
          f"{e['med']:+8.2f} [{e['p25']:+6.1f},{e['p75']:+5.1f}] {e['area']:8.0f} | "
          f"{l['med']:+8.2f} [{l['p25']:+6.1f},{l['p75']:+5.1f}] {l['area']:8.0f}")
