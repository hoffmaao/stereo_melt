"""E1 score: nonlinear FEniCSx flowline (vendored Stubblefield model) vs the
paper's steady linearized predictions.

Upper surface: Green's-function convolution (notebook 4_nonlinear recipe).
Lower surface: vendored linear-model spectral operator s_steady on a y-uniform
ridge (ky=0 modes = flowline limit).
Transient: center-point incision s(0,t) vs time-dependent compute_s.

Conventions: m > 0 = melt (Stubblefield); h = upper surface, s = lower.
"""
import json
import sys

import numpy as np
from scipy.signal import convolve

RESULT = "/wd2/projects/stereo_melt/examples/elmer_synth/results/e1_nonlinear_result.npz"
OUT_PNG = "/wd2/projects/stereo_melt/examples/elmer_synth/results/e1_score_vs_linear.png"
OUT_JSON = "/wd2/projects/stereo_melt/examples/elmer_synth/results/e1_score_vs_linear.json"
LINMOD = "/wd2/projects/stereo_melt/vendor/linear-shelf-melt/linear-model"

d = np.load(RESULT)
h, s, x, t = d["h"], d["s"], d["x"], d["t"]
H, m0, stdev, t_r = float(d["H"]), float(d["m0"]), float(d["stdev"]), float(d["t_r"])
yr = 3.154e7

# --- anomalies at final time, far-field (|x|>15 km) mean removed -------------
far = np.abs(x) > 15e3
h_anom = (h[:, -1] - h[:, 0]) - (h[:, -1] - h[:, 0])[far].mean()
s_anom = (s[:, -1] - s[:, 0]) - (s[:, -1] - s[:, 0])[far].mean()

# --- steady linear upper surface via the paper's Green's function ------------
m_x = m0 * np.exp(-(x**2) / stdev**2)          # m/s, melt > 0
M = m_x * t_r / H                              # nondimensional melt
G = (np.pi / 4.0) * (1.0 / np.cosh(np.pi * x / (2 * H)) ** 2) * (
    -3.0 + (np.pi * x / H) * np.tanh(np.pi * x / (2 * H))
)
dx = float(x[1] - x[0])
h_lin = convolve(G, M, mode="same") * dx / H * H   # nondim response -> metres

# --- steady linear lower surface via the vendored spectral operator ----------
sys.path.insert(0, LINMOD)
import params as lp  # noqa: E402  (their nondim grid: x in units of H, L=40)
from operators import s_steady, h_steady  # noqa: E402

m_on_lin = m0 * np.exp(-((lp.x0 * H) ** 2) / stdev**2) * t_r / H
# steady operators still index their (Nt,Nx,Ny) k-grid: tile the ridge in t
M_ridge = np.ascontiguousarray(
    np.broadcast_to(m_on_lin[None, :, None], (lp.Nt, lp.Nx, lp.Ny))
)
s_lin_nd = s_steady(M_ridge, alpha=0.0)[0, :, lp.Ny // 2]
h_lin_op_nd = h_steady(M_ridge, alpha=0.0)[0, :, lp.Ny // 2]
x_lin = lp.x0 * H
s_lin = np.interp(x, x_lin, s_lin_nd) * H
h_lin_op = np.interp(x, x_lin, h_lin_op_nd) * H

# --- metrics ------------------------------------------------------------------
mid = np.abs(x) < 15e3
i0 = np.argmin(np.abs(x))


def score(a, b, label):
    r = float(np.corrcoef(a[mid], b[mid])[0, 1])
    amp = float(a[i0] / b[i0]) if abs(b[i0]) > 1e-12 else np.nan
    rms = float(np.sqrt(np.mean((a[mid] - b[mid]) ** 2)))
    print(f"{label}: center NL={a[i0]:+.2f} m LIN={b[i0]:+.2f} m "
          f"ratio={amp:.3f} r={r:.4f} rms={rms:.2f} m")
    return {"center_nl_m": float(a[i0]), "center_lin_m": float(b[i0]),
            "amp_ratio": amp, "pearson_r": r, "rms_m": rms}


print(f"t_f = {t[-1]/yr:.1f} yr = {t[-1]/t_r:.0f} t_r; melt peak {m0*yr:.1f} m/yr, sigma {stdev/1e3:.2f} km")
res = {
    "upper_vs_green": score(h_anom, h_lin, "upper h vs Green fn"),
    "upper_vs_operator": score(h_anom, h_lin_op, "upper h vs spectral op"),
    "lower_vs_operator": score(s_anom, s_lin, "lower s vs spectral op"),
    "meta": {"t_f_yr": float(t[-1] / yr), "t_r_yr": t_r / yr,
             "m0_m_yr": m0 * yr, "stdev_km": stdev / 1e3},
}

with open(OUT_JSON, "w") as f:
    json.dump(res, f, indent=2)

# --- figure -------------------------------------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(x / 1e3, h_anom, "forestgreen", lw=2, label="h nonlinear FEM")
ax[0].plot(x / 1e3, h_lin, "k--", lw=1.5, label="h steady linear (Green)")
ax[0].plot(x / 1e3, h_lin_op, "r:", lw=1.5, label="h steady linear (operator)")
ax[0].set_xlabel("x (km)"); ax[0].set_ylabel("upper-surface anomaly (m)")
ax[0].legend(fontsize=8); ax[0].set_title("upper surface")
ax[1].plot(x / 1e3, s_anom, "midnightblue", lw=2, label="s nonlinear FEM")
ax[1].plot(x / 1e3, s_lin, "k--", lw=1.5, label="s steady linear (operator)")
ax[1].set_xlabel("x (km)"); ax[1].set_ylabel("lower-surface anomaly (m)")
ax[1].legend(fontsize=8); ax[1].set_title("lower surface (basal incision)")
fig.suptitle("E1: Stubblefield nonlinear FEM vs steady linear theory (alpha=0)")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140)
print(f"wrote {OUT_PNG}\nwrote {OUT_JSON}")
