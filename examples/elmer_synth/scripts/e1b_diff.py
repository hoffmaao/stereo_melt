"""Exact anomaly response by differencing the E1b melt run against the m0=0
control at the final (steady) state: dH ramp/plateau, hydrostatic partition,
advective smearing, and the pure-advection transit integral prediction."""
import numpy as np

_REPO = __import__("pathlib").Path(__file__).resolve().parents[3]
R = f"{_REPO}/examples/elmer_synth/results/"
dm = np.load(R + "e1b_throughflow_result.npz")
dc = np.load(R + "e1b_throughflow_control_m0.npz")
yr = 3.154e7
rho_i, rho_w = 917.0, 1020.0
r = rho_i / rho_w

x = dm["x"]
m0, stdev, xc, u0, t_r = (float(dm[k]) for k in ("m0", "stdev", "xc", "u0", "t_r"))
Hm = dm["h"][:, -1] - dm["s"][:, -1]
Hc = dc["h"][:, -1] - dc["s"][:, -1]
dH = Hm - Hc
dh = dm["h"][:, -1] - dc["h"][:, -1]
ds = dm["s"][:, -1] - dc["s"][:, -1]
ub_m = 0.5 * (dm["u_top"][:, -1] + dm["u_bot"][:, -1]) * yr
ub_c = 0.5 * (dc["u_top"][:, -1] + dc["u_bot"][:, -1]) * yr

sel = slice(3, -3)  # trim boundary columns
xs = x[sel]

print(f"[dH] upstream (x<xc-4sig): {dH[sel][xs < xc-4*stdev].mean():+.3f} m; "
      f"plateau (0<x<15 km): {dH[sel][(xs > 0) & (xs < 15e3)].mean():+.3f} m; "
      f"min {dH[sel].min():+.3f} m @ x={xs[np.argmin(dH[sel])]/1e3:+.2f} km")

# ramp position/width: normalize dH to [0,1] between upstream and plateau
up = dH[sel][xs < xc - 4*stdev].mean()
pl = dH[sel][(xs > 0) & (xs < 15e3)].mean()
ramp = (dH[sel] - up) / (pl - up)
zone = (xs > xc - 5*stdev) & (xs < 10e3)
xr, rr = xs[zone], ramp[zone]
x10 = xr[np.searchsorted(rr, 0.1)]
x50 = xr[np.searchsorted(rr, 0.5)]
x90 = xr[np.searchsorted(rr, 0.9)]
print(f"[ramp] 10/50/90%: {x10/1e3:+.2f} / {x50/1e3:+.2f} / {x90/1e3:+.2f} km; "
      f"width(10-90) {(x90-x10)/1e3:.2f} km; midpoint shift vs xc "
      f"{(x50-xc)/1e3:+.2f} km = {(x50-xc)/(u0*t_r):+.2f} u0*t_r "
      f"(Gaussian source alone -> erf ramp centered at xc)")

# hydrostatic partition of the response
mask = (xs > xc + 2*stdev) & (xs < 15e3)  # settled response region
ratio = dh[sel][mask] / dH[sel][mask]
print(f"[partition] dh/dH settled region: mean {ratio.mean():.4f} "
      f"(hydrostatic 1-r = {1-r:.4f}); range [{ratio.min():.4f}, {ratio.max():.4f}]")

# pure-advection transit integral: dH_inf = -int m(x)/u(x) dx (no dyn feedback)
m_x = m0 * yr * np.exp(-(((x - xc) / stdev) ** 2))
pred = -np.trapz(m_x / ub_m, x)
print(f"[transit] pure-advection prediction dH_inf = {pred:+.3f} m; "
      f"measured plateau {pl:+.3f} m; dynamic+diffusive residual "
      f"{pl - pred:+.3f} m ({100*(pl-pred)/abs(pred):+.1f}%)")

# dynamic velocity response
print(f"[du] max|ub_melt-ub_ctrl| {np.abs(ub_m-ub_c)[sel].max():.1f} m/yr "
      f"@ x={xs[np.argmax(np.abs(ub_m-ub_c)[sel])]/1e3:+.2f} km "
      f"(thinner ice -> weaker spreading)")
