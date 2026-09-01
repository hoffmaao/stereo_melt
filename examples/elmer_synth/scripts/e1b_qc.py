"""QC of an E1b through-flow result: steadiness, steady mass-budget closure
d(ubar*H)/dx = -m, hydrostatic freeboard deviation, and the advected imprint
of the ocean-fixed melt anomaly. Usage: e1b_qc.py [result.npz]"""
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "/wd2/projects/stereo_melt/examples/elmer_synth/results/e1b_throughflow_result.npz")
d = np.load(PATH)
h, s, x, t = d["h"], d["s"], d["x"], d["t"]
u_top, u_bot = d["u_top"], d["u_bot"]
H0, m0, stdev, xc, u0, t_r = (float(d[k]) for k in
                              ("H", "m0", "stdev", "xc", "u0", "t_r"))
yr = 3.154e7
rho_i, rho_w = 917.0, 1020.0
sl = H0 * rho_i / rho_w
H = h - s

print(f"result: {PATH}")
print(f"grid nx={x.size}, nt={t.size}, t_f={t[-1]/yr:.0f} yr; "
      f"m0={m0*yr:.1f} m/yr @ xc={xc/1e3:+.1f} km, u0={u0*yr:.0f} m/yr, "
      f"t_r={t_r/yr:.2f} yr, u0*t_r={u0*t_r/1e3:.2f} km")

# 1 -- steadiness over the last 20 yr
i20 = np.searchsorted(t, t[-1] - 20 * yr)
dHdt = (H[:, -1] - H[:, i20]) / ((t[-1] - t[i20]) / yr)
print(f"\n[steadiness] max|dH/dt| last 20 yr = {np.abs(dHdt).max():.4f} m/yr "
      f"(mean {dHdt.mean():+.4f})")

# 2 -- final profiles
Hf = H[:, -1]
ubar = 0.5 * (u_top[:, -1] + u_bot[:, -1]) * yr
shear = (u_top[:, -1] - u_bot[:, -1]) * yr
print(f"[profiles] H: inflow {Hf[0]:.1f} m -> front {Hf[-1]:.1f} m "
      f"(min {Hf.min():.1f} @ x={x[np.argmin(Hf)]/1e3:+.1f} km)")
print(f"           ubar: inflow {ubar[0]:.0f} -> front {ubar[-1]:.0f} m/yr; "
      f"max|u_top-u_bot| {np.abs(shear).max():.2f} m/yr")

# 3 -- steady mass budget: d(ubar H)/dx + m = 0
flux = ubar * Hf
m_x = m0 * yr * np.exp(-(((x - xc) / stdev) ** 2))
div = np.gradient(flux, x)
resid = div + m_x
sel = slice(5, -5)
print(f"[budget] flux inflow {flux[0]:.3e} -> front {flux[-1]:.3e} m^2/yr; "
      f"drop {flux[0]-flux[-1]:.0f} vs integral m {np.trapz(m_x, x):.0f} "
      f"-> closure ratio {(flux[0]-flux[-1])/np.trapz(m_x, x):.4f}")
print(f"         pointwise resid (interior): mean {resid[sel].mean():+.3f}, "
      f"rms {np.sqrt((resid[sel]**2).mean()):.3f}, "
      f"max|.| {np.abs(resid[sel]).max():.3f} m/yr "
      f"(vs m0 {m0*yr:.1f})")

# 4 -- hydrostasy: freeboard vs (1 - rho_i/rho_w) H
fb = h[:, -1] - sl
fb_hyd = (1.0 - rho_i / rho_w) * Hf
dev = fb - fb_hyd
near = np.abs(x - xc) < 2 * stdev
print(f"[hydrostasy] freeboard-dev: mean {dev.mean():+.3f} m, "
      f"range [{dev.min():+.3f}, {dev.max():+.3f}] m; "
      f"at anomaly (|x-xc|<2sig): {dev[near].mean():+.3f} m")

# 5 -- anomaly imprint (approximate, pending the m0=0 control run):
# fit the smooth spreading background H_bg(x) on columns far from the anomaly
# and look at the residual scoop position vs xc.
far = (x < xc - 3 * stdev) | (x > xc + 6 * stdev)
far &= (x > x[0] + 2e3) & (x < x[-1] - 2e3)
cf = np.polyfit(x[far], Hf[far], 3)
scoop = Hf - np.polyval(cf, x)
j = np.argmin(scoop)
zone = np.abs(x - xc) < 4 * stdev
cent = np.sum(x[zone] * (-scoop[zone]).clip(0)) / np.sum((-scoop[zone]).clip(0))
print(f"[imprint] scoop min {scoop.min():.2f} m @ x={x[j]/1e3:+.2f} km; "
      f"weighted centroid {cent/1e3:+.2f} km; xc={xc/1e3:+.1f} km -> "
      f"downstream shift {(cent-xc)/1e3:+.2f} km = "
      f"{(cent-xc)/(u0*t_r):+.2f} u0*t_r")
