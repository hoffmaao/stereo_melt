"""E1 pilot: run Stubblefield's vendored FEniCSx nonlinear flowline model
with the paper's Gaussian melt anomaly; dump surfaces for scoring."""
import sys
import time

import numpy as np

sys.path.insert(0, "/wd2/projects/stereo_melt/vendor/linear-shelf-melt/nonlinear-model")

import smb  # noqa: E402
from main import solve  # noqa: E402
from params import H, L, nt, t_f, t_r  # noqa: E402

OUT = "/tmp/claude-15579/-wd2-projects-stereo-melt/f1c98541-2f78-4e75-95c9-3c1e9af04c53/scratchpad/e1_nonlinear_result.npz"

m0 = 5.0 / 3.154e7  # 5 m/yr peak basal melt, in m/s
stdev = 10.0 * H / 3.0  # paper's anomaly width (~1.67 km)

m = lambda x, t: smb.smb_s(x, t, m0, stdev)  # noqa: E731
a = lambda x, t: smb.smb_h(x, t, m0, stdev)  # noqa: E731

print(f"E1 start: L={L/1e3:.0f} km, H={H:.0f} m, nt={nt}, t_f/t_r={t_f/t_r:.1f}", flush=True)
t0 = time.time()
h, s, u, w, x, x_vel, z_vel0 = solve(a, m)
print(f"\nE1 solve done in {(time.time()-t0)/60:.1f} min", flush=True)

np.savez(
    OUT,
    h=h, s=s, u=u, w=w, x=x, x_vel=x_vel, z_vel0=z_vel0,
    t=np.linspace(0.0, t_f, nt), t_r=t_r, H=H, L=L, m0=m0, stdev=stdev,
)
print(f"saved {OUT}", flush=True)
