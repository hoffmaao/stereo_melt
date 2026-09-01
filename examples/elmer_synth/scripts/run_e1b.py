"""E1b pilot: through-flow flowline (Eulerian window) with an ocean-fixed
Gaussian basal melt anomaly; uniform prefactor. Dumps surfaces + surface/basal
velocities for melt-solver evaluation and the alpha-theory cross-check."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/wd2/projects/stereo_melt/examples/elmer_synth/flowline")

import smb  # noqa: E402
from main import solve  # noqa: E402
from params import H, L, m0, nt, stdev, t_f, t_r, u0, xc, yr  # noqa: E402

OUT = os.environ.get(
    "E1B_OUT",
    "/wd2/projects/stereo_melt/examples/elmer_synth/results/e1b_throughflow_result.npz")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

alpha = u0 * t_r / H
print(f"E1b start: L={L/1e3:.0f} km, H={H:.0f} m, u0={u0*yr:.0f} m/yr "
      f"(alpha={alpha:.2f}), melt {m0*yr:.1f} m/yr @ xc={xc/1e3:+.0f} km "
      f"sigma={stdev/1e3:.2f} km, t_f={t_f/yr:.0f} yr, nt={nt}", flush=True)

t0 = time.time()
h, s, x, u_top, w_top, u_bot, w_bot, t, t_saves = solve(smb.smb_h, smb.smb_s)
print(f"\nE1b solve done in {(time.time()-t0)/60:.1f} min", flush=True)

np.savez(
    OUT,
    h=h, s=s, x=x, t=t,
    u_top=u_top, w_top=w_top, u_bot=u_bot, w_bot=w_bot, t_saves=t_saves,
    H=H, L=L, m0=m0, stdev=stdev, xc=xc, u0=u0, t_r=t_r,
)
print(f"saved {OUT}", flush=True)
