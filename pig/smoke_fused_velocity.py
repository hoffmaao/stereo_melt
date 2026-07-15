#!/usr/bin/env python
"""Smoke-test + quantify the fused (Kalman+EOF+1km) velocity vs raw ase-quarterly.

Validates the new PIG_VELOCITY=fused path on the REAL PIG stack and measures the
div(u) noise reduction -- the lever for a less-noisy H*div(u) melt term (the
Shean-matching goal). Prints div(u) percentiles + MAD over floating ice for both.
"""
import os
import numpy as np

from pig import config  # noqa: F401
from pig.run_melt import load_stack, load_floating_mask, load_velocity_on_grid
from stereo_melt.kinematics import divergence

print("loading 250 m is2ctempo stack...", flush=True)
stack = load_stack(stack_prefix="pig_stack_250m_is2ctempo")
floating = load_floating_mask(stack)
stack = stack.where(floating)
fm = floating.values.astype(bool)
print(f"  stack time={stack.sizes['time']} grid={stack.sizes['y']}x{stack.sizes['x']} "
      f"floating cells={int(fm.sum())}", flush=True)


def div_stats(tag, source_env):
    os.environ["PIG_VELOCITY"] = source_env
    vx, vy, src = load_velocity_on_grid(stack)
    vxm = vx.mean("time") if "time" in vx.dims else vx
    vym = vy.mean("time") if "time" in vy.dims else vy
    du = divergence(vxm, vym).values  # 1/yr
    a = du[fm]
    a = a[np.isfinite(a)]
    p = np.percentile(a, [1, 5, 50, 95, 99])
    mad = float(np.median(np.abs(a - np.median(a))))
    print(f"\n[{tag}] {src}", flush=True)
    print(f"  div(u) 1/yr  p[1,5,50,95,99]={np.round(p, 4)}  MAD={mad:.4f}  "
          f"|div|>0.1 frac={float(np.mean(np.abs(a) > 0.1)):.3f}  n={a.size}", flush=True)
    return mad, float(np.mean(np.abs(a) > 0.1))


raw_mad, raw_hi = div_stats("RAW ase-quarterly", "ase-quarterly")
fus_mad, fus_hi = div_stats("FUSED kalman+1km", "fused")
print(f"\n=== div(u) MAD: raw={raw_mad:.4f} -> fused={fus_mad:.4f} "
      f"({100 * (1 - fus_mad / raw_mad):.0f}% lower)   "
      f"|div|>0.1 frac: {raw_hi:.3f} -> {fus_hi:.3f} ===", flush=True)
print("SMOKE OK", flush=True)
