# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

"""Wiring gate for the driver-facing ``variational_melt_rate`` adapter.

Not a physics validation (that is E2a against the Elmer/Ice twin) -- this checks
the xarray plumbing the real-basin drivers depend on:

  [1] a synthetic positive-melt channel forwarded through the operator is
      recovered with the **Shean** sign (negative = melt), self-consistently;
  [2] production densities (918 / 1027) are used and reported;
  [3] a flat control surface (no anomaly) recovers ~zero melt;
  [4] the fit-quality diagnostic is populated and reports a good fit when the
      observed anomaly *is* something the operator can produce.
"""

import sys

import numpy as np
import pandas as pd  # noqa: F401  (import before torch: libstdc++ ordering)
import xarray as xr

import torch  # noqa: E402

from stereo_melt.constants import rhoi, rhow  # noqa: E402
from stereo_melt.dynamics.stubblefield_forward import (  # noqa: E402
    StubblefieldForward,
    stubblefield_forward_multiplier,
    variational_melt_rate,
)

NY, NX = 48, 64
RES = 500.0          # m posting
H = 400.0            # m reference thickness
UX, UY = 300.0, 0.0  # m/yr mean flow (along +x)
M0 = 10.0            # m/yr peak melt (positive = melt, Stubblefield sense)
# freeboard that floats to H via freeboard_to_thickness (d=0): f = H*(rho_w-rho_i)/rho_w
FREEBOARD = H * (rhow - rhoi) / rhow


def _make_stack(anomaly: np.ndarray) -> tuple:
    """Wrap a surface anomaly as a (time, y, x) stack + mask + velocity fields."""
    x = np.arange(NX) * RES
    y = -np.arange(NY) * RES  # descending (EPSG:3031 north-up)
    surf = FREEBOARD + anomaly
    h = xr.DataArray(
        np.stack([surf, surf]), dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": y, "x": x})
    mask = xr.DataArray(np.ones((NY, NX), bool), dims=("y", "x"),
                        coords={"y": y, "x": x})
    vx = xr.DataArray(np.full((NY, NX), UX), dims=("y", "x"),
                      coords={"y": y, "x": x})
    vy = xr.DataArray(np.full((NY, NX), UY), dims=("y", "x"),
                      coords={"y": y, "x": x})
    return h, mask, vx, vy


def _forward_surface(m_true: np.ndarray) -> np.ndarray:
    """Surface anomaly the operator produces from a known melt field."""
    M_h = stubblefield_forward_multiplier(2 * NY, 2 * NX, RES, RES, H, UX, UY)
    fwd = StubblefieldForward(M_h, NY, NX)
    return fwd(torch.from_numpy(m_true)).detach().cpu().numpy()


def main() -> int:
    yy, xx = np.mgrid[0:NY, 0:NX]
    cy, cx = NY // 2, NX // 2
    m_true = M0 * np.exp(-(((xx - cx) * RES) ** 2 + ((yy - cy) * RES) ** 2)
                         / (2.0 * (2.0 * RES) ** 2))            # positive = melt

    dzs = _forward_surface(m_true)
    ok = True

    # [1] sign + self-consistency
    h, mask, vx, vy = _make_stack(dzs)
    ds = variational_melt_rate(h, vx, vy, floating_mask=mask, d=0.0,
                               rep="grid", iters=2500, lr=3e-3)
    mr = ds["melt_rate"].values
    center = float(mr[cy, cx])
    core = m_true > 0.5 * M0
    core_mean = float(np.nanmean(mr[core]))
    # recovered should track -m_true (Shean): strong negative correlation with m_true
    good = np.isfinite(mr)
    corr = float(np.corrcoef(mr[good], -m_true[good])[0, 1])
    amp = -center / M0  # self-consistent recovery ~ O(1)
    sign_ok = center < 0 and core_mean < 0 and corr > 0.9 and 0.3 < amp < 3.0
    print(f"[1] channel: center {center:+.2f}  core_mean {core_mean:+.2f} "
          f"corr(-m_true) {corr:.3f}  amp {amp:.2f}  "
          f"{'PASS' if sign_ok else 'FAIL'}")
    ok &= sign_ok

    # [2] production densities used + reported
    dens_ok = (ds.attrs["rho_i"] == rhoi == 918 and ds.attrs["rho_w"] == rhow == 1027)
    print(f"[2] densities rho_i={ds.attrs['rho_i']} rho_w={ds.attrs['rho_w']} "
          f"(H_ref={ds.attrs['H_ref_m']:.1f} m)  {'PASS' if dens_ok else 'FAIL'}")
    ok &= dens_ok

    # [3] flat control -> ~zero melt
    hc, maskc, vxc, vyc = _make_stack(np.zeros((NY, NX)))
    dsc = variational_melt_rate(hc, vxc, vyc, floating_mask=maskc, d=0.0,
                                rep="grid", iters=1500, lr=3e-3)
    ctrl_max = float(np.nanmax(np.abs(dsc["melt_rate"].values)))
    ctrl_ok = ctrl_max < 0.1 * M0
    print(f"[3] flat control: max|melt| {ctrl_max:.3f} (< {0.1 * M0}) "
          f"{'PASS' if ctrl_ok else 'FAIL'}")
    ok &= ctrl_ok

    # [4] fit-quality diagnostic: the surface here was MADE by the operator, so
    # (after the high-pass) it must be reproducible -- var_explained near 1.
    have_vars = {"dzs_obs", "dzs_fit"} <= set(ds.data_vars)
    ve = float(ds.attrs["fit_var_explained"])
    resid_ok = ds.attrs["fit_rms_resid_m"] < ds.attrs["fit_rms_obs_m"]
    fit_ok = have_vars and ve > 0.95 and resid_ok
    print(f"[4] fit diag: var_explained {ve:.3f}  rms_resid "
          f"{ds.attrs['fit_rms_resid_m']:.3f} m (obs {ds.attrs['fit_rms_obs_m']:.3f} m)  "
          f"vars={sorted(ds.data_vars)}  {'PASS' if fit_ok else 'FAIL'}")
    ok &= fit_ok

    print("\n" + ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
