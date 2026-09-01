# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

"""Gate for ``BlendedStubblefieldForward`` -- the spatially varying operator.

``StubblefieldForward`` collapses a shelf to one reference thickness and one mean
velocity. That is a synthetic-twin assumption; a real shelf spans a large
velocity range, so ``alpha ~ u t_r / H`` is wrong nearly everywhere.

The operator is **not spatially compact** -- long-wavelength modes relax slowly
and so advect far downstream before damping (impulse response 5% of peak at
~8 km for 200 m/yr but ~75 km at 2000 m/yr) -- so the melt field cannot be cut
into tiles. Instead each geometry bin's multiplier is applied globally and the
responses are blended. This gate checks the properties the drivers rely on:

  [0] the non-compactness that motivates the design is real, and grows with
      velocity (a regression test on the premise itself);
  [1] the blend weights are an exact partition of unity;
  [2] with spatially CONSTANT geometry the blended operator reproduces the
      monolithic one to machine precision -- switching it on cannot move a
      uniform-flow answer (this is what tiling could NOT do: it truncated the
      tail, 2.5% error at 200 m/yr degrading to 47% at 2000 m/yr);
  [3] with spatially VARYING flow it tracks a piecewise-local reference better
      than the monolithic single-mean-``u`` operator -- the reason the class
      exists. The reference is built independently (the monolithic operator run
      at each half's own velocity), so this is not an inverse crime;
  [4] the driver path ``variational_melt_rate(..., n_bins=...)`` runs, reports
      the binning in its attrs, and still recovers the Shean sign.
"""

import sys

import numpy as np
import pandas as pd  # noqa: F401  (import before torch: libstdc++ ordering)
import xarray as xr

import torch  # noqa: E402

from stereo_melt.constants import rhoi, rhow  # noqa: E402
from stereo_melt.dynamics.stubblefield_forward import (  # noqa: E402
    BlendedStubblefieldForward,
    StubblefieldForward,
    stubblefield_forward_multiplier,
    variational_melt_rate,
)

NY, NX = 128, 160   # 64 x 80 km at 500 m
RES = 500.0         # m posting
H = 400.0           # m reference thickness
U_SLOW, U_FAST = 200.0, 2000.0   # m/yr; a PIG-like range across the domain
M0 = 10.0           # m/yr melt amplitude
FREEBOARD = H * (rhow - rhoi) / rhow


def _smooth_melt(seed: int = 0) -> np.ndarray:
    """Band-limited random melt field (structure at scales the operator passes)."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(NY, RES)[:, None]
    kx = np.fft.fftfreq(NX, RES)[None, :]
    k = np.hypot(ky, kx)
    spec = np.fft.fft2(rng.standard_normal((NY, NX)))
    spec[(k > 1.0 / 4000.0) | (k < 1.0 / 16000.0)] = 0.0     # keep 4-16 km
    m = np.real(np.fft.ifft2(spec))
    return M0 * m / np.abs(m).max()


def _mono(m: np.ndarray, ux: float, uy: float = 0.0) -> np.ndarray:
    """Monolithic forward response at a single velocity."""
    M_h = stubblefield_forward_multiplier(2 * NY, 2 * NX, RES, RES, H, ux, uy)
    return StubblefieldForward(M_h, NY, NX)(torch.from_numpy(m)).detach().numpy()


def _blended(m: np.ndarray, ux_field: np.ndarray, n_bins: int) -> np.ndarray:
    fwd = BlendedStubblefieldForward(
        NY, NX, RES, RES, np.full((NY, NX), H), ux_field, np.zeros((NY, NX)),
        n_bins=n_bins, blend_px=8.0)
    return fwd(torch.from_numpy(m)).detach().numpy()


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def main() -> int:
    ok = True
    m = _smooth_melt()
    xs = np.arange(NX)
    split = NX // 2

    # [0] the premise: the impulse response is NOT compact, and its reach grows
    # with velocity. If this ever fails, tiling would have been viable and this
    # whole class is unnecessary complexity.
    big = 1024
    reach = {}
    for U in (U_SLOW, U_FAST):
        M_h = stubblefield_forward_multiplier(big, big, RES, RES, H, U, 0.0)
        imp = np.zeros((big, big))
        imp[big // 2, big // 2] = 1.0
        psf = np.real(np.fft.ifft2(M_h * np.fft.fft2(imp)))
        prof = np.abs(psf[big // 2, big // 2:])
        below = np.where(prof < 0.05 * prof.max())[0]
        reach[U] = (below[0] * RES / 1000.0) if len(below) else np.inf
    compact_ok = reach[U_SLOW] > 4.0 and reach[U_FAST] > 3.0 * reach[U_SLOW]
    print(f"[0] non-compact kernel: downstream 5% reach "
          f"{reach[U_SLOW]:.0f} km at {U_SLOW:.0f} m/yr -> "
          f"{reach[U_FAST]:.0f} km at {U_FAST:.0f} m/yr  "
          f"{'PASS' if compact_ok else 'FAIL'}")
    ok &= compact_ok

    # [1] partition of unity.
    fwd = BlendedStubblefieldForward(
        NY, NX, RES, RES, np.full((NY, NX), H),
        np.where(xs[None, :] < split, U_SLOW, U_FAST) * np.ones((NY, 1)),
        np.zeros((NY, NX)), n_bins=4, blend_px=8.0)
    wsum = fwd.w.cpu().numpy().sum(0)
    pou_err = float(np.abs(wsum - 1.0).max())
    # only two distinct geometries exist here, so the requested 4 must clamp to 2
    pou_ok = pou_err < 1e-12 and fwd.n_bins == 2
    print(f"[1] partition of unity: max|sum(w) - 1| = {pou_err:.2e}, "
          f"4 bins requested -> {fwd.n_bins} used (2 distinct geometries)  "
          f"{'PASS' if pou_ok else 'FAIL'}")
    ok &= pou_ok

    # [2] constant geometry -> reduces to the monolithic operator EXACTLY.
    for U in (U_SLOW, U_FAST):
        ref = _mono(m, U)
        bl = _blended(m, np.full((NY, NX), U), n_bins=4)
        err = float(np.abs(bl - ref).max())
        red_ok = err < 1e-9
        print(f"[2] constant geometry at {U:6.0f} m/yr reduces to monolithic: "
              f"max|diff| {err:.2e} m  {'PASS' if red_ok else 'FAIL'}")
        ok &= red_ok

    # [3] spatially varying flow. Reference: the monolithic operator evaluated at
    # each half's OWN velocity, scored on the interior of each half (a buffer
    # keeps the seam, where "local" is ill-defined, out of the score).
    ux_field = np.where(xs[None, :] < split, U_SLOW, U_FAST) * np.ones((NY, 1))
    local_ref = np.where(xs[None, :] < split, _mono(m, U_SLOW), _mono(m, U_FAST))
    buf = 24
    interior = np.zeros((NY, NX), bool)
    interior[buf:NY - buf, buf:split - buf] = True
    interior[buf:NY - buf, split + buf:NX - buf] = True

    bl_v = _blended(m, ux_field, n_bins=4)
    mono_v = _mono(m, 0.5 * (U_SLOW + U_FAST))      # the single-mean-u alternative
    e_bl = _rel_rms(bl_v[interior], local_ref[interior])
    e_mono = _rel_rms(mono_v[interior], local_ref[interior])
    var_ok = e_bl < 0.5 * e_mono
    print(f"[3] varying u ({U_SLOW:.0f}-{U_FAST:.0f} m/yr): rel-rms vs local ref  "
          f"blended {e_bl:.4f}  monolithic-mean-u {e_mono:.4f}  "
          f"({e_mono / max(e_bl, 1e-12):.1f}x better)  "
          f"{'PASS' if var_ok else 'FAIL'}")
    ok &= var_ok

    # [4] driver path: n_bins end-to-end through variational_melt_rate. The melt
    # sits well inside the FAST half, where the local geometry is the same U_FAST
    # the reference surface was made with -- so the operator is locally correct
    # there without the fit ever seeing the blended operator's own output.
    x = np.arange(NX) * RES
    y = -np.arange(NY) * RES                        # descending (EPSG:3031)
    coords = {"y": y, "x": x}
    yy, xx = np.mgrid[0:NY, 0:NX]
    cy, cx = NY // 2, split + NX // 4
    m_true = M0 * np.exp(-(((xx - cx) * RES) ** 2 + ((yy - cy) * RES) ** 2)
                         / (2.0 * (2.0 * RES) ** 2))           # positive = melt
    surf = FREEBOARD + _mono(m_true, U_FAST)
    h = xr.DataArray(np.stack([surf, surf]), dims=("time", "y", "x"),
                     coords={"time": [0, 1], **coords})
    mask = xr.DataArray(np.ones((NY, NX), bool), dims=("y", "x"), coords=coords)
    vx = xr.DataArray(ux_field, dims=("y", "x"), coords=coords)
    vy = xr.DataArray(np.zeros((NY, NX)), dims=("y", "x"), coords=coords)

    ds = variational_melt_rate(h, vx, vy, floating_mask=mask, d=0.0, rep="grid",
                               iters=1500, lr=3e-3, n_bins=4, blend_km=4.0)
    mr = ds["melt_rate"].values
    fast = np.isfinite(mr) & (xs[None, :] >= split)     # score where the melt is
    corr = float(np.corrcoef(mr[fast], -m_true[fast])[0, 1])
    # thickness varies continuously here, so all 4 requested bins are realizable;
    # the attrs report the count actually USED (clamping is checked in [1])
    attrs_ok = (ds.attrs["n_bins"] == 4 and ds.attrs["blend_km"] == 4.0
                and "blended operator (4 geometry bins" in ds.attrs["note"])
    drv_ok = attrs_ok and float(mr[cy, cx]) < 0 and corr > 0.7
    print(f"[4] driver n_bins path: center {float(mr[cy, cx]):+.2f} "
          f"corr(-m_true) {corr:.3f}  n_bins={ds.attrs['n_bins']} "
          f"var_explained={ds.attrs['fit_var_explained']:.3f}  "
          f"{'PASS' if drv_ok else 'FAIL'}")
    ok &= drv_ok

    print("\n" + ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
