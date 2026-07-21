"""Gate: bridging restoration filter identities.

1. Along-flow single-k: the filter is the exact complex inverse of the
   model transfer T (amplitude AND phase restored) where the cap doesn't
   bind.
2. Across-flow single-k: the filter is the identity (budget dynamics
   recovers flow-parallel ridges without help).
3. The modulus cap binds at short wavelengths.
4. DC / spatial mean passes through untouched; NaN geometry is preserved.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY stereo_melt/tests/gate_bridging_restoration.py
"""
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")
from stereo_melt.backend import asarray, to_numpy, xp  # noqa: E402
from stereo_melt.dynamics import (  # noqa: E402
    bridging_restoration,
    bridging_restoration_filter,
)

NY, NX, RES = 61, 121, 200.0
H, U, ETA = 500.0, 850.0, 1e14
y = np.arange(NY, dtype=float)[::-1] * RES
x = np.arange(NX, dtype=float) * RES
X, Y = np.meshgrid(x, y)
COORDS = {"y": y, "x": x}
KW = dict(H=H, eta_bar=ETA, alpha_scale=0.34)


def da(arr):
    return xr.DataArray(arr, dims=("y", "x"), coords=COORDS)


def forward_through_T(m_true, ux, uy, lift_cap=6.0):
    """Model 'recovered' field: m_rec = IFFT(T * FFT(m_true))."""
    _, T = bridging_restoration_filter(NY, NX, RES, RES, H, ux, uy,
                                       eta_bar=ETA, alpha_scale=0.34,
                                       lift_cap=lift_cap)
    return to_numpy(xp.fft.ifft2(T * xp.fft.fft2(asarray(m_true))).real)


def main():
    n_fail = 0

    # 1. along-flow single-k at ~3H (16 cycles across the domain = an exact
    # FFT mode, lambda = 1512 m) — deep in the damped band, cap not binding.
    # Mirror padding makes the round trip approximate (the even extension
    # is not a pure mode), so check the interior at a few-percent tolerance.
    kx1 = 2.0 * np.pi * 16 / (NX * RES)
    m_true = -5.0 * np.cos(kx1 * X)
    m_rec = forward_through_T(m_true, U, 0.0)
    damped = float(np.abs(m_rec).max() / np.abs(m_true).max())
    rest = bridging_restoration(da(m_rec), U, 0.0, lift_cap=6.0, **KW)
    inner = (slice(8, -8), slice(12, -12))
    err = float(np.abs(rest.values[inner] - m_true[inner]).max()
                / np.abs(m_true).max())
    # 10% max-norm budget: ~5% deliberate band-edge attenuation at 3H
    # (Butterworth half-power at 2.5H) + mirror-pad leakage.
    ok = err < 0.10 and damped < 0.75
    n_fail += not ok
    print(f"[1] along-flow k (lam={2*np.pi/kx1:.0f} m): T damped to "
          f"{damped:.3f}, restored interior rel err {err:.2e} "
          f"{'PASS' if ok else 'FAIL'}")

    # 2. across-flow single-k: identity
    ky1 = 2.0 * np.pi * 4 / (NY * RES)
    m_y = -5.0 * np.cos(ky1 * Y)
    rest_y = bridging_restoration(da(m_y), U, 0.0, lift_cap=6.0, **KW)
    err_y = float(np.abs(rest_y.values - m_y).max())
    ok = err_y < 1e-9
    n_fail += not ok
    print(f"[2] across-flow k: max change {err_y:.2e} "
          f"{'PASS' if ok else 'FAIL'}")

    # 3. cap binds at short lambda
    cap = 3.0
    F, _ = bridging_restoration_filter(NY, NX, RES, RES, H, U, 0.0,
                                       eta_bar=ETA, alpha_scale=0.34,
                                       lift_cap=cap)
    fmax = float(to_numpy(xp.abs(F)).max())
    ok = fmax <= cap + 1e-9 and fmax >= cap - 1e-6
    n_fail += not ok
    print(f"[3] modulus cap: max|F| = {fmax:.6f} (cap {cap}) "
          f"{'PASS' if ok else 'FAIL'}")

    # 4. DC pass-through + NaN geometry
    m_c = np.full((NY, NX), -3.7)
    rest_c = bridging_restoration(da(m_c), U, 0.0, **KW)
    err_c = float(np.abs(rest_c.values - m_c).max())
    m_n = m_true.copy()
    m_n[10:20, 30:50] = np.nan
    rest_n = bridging_restoration(da(m_n), U, 0.0, **KW)
    nan_ok = np.array_equal(np.isnan(rest_n.values), np.isnan(m_n))
    ok = err_c < 1e-9 and nan_ok
    n_fail += not ok
    print(f"[4] DC unchanged ({err_c:.2e}) + NaN geometry preserved "
          f"({nan_ok}) {'PASS' if ok else 'FAIL'}")

    # 5. rotated flow: with flow along +y the x-varying pattern is
    # across-flow, so the filter must reduce to the identity.
    rest_rot = bridging_restoration(da(m_true), 0.0, U, **KW)
    err_rot = float(np.abs(rest_rot.values - m_true).max())
    ok = err_rot < 1e-9  # pattern is across-flow now: identity expected
    n_fail += not ok
    print(f"[5] flow along y, pattern along x -> identity: max change "
          f"{err_rot:.2e} {'PASS' if ok else 'FAIL'}")

    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
