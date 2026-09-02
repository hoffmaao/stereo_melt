"""Gate: regularized bridging inverse (Wiener) well-posedness identities.

1. Round-trip on recoverable single-k modes, along-flow AND across-flow:
   ``bridging_inverse(T[m_true]) ~ m_true`` to a few % (amplitude + phase) on
   BOTH axes — the well-posedness claim (contrast: the restoration post-filter
   only inverts along-flow and forces identity across-flow via its w-weight).
2. Unrecoverable modes are SUPPRESSED, not amplified: ``|W|`` obeys the exact
   Wiener bound ``1/(2*sqrt(lam))`` (no ``1/T`` blow-up).
3. Larger ``lam`` -> stronger suppression of a deep-band mode.
4. DC / spatial mean passes through untouched; NaN geometry preserved.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_bridging_inverse.py
"""
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from stereo_melt.backend import asarray, to_numpy, xp  # noqa: E402
from stereo_melt.dynamics import (  # noqa: E402
    bridging_inverse,
    bridging_inverse_filter,
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


def forward_through_T(m_true, ux, uy):
    """Model 'recovered' hydrostatic field: m_rec = IFFT(T * FFT(m_true))."""
    _, T = bridging_restoration_filter(NY, NX, RES, RES, H, ux, uy,
                                       eta_bar=ETA, alpha_scale=0.34)
    return to_numpy(xp.fft.ifft2(T * xp.fft.fft2(asarray(m_true))).real)


def main():
    n_fail = 0

    # 1a. along-flow single-k (~3H): T damps hard; the inverse recovers it.
    kx1 = 2.0 * np.pi * 16 / (NX * RES)
    m_x = -5.0 * np.cos(kx1 * X)
    rec_x = forward_through_T(m_x, U, 0.0)
    damped_x = float(np.abs(rec_x).max() / 5.0)
    inv_x = bridging_inverse(da(rec_x), U, 0.0, lam=1e-3, **KW)
    inner = (slice(8, -8), slice(12, -12))
    err_x = float(np.abs(inv_x.values[inner] - m_x[inner]).max() / 5.0)
    ok = err_x < 0.05
    n_fail += not ok
    print(f"[1a] along-flow k (lam={2*np.pi/kx1:.0f} m): T damped {damped_x:.3f}"
          f", inverse interior rel err {err_x:.3f} {'PASS' if ok else 'FAIL'}")

    # 1b. across-flow single-k: lightly damped; round-trip also recovers it
    # (full-T deconvolution, no w-weight — well-posed on both axes).
    ky1 = 2.0 * np.pi * 4 / (NY * RES)
    m_y = -5.0 * np.cos(ky1 * Y)
    rec_y = forward_through_T(m_y, U, 0.0)
    damped_y = float(np.abs(rec_y).max() / 5.0)
    inv_y = bridging_inverse(da(rec_y), U, 0.0, lam=1e-3, **KW)
    err_y = float(np.abs(inv_y.values[inner] - m_y[inner]).max() / 5.0)
    ok = err_y < 0.05
    n_fail += not ok
    print(f"[1b] across-flow k (lam={2*np.pi/ky1:.0f} m): T damped {damped_y:.3f}"
          f", inverse rel err {err_y:.3f} {'PASS' if ok else 'FAIL'}")

    # 2. Wiener modulus bound: max|W| <= 1/(2*sqrt(lam)) exactly (no 1/T blowup).
    for lam in (1e-3, 1e-2):
        W, _ = bridging_inverse_filter(NY, NX, RES, RES, H, U, 0.0,
                                       eta_bar=ETA, alpha_scale=0.34, lam=lam)
        wmax = float(to_numpy(xp.abs(W)).max())
        bound = 1.0 / (2.0 * np.sqrt(lam))
        ok = wmax <= bound + 1e-6
        n_fail += not ok
        print(f"[2] lam={lam:g}: max|W|={wmax:.3f} <= Wiener bound {bound:.3f} "
              f"{'PASS' if ok else 'FAIL'}")

    # 3. stronger regularization suppresses a deep-band (short-lambda) mode more.
    kshort = 2.0 * np.pi * 40 / (NX * RES)  # ~1.2H, well into the damped band
    m_s = -5.0 * np.cos(kshort * X)
    rec_s = forward_through_T(m_s, U, 0.0)
    amps = []
    for lam in (1e-3, 1e-1):
        inv_s = bridging_inverse(da(rec_s), U, 0.0, lam=lam, **KW)
        amps.append(float(np.abs(inv_s.values[inner]).max()))
    ok = amps[1] < amps[0]
    n_fail += not ok
    print(f"[3] deep-band suppression: |m| {amps[0]:.3f} (lam=1e-3) -> "
          f"{amps[1]:.3f} (lam=1e-1) {'PASS' if ok else 'FAIL'}")

    # 4. DC pass-through + NaN geometry.
    m_c = np.full((NY, NX), -3.7)
    inv_c = bridging_inverse(da(m_c), U, 0.0, **KW)
    err_c = float(np.abs(inv_c.values - m_c).max())
    m_n = m_x.copy()
    m_n[10:20, 30:50] = np.nan
    inv_n = bridging_inverse(da(m_n), U, 0.0, **KW)
    nan_ok = np.array_equal(np.isnan(inv_n.values), np.isnan(m_n))
    ok = err_c < 1e-9 and nan_ok
    n_fail += not ok
    print(f"[4] DC unchanged ({err_c:.2e}) + NaN geometry preserved ({nan_ok}) "
          f"{'PASS' if ok else 'FAIL'}")

    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
