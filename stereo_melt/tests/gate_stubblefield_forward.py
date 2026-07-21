"""Gate: differentiable Stubblefield forward operator + variational melt inverse.

1. The torch forward layer reproduces the numpy spectral transfer to ~machine
   precision (wiring correct).
2. Gradients flow to the melt representation (the layer is differentiable).
3. Self-test / machinery: fitting the operator's OWN forward of a known
   single-k melt recovers the amplitude to <3% -- isolates optimizer/plumbing
   error from forward-model error (which is a separate, calibrated-per-geometry
   quantity, so it does NOT belong in a machinery gate).
4. The self-test is INVARIANT to the ``eta_bar`` calibration knob (forward and
   inverse share the operator), confirming that on real truth the residual is a
   forward-model choice, not a machinery defect.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY stereo_melt/tests/gate_stubblefield_forward.py
"""
import sys

import numpy as np

sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")
# Import the library (pulls in pandas via xarray) BEFORE torch: torch loads a
# newer libstdc++ that shadows the system one pandas' C-extensions link against.
from stereo_melt.dynamics.stubblefield_forward import (  # noqa: E402
    GridMelt,
    StubblefieldForward,
    stubblefield_forward_multiplier,
    variational_melt_inverse,
)

import torch  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

NY, NX, RES = 61, 121, 200.0
H, U, V = 500.0, 780.0, 0.0
LAM_M = 1500.0          # across-flow ridge wavelength (~3H)
AMP = 0.08              # true melt amplitude, m/yr
y = np.arange(NY, dtype=float)[::-1] * RES
x = np.arange(NX, dtype=float) * RES
X, Y = np.meshgrid(x, y)
INNER = (slice(6, -6), slice(10, -10))


def synth_dzs(m_true, eta_bar):
    """Surface anomaly the operator itself produces from ``m_true`` (m/yr)."""
    M_h = stubblefield_forward_multiplier(2 * NY, 2 * NX, RES, RES, H, U, V,
                                          eta_bar=eta_bar)
    fwd = StubblefieldForward(M_h, NY, NX)
    with torch.no_grad():
        return fwd(torch.from_numpy(m_true)).cpu().numpy()


def recovered_amp(m_rec):
    """Across-flow cosine amplitude of the recovered melt (LSQ on the interior)."""
    k = 2.0 * np.pi / LAM_M
    sy, sx = INNER
    prof = np.nanmean(m_rec[sy, sx], axis=1)   # average over x -> y-profile
    yy = y[sy]
    A = np.column_stack([np.cos(k * yy), np.sin(k * yy), np.ones_like(yy)])
    c, *_ = np.linalg.lstsq(A, prof, rcond=None)
    return float(np.hypot(c[0], c[1]))


def main():
    n_fail = 0
    k = 2.0 * np.pi / LAM_M
    m_true = AMP * np.cos(k * Y)

    # 1. torch forward layer == numpy spectral transfer.
    M_h = stubblefield_forward_multiplier(2 * NY, 2 * NX, RES, RES, H, U, V)
    fwd = StubblefieldForward(M_h, NY, NX)
    with torch.no_grad():
        torch_out = fwd(torch.from_numpy(m_true)).cpu().numpy()
    mp = np.pad(m_true, ((0, NY), (0, NX)), mode="symmetric")
    np_out = np.fft.ifft2(M_h * np.fft.fft2(mp)).real[:NY, :NX]
    err = float(np.abs(torch_out - np_out).max())
    ok = err < 1e-10
    n_fail += not ok
    print(f"[1] torch layer vs numpy transfer: max|diff| {err:.2e} "
          f"{'PASS' if ok else 'FAIL'}")

    # 2. gradients flow to the melt parameters.
    rep = GridMelt(NY, NX)
    pred = fwd(rep())
    loss = ((pred - torch.from_numpy(torch_out)) ** 2).mean()
    loss.backward()
    grad = rep.m.grad
    ok = grad is not None and torch.isfinite(grad).all() and float(grad.abs().max()) > 0
    n_fail += not ok
    print(f"[2] differentiable: finite non-zero grad "
          f"(max|g|={float(grad.abs().max()):.2e}) {'PASS' if ok else 'FAIL'}")

    # 3. machinery self-test: recover the operator's own forward to <3%.
    dzs = synth_dzs(m_true, eta_bar=1e14)
    mask = np.isfinite(dzs)
    res = variational_melt_inverse(dzs, mask, RES, RES, H, U, V,
                                   rep="grid", iters=3000, lr=3e-3)
    ratio = recovered_amp(res.melt) / AMP
    ok = abs(ratio - 1.0) < 0.03
    n_fail += not ok
    print(f"[3] self-test machinery: recovered/true amp {ratio:.3f} "
          f"{'PASS' if ok else 'FAIL'}")

    # 4. self-test is invariant to the eta_bar calibration knob.
    ratios = []
    for eta in (6e13, 2e14):
        dzs_e = synth_dzs(m_true, eta_bar=eta)
        res_e = variational_melt_inverse(dzs_e, np.isfinite(dzs_e), RES, RES,
                                         H, U, V, rep="grid", eta_bar=eta,
                                         iters=3000, lr=3e-3)
        ratios.append(recovered_amp(res_e.melt) / AMP)
    ok = all(abs(r - 1.0) < 0.03 for r in ratios)
    n_fail += not ok
    print(f"[4] eta_bar-invariant self-test: ratios {ratios[0]:.3f}, "
          f"{ratios[1]:.3f} {'PASS' if ok else 'FAIL'}")

    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILURES'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
