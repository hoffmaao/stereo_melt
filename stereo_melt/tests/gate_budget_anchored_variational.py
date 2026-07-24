"""Gate: budget-anchored null-space + in-model background for the variational inverse.

Proves the two mechanisms that turn the DC-blind forward-fit into a level-carrying
fused solver, on a uniform slab where the operator is exact (no inverse crime for
the *level*: the truth surface is built forward, but the questions asked -- does the
anchor restore the mean, does the in-model background replace the high-pass -- are
about the inverse's conditioning, not the operator's fidelity):

  1. IN-MODEL BACKGROUND. Feed the RAW surface (reference ramp + DC pedestal +
     channel anomaly) with ``bg_degree=1`` and no high-pass. The recovered melt
     must still match the channel pattern, and the fitted background must recover
     the planted ramp -- i.e. the polynomial-in-the-model replaces the Gaussian
     high-pass without cutting the melt band.
  2. BUDGET-ANCHORED NULL-SPACE. With ``m_prior`` set to the smooth (budget)
     part of the truth, the recovered melt must carry that absolute level; with
     ``m_prior=None`` the same fit is DC-blind (mean ~ 0). The difference of the
     two means must equal the prior's level -- the anchor, not the data, supplies
     the mean the operator cannot see.
  3. BACKWARD COMPAT. With both defaulted on a pure anomaly, the fit reduces to
     the legacy toward-zero pin (``bg_field is None``) and still recovers the
     channel.

Run: ``$PY stereo_melt/tests/gate_budget_anchored_variational.py``
"""
from __future__ import annotations

import sys

import numpy as np

# Import the library (which pulls xarray/pandas) BEFORE torch: torch-first pins
# the stale system libstdc++ and breaks pandas' extension load in this env.
from stereo_melt.dynamics.stubblefield_forward import (  # noqa: E402
    StubblefieldForward,
    stubblefield_forward_multiplier,
    variational_melt_inverse,
)
import torch  # noqa: E402

NY = NX = 72
DX = DY = 200.0
H = 500.0
U0X, U0Y = 600.0, 0.0
ETA_BAR, ALPHA_SCALE = 1e14, 0.34
LAM, ITERS, LR = 1e-4, 3000, 3e-3


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _forward(fwd: StubblefieldForward, m: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return fwd(torch.from_numpy(np.ascontiguousarray(m, float))).cpu().numpy()


def main() -> int:
    M_h = stubblefield_forward_multiplier(
        2 * NY, 2 * NX, DX, DY, H, U0X, U0Y, eta_bar=ETA_BAR, alpha_scale=ALPHA_SCALE)
    fwd = StubblefieldForward(M_h, NY, NX)

    yy, xx = np.mgrid[0:NY, 0:NX].astype(float)
    xs = xx / (NX - 1) * 2.0 - 1.0
    ys = yy / (NY - 1) * 2.0 - 1.0

    # truth melt (Stubblefield sign, m>0 = melt): a channel-scale Gaussian on top
    # of a smooth "budget" field the channel fit does not know about.
    sig = 3.0 * H / DX                      # ~3 ice thicknesses, in pixels
    cx, cy = NX * 0.45, NY * 0.52
    m_channel = 5.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
    m_long = 1.5 + 0.5 * np.sin(2 * np.pi * xx / NX)      # smooth, domain-scale
    m_true = m_channel + m_long

    # raw observed surface: DC-blind operator response + a reference ramp/pedestal
    bg_true = 15.0 + 8.0 * xs - 5.0 * ys
    hm = _forward(fwd, m_true) + bg_true
    mask = np.ones((NY, NX), bool)

    common = dict(eta_bar=ETA_BAR, alpha_scale=ALPHA_SCALE, lam=LAM,
                  iters=ITERS, lr=LR)

    SHIFT = 3.0
    # (1)+(2) fused: in-model background AND budget anchor
    fused = variational_melt_inverse(hm, mask, DX, DY, H, U0X, U0Y,
                                     m_prior=m_long, bg_degree=1, **common)
    # (2) same, prior shifted by a constant: the level must follow 1:1
    fused_shift = variational_melt_inverse(hm, mask, DX, DY, H, U0X, U0Y,
                                           m_prior=m_long + SHIFT, bg_degree=1, **common)
    # (1) background only, no anchor -> recovers the pattern, DC-blind level
    bgonly = variational_melt_inverse(hm, mask, DX, DY, H, U0X, U0Y,
                                      m_prior=None, bg_degree=1, **common)
    # (3) legacy: pure anomaly, both defaulted
    legacy = variational_melt_inverse(_forward(fwd, m_channel), mask, DX, DY,
                                      H, U0X, U0Y, **common)

    checks = []

    def chk(name, cond, detail):
        checks.append((name, bool(cond), detail))

    # (1) in-model background replaces the high-pass
    chk("bg: background recovered",
        _corr(fused.bg_field, bg_true) > 0.9 and abs(fused.bg_field.mean() - 15.0) < 2.0,
        f"corr={_corr(fused.bg_field, bg_true):.3f} mean={fused.bg_field.mean():.2f} (truth 15)")
    chk("bg: channel pattern recovered from raw surface",
        _corr(bgonly.melt, m_true) > 0.85,
        f"corr(bgonly, truth)={_corr(bgonly.melt, m_true):.3f}")

    # (2) anchor supplies the level the operator cannot see. The DC of the melt
    # is unobservable (DC-blind operator), so it must come from the prior: the
    # recovered level tracks the prior, and shifting the prior shifts it 1:1.
    chk("anchor: level tracks the prior",
        abs(fused.melt.mean() - m_long.mean()) < 0.5,
        f"mean(fused)={fused.melt.mean():.3f} prior={m_long.mean():.3f}")
    chk("anchor: shifting the prior shifts the level 1:1",
        abs((fused_shift.melt.mean() - fused.melt.mean()) - SHIFT) < 0.25,
        f"dmean={fused_shift.melt.mean() - fused.melt.mean():.3f} (imposed {SHIFT})")
    chk("fused: pattern + level",
        _corr(fused.melt, m_true) > 0.9 and abs(fused.melt.mean() - m_long.mean()) < 0.5,
        f"corr={_corr(fused.melt, m_true):.3f} mean={fused.melt.mean():.3f} prior={m_long.mean():.3f}")

    # (3) backward compatibility
    chk("legacy: defaults reduce to toward-zero pin",
        legacy.bg_field is None and _corr(legacy.melt, m_channel) > 0.9,
        f"bg_field={'None' if legacy.bg_field is None else 'set'} "
        f"corr={_corr(legacy.melt, m_channel):.3f}")

    ok = all(c for _, c, _ in checks)
    print("=" * 72)
    for name, c, detail in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name:44s} {detail}")
    print("=" * 72)
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
