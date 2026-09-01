"""Gate: spatially varying viscosity (eta_field) through the blended operator.

The 2026-07-25 PIG diagnostic showed the fused inverse's forward operator
overstates kept-band surface relief ~3.5x with a hand-set scalar eta_bar, and
pumps coherent stripes at a systematic angle — both symptoms of a uniform
viscosity on real geometry. ``eta_field`` lets a momentum-balance-inferred
effective viscosity vary per geometry bin. This gate proves the seam:

  1. CONSTANT REDUCTION. A constant eta_field must collapse to one bin and
     reproduce the scalar-eta ``StubblefieldForward`` response bit-for-bit.
  2. ETA MATTERS. The uniform responses at the two test viscosities differ
     by a real margin on the test pattern (guards against eta being silently
     ignored, which would let rung 3 pass trivially on one half).
  3. PER-BIN OVERRIDE. On a two-half eta field, the response in each half's
     interior must match the corresponding uniform-eta operator, not the
     scalar fallback passed in mult_kwargs.
  4. PLUMBING. ``variational_melt_inverse(eta_field=...)`` without ``n_bins``
     raises (the monolithic operator has no geometry to vary).

Run: ``$PY stereo_melt/tests/gate_eta_field_operator.py``
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
# Library (pulls pandas) BEFORE torch: torch-first pins the stale system
# libstdc++ and breaks pandas' extension load in this env.
from stereo_melt.dynamics.stubblefield_forward import (  # noqa: E402
    BlendedStubblefieldForward,
    StubblefieldForward,
    stubblefield_forward_multiplier,
    variational_melt_inverse,
)
import torch  # noqa: E402

NY, NX = 96, 192
DX = DY = 200.0
H = 500.0
U0X, U0Y = 600.0, 0.0
ETA1, ETA2 = 1e14, 3e13
ALPHA = 0.34
MK = dict(eta_bar=ETA1, alpha_scale=ALPHA)


def _uniform(eta):
    M = stubblefield_forward_multiplier(2 * NY, 2 * NX, DX, DY, H, U0X, U0Y,
                                        eta_bar=eta, alpha_scale=ALPHA)
    return StubblefieldForward(M, NY, NX)


def _fwd(op, m):
    with torch.no_grad():
        return op(torch.from_numpy(np.ascontiguousarray(m, float))).cpu().numpy()


def main() -> int:
    yy, xx = np.mgrid[0:NY, 0:NX].astype(float)
    sig = 4.0 * H / DX
    # one kernel-band bump per half, so each half's response is probed locally
    m = (5.0 * np.exp(-((xx - NX * 0.25) ** 2 + (yy - NY * 0.5) ** 2)
                      / (2 * sig ** 2))
         + 5.0 * np.exp(-((xx - NX * 0.75) ** 2 + (yy - NY * 0.5) ** 2)
                        / (2 * sig ** 2)))

    ones = np.ones((NY, NX))
    r_u1 = _fwd(_uniform(ETA1), m)
    r_u2 = _fwd(_uniform(ETA2), m)

    checks = []

    def chk(name, cond, detail):
        checks.append((name, bool(cond), detail))

    def rms(a):
        return float(np.sqrt(np.mean(np.asarray(a) ** 2)))

    # (1) constant eta_field == scalar operator, bit-for-bit
    b_const = BlendedStubblefieldForward(NY, NX, DX, DY, H * ones, U0X * ones,
                                         U0Y * ones, eta_field=ETA1 * ones,
                                         n_bins=4, **MK)
    r_const = _fwd(b_const, m)
    chk("constant eta_field reduces to the scalar operator",
        b_const.n_bins == 1 and np.allclose(r_const, r_u1, atol=1e-10),
        f"n_bins={b_const.n_bins} max|diff|={np.max(np.abs(r_const - r_u1)):.2e}")

    # (2) the two viscosities give measurably different responses
    dref = rms(r_u1 - r_u2) / rms(r_u1)
    chk("eta changes the response (difference floor)", dref > 0.05,
        f"rel rms diff={dref:.3f}")

    # (3) two-half eta field: each interior matches its own uniform operator
    eta2d = np.where(xx < NX / 2, ETA1, ETA2)
    b_half = BlendedStubblefieldForward(NY, NX, DX, DY, H * ones, U0X * ones,
                                        U0Y * ones, eta_field=eta2d,
                                        n_bins=2, blend_px=2.0, **MK)
    r_half = _fwd(b_half, m)
    mar = 24
    left = (slice(None), slice(0, NX // 2 - mar))
    right = (slice(None), slice(NX // 2 + mar, NX))
    e_l = rms(r_half[left] - r_u1[left]) / max(rms(r_u1[left]), 1e-30)
    e_r = rms(r_half[right] - r_u2[right]) / max(rms(r_u2[right]), 1e-30)
    # the wrong-eta error in the same interiors, for scale
    w_r = rms(r_u1[right] - r_u2[right]) / max(rms(r_u2[right]), 1e-30)
    chk("two-half field: left interior = eta1 operator", e_l < 0.02,
        f"rel err={e_l:.4f}")
    chk("two-half field: right interior = eta2 operator (not the scalar fallback)",
        e_r < 0.25 * w_r and e_r < 0.05,
        f"rel err={e_r:.4f} (wrong-eta would be {w_r:.3f})")

    # (4) plumbing guard
    try:
        variational_melt_inverse(np.zeros((NY, NX)), np.ones((NY, NX), bool),
                                 DX, DY, H, U0X, U0Y, eta_field=eta2d,
                                 iters=1, **MK)
        raised = False
    except ValueError:
        raised = True
    chk("eta_field without n_bins raises", raised, f"raised={raised}")

    ok = all(c for _, c, _ in checks)
    print("=" * 72)
    for name, c, detail in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name:58s} {detail}")
    print("=" * 72)
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
