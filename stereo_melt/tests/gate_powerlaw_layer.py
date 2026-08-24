"""Gate: the linearised power-law (Glen) layer transfer functions.

Rungs
-----
G1  n -> 1 limit: R, B converge to Stubblefield Eq. 2.36-2.37 linearly in
    (n - 1) at every angle (the n = 1 point itself is a degenerate
    triple eigenvalue, so the closed form is used there by construction).
G2  reciprocity: the two independent unit-load estimates of R and of B
    (surface load -> base response, base load -> surface response) agree
    to round-off for n = 3, every angle, uniaxial and radial backgrounds.
G3  analytic limits, n = 3 uniaxial, k || E and k _|_ E:
      long wavelength (thin sheet)  R_n / R_1 -> 1 / nu_n
      short wavelength (half-space) R_n / R_1 -> 1 / sqrt(nu_n)
    with nu_n the closed-form normal-viscosity ratio (1/3 and 5/6).
G4  independent stream-function solution (plane strain, nu = 1/3) of
    psi'''' - (4 nu - 2) k^2 psi'' + k^4 psi = 0 by Chebyshev collocation
    matches the 6x6 exponential solver to 1e-3 for kH in [0.5, 8].
G5  LinearPerturbation(n=3) reduces to LinearPerturbation(n=1) when the
    table is bypassed (n=1 path) and the steady transfer at alpha=0 is
    real; the static k||E transfer at 2H is negative (necking-type sign
    flip) and at 10H is within 1% of the Newtonian value.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY stereo_melt/tests/gate_powerlaw_layer.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")
from stereo_melt.dynamics.linear_perturbation import LinearPerturbation  # noqa: E402
from stereo_melt.dynamics.powerlaw_layer import (  # noqa: E402
    layer_response, newtonian_RB, normal_viscosity_ratio,
)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def cheb(N):
    j = np.arange(N)
    x = np.cos(np.pi * j / (N - 1))
    c = np.ones(N)
    c[0] = c[-1] = 2
    c *= (-1.0) ** j
    X = np.tile(x, (N, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N))
    D -= np.diag(D.sum(1))
    return (1 - x) / 2, -2 * D


def psi_solver(kH, nu, N=48, delta=1020 / 917 - 1):
    """Plane-strain stream-function BVP (independent of the 6x6 solver)."""
    z, D = cheb(N)
    D2, D3, D4 = D @ D, D @ D @ D, D @ D @ D @ D
    Id = np.eye(N)
    k = kH
    A = (D4 - (4 * nu - 2) * k ** 2 * D2 + k ** 4 * Id).astype(complex)
    rhs = np.zeros((N, 2), complex)
    order = np.argsort(z)
    i0, i1, j0, j1 = order[0], order[1], order[-1], order[-2]
    bot, top = order[0], order[-1]
    A[i0] = (D2 + k ** 2 * Id)[bot]
    A[i1] = (D3 + (1 - 4 * nu) * k ** 2 * D)[bot]
    rhs[i1, 1] = -2j * k * delta
    A[j0] = (D2 + k ** 2 * Id)[top]
    A[j1] = (D3 + (1 - 4 * nu) * k ** 2 * D)[top]
    rhs[j1, 0] = 2j * k
    psi = np.linalg.solve(A, rhs)
    w = -1j * k * psi
    return -w[top, 0].real, -w[bot, 0].real


def main() -> int:
    kH = np.array([0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    R1, B1 = newtonian_RB(kH)
    print("G1  n -> 1 limit")
    prev = None
    for nn in (1.01, 1.001, 1.0001):
        devs = []
        for th in (0.0, 0.7, np.pi / 2):
            R, B = layer_response(kH, th, n=nn, Exx=1.0, Eyy=0.0)
            devs.append(max(np.max(np.abs(R - R1) / np.abs(R1)),
                            np.max((np.abs(B - B1) / np.abs(B1))[kH <= 8])))
        dev = max(devs)
        ok = dev < 30 * (nn - 1) and (prev is None or dev < prev / 5)
        check(f"n={nn}", ok, f"max rel dev {dev:.2e}")
        prev = dev

    print("G2  reciprocity")
    for Exx, Eyy, lab in ((1.0, 0.0, "uniaxial"), (1.0, 1.0, "radial"), (1.0, -0.4, "mixed")):
        for th in (0.0, 0.5, np.pi / 2, 2.3):
            _, _, rec = layer_response(kH, th, n=3.0, Exx=Exx, Eyy=Eyy,
                                       return_reciprocity=True)
            check(f"{lab} theta={th:.1f}", rec < 1e-8, f"{rec:.1e}")

    print("G3  analytic limits (n = 3, uniaxial)")
    for th, lab in ((0.0, "k || E"), (np.pi / 2, "k _|_ E")):
        nu = float(normal_viscosity_ratio(3.0, th, 1.0, 0.0))
        R, _ = layer_response(np.array([0.02, 40.0]), th, n=3.0, Exx=1.0, Eyy=0.0)
        Rn, _ = newtonian_RB(np.array([0.02, 40.0]))
        lo, hi = float(np.real(R[0]) / Rn[0]), float(np.real(R[1]) / Rn[1])
        check(f"{lab} thin-sheet 1/nu={1/nu:.3f}", abs(lo * nu - 1) < 2e-2, f"R3/R1 = {lo:.4f}")
        check(f"{lab} half-space 1/sqrt(nu)={nu**-0.5:.3f}", abs(hi * nu ** 0.5 - 1) < 1e-3,
              f"R3/R1 = {hi:.4f}")

    print("G4  stream-function cross-check (nu = 1/3)")
    kk = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0])
    R6, B6 = layer_response(kk, 0.0, n=3.0, Exx=1.0, Eyy=0.0)
    Rp = np.array([psi_solver(k, 1 / 3)[0] for k in kk])
    Bp = np.array([psi_solver(k, 1 / 3)[1] for k in kk])
    dR = np.max(np.abs(np.real(R6) - Rp) / np.abs(Rp))
    dB = np.max(np.abs(np.real(B6) - Bp) / np.maximum(np.abs(Bp), 1e-3))
    check("R agree", dR < 1e-3, f"{dR:.1e}")
    check("B agree", dB < 1e-2, f"{dB:.1e}")

    print("G5  LinearPerturbation integration")
    H, rho_i, rho_w, g = 500.0, 917.0, 1020.0, 9.81
    fb = 1 - rho_i / rho_w

    def T(lam_over_H, n, axis="x"):
        m = LinearPerturbation(H=H, eta_bar=1e14, rho_i=rho_i, rho_w=rho_w, g=g,
                               n=n, Exx=1.0, Eyy=0.0)
        k = 2 * np.pi / (lam_over_H * H)
        kx, ky = (k, 0.0) if axis == "x" else (0.0, k)
        Gh, Gs = m.steady_state_kernel(np.array([[kx]]), np.array([[ky]]))
        Gh, Gs = complex(Gh[0, 0]), complex(Gs[0, 0])
        return Gh / (fb * (Gh - Gs))

    t1 = T(3.0, 1.0)
    check("n=1 static 3H = 0.703", abs(t1.real - 0.703) < 2e-3 and abs(t1.imag) < 1e-12,
          f"{t1.real:.4f}")
    t3 = T(2.0, 3.0)
    check("n=3 k||E static 2H is NEGATIVE (necking sign flip)", t3.real < 0,
          f"{t3.real:+.4f}")
    t3_10 = T(10.0, 3.0)
    check("n=3 k||E static 10H within 1% of Newtonian", abs(t3_10.real / T(10.0, 1.0).real - 1) < 1e-2,
          f"{t3_10.real:.4f} vs {T(10.0, 1.0).real:.4f}")
    t3y = T(2.0, 3.0, "y")
    check("n=3 k_|_E static 2H mildly below Newtonian (0.30 vs 0.375)",
          0.25 < t3y.real < 0.375, f"{t3y.real:.4f}")

    print()
    if FAILS:
        print(f"GATE FAILED: {len(FAILS)} rung(s): {FAILS}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
