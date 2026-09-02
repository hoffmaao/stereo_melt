# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Relaxation and buoyancy transfer functions of a **power-law** ice layer.

Stubblefield, Wearing & Meyer (2023) derive the surface-relaxation function
:math:`R(k)` and the buoyancy transfer function :math:`B(k)` of a floating
layer by solving the perturbed Stokes problem for a **Newtonian** fluid
(their Eq. 2.24–2.29, 2.36–2.37) and note that *"a non-Newtonian reference
state results in a perturbed momentum balance that depends on the stress
exponent and background principal strain rates"* without pursuing it. This
module pursues it.

Linearised Glen rheology
------------------------
With :math:`\tau = 2\eta(\dot\varepsilon_e)\,\dot\varepsilon`,
:math:`\eta = \tfrac12 B\,\dot\varepsilon_e^{(1-n)/n}` and
:math:`\dot\varepsilon_e^2 = \tfrac12\dot\varepsilon:\dot\varepsilon`, a
perturbation :math:`\dot\varepsilon'` about a background strain rate
:math:`\dot\varepsilon^0` carries the stress

.. math::
    \tau' = 2\eta^0\Bigl[\dot\varepsilon'
            - \frac{n-1}{n}\,
              \frac{\dot\varepsilon^0\,(\dot\varepsilon^0:\dot\varepsilon')}
                   {\dot\varepsilon^0:\dot\varepsilon^0}\Bigr],

where :math:`\eta^0` is the *secant* viscosity at the background effective
strain rate. The tangent operator is symmetric and positive definite but
**anisotropic**: strain-rate perturbations parallel to the background
strain tensor see :math:`\eta^0/n`, perturbations orthogonal to it (every
shear component, since the background has no shear) see the full
:math:`\eta^0`. For the depth-independent extensional background of
Stubblefield's reference state, :math:`\dot\varepsilon^0 =
\mathrm{diag}(E_{xx}, E_{yy}, -E)`, the normal-stress reduction for a plane
wave at angle :math:`\theta` to the :math:`x` axis is

.. math::
    \nu_n(\theta) = 1 - \frac{n-1}{2n}\,
        \frac{\bigl[(2E_{xx}+E_{yy})\cos^2\theta + (2E_{yy}+E_{xx})\sin^2\theta\bigr]^2}
             {E_{xx}^2 + E_{yy}^2 + E^2},

which is :math:`1/n` for :math:`k \parallel` a uniaxial extension axis,
:math:`1 - (n-1)/(4n)` (0.83 at :math:`n=3`) for :math:`k \perp` it, and
:math:`1 - 3(n-1)/(4n)` (0.5 at :math:`n=3`) for radial spreading. Off the
principal axes the transverse velocity component couples in, so the layer
problem is solved in full primitive variables rather than with the
Newtonian stream-function reduction.

What is computed
----------------
For each dimensionless wavenumber :math:`k' = kH` and angle :math:`\theta`,
the layer problem with unit topographic loads (surface :math:`h`, base
:math:`s`, Stubblefield Eq. 2.14–2.15) is solved exactly — the layer is
homogeneous, so the :math:`z` dependence is a sum of six exponentials
(eigenmodes of the first-order system :math:`Y' = MY`), fixed by the six
traction conditions — and the vertical velocities at the two surfaces give, in
units of :math:`1/t_r` with :math:`t_r = 2\eta^0/(\rho_i g H)`,

.. math::
    \hat w|_{z=H} = -R\,\hat h - \delta B\,\hat s, \qquad
    \hat w|_{z=0} = -B\,\hat h - \delta R\,\hat s,

exactly the normalisation of Eq. 2.35 / 2.41 with the extensional
:math:`R_0, B_0` terms omitted (as the production kernel omits them). At
:math:`n = 1` the result reproduces Eq. 2.36–2.37 to round-off — that is
the gate. Reciprocity (the two independent estimates of :math:`B`, and of
:math:`R`, from the two unit loads) is checked and reported.

Only the *ratio* :math:`E_{xx}:E_{yy}` and the angle enter the anisotropy;
the magnitude of the background strain rate enters through
:math:`\eta^0` (and :math:`t_r`), which the caller supplies.
"""

from __future__ import annotations

import functools

import numpy as np

__all__ = [
    "layer_response",
    "layer_response_table",
    "normal_viscosity_ratio",
    "newtonian_RB",
    "glen_secant_viscosity",
]


def newtonian_RB(kH):
    """Stubblefield Eq. 2.36–2.37 (the closed form), for the gate."""
    k = np.asarray(kH, dtype=float)
    em2k = np.exp(-2.0 * k)
    em4k = em2k * em2k
    emk = np.exp(-k)
    em3k = em2k * emk
    denom = k * (1.0 - 2.0 * (1.0 + 2.0 * k ** 2) * em2k + em4k)
    R = (1.0 + 4.0 * k * em2k - em4k) / denom
    B = 2.0 * ((k + 1.0) * emk + (k - 1.0) * em3k) / denom
    return R, B


def normal_viscosity_ratio(n: float, theta, Exx: float, Eyy: float):
    r"""Closed-form :math:`\nu_n(\theta)`: tangent/secant viscosity for the
    normal-stress perturbation of a plane wave at angle ``theta`` (rad) to
    the :math:`x` axis. Diagnostic only — the solver uses the full tensor."""
    E = Exx + Eyy
    S = Exx ** 2 + Eyy ** 2 + E ** 2
    if S == 0:
        return np.ones_like(np.asarray(theta, dtype=float))
    c, s = np.cos(theta), np.sin(theta)
    num = (2 * Exx + Eyy) * c ** 2 + (2 * Eyy + Exx) * s ** 2
    return 1.0 - (n - 1.0) / (2.0 * n) * num ** 2 / S


def _system_matrix(kx: float, ky: float, n: float, Exx: float, Eyy: float):
    r"""The constant-coefficient first-order system :math:`Y' = M Y` of the
    linearised power-law layer, :math:`Y = [\hat u, \hat u', \hat v,
    \hat v', \hat w, \hat p]`, in units :math:`2\eta^0 = 1`.

    Derived from the three momentum components, continuity and the tangent
    stress (module docstring); at :math:`n = 1` it is the Newtonian Stokes
    system. Also returns the boundary operators ``(shear_x, shear_y,
    normal)`` as rows acting on ``Y`` (traction-free shear; the normal
    row is :math:`-\hat p + \hat\tau_{zz}`).
    """
    E = Exx + Eyy
    S = Exx ** 2 + Eyy ** 2 + E ** 2
    c = (n - 1.0) / n if (n != 1.0 and S > 0) else 0.0
    k2 = kx ** 2 + ky ** 2
    if S > 0:
        ax, ay = (Exx + E) / S, (Eyy + E) / S  # q = i[ax kx u + ay ky v]
    else:
        ax = ay = 0.0
    M = np.zeros((6, 6), dtype=complex)
    M[0, 1] = 1.0
    M[1, 0] = k2 - 2.0 * c * Exx * ax * kx ** 2
    M[1, 2] = -2.0 * c * Exx * ay * kx * ky
    M[1, 5] = 2j * kx
    M[2, 3] = 1.0
    M[3, 0] = -2.0 * c * Eyy * ax * kx * ky
    M[3, 2] = k2 - 2.0 * c * Eyy * ay * ky ** 2
    M[3, 5] = 2j * ky
    M[4, 0] = -1j * kx
    M[4, 2] = -1j * ky
    M[5, 1] = 1j * kx * (-0.5 + c * E * ax)
    M[5, 3] = 1j * ky * (-0.5 + c * E * ay)
    M[5, 4] = -0.5 * k2
    shear_x = np.array([0, 1, 0, 0, 1j * kx, 0], dtype=complex)   # u' + i kx w
    shear_y = np.array([0, 0, 0, 1, 1j * ky, 0], dtype=complex)   # v' + i ky w
    normal = np.array([1j * kx * (-1.0 + c * E * ax), 0,
                       1j * ky * (-1.0 + c * E * ay), 0, 0, -1.0], dtype=complex)
    return M, shear_x, shear_y, normal


def _solve_one(kH: float, theta: float, n: float, Exx: float, Eyy: float,
               delta: float):
    """Unit-load responses of the layer at one (kH, theta): exact
    exponential-basis solution of the 6x6 two-point problem, with growing
    modes referenced to z = 1 so every basis function is bounded on [0, 1]
    (no e^{+kH} overflow / cancellation)."""
    kx = kH * np.cos(theta)
    ky = kH * np.sin(theta)
    M, shx, shy, nrm = _system_matrix(kx, ky, n, Exx, Eyy)
    lam, V = np.linalg.eig(M)
    grow = lam.real > 0
    # basis phi_j(z) = e^{lam z} (decaying) or e^{lam (z-1)} (growing)
    phi0 = np.where(grow, np.exp(-lam), 1.0)   # value at z = 0
    phi1 = np.where(grow, 1.0, np.exp(lam))    # value at z = 1
    Y0 = V * phi0[None, :]
    Y1 = V * phi1[None, :]
    A = np.vstack([shx @ Y0, shy @ Y0, nrm @ Y0, shx @ Y1, shy @ Y1, nrm @ Y1])
    rhs = np.zeros((6, 2), dtype=complex)   # columns: load h = 1 | load s = 1
    rhs[5, 0] = -1.0        # top: -p + tzz = -h
    rhs[2, 1] = delta       # base: -p + tzz = delta s
    a = np.linalg.solve(A, rhs)
    w_top = (Y1[4] @ a)
    w_bot = (Y0[4] @ a)
    R_h, B_h = -w_top[0], -w_bot[0]
    B_s, R_s = -w_top[1] / delta, -w_bot[1] / delta
    return R_h, B_h, R_s, B_s


def layer_response(kH, theta=0.0, n: float = 1.0, Exx: float = 1.0,
                   Eyy: float = 0.0, delta: float = 1020.0 / 917.0 - 1.0,
                   return_reciprocity: bool = False):
    r"""Return ``(R, B)`` of the (power-law) layer at dimensionless
    wavenumber(s) ``kH`` and angle(s) ``theta``; broadcast together.

    ``n=1`` reproduces :func:`newtonian_RB`. ``Exx, Eyy`` are the background
    principal strain rates (any units; only their ratio and sign pattern
    matter here). ``delta`` = rho_w/rho_i - 1 enters only the base load and
    cancels out of the reported ``R, B``. With ``return_reciprocity`` the
    max relative mismatch between the two independent estimates of ``R``
    and of ``B`` is returned as a third value.
    """
    kH = np.asarray(kH, dtype=float)
    theta = np.asarray(theta, dtype=float)
    kH_b, th_b = np.broadcast_arrays(kH, theta)
    R = np.empty(kH_b.shape, dtype=complex)
    B = np.empty(kH_b.shape, dtype=complex)
    rec = 0.0
    for idx in np.ndindex(kH_b.shape):
        k = float(kH_b[idx])
        if k <= 0:
            R[idx] = B[idx] = 0.0
            continue
        R_h, B_h, R_s, B_s = _solve_one(k, float(th_b[idx]), n, Exx, Eyy,
                                        delta)
        R[idx] = 0.5 * (R_h + R_s)
        B[idx] = 0.5 * (B_h + B_s)
        rec = max(rec,
                  abs(R_h - R_s) / max(abs(R_h), 1e-300),
                  abs(B_h - B_s) / max(abs(B_h), 1e-300))
    # the layer responses are real for a real (non-advected) problem
    R = R.real if np.max(np.abs(R.imag)) < 1e-8 * max(np.max(np.abs(R)), 1e-300) else R
    B = B.real if np.max(np.abs(B.imag)) < 1e-8 * max(np.max(np.abs(B)), 1e-300) else B
    if return_reciprocity:
        return R, B, rec
    return R, B


@functools.lru_cache(maxsize=32)
def layer_response_table(n: float, Exx: float, Eyy: float,
                         kH_min: float = 1e-2, kH_max: float = 60.0,
                         n_k: int = 120, n_theta: int = 25,
                         delta: float = 1020.0 / 917.0 - 1.0):
    r"""Tabulate ``R, B`` on a log-spaced ``kH`` × ``theta`` ∈ [0, π] grid
    for interpolation onto FFT wavenumber grids (``theta`` and ``theta+π``
    are equivalent; the table covers [0, π] inclusive). Cached on its
    arguments (~3000 exact solves, ~10 s) — only the strain-rate RATIO
    enters, so callers should normalise ``(Exx, Eyy)`` before calling."""
    kH = np.logspace(np.log10(kH_min), np.log10(kH_max), n_k)
    th = np.linspace(0.0, np.pi, n_theta)
    R, B = layer_response(kH[:, None], th[None, :], n=n, Exx=Exx, Eyy=Eyy,
                          delta=delta)
    return kH, th, np.real(R), np.real(B)


def glen_secant_viscosity(exx, eyy, exy, A: float, n: float = 3.0,
                          eps_min: float = 1e-12):
    r"""Secant viscosity :math:`\eta^0 = \tfrac12 A^{-1/n}
    \dot\varepsilon_e^{(1-n)/n}` (Pa s) from horizontal strain-rate
    components (1/s) of a shelf in plane stress (``ezz = -(exx+eyy)``),
    :math:`\dot\varepsilon_e^2 = e_{xx}^2 + e_{yy}^2 + e_{xx}e_{yy}
    + e_{xy}^2`. ``A`` in Pa^-n s^-1."""
    exx = np.asarray(exx, dtype=float)
    eyy = np.asarray(eyy, dtype=float)
    exy = np.asarray(exy, dtype=float)
    ee = np.sqrt(np.maximum(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2,
                            eps_min ** 2))
    return 0.5 * A ** (-1.0 / n) * ee ** ((1.0 - n) / n)
