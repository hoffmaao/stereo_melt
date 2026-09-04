# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Linearized forward model of ice-shelf topography response to basal melt.

Implements the Fourier-space forward operator of Stubblefield, Wearing
and Meyer (2023, *Proc. R. Soc. A* 479: 20230290) for a uniform-thickness,
Newtonian, floating ice shelf with depth-independent extension and a
prescribed across-channel background advection. Given a basal melt-rate
perturbation :math:`m(x, y, t)`, the model returns the surface elevation
perturbation :math:`h(x, y, t)` and optionally the basal elevation
perturbation :math:`s(x, y, t)`.

Physics in one summary
----------------------

Non-dimensionalize with reference thickness :math:`H`, viscous
relaxation time :math:`t_r = 2\bar\eta / (\rho_i g H)`, and velocity
scale :math:`H/t_r`. Define the flotation factor
:math:`\delta = \rho_w/\rho_i - 1`, the across-channel advection
parameter :math:`\alpha = \bar u_0\,t_r/H`, and the extensional
parameter :math:`\gamma = E\,t_r`.

In Fourier space at dimensionless wavenumber :math:`k' = kH`, the
topographic-relaxation function :math:`R(k')` and the buoyancy transfer
function :math:`B(k')` are (Stubblefield Eq. 2.36–2.37)

.. math::
    R = \frac{e^{4k'} + 4k'e^{2k'} - 1}
              {k'(e^{4k'} - 2(1 + 2k'^2)e^{2k'} + 1)},
    \qquad
    B = \frac{2\bigl[(k'+1)e^{3k'} + (k'-1)e^{k'}\bigr]}
              {k'(e^{4k'} - 2(1 + 2k'^2)e^{2k'} + 1)}.

Both diverge as :math:`k' \to 0`; a Tikhonov-style regularization
:math:`R_\theta = 1/(\theta + 1/R)`, :math:`B_\theta = 1/(\theta + 1/B)`
with :math:`\theta \sim 10^{-14}` tames the singularity while
preserving the long-wavelength limit (Appendix A of the paper).

The two eigenvalues of the coupled :math:`(\hat h, \hat s)` system are
(Eq. 3.4–3.6)

.. math::
    \lambda_\pm = \gamma - \left(i\,k'_x\,\alpha
                  - \frac{\delta + 1}{2}R \mp \frac{\mu}{2}\right),
    \qquad
    \mu = \sqrt{4\delta B^2 + R^2(\delta - 1)^2}.

The inflow parameter :math:`\alpha` enters only through :math:`ik'_x`,
so the across-channel axis is the :math:`x`-direction and the caller
is expected to rotate the tile into the flow-aligned frame.

Time evolution (Eq. 3.10–3.11):

.. math::
    \hat h(k, t) = \int_0^t K_h(k,\,t-\tilde t)\,\hat m(k, \tilde t)\,d\tilde t,
    \qquad
    K_h(k, t) = -\frac{\delta B}{\mu}\bigl(e^{\lambda_+ t} - e^{\lambda_- t}\bigr),

and analogously for :math:`s` with

.. math::
    K_s(k, t) = \frac{1}{2\mu}\bigl[\,(\mu + (1-\delta)R)e^{\lambda_+ t}
               + (\mu - (1-\delta)R)e^{\lambda_- t}\,\bigr].

For stationary forcing :math:`m(x, y)` the time integral has the closed
form :math:`\hat h(k, t) = -\delta B/\mu \cdot \bigl[(e^{\lambda_+ t}
- 1)/\lambda_+ - (e^{\lambda_- t} - 1)/\lambda_-\bigr]\,\hat m(k)`,
and the steady-state :math:`t \to \infty` limit collapses to the Green's
function of Eq. 3.12.

The :math:`k=0` (DC) bin, across the spectral representations
------------------------------------------------------------
Several operators in :mod:`stereo_melt.dynamics` are diagonal in a spectral
basis, and each has to decide what the :math:`k=0` bin means. They do **not**
all agree, and that is deliberate -- a mean-free perturbation operator and a
forward kernel with a physical long-wavelength limit want opposite things. The
full table, so the difference is explicit rather than implicit:

==================================================  ==========================
operator                                            ``k = 0`` bin
==================================================  ==========================
:meth:`LinearPerturbation.transfer_functions`       ``R = B = 0``. Both diverge
                                                    as :math:`k\to 0`; zeroing
                                                    dodges the 0/0, and every
                                                    caller that needs the true
                                                    limit patches it back.
:meth:`LinearPerturbation.kernel_time_integral_stationary`  analytic
                                                    :math:`I_h(0,t)`,
                                                    :math:`I_s(0,t)` -- finite
                                                    and physical.
:meth:`LinearPerturbation.steady_state_kernel`      analytic
                                                    :math:`t\to\infty` limit of
                                                    the same expressions, so
                                                    :func:`steady_state` and
                                                    ``forward(stationary=True)``
                                                    agree at DC.
:func:`inverse_stationary`, :func:`inverse_dhdt`    structurally blind (they
                                                    invert the zeroed kernel);
                                                    ``recover_dc=True`` splices
                                                    the mean back from a 1-D OLS
                                                    slope of the basin-mean
                                                    :math:`h(t)`.
:class:`~.perturbation_dct.PerturbationForwardOpDCT`  same as the FFT parent --
                                                    it only swaps the wavenumber
                                                    lattice, and DCT bin 0 is
                                                    :math:`k=0` exactly as
                                                    ``fftfreq`` bin 0 is.
:func:`~.stubblefield_forward.stubblefield_forward_multiplier`  pinned to **0**:
                                                    DC-blind BY POLICY, because
                                                    the variational inverse it
                                                    feeds is fitted against a
                                                    high-passed target. See its
                                                    docstring.
:func:`~.budget_bridging.bridging_transfer_multiplier`  pinned to **1**, the
                                                    analytic limit of
                                                    :math:`G_h/(f_b(G_h-G_s))`
                                                    (checked against the kernel
                                                    before pinning).
:func:`~.budget_bridging.normalized_bridging_multiplier`  pinned to **1**; its
                                                    plateau is measured over the
                                                    RESOLVED bins only.
:func:`~.bridging_restoration.bridging_restoration_filter`,
:func:`~.bridging_restoration.bridging_inverse_filter`  pinned to **1**
                                                    (identity / DC passthrough:
                                                    the spatial mean is
                                                    authoritative, not lifted).
==================================================  ==========================

The three ``pinned`` entries are the ones that intentionally override the
kernel: two because unity is what a *ratio* tends to in the hydrostatic limit,
one because the operator's own input has had its mean deleted. Everything else
either carries the physical limit or is honestly blind and says so.

Implementation notes
--------------------

- 2D spatial FFTs over ``(y, x)``; time convolution by direct sum for
  general time-varying :math:`m`, or the closed-form integral for the
  stationary case.
- All arithmetic goes through :mod:`stereo_melt.backend`, so setting
  ``STEREO_MELT_BACKEND=cupy`` dispatches the FFTs and element-wise
  operations to the GPU via ``cupy.fft``.
- The caller is responsible for rotating the tile so the channel is
  along the :math:`y` axis and for cropping a rectangle with
  approximately-uniform background flow and ice thickness.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from ..backend import asarray, backend, to_numpy, xp
from ..constants import rhoi, rhow

__all__ = [
    "LinearPerturbation",
    "forward",
    "steady_state",
    "inverse_stationary",
    "inverse_dhdt",
    "_dctn",
    "_idctn",
    "_dct_wavenumber_grids",
]

G_GRAVITY = 9.81
SECONDS_PER_YEAR = 86400.0 * 365.25


def _expm1_over(lam, t):
    r"""Return :math:`\int_0^t e^{\lambda \tau}\,d\tau = (e^{\lambda t} - 1)/\lambda`.

    Uses :func:`expm1` to avoid catastrophic cancellation for small
    :math:`|\lambda t|`, and a Taylor fallback for :math:`|\lambda t|`
    below :math:`10^{-6}` where ``expm1`` itself still loses precision
    from the final division.
    """
    lt = lam * t
    tiny = 1e-30
    lam_safe = xp.where(xp.abs(lam) > tiny, lam, xp.asarray(tiny, dtype=lam.dtype))
    exact = xp.expm1(lt) / lam_safe
    # Taylor: t(1 + λt/2 + (λt)²/6) — precise to O((λt)³).
    taylor = t * (1.0 + 0.5 * lt + lt * lt / 6.0)
    return xp.where(xp.abs(lt) < 1e-6, taylor, exact)


class LinearPerturbation:
    r"""Linearized ice-shelf response model of Stubblefield 2023.

    Parameters
    ----------
    H : float
        Reference ice thickness in meters. Sets the non-dimensional
        wavenumber :math:`k' = kH` and the thickness scale.
    eta_bar : float
        Column-averaged dynamic viscosity of ice in Pa s. Typical
        values :math:`10^{13}-10^{15}`. Defaults to :math:`10^{14}`
        (Stubblefield §4 reference).
    rho_i, rho_w : float
        Ice and seawater densities in kg m\ :sup:`-3`.
    g : float
        Gravitational acceleration in m s\ :sup:`-2`.
    alpha : float
        Across-channel advection parameter :math:`\bar u_0 t_r / H`
        (dimensionless). Positive values correspond to background flow
        crossing the channel from negative to positive :math:`x`.
    gamma : float
        Extensional parameter :math:`E\,t_r` (dimensionless).
    theta : float
        Long-wavelength regularization for the transfer functions
        (Stubblefield Appendix A). The solution is insensitive over
        the range :math:`10^{-16}-10^{-10}` for typical parameters.
    n : float
        Glen stress exponent of the **perturbation rheology**. ``1``
        (default) is Stubblefield's Newtonian layer. For ``n > 1`` the
        relaxation/buoyancy functions are those of the linearised
        power-law layer (:mod:`.powerlaw_layer`): the tangent viscosity
        about the background strain rate is anisotropic — :math:`\eta^0/n`
        for normal perturbations along a uniaxial extension axis, the full
        :math:`\eta^0` for shear — so ``R, B`` depend on the angle of
        :math:`\mathbf k` to the principal strain axes. ``eta_bar`` is then
        the **secant** viscosity at the background effective strain rate
        (:func:`.powerlaw_layer.glen_secant_viscosity`).
    Exx, Eyy : float
        Background principal strain rates (only their ratio and signs
        enter the anisotropy; ``Exx`` along :math:`x`). Default uniaxial
        extension along :math:`x`, the flow-aligned-frame convention.
    Ephi : float
        Rotation (rad, counter-clockwise from :math:`+x`) of the ``Exx``
        principal axis, for tiles that are not in the flow-aligned frame.

    Attributes
    ----------
    tr : float
        Viscous relaxation time :math:`t_r = 2\bar\eta/(\rho_i g H)`
        in seconds.
    delta : float
        Flotation factor :math:`\rho_w/\rho_i - 1`.
    """

    def __init__(
        self,
        H: float,
        eta_bar: float = 1e14,
        rho_i: float = rhoi,
        rho_w: float = rhow,
        g: float = G_GRAVITY,
        alpha: float = 0.0,
        alpha_y: float = 0.0,
        gamma: float = 0.0,
        theta: float = 1e-14,
        n: float = 1.0,
        Exx: float = 1.0,
        Eyy: float = 0.0,
        Ephi: float = 0.0,
    ):
        self.H = float(H)
        self.eta_bar = float(eta_bar)
        self.rho_i = float(rho_i)
        self.rho_w = float(rho_w)
        self.g = float(g)
        # Backward-compatible single-axis advection: ``alpha`` is the
        # across-channel (x) component (Stubblefield's convention);
        # ``alpha_y`` adds along-channel advection so 2-D Eulerian
        # backgrounds can be handled without a frame rotation.
        self.alpha = float(alpha)
        self.alpha_y = float(alpha_y)
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.tr = 2.0 * self.eta_bar / (self.rho_i * self.g * self.H)
        self.delta = self.rho_w / self.rho_i - 1.0
        self.n = float(n)
        self.Exx = float(Exx)
        self.Eyy = float(Eyy)
        self.Ephi = float(Ephi)  # rotation (rad) of the Exx principal axis from x
        self._rb_table = None

    def _powerlaw_RB(self, kmag, kx, ky):
        r"""``R, B`` of the linearised power-law layer on the wavenumber
        grid, by bilinear interpolation (log :math:`kH` × angle) of a table
        computed once per instance. Angle is taken modulo :math:`\pi`
        (the layer response is even in :math:`\mathbf k`)."""
        from .powerlaw_layer import layer_response_table, newtonian_RB
        if self._rb_table is None:
            # normalise the strain-rate pair so the cache hits on the ratio
            scale = max(abs(self.Exx), abs(self.Eyy), 1e-300)
            self._rb_table = layer_response_table(
                round(self.n, 6), round(self.Exx / scale, 6),
                round(self.Eyy / scale, 6), delta=round(self.delta, 9))
        kH_t, th_t, R_t, B_t = self._rb_table
        kmag_np = to_numpy(kmag)
        kx_np, ky_np = to_numpy(kx), to_numpy(ky)
        # arctan2 in the array frame (plain-fftfreq ky, y descending) gives
        # -phi_map; the response is even in theta, so the map-frame axis
        # rotation Ephi (CCW from +x) enters the table angle with a + sign
        th = np.mod(np.arctan2(ky_np, kx_np) + self.Ephi, np.pi)
        lk = np.log(np.clip(kmag_np, kH_t[0], kH_t[-1]))
        lk_t = np.log(kH_t)
        ik = np.clip(np.searchsorted(lk_t, lk) - 1, 0, lk_t.size - 2)
        it = np.clip(np.searchsorted(th_t, th) - 1, 0, th_t.size - 2)
        wk = (lk - lk_t[ik]) / (lk_t[ik + 1] - lk_t[ik])
        wt = (th - th_t[it]) / (th_t[it + 1] - th_t[it])

        def interp(T):
            return ((1 - wk) * (1 - wt) * T[ik, it] + wk * (1 - wt) * T[ik + 1, it]
                    + (1 - wk) * wt * T[ik, it + 1] + wk * wt * T[ik + 1, it + 1])
        R = interp(R_t)
        B = interp(B_t)
        # beyond the table the Newtonian tail is fine: R scales by its
        # half-space ratio, B is exponentially negligible
        hi = kmag_np > kH_t[-1]
        if hi.any():
            R1, B1 = newtonian_RB(kmag_np[hi])
            R1e, _ = newtonian_RB(np.full(hi.sum(), kH_t[-1]))
            R[hi] = R1 * interp(R_t)[hi] / R1e
            B[hi] = 0.0
        return asarray(R), asarray(B)

    def transfer_functions(self, kx, ky):
        r"""Return :math:`(R, B, \lambda_+, \lambda_-, \mu)` at wavenumber ``(kx, ky)``.

        Parameters
        ----------
        kx, ky : ndarray
            Wavenumber grids in rad m\ :sup:`-1` (output of
            ``2*pi*fftfreq``). Any shape; operations are element-wise.
            Arrays on the active backend.

        Returns
        -------
        R, B : ndarray
            Dimensionless relaxation and buoyancy transfer functions.
        lam_plus, lam_minus : ndarray
            Dimensionless eigenvalues of the coupled system.
        mu : ndarray
            :math:`\mu(k)` as defined by Stubblefield Eq. 3.6.
        """
        H = self.H
        kmag = xp.sqrt(kx**2 + ky**2) * H
        # Floor away from zero so the naive formulas don't divide by 0.
        # We overwrite the k=0 entry at the end.
        kmag_safe = xp.where(kmag > 0, kmag, xp.asarray(1e-30))

        # Factor out exp(4k) from numerator and denominator to keep the
        # exponentials bounded for all k >= 0. Writing R and B in terms
        # of exp(-k) instead of exp(+k) avoids float overflow near Nyquist.
        em2k = xp.exp(-2.0 * kmag_safe)
        em4k = em2k * em2k
        emk = xp.exp(-kmag_safe)
        em3k = em2k * emk

        denom = kmag_safe * (1.0 - 2.0 * (1.0 + 2.0 * kmag_safe**2) * em2k + em4k)

        R_num = 1.0 + 4.0 * kmag_safe * em2k - em4k
        R = R_num / denom

        B_num = 2.0 * ((kmag_safe + 1.0) * emk + (kmag_safe - 1.0) * em3k)
        B = B_num / denom
        if self.n != 1.0:
            R, B = self._powerlaw_RB(kmag_safe, kx, ky)

        # Long-wavelength regularization (Appendix A)
        R_reg = 1.0 / (self.theta + 1.0 / R)
        B_reg = 1.0 / (self.theta + 1.0 / B)

        # k=0 component is the spatial mean; set to zero so perturbations
        # carry no DC offset. Downstream FFTs handle the rest.
        zero_mask = kmag <= 0
        R_reg = xp.where(zero_mask, xp.asarray(0.0), R_reg)
        B_reg = xp.where(zero_mask, xp.asarray(0.0), B_reg)

        mu = xp.sqrt(4.0 * self.delta * B_reg**2 + R_reg**2 * (self.delta - 1.0) ** 2)

        # Eigenvalues of A = [[γ - ikα - R, -Bδ], [-B, γ - ikα - δR]]:
        # λ± = γ - i(αx·kx' + αy·ky') - R(1+δ)/2 ± μ/2.
        # Two-component α generalises Stubblefield's uniform across-channel
        # inflow to uniform 2-D advection — needed for Eulerian wrappers
        # that derive (αx, αy) from MEaSUREs velocity without first
        # rotating the tile to a flow-aligned frame.
        kx_prime = kx * H
        ky_prime = ky * H
        adv = 1j * (kx_prime * self.alpha + ky_prime * self.alpha_y)
        common = adv + 0.5 * (self.delta + 1.0) * R_reg
        lam_plus = self.gamma - (common - 0.5 * mu)
        lam_minus = self.gamma - (common + 0.5 * mu)

        return R_reg, B_reg, lam_plus, lam_minus, mu

    def kernel_time_integral_stationary(self, kx, ky, t_ndim):
        r"""Return :math:`\int_0^t K_h(k,\tau)\,d\tau` and its :math:`s`-analog.

        Closed-form time integral for a stationary forcing, evaluated at
        the non-dimensional time ``t_ndim``. Used for the stationary
        path in :func:`forward`.

        Parameters
        ----------
        kx, ky : ndarray
            Wavenumber grids in rad m\ :sup:`-1`.
        t_ndim : ndarray or float
            Non-dimensional time ``t/tr``. Broadcastable against
            ``kx, ky``.

        Returns
        -------
        I_h, I_s : ndarray
            Time-integrated kernels; :math:`\hat h(k, t) = I_h\,\hat m(k)`
            for stationary :math:`\hat m`, and analogously for :math:`s`.
        """
        R, B, lp, lm, mu = self.transfer_functions(kx, ky)

        # (e^{λt} - 1)/λ is singular at λ=0 only in appearance — its true
        # value is t. For small |λt| the naive form suffers catastrophic
        # cancellation (exp rounds to 1.0 in float64). Use expm1 and a
        # Taylor fallback for |λt| < 1e-6.
        Ip = _expm1_over(lp, t_ndim)
        Im = _expm1_over(lm, t_ndim)

        tiny = 1e-30
        mu_safe = xp.where(xp.abs(mu) > tiny, mu, xp.asarray(tiny))
        I_h = -(self.delta * B / mu_safe) * (Ip - Im)

        # K_s = 1/(2μ)·[(μ + (1-δ)R) e^{λ+ t} + (μ - (1-δ)R) e^{λ- t}]
        # Integral ∫₀ᵗ e^{λτ}dτ = (e^{λt} - 1)/λ, which is what Ip/Im return.
        I_s = (1.0 / (2.0 * mu_safe)) * (
            (mu + (1.0 - self.delta) * R) * Ip + (mu - (1.0 - self.delta) * R) * Im
        )

        # k=0 analytic limit. transfer_functions hard-zeros R and B at
        # the DC mode to dodge a 0/0 in the naive R = R_num/denom
        # evaluation, which propagates a spurious zero into I_h, I_s
        # here. The actual k=0 limit is finite and physical, and
        # steady_state_kernel applies its t -> infinity value: with no
        # spatial gradients the bending operator vanishes, leaving a
        # coupled (h, s) system whose eigenvalues are
        #   λ+(0) = γ - δ/(2(δ+1)),     λ-(0) → -∞,
        # the prefactor δB/μ → δ/(δ+1), and Im → 0. The time-integrated
        # kernels collapse to a single exponential relaxation:
        #   I_h(0, t) = -(δ/(δ+1)) · expm1(λ+(0)·t) / λ+(0)
        #   I_s(0, t) =  (1/(δ+1)) · expm1(λ+(0)·t) / λ+(0)
        # In the long-time limit I_h → -2, matching steady_state_kernel's
        # G_h analytic value, which is the right cross-check. λ+(0) ≥ 0
        # leaves the mode unrelaxed; steady_state_kernel returns NaN there
        # while this integral stays finite at finite t.
        kmag = xp.sqrt(kx ** 2 + ky ** 2)
        zero_mask = kmag <= 0
        if bool(xp.any(zero_mask)):
            delta = self.delta
            c0 = delta / (2.0 * (delta + 1.0))
            lam0 = xp.asarray(self.gamma - c0, dtype=xp.float64)
            I0 = _expm1_over(lam0, t_ndim)
            I_h_zero = -(delta / (delta + 1.0)) * I0
            I_s_zero = I0 / (delta + 1.0)
            I_h = xp.where(zero_mask, xp.asarray(I_h_zero, dtype=I_h.dtype), I_h)
            I_s = xp.where(zero_mask, xp.asarray(I_s_zero, dtype=I_s.dtype), I_s)

        return I_h, I_s

    def steady_state_kernel(self, kx, ky):
        r"""Return the steady-state Green's function ``ĥ_e / m̂`` and its ``s`` analog.

        Evaluates Stubblefield Eq. 3.12–3.13 directly. Dimensional:
        multiply :math:`\hat m` by ``H`` conversion internally if used
        by :func:`steady_state`.

        The :math:`k=0` bin carries the analytic long-wavelength limit — the
        :math:`t\to\infty` value of the very expressions
        :meth:`kernel_time_integral_stationary` integrates, so
        :func:`steady_state` and ``forward(stationary=True)`` agree at DC:

        .. math::
            \lambda_0 = \gamma - \frac{\delta}{2(\delta+1)}, \qquad
            G_h(0) = \frac{\delta/(\delta+1)}{\lambda_0}, \qquad
            G_s(0) = -\frac{1/(\delta+1)}{\lambda_0}.

        :math:`\alpha` does not enter, since :math:`k'_x = k'_y = 0` there.
        At :math:`\gamma = 0` this is :math:`G_h(0) = -2` and
        :math:`G_s(0) = 2/\delta`, so :math:`\hat h/\hat s = -\delta` — exact
        hydrostatic flotation, an infinite-wavelength load carrying no bridging
        — and the flotation departure :math:`T = G_h/(f_b(G_h - G_s))` is
        exactly 1 for **any** :math:`\gamma`, because :math:`\lambda_0`
        cancels.

        The limit is the Newtonian one (it comes from the :math:`k\to 0`
        asymptotics of :math:`R` and :math:`B`), and is applied for
        ``n != 1`` too, matching :meth:`kernel_time_integral_stationary`.

        **No stability screening is applied, at DC or anywhere else.** A mode
        has a steady state only where :math:`\operatorname{Re}\lambda_+ < 0`;
        where it does not, the formula still returns a finite number that is
        the analytic continuation, not a physical steady amplitude. This
        method does not distinguish the two -- masking one bin while returning
        finite values for an equally unrelaxed band would be worse than not
        screening at all. Ask :meth:`unrelaxed_modes` which bins are affected;
        see it for the criterion.
        """
        R, B, lp, lm, mu = self.transfer_functions(kx, ky)
        # ĥ_e = -δ B m̂ / [δ(R² - B²) + (i(αx kx'+αy ky') − γ)(δ+1) R + (i(αx kx'+αy ky') − γ)²]
        # ŝ_e = (R + i(αx kx'+αy ky') − γ) m̂ / [ same denom ]
        kx_prime = kx * self.H
        ky_prime = ky * self.H
        q = 1j * (kx_prime * self.alpha + ky_prime * self.alpha_y) - self.gamma
        denom = self.delta * (R**2 - B**2) + q * (self.delta + 1.0) * R + q**2
        tiny = 1e-30
        denom_safe = xp.where(xp.abs(denom) > tiny, denom, xp.asarray(tiny))
        G_h = -self.delta * B / denom_safe
        G_s = (R + q) / denom_safe
        kmag = xp.sqrt(kx**2 + ky**2)
        zero = kmag <= 0
        if bool(xp.any(zero)):
            delta = self.delta
            lam0 = np.float64(self.dc_eigenvalue())
            with np.errstate(divide="ignore", invalid="ignore"):
                g_h0 = (delta / (delta + 1.0)) / lam0
                g_s0 = -(1.0 / (delta + 1.0)) / lam0
            G_h = xp.where(zero, xp.asarray(g_h0, dtype=G_h.dtype), G_h)
            G_s = xp.where(zero, xp.asarray(g_s0, dtype=G_s.dtype), G_s)
        return G_h, G_s

    def dc_eigenvalue(self) -> float:
        r"""Return :math:`\lambda_0`, the true :math:`k=0` eigenvalue.

        :math:`\lambda_0 = \gamma - \delta/(2(\delta+1))`. This is NOT what
        :meth:`transfer_functions` reports at DC: there ``R`` and ``B`` are
        hard-zeroed, so it returns :math:`\gamma`. The finite limit comes from
        the subleading behaviour of :math:`B/R \to 1`, and :math:`\alpha` does
        not enter because :math:`k'_x = k'_y = 0`.
        """
        return float(self.gamma - self.delta / (2.0 * (self.delta + 1.0)))

    def unrelaxed_modes(self, kx, ky):
        r"""Boolean mask of wavenumbers with **no steady state**.

        A mode relaxes to a steady amplitude only where
        :math:`\operatorname{Re}\lambda_+ < 0`; this returns
        :math:`\operatorname{Re}\lambda_+ \ge 0`. Where it is ``True``,
        :meth:`steady_state_kernel` still returns a finite number, but that
        number is an analytic continuation, not a physical steady state.

        Two regimes, both driven by :math:`\gamma` (extension,
        :math:`\gamma = E t_r`; for a divergent flow
        :math:`\gamma = \nabla\!\cdot\!u\,t_r`):

        * **High wavenumbers.** For ``n = 1`` the large-:math:`k'` asymptotics
          are :math:`R \to 1/k'` and :math:`B \to 0`, so
          :math:`\mu \to R(1-\delta)` and
          :math:`\operatorname{Re}\lambda_+ \to \gamma - \delta R =
          \gamma - \delta/k'`. Every :math:`k' = kH > \delta/\gamma` is
          therefore unrelaxed -- every wavelength below
          :math:`\lambda_c = 2\pi H\gamma/\delta`
          (:meth:`unrelaxed_cutoff_wavelength_m`). ANY :math:`\gamma > 0`
          leaves such a band; it is simply off the grid until :math:`\gamma`
          is large enough. At :math:`H = 500` m and
          :math:`\bar\eta = 10^{14}` Pa s, :math:`t_r = 1.41` yr, so an
          ordinary trunk divergence of 0.04 yr\ :sup:`-1` puts the cut at
          ~1.5 km.
        * **The DC bin**, evaluated at :math:`\lambda_0` from
          :meth:`dc_eigenvalue` rather than at the hard-zeroed :math:`\gamma`
          :meth:`transfer_functions` would report. It is unrelaxed once
          :math:`\gamma \ge \delta/(2(\delta+1))`, and because
          :math:`\operatorname{Re}\lambda_+` grows with :math:`k'`, by then
          every mode is unrelaxed.

        Numerical caveat: below :math:`k' \sim 10^{-2}` the Tikhonov cap
        (:math:`R \to 1/\theta`) makes
        :math:`\tfrac12(\delta+1)R - \tfrac12\mu` a difference of numbers
        :math:`\sim 1/\theta` cancelling to :math:`O(10^{-2})`, so the sign
        there is float noise -- which is exactly why DC is special-cased
        analytically. Real FFT grids do not reach that band (a 250 m posting
        on a 32 km tile has :math:`k'_{\min} \approx 0.1`).
        """
        _R, _B, lam_plus, _lm, _mu = self.transfer_functions(kx, ky)
        re_lp = xp.real(lam_plus)
        kmag = xp.sqrt(kx**2 + ky**2)
        zero = kmag <= 0
        if bool(xp.any(zero)):
            re_lp = xp.where(zero, xp.asarray(self.dc_eigenvalue()), re_lp)
        return re_lp >= 0.0

    def unrelaxed_cutoff_wavelength_m(self) -> float:
        r"""Wavelength below which modes are unrelaxed, :math:`2\pi H\gamma/\delta`.

        The large-:math:`k'` asymptote of :meth:`unrelaxed_modes` (exact to
        ~0.03 % an order of magnitude above the cut, looser as :math:`\gamma`
        approaches :math:`\delta/(2(\delta+1))`, where the band swallows the
        whole spectrum). ``inf`` when :math:`\gamma \le 0`, i.e. no unrelaxed
        band at all -- which is every production path here, all of which run
        :math:`\gamma = 0`.
        """
        if self.gamma <= 0.0:
            return float("inf")
        return float(2.0 * np.pi * self.H * self.gamma / self.delta)


def _wavenumber_grids(nx: int, ny: int, dx: float, dy: float):
    r"""Return ``(kx, ky)`` rad/m grids broadcast to ``(ny, nx)``."""
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    return asarray(kx), asarray(ky)


def _dct_wavenumber_grids(nx: int, ny: int, dx: float, dy: float):
    r"""Return ``(kx, ky)`` rad/m grids on the DCT-II frequency lattice.

    The DCT-II of an :math:`N`-length signal samples wavenumbers
    :math:`k_j = j\pi/(N\Delta x)` for :math:`j = 0, 1, \dots, N-1`.
    Real-valued, broadcast to ``(ny, nx)``; drop-in compatible with
    :meth:`LinearPerturbation.transfer_functions` because the kernel
    depends only on :math:`|\mathbf k|`.
    """
    kx_1d = np.pi * np.arange(nx) / (nx * dx)
    ky_1d = np.pi * np.arange(ny) / (ny * dy)
    kx, ky = np.meshgrid(asarray(kx_1d), asarray(ky_1d), indexing="xy")
    return kx, ky


def _dctn(arr):
    r"""Backend-aware orthonormal 2-D DCT-II on the last two axes."""
    if backend == "cupy":
        from cupyx.scipy.fft import dctn as _d
    else:
        from scipy.fft import dctn as _d
    return _d(arr, type=2, norm="ortho", axes=(-2, -1))


def _idctn(arr):
    r"""Backend-aware orthonormal inverse DCT-II (= DCT-III with ortho norm)."""
    if backend == "cupy":
        from cupyx.scipy.fft import idctn as _i
    else:
        from scipy.fft import idctn as _i
    return _i(arr, type=2, norm="ortho", axes=(-2, -1))


def _times_to_seconds(time_values, relative: bool = True) -> np.ndarray:
    r"""Convert a time coordinate to seconds.

    If ``relative=True`` (default), subtract the first entry so the output
    runs from zero — appropriate for a time-varying ``m`` stack whose
    perturbation effectively turns on at the first epoch. If
    ``relative=False``, the values are treated as elapsed seconds since
    :math:`t = 0` — appropriate for stationary forcing with a caller-
    supplied evaluation time axis.
    """
    arr = np.asarray(time_values)
    if np.issubdtype(arr.dtype, np.datetime64):
        if relative:
            return (arr - arr[0]).astype("timedelta64[s]").astype(np.float64)
        # Assume datetime64 inputs always need a relative anchor; emit seconds from arr[0]
        return (arr - arr[0]).astype("timedelta64[s]").astype(np.float64)
    vals = arr.astype(np.float64)
    if relative:
        vals = vals - float(vals[0])
    return vals


def forward(
    m: xr.DataArray,
    H: float,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    alpha: float = 0.0,
    alpha_y: float = 0.0,
    gamma: float = 0.0,
    theta: float = 1e-14,
    times: xr.DataArray | np.ndarray | None = None,
    stationary: bool = False,
    return_basal: bool = False,
):
    r"""Predict surface elevation anomaly from a basal melt-rate field.

    Parameters
    ----------
    m : xarray.DataArray
        Basal melt-rate anomaly. Either ``dims=(y, x)`` (stationary
        forcing — set ``stationary=True``) or ``dims=(time, y, x)``
        (time-varying forcing). Units: m ice yr\ :sup:`-1`.
    H : float
        Reference ice thickness at the tile, meters.
    eta_bar, rho_i, rho_w, g : float
        Physical constants (see :class:`LinearPerturbation`).
    alpha, gamma : float
        Across-channel advection and extensional parameters
        (dimensionless).
    theta : float
        Long-wavelength regularization.
    times : xarray.DataArray or array-like, optional
        Times at which to evaluate ``h`` when ``m`` is stationary. May
        be datetime64 or numeric seconds since the start. Ignored if
        ``m`` carries a ``time`` dimension.
    stationary : bool
        If True, treat ``m`` as time-invariant and use the closed-form
        time integral. If False, expect ``m`` to carry a ``time`` dim
        and perform a trapezoidal time convolution.
    return_basal : bool
        If True, also return the basal elevation anomaly ``s``.

    Returns
    -------
    h : xarray.DataArray
        Surface elevation anomaly in meters, ``dims=(time, y, x)``.
    s : xarray.DataArray, optional
        Basal elevation anomaly in meters, same shape. Returned iff
        ``return_basal=True``.
    """
    model = LinearPerturbation(
        H=H,
        eta_bar=eta_bar,
        rho_i=rho_i,
        rho_w=rho_w,
        g=g,
        alpha=alpha,
        alpha_y=alpha_y,
        gamma=gamma,
        theta=theta,
    )

    x_coords = m["x"].values
    y_coords = m["y"].values
    ny = m.sizes["y"]
    nx = m.sizes["x"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))

    kx, ky = _wavenumber_grids(nx, ny, dx, dy)

    if stationary:
        if "time" in m.dims:
            raise ValueError("stationary=True but m has a 'time' dim; pass a (y, x) field")
        if times is None:
            raise ValueError("stationary=True requires a 'times' argument")
        t_vals = np.asarray(times.values if isinstance(times, xr.DataArray) else times)
        t_secs = _times_to_seconds(t_vals, relative=False)
        # Convert user m from m/yr to m/s so downstream tr·I_h·m̂ gives meters.
        m_arr = asarray(m.values.astype(np.float64) / SECONDS_PER_YEAR)
        m_hat = xp.fft.fft2(m_arr)

        h_out = np.empty((len(t_secs), ny, nx), dtype=np.float64)
        s_out = np.empty((len(t_secs), ny, nx), dtype=np.float64) if return_basal else None

        for i, t_s in enumerate(t_secs):
            t_ndim = t_s / model.tr
            I_h, I_s = model.kernel_time_integral_stationary(kx, ky, t_ndim)
            # Non-dim derivation: ĥ_dim(k, t) = tr · I_h^ndim · m̂_dim (SI).
            h_hat = (model.tr * I_h) * m_hat
            h_field = xp.fft.ifft2(h_hat).real
            h_out[i] = to_numpy(h_field)
            if return_basal:
                s_hat = (model.tr * I_s) * m_hat
                s_out[i] = to_numpy(xp.fft.ifft2(s_hat).real)

        coords = {
            "time": (times.values if isinstance(times, xr.DataArray) else np.asarray(times)),
            "y": y_coords,
            "x": x_coords,
        }
        h_da = xr.DataArray(h_out, dims=("time", "y", "x"), coords=coords, name="h")
        if return_basal:
            s_da = xr.DataArray(s_out, dims=("time", "y", "x"), coords=coords, name="s")
            return h_da, s_da
        return h_da

    # Time-varying path
    if "time" not in m.dims:
        raise ValueError("non-stationary forward requires m with a 'time' dim")

    t_vals = m["time"].values
    t_secs = _times_to_seconds(t_vals, relative=True)
    n_t = len(t_secs)
    # Convert user m/yr to m/s; ĥ_dim = ∫ K_h · m̂_SI dt gives meters.
    m_arr = asarray(m.values.astype(np.float64) / SECONDS_PER_YEAR)
    m_hat = xp.fft.fft2(m_arr, axes=(-2, -1))  # (t, ky, kx)

    h_out = np.empty((n_t, ny, nx), dtype=np.float64)
    s_out = np.empty((n_t, ny, nx), dtype=np.float64) if return_basal else None

    # Precompute transfer functions; eigenvalues don't depend on t.
    R, B, lp, lm, mu = model.transfer_functions(kx, ky)
    tiny = 1e-30
    mu_safe = xp.where(xp.abs(mu) > tiny, mu, xp.asarray(tiny))

    for n in range(n_t):
        # Trapezoidal ∫₀^{t_n} K_h(t_n - t̃) m̂(t̃) dt̃
        h_hat_n = xp.zeros((ny, nx), dtype=xp.complex128)
        s_hat_n = xp.zeros((ny, nx), dtype=xp.complex128) if return_basal else None
        for i in range(n + 1):
            t_ndim = (t_secs[n] - t_secs[i]) / model.tr
            e_p = xp.exp(lp * t_ndim)
            e_m = xp.exp(lm * t_ndim)
            K_h = -(model.delta * B / mu_safe) * (e_p - e_m)
            # Trapezoid weight
            if i == 0 or i == n:
                w = 0.5
            else:
                w = 1.0
            dt_i = (t_secs[min(i + 1, n_t - 1)] - t_secs[max(i - 1, 0)]) / 2.0
            if i == 0 and n_t > 1:
                dt_i = t_secs[1] - t_secs[0]
            elif i == n_t - 1 and n_t > 1:
                dt_i = t_secs[n_t - 1] - t_secs[n_t - 2]
            # Non-dimensional integration: dt_ndim = dt_dim / tr,
            # result in non-dim h; conversion cancels a factor of tr per the stationary branch.
            h_hat_n = h_hat_n + w * K_h * m_hat[i] * dt_i
            if return_basal:
                K_s = (1.0 / (2.0 * mu_safe)) * (
                    (mu + (1.0 - model.delta) * R) * e_p + (mu - (1.0 - model.delta) * R) * e_m
                )
                s_hat_n = s_hat_n + w * K_s * m_hat[i] * dt_i
        h_field = xp.fft.ifft2(h_hat_n).real
        h_out[n] = to_numpy(h_field)
        if return_basal:
            s_out[n] = to_numpy(xp.fft.ifft2(s_hat_n).real)

    coords = {"time": t_vals, "y": y_coords, "x": x_coords}
    h_da = xr.DataArray(h_out, dims=("time", "y", "x"), coords=coords, name="h")
    if return_basal:
        s_da = xr.DataArray(s_out, dims=("time", "y", "x"), coords=coords, name="s")
        return h_da, s_da
    return h_da


def steady_state(
    m: xr.DataArray,
    H: float,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    alpha: float = 0.0,
    alpha_y: float = 0.0,
    gamma: float = 0.0,
    theta: float = 1e-14,
    return_basal: bool = False,
):
    r"""Return the steady-state :math:`(h_e, s_e)` response to a stationary :math:`m`.

    Evaluates Stubblefield Eq. 3.12–3.13 directly in Fourier space
    (no time integration). Convenience wrapper for the :math:`t
    \to \infty` limit; expect agreement with
    :func:`forward` at large :math:`t/t_r`, including the :math:`k=0`
    mode (a spatially uniform melt does thin the shelf, and
    :meth:`LinearPerturbation.steady_state_kernel` carries the analytic DC
    limit of that relaxation).

    A steady state only exists for modes with
    :math:`\operatorname{Re}\lambda_+ < 0`. Where some do not, the returned
    field still contains numbers -- the analytic continuation -- so this warns
    (``RuntimeWarning``) naming :math:`\gamma`, the cutoff wavelength and how
    many bins are affected, rather than either failing or staying silent. See
    :meth:`LinearPerturbation.unrelaxed_modes`. With :math:`\gamma = 0` (every
    production path here) there are none and nothing is warned.

    Parameters
    ----------
    m : xarray.DataArray, dims ``(y, x)``
        Stationary basal melt-rate anomaly (m ice yr\ :sup:`-1`).
    H, eta_bar, rho_i, rho_w, g, alpha, gamma, theta
        As for :func:`forward`.
    return_basal : bool
        If True, also return the basal anomaly.

    Returns
    -------
    h : xarray.DataArray, dims ``(y, x)``
        Steady-state surface elevation anomaly, meters.
    s : xarray.DataArray, optional
        Steady-state basal anomaly, meters. Returned iff
        ``return_basal=True``.
    """
    if "time" in m.dims:
        raise ValueError("steady_state expects a (y, x) field, not a time stack")

    model = LinearPerturbation(
        H=H,
        eta_bar=eta_bar,
        rho_i=rho_i,
        rho_w=rho_w,
        g=g,
        alpha=alpha,
        alpha_y=alpha_y,
        gamma=gamma,
        theta=theta,
    )

    x_coords = m["x"].values
    y_coords = m["y"].values
    ny = m.sizes["y"]
    nx = m.sizes["x"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))

    kx, ky = _wavenumber_grids(nx, ny, dx, dy)
    bad = model.unrelaxed_modes(kx, ky)
    n_bad = int(to_numpy(bad).sum())
    if n_bad:
        lam_c = model.unrelaxed_cutoff_wavelength_m()
        warnings.warn(
            f"steady_state: {n_bad} of {to_numpy(bad).size} wavenumbers have "
            f"Re(lam_+) >= 0 (no steady state) at gamma={model.gamma:g}; the "
            f"unrelaxed band is wavelengths below ~{lam_c:.4g} m"
            + (" and includes k=0, so no mode relaxes"
               if model.dc_eigenvalue() >= 0 else "")
            + ". The returned field is the analytic continuation there, not a "
            "steady state.", RuntimeWarning, stacklevel=2)
    G_h, G_s = model.steady_state_kernel(kx, ky)

    # Convert user m (m/yr) to SI (m/s); ĥ_dim = tr · G_h · m̂_SI gives meters.
    m_arr = asarray(m.values.astype(np.float64) / SECONDS_PER_YEAR)
    m_hat = xp.fft.fft2(m_arr)

    h_hat = (model.tr * G_h) * m_hat
    h_field = to_numpy(xp.fft.ifft2(h_hat).real)
    h_da = xr.DataArray(h_field, dims=("y", "x"), coords={"y": y_coords, "x": x_coords}, name="h")

    if not return_basal:
        return h_da

    s_hat = (model.tr * G_s) * m_hat
    s_field = to_numpy(xp.fft.ifft2(s_hat).real)
    s_da = xr.DataArray(s_field, dims=("y", "x"), coords={"y": y_coords, "x": x_coords}, name="s")
    return h_da, s_da


def inverse_dhdt(
    dh_dt: xr.DataArray,
    H: float,
    t_secs: np.ndarray,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    alpha: float = 0.0,
    alpha_y: float = 0.0,
    gamma: float = 0.0,
    theta: float = 1e-14,
    reg: float = 1e-3,
    transform: str = "fft",
    recover_dc: bool = True,
    a_dot_dc: float = 0.0,
) -> xr.DataArray:
    r"""Single-step Fourier inverse from a per-pixel dh/dt field.

    Reformulation of the closed-form Stubblefield inverse that decouples the
    per-pixel time fit from the per-wavenumber Fourier inverse. Given an
    observation window with epoch times ``t_secs`` and a stationary forcing,
    the OLS-slope estimator is linear in :math:`\hat h`,

    .. math::
        \hat b(\mathbf k) =
            \frac{\sum_i (t_i - \bar t)\,\hat h(\mathbf k, t_i)}
                 {\sum_i (t_i - \bar t)^2}
            = K_\mathrm{dot}(\mathbf k)\,\hat m(\mathbf k),

    where the OLS-slope kernel is

    .. math::
        K_\mathrm{dot}(\mathbf k) =
            \frac{\sum_i (t_i - \bar t)\,A_i(\mathbf k)}
                 {\sum_i (t_i - \bar t)^2},
        \quad A_i(\mathbf k) = t_r\,I_h(\mathbf k, t_i / t_r).

    Tikhonov inverse:

    .. math::
        \hat m(\mathbf k) =
            \frac{\hat b(\mathbf k)\,K_\mathrm{dot}^*(\mathbf k)}
                 {|K_\mathrm{dot}(\mathbf k)|^2 + \lambda^2}.

    Why this beats :func:`inverse_stationary` under sparse coverage: the
    per-pixel OLS slope is computed independently per pixel using *only*
    that pixel's valid epochs, so coverage gaps don't inject false
    ``h_anom = 0`` observations into the LSQ. The Fourier step then sees
    a single 2-D field with no temporal masking. Coverage gaps still hurt
    (a pixel with one valid epoch can't constrain a slope), but no longer
    bleed across pixels via the FFT zero-fill cliff.

    Parameters
    ----------
    dh_dt : xarray.DataArray, dims ``(y, x)``
        Per-pixel time slope of surface elevation (or freeboard) in m s\ :sup:`-1`.
        NaN-valued pixels are filled with the mean of the finite cells before
        the Fourier transform; supply a floating-mask + finite-domain field
        for best results.
    H : float
        Reference ice thickness in meters.
    t_secs : np.ndarray
        Observation epoch times in seconds (relative). Used to construct
        :math:`(t_i - \bar t)` for the kernel; should match the times used
        when fitting ``dh_dt``.
    eta_bar, rho_i, rho_w, g, alpha, alpha_y, gamma, theta : float
        Same as :func:`inverse_stationary`.
    reg : float
        Tikhonov regularization on :math:`|K_\mathrm{dot}|`, scaled
        internally by the RMS of :math:`|K_\mathrm{dot}|` so the
        user-facing value is dimensionless and dataset-independent
        (matches the convention in :func:`inverse_stationary`). Typical
        range 1e-2 to 1; ``0.1`` is a good starting point. Larger values
        damp low-:math:`\mathbf k` modes more aggressively, suppressing
        the per-pixel OLS noise floor at the cost of some smoothing.
    transform : {"fft", "dct"}
        Spatial basis. FFT for periodic domains, DCT for reflective.
    recover_dc : bool
        If True (default), splice the mass-balance DC mode back into
        the recovered ``m``. The Stubblefield kernel zeros ``k=0`` by
        construction (the membrane response to a uniform melt is
        degenerate), so the spectral inverse loses any uniform offset
        in ``m``. The mass-balance constraint (Shean convention,
        positive = accretion)
        :math:`\overline{\dot b} = R\,\overline{\partial h/\partial t}
        - \overline{\dot a}` (with :math:`R = \rho_w/(\rho_w-\rho_i)`)
        recovers the offset directly from the spatial mean of
        ``dh_dt``. Only valid when bulk advection through the domain
        boundary is small (i.e. no net flux div over the tile); for
        basins where ``alpha`` and ``gamma`` are non-trivial, prefer
        the Eulerian / Lagrangian path-int solvers for the mean.
    a_dot_dc : float
        Spatial-mean SMB rate (m ice / yr) folded into the DC splice
        when ``recover_dc=True``. Defaults to 0.

    Returns
    -------
    xarray.DataArray
        Recovered stationary basal mass balance, dims ``(y, x)``, units
        m ice yr\ :sup:`-1`. Sign convention follows Shean 2019:
        **negative = melt, positive = accretion**. (The internal
        kernel uses Stubblefield's positive=melt convention; the
        output is negated before return so the public API is uniform
        across all solvers in this package.)
    """
    if transform not in ("fft", "dct"):
        raise ValueError(f"transform must be 'fft' or 'dct', got {transform!r}")

    model = LinearPerturbation(
        H=H, eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
        alpha=alpha, alpha_y=alpha_y, gamma=gamma, theta=theta,
    )

    x_coords = dh_dt["x"].values
    y_coords = dh_dt["y"].values
    ny = dh_dt.sizes["y"]
    nx = dh_dt.sizes["x"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[0] - y_coords[1]))

    if transform == "fft":
        kx, ky = _wavenumber_grids(nx, ny, dx, dy)
    else:
        kx, ky = _dct_wavenumber_grids(nx, ny, dx, dy)

    # OLS-slope kernel K_dot(k)
    t_centered = np.asarray(t_secs, dtype=np.float64)
    t_centered = t_centered - t_centered.mean()
    denom = float((t_centered ** 2).sum())
    if denom <= 0:
        raise ValueError("t_secs must span more than one epoch")

    K = xp.zeros_like(kx, dtype=xp.complex128)
    for i, t_s in enumerate(t_secs):
        I_h, _ = model.kernel_time_integral_stationary(kx, ky, float(t_s) / model.tr)
        A_i = model.tr * I_h
        K = K + float(t_centered[i]) * A_i
    K = K / denom
    if transform == "dct":
        K = xp.real(K)

    # Fill NaNs with the finite-cell mean for the transform; the user is
    # expected to mask the output to the floating area afterwards.
    dh_vals = dh_dt.values.astype(np.float64)
    finite = np.isfinite(dh_vals)
    if not finite.any():
        raise ValueError("dh_dt contains no finite cells")
    fill = float(dh_vals[finite].mean())
    dh_filled = asarray(np.where(finite, dh_vals, fill))

    k_rms = float(to_numpy(xp.sqrt((xp.abs(K) ** 2).mean())))
    lam2 = (reg * k_rms) ** 2

    if transform == "fft":
        b_hat = xp.fft.fft2(dh_filled)
        m_hat = b_hat * xp.conj(K) / (xp.abs(K) ** 2 + lam2)
        m_si = to_numpy(xp.fft.ifft2(m_hat).real)
    else:
        b_hat = _dctn(dh_filled)
        m_hat = b_hat * K / (K * K + lam2)
        m_si = to_numpy(_idctn(m_hat))

    if recover_dc:
        # Mass-balance DC mode (no advection through boundary, stationary
        # forcing). Internally we stay in Stubblefield's positive=melt
        # convention: mean(dh/dt) * R = -mean(m) + mean(a_dot), with
        # R = rho_w / (rho_w - rho_i). The spectral inverse zeros m at
        # k=0 by construction (kernel R_reg, B_reg both 0 there); we
        # restore the spatial-mean offset analytically. The final
        # negation below flips both the AC and DC modes to the Shean
        # public convention (positive = accretion).
        R_hydro = rho_w / (rho_w - rho_i)
        a_dot_si = float(a_dot_dc) / SECONDS_PER_YEAR
        mean_dh_dt = float(dh_vals[finite].mean())
        m_dc_si = -R_hydro * mean_dh_dt + a_dot_si
        m_si = m_si - m_si.mean() + m_dc_si

    # Internal Stubblefield kernel maps positive m -> downward h (melt-as-
    # positive). Negate to publish the package-wide Shean convention
    # (positive = accretion, negative = melt).
    m_m_per_yr = -m_si * SECONDS_PER_YEAR
    return xr.DataArray(
        m_m_per_yr, dims=("y", "x"), coords={"y": y_coords, "x": x_coords},
        name="m",
        attrs={
            "units": "m ice yr^-1; Shean convention: negative = melt, positive = accretion",
            "H": H, "eta_bar": eta_bar, "alpha": alpha, "gamma": gamma,
            "reg": reg, "transform": transform,
            "t_window_yr": float((t_secs[-1] - t_secs[0]) / SECONDS_PER_YEAR),
            "n_epochs": int(len(t_secs)),
            "method": "dh_dt single-step Fourier inverse",
        },
    )


def inverse_stationary(
    h_stack: xr.DataArray,
    H: float,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    alpha: float = 0.0,
    alpha_y: float = 0.0,
    gamma: float = 0.0,
    theta: float = 1e-14,
    reg: float = 1e-3,
    reference: str = "first_epoch",
    transform: str = "fft",
    recover_dc: bool = True,
    a_dot_dc: float = 0.0,
) -> xr.DataArray:
    r"""Invert a surface-elevation stack for a stationary basal melt rate.

    Assumes :math:`m(x, y)` is time-invariant over the observation
    window. For a stationary forcing the forward operator is diagonal
    in Fourier space at each observation time,

    .. math::
        \hat h(k, t_i) = \underbrace{t_r\,I_h(k, t_i)}_{A_i(k)}\,\hat m(k),

    and the space-time inverse decouples into one least-squares problem
    per wavenumber with a closed-form Tikhonov solution

    .. math::
        \hat m(k) = \frac{\sum_i A_i^\ast(k)\,\hat h(k, t_i)}
                         {\sum_i |A_i(k)|^2 + \lambda^2}.

    This is the proposal's Eq. 3–4 restricted to stationary
    :math:`m`; the general time-varying inverse adds a conjugate-
    gradient sweep over the LinearOperator built from :func:`forward`.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Observed surface-elevation anomaly time series, meters. See
        ``reference`` for the assumed anomaly definition.
    H, eta_bar, rho_i, rho_w, g, alpha, gamma, theta
        Same as :func:`forward`.
    reg : float
        Tikhonov regularization :math:`\lambda` scaled relative to the
        per-wavenumber RMS of :math:`|A_i(k)|`. Larger values smooth
        the inversion; typical range :math:`10^{-4}-10^{-2}`.
    reference : {"first_epoch", "time_mean"}
        How ``h_stack`` is interpreted as a perturbation anomaly:

        - ``"first_epoch"`` (default): ``h_stack[0]`` is the
          ``t = 0`` reference (perturbation has built up to
          ``h_stack[i]`` by ``t_i``). Kernel ``A_i = t_r · I_h(t_i)``.
        - ``"time_mean"``: ``h_stack[i]`` is the deviation from the
          per-pixel time mean over the observation window. Kernel
          ``A_i = t_r · (I_h(t_i) - ⟨I_h⟩)``. Use this when the first
          epoch has incomplete spatial coverage (e.g. REMA strips),
          because the time-mean anomaly preserves per-pixel coverage
          while the first-epoch anomaly zeros every pixel the anchor
          didn't observe.
    transform : {"fft", "dct"}
        Spatial basis used to diagonalize the (translation-invariant /
        reflection-symmetric) Stubblefield kernel. ``"fft"`` uses the
        2-D periodic basis with ``2π·fftfreq`` wavenumbers — appropriate
        when the data envelope fills the analysis grid or has been
        padded to a quasi-periodic continuation. ``"dct"`` uses the
        DCT-II reflective basis with wavenumbers ``k_j = jπ/(NΔ)`` —
        appropriate when the data envelope occupies a fraction of the
        bounding box and a periodic wrap would couple the field to its
        opposite edge. The kernel ``A_i(k)`` depends only on ``|k|`` so
        it remains diagonal in either basis; the closed-form Tikhonov
        normal equations are identical, with conjugation collapsing to
        a no-op in DCT space (real arithmetic when ``α = α_y = 0``).

    Returns
    -------
    xarray.DataArray
        Recovered stationary basal mass balance, dims ``(y, x)``, units
        m ice yr\ :sup:`-1`. Sign convention follows Shean 2019:
        **negative = melt, positive = accretion**.
    """
    if "time" not in h_stack.dims:
        raise ValueError("inverse_stationary requires a 'time' dim on h_stack")
    if reference not in ("first_epoch", "time_mean"):
        raise ValueError(
            f"reference must be 'first_epoch' or 'time_mean', got {reference!r}"
        )
    if transform not in ("fft", "dct"):
        raise ValueError(f"transform must be 'fft' or 'dct', got {transform!r}")

    model = LinearPerturbation(
        H=H,
        eta_bar=eta_bar,
        rho_i=rho_i,
        rho_w=rho_w,
        g=g,
        alpha=alpha,
        alpha_y=alpha_y,
        gamma=gamma,
        theta=theta,
    )

    x_coords = h_stack["x"].values
    y_coords = h_stack["y"].values
    ny = h_stack.sizes["y"]
    nx = h_stack.sizes["x"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))

    if transform == "fft":
        kx, ky = _wavenumber_grids(nx, ny, dx, dy)
    else:
        kx, ky = _dct_wavenumber_grids(nx, ny, dx, dy)

    t_secs = _times_to_seconds(h_stack["time"].values, relative=True)

    # Pre-compute A_i per epoch; for time_mean reference we subtract the
    # epoch-average kernel from each A_i. The h_stack input is already
    # mean-subtracted by the caller in that mode, so the LS problem
    #     min_m̂ Σ_i || A_i m̂ - ĥ_i ||²
    # remains well-posed with the same closed-form Tikhonov solution.
    A_list = []
    for t_s in t_secs:
        t_ndim = t_s / model.tr
        I_h, _ = model.kernel_time_integral_stationary(kx, ky, t_ndim)
        A_i = model.tr * I_h
        if transform == "dct":
            # DCT-II of a real signal is real-valued. The Stubblefield
            # kernel A_i(k) only acquires an imaginary part through the
            # advection term i·(αx·kx + αy·ky); with α=α_y=0 (Lagrangian
            # frame) it is purely real. Take Re(A_i) so the basis match
            # holds defensively even if a caller leaves α non-zero.
            A_i = xp.real(A_i)
        A_list.append(A_i)
    if reference == "time_mean":
        A_mean = sum(A_list) / len(A_list)
        A_list = [A_i - A_mean for A_i in A_list]

    # Accumulate normal-equation terms per wavenumber:
    #   num(k) = Σ_i A_i*(k) · ĥ(k, t_i)
    #   den(k) = Σ_i |A_i(k)|^2
    if transform == "fft":
        num = xp.zeros((ny, nx), dtype=xp.complex128)
        den = xp.zeros((ny, nx), dtype=xp.float64)
        for i, A_i in enumerate(A_list):
            h_slice = asarray(h_stack.isel(time=i).values.astype(np.float64))
            h_hat_i = xp.fft.fft2(h_slice)
            num = num + xp.conj(A_i) * h_hat_i
            den = den + (A_i.real * A_i.real + A_i.imag * A_i.imag)
    else:
        # DCT closed form: A_i and ĥ_i are real, so conj() is a no-op
        # and the normal equations stay in real arithmetic throughout.
        num = xp.zeros((ny, nx), dtype=xp.float64)
        den = xp.zeros((ny, nx), dtype=xp.float64)
        for i, A_i in enumerate(A_list):
            h_slice = asarray(h_stack.isel(time=i).values.astype(np.float64))
            h_hat_i = _dctn(h_slice)
            num = num + A_i * h_hat_i
            den = den + A_i * A_i

    # Per-wavenumber Tikhonov: scale reg by the RMS |A| to make `reg`
    # dimensionless and dataset-independent.
    a_rms = float(to_numpy(xp.sqrt(den.mean())))
    lam2 = (reg * a_rms) ** 2

    m_hat_si = num / (den + lam2)
    if transform == "fft":
        m_si = to_numpy(xp.fft.ifft2(m_hat_si).real)
    else:
        m_si = to_numpy(_idctn(m_hat_si))

    if recover_dc:
        # Splice the mass-balance DC mode back in. The stationary
        # spectral inverse is structurally blind to k=0 (kernel R_reg,
        # B_reg both zero there); we recover the spatial-mean melt rate
        # from a 1D OLS slope of the basin-mean h(t) time series.
        # Internally we stay in Stubblefield's positive=melt convention
        # (m_DC = -R · ⟨dh/dt⟩ + ⟨a_dot⟩, no advection through tile);
        # the final negation below flips everything to the Shean public
        # convention.
        h_arr = h_stack.values
        h_basin_mean = np.array(
            [float(np.nanmean(h_arr[k])) for k in range(h_arr.shape[0])],
            dtype=np.float64,
        )
        valid = np.isfinite(h_basin_mean) & np.isfinite(t_secs)
        if int(valid.sum()) >= 2:
            t_v = np.asarray(t_secs)[valid]
            h_v = h_basin_mean[valid]
            t_c = t_v - t_v.mean()
            denom_t = float((t_c * t_c).sum())
            if denom_t > 0:
                mean_dh_dt = float(((h_v - h_v.mean()) * t_c).sum() / denom_t)
                R_hydro = rho_w / (rho_w - rho_i)
                a_dot_si = float(a_dot_dc) / SECONDS_PER_YEAR
                m_dc_si = -R_hydro * mean_dh_dt + a_dot_si
                m_si = m_si - m_si.mean() + m_dc_si

    # Internal Stubblefield kernel maps positive m -> downward h (melt-as-
    # positive). Negate to publish the package-wide Shean convention
    # (positive = accretion, negative = melt).
    m_m_per_yr = -m_si * SECONDS_PER_YEAR

    return xr.DataArray(
        m_m_per_yr,
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        name="m",
        attrs={
            "units": "m ice yr^-1; Shean convention: negative = melt, positive = accretion",
            "H": H,
            "eta_bar": eta_bar,
            "alpha": alpha,
            "gamma": gamma,
            "reg": reg,
            "transform": transform,
            "recover_dc": int(bool(recover_dc)),
        },
    )
