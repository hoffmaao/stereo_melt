"""Pseudo-spectral forward + adjoint + CG inverter for time-dependent
linear-perturbation inversion.

This module implements Step 1 of both ``plan_eulerian.md`` and
``plan_lagrangian.md`` in the Level-0 (constant-background) limit. The
forward operator is a time-domain convolution using the analytic
Stubblefield kernel ``K_h(k, t)``; the spatial structure is diagonalized
by FFTs (the "pseudo-spectral" part) while the time axis is integrated
by trapezoidal quadrature.

Companion to :mod:`stereo_melt.dynamics.linear_perturbation`, which
provides the closed-form steady-state forward/inverse for the same
governing equations. The pseudo-spectral path keeps the same physics
but exposes a forward + adjoint pair so a CG-based time-dependent
inverse can recover ``m'(x, y, t)`` (rather than the time-mean).

Architecture
------------

- :class:`PerturbationForwardOp` --- forward and adjoint as a pair of
  methods on a single class, parameterised by background state
  ``(H, eta_bar, alpha, gamma, ...)`` and the observation time axis.
  The forward wraps the time-varying path of
  :func:`stereo_melt.dynamics.linear_perturbation.forward`. The adjoint
  is the transposed time convolution with the conjugate kernel.
- :func:`cg_invert` --- conjugate-gradient solver on the normal
  equations with a simple Tikhonov regulariser.
- Frame-specific wrappers (Eulerian / Lagrangian) live in their own
  modules and compose this operator with frame-specific data
  preprocessing (e.g. trajectory advection for Lagrangian).

Notes
-----

This is Level 0 in the plan hierarchy: single uniform background per
inversion. Tiling, δH expansion, WKB, and full FE are deferred.
TV regularisation is also deferred --- the inner CG loop here is
exactly the weighted-L2 sub-problem the eventual lagged-diffusivity
outer loop will call.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..backend import asarray, to_numpy, xp
from ..constants import rhoi, rhow
from .linear_perturbation import LinearPerturbation, _times_to_seconds, _wavenumber_grids

__all__ = [
    "PerturbationForwardOp",
    "LinearizedHForwardOp",
    "cg_invert",
    "cg_invert_stationary",
    "cg_invert_basis",
    "make_temporal_basis",
    "inverse_time_varying",
]

G_GRAVITY = 9.81
SECONDS_PER_YEAR = 86400.0 * 365.25


def _trapezoid_dt_weights(t_secs: np.ndarray) -> np.ndarray:
    r"""Composite-trapezoidal per-sample weights for ``Σ_i w_i Δt_i f_i``.

    Returns a 1-D array of length ``len(t_secs)`` whose entries already
    include the ``w_i Δt_i`` product (so the caller just multiplies the
    integrand and sums). Endpoints get half weight; interior gets the
    average of neighbouring intervals.
    """
    n = len(t_secs)
    if n == 1:
        return np.array([1.0], dtype=np.float64)
    w = np.empty(n, dtype=np.float64)
    w[0] = 0.5 * (t_secs[1] - t_secs[0])
    w[-1] = 0.5 * (t_secs[-1] - t_secs[-2])
    for i in range(1, n - 1):
        w[i] = 0.5 * (t_secs[i + 1] - t_secs[i - 1])
    return w


class PerturbationForwardOp:
    r"""Time-dependent linear-perturbation forward + adjoint.

    For uniform background ``H, eta_bar, alpha, gamma`` and a regular
    spatial grid, applies

    .. math::
        \hat h(\mathbf{k}, t_n) = \sum_{i \le n} w_i \Delta t_i \,
            K_h(\mathbf{k},\, t_n - t_i)\, \hat m(\mathbf{k}, t_i)

    where ``K_h`` is the Stubblefield surface-response kernel from
    :func:`LinearPerturbation.transfer_functions`, ``w_i Δt_i`` are
    composite-trapezoidal weights, and the spectral coordinates are
    diagonalised by 2-D FFT.

    The adjoint computes the transpose of the same operator (with
    complex-conjugate kernel) so a Gauss--Newton / CG inverse can be
    built without autodiff.

    Parameters
    ----------
    H : float
        Reference ice thickness, m.
    times : array-like or xarray.DataArray
        Observation times. Datetime64 or numeric seconds. Internally
        reduced to seconds-since-first-epoch.
    nx, ny : int
        Spatial grid sizes (note ordering: ``ny`` rows, ``nx`` columns).
    dx, dy : float
        Grid spacing, meters.
    eta_bar, rho_i, rho_w, g, alpha, gamma, theta : float
        Same as :class:`LinearPerturbation`.

    Attributes
    ----------
    n_t : int
    tr : float
        Viscous relaxation time, s.
    dt_weights : ndarray, shape (n_t,)
        Composite-trapezoidal weights ``w_i Δt_i`` in seconds.
    """

    def __init__(
        self,
        H: float,
        times,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        eta_bar: float = 1e14,
        rho_i: float = rhoi,
        rho_w: float = rhow,
        g: float = G_GRAVITY,
        alpha: float = 0.0,
        alpha_y: float = 0.0,
        gamma: float = 0.0,
        theta: float = 1e-14,
    ):
        self.model = LinearPerturbation(
            H=H, eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
            alpha=alpha, alpha_y=alpha_y, gamma=gamma, theta=theta,
        )
        self.tr = self.model.tr
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)

        if isinstance(times, xr.DataArray):
            t_vals = times.values
        else:
            t_vals = np.asarray(times)
        self.times = t_vals
        self.t_secs = _times_to_seconds(t_vals, relative=True)
        self.n_t = len(self.t_secs)
        self.dt_weights = _trapezoid_dt_weights(self.t_secs)

        kx, ky = _wavenumber_grids(nx, ny, dx, dy)
        self.kx = kx
        self.ky = ky
        R, B, lp, lm, mu = self.model.transfer_functions(kx, ky)
        tiny = 1e-30
        self.R = R
        self.B = B
        self.lp = lp
        self.lm = lm
        self.mu = mu
        self.mu_safe = xp.where(xp.abs(mu) > tiny, mu, xp.asarray(tiny))

    # ------------------------------------------------------------------
    # Stationary helpers
    # ------------------------------------------------------------------
    def _build_K_cum(self):
        r"""Precompute the cumulative time-integrated kernel per epoch.

        For a stationary unknown :math:`m(\mathbf{x})`, the time
        convolution at each observation epoch collapses analytically:

        .. math::
            \hat h(\mathbf{k}, t_n) = \hat m(\mathbf{k})
                \sum_{i \le n} w_i \Delta t_i\, K_h(\mathbf{k},\, t_n - t_i)
                \;\equiv\; \hat m(\mathbf{k}) \cdot K_{\text{cum}}[n](\mathbf{k}).

        Storing :math:`K_{\text{cum}}[n]` once turns the
        :class:`forward_stationary` and :class:`adjoint_stationary`
        inner loops from :math:`O(n_t^2)` to :math:`O(n_t)` per CG
        iteration. Cached on first call; size
        ``(n_t, ny, nx) complex128``.
        """
        if hasattr(self, "_K_cum_cached"):
            return self._K_cum_cached
        K_cum = xp.zeros((self.n_t, self.ny, self.nx), dtype=xp.complex128)
        for n in range(self.n_t):
            for i in range(n + 1):
                K_cum[n] = (
                    K_cum[n]
                    + self.dt_weights[i]
                    * self._Kh(self.t_secs[n] - self.t_secs[i])
                )
        self._K_cum_cached = K_cum
        return K_cum

    def _build_K_cum_centered(self):
        r"""Cumulative kernel with the time-mean removed across ``n``.

        Used when the observed ``h_obs`` has been per-pixel time-mean
        subtracted (so that ``h_obs(t,x,y) := h_lag(t,x,y) - <h_lag>_t``).
        The forward model on this centered observation is

        .. math::
            h_{\text{obs}}(\mathbf{k}, t_n)
              = (K_{\text{cum}}[n] - \langle K_{\text{cum}} \rangle_n) \,
                \hat m(\mathbf{k}),

        which is what :meth:`forward_stationary` returns when
        ``reference == "time_mean"``. Avoids the trivial-zero bug
        where centered data is fit against an uncentered forward.
        """
        if hasattr(self, "_K_cum_centered_cached"):
            return self._K_cum_centered_cached
        K_cum = self._build_K_cum()
        K_mean = K_cum.mean(axis=0, keepdims=True)
        self._K_cum_centered_cached = K_cum - K_mean
        return self._K_cum_centered_cached

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------
    def _Kh(self, dt_seconds: float):
        r"""Return ``K_h(k, dt)`` evaluated on the spectral grid.

        ``K_h(k, t) = -(δ B / μ) (e^{λ_+ t} - e^{λ_- t})`` with
        non-dimensional time ``t / t_r``.
        """
        t_ndim = dt_seconds / self.tr
        e_p = xp.exp(self.lp * t_ndim)
        e_m = xp.exp(self.lm * t_ndim)
        return -(self.model.delta * self.B / self.mu_safe) * (e_p - e_m)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, m: np.ndarray) -> np.ndarray:
        r"""Apply the forward operator to ``m(t, y, x)`` in m/yr.

        Returns ``h(t, y, x)`` in meters with the same shape.
        """
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (self.n_t, self.ny, self.nx):
            raise ValueError(
                f"m has shape {m.shape}, expected ({self.n_t}, {self.ny}, {self.nx})"
            )
        m_si = asarray(m / SECONDS_PER_YEAR)
        m_hat = xp.fft.fft2(m_si, axes=(-2, -1))

        h_out = np.empty((self.n_t, self.ny, self.nx), dtype=np.float64)
        for n in range(self.n_t):
            h_hat_n = xp.zeros((self.ny, self.nx), dtype=xp.complex128)
            for i in range(n + 1):
                K = self._Kh(self.t_secs[n] - self.t_secs[i])
                h_hat_n = h_hat_n + (self.dt_weights[i] * K) * m_hat[i]
            h_out[n] = to_numpy(xp.fft.ifft2(h_hat_n).real)
        return h_out

    # ------------------------------------------------------------------
    # Adjoint
    # ------------------------------------------------------------------
    def adjoint(self, h: np.ndarray) -> np.ndarray:
        r"""Apply the adjoint operator to ``h(t, y, x)`` in meters.

        Returns ``m_adj(t, y, x)`` in m/yr.

        The forward writes
        ``ĥ(t_n) = Σ_{i ≤ n} w_i K(t_n - t_i) m̂(t_i)``
        so the adjoint is
        ``m̂_adj(t_i) = Σ_{n ≥ i} w_i conj(K(t_n - t_i)) ĥ(t_n)``.
        Verified against ``<G m, h> = <m, G^T h>`` to ~1e-10 in the
        sanity test.
        """
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (self.n_t, self.ny, self.nx):
            raise ValueError(
                f"h has shape {h.shape}, expected ({self.n_t}, {self.ny}, {self.nx})"
            )
        h_arr = asarray(h)
        h_hat = xp.fft.fft2(h_arr, axes=(-2, -1))

        m_adj_out = np.empty((self.n_t, self.ny, self.nx), dtype=np.float64)
        for i in range(self.n_t):
            m_hat_i = xp.zeros((self.ny, self.nx), dtype=xp.complex128)
            wi = self.dt_weights[i]
            for n in range(i, self.n_t):
                K_conj = xp.conj(self._Kh(self.t_secs[n] - self.t_secs[i]))
                m_hat_i = m_hat_i + (wi * K_conj) * h_hat[n]
            m_adj_si = xp.fft.ifft2(m_hat_i).real
            # Forward converts m m/yr -> m/s by /SECONDS_PER_YEAR.
            # Adjoint of that scalar multiplication is the same /SECONDS_PER_YEAR
            # to keep <Gm, h>_(meters) = <m, G^T h>_(m/yr).
            m_adj_out[i] = to_numpy(m_adj_si) / SECONDS_PER_YEAR
        return m_adj_out

    # ------------------------------------------------------------------
    # Stationary forward / adjoint (single 2-D unknown m(x, y))
    # ------------------------------------------------------------------
    def forward_stationary(
        self, m_2d: np.ndarray, reference: str = "anchor_first"
    ) -> np.ndarray:
        r"""Forward operator for a stationary basal melt rate.

        Given ``m(y, x)`` in m/yr, returns ``h(t, y, x)`` in meters via

        .. math::
            \hat h(\mathbf{k}, t_n) = K_{\text{cum,ref}}[n](\mathbf{k}) \,
                \hat m(\mathbf{k}),

        where ``K_{cum,ref}`` is :meth:`_build_K_cum` for
        ``reference="anchor_first"`` (the convention that
        ``h(t=0) = 0``) and :meth:`_build_K_cum_centered` for
        ``reference="time_mean"`` (the convention that
        ``<h>_t = 0`` per pixel).
        """
        m_2d = np.asarray(m_2d, dtype=np.float64)
        if m_2d.shape != (self.ny, self.nx):
            raise ValueError(
                f"m_2d has shape {m_2d.shape}, expected ({self.ny}, {self.nx})"
            )
        m_si = asarray(m_2d / SECONDS_PER_YEAR)
        m_hat = xp.fft.fft2(m_si)
        K = self._kernel_for_reference(reference)
        h_out = np.empty((self.n_t, self.ny, self.nx), dtype=np.float64)
        for n in range(self.n_t):
            h_out[n] = to_numpy(xp.fft.ifft2(K[n] * m_hat).real)
        return h_out

    def adjoint_stationary(
        self, h: np.ndarray, reference: str = "anchor_first"
    ) -> np.ndarray:
        r"""Adjoint of :meth:`forward_stationary` for the same ``reference``."""
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (self.n_t, self.ny, self.nx):
            raise ValueError(
                f"h has shape {h.shape}, expected ({self.n_t}, {self.ny}, {self.nx})"
            )
        h_arr = asarray(h)
        h_hat = xp.fft.fft2(h_arr, axes=(-2, -1))
        K = self._kernel_for_reference(reference)
        m_hat_adj = xp.zeros((self.ny, self.nx), dtype=xp.complex128)
        for n in range(self.n_t):
            m_hat_adj = m_hat_adj + xp.conj(K[n]) * h_hat[n]
        m_adj_si = xp.fft.ifft2(m_hat_adj).real
        return to_numpy(m_adj_si) / SECONDS_PER_YEAR

    def _kernel_for_reference(self, reference: str):
        if reference == "anchor_first":
            return self._build_K_cum()
        if reference == "time_mean":
            return self._build_K_cum_centered()
        raise ValueError(
            f"reference must be 'anchor_first' or 'time_mean', got {reference!r}"
        )

    # ------------------------------------------------------------------
    # Reduced temporal-basis forward / adjoint
    #   m(x, y, t) = Σ_j c_j(x, y) φ_j(t),  J ≪ n_t
    # ------------------------------------------------------------------
    def build_basis_kernels(self, phi):
        r"""Precompute ``A[n, j](k) = Σ_{i≤n} w_i K_h(t_n - t_i) φ_j(t_i)``.

        ``phi`` is a real ``(n_t, J)`` array of the temporal basis
        functions sampled at the observation epochs (see
        :func:`make_temporal_basis`). Returns a backend array of shape
        ``(n_t, J, ny, nx)`` complex128.

        For ``φ_0 ≡ 1`` the ``j = 0`` slice is exactly
        :meth:`_build_K_cum`, so :meth:`forward_basis` at ``J = 1``
        coincides with :meth:`forward_stationary` (``reference="anchor_first"``)
        --- the reduced basis is the minimal generalisation of the
        stationary operator, not a new one.
        """
        phi = np.asarray(phi, dtype=np.float64)
        if phi.ndim != 2 or phi.shape[0] != self.n_t:
            raise ValueError(f"phi must be (n_t={self.n_t}, J), got {phi.shape}")
        J = phi.shape[1]
        A = xp.zeros((self.n_t, J, self.ny, self.nx), dtype=xp.complex128)
        for n in range(self.n_t):
            for i in range(n + 1):
                Kh = self._Kh(self.t_secs[n] - self.t_secs[i])  # (ny, nx)
                wi = self.dt_weights[i]
                for j in range(J):
                    A[n, j] = A[n, j] + (wi * float(phi[i, j])) * Kh
        return A

    def forward_basis(self, coeffs, A):
        r"""Map basis coefficient fields ``c_j(y, x)`` (m/yr) to ``h(t, y, x)`` (m).

        ``coeffs`` has shape ``(J, ny, nx)``; ``A`` is the output of
        :meth:`build_basis_kernels`. Returns ``h`` of shape
        ``(n_t, ny, nx)``.
        """
        coeffs = np.asarray(coeffs, dtype=np.float64)
        J = A.shape[1]
        if coeffs.shape != (J, self.ny, self.nx):
            raise ValueError(
                f"coeffs shape {coeffs.shape}, expected ({J}, {self.ny}, {self.nx})"
            )
        c_hat = xp.fft.fft2(asarray(coeffs / SECONDS_PER_YEAR), axes=(-2, -1))
        h_out = np.empty((self.n_t, self.ny, self.nx), dtype=np.float64)
        for n in range(self.n_t):
            h_hat = xp.zeros((self.ny, self.nx), dtype=xp.complex128)
            for j in range(J):
                h_hat = h_hat + A[n, j] * c_hat[j]
            h_out[n] = to_numpy(xp.fft.ifft2(h_hat).real)
        return h_out

    def adjoint_basis(self, h, A):
        r"""Adjoint of :meth:`forward_basis`: ``h(t,y,x)`` (m) → ``c_j(y,x)`` (m/yr).

        The forward writes ``ĥ(t_n) = Σ_j A[n,j] ĉ_j`` so the adjoint is
        ``ĉ_j^adj = Σ_n conj(A[n,j]) ĥ(t_n)``. Satisfies
        ``⟨G c, h⟩ = ⟨c, G† h⟩`` (checked in the sanity test).
        """
        h = np.asarray(h, dtype=np.float64)
        J = A.shape[1]
        if h.shape != (self.n_t, self.ny, self.nx):
            raise ValueError(
                f"h shape {h.shape}, expected ({self.n_t}, {self.ny}, {self.nx})"
            )
        h_hat = xp.fft.fft2(asarray(h), axes=(-2, -1))  # (n_t, ny, nx)
        c_adj = np.empty((J, self.ny, self.nx), dtype=np.float64)
        for j in range(J):
            acc = xp.zeros((self.ny, self.nx), dtype=xp.complex128)
            for n in range(self.n_t):
                acc = acc + xp.conj(A[n, j]) * h_hat[n]
            c_adj[j] = to_numpy(xp.fft.ifft2(acc).real) / SECONDS_PER_YEAR
        return c_adj


# ---------------------------------------------------------------------------
# Variable-H linearization
# ---------------------------------------------------------------------------


class LinearizedHForwardOp:
    r"""Stationary forward operator with first-order linearization in H(x,y).

    The Stubblefield kernel ``K_cum[n](k) ≡ Σ_{i ≤ n} w_i K_h(k, t_n - t_i)``
    is smooth in the reference thickness ``H``. For a basin where ``H``
    varies as ``H(x,y) = H_ref + ΔH(x,y)`` with ``|ΔH| / H_ref`` modest
    (~20% on Beardmore), Taylor-expand around the spatial mean

    .. math::
        K_{\text{cum}}[n]\bigl(\mathbf{k}, H(\mathbf{x})\bigr)
            \approx K_0[n](\mathbf{k}) + \Delta H(\mathbf{x})\,
                    K_1[n](\mathbf{k}),

    where :math:`K_0 = K_{\text{cum}}|_{H_{\text{ref}}}` and
    :math:`K_1 = \partial K_{\text{cum}} / \partial H |_{H_{\text{ref}}}`.
    The forward operator becomes

    .. math::
        h_n(\mathbf{x}) = \mathcal{F}^{-1}[K_0[n]\,\hat m]
                        + \Delta H(\mathbf{x})\;
                          \mathcal{F}^{-1}[K_1[n]\,\hat m].

    By a one-line inner-product identity the adjoint is

    .. math::
        m_{\text{adj}} = \sum_n \mathcal{F}^{-1}[K_0^*[n]\,\hat h_n]
                       + \sum_n \mathcal{F}^{-1}[K_1^*[n]\,
                                 \mathcal{F}(\Delta H \cdot h_n)],

    self-adjoint by construction. Plugs into :func:`cg_invert_stationary`
    by exposing the same ``forward_stationary`` / ``adjoint_stationary``
    / ``_kernel_for_reference`` API as :class:`PerturbationForwardOp`.

    ``K_1`` is computed numerically by centered finite difference: build
    two :class:`PerturbationForwardOp` instances at
    ``H_ref·(1 ± eps)`` and take ``(K_+ − K_−) / (2·eps·H_ref)``. ``γ``
    is shifted along with ``H`` so that the FD captures both the
    ``k' = kH`` rescaling and the ``γ ∝ 1/H`` dependence in
    ``λ_±``.

    Parameters
    ----------
    H_field : ndarray, shape (ny, nx)
        Reference ice thickness per pixel, meters. Nan-cells are
        replaced by the spatial mean over finite cells.
    H_ref : float, optional
        Anchor for the Taylor expansion. Defaults to the spatial mean
        of ``H_field`` over finite cells.
    eps : float, default 0.05
        Relative finite-difference step (so the FD probes
        ``H_ref·(1±eps)``).
    div_u_mean : float, optional
        Spatial-mean divergence of velocity (1/s) used to scale
        ``γ = ⟨∇·u⟩·t_r(H)`` consistently with the H-shifted FD ops.
        If None, ``γ`` is held at the value passed via ``gamma`` for
        all three ops (constant-γ approximation).
    Other parameters mirror :class:`PerturbationForwardOp`.
    """

    def __init__(
        self,
        H_field,
        times,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        H_ref: float | None = None,
        eta_bar: float = 1e14,
        rho_i: float = rhoi,
        rho_w: float = rhow,
        g: float = G_GRAVITY,
        alpha: float = 0.0,
        alpha_y: float = 0.0,
        gamma: float = 0.0,
        theta: float = 1e-14,
        eps: float = 0.05,
        div_u_mean: float | None = None,
    ):
        H_arr = np.asarray(H_field, dtype=np.float64)
        if H_arr.shape != (ny, nx):
            raise ValueError(
                f"H_field has shape {H_arr.shape}, expected ({ny}, {nx})"
            )
        finite = np.isfinite(H_arr)
        if not finite.any():
            raise ValueError("H_field has no finite cells")
        H_mean = float(H_arr[finite].mean())
        if H_ref is None:
            H_ref = H_mean
        self.H_ref = float(H_ref)
        # Replace NaNs with H_ref so ΔH is well-defined everywhere.
        H_filled = np.where(finite, H_arr, self.H_ref)
        self.dH = asarray(H_filled - self.H_ref)  # (ny, nx) on backend

        self.eps = float(eps)
        self.eps_H = self.eps * self.H_ref

        # γ at H_ref±eps. If div_u_mean is provided, recompute γ at each
        # FD step so the FD captures the t_r(H)-driven γ shift.
        def _gamma_at(H_value: float) -> float:
            if div_u_mean is None:
                return float(gamma)
            tr_local = 2.0 * eta_bar / (rho_i * g * H_value)
            return float(div_u_mean) * tr_local

        gamma_0 = float(gamma) if div_u_mean is None else _gamma_at(self.H_ref)
        gamma_p = _gamma_at(self.H_ref * (1.0 + self.eps))
        gamma_m = _gamma_at(self.H_ref * (1.0 - self.eps))

        common_kw = dict(
            times=times, nx=nx, ny=ny, dx=dx, dy=dy,
            eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
            alpha=alpha, alpha_y=alpha_y, theta=theta,
        )
        self.op_0 = PerturbationForwardOp(H=self.H_ref, gamma=gamma_0, **common_kw)
        self.op_plus = PerturbationForwardOp(
            H=self.H_ref * (1.0 + self.eps), gamma=gamma_p, **common_kw,
        )
        self.op_minus = PerturbationForwardOp(
            H=self.H_ref * (1.0 - self.eps), gamma=gamma_m, **common_kw,
        )

        self.tr = self.op_0.tr
        self.n_t = self.op_0.n_t
        self.times = self.op_0.times
        self.nx = nx
        self.ny = ny
        self.dx = float(dx)
        self.dy = float(dy)
        # Spectral grid is invariant in (nx, ny, dx, dy); expose so the
        # Sobolev-H¹ smoothness term in cg_invert_stationary works for v3.
        self.kx = self.op_0.kx
        self.ky = self.op_0.ky

    def _K_pair_for_reference(self, reference: str):
        K0 = self.op_0._kernel_for_reference(reference)
        K_plus = self.op_plus._kernel_for_reference(reference)
        K_minus = self.op_minus._kernel_for_reference(reference)
        K1 = (K_plus - K_minus) / (2.0 * self.eps_H)
        return K0, K1

    def _kernel_for_reference(self, reference: str):
        # cg_invert_stationary uses this only for Tikhonov scaling. Returning
        # K0 (the central kernel) gives a well-defined per-wavenumber RMS
        # without inflating the scale by the small ΔH-correction term.
        return self.op_0._kernel_for_reference(reference)

    def forward_stationary(self, m_2d, reference: str = "time_mean"):
        m_2d = np.asarray(m_2d, dtype=np.float64)
        if m_2d.shape != (self.ny, self.nx):
            raise ValueError(
                f"m_2d has shape {m_2d.shape}, expected ({self.ny}, {self.nx})"
            )
        m_si = asarray(m_2d / SECONDS_PER_YEAR)
        m_hat = xp.fft.fft2(m_si)
        K0, K1 = self._K_pair_for_reference(reference)
        h_out = np.empty((self.n_t, self.ny, self.nx), dtype=np.float64)
        for n in range(self.n_t):
            term0 = xp.fft.ifft2(K0[n] * m_hat).real
            term1 = self.dH * xp.fft.ifft2(K1[n] * m_hat).real
            h_out[n] = to_numpy(term0 + term1)
        return h_out

    def adjoint_stationary(self, h, reference: str = "time_mean"):
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (self.n_t, self.ny, self.nx):
            raise ValueError(
                f"h has shape {h.shape}, expected ({self.n_t}, {self.ny}, {self.nx})"
            )
        K0, K1 = self._K_pair_for_reference(reference)
        m_adj_hat = xp.zeros((self.ny, self.nx), dtype=xp.complex128)
        for n in range(self.n_t):
            h_n = asarray(h[n])
            h_hat_n = xp.fft.fft2(h_n)
            # Op_0 adjoint: <Op_0(m), h_n> = <m, conj(K0[n]) FFT(h_n)>
            m_adj_hat = m_adj_hat + xp.conj(K0[n]) * h_hat_n
            # Op_1 adjoint with ΔH multiplication: <ΔH·Op_1(m), h_n>
            #     = <Op_1(m), ΔH·h_n>
            #     = <m, conj(K1[n]) FFT(ΔH·h_n)>
            dH_h = self.dH * h_n
            m_adj_hat = m_adj_hat + xp.conj(K1[n]) * xp.fft.fft2(dH_h)
        m_adj_si = xp.fft.ifft2(m_adj_hat).real
        return to_numpy(m_adj_si) / SECONDS_PER_YEAR


# ---------------------------------------------------------------------------
# CG inverter
# ---------------------------------------------------------------------------


def cg_invert(
    op: PerturbationForwardOp,
    h_obs: np.ndarray,
    weight: np.ndarray | None = None,
    tikhonov: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    r"""Solve ``min_m  ‖W (G m - h_obs)‖² + λ² ‖m‖²`` by CG on the normal
    equations.

    Parameters
    ----------
    op : PerturbationForwardOp
    h_obs : ndarray, shape ``(n_t, ny, nx)``
        Observed surface elevation perturbation, meters. NaN entries
        are masked (treated as missing data; their residual is set to
        zero before the adjoint and they contribute neither to the
        gradient nor the curvature).
    weight : ndarray or None
        Optional per-pixel/per-time weights ``w(t, y, x)`` (e.g.
        ``1 / sigma`` from the geodiff jitter sidecar). If None, a
        uniform weight of 1 is used over finite cells.
    tikhonov : float
        ``λ`` in ``J(m) = ‖W(Gm - h)‖² + λ² ‖m‖²``. Scaled relative
        to the per-wavenumber RMS of |G|; pass ``λ ≈ 1e-3 .. 1e-1``.
    max_iter, tol : iteration cap and convergence threshold for
        ``‖r_k‖ / ‖r_0‖``.
    verbose : if True, print per-iteration residual norm.

    Returns
    -------
    m_recovered : ndarray, shape ``(n_t, ny, nx)``, m/yr.
    info : dict with keys ``n_iter``, ``residual_norm``,
        ``residual_norm_initial``, ``converged``.
    """
    h_obs = np.asarray(h_obs, dtype=np.float64)
    if h_obs.shape != (op.n_t, op.ny, op.nx):
        raise ValueError(f"h_obs shape mismatch: got {h_obs.shape}")

    finite_mask = np.isfinite(h_obs)
    h_safe = np.where(finite_mask, h_obs, 0.0)
    if weight is None:
        W = finite_mask.astype(np.float64)
    else:
        W = np.where(finite_mask, np.asarray(weight, dtype=np.float64), 0.0)

    # Tikhonov scaled by per-wavenumber kernel RMS so `tikhonov` stays
    # dimensionless and dataset-independent. Estimate via the magnitude
    # of ``Σ_n w_n K(t_n - t_0)`` averaged over k --- the steady-state
    # build-up at the longest baseline.
    K0 = op._Kh(op.t_secs[-1] - op.t_secs[0])
    a_rms = float(to_numpy(xp.sqrt(xp.mean(xp.abs(K0) ** 2))))
    lam = tikhonov * a_rms
    lam2 = lam * lam

    def Aop(m_vec):
        m = m_vec.reshape(op.n_t, op.ny, op.nx)
        Gm = op.forward(m)
        WGm = W * Gm
        GtWGm = op.adjoint(W * WGm)
        return (GtWGm + lam2 * m).reshape(-1)

    rhs = op.adjoint(W * (W * h_safe)).reshape(-1)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0:
        return np.zeros((op.n_t, op.ny, op.nx)), dict(
            n_iter=0, residual_norm=0.0, residual_norm_initial=0.0, converged=True,
        )

    # CG on (G^T W^2 G + λ² I) m = G^T W^2 h_obs.
    m = np.zeros_like(rhs)
    r = rhs - Aop(m)
    p = r.copy()
    rs_old = float(r @ r)
    rs0 = rs_old
    info = {
        "n_iter": 0,
        "residual_norm": float(np.sqrt(rs_old)),
        "residual_norm_initial": float(np.sqrt(rs0)),
        "converged": False,
    }
    for k in range(max_iter):
        Ap = Aop(p)
        alpha = rs_old / float(p @ Ap)
        m += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        info["n_iter"] = k + 1
        info["residual_norm"] = float(np.sqrt(rs_new))
        if verbose:
            print(f"  CG iter {k + 1}: ‖r‖={info['residual_norm']:.3e}")
        if rs_new < (tol * tol) * rs0:
            info["converged"] = True
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return m.reshape(op.n_t, op.ny, op.nx), info


def cg_invert_stationary(
    op: PerturbationForwardOp,
    h_obs: np.ndarray,
    weight: np.ndarray | None = None,
    tikhonov: float = 1e-3,
    length_scale_m: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    reference: str = "anchor_first",
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    r"""Solve for a stationary :math:`m(x, y)` by CG-LSQ on the normal equations.

    Minimises

    .. math::
        J(m) = \sum_t \big\| W_t \cdot (G_{\text{stat}}(m)_t - h_t) \big\|^2
               + \lambda^2 \|m\|^2,

    where :math:`G_{\text{stat}}` is :meth:`PerturbationForwardOp.forward_stationary`
    (a single 2-D unknown drives the time-dependent kernel response). NaN
    cells in ``h_obs`` are masked out by ``W`` (zero weight) so they
    contribute neither to the gradient nor the curvature --- that is the
    LSQ-with-data convention requested for the strip-coverage-gappy
    Beardmore / Nansen stacks.

    Parameters
    ----------
    op : PerturbationForwardOp
    h_obs : ndarray, shape ``(n_t, ny, nx)``
        Observed surface elevation perturbation, meters. NaN entries
        are treated as missing data; they do **not** get filled with zero.
    weight : ndarray, optional
        Optional per-pixel/per-time weight ``w(t, y, x)`` (e.g.
        ``1 / sigma`` from the geodiff jitter sidecar). NaN cells in
        ``h_obs`` get zero weight regardless.
    tikhonov : float
        Dimensionless Tikhonov scale; multiplied by the per-wavenumber
        RMS of the longest-time cumulative kernel before squaring.
    length_scale_m : float, default 0.0
        Sobolev-H¹ smoothness length, meters. ``0`` reverts to pure
        Tikhonov ``λ²‖m‖²``; ``L > 0`` adds a high-k penalty
        ``λ² L² ‖∇m‖²`` so features finer than ``L`` are damped while
        channel- and basin-scale structure passes through.
    max_iter, tol : iteration cap and convergence threshold for
        ``‖r_k‖ / ‖r_0‖``.
    verbose : per-iteration residual norms.

    Returns
    -------
    m_recovered : ndarray, shape ``(ny, nx)``, m/yr.
    info : dict with keys ``n_iter``, ``residual_norm``,
        ``residual_norm_initial``, ``converged``.
    """
    h_obs = np.asarray(h_obs, dtype=np.float64)
    if h_obs.shape != (op.n_t, op.ny, op.nx):
        raise ValueError(f"h_obs shape mismatch: got {h_obs.shape}")

    finite_mask = np.isfinite(h_obs)
    h_safe = np.where(finite_mask, h_obs, 0.0)
    if weight is None:
        W = finite_mask.astype(np.float64)
    else:
        W = np.where(finite_mask, np.asarray(weight, dtype=np.float64), 0.0)

    K = op._kernel_for_reference(reference)
    # Scale Tikhonov by the per-wavenumber RMS of the *time-summed* kernel
    # |sum_n K[n]|: that is the diagonal of G_stat^T G_stat in spectral
    # space (modulo the W mask). The kernel K has SI units of seconds
    # because it folds in dt_weights; forward_stationary internally
    # converts m m/yr -> m/s and adjoint_stationary converts back, so the
    # effective scaling of G†G acting on m_2d (m/yr) carries an
    # implicit /SECONDS_PER_YEAR² factor. We pre-divide by SECONDS_PER_YEAR
    # so the resulting `lam²` matches G†G's actual numerical scale and
    # `tikhonov` stays a dimensionless data-vs-regularization knob.
    K_sum_sq = float(
        to_numpy(xp.mean(xp.sum(xp.abs(K) ** 2, axis=0)))
    )
    lam = tikhonov * np.sqrt(max(K_sum_sq, 1e-30)) / SECONDS_PER_YEAR
    lam2 = lam * lam

    # Sobolev-H¹ prior: penalize ‖(I + L²·(-∇²))^(1/2) m‖² = ‖m‖² + L²‖∇m‖².
    # Spectral multiplier S(k) = 1 + L²·(kx² + ky²) is real, positive, and
    # self-adjoint, so the modified normal operator stays SPD for CG. The
    # term scales like k² at high k, so it preferentially damps the small-
    # scale ringing modes the kernel can't constrain while leaving channel-
    # and basin-scale signal alone (Beardmore basal channels ≳ a few × L).
    if length_scale_m and length_scale_m > 0.0:
        L2 = float(length_scale_m) ** 2
        S_smooth = 1.0 + L2 * (op.kx ** 2 + op.ky ** 2)
    else:
        S_smooth = None

    def Aop(m_vec):
        m_2d = m_vec.reshape(op.ny, op.nx)
        Gm = op.forward_stationary(m_2d, reference=reference)
        WGm = W * Gm
        GtWGm = op.adjoint_stationary(W * WGm, reference=reference)
        if S_smooth is None:
            reg_term = lam2 * m_2d
        else:
            m_hat = xp.fft.fft2(asarray(m_2d))
            reg_hat = S_smooth * m_hat
            reg_term = lam2 * to_numpy(xp.fft.ifft2(reg_hat).real)
        return (GtWGm + reg_term).reshape(-1)

    rhs = op.adjoint_stationary(
        W * (W * h_safe), reference=reference
    ).reshape(-1)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0:
        return np.zeros((op.ny, op.nx)), dict(
            n_iter=0, residual_norm=0.0, residual_norm_initial=0.0, converged=True,
        )

    m = np.zeros_like(rhs)
    r = rhs - Aop(m)
    p = r.copy()
    rs_old = float(r @ r)
    rs0 = rs_old
    info = {
        "n_iter": 0,
        "residual_norm": float(np.sqrt(rs_old)),
        "residual_norm_initial": float(np.sqrt(rs0)),
        "converged": False,
    }
    for k in range(max_iter):
        Ap = Aop(p)
        alpha = rs_old / float(p @ Ap)
        m += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        info["n_iter"] = k + 1
        info["residual_norm"] = float(np.sqrt(rs_new))
        if verbose:
            print(f"  CG-stat iter {k + 1}: ‖r‖={info['residual_norm']:.3e}")
        if rs_new < (tol * tol) * rs0:
            info["converged"] = True
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return m.reshape(op.ny, op.nx), info


# ---------------------------------------------------------------------------
# Reduced temporal-basis inverse
# ---------------------------------------------------------------------------


def make_temporal_basis(t_secs, kind: str = "trend"):
    r"""Return ``(phi, names)`` for the reduced temporal basis.

    ``phi`` is a real ``(n_t, J)`` array; time is normalised to the
    observation window so the recovered coefficients come out in m/yr and
    are directly interpretable.

    ==================  ==================================  ===  ==============================
    ``kind``            :math:`\{\varphi_j\}`               J    coefficient meaning
    ==================  ==================================  ===  ==============================
    ``"const"``         :math:`\{1\}`                       1    ≡ stationary melt
    ``"trend"``         :math:`\{1,\,(t-\bar t)/T\}`        2    ``c1`` = Δmelt across window
    ``"trend+annual"``  ``+ \{\sin,\cos\}(2\pi t/1\,yr)``   4    ``c2,c3`` = seasonal amplitude
    ==================  ==================================  ===  ==============================

    ``T`` is the full window span ``t[-1]-t[0]``, so ``φ_trend`` runs over
    ``[-0.5, 0.5]`` and ``c1`` is the melt change from the first to the
    last epoch.
    """
    t = np.asarray(t_secs, dtype=np.float64)
    n = len(t)
    if n < 2:
        raise ValueError("temporal basis needs at least two epochs")
    t0 = float(t[0])
    span = max(float(t[-1] - t[0]), 1e-30)
    tbar = float(t.mean())

    cols = [np.ones(n, dtype=np.float64)]
    names = ["const"]
    if kind == "const":
        pass
    elif kind == "trend":
        cols.append((t - tbar) / span)
        names.append("trend")
    elif kind == "trend+annual":
        cols.append((t - tbar) / span)
        names.append("trend")
        w = 2.0 * np.pi / SECONDS_PER_YEAR
        cols.append(np.sin(w * (t - t0)))
        names.append("sin_annual")
        cols.append(np.cos(w * (t - t0)))
        names.append("cos_annual")
    else:
        raise ValueError(
            f"unknown temporal basis {kind!r}; use 'const', 'trend', or 'trend+annual'"
        )
    phi = np.stack(cols, axis=1)
    return phi, names


def _cumulative_integral(col, t_secs):
    r"""Cumulative trapezoidal integral ``Φ(t_i) = ∫_0^{t_i} f dτ`` of a 1-D signal.

    ``col`` and ``t_secs`` are 1-D arrays of equal length; returns an array of
    the same length with ``Φ(t_0) = 0``. Used to map a temporal basis
    ``φ_j(t)`` to its antiderivative for the mass-balance DC fit.
    """
    col = np.asarray(col, dtype=np.float64)
    t = np.asarray(t_secs, dtype=np.float64)
    out = np.zeros_like(col)
    if len(col) > 1:
        incr = 0.5 * (col[1:] + col[:-1]) * (t[1:] - t[:-1])
        out[1:] = np.cumsum(incr)
    return out


def cg_invert_basis(
    op: PerturbationForwardOp,
    h_obs: np.ndarray,
    phi: np.ndarray,
    weight: np.ndarray | None = None,
    tikhonov: float = 1e-2,
    length_scale_m: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    r"""Solve for reduced-basis coefficients ``c_j(y, x)`` by CG-LSQ.

    Minimises

    .. math::
        J(c) = \sum_t \big\| W_t \cdot (G_{\text{basis}}(c)_t - h_t) \big\|^2
               + \lambda^2 \| S^{1/2} c \|^2,

    where :math:`G_{\text{basis}}` is :meth:`PerturbationForwardOp.forward_basis`
    (``m(x,y,t) = Σ_j c_j(x,y) φ_j(t)`` driving the time-dependent kernel),
    ``W = isfinite(h_obs)`` masks coverage gaps (**NaN is masked, never
    zero-filled** --- unlike the two frame wrappers today), and ``S`` is the
    optional spatial Sobolev-H¹ multiplier ``1 + L²|k|²``.

    This is :func:`cg_invert_stationary` carrying a mode index ``j``; at
    ``J = 1`` (``phi = ones``) it reduces to it exactly.

    Parameters
    ----------
    op : PerturbationForwardOp
    h_obs : ndarray ``(n_t, ny, nx)``
        Surface elevation anomaly, meters. NaN = missing (masked).
    phi : ndarray ``(n_t, J)``
        Temporal basis sampled at the epochs (see :func:`make_temporal_basis`).
    weight : ndarray ``(n_t, ny, nx)``, optional
        Per-sample weight; NaN cells in ``h_obs`` get zero weight regardless.
    tikhonov : float
        Dimensionless Tikhonov scale (× per-mode kernel RMS before squaring).
    length_scale_m : float
        Sobolev-H¹ smoothness length (m); ``0`` = pure Tikhonov.
    max_iter, tol, verbose : CG controls.

    Returns
    -------
    coeffs : ndarray ``(J, ny, nx)``, m/yr.
    info : dict with ``n_iter``, ``residual_norm``,
        ``residual_norm_initial``, ``converged``.
    """
    h_obs = np.asarray(h_obs, dtype=np.float64)
    if h_obs.shape != (op.n_t, op.ny, op.nx):
        raise ValueError(f"h_obs shape mismatch: got {h_obs.shape}")

    A = op.build_basis_kernels(phi)
    J = A.shape[1]

    finite_mask = np.isfinite(h_obs)
    h_safe = np.where(finite_mask, h_obs, 0.0)
    if weight is None:
        W = finite_mask.astype(np.float64)
    else:
        W = np.where(finite_mask, np.asarray(weight, dtype=np.float64), 0.0)

    # Tikhonov scaled by the per-mode RMS of the time-summed kernel
    # |Σ_n A[n,j]|², matching cg_invert_stationary's convention (the
    # kernel carries dt_weights in seconds; forward_basis converts
    # m/yr → m/s, so pre-dividing by SECONDS_PER_YEAR puts lam² on the
    # same numerical scale as G†G acting on m/yr coefficients).
    K_sum_sq = float(to_numpy(xp.mean(xp.sum(xp.abs(A) ** 2, axis=0))))
    lam = tikhonov * np.sqrt(max(K_sum_sq, 1e-30)) / SECONDS_PER_YEAR
    lam2 = lam * lam

    if length_scale_m and length_scale_m > 0.0:
        L2 = float(length_scale_m) ** 2
        S_smooth = 1.0 + L2 * (op.kx ** 2 + op.ky ** 2)
    else:
        S_smooth = None

    def Aop(c_vec):
        c = c_vec.reshape(J, op.ny, op.nx)
        Gm = op.forward_basis(c, A)
        WGm = W * Gm
        GtWGm = op.adjoint_basis(W * WGm, A)
        if S_smooth is None:
            reg_term = lam2 * c
        else:
            reg_term = np.empty_like(c)
            for j in range(J):
                ch = xp.fft.fft2(asarray(c[j]))
                reg_term[j] = lam2 * to_numpy(xp.fft.ifft2(S_smooth * ch).real)
        return (GtWGm + reg_term).reshape(-1)

    rhs = op.adjoint_basis(W * (W * h_safe), A).reshape(-1)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0:
        return np.zeros((J, op.ny, op.nx)), dict(
            n_iter=0, residual_norm=0.0, residual_norm_initial=0.0, converged=True,
        )

    c = np.zeros_like(rhs)
    r = rhs - Aop(c)
    p = r.copy()
    rs_old = float(r @ r)
    rs0 = rs_old
    info = {
        "n_iter": 0,
        "residual_norm": float(np.sqrt(rs_old)),
        "residual_norm_initial": float(np.sqrt(rs0)),
        "converged": False,
    }
    for k in range(max_iter):
        Ap = Aop(p)
        alpha = rs_old / float(p @ Ap)
        c += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        info["n_iter"] = k + 1
        info["residual_norm"] = float(np.sqrt(rs_new))
        if verbose:
            print(f"  CG-basis iter {k + 1}: ‖r‖={info['residual_norm']:.3e}")
        if rs_new < (tol * tol) * rs0:
            info["converged"] = True
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return c.reshape(J, op.ny, op.nx), info


def inverse_time_varying(
    h_stack: xr.DataArray,
    H: float,
    *,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    alpha: float = 0.0,
    alpha_y: float = 0.0,
    gamma: float = 0.0,
    theta: float = 1e-14,
    temporal_basis: str = "trend",
    reg: float = 1e-2,
    length_scale_m: float = 0.0,
    recover_dc: bool = True,
    a_dot_dc: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> xr.DataArray:
    r"""Invert a surface-elevation stack for a *time-varying* basal melt rate.

    The frame-agnostic public entry point for the reduced-temporal-basis
    Stubblefield inverse (design note ``literature/plan_timevarying_inverse.md``
    §3.1/§3.5). Parameterises the unknown as
    :math:`m(x, y, t) = \sum_j c_j(x, y)\,\varphi_j(t)` with a small basis
    ``J ≪ n_t`` (see :func:`make_temporal_basis`), solves the masked space-time
    normal equations for the coefficient fields by CG (:func:`cg_invert_basis`),
    and reconstructs ``m(t, y, x)``.

    Relationship to the other inverses:

    - ``temporal_basis="const"`` (``J=1``) reproduces
      :func:`stereo_melt.dynamics.inverse_stationary` (masked-CG variant): a
      single time-invariant field.
    - ``temporal_basis="trend"`` / ``"trend+annual"`` recover a melt *trend* /
      *seasonal cycle* — the capability the stationary inverses cannot express.

    Caller supplies the background ``(alpha, alpha_y, gamma)`` — pass ``0`` in a
    Lagrangian comoving frame, or the velocity-derived values in the Eulerian
    frame (mirroring :func:`pseudospectral_eulerian_inverse`).

    Coverage handling: NaN cells in ``h_stack`` are treated as *missing* and
    masked out of the fit (never zero-filled). The anomaly is taken relative to
    the first epoch (``h - h[0]``), matching the causal-kernel convention
    ``h(t=0)=0``.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Surface-elevation stack, meters. NaN = missing (masked).
    H : float
        Reference ice thickness, meters.
    eta_bar, rho_i, rho_w, g, alpha, alpha_y, gamma, theta : float
        Physics/background, as in :class:`PerturbationForwardOp`.
    temporal_basis : {"const", "trend", "trend+annual"}
        Temporal parameterisation of ``m`` (:func:`make_temporal_basis`).
    reg : float
        Dimensionless Tikhonov scale for the coefficient CG.
    length_scale_m : float
        Spatial Sobolev-H¹ smoothness length (m). Strongly recommended on real
        (noisy) data — it tames the deconvolution noise amplification that the
        kernel's high-``k`` damping would otherwise inflate; ``≈ 4·H`` is a
        good starting point (design note §3.4).
    recover_dc : bool
        If True (default), splice the spatial-mean (``k=0``) melt back in per
        temporal mode (design note §3.3). The Stubblefield kernel zeros ``k=0``
        in *both* directions, so the spectral inverse is blind to the
        spatial-mean melt and to the basin-mean surface change it drives; the
        mean is recovered instead from the observed basin-mean anomaly time
        series ``\bar h(t)`` through the hydrostatic mass balance
        ``\overline{m}(t) = \dot a - R\,d\bar h/dt`` (``R = ρ_w/(ρ_w-ρ_i)``),
        projected onto the temporal basis by least squares. **Only valid when
        bulk advection through the tile boundary is small** — for a
        strongly-advective Eulerian tile prefer the Eul/Lag solvers for the
        mean and set ``recover_dc=False``.
    a_dot_dc : float
        Spatial-mean SMB rate (m ice / yr) folded into the DC mass balance.
    max_iter, tol, verbose : CG controls.

    Returns
    -------
    xarray.DataArray, dims ``(time, y, x)``
        Recovered basal melt rate, m ice yr\ :sup:`-1`, **Shean convention:
        negative = melt, positive = accretion**.

    Notes
    -----
    The AC (spatial-anomaly) structure is the physically-constrained part; with
    ``recover_dc=True`` the per-epoch spatial-mean level is spliced from mass
    balance. Even so, treat this as a time-resolved non-hydrostatic
    *correction / diagnostic* whose amplitude scales with the assumed viscosity
    (``∝ 1/η̄``), not a mass-budget melt product interchangeable with the
    Eulerian / Lagrangian solvers.
    """
    if "time" not in h_stack.dims:
        raise ValueError("inverse_time_varying requires a 'time' dim on h_stack")
    if h_stack.sizes["time"] < 2:
        raise ValueError("need at least two epochs for the time-varying inverse")

    x_coords = h_stack["x"].values
    y_coords = h_stack["y"].values
    ny = h_stack.sizes["y"]
    nx = h_stack.sizes["x"]
    dx = float(abs(x_coords[1] - x_coords[0]))
    dy = float(abs(y_coords[1] - y_coords[0]))
    times = h_stack["time"].values

    op = PerturbationForwardOp(
        H=H, times=times, nx=nx, ny=ny, dx=dx, dy=dy,
        eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
        alpha=alpha, alpha_y=alpha_y, gamma=gamma, theta=theta,
    )
    phi, names = make_temporal_basis(op.t_secs, temporal_basis)

    # Anomaly vs first epoch; KEEP NaN so cg_invert_basis's W masks coverage
    # gaps (do NOT fillna(0) — that would defeat the mask, design note G1).
    h_anom = (h_stack - h_stack.isel(time=0)).values.astype(np.float64)

    coeffs, info = cg_invert_basis(
        op, h_anom, phi, tikhonov=reg, length_scale_m=length_scale_m,
        max_iter=max_iter, tol=tol, verbose=verbose,
    )
    J = coeffs.shape[0]

    # Per-mode DC splice (design note §3.3). The kernel zeros k=0 in both
    # directions, so each coefficient field's spatial mean ⟨c_j⟩ is
    # unconstrained by the spectral inverse. Recover it from the observed
    # basin-mean anomaly h̄(t) via the hydrostatic mass balance
    #     m̄(t) = ȧ - R·dh̄/dt,   R = ρ_w/(ρ_w-ρ_i),
    # with m̄(t) = Σ_j ⟨c_j⟩ φ_j(t). Integrating (h̄(0)=0):
    #     R·h̄(t) - ȧ·t = -Σ_j ⟨c_j⟩ Φ_j(t),   Φ_j = ∫_0^t φ_j dτ,
    # a linear least-squares system for ⟨c_j⟩ (internal positive=melt units).
    dc_ok = 0
    if recover_dc:
        h_bar = np.array(
            [float(np.nanmean(h_anom[i])) if np.isfinite(h_anom[i]).any() else np.nan
             for i in range(op.n_t)],
            dtype=np.float64,
        )
        valid = np.isfinite(h_bar)
        if int(valid.sum()) > J:
            R_hydro = rho_w / (rho_w - rho_i)
            a_dot_si = float(a_dot_dc) / SECONDS_PER_YEAR
            Phi = np.stack(
                [_cumulative_integral(phi[:, j], op.t_secs) for j in range(J)], axis=1
            )  # (n_t, J), seconds
            b = a_dot_si * op.t_secs - R_hydro * h_bar        # (n_t,), meters
            M = Phi[valid]
            cbar_si, *_ = np.linalg.lstsq(M, b[valid], rcond=None)  # m/s
            cbar_myr = cbar_si * SECONDS_PER_YEAR
            for j in range(J):
                coeffs[j] = coeffs[j] - float(coeffs[j].mean()) + float(cbar_myr[j])
            dc_ok = 1

    # Reconstruct m(t) = Σ_j c_j φ_j(t); negate to Shean (negative = melt).
    m = np.empty((op.n_t, ny, nx), dtype=np.float64)
    for i in range(op.n_t):
        acc = np.zeros((ny, nx), dtype=np.float64)
        for j in range(J):
            acc = acc + coeffs[j] * float(phi[i, j])
        m[i] = acc
    m_shean = -m

    return xr.DataArray(
        m_shean, dims=("time", "y", "x"),
        coords={"time": times, "y": y_coords, "x": x_coords},
        name="melt_rate",
        attrs={
            "equation": "Stubblefield 2023 reduced-basis time-varying linear inverse",
            "units": "m ice yr^-1; Shean convention: negative = melt, positive = accretion",
            "temporal_basis": temporal_basis,
            "basis_modes": ",".join(names),
            "H": H, "eta_bar": eta_bar, "alpha": alpha, "alpha_y": alpha_y,
            "gamma": gamma, "reg": reg, "length_scale_m": length_scale_m,
            "tr_yr": op.tr / SECONDS_PER_YEAR,
            "cg_iter": int(info["n_iter"]),
            "cg_converged": bool(info["converged"]),
            "dc_recovered": int(dc_ok),
            "a_dot_dc": float(a_dot_dc),
            "note": "time-resolved non-hydrostatic melt (AC constrained by kernel, DC "
                    "spliced from mass balance); amplitude ∝ 1/eta_bar; a correction/"
                    "diagnostic, not a mass-budget product",
        },
    )
