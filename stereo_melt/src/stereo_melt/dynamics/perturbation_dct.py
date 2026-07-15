r"""DCT-II variant of the Stubblefield linear-perturbation forward operator.

Drop-in replacement for :class:`stereo_melt.dynamics.pseudospectral.PerturbationForwardOp`
that diagonalises the spatial operator with a 2-D DCT-II instead of a 2-D FFT.

Why DCT instead of FFT
----------------------

The FFT path treats the analysis grid as periodic. For ice-shelf altimetry
stacks where the floating mask occupies a fraction of the FFT bounding box,
that periodic boundary couples the data envelope to its mirror across the
domain — a periodic-wrap leakage that drives the high-frequency ringing and
boundary blow-ups seen in the constant-:math:`\bar H` CG runs at small
Tikhonov.

The DCT-II implicitly extends the grid by reflection across both the left/
right and top/bottom edges, so the implicit boundary condition is

.. math::
    \frac{\partial f}{\partial x}\bigg|_{\text{boundary}} = 0,

which matches "no information past the data envelope" much better than
periodic wrap. Because the Stubblefield kernel :math:`K_h(\mathbf k, t)`
depends only on :math:`|\mathbf k|`, the operator is reflection-symmetric and
remains diagonal in the DCT basis. The DCT-equivalent wavenumber on a length-
:math:`N` axis with cell spacing :math:`\Delta x` is

.. math::
    k_j = \frac{j \pi}{N \Delta x}, \quad j = 0, 1, \dots, N-1.

That is the only structural change: the forward / adjoint methods are the
same time-convolution against ``K_h``, just transformed via DCT-II instead
of FFT2.

Composes with the existing :func:`stereo_melt.dynamics.pseudospectral.cg_invert`
and :func:`cg_invert_stationary` — those routines only see the abstract
forward / adjoint API.
"""

from __future__ import annotations

import numpy as np

from ..backend import asarray, to_numpy, xp
from .linear_perturbation import _dct_wavenumber_grids, _dctn, _idctn
from .pseudospectral import PerturbationForwardOp, SECONDS_PER_YEAR

__all__ = ["PerturbationForwardOpDCT"]


class PerturbationForwardOpDCT(PerturbationForwardOp):
    r"""DCT-II variant of :class:`PerturbationForwardOp`.

    Same constructor signature, same kernel machinery; the only structural
    differences are

    1. wavenumber grid: ``k_j = j π / (N Δ)`` instead of ``2π · fftfreq(N) / Δ``,
    2. spatial transform: ``dctn(., type=2, norm="ortho")`` instead of ``fft2``,
    3. complex kernel handled by taking ``Re(K m_dct)`` before the inverse
       transform — justified because the DCT-II basis is the FFT of the
       reflection-extended signal, which is real-valued, so any imaginary
       parts of :math:`K(|k|)` cancel by the same parity argument that keeps
       the FFT path's output real.

    Adjoint stays a self-adjoint multiplication (``Re(K)`` is its own
    transpose in DCT basis), so :func:`cg_invert_stationary` works without
    modification.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the FFT-based wavenumber grid with the DCT-II one and
        # rebuild the transfer functions on it.
        kx, ky = _dct_wavenumber_grids(self.nx, self.ny, self.dx, self.dy)
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
        # Stationary kernel caches built on the FFT grid by the parent are
        # invalid; drop them so they get rebuilt on first access.
        for attr in ("_K_cum_cached", "_K_cum_centered_cached"):
            if hasattr(self, attr):
                delattr(self, attr)

    # ------------------------------------------------------------------
    # Forward / adjoint (time-dependent)
    # ------------------------------------------------------------------
    def forward(self, m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (self.n_t, self.ny, self.nx):
            raise ValueError(
                f"m has shape {m.shape}, expected ({self.n_t}, {self.ny}, {self.nx})"
            )
        m_si = asarray(m / SECONDS_PER_YEAR)
        m_dct = _dctn(m_si)  # real-valued

        h_out = np.empty((self.n_t, self.ny, self.nx), dtype=np.float64)
        for n in range(self.n_t):
            h_dct = xp.zeros((self.ny, self.nx), dtype=xp.float64)
            for i in range(n + 1):
                K = self._Kh(self.t_secs[n] - self.t_secs[i])
                h_dct = h_dct + (self.dt_weights[i] * xp.real(K)) * m_dct[i]
            h_out[n] = to_numpy(_idctn(h_dct))
        return h_out

    def adjoint(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (self.n_t, self.ny, self.nx):
            raise ValueError(
                f"h has shape {h.shape}, expected ({self.n_t}, {self.ny}, {self.nx})"
            )
        h_dct = _dctn(asarray(h))  # (n_t, ny, nx), real

        m_adj_out = np.empty((self.n_t, self.ny, self.nx), dtype=np.float64)
        for i in range(self.n_t):
            m_dct_i = xp.zeros((self.ny, self.nx), dtype=xp.float64)
            wi = self.dt_weights[i]
            for n in range(i, self.n_t):
                K = self._Kh(self.t_secs[n] - self.t_secs[i])
                # Adjoint of multiplication by Re(K) is multiplication by Re(K).
                m_dct_i = m_dct_i + (wi * xp.real(K)) * h_dct[n]
            m_adj_si = _idctn(m_dct_i)
            m_adj_out[i] = to_numpy(m_adj_si) / SECONDS_PER_YEAR
        return m_adj_out

    # ------------------------------------------------------------------
    # Stationary forward / adjoint (single 2-D unknown m(x, y))
    # ------------------------------------------------------------------
    def forward_stationary(
        self, m_2d: np.ndarray, reference: str = "anchor_first"
    ) -> np.ndarray:
        m_2d = np.asarray(m_2d, dtype=np.float64)
        if m_2d.shape != (self.ny, self.nx):
            raise ValueError(
                f"m_2d has shape {m_2d.shape}, expected ({self.ny}, {self.nx})"
            )
        m_si = asarray(m_2d / SECONDS_PER_YEAR)
        m_dct = _dctn(m_si)
        K = self._kernel_for_reference(reference)  # (n_t, ny, nx) complex
        h_out = np.empty((self.n_t, self.ny, self.nx), dtype=np.float64)
        for n in range(self.n_t):
            h_out[n] = to_numpy(_idctn(xp.real(K[n]) * m_dct))
        return h_out

    def adjoint_stationary(
        self, h: np.ndarray, reference: str = "anchor_first"
    ) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (self.n_t, self.ny, self.nx):
            raise ValueError(
                f"h has shape {h.shape}, expected ({self.n_t}, {self.ny}, {self.nx})"
            )
        h_dct = _dctn(asarray(h))  # (n_t, ny, nx), real
        K = self._kernel_for_reference(reference)
        m_dct_adj = xp.zeros((self.ny, self.nx), dtype=xp.float64)
        for n in range(self.n_t):
            m_dct_adj = m_dct_adj + xp.real(K[n]) * h_dct[n]
        m_adj_si = _idctn(m_dct_adj)
        return to_numpy(m_adj_si) / SECONDS_PER_YEAR
