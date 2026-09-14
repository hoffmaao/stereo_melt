# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Differentiable Stubblefield forward operator and variational melt inverse.

The hydrostatic and budget inverses read the surface as freeboard-scaled
thickness, so their recovered melt inherits the ice shelf's bridging transfer
(see :mod:`stereo_melt.dynamics.bridging_restoration`). Two ways to undo that
transfer have a decisive difference:

* **Post-hoc deconvolution** (``bridging_inverse``) divides the *recovered*
  Eulerian melt by the transfer :math:`T`. This over-lifts pure across-flow
  ridges, because the Eulerian solver already recovers those through the
  advective term :math:`u\,\partial H/\partial x` — deconvolving with :math:`T`
  double-counts that dynamics (E2a ``cosy``: 1.07 without, 1.54 with).

* **Forward-fit** (this module) puts the transfer in the *forward* model and
  fits melt to the observed surface. The across-flow dynamics live inside the
  Green's function :math:`G_h` itself, so there is no double count and no
  angular weight is needed: the same fit recovers along-flow, oblique, and
  across-flow melt through one operator.

Forward operator
----------------

Given a basal melt-rate perturbation :math:`m(x, y)` in m/yr, the steady
linearized surface-elevation response (Stubblefield, Wearing and Meyer 2023;
:mod:`~stereo_melt.dynamics.linear_perturbation`) is a spectral multiply,

.. math::
    \widehat{\delta z_s}(\mathbf k) = M_h(\mathbf k)\,\hat m(\mathbf k),
    \qquad
    M_h = \frac{t_r\, G_h(\mathbf k)}{\text{SPY}},

where :math:`G_h` is the advected steady surface Green's function, :math:`t_r`
the viscous relaxation time, and SPY converts m/yr melt to meters of surface.
:func:`stubblefield_forward_multiplier` builds :math:`M_h` (pure numpy);
:class:`StubblefieldForward` wraps it as a fixed differentiable ``torch`` layer.
The field is symmetric mirror-padded to twice its size before the FFT so
non-periodic tile edges do not wrap through the operator.

Variational inverse
-------------------

:func:`variational_melt_inverse` fits melt to an observed surface anomaly
:math:`\delta z_s = z_s - z_s^{\rm control}` by minimizing

.. math::
    \| \mathcal F(m) - \delta z_s \|^2_{\rm mask} + \lambda \|m\|^2

over either a dense grid (:class:`GridMelt`) or a SIREN coordinate network
(:class:`SirenMelt`; an implicit neural representation, Sitzmann et al. 2020)
via Adam. The two representations agree to the optimizer noise floor
(E2a ``cosy``: grid 0.725, SIREN 0.723), so the recovery is
representation-agnostic.

Calibration
-----------

The operator carries the same two calibration knobs as the transfer:
``alpha_scale`` (advection weight; E1b along-flow twin favors
:math:`\bar\eta_{\rm eff} = \bar\eta/3`, ``alpha_scale = 0.34``) and
``eta_bar`` (viscosity, which sets the advection-free static response an
across-flow ridge sees, since :math:`\alpha k'_x = 0` there). On the E2a
``cosy`` twin the uncalibrated static kernel over-predicts the Elmer surface
response, so the forward-fit under-recovers to 0.725; the recovery is monotonic
in ``eta_bar`` and reaches 1.0 at :math:`\bar\eta_{\rm eff}\approx\bar\eta/1.67`
(``eta_bar = 6e13``). Both knobs are caller-supplied study parameters, not
library constants — the residual is a forward-model-accuracy problem, calibrated
per geometry against truth, not a code defect (the machinery self-test recovers
1.0; see ``tests/gate_stubblefield_forward.py``).

``torch`` note: like :mod:`stereo_melt.dynamics.streamline_pinn`, this module
imports ``torch`` at top level and is therefore **not** re-exported from
``stereo_melt.dynamics.__init__`` — import it explicitly to keep a bare
``import stereo_melt.dynamics`` free of the ``torch`` dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

# torch is imported *after* the numpy/xarray/scipy C-extensions so its bundled
# libstdc++ does not shadow theirs (see tests/gate_stubblefield_forward.py).
import torch  # noqa: E402

from ..constants import rhoi, rhow  # noqa: E402
from ..freeboard import freeboard_to_thickness  # noqa: E402
from .linear_perturbation import (  # noqa: E402
    G_GRAVITY,
    SECONDS_PER_YEAR,
    LinearPerturbation,
    _wavenumber_grids,
)

__all__ = [
    "stubblefield_forward_multiplier",
    "StubblefieldForward",
    "BlendedStubblefieldForward",
    "GridMelt",
    "SirenMelt",
    "variational_melt_inverse",
    "variational_melt_rate",
    "MeltInverseResult",
]


def _keep_band_np(a: np.ndarray, mask: np.ndarray, sigma_px: float,
                  short_lambda_px: float | None = None) -> np.ndarray:
    """Numpy replica of the fit's band projector (see the ``anchor_lp`` /
    ``anchor_short`` machinery in :func:`variational_melt_inverse`): the
    component of ``a`` in the KEPT band, mask-zero-filled, mirror-padded.
    Used so the reported fit diagnostics live in exactly the subspace the
    objective saw."""
    a0 = np.where(mask, a, 0.0)
    ny, nx = a0.shape
    py, px = ny // 2, nx // 2
    f = np.pad(a0, ((py, py), (px, px)), mode="reflect")
    ky = np.fft.fftfreq(f.shape[0])[:, None]
    kx = np.fft.rfftfreq(f.shape[1])[None, :]
    kk = np.hypot(ky, kx)

    def cos_lp(k_cut: float) -> np.ndarray:
        k_lo, k_hi = k_cut / 1.2, k_cut * 1.2
        t = np.clip((k_hi - kk) / (k_hi - k_lo), 0.0, 1.0)
        return np.where(kk <= k_lo, 1.0, np.where(kk >= k_hi, 0.0,
                        0.5 - 0.5 * np.cos(math.pi * t)))

    keep = 1.0 - cos_lp(math.sqrt(2.0 * math.log(2.0))
                        / (2.0 * math.pi * float(sigma_px)))
    if short_lambda_px:
        keep = keep * cos_lp(1.0 / float(short_lambda_px))
    out = np.fft.irfft2(np.fft.rfft2(f) * keep, s=f.shape)
    return out[py:py + ny, px:px + nx]


def _nan_gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing (smooth value / smooth mask)."""
    m = np.isfinite(a)
    a0 = np.where(m, a, 0.0)
    return gaussian_filter(a0, sigma) / np.maximum(
        gaussian_filter(m.astype(float), sigma), 1e-6)


def _poly_basis(
    ny: int, nx: int, degree: int, extra: np.ndarray | None = None
) -> np.ndarray:
    """Low-order polynomial background basis on a ``(ny, nx)`` grid.

    Columns are the monomials ``x^i y^j`` with ``i + j <= degree`` over
    coordinates normalized to ``[-1, 1]`` (so ``degree=0`` -> DC only,
    ``degree=1`` -> plane ``{1, x, y}``, ``degree=2`` -> quadratic, 6 columns).
    Returns an ``(ny*nx, p)`` array (row-major, matching ``.ravel()``). Optional
    ``extra`` regressors (e.g. a hydrostatic reference surface), shape
    ``(ny, nx)`` or ``(ny, nx, k)``, are appended with non-finite cells zeroed.
    """
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    xs = xx / max(nx - 1, 1) * 2.0 - 1.0
    ys = yy / max(ny - 1, 1) * 2.0 - 1.0
    cols = [(xs ** i) * (ys ** (total - i))
            for total in range(int(degree) + 1) for i in range(total + 1)]
    B = np.stack([c.ravel() for c in cols], axis=1)
    if extra is not None:
        ex = np.asarray(extra, dtype=float)
        ex = ex[..., None] if ex.ndim == 2 else ex
        ex = np.where(np.isfinite(ex), ex, 0.0).reshape(ny * nx, -1)
        B = np.concatenate([B, ex], axis=1)
    return B


def stubblefield_forward_multiplier(
    ny: int,
    nx: int,
    dx: float,
    dy: float,
    H: float,
    ux_myr: float,
    uy_myr: float,
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    gamma: float = 0.0,
    theta: float = 1e-14,
) -> np.ndarray:
    r"""Complex steady multiplier :math:`M_h = t_r G_h/\text{SPY}` (m/yr -> m).

    Built on the ``(ny, nx)`` FFT wavenumber grid at posting ``(dx, dy)``. Pass
    the **padded** shape (e.g. ``2*ny, 2*nx``) when the field will be
    mirror-padded before transforming, as :class:`StubblefieldForward` does.
    ``alpha_{x,y} = alpha_scale * u_{x,y} * t_r / H`` (E1b advection calibration).

    ``M_h[0, 0] = 0`` — this operator is deliberately **DC-blind**, which is a
    policy of this consumer and not the kernel's own :math:`k=0` value.
    :meth:`~stereo_melt.dynamics.linear_perturbation.LinearPerturbation.steady_state_kernel`
    carries the physical long-wavelength limit :math:`G_h(0) = -2` (at
    :math:`\gamma = 0`), i.e. the full hydrostatic plateau, and that is right
    for a mass-budget operator. It is wrong here: the variational inverse this
    multiplier feeds is fitted against a **high-passed** target
    (:func:`variational_melt_rate` subtracts a Gaussian background, or projects
    a fitted background out of the residual), so the domain mean has been
    removed from the data by construction. A live DC bin would let the
    optimiser fit a uniform melt level to whatever edge and NaN residual the
    high-pass leaves — an artefact with nothing to constrain it. Zeroing the
    bin instead keeps the mask-mean an exact null direction, which is what
    :func:`variational_melt_inverse`'s mean pin and the "channel correction,
    not a mass-budget melt" contract both assume. The recovered level therefore
    comes from a prior (``m_prior``/``bg_degree``) or not at all.
    """
    model = LinearPerturbation(H=H, eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w,
                               g=g, gamma=gamma, theta=theta)
    a_fac = alpha_scale * model.tr / (H * SECONDS_PER_YEAR)
    model.alpha = float(ux_myr) * a_fac
    model.alpha_y = float(uy_myr) * a_fac
    kx, ky = _wavenumber_grids(nx, ny, dx, dy)
    G_h, _ = model.steady_state_kernel(kx, ky)
    M = model.tr * np.asarray(G_h) / SECONDS_PER_YEAR
    M[0, 0] = 0.0
    return M


def _pad2x(m: torch.Tensor) -> torch.Tensor:
    """Differentiable symmetric doubling (matches ``np.pad(..., 'symmetric')``)."""
    m = torch.cat([m, torch.flip(m, dims=[0])], dim=0)
    m = torch.cat([m, torch.flip(m, dims=[1])], dim=1)
    return m


class StubblefieldForward(torch.nn.Module):
    """Fixed differentiable linear layer :math:`m \\to \\delta z_s`.

    Mirror-pads the melt field to twice its size, multiplies by the padded
    spectral multiplier ``M_h``, inverse-transforms, and crops back. ``M_h`` is
    a fixed (non-trainable) buffer, so gradients flow only to the melt
    representation.
    """

    def __init__(self, M_h: np.ndarray, ny: int, nx: int):
        super().__init__()
        self.ny, self.nx = int(ny), int(nx)
        self.register_buffer("M", torch.from_numpy(np.ascontiguousarray(M_h)))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        padded = _pad2x(m).to(torch.complex128)
        dzs = torch.fft.ifft2(self.M * torch.fft.fft2(padded))
        return dzs.real[:self.ny, :self.nx]


from .geometry_bins import kmeans_geometry as _kmeans_geometry  # noqa: E402


class BlendedStubblefieldForward(torch.nn.Module):
    r"""Spatially varying forward operator: globally applied multipliers, blended.

    :class:`StubblefieldForward` applies one multiplier built from a single
    reference thickness and a single mean velocity. That is defensible on a
    synthetic twin with uniform flow, but not on a real shelf: Pine Island spans
    roughly 300 to 4000 m/yr, so one :math:`\alpha \propto u\,t_r/H` is wrong
    nearly everywhere.

    **Why not tiles.** The obvious fix -- cut the domain into tiles, give each its
    own :math:`M_h`, overlap-add -- is wrong here, because this operator is not
    spatially compact. Long-wavelength modes relax slowly under viscous bridging,
    so they advect far downstream before they damp: the impulse response falls to
    5% of peak only after ~8 km at 200 m/yr but ~75 km at 2000 m/yr and ~150 km at
    4000 m/yr. Any tile small enough to localize the geometry truncates that tail
    (a measured 47% forward error at 2000 m/yr, spatially uniform, and *not*
    removable by high-passing).

    So instead of localizing the *melt*, this localizes the *operator*: each
    multiplier is applied to the **whole** domain -- preserving the full
    downstream tail -- and the global responses are recombined with smooth
    spatial weights,

    .. math:: \delta z_s(x) = \sum_b w_b(x)\,\big[\mathcal{F}^{-1} M_b
              \mathcal{F} m\big](x), \qquad \sum_b w_b(x) = 1.

    Geometry is clustered into ``n_bins`` groups of similar :math:`(H, u_x, u_y)`
    (deterministically, see :func:`_kmeans_geometry`), each contributing one
    multiplier, and the hard bin indicators are Gaussian-smoothed over
    ``blend_px`` and renormalized so the weights are an exact partition of unity.
    With a spatially constant geometry every cell lands in one bin, so the result
    is **bit-identical** to :class:`StubblefieldForward`.

    Cost is one forward FFT plus ``n_bins`` inverse FFTs per call -- cheaper than
    a correctly haloed tiling, and independent of how finely the geometry varies.

    Parameters
    ----------
    ny, nx, dx, dy :
        Grid shape and posting (m).
    H_field, ux_field, uy_field : (ny, nx) arrays
        Local thickness (m) and velocity (m/yr). Non-finite cells (off-shelf) do
        not vote on the clustering and take the nearest valid weights.
    eta_field : (ny, nx) array, optional
        Spatially varying effective viscosity (Pa s). When given it joins the
        clustering features and each bin's multiplier is built with its own
        ``eta_bar`` (overriding the scalar in ``mult_kwargs``).
    n_bins :
        Number of distinct multipliers. Cost is linear in it.
    blend_px :
        Gaussian sigma (pixels) used to soften the bin indicators.
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        dx: float,
        dy: float,
        H_field: np.ndarray,
        ux_field: np.ndarray,
        uy_field: np.ndarray,
        *,
        eta_field: np.ndarray | None = None,
        n_bins: int = 6,
        blend_px: float = 8.0,
        **mult_kwargs,
    ):
        super().__init__()
        self.ny, self.nx = int(ny), int(nx)

        H_field = np.asarray(H_field, dtype=float)
        ux_field = np.asarray(ux_field, dtype=float)
        uy_field = np.asarray(uy_field, dtype=float)
        valid = (np.isfinite(H_field) & np.isfinite(ux_field)
                 & np.isfinite(uy_field) & (H_field > 0))
        cols = [H_field, ux_field, uy_field]
        if eta_field is not None:
            # Spatially varying effective viscosity (e.g. inferred from a
            # momentum-balance inversion): eta joins the geometry clustering and
            # each bin's multiplier is built with its own eta_bar, overriding
            # the scalar from mult_kwargs. Soft shear margins and stiff trunk
            # ice then get different bridging responses.
            eta_field = np.asarray(eta_field, dtype=float)
            valid &= np.isfinite(eta_field) & (eta_field > 0)
            cols.append(eta_field)
        if not valid.any():
            raise ValueError("no finite (H, ux, uy[, eta]) cells for the "
                             "blended operator")

        feats = np.stack([c[valid] for c in cols], axis=1)
        n_bins = max(1, min(int(n_bins), len(np.unique(feats, axis=0))))
        if n_bins == 1:
            lab_v = np.zeros(len(feats), dtype=int)
            cent = feats.mean(0, keepdims=True)
        else:
            lab_v, cent = _kmeans_geometry(feats, n_bins)
        self.n_bins = n_bins
        self.bin_geometry = [tuple(float(c) for c in row) for row in cent]

        def _bin_kwargs(row):
            if eta_field is None:
                return mult_kwargs
            return {**mult_kwargs, "eta_bar": row[3]}

        mults = [stubblefield_forward_multiplier(
                     2 * self.ny, 2 * self.nx, dx, dy, row[0], row[1], row[2],
                     **_bin_kwargs(row))
                 for row in self.bin_geometry]
        self.register_buffer(
            "M", torch.from_numpy(np.ascontiguousarray(np.stack(mults))))

        # Soft weights: smooth each hard indicator, then renormalize. Off-shelf
        # cells carry no indicator of their own, so smoothing lets the nearest
        # on-shelf bins fill them in and the partition of unity still holds.
        w = np.zeros((n_bins, self.ny, self.nx))
        for b in range(n_bins):
            ind = np.zeros((self.ny, self.nx))
            ind[valid] = (lab_v == b)
            w[b] = gaussian_filter(ind, blend_px, mode="nearest")
        tot = w.sum(0)
        flat = tot < 1e-8            # unreachable by smoothing: fall back to bin 0
        w[0][flat] = 1.0
        tot = np.maximum(w.sum(0), 1e-30)
        self.register_buffer("w", torch.from_numpy(w / tot))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        F = torch.fft.fft2(_pad2x(m).to(torch.complex128))
        resp = torch.fft.ifft2(self.M * F).real[:, :self.ny, :self.nx]
        return (self.w * resp).sum(0)


class GridMelt(torch.nn.Module):
    """Dense per-pixel melt field (the direct representation)."""

    def __init__(self, ny: int, nx: int):
        super().__init__()
        self.m = torch.nn.Parameter(torch.zeros(int(ny), int(nx)))

    def forward(self) -> torch.Tensor:
        return self.m


class SirenMelt(torch.nn.Module):
    """SIREN coordinate network :math:`(x, y)\\in[-1,1]^2 \\to m`.

    A small implicit neural representation with periodic (sine) activations
    (Sitzmann et al. 2020); a smoother, resolution-independent alternative to
    the dense :class:`GridMelt`. Recovers the same amplitude to the optimizer
    noise floor, at ~4-8x the cost per iteration.
    """

    def __init__(self, ny: int, nx: int, hidden: int = 128, layers: int = 4,
                 w0: float = 30.0, scale: float = 0.1):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, int(ny)),
                                torch.linspace(-1, 1, int(nx)), indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy], -1).reshape(-1, 2))
        self.w0, self.scale = float(w0), float(scale)
        dims = [2] + [hidden] * layers + [1]
        self.lin = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        with torch.no_grad():  # SIREN weight init
            for i, lyr in enumerate(self.lin):
                b = 1.0 / dims[i] if i == 0 else math.sqrt(6.0 / dims[i]) / w0
                lyr.weight.uniform_(-b, b)
        self.ny, self.nx = int(ny), int(nx)

    def forward(self) -> torch.Tensor:
        h = self.coords
        for i, lyr in enumerate(self.lin[:-1]):
            h = torch.sin((self.w0 if i == 0 else 1.0) * lyr(h))
        return (self.lin[-1](h) * self.scale).reshape(self.ny, self.nx)


@dataclass
class MeltInverseResult:
    """Output of :func:`variational_melt_inverse`."""

    melt: np.ndarray                    # (ny, nx) recovered basal melt, m/yr
    loss_history: list                  # data-misfit term per recorded iteration
    n_iter: int
    dzs_fit: np.ndarray | None = None   # (ny, nx) modeled surface anomaly, m
    # fitted background surface (m) when bg_degree is set: the reference state
    # projected out of the data residual, so full model = dzs_fit + bg_field
    bg_field: np.ndarray | None = None
    # geometry bins actually used (the request is clamped to the number of
    # distinct geometries present); 1 means the monolithic operator
    n_bins_used: int = 1


def variational_melt_inverse(
    dzs_obs: np.ndarray,
    mask: np.ndarray,
    dx: float,
    dy: float,
    H: float,
    ux_myr: float,
    uy_myr: float,
    *,
    rep: str = "grid",
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    lam: float = 1e-4,
    iters: int = 4000,
    lr: float = 3e-3,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    siren_kwargs: dict | None = None,
    log_every: int = 0,
    H_field: np.ndarray | None = None,
    ux_field: np.ndarray | None = None,
    uy_field: np.ndarray | None = None,
    eta_field: np.ndarray | None = None,
    n_bins: int | None = None,
    blend_px: float = 8.0,
    m_prior: np.ndarray | None = None,
    anchor_lp_sigma_px: float | None = None,
    anchor_short_lambda_px: float | None = None,
    bg_degree: int | None = None,
    bg_extra: np.ndarray | None = None,
) -> MeltInverseResult:
    r"""Fit basal melt to an observed surface (anomaly) through the forward operator.

    Minimizes ``||(I - P_B)(Forward(m) - dzs_obs)||^2_mask + lam*||m - m_prior||^2``
    by Adam, with the Stubblefield transfer as a fixed differentiable forward
    layer. ``P_B`` projects out an optional low-order background basis (see
    ``bg_degree``), and ``m_prior`` optionally anchors the null-space to a budget
    field. With both defaulted this reduces to
    ``||Forward(m) - dzs_obs||^2 + lam*||m||^2``.

    Parameters
    ----------
    dzs_obs, mask : (ny, nx) arrays
        Observed surface-elevation anomaly (m) and a boolean fit mask (the
        scored / trustworthy region). Non-finite ``dzs_obs`` cells should be
        excluded by ``mask``.
    dx, dy, H, ux_myr, uy_myr :
        Posting (m), reference thickness (m), and mean velocity components
        (m/yr) — the operator geometry.
    rep : {"grid", "siren"}
        Melt representation: dense grid or SIREN coordinate network.
    eta_bar, alpha_scale :
        Forward-model calibration (see the module docstring). Study parameters.
    lam, iters, lr :
        Tikhonov weight, Adam iterations, and learning rate.
    log_every :
        If > 0, print the data/reg terms every ``log_every`` iterations (heavy
        fits run under ``nohup``; the log is the only visibility).
    eta_field : (ny, nx) array, optional
        Spatially varying effective (Newtonian-equivalent) viscosity, Pa s —
        e.g. a momentum-balance inversion linearized about the observed strain
        rate. Joins the geometry clustering, and each bin's multiplier is built
        with its own ``eta_bar`` (the scalar ``eta_bar`` is then only a
        fallback for cells outside the field). Requires ``n_bins``.
    H_field, ux_field, uy_field, n_bins, blend_px :
        Set ``n_bins`` to use the spatially varying
        :class:`BlendedStubblefieldForward` instead of the single-``(H, u)``
        operator: the geometry fields are clustered into ``n_bins`` multipliers,
        each applied globally and blended over ``blend_px``. Required on real
        shelves, where one mean velocity is wrong nearly everywhere. Fields
        default to constants from the scalars (which reduces to the monolithic
        operator exactly).
    m_prior : (ny, nx) array, optional
        Budget-anchored null-space: the Tikhonov term becomes
        ``lam*||m - m_prior||^2``, pulling melt toward this field (a budget/level
        melt in Stubblefield sign, m>0 = melt) instead of zero. The DC-blind
        operator cannot constrain the mean or the global ramp, so those modes
        relax to ``m_prior`` while the data only moves melt off it where the
        operator has gain (the channel band). ``None`` = the toward-zero pin.
    anchor_lp_sigma_px : float, optional
        Prior-owned long-wavelength band: Gaussian sigma (pixels) of a low-pass
        that splits the problem in two, consistently on both sides of the
        objective. Melt side: the low-pass component of the fitted deviation
        ``dm`` is projected out every step (the scale generalization of the
        mask-mean pin), so structure broader than this scale comes from
        ``m_prior`` *by construction* — no scalar ``lam`` can otherwise separate
        the genuine channel-band correction from long-λ melt modes recruited to
        absorb unmodeled smooth surface structure (a DATA-FAVORED leak). Data
        side: the same low-pass is removed from the residual, blinding the fit
        to the scales the model refuses to explain — without this the leak does
        not vanish but ALIASES into superposed channel-band wiggles (worse).
        Choose so the cutoff wavelength (~5.3 sigma) sits above BOTH the
        bridging knee ``2*pi*H`` (where the budget prior is near-exact anyway)
        AND the melt anomalies' own spectral support: ~``4*H/dx`` (cutoff
        ~21 H) in practice — the E2a A/B showed ``2*H/dx`` (cutoff ~10.7 H)
        already clips band-scale corrections, whose content peaks near 12.6 H
        for a 3.3 H-wide channel. ``None`` = pin only the mask mean (legacy).
    anchor_short_lambda_px : float, optional
        Short-wavelength edge of the correction band (a cutoff WAVELENGTH in
        pixels, not a sigma), active only with ``anchor_lp_sigma_px``. Below
        this scale the operator transfer sits on its floor, so any fitted melt
        there is amplified noise by construction — with the long-wavelength
        residual blinded, the optimizer otherwise pumps exactly that junk
        (E2a patch: lam had to rise to ~1 to suppress it, killing genuine
        corrections elsewhere). The pin becomes a band-pass and the residual is
        blinded to match, which returns ``lam`` to a benign default. ~2.5 H in
        pixels (the ``bridging_restoration`` band limit; the transfer knee is
        ``2*pi*H``). ``None`` = long-side pin only.
    bg_degree : int, optional
        Degree of a polynomial background basis (0=DC, 1=plane, 2=quadratic)
        fitted *in the model* and projected out of the data residual each step
        (variable projection). Replaces the input high-pass: it removes the
        reference state the linearization expands about without cutting a
        wavelength band from the melt signal, so the raw (un-high-passed) surface
        is then the correct input. ``None`` = no background term (the input must
        already be an anomaly).
    bg_extra : (ny, nx) or (ny, nx, k) array, optional
        Extra background regressors appended to the polynomial basis (e.g. the
        hydrostatic surface of the thickness field). Non-finite cells are zeroed.

    Returns
    -------
    MeltInverseResult
        Recovered melt (m/yr), the data-misfit history, and the iteration count.
    """
    ny, nx = dzs_obs.shape
    if mask.shape != dzs_obs.shape:
        raise ValueError(f"mask shape {mask.shape} != dzs_obs {dzs_obs.shape}")
    if rep not in ("grid", "siren"):
        raise ValueError(f"rep must be 'grid' or 'siren', got {rep!r}")

    mult_kw = dict(eta_bar=eta_bar, alpha_scale=alpha_scale,
                   rho_i=rho_i, rho_w=rho_w, g=g)
    if n_bins is None:
        if eta_field is not None:
            raise ValueError("eta_field requires n_bins (the blended operator)")
        M_h = stubblefield_forward_multiplier(
            2 * ny, 2 * nx, dx, dy, H, ux_myr, uy_myr, **mult_kw)
        fwd = StubblefieldForward(M_h, ny, nx)
    else:
        def _fld(a, v):
            return np.full((ny, nx), float(v)) if a is None else np.asarray(a)
        fwd = BlendedStubblefieldForward(
            ny, nx, dx, dy, _fld(H_field, H), _fld(ux_field, ux_myr),
            _fld(uy_field, uy_myr), eta_field=eta_field,
            n_bins=int(n_bins), blend_px=blend_px, **mult_kw)
        if log_every:
            gh = np.array([g[0] for g in fwd.bin_geometry])
            gu = np.hypot([g[1] for g in fwd.bin_geometry],
                          [g[2] for g in fwd.bin_geometry])
            eta_note = ""
            if len(fwd.bin_geometry[0]) > 3:
                ge = np.array([g[3] for g in fwd.bin_geometry])
                eta_note = f"  eta {ge.min():.2e}-{ge.max():.2e} Pa s"
            print(f"    blended operator: {fwd.n_bins} geometry bins  "
                  f"H {gh.min():.0f}-{gh.max():.0f} m  "
                  f"|u| {gu.min():.0f}-{gu.max():.0f} m/yr{eta_note}", flush=True)
    rep_net = (GridMelt(ny, nx) if rep == "grid"
               else SirenMelt(ny, nx, **(siren_kwargs or {})))

    dzs_t = torch.from_numpy(np.ascontiguousarray(dzs_obs, dtype=np.float64))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask, dtype=bool))

    # Budget-anchored null-space: pull melt toward m_prior (the budget/level
    # field) instead of zero, so the modes the DC-blind operator cannot see
    # relax to the budget rather than vanishing. None reproduces the toward-zero
    # pin exactly.
    anchor = m_prior is not None
    m_prior_t = (torch.zeros(ny, nx, dtype=torch.float64) if not anchor
                 else torch.from_numpy(np.ascontiguousarray(
                     np.where(np.isfinite(m_prior), m_prior, 0.0),
                     dtype=np.float64)))

    # Background-in-the-model: a low-order basis for the reference state the
    # linearization expands about (the shape a Gaussian high-pass used to strip
    # from the input). We instead project it out of the data residual each step
    # (variable projection), so the raw surface goes in unfiltered and no
    # melt-bearing band is discarded. Qb spans the basis on the masked cells.
    Qb = Rinv = B_full_t = None
    if bg_degree is not None:
        B_full_t = torch.from_numpy(np.ascontiguousarray(
            _poly_basis(ny, nx, int(bg_degree), extra=bg_extra)))
        B_m = B_full_t[torch.from_numpy(np.ascontiguousarray(mask, bool).ravel())]
        if B_m.shape[0] <= B_m.shape[1]:
            raise ValueError(
                f"background basis ({B_m.shape[1]} cols) >= masked cells "
                f"({B_m.shape[0]}); lower bg_degree")
        Qb, Rb = torch.linalg.qr(B_m)
        Rinv = torch.linalg.inv(Rb)
        if log_every:
            print(f"    background projection: degree {bg_degree}, "
                  f"{B_m.shape[1]} basis columns removed from the residual",
                  flush=True)

    # Reparametrize as m = m_prior + dm: the network fits the deviation dm (init
    # 0) and reg pulls dm toward 0, so where the DC-blind operator sees nothing
    # the melt equals the prior. A plain ||m - m_prior||^2 penalty has the same
    # minimum but converges far too slowly in the flat null-space direction.
    # The mask-mean of dm is the exact null direction (the operator's k=0 bin
    # is pinned to zero in stubblefield_forward_multiplier, so it maps a
    # constant to nothing): Adam's per-pixel normalization drifts it even though the data
    # gradient is mean-zero, so we pin it -- the melt's unobservable level then
    # comes from the prior, not the optimizer. m_prior=None => dm=m (legacy).
    # anchor_lp_sigma_px generalizes the pin from the mean to the whole
    # long-wavelength band: dm <- dm - LP(dm), so the prior owns every scale
    # the low-pass keeps and the data can only shape the channel band. LP MUST
    # be a true spectral projector (idempotent): a spatial Gaussian I-G leaks
    # G(1-G) (max 25% at its half-power band) and is invertible at low k, so
    # the optimizer tunnels through it and near-cutoff junk contaminates the
    # fit (first gate attempt). We therefore build a 0/1 low-pass mask (narrow
    # raised-cosine ramp against Gibbs) on the mirror-padded FFT grid, with the
    # cutoff at the Gaussian-equivalent half-power wavelength ~5.34*sigma.
    rm_W = None                      # spectral mask of the REMOVED band(s)
    if anchor_lp_sigma_px:
        py_, px_ = ny // 2, nx // 2
        ky = np.fft.fftfreq(ny + 2 * py_)[:, None]
        kx = np.fft.rfftfreq(nx + 2 * px_)[None, :]
        kk = np.hypot(ky, kx)

        def _cos_lp(k_cut: float) -> np.ndarray:
            """1 below the cut, 0 above, raised-cosine ramp over /1.2..*1.2."""
            k_lo, k_hi = k_cut / 1.2, k_cut * 1.2
            t = np.clip((k_hi - kk) / (k_hi - k_lo), 0.0, 1.0)
            return np.where(kk <= k_lo, 1.0, np.where(kk >= k_hi, 0.0,
                            0.5 - 0.5 * np.cos(math.pi * t)))

        s = float(anchor_lp_sigma_px)
        k_c = math.sqrt(2.0 * math.log(2.0)) / (2.0 * math.pi * s)
        keep = 1.0 - _cos_lp(k_c)          # long side: prior-owned band out
        note = f"cutoff lambda ~{1.0 / k_c:.0f} px"
        if anchor_short_lambda_px:
            # short side: below the transfer floor any melt is amplified
            # noise -- remove it from the fittable band and the residual alike
            keep = keep * _cos_lp(1.0 / float(anchor_short_lambda_px))
            note += f", short cut {float(anchor_short_lambda_px):.0f} px"
        rm_W = torch.from_numpy(np.ascontiguousarray(1.0 - keep))
        if log_every:
            print(f"    anchor band pin: {note} (spectral projector) -- "
                  f"long-wavelength dm follows the prior; data residual "
                  f"blinded to the kept band", flush=True)

    def _rm(field: torch.Tensor) -> torch.Tensor:
        """The removed-band component of a field (mirror-padded rfft2)."""
        py, px = ny // 2, nx // 2
        f = torch.nn.functional.pad(field[None, None], (px, px, py, py),
                                    mode="reflect")[0, 0]
        F = torch.fft.rfft2(f)
        out = torch.fft.irfft2(F * rm_W.to(F.dtype), s=f.shape)
        return out[py:py + ny, px:px + nx]

    def _pin(dm: torch.Tensor) -> torch.Tensor:
        if rm_W is not None:
            dm = dm - _rm(dm)
        if anchor:
            dm = dm - dm[mask_t].mean()
        return dm

    def _blind(r_field: torch.Tensor) -> torch.Tensor:
        # Consistency of the projection contract: the data term must not SEE
        # the bands the model refuses to explain -- long-wavelength residual
        # would otherwise be mis-attributed by ALIASING into channel-band
        # wiggles (the gate's dome-leak failure without this), and
        # sub-transfer-floor residual would be chased with amplified melt
        # noise. Zero-fill outside the mask before filtering (NaN-safe).
        if rm_W is None:
            return r_field
        r0 = torch.where(mask_t, r_field, torch.zeros((), dtype=r_field.dtype))
        return r_field - _rm(r0)

    opt = torch.optim.Adam(rep_net.parameters(), lr=lr)
    history: list = []
    for it in range(iters):
        opt.zero_grad()
        dm = _pin(rep_net())
        m = dm + m_prior_t
        pred = fwd(m)
        r = _blind(pred - dzs_t)[mask_t]
        if Qb is not None:                       # remove the fitted background
            r = r - Qb @ (Qb.t() @ r)
        data = (r ** 2).mean()
        reg = (dm ** 2).mean()
        loss = data + lam * reg
        loss.backward()
        opt.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            history.append(float(data.item()))
            print(f"    it {it:5d}  data {data.item():.3e}  reg {reg.item():.3e}",
                  flush=True)

    with torch.no_grad():
        m_final = _pin(rep_net()) + m_prior_t
        melt = m_final.cpu().numpy()
        dzs_fit_t = fwd(m_final)
        dzs_fit = dzs_fit_t.cpu().numpy()
        bg_field = None
        if Qb is not None:
            r_fin = _blind(dzs_fit_t - dzs_t)[mask_t]
            theta = -(Rinv @ (Qb.t() @ r_fin))    # coeffs in the raw basis
            bg_field = (B_full_t @ theta).reshape(ny, nx).cpu().numpy()
    return MeltInverseResult(melt=melt, loss_history=history, n_iter=iters,
                             dzs_fit=dzs_fit, bg_field=bg_field,
                             n_bins_used=getattr(fwd, "n_bins", 1))


def variational_melt_rate(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    floating_mask: xr.DataArray | None = None,
    d: xr.DataArray | float = 0.0,
    *,
    rep: str = "grid",
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    lam: float = 1e-4,
    iters: int = 4000,
    lr: float = 3e-3,
    sigma_hp_H: float = 5.0,
    rho_w: float = rhow,
    rho_i: float = rhoi,
    g: float = G_GRAVITY,
    siren_kwargs: dict | None = None,
    log_every: int = 0,
    n_bins: int | None = None,
    blend_km: float = 4.0,
    eta_field: xr.DataArray | None = None,
    m_prior: xr.DataArray | None = None,
    anchor_lp_sigma_H: float | None = None,
    anchor_short_lambda_H: float | None = 2.5,
    bg_degree: int | None = None,
    bg_extra: xr.DataArray | None = None,
) -> xr.Dataset:
    r"""Basal melt rate from the surface (anomaly) by variational forward-fit.

    Driver-facing xarray wrapper around :func:`variational_melt_inverse`:
    strip-robust time-**median** surface -> remove the reference state (legacy
    high-pass of the input, or -- with ``bg_degree`` -- an in-model background
    fitted and projected out of the residual) -> fit melt through the
    Stubblefield forward operator on the floating bbox, optionally anchored to a
    budget field via ``m_prior`` -> Shean sign (negative = melt).

    Unlike a direct Fourier division of the surface by the transfer (which
    over-lifts across-flow ridges), this puts the bridging transfer in the
    *forward* model and fits, so along-flow, oblique, and across-flow melt come
    through one operator with no angular weight.

    Set ``n_bins`` to run the spatially varying
    :class:`BlendedStubblefieldForward`, which clusters the shelf into that many
    thickness/velocity geometries and blends their globally applied responses.
    Without it the operator collapses the shelf to one ``H_ref`` and one mean
    ``(u0x, u0y)``, which is a synthetic-twin assumption -- on a real shelf
    spanning a large velocity range it is wrong nearly everywhere, so ``n_bins``
    is the appropriate setting for production. Either way the operator is
    DC-blind (:func:`stubblefield_forward_multiplier` pins ``k=0`` to zero, by
    policy rather than by accident -- see there): the recovered field is a
    channel-scale pattern correction, **not** a mass-budget melt, and is not
    interchangeable with the Eulerian/Lagrangian solvers.

    Parameters
    ----------
    h_stack : xarray.DataArray ``(time, y, x)``
        Geoid-referenced, corrected surface-elevation stack, meters.
    vx, vy : xarray.DataArray
        Column-averaged velocity (m/yr); the time/shelf mean sets the advection.
    floating_mask : xarray.DataArray, optional
        Floating-ice mask; the inversion is reported only there.
    d : firn air content (m), for the reference thickness.
    rep, eta_bar, alpha_scale, lam, iters, lr :
        Passed through to :func:`variational_melt_inverse` (see there and the
        module docstring; ``eta_bar``/``alpha_scale`` are per-geometry study
        parameters, not constants).
    eta_field : xarray.DataArray, optional
        Spatially varying effective (Newtonian-equivalent) viscosity on the
        stack grid, Pa s — e.g. an icepack/momentum-balance inversion
        linearized about the observed strain rate,
        :math:`\bar\eta = \tfrac12 A^{-1/n}\,\dot\varepsilon_e^{(1-n)/n}`.
        Joins the geometry clustering of the blended operator so soft shear
        margins and stiff trunk ice get different bridging responses; the
        scalar ``eta_bar`` remains the study-parameter fallback. Requires
        ``n_bins``.
    m_prior : xarray.DataArray, optional
        Budget/level melt field (Shean sign, negative = melt) on the stack grid.
        Anchors the fit's null-space (mean + global ramp, which the DC-blind
        operator cannot see) to the budget, so the fused melt carries an absolute
        level and flux/gain become meaningful. Negated internally to the
        operator's sign. ``None`` = the DC-blind toward-zero pin.
    anchor_lp_sigma_H : float, optional
        Prior-owned long-wavelength band, as a Gaussian sigma in units of the
        reference thickness ``H_ref`` (converted to pixels and passed to
        :func:`variational_melt_inverse` as ``anchor_lp_sigma_px``). ~2.0 puts
        the half-power split at ~10.7 H, above the bridging knee ``2*pi*H``:
        melt structure broader than that follows ``m_prior`` by construction,
        which stops the data term from recruiting long-λ melt modes to absorb
        large-scale surface structure the background basis cannot represent
        (the failure mode on localized-anomaly geometries). ``None`` = mask-mean
        pin only.
    anchor_short_lambda_H : float, optional
        Short-wavelength edge of the correction band (cutoff wavelength,
        x ``H_ref``), active only with ``anchor_lp_sigma_H``. Below it the
        operator transfer sits on its floor, so fitted melt there is amplified
        noise; removing the band from both the fit and the residual keeps the
        freed optimizer from chasing unfittable short-scale content once the
        long wavelengths are blinded (E2a patch behavior), and returns ``lam``
        to a benign default. Default 2.5 (the ``bridging_restoration`` band
        limit). ``None`` = long-side pin only.
    bg_degree : int, optional
        Fit a polynomial background of this degree (1=plane, 2=quadratic) in the
        model and feed the RAW median surface, instead of the legacy Gaussian
        high-pass (``sigma_hp_H``, used only when ``bg_degree is None``). Removes
        the reference state without cutting a melt-bearing wavelength band.
    bg_extra : xarray.DataArray, optional
        Extra background regressor on the stack grid (e.g. the hydrostatic
        surface of the thickness field), appended to the polynomial basis.

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` (m ice/yr, Shean sign), the observed high-passed surface
        anomaly ``dzs_obs`` and the operator's fit to it ``dzs_fit``, plus
        ``H_ref_m``, ``t_r_yr``, ``u0x_myr``/``u0y_myr``, ``eta_bar``,
        ``alpha_scale``, ``lam``, ``iters``, ``rep`` and the fit-quality attrs
        ``fit_rms_resid_m`` / ``fit_rms_obs_m`` / ``fit_var_explained``. On a real
        shelf there is no truth, so how much of ``dzs_obs`` the operator can
        reproduce is the only available self-diagnostic: a low
        ``fit_var_explained`` means the surface structure is not something this
        (single-``H``, single-mean-``u``, steady) operator can make from *any*
        melt field, and the recovered melt should be read with that caveat.
        With the band contract active this becomes a KEPT-BAND variance
        explained and a working channel detector (E2a: ~0.7 where a genuine
        channel exists vs ~0.1 on a channel-free shelf, where the correction is
        noise): low values say trust the prior — raise ``lam`` or report the
        budget melt alone.
    """
    # strip-robust surface (median over time, not mean)
    h_med = h_stack.median("time", skipna=True)
    if floating_mask is not None:
        h_med = h_med.where(floating_mask)
    res = float(abs(h_stack["x"].values[1] - h_stack["x"].values[0]))

    fl = np.isfinite(h_med.values)
    if floating_mask is not None:
        fl &= floating_mask.values.astype(bool)
    if not fl.any():
        raise ValueError("no finite floating-shelf surface for the variational inverse")

    H_ref = float(freeboard_to_thickness(float(np.nanmean(h_med.values[fl])), d=0.0,
                                         rho_w=rho_w, rho_i=rho_i))
    if isinstance(d, xr.DataArray):
        H_ref = float(np.nanmean(
            freeboard_to_thickness(h_med, d=d, rho_w=rho_w, rho_i=rho_i).values[fl]))
    if not np.isfinite(H_ref) or H_ref <= 0:
        raise ValueError(f"derived H_ref={H_ref} is not positive finite")
    t_r_yr = (2.0 * eta_bar / (rho_i * g * H_ref)) / SECONDS_PER_YEAR

    # advection from the mean flow
    vxm = vx.mean("time") if "time" in vx.dims else vx
    vym = vy.mean("time") if "time" in vy.dims else vy
    flda = xr.DataArray(fl, dims=h_med.dims, coords=h_med.coords)
    u0x = float(vxm.where(flda).mean(skipna=True))
    u0y = float(vym.where(flda).mean(skipna=True))
    u0x = 0.0 if not np.isfinite(u0x) else u0x
    u0y = 0.0 if not np.isfinite(u0y) else u0y

    # crop to the floating bbox
    ys, xs = np.where(fl)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    hm = h_med.values[y0:y1, x0:x1]
    fl_crop = fl[y0:y1, x0:x1]
    # Remove the reference state the operator expands about, two ways:
    #   bg_degree is None -> legacy Gaussian high-pass of the INPUT (a band cut)
    #   bg_degree set      -> feed the raw surface; the background is fitted in
    #                         the model and projected out of the residual.
    if bg_degree is None:
        dzs = hm - _nan_gauss(hm, sigma_hp_H * H_ref / res)
    else:
        dzs = hm.copy()
    fit_mask = np.isfinite(dzs) & fl_crop
    dzs = np.where(np.isfinite(dzs), dzs, 0.0)

    # Budget-anchored null-space prior (Shean sign -> Stubblefield sign) and any
    # extra background regressor, cropped to the fit bbox.
    m_prior_crop = None
    if m_prior is not None:
        mp = m_prior.mean("time") if "time" in m_prior.dims else m_prior
        m_prior_crop = -np.asarray(mp.values)[y0:y1, x0:x1]
    bg_extra_crop = (None if bg_extra is None
                     else np.asarray(bg_extra.values)[y0:y1, x0:x1])

    # Local geometry for the blended operator: thickness and velocity on the same
    # crop, masked off the shelf so the geometry clustering is not pulled by open
    # ocean or grounded ice.
    H_crop = ux_crop = uy_crop = eta_crop = None
    if n_bins is not None:
        H_full = freeboard_to_thickness(h_med, d=d, rho_w=rho_w, rho_i=rho_i).values
        H_crop = np.where(fl_crop, H_full[y0:y1, x0:x1], np.nan)
        ux_crop = np.where(fl_crop, vxm.values[y0:y1, x0:x1], np.nan)
        uy_crop = np.where(fl_crop, vym.values[y0:y1, x0:x1], np.nan)
        if eta_field is not None:
            eta_crop = np.where(
                fl_crop, np.asarray(eta_field.values)[y0:y1, x0:x1], np.nan)

    result = variational_melt_inverse(
        dzs, fit_mask, res, res, H_ref, u0x, u0y,
        rep=rep, eta_bar=eta_bar, alpha_scale=alpha_scale, lam=lam,
        iters=iters, lr=lr, rho_i=rho_i, rho_w=rho_w, g=g,
        siren_kwargs=siren_kwargs, log_every=log_every,
        H_field=H_crop, ux_field=ux_crop, uy_field=uy_crop,
        eta_field=eta_crop, n_bins=n_bins,
        blend_px=blend_km * 1000.0 / res,
        m_prior=m_prior_crop,
        anchor_lp_sigma_px=(None if anchor_lp_sigma_H is None
                            else anchor_lp_sigma_H * H_ref / res),
        anchor_short_lambda_px=(None if (anchor_lp_sigma_H is None
                                         or anchor_short_lambda_H is None)
                                else anchor_short_lambda_H * H_ref / res),
        bg_degree=bg_degree, bg_extra=bg_extra_crop)

    def _embed(crop: np.ndarray, name: str) -> xr.DataArray:
        """Place a bbox-cropped field back on the full grid, masked to the fit."""
        full = np.full(h_med.shape, np.nan, dtype=np.float64)
        full[y0:y1, x0:x1] = np.where(fit_mask, crop, np.nan)
        full[~fl] = np.nan
        return xr.DataArray(full, dims=h_med.dims, coords=h_med.coords, name=name)

    # Stubblefield m>0 = melt -> Shean negate; report only where fit
    melt = _embed(-result.melt, "melt_rate")

    # Fit quality: the share of the observed surface anomaly the operator
    # reproduces (the only no-truth self-diagnostic). With an in-model background
    # the anomaly is hm - bg_field and the model is dzs_fit + bg_field; with the
    # legacy high-pass the anomaly is the high-passed dzs and the model dzs_fit.
    if result.bg_field is not None:
        dzs_anom = dzs - result.bg_field
        resid_f = result.dzs_fit + result.bg_field - dzs
    else:
        dzs_anom = dzs
        resid_f = result.dzs_fit - dzs
    if anchor_lp_sigma_H is not None:
        # Evaluate in the kept band only -- the objective the fit actually
        # saw. Long-wavelength residual is prior-owned by contract and
        # sub-transfer-floor residual is unfittable noise; counting either
        # here would swamp the diagnostic (they are not the fit's to explain).
        short_px = (None if anchor_short_lambda_H is None
                    else anchor_short_lambda_H * H_ref / res)
        dzs_anom = _keep_band_np(dzs_anom, fit_mask,
                                 anchor_lp_sigma_H * H_ref / res, short_px)
        resid_f = _keep_band_np(resid_f, fit_mask,
                                anchor_lp_sigma_H * H_ref / res, short_px)
    resid = resid_f[fit_mask]
    obs = dzs_anom[fit_mask]
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))
    rms_obs = float(np.sqrt(np.mean(obs ** 2)))
    var_expl = float(1.0 - np.var(resid) / np.var(obs)) if np.var(obs) > 0 else np.nan

    data_vars = {
        "melt_rate": melt,
        "dzs_obs": _embed(dzs_anom, "dzs_obs"),
        "dzs_fit": _embed(result.dzs_fit, "dzs_fit"),
    }
    if result.bg_field is not None:
        data_vars["dzs_bg"] = _embed(result.bg_field, "dzs_bg")

    return xr.Dataset(
        data_vars,
        attrs={
            "fit_rms_resid_m": rms_resid,
            "fit_rms_obs_m": rms_obs,
            "fit_var_explained": var_expl,
            "fit_n_cells": int(fit_mask.sum()),
            "equation": ("Stubblefield 2023 forward-fit: "
                         "min ||(I-P_B)(Forward(m)-h)||^2 + lam||m-m_prior||^2"),
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "method": (
                f"variational forward-operator inverse ({rep})"
                + ("; budget-anchored null-space + in-model background"
                   if (m_prior is not None or bg_degree is not None)
                   else "; DC-blind channel correction")
                + ("; long-wavelength band pinned to prior and blinded in data"
                   if anchor_lp_sigma_H is not None else "")),
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref, "t_r_yr": t_r_yr,
            "u0x_myr": u0x, "u0y_myr": u0y,
            "alpha_scale": alpha_scale, "lam": lam, "iters": iters, "rep": rep,
            "sigma_hp_H": (-1.0 if bg_degree is not None else sigma_hp_H),
            "bg_degree": (-1 if bg_degree is None else int(bg_degree)),
            "budget_anchored": int(m_prior is not None),
            "anchor_lp_sigma_H": (-1.0 if anchor_lp_sigma_H is None
                                  else float(anchor_lp_sigma_H)),
            "anchor_short_lambda_H": (
                -1.0 if (anchor_lp_sigma_H is None
                         or anchor_short_lambda_H is None)
                else float(anchor_short_lambda_H)),
            "n_bins": -1 if n_bins is None else int(result.n_bins_used),
            "blend_km": -1.0 if n_bins is None else float(blend_km),
            "eta_field": int(eta_field is not None),
            # aliases so this Dataset slots into run_melt's linv plot/save plumbing
            "gamma_dimless": 0.0, "tr_yr": t_r_yr,
            "note": (
                (f"blended operator ({result.n_bins_used} geometry bins, "
                 "globally applied)"
                 if n_bins else "first-order (single H, single mean u)")
                + ("; budget-anchored, carries level" if m_prior is not None
                   else "; pattern diagnostic, not mass-budget melt")),
        },
    )
