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


def _nan_gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing (smooth value / smooth mask)."""
    m = np.isfinite(a)
    a0 = np.where(m, a, 0.0)
    return gaussian_filter(a0, sigma) / np.maximum(
        gaussian_filter(m.astype(float), sigma), 1e-6)


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
    """
    model = LinearPerturbation(H=H, eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w,
                               g=g, gamma=gamma, theta=theta)
    a_fac = alpha_scale * model.tr / (H * SECONDS_PER_YEAR)
    model.alpha = float(ux_myr) * a_fac
    model.alpha_y = float(uy_myr) * a_fac
    kx, ky = _wavenumber_grids(nx, ny, dx, dy)
    G_h, _ = model.steady_state_kernel(kx, ky)
    return model.tr * np.asarray(G_h) / SECONDS_PER_YEAR


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


def _kmeans_geometry(
    feats: np.ndarray, n_bins: int, iters: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 1-D-seeded Lloyd clustering of standardized ``(H, ux, uy)``.

    Seeded from evenly spaced quantiles along the first principal component (no
    RNG), so a given geometry always yields the same bins -- a solver that
    silently changed its operator between runs would be unusable for A/B work.
    Returns the per-cell label and the ``(n_bins, 3)`` centroids in feature units.
    """
    mu = feats.mean(0)
    sd = feats.std(0)
    sd[sd <= 0] = 1.0
    z = (feats - mu) / sd
    # principal direction via the power method on the covariance (no scipy/sklearn)
    C = np.cov(z.T) + 1e-12 * np.eye(z.shape[1])
    v = np.ones(z.shape[1]) / math.sqrt(z.shape[1])
    for _ in range(100):
        v = C @ v
        v /= max(np.linalg.norm(v), 1e-30)
    proj = z @ v
    qs = np.quantile(proj, (np.arange(n_bins) + 0.5) / n_bins)
    cent = np.stack([z[np.argmin(np.abs(proj - q))] for q in qs])

    lab = np.zeros(len(z), dtype=int)
    for _ in range(iters):
        d = ((z[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for b in range(n_bins):
            sel = lab == b
            if sel.any():
                cent[b] = z[sel].mean(0)
    return lab, cent * sd + mu


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
        if not valid.any():
            raise ValueError("no finite (H, ux, uy) cells for the blended operator")

        feats = np.stack([H_field[valid], ux_field[valid], uy_field[valid]], axis=1)
        n_bins = max(1, min(int(n_bins), len(np.unique(feats, axis=0))))
        if n_bins == 1:
            lab_v = np.zeros(len(feats), dtype=int)
            cent = feats.mean(0, keepdims=True)
        else:
            lab_v, cent = _kmeans_geometry(feats, n_bins)
        self.n_bins = n_bins
        self.bin_geometry = [tuple(float(c) for c in row) for row in cent]

        mults = [stubblefield_forward_multiplier(
                     2 * self.ny, 2 * self.nx, dx, dy, H_b, ux_b, uy_b,
                     **mult_kwargs)
                 for H_b, ux_b, uy_b in self.bin_geometry]
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
    n_bins: int | None = None,
    blend_px: float = 8.0,
) -> MeltInverseResult:
    r"""Fit basal melt to an observed surface anomaly through the forward operator.

    Minimizes ``||Forward(m) - dzs_obs||^2_mask + lam*||m||^2`` by Adam, with
    the Stubblefield transfer as a fixed differentiable forward layer.

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
    H_field, ux_field, uy_field, n_bins, blend_px :
        Set ``n_bins`` to use the spatially varying
        :class:`BlendedStubblefieldForward` instead of the single-``(H, u)``
        operator: the geometry fields are clustered into ``n_bins`` multipliers,
        each applied globally and blended over ``blend_px``. Required on real
        shelves, where one mean velocity is wrong nearly everywhere. Fields
        default to constants from the scalars (which reduces to the monolithic
        operator exactly).

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
        M_h = stubblefield_forward_multiplier(
            2 * ny, 2 * nx, dx, dy, H, ux_myr, uy_myr, **mult_kw)
        fwd = StubblefieldForward(M_h, ny, nx)
    else:
        def _fld(a, v):
            return np.full((ny, nx), float(v)) if a is None else np.asarray(a)
        fwd = BlendedStubblefieldForward(
            ny, nx, dx, dy, _fld(H_field, H), _fld(ux_field, ux_myr),
            _fld(uy_field, uy_myr), n_bins=int(n_bins), blend_px=blend_px,
            **mult_kw)
        if log_every:
            gh = np.array([g[0] for g in fwd.bin_geometry])
            gu = np.hypot([g[1] for g in fwd.bin_geometry],
                          [g[2] for g in fwd.bin_geometry])
            print(f"    blended operator: {fwd.n_bins} geometry bins  "
                  f"H {gh.min():.0f}-{gh.max():.0f} m  "
                  f"|u| {gu.min():.0f}-{gu.max():.0f} m/yr", flush=True)
    rep_net = (GridMelt(ny, nx) if rep == "grid"
               else SirenMelt(ny, nx, **(siren_kwargs or {})))

    dzs_t = torch.from_numpy(np.ascontiguousarray(dzs_obs, dtype=np.float64))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask, dtype=bool))
    opt = torch.optim.Adam(rep_net.parameters(), lr=lr)
    history: list = []
    for it in range(iters):
        opt.zero_grad()
        m = rep_net()
        pred = fwd(m)
        data = ((pred - dzs_t)[mask_t] ** 2).mean()
        reg = (m ** 2).mean()
        loss = data + lam * reg
        loss.backward()
        opt.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            history.append(float(data.item()))
            print(f"    it {it:5d}  data {data.item():.3e}  reg {reg.item():.3e}",
                  flush=True)

    with torch.no_grad():
        m_final = rep_net()
        melt = m_final.cpu().numpy()
        dzs_fit = fwd(m_final).cpu().numpy()
    return MeltInverseResult(melt=melt, loss_history=history, n_iter=iters,
                             dzs_fit=dzs_fit,
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
) -> xr.Dataset:
    r"""Basal melt rate from the surface anomaly by variational forward-fit.

    Driver-facing xarray wrapper around :func:`variational_melt_inverse`:
    strip-robust time-**median** surface -> high-pass control anomaly (remove the
    ``> sigma_hp_H * H_ref`` regional shape) -> fit melt through the Stubblefield
    forward operator on the floating bbox -> Shean sign (negative = melt).

    Unlike
    :func:`~stereo_melt.dynamics.stubblefield_inverse.stubblefield_inverse_melt_rate`
    (a direct Fourier division that over-lifts across-flow ridges), this puts the
    bridging transfer in the *forward* model and fits, so along-flow, oblique, and
    across-flow melt come through one operator with no angular weight.

    Set ``n_bins`` to run the spatially varying
    :class:`BlendedStubblefieldForward`, which clusters the shelf into that many
    thickness/velocity geometries and blends their globally applied responses.
    Without it the operator collapses the shelf to one ``H_ref`` and one mean
    ``(u0x, u0y)``, which is a synthetic-twin assumption -- on a real shelf
    spanning a large velocity range it is wrong nearly everywhere, so ``n_bins``
    is the appropriate setting for production. Either way the operator is
    DC-blind (it zeros ``k=0``): the recovered field is a channel-scale pattern
    correction, **not** a mass-budget melt, and is not interchangeable with the
    Eulerian/Lagrangian solvers.

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

    # crop to the floating bbox; build the DC-blind control anomaly
    ys, xs = np.where(fl)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    hm = h_med.values[y0:y1, x0:x1]
    fl_crop = fl[y0:y1, x0:x1]
    dzs = hm - _nan_gauss(hm, sigma_hp_H * H_ref / res)
    fit_mask = np.isfinite(dzs) & fl_crop
    dzs = np.where(np.isfinite(dzs), dzs, 0.0)

    # Local geometry for the blended operator: thickness and velocity on the same
    # crop, masked off the shelf so the geometry clustering is not pulled by open
    # ocean or grounded ice.
    H_crop = ux_crop = uy_crop = None
    if n_bins is not None:
        H_full = freeboard_to_thickness(h_med, d=d, rho_w=rho_w, rho_i=rho_i).values
        H_crop = np.where(fl_crop, H_full[y0:y1, x0:x1], np.nan)
        ux_crop = np.where(fl_crop, vxm.values[y0:y1, x0:x1], np.nan)
        uy_crop = np.where(fl_crop, vym.values[y0:y1, x0:x1], np.nan)

    result = variational_melt_inverse(
        dzs, fit_mask, res, res, H_ref, u0x, u0y,
        rep=rep, eta_bar=eta_bar, alpha_scale=alpha_scale, lam=lam,
        iters=iters, lr=lr, rho_i=rho_i, rho_w=rho_w, g=g,
        siren_kwargs=siren_kwargs, log_every=log_every,
        H_field=H_crop, ux_field=ux_crop, uy_field=uy_crop, n_bins=n_bins,
        blend_px=blend_km * 1000.0 / res)

    def _embed(crop: np.ndarray, name: str) -> xr.DataArray:
        """Place a bbox-cropped field back on the full grid, masked to the fit."""
        full = np.full(h_med.shape, np.nan, dtype=np.float64)
        full[y0:y1, x0:x1] = np.where(fit_mask, crop, np.nan)
        full[~fl] = np.nan
        return xr.DataArray(full, dims=h_med.dims, coords=h_med.coords, name=name)

    # Stubblefield m>0 = melt -> Shean negate; report only where fit
    melt = _embed(-result.melt, "melt_rate")

    # Fit quality: with no truth on a real shelf, the share of the observed
    # high-passed surface the operator reproduces is the only self-diagnostic.
    resid = (result.dzs_fit - dzs)[fit_mask]
    obs = dzs[fit_mask]
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))
    rms_obs = float(np.sqrt(np.mean(obs ** 2)))
    var_expl = float(1.0 - np.var(resid) / np.var(obs)) if np.var(obs) > 0 else np.nan

    return xr.Dataset(
        {
            "melt_rate": melt,
            "dzs_obs": _embed(dzs, "dzs_obs"),
            "dzs_fit": _embed(result.dzs_fit, "dzs_fit"),
        },
        attrs={
            "fit_rms_resid_m": rms_resid,
            "fit_rms_obs_m": rms_obs,
            "fit_var_explained": var_expl,
            "fit_n_cells": int(fit_mask.sum()),
            "equation": "Stubblefield 2023 forward-fit: min ||Forward(m)-dzs||^2 + lam||m||^2",
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "method": f"variational forward-operator inverse ({rep}); DC-blind channel correction",
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref, "t_r_yr": t_r_yr,
            "u0x_myr": u0x, "u0y_myr": u0y,
            "alpha_scale": alpha_scale, "lam": lam, "iters": iters, "rep": rep,
            "sigma_hp_H": sigma_hp_H,
            "n_bins": -1 if n_bins is None else int(result.n_bins_used),
            "blend_km": -1.0 if n_bins is None else float(blend_km),
            # aliases so this Dataset slots into run_melt's linv plot/save plumbing
            "gamma_dimless": 0.0, "tr_yr": t_r_yr,
            "note": (
                (f"blended operator ({result.n_bins_used} geometry bins, "
                 "globally applied)"
                 if n_bins else "first-order (single H, single mean u)")
                + "; pattern diagnostic, not mass-budget melt"),
        },
    )
