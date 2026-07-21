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
import torch

from ..constants import rhoi, rhow
from .linear_perturbation import (
    G_GRAVITY,
    SECONDS_PER_YEAR,
    LinearPerturbation,
    _wavenumber_grids,
)

__all__ = [
    "stubblefield_forward_multiplier",
    "StubblefieldForward",
    "GridMelt",
    "SirenMelt",
    "variational_melt_inverse",
    "MeltInverseResult",
]


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

    melt: np.ndarray          # (ny, nx) recovered basal melt, m/yr
    loss_history: list        # data-misfit term per recorded iteration
    n_iter: int


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

    M_h = stubblefield_forward_multiplier(
        2 * ny, 2 * nx, dx, dy, H, ux_myr, uy_myr,
        eta_bar=eta_bar, alpha_scale=alpha_scale,
        rho_i=rho_i, rho_w=rho_w, g=g)
    fwd = StubblefieldForward(M_h, ny, nx)
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

    melt = rep_net().detach().cpu().numpy()
    return MeltInverseResult(melt=melt, loss_history=history, n_iter=iters)
