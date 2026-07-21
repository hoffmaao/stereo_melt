# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Streamline-frame physics-informed network for basal melt inversion.

A Lagrangian PINN that takes Lagrangian particle labels
:math:`(x_{gl}, y_{gl}, \tau)` — grounding-line crossing point and age
since crossing — and predicts the freeboard :math:`\hat h` at that
streamline coordinate. Basal melt rate is the autodiff residual of
Shean Eq. 10 in streamline coordinates:

.. math::

    \hat m(x_{gl}, y_{gl}, \tau)
        := \frac{\partial \hat H}{\partial \tau}
         + \hat H \, \nabla\!\cdot\mathbf{u}
         - \dot a

where :math:`\hat H` is the hydrostatic inversion of :math:`\hat h` and
forcings :math:`(\nabla\!\cdot\mathbf{u}, \dot a)` are sampled at the
observation pixel. Mass balance is exactly satisfied by construction —
no PDE-residual loss term, only data loss on :math:`\hat h`. Physics
enters through the architecture rather than a soft penalty: the
conservation law is embedded in the forward map, so the network only
has to fit the freeboard while mass balance holds automatically.

v1 assumptions:
- stationary melt rate (no absolute-time input; time-averaged forcings)
- hydrostatic only (Stubblefield non-hydrostatic correction deferred)
- static firn air content (no Dd/Dt term, per project_dhf_dt_gap)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import xarray as xr
from torch import nn

if TYPE_CHECKING:
    from ..kinematics import StreamlineDataset

__all__ = [
    "StreamlinePINN",
    "fit_streamline_pinn",
    "predict_melt_grid",
    "PINNFitResult",
]


# ----------------------------------------------------------------------
# Architecture: SiLU residual MLP with 3-D input (streamline labels),
# a single h_hat head, and an optional Fourier-feature embedding on τ.
# ----------------------------------------------------------------------


class _ResBlock(nn.Module):
    """SiLU residual block."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, h):
        return self.act(h + self.lin2(self.act(self.lin1(h))))


class _FourierFeatures(nn.Module):
    r"""Random Fourier features on a 1-D input (here: normalized τ).

    Maps a scalar :math:`\tau \in [0, 1]` to a ``2 * n_features``-dim
    vector :math:`[\sin(\omega_i \tau), \cos(\omega_i \tau)]`. The
    frequencies :math:`\omega_i` are fixed (not trained) so the
    embedding is a deterministic feature map, not a learned one. This
    fights the spectral bias an unembedded MLP exhibits along the time
    axis when fitting multi-year trajectories.
    """

    def __init__(self, n_features: int, sigma: float = 4.0) -> None:
        super().__init__()
        rng = np.random.default_rng(0)
        omega = rng.normal(scale=sigma * 2 * np.pi, size=n_features).astype(np.float32)
        self.register_buffer("omega", torch.from_numpy(omega))

    def forward(self, t):
        # t: (N,) or (N, 1) -> (N, 2 * n_features)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        proj = t * self.omega[None, :]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class StreamlinePINN(nn.Module):
    r"""Streamline-frame PINN: (x_gl, y_gl, τ) → h_hat.

    Predicts freeboard at the Lagrangian coordinate. Basal melt rate is
    recovered post-hoc via autodiff of the mass-balance equation; see
    :meth:`melt_from_h`.

    Inputs are normalized inside the network — pass raw meters and
    years; the model holds the normalization scales as buffers so they
    travel with the checkpoint.

    Per-strip h offset
    ------------------
    When ``n_strips > 0`` the model carries an additive offset
    :math:`\delta_t` per epoch (shape ``(n_strips,)``). During training
    we add :math:`\delta_{t_{obs}}` to ``h_hat`` so the loss can absorb
    constant-in-time coregistration biases. The offset is excluded at
    prediction time (its callers go through :meth:`forward` directly
    without strip_idx). Because :math:`\delta_t` does not depend on
    :math:`\tau`, it has zero contribution to :math:`\partial H/\partial \tau`
    and therefore cannot pollute the recovered melt rate.
    """

    def __init__(
        self,
        x_scale: tuple[float, float],
        y_scale: tuple[float, float],
        tau_max: float,
        hidden_dim: int = 256,
        n_hidden_layers: int = 6,
        n_fourier_features: int = 16,
        n_strips: int = 0,
    ) -> None:
        super().__init__()
        # Clamp ranges so degenerate single-line GLs (synthetic and some real
        # narrow tongues) don't divide-by-zero in _normalize. 1 m floor is
        # well below the ~250 m grid resolution we ever care about.
        x_min, x_max = float(x_scale[0]), float(x_scale[1])
        y_min, y_max = float(y_scale[0]), float(y_scale[1])
        if x_max - x_min < 1.0:
            x_max = x_min + 1.0
        if y_max - y_min < 1.0:
            y_max = y_min + 1.0
        self.register_buffer("x_min", torch.tensor(x_min, dtype=torch.float32))
        self.register_buffer("x_max", torch.tensor(x_max, dtype=torch.float32))
        self.register_buffer("y_min", torch.tensor(y_min, dtype=torch.float32))
        self.register_buffer("y_max", torch.tensor(y_max, dtype=torch.float32))
        self.register_buffer("tau_max", torch.tensor(max(float(tau_max), 1e-3), dtype=torch.float32))

        self.fourier = _FourierFeatures(n_fourier_features)
        in_dim = 2 + 2 * n_fourier_features  # (x, y) + sin/cos features on τ

        self.inp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        self.trunk = nn.ModuleList([_ResBlock(hidden_dim) for _ in range(n_hidden_layers)])
        self.head_h = nn.Linear(hidden_dim, 1)

        self.n_strips = int(n_strips)
        if self.n_strips > 0:
            self.strip_offset = nn.Parameter(torch.zeros(self.n_strips, dtype=torch.float32))
        else:
            self.register_parameter("strip_offset", None)

    def _normalize(self, x_gl, y_gl, tau):
        xn = 2.0 * (x_gl - self.x_min) / (self.x_max - self.x_min) - 1.0
        yn = 2.0 * (y_gl - self.y_min) / (self.y_max - self.y_min) - 1.0
        tn = tau / self.tau_max
        return xn, yn, tn

    def forward(self, x_gl, y_gl, tau, strip_idx=None):
        r"""Predict ``h_hat`` at the Lagrangian coordinate.

        If ``strip_idx`` is provided AND the model was built with
        ``n_strips > 0``, the per-strip offset is added to the output.
        Prediction-time callers (``predict_melt_grid``) leave strip_idx
        as None so they get the strip-mean h_hat.
        """
        xn, yn, tn = self._normalize(x_gl, y_gl, tau)
        # Stack spatial + Fourier(τ); inputs may arrive as (N,) or (N, 1).
        xn = xn.flatten()
        yn = yn.flatten()
        tn = tn.flatten()
        spatial = torch.stack([xn, yn], dim=-1)  # (N, 2)
        time_feat = self.fourier(tn)  # (N, 2 * n_fourier_features)
        z = torch.cat([spatial, time_feat], dim=-1)
        h = self.inp(z)
        for blk in self.trunk:
            h = blk(h)
        h_hat = self.head_h(h).squeeze(-1)
        if strip_idx is not None and self.strip_offset is not None:
            h_hat = h_hat + self.strip_offset[strip_idx]
        return h_hat

    def melt_from_h(
        self,
        x_gl,
        y_gl,
        tau,
        a_dot,
        vdiv,
        d_fac,
        rho_w: float = 1027.0,
        rho_i: float = 918.0,
    ):
        r"""Recover :math:`(\hat h, \hat H, \hat m)` at the given streamline coords.

        Uses ``torch.autograd.grad`` to compute :math:`\partial \hat H / \partial \tau`
        with the seed coords held fixed. The returned :math:`\hat m`
        satisfies Shean Eq. 10 by construction (Shean sign convention:
        positive = accretion, negative = melt).

        Inputs are (N,) tensors with ``requires_grad`` set on ``tau``.
        """
        if not tau.requires_grad:
            tau = tau.detach().clone().requires_grad_(True)
        h_hat = self(x_gl, y_gl, tau)
        # Hydrostatic inversion (static FAC, v1)
        H_hat = (rho_w / (rho_w - rho_i)) * (h_hat - d_fac) + d_fac
        # Material derivative in streamline coords ≡ ∂/∂τ
        (dH_dtau,) = torch.autograd.grad(
            outputs=H_hat.sum(),
            inputs=tau,
            create_graph=True,
        )
        # Shean Eq. 10 in Lagrangian form (Shean convention: positive = accretion)
        m_hat = dH_dtau + H_hat * vdiv - a_dot
        return h_hat, H_hat, m_hat


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


@dataclass
class PINNFitResult:
    """Container for a fitted model and its training diagnostics."""

    model: "StreamlinePINN"
    train_losses: list[float]
    test_losses: list[float]
    n_train: int
    n_test: int


def fit_streamline_pinn(
    dataset: "StreamlineDataset",
    *,
    hidden_dim: int = 256,
    n_hidden_layers: int = 6,
    n_fourier_features: int = 16,
    n_epochs: int = 15,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    test_frac: float = 0.2,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> PINNFitResult:
    r"""Train a :class:`StreamlinePINN` on a :class:`StreamlineDataset`.

    Loss: ``mean( ((h_hat - h_obs) / sigma) ** 2 )`` — Gaussian
    likelihood with per-observation noise. No PDE-residual term; mass
    balance is enforced by construction in :meth:`StreamlinePINN.melt_from_h`.

    Split: random ``test_frac`` of rows held out for a test-loss
    diagnostic. For spatial generalization use a held-out region
    upstream of the dataset (e.g. a basin sub-mask) instead.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Stack to (N, ...) float32 tensors. strip_idx stays int64.
    float_cols = {
        "x_gl": dataset.x_gl,
        "y_gl": dataset.y_gl,
        "tau": dataset.tau,
        "h_obs": dataset.h_obs,
        "sigma": dataset.sigma,
        "a_dot": dataset.a_dot,
        "vdiv": dataset.vdiv,
        "d_fac": dataset.d_fac,
    }
    n = len(dataset)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    def _tf(arr, idx):
        return torch.from_numpy(np.asarray(arr[idx], dtype=np.float32)).to(device)

    def _ti(arr, idx):
        return torch.from_numpy(np.asarray(arr[idx], dtype=np.int64)).to(device)

    tensors_train = {k: _tf(v, train_idx) for k, v in float_cols.items()}
    tensors_train["strip_idx"] = _ti(dataset.strip_idx, train_idx)
    tensors_test = {k: _tf(v, test_idx) for k, v in float_cols.items()}
    tensors_test["strip_idx"] = _ti(dataset.strip_idx, test_idx)

    model = StreamlinePINN(
        x_scale=(float(dataset.x_gl.min()), float(dataset.x_gl.max())),
        y_scale=(float(dataset.y_gl.min()), float(dataset.y_gl.max())),
        tau_max=float(dataset.tau.max()),
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_fourier_features=n_fourier_features,
        n_strips=int(dataset.n_strips),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _data_loss(t: dict, idx_slice=None) -> torch.Tensor:
        if idx_slice is None:
            x_gl = t["x_gl"]
            y_gl = t["y_gl"]
            tau = t["tau"]
            h_obs = t["h_obs"]
            sigma = t["sigma"]
            strip_idx = t["strip_idx"]
        else:
            x_gl = t["x_gl"][idx_slice]
            y_gl = t["y_gl"][idx_slice]
            tau = t["tau"][idx_slice]
            h_obs = t["h_obs"][idx_slice]
            sigma = t["sigma"][idx_slice]
            strip_idx = t["strip_idx"][idx_slice]
        h_hat = model(x_gl, y_gl, tau, strip_idx=strip_idx)
        return (((h_hat - h_obs) / sigma) ** 2).mean()

    n_train = train_idx.size
    train_losses: list[float] = []
    test_losses: list[float] = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        # Random minibatch ordering
        order = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        for start in range(0, n_train, batch_size):
            sl = order[start : start + batch_size]
            optim.zero_grad(set_to_none=True)
            loss = _data_loss(tensors_train, sl)
            loss.backward()
            optim.step()
            loss_sum += float(loss.item()) * sl.numel()
        train_losses.append(loss_sum / n_train)

        model.eval()
        with torch.no_grad():
            if n_test > 0:
                test_losses.append(float(_data_loss(tensors_test).item()))
            else:
                test_losses.append(float("nan"))
        if verbose:
            print(
                f"  epoch {epoch:3d}/{n_epochs}  "
                f"train_loss={train_losses[-1]:.4e}  "
                f"test_loss={test_losses[-1]:.4e}"
            )

    return PINNFitResult(
        model=model,
        train_losses=train_losses,
        test_losses=test_losses,
        n_train=n_train,
        n_test=n_test,
    )


# ----------------------------------------------------------------------
# Prediction onto an Eulerian (y, x) grid
# ----------------------------------------------------------------------


def predict_melt_grid(
    model: "StreamlinePINN",
    trajectories: dict[str, np.ndarray],
    a_dot: xr.DataArray,
    vdiv: xr.DataArray,
    d_fac: xr.DataArray,
    rho_w: float = 1027.0,
    rho_i: float = 918.0,
    device: str | None = None,
    batch_size: int = 65536,
) -> xr.Dataset:
    r"""Evaluate the trained PINN on an Eulerian grid.

    For each grid pixel where backward advection succeeded, evaluates
    the model at its ``(x_gl, y_gl, tau)`` label, computes
    :math:`(\hat h, \hat H, \hat m)` via :meth:`StreamlinePINN.melt_from_h`,
    and reshapes to (Y, X). Pixels whose backward trajectory failed
    (left domain or hit max_tau) are masked NaN.

    Returns a Dataset with variables ``h_hat``, ``H_hat``, ``melt_rate``,
    ``tau``, ``x_gl``, ``y_gl`` on the Eulerian (y, x) grid of ``a_dot``.
    """
    if device is None:
        device = next(model.parameters()).device.type

    ny, nx = trajectories["tau"].shape
    if a_dot.shape != (ny, nx):
        raise ValueError(f"a_dot shape {a_dot.shape} != trajectory grid ({ny}, {nx})")

    tau_flat = trajectories["tau"].ravel()
    x_gl_flat = trajectories["x_gl"].ravel()
    y_gl_flat = trajectories["y_gl"].ravel()
    a_dot_flat = a_dot.values.ravel()
    vdiv_flat = vdiv.values.ravel()
    d_fac_flat = d_fac.values.ravel()

    valid = np.isfinite(tau_flat) & np.isfinite(a_dot_flat) & np.isfinite(vdiv_flat)
    n_valid = int(valid.sum())

    h_hat_out = np.full(tau_flat.shape, np.nan, dtype=np.float64)
    H_hat_out = np.full(tau_flat.shape, np.nan, dtype=np.float64)
    m_out = np.full(tau_flat.shape, np.nan, dtype=np.float64)

    if n_valid == 0:
        # Nothing to do; return NaN fields
        ds = xr.Dataset(
            {
                "h_hat": (("y", "x"), h_hat_out.reshape(ny, nx)),
                "H_hat": (("y", "x"), H_hat_out.reshape(ny, nx)),
                "melt_rate": (("y", "x"), m_out.reshape(ny, nx)),
                "tau": (("y", "x"), trajectories["tau"]),
                "x_gl": (("y", "x"), trajectories["x_gl"]),
                "y_gl": (("y", "x"), trajectories["y_gl"]),
            },
            coords={"y": a_dot["y"].values, "x": a_dot["x"].values},
            attrs={"convention": "Shean (positive=accretion, negative=melt)"},
        )
        return ds

    idx_valid = np.where(valid)[0]
    model.eval()
    # We need grad to flow through τ even at eval — torch.no_grad would kill it.
    for start in range(0, n_valid, batch_size):
        sl = idx_valid[start : start + batch_size]
        x_gl_b = torch.from_numpy(x_gl_flat[sl].astype(np.float32)).to(device)
        y_gl_b = torch.from_numpy(y_gl_flat[sl].astype(np.float32)).to(device)
        tau_b = torch.from_numpy(tau_flat[sl].astype(np.float32)).to(device).requires_grad_(True)
        a_b = torch.from_numpy(a_dot_flat[sl].astype(np.float32)).to(device)
        v_b = torch.from_numpy(vdiv_flat[sl].astype(np.float32)).to(device)
        d_b = torch.from_numpy(d_fac_flat[sl].astype(np.float32)).to(device)

        h_hat, H_hat, m_hat = model.melt_from_h(
            x_gl_b, y_gl_b, tau_b, a_b, v_b, d_b, rho_w=rho_w, rho_i=rho_i
        )
        h_hat_out[sl] = h_hat.detach().cpu().numpy().astype(np.float64)
        H_hat_out[sl] = H_hat.detach().cpu().numpy().astype(np.float64)
        m_out[sl] = m_hat.detach().cpu().numpy().astype(np.float64)

    return xr.Dataset(
        {
            "h_hat": (("y", "x"), h_hat_out.reshape(ny, nx)),
            "H_hat": (("y", "x"), H_hat_out.reshape(ny, nx)),
            "melt_rate": (("y", "x"), m_out.reshape(ny, nx)),
            "tau": (("y", "x"), trajectories["tau"]),
            "x_gl": (("y", "x"), trajectories["x_gl"]),
            "y_gl": (("y", "x"), trajectories["y_gl"]),
        },
        coords={"y": a_dot["y"].values, "x": a_dot["x"].values},
        attrs={"convention": "Shean (positive=accretion, negative=melt)"},
    )
