# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.
r"""Bayesian physics-informed space-time assimilation of a DEM stack (JAX).

An Eulerian B-PINN (Yang, Meng & Karniadakis 2021) for the ice-shelf mass
budget. Two networks: a space-time thickness surrogate

.. math::  H_\theta(x, y, t) = H_0 + s_H\,\mathrm{MLP}_\theta(\phi(x, y, t))

fitted to every finite pixel of every epoch of the (tilt-corrected,
hydrostatic) thickness stack, and a melt-rate field

.. math::  \dot b_\varphi(x, y) = s_b\,\mathrm{MLP}_\varphi(\phi(x, y))

(steady over the window; Shean sign, negative = melt). With ``base_field`` (default) the surrogate is
:math:`H_\theta = H_{\rm base}(x, y) + s_H\,\mathrm{MLP}_\theta`, where
:math:`H_{\rm base}` is the lightly smoothed per-pixel temporal median of the
observations sampled bilinearly (differentiable), so the network only has to
carry anomalies and the time evolution rather than the hundreds of metres of
static structure of a real trunk. Both take random
Fourier features at several wavelength bands so the surrogate can carry
kilometre-scale structure advecting at kilometres per year. The physics
enters as a likelihood on the residual of the thickness equation at
collocation points :math:`(x_c, y_c, t_c)` drawn over the domain and window,

.. math::  r = \partial_t H_\theta + H_\theta\,\nabla\!\cdot\mathbf u
              + \mathbf u\cdot\nabla H_\theta - \dot a - \dot b_\varphi
              \sim \mathcal N(0, \sigma_r^2),

with the surrogate's derivatives by autodiff and the velocity, its
divergence and the SMB sampled bilinearly from their grids (velocity may be
time-varying). Observations enter as Gaussian or Student-t likelihoods with
scale ``sigma_h_m``, after subtracting a per-epoch plane
:math:`\alpha_{z,k} + \alpha_{x,k}(x - x_c) + \alpha_{y,k}(y - y_c)` of nuisance
parameters (``epoch_planes``; Gaussian priors ``plane_prior_z_m`` and
``plane_prior_xy_m_per_km``), because strip datum and tilt errors are the
dominant, per-epoch, spatially coherent part of the noise and a surrogate
smooth in time cannot represent them. Weights carry Gaussian priors. The posterior is
explored three ways, from cheap to faithful:

1. MAP (Adam) - the point estimate;
2. an ensemble of MAP fits over seeds with an epoch bootstrap (resampling
   whole DEMs, because strip errors are per-epoch and spatially coherent);
3. NUTS (blackjax) over the melt network with the surrogate held at MAP
   (``hmc_samples > 0``), or jointly over both networks (``hmc_joint``).

Why Eulerian space-time rather than a per-pixel trend: the Eulerian budget
solver differences a 14-year per-pixel trend against :math:`\nabla\cdot(H\mathbf u)`
evaluated on the time-mean thickness, so channels and keels advected through
a pixel leave an uncancelled :math:`\mathbf u\cdot\nabla H` that reads as
melt/accretion dipoles (2026-09-06 accretion diagnosis). A surrogate that is
smooth in :math:`t` but resolves the moving features evaluates
:math:`\partial_t H` and :math:`\mathbf u\cdot\nabla H` consistently at the
same point in space-time, so the advection cancels where it should.

With ``transfer=True`` the surrogate is the TRUE thickness and the observation
operator carries the Stubblefield bridging transfer: for each observed epoch
the surrogate is evaluated on the grid, its least-squares plane is removed,
the field is reflect-padded to twice its size and multiplied in the FFT domain
by :math:`M(k) = 1 + (T(k) - 1)\,(1 - L(k))` — ``T`` from
:func:`~stereo_melt.dynamics.bridging_restoration._bridging_transfer` (the
same advected Newtonian transfer the monolithic and restored-budget solvers
use; ``eta_bar``, ``alpha_scale``, ``H_ref_m``, ``u_ref_myr``, the last
defaulting to the mean speed over ``data.domain`` — time-averaged first for a
time-varying velocity — because a whole-grid mean of the gap-filled velocity
biases it low, and with it the along-flow asymmetry of ``M``, on any window
with velocity gaps) and ``L`` a
Gaussian low-pass at ``transfer_bg_sigma_H`` ice thicknesses that keeps the
operator at unity on the long wavelengths where ``T`` is unity anyway — then
cropped and compared to the hydrostatic thickness observations. Data
mini-batches are then whole epochs (``batch_epochs``). Without the transfer
(the default) a hydrostatic budget can only see :math:`|T|` of channel-scale
melt (0.06 at 2H, 0.34 at 3H on the PIGREAL twin, 2026-09-12), which is why
it matched the Eulerian ceiling there.

JAX, optax and blackjax are imported lazily; the module imports without them.
Validate on the DEM-stack twins (truth known) before any real basin:
``examples/elmer_synth/scripts/run_bpinn_twin.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BPINNConfig", "BPINNData", "BPINNResult", "prepare_bpinn_data", "fit_bpinn"]


@dataclass
class BPINNConfig:
    hidden: int = 128
    layers: int = 4
    n_fourier: int = 32
    xy_scales_km: tuple = (0.5, 1.0, 2.0, 4.0, 8.0)
    t_scales_yr: tuple = (0.5, 1.0, 2.0, 5.0)
    melt_hidden: int = 64
    melt_layers: int = 3
    melt_scales_km: tuple = (1.0, 2.0, 4.0, 8.0)
    H_scale_m: float = 50.0
    base_field: bool = True
    base_smooth_px: float = 2.0
    b_scale_myr: float = 10.0
    sigma_h_m: float = 2.0
    nu: float | None = 4.0
    sigma_r_myr: float = 1.0
    n_col_slices: int = 24
    weight_prior_sd: float = 1.0
    n_steps: int = 20000
    batch_obs: int = 8192
    batch_col: int = 8192
    lr: float = 1e-3
    grad_clip: float = 10.0
    seed: int = 0
    ensemble: int = 1
    bootstrap_epochs: bool = True
    hmc_samples: int = 0
    hmc_warmup: int = 200
    hmc_joint: bool = False
    log_every: int = 1000
    epoch_planes: bool = True
    plane_prior_z_m: float = 20.0
    plane_prior_xy_m_per_km: float = 1.0
    transfer: bool = False
    eta_bar: float = 1e14
    alpha_scale: float = 0.34
    transfer_bg_sigma_H: float = 3.0
    batch_epochs: int = 16
    H_ref_m: float | None = None
    u_ref_myr: float | None = None


@dataclass
class BPINNData:
    """Normalised, flattened inputs (see :func:`prepare_bpinn_data`)."""
    x_km: np.ndarray
    y_km: np.ndarray
    t_yr: np.ndarray
    H_obs: np.ndarray            # (K, ny, nx), NaN where absent
    epoch_of_obs: np.ndarray     # (N,) epoch index of each finite observation
    obs_xyt: np.ndarray          # (N, 3) km, km, yr
    obs_H: np.ndarray            # (N,)
    vx: np.ndarray               # (K or 1, ny, nx) km/yr
    vy: np.ndarray
    divu: np.ndarray             # (K or 1, ny, nx) 1/yr
    a_dot: np.ndarray            # (ny, nx) m/yr
    domain: np.ndarray           # (ny, nx) bool: where collocation points live
    H0: float
    extras: dict = field(default_factory=dict)


@dataclass
class BPINNResult:
    melt_mean: np.ndarray
    melt_sd: np.ndarray
    samples: np.ndarray          # (S, ny, nx)
    map_melt: np.ndarray
    obs_rms_m: float
    loss_history: np.ndarray
    config: BPINNConfig
    extras: dict = field(default_factory=dict)


def prepare_bpinn_data(H_obs, x, y, t_yr, vx, vy, a_dot=None, domain=None,
                       rho_i: float = 917.0, rho_w: float = 1027.0, vt_yr=None) -> BPINNData:
    """Flatten a thickness stack and its forcings for :func:`fit_bpinn`.

    ``H_obs`` (K, ny, nx) hydrostatic thickness in metres, NaN where absent;
    ``x``/``y`` in metres (y descending, north-up); ``t_yr`` decimal years;
    ``vx``/``vy`` in m/yr, 2-D (static) or 3-D (K, ny, nx, time-varying);
    ``a_dot`` surface mass balance in m ice/yr (default 0); ``domain`` bool
    (ny, nx) where the physics is enforced (default: any finite observation),
    intersected either way with the pixels where the velocity and its centred
    divergence stencil are finite, so a pixel next to a velocity gap does not
    carry the one-sided difference against a filled zero;
    ``vt_yr`` the decimal-year axis of a time-varying velocity (else the
    velocity epochs are assumed to span the stack window uniformly).
    """
    H_obs = np.asarray(H_obs, float)
    K, ny, nx = H_obs.shape
    x_km = np.asarray(x, float) / 1e3
    y_km = np.asarray(y, float) / 1e3
    t_yr = np.asarray(t_yr, float)
    vx = np.asarray(vx, float) / 1e3
    vy = np.asarray(vy, float) / 1e3
    if vx.ndim == 2:
        vx, vy = vx[None], vy[None]
    dx = float(x_km[1] - x_km[0])
    dy = float(y_km[1] - y_km[0])          # negative for a north-up grid
    divu = np.gradient(vx, dx, axis=2) + np.gradient(vy, dy, axis=1)
    a_dot = np.zeros((ny, nx)) if a_dot is None else np.asarray(a_dot, float)
    fin = np.isfinite(H_obs)
    if domain is None:
        domain = fin.any(axis=0)
    domain = (np.asarray(domain, bool) & np.isfinite(vx).all(axis=0)
              & np.isfinite(vy).all(axis=0) & np.isfinite(divu).all(axis=0))
    k, j, i = np.nonzero(fin & domain[None])
    obs_xyt = np.stack([x_km[i], y_km[j], t_yr[k]], axis=1)
    H0 = float(np.nanmedian(H_obs))
    return BPINNData(x_km, y_km, t_yr, H_obs, k, obs_xyt, H_obs[k, j, i],
                     np.nan_to_num(vx), np.nan_to_num(vy), np.nan_to_num(divu),
                     np.nan_to_num(a_dot), domain, H0,
                     {"rho_i": float(rho_i), "rho_w": float(rho_w), "dx_km": dx,
                      "vt_yr": None if vt_yr is None else np.asarray(vt_yr, float)})


# ----------------------------------------------------------------------
# networks (pure JAX pytrees)
# ----------------------------------------------------------------------
def _init_mlp(key, in_dim, hidden, layers, out_dim):
    import jax
    import jax.numpy as jnp
    dims = [in_dim] + [hidden] * layers + [out_dim]
    params = []
    for a, b in zip(dims[:-1], dims[1:]):
        key, sub = jax.random.split(key)
        params.append({"W": jax.random.normal(sub, (a, b)) * jnp.sqrt(2.0 / a), "b": jnp.zeros((b,))})
    params[-1]["W"] = params[-1]["W"] * 0.1
    return params


def _mlp(params, h):
    import jax
    for layer in params[:-1]:
        h = jax.nn.silu(h @ layer["W"] + layer["b"])
    return h @ params[-1]["W"] + params[-1]["b"]


def _fourier_matrix(key, in_dim, n_per_scale, scales):
    """Rows of B ~ N(0, 1/λ²) per wavelength band λ, so features ~ sin(2π B·u)."""
    import jax
    import jax.numpy as jnp
    mats = []
    for lam in scales:
        key, sub = jax.random.split(key)
        mats.append(jax.random.normal(sub, (in_dim, n_per_scale)) / lam)
    return jnp.concatenate(mats, axis=1)


def _features(B, u):
    import jax.numpy as jnp
    z = 2.0 * jnp.pi * (u @ B)
    return jnp.concatenate([jnp.sin(z), jnp.cos(z)], axis=-1)


def fit_bpinn(data: BPINNData, cfg: BPINNConfig | None = None, truth=None) -> BPINNResult:
    """Fit the space-time B-PINN; see the module docstring for the model."""
    import jax
    import jax.numpy as jnp
    import optax

    cfg = cfg or BPINNConfig()
    key = jax.random.PRNGKey(cfg.seed)
    ny, nx = data.domain.shape
    K = data.H_obs.shape[0]
    x0, x1 = float(data.x_km[0]), float(data.x_km[-1])
    y0, y1 = float(data.y_km[-1]), float(data.y_km[0])     # y descending → (min, max)
    t0, t1 = float(data.t_yr.min()), float(data.t_yr.max())
    # grids for bilinear sampling: pixel index space
    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)                               # positive spacing (y1 = max)
    vx_g, vy_g, divu_g = jnp.asarray(data.vx), jnp.asarray(data.vy), jnp.asarray(data.divu)
    a_g = jnp.asarray(data.a_dot)
    tv = vx_g.shape[0] > 1

    def _pix(x, y):
        return (y1 - y) / dy, (x - x0) / dx                # row, col (fractional)

    def _sample2(g, x, y):
        from jax.scipy.ndimage import map_coordinates
        r, c = _pix(x, y)
        return map_coordinates(g, [r, c], order=1, mode="nearest")

    vt = data.extras.get("vt_yr")
    vt_j = None if vt is None else jnp.asarray(vt)

    def _sample3(g, x, y, t):
        from jax.scipy.ndimage import map_coordinates
        r, c = _pix(x, y)
        if not tv:
            return map_coordinates(g[0], [r, c], order=1, mode="nearest")
        if vt_j is not None:
            ti = jnp.interp(t, vt_j, jnp.arange(g.shape[0], dtype=float))
        else:
            ti = (t - t0) / max(t1 - t0, 1e-9) * (g.shape[0] - 1)
        return map_coordinates(g, [ti, r, c], order=1, mode="nearest")

    # feature matrices (fixed) and parameters
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    B_xy = _fourier_matrix(k1, 2, cfg.n_fourier, cfg.xy_scales_km)
    B_t = _fourier_matrix(k2, 1, cfg.n_fourier, cfg.t_scales_yr)
    B_b = _fourier_matrix(k3, 2, cfg.n_fourier, cfg.melt_scales_km)
    in_H = 2 * B_xy.shape[1] + 2 * B_t.shape[1] + 3
    in_b = 2 * B_b.shape[1] + 2
    xc, yc, tc = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (t0 + t1)
    Lx, Ly, Lt = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9), max(t1 - t0, 1e-9)

    if cfg.base_field:
        from scipy import ndimage as _ndi
        med = np.nanmedian(data.H_obs, axis=0)
        fin = np.isfinite(med)
        filled = np.where(fin, med, data.H0)
        if fin.any() and not fin.all():
            idx = _ndi.distance_transform_edt(~fin, return_distances=False, return_indices=True)
            filled = filled[tuple(idx)]
        if cfg.base_smooth_px > 0:
            filled = _ndi.gaussian_filter(filled, cfg.base_smooth_px)
        base_g = jnp.asarray(filled)
        print(f"  [bpinn] base field: temporal median, {int(fin.sum())} px, range {np.nanmin(med):.0f}..{np.nanmax(med):.0f} m, "
              f"smoothed {cfg.base_smooth_px:g} px; network carries anomalies (scale {cfg.H_scale_m:g} m)", flush=True)
    else:
        base_g = None

    def H_net(theta, x, y, t):
        u_km = jnp.stack([x - xc, y - yc], -1)                 # physical offsets (km)
        u = jnp.stack([(x - xc) / Lx, (y - yc) / Ly], -1)      # normalised, linear inputs
        f = jnp.concatenate([_features(B_xy, u_km), _features(B_t, (t - tc)[..., None]),
                             u, ((t - tc) / Lt)[..., None]], -1)
        base = _sample2(base_g, x, y) if base_g is not None else data.H0
        return base + cfg.H_scale_m * _mlp(theta, f)[..., 0]

    def b_net(phi, x, y):
        u_km = jnp.stack([x - xc, y - yc], -1)
        u = jnp.stack([(x - xc) / Lx, (y - yc) / Ly], -1)
        f = jnp.concatenate([_features(B_b, u_km), u], -1)
        return cfg.b_scale_myr * _mlp(phi, f)[..., 0]

    def residual_parts(theta, phi, x, y, t):
        H_scalar = lambda xx, yy, tt: H_net(theta, xx, yy, tt)  # noqa: E731
        H = H_scalar(x, y, t)
        Hx = jax.vmap(jax.grad(H_scalar, 0))(x, y, t)
        Hy = jax.vmap(jax.grad(H_scalar, 1))(x, y, t)
        Ht = jax.vmap(jax.grad(H_scalar, 2))(x, y, t)
        u, v = _sample3(vx_g, x, y, t), _sample3(vy_g, x, y, t)
        return dict(H=H, Hx=Hx, Hy=Hy, Ht=Ht, u=u, v=v, divu=_sample3(divu_g, x, y, t), a=_sample2(a_g, x, y), b=b_net(phi, x, y))

    def residual(theta, phi, x, y, t):
        """Thickness-equation residual at points (km, km, yr) → m/yr."""
        H_scalar = lambda xx, yy, tt: H_net(theta, xx, yy, tt)  # noqa: E731
        H = H_scalar(x, y, t)
        Hx = jax.vmap(jax.grad(H_scalar, 0))(x, y, t)          # m/km
        Hy = jax.vmap(jax.grad(H_scalar, 1))(x, y, t)
        Ht = jax.vmap(jax.grad(H_scalar, 2))(x, y, t)          # m/yr
        u, v = _sample3(vx_g, x, y, t), _sample3(vy_g, x, y, t)
        return Ht + H * _sample3(divu_g, x, y, t) + u * Hx + v * Hy - _sample2(a_g, x, y) - b_net(phi, x, y)

    # ---- bridging transfer as the observation operator (optional) ----
    if cfg.transfer:
        from .bridging_restoration import _bridging_transfer
        H_ref = cfg.H_ref_m if cfg.H_ref_m is not None else float(data.H0)
        if cfg.u_ref_myr is not None:
            u_ref = float(cfg.u_ref_myr)
        else:
            spd = np.hypot(data.vx, data.vy).mean(axis=0)[data.domain]
            if spd.size == 0:
                raise ValueError("empty domain: cannot default u_ref_myr, pass BPINNConfig(u_ref_myr=...)")
            u_ref = float(spd.mean() * 1e3)
            print(f"  [bpinn] u_ref default: mean speed over {spd.size} domain px = {u_ref:.0f} m/yr", flush=True)
        Py, Px = 2 * ny, 2 * nx
        T_np = _bridging_transfer(Py, Px, dx * 1e3, dy * 1e3, H_ref, u_ref, 0.0, cfg.eta_bar, cfg.alpha_scale,
                                  data.extras.get("rho_i", 917.0), data.extras.get("rho_w", 1027.0), 9.81, 0.0, 0.0)[0]
        kxp = 2 * np.pi * np.fft.fftfreq(Px, dx * 1e3)
        kyp = 2 * np.pi * np.fft.fftfreq(Py, dy * 1e3)
        K2 = kxp[None, :] ** 2 + kyp[:, None] ** 2
        sig_bg = cfg.transfer_bg_sigma_H * H_ref
        L_np = np.exp(-0.5 * K2 * sig_bg ** 2)
        M_op = jnp.asarray(1.0 + (T_np - 1.0) * (1.0 - L_np))
        Xg, Yg = np.meshgrid(data.x_km, data.y_km)
        Xd = np.stack([np.ones(ny * nx), (Xg.ravel() - xc), (Yg.ravel() - yc)], 1)
        P_pinv = jnp.asarray(np.linalg.pinv(Xd))          # (3, ny*nx)
        Xd_j = jnp.asarray(Xd)
        i_t2 = np.argmin(np.abs(kxp - 2 * np.pi / (2 * H_ref)))
        i_t3 = np.argmin(np.abs(kxp - 2 * np.pi / (3 * H_ref)))
        print(f"  [bpinn] transfer ON: H_ref {H_ref:.0f} m, u_ref {u_ref:.0f} m/yr, eta {cfg.eta_bar:.1e}, "
              f"alpha {cfg.alpha_scale}; |T| along-flow at 2H/3H = {abs(T_np[0, i_t2]):.2f}/{abs(T_np[0, i_t3]):.2f}; "
              f"background low-pass sigma {sig_bg/1e3:.1f} km; padded FFT {Py}x{Px}", flush=True)
        H_dense = jnp.asarray(np.nan_to_num(data.H_obs))
        M_dense = jnp.asarray(np.isfinite(data.H_obs) & data.domain[None])
        gxj, gyj = jnp.asarray(Xg.ravel()), jnp.asarray(Yg.ravel())
        t_epochs = jnp.asarray(data.t_yr)

        def apparent_epoch(theta, tk):
            """Surrogate on the grid at epoch time tk → hydrostatic-apparent thickness (ny, nx)."""
            h = H_net(theta, gxj, gyj, jnp.full_like(gxj, tk))          # (ny*nx,)
            coef = P_pinv @ h
            pl = Xd_j @ coef
            a = (h - pl).reshape(ny, nx)
            ap = jnp.pad(a, ((0, ny), (0, nx)), mode="reflect")
            out = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(ap) * M_op))[:ny, :nx]
            return out + pl.reshape(ny, nx)

        def apparent_many(theta, ts):
            """Sequential (memory-safe) evaluation over many epochs → (n, ny, nx)."""
            return jax.lax.map(lambda tk: apparent_epoch(theta, tk), ts)

        def nll_obs_epochs(theta, ks, pl=None):
            pred = jax.vmap(lambda tk: apparent_epoch(theta, tk))(t_epochs[ks])   # (B, ny, nx)
            if pl is not None:
                pred = pred + (pl["z"][ks][:, None, None] + pl["x"][ks][:, None, None] * (Xg_j - xc)
                               + pl["y"][ks][:, None, None] * (Yg_j - yc))
            z = (H_dense[ks] - pred) / cfg.sigma_h_m
            if cfg.nu is None:
                per = 0.5 * z ** 2
            else:
                per = 0.5 * (cfg.nu + 1.0) * jnp.log1p(z ** 2 / cfg.nu)
            return jnp.sum(jnp.where(M_dense[ks], per, 0.0), axis=(1, 2))   # (B,) per-epoch sums

        Xg_j, Yg_j = jnp.asarray(Xg), jnp.asarray(Yg)
        n_fin_k = jnp.asarray((np.isfinite(data.H_obs) & data.domain[None]).sum(axis=(1, 2)), float)

        def nll_obs_epochs_seq(theta, pl=None):
            """Full-epoch data likelihood with sequential epoch evaluation (HMC)."""
            def one(carry, k):
                pred = apparent_epoch(theta, t_epochs[k])
                if pl is not None:
                    pred = pred + pl["z"][k] + pl["x"][k] * (Xg_j - xc) + pl["y"][k] * (Yg_j - yc)
                z = (H_dense[k] - pred) / cfg.sigma_h_m
                per = 0.5 * z ** 2 if cfg.nu is None else 0.5 * (cfg.nu + 1.0) * jnp.log1p(z ** 2 / cfg.nu)
                return carry + jnp.sum(jnp.where(M_dense[k], per, 0.0)), None
            total, _ = jax.lax.scan(one, 0.0, jnp.arange(K))
            return total

    # observation / collocation pools
    obs_xyt = jnp.asarray(data.obs_xyt)
    obs_H = jnp.asarray(data.obs_H)
    obs_k = jnp.asarray(data.epoch_of_obs)
    N_obs = obs_H.shape[0]
    dom_j, dom_i = np.nonzero(data.domain)
    dom_xy = jnp.asarray(np.stack([data.x_km[dom_i], data.y_km[dom_j]], 1))
    N_col = dom_xy.shape[0] * cfg.n_col_slices           # nominal number of physics "observations"

    def plane(pl, k, x, y):
        if pl is None:
            return 0.0
        return pl["z"][k] + pl["x"][k] * (x - xc) + pl["y"][k] * (y - yc)

    def nll_obs(theta, xyt, H, pl=None, k=None):
        z = (H - H_net(theta, xyt[:, 0], xyt[:, 1], xyt[:, 2]) - plane(pl, k, xyt[:, 0], xyt[:, 1])) / cfg.sigma_h_m
        if cfg.nu is None:
            return 0.5 * jnp.sum(z ** 2)
        return 0.5 * (cfg.nu + 1.0) * jnp.sum(jnp.log1p(z ** 2 / cfg.nu))

    def nll_col(theta, phi, xy, t):
        r = residual(theta, phi, xy[:, 0], xy[:, 1], t) / cfg.sigma_r_myr
        return 0.5 * jnp.sum(r ** 2)

    def log_prior(params):
        lp = 0.0
        for name, sub in params.items():
            if name == "planes":
                lp += -0.5 * (jnp.sum(sub["z"] ** 2) / cfg.plane_prior_z_m ** 2
                              + (jnp.sum(sub["x"] ** 2) + jnp.sum(sub["y"] ** 2)) / cfg.plane_prior_xy_m_per_km ** 2)
            else:
                lp += -0.5 * sum(jnp.sum(w ** 2) for w in jax.tree_util.tree_leaves(sub)) / cfg.weight_prior_sd ** 2
        return lp

    def loss_fn(params, key, obs_idx, epoch_w):
        theta, phi = params["theta"], params["phi"]
        k_o, k_c, k_t = jax.random.split(key, 3)
        ic = jax.random.choice(k_c, dom_xy.shape[0], (cfg.batch_col,))
        xy = dom_xy[ic] + jax.random.uniform(k_c, (cfg.batch_col, 2), minval=-0.5, maxval=0.5) * jnp.array([dx, dy])
        tcol = jax.random.uniform(k_t, (cfg.batch_col,), minval=t0, maxval=t1)
        if cfg.transfer:
            # Importance sampling: draw epochs with probability ∝ (bootstrap weight × finite
            # pixel count); the estimator Σ_k w_k L_k ≈ mean_b[w_kb L_kb / p_kb] is unbiased and
            # low-variance because L_k ∝ n_k ∝ p_k (PIG windows: 3 % finite, wildly uneven epochs).
            pk = epoch_w * n_fin_k
            pk = pk / jnp.sum(pk)
            ks = jax.random.choice(k_o, K, (cfg.batch_epochs,), p=pk)
            lk = nll_obs_epochs(theta, ks, params.get("planes"))
            lo = jnp.mean(epoch_w[ks] * lk / jnp.maximum(pk[ks], 1e-12))
        else:
            io = jax.random.choice(k_o, obs_idx, (cfg.batch_obs,))
            lo = nll_obs(theta, obs_xyt[io], obs_H[io], params.get("planes"), obs_k[io]) * (obs_idx.shape[0] / cfg.batch_obs)
        lc = nll_col(theta, phi, xy, tcol) * (N_col / cfg.batch_col)
        return lo + lc - log_prior(params), (lo, lc)

    def _fit_map(seed, obs_idx, epoch_w):
        k = jax.random.PRNGKey(seed)
        k, ka, kb = jax.random.split(k, 3)
        params = {"theta": _init_mlp(ka, in_H, cfg.hidden, cfg.layers, 1),
                  "phi": _init_mlp(kb, in_b, cfg.melt_hidden, cfg.melt_layers, 1)}
        if cfg.epoch_planes:
            params["planes"] = {"z": jnp.zeros((K,)), "x": jnp.zeros((K,)), "y": jnp.zeros((K,))}
        sched = optax.cosine_decay_schedule(cfg.lr, cfg.n_steps, alpha=0.02)
        opt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(sched))
        state = opt.init(params)

        @jax.jit
        def step(params, state, key):
            (loss, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(params, key, obs_idx, epoch_w)
            upd, state = opt.update(g, state, params)
            return optax.apply_updates(params, upd), state, loss, parts

        hist = []
        for it in range(cfg.n_steps):
            k, sub = jax.random.split(k)
            params, state, loss, parts = step(params, state, sub)
            if it % cfg.log_every == 0 or it == cfg.n_steps - 1:
                hist.append([it, float(loss), float(parts[0]), float(parts[1])])
                print(f"    step {it:6d}  loss {float(loss):.4e}  obs {float(parts[0]):.4e}  phys {float(parts[1]):.4e}", flush=True)
        return params, np.array(hist)

    X, Y = np.meshgrid(data.x_km, data.y_km)
    gx, gy = jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())
    eval_b = jax.jit(lambda phi: b_net(phi, gx, gy).reshape(ny, nx))
    eval_H = jax.jit(lambda theta, xyt: H_net(theta, xyt[:, 0], xyt[:, 1], xyt[:, 2]))

    if cfg.n_steps <= 0:   # debug hook: residual parts at init
        kd = jax.random.PRNGKey(0)
        params = {"theta": _init_mlp(kd, in_H, cfg.hidden, cfg.layers, 1), "phi": _init_mlp(kd, in_b, cfg.melt_hidden, cfg.melt_layers, 1)}
        xy = dom_xy[:512]
        parts = residual_parts(params["theta"], params["phi"], xy[:, 0], xy[:, 1], jnp.full((512,), tc))
        return {k: np.asarray(v) for k, v in parts.items()}

    samples, hists = [], []
    all_idx = jnp.arange(N_obs)
    epoch_of_obs = np.asarray(data.epoch_of_obs)
    for m in range(max(cfg.ensemble, 1)):
        seed = cfg.seed + m
        if cfg.bootstrap_epochs and m > 0:
            rng = np.random.default_rng(seed)
            keep = rng.choice(K, K, replace=True)
            counts = np.bincount(keep, minlength=K)
            idx = np.repeat(np.arange(N_obs), counts[epoch_of_obs])   # epoch multiplicity
            obs_idx = jnp.asarray(idx)
            epoch_w = jnp.asarray(counts, float)
        else:
            obs_idx = all_idx
            epoch_w = jnp.ones((K,))
        print(f"  [bpinn] MAP fit {m + 1}/{cfg.ensemble} seed={seed} obs={int(obs_idx.shape[0])} "
              f"{'epochs/step=' + str(cfg.batch_epochs) if cfg.transfer else ''} col/step={cfg.batch_col} steps={cfg.n_steps}", flush=True)
        params, hist = _fit_map(seed, obs_idx, epoch_w)
        if m == 0:
            map_params, map_hist = params, hist
        samples.append(np.asarray(eval_b(params["phi"])))
        hists.append(hist)
    map_melt = samples[0]

    if cfg.hmc_samples > 0:
        import blackjax
        theta_map = map_params["theta"]

        def logdens_phi(phi):
            # full-batch physics likelihood on a fixed collocation design + prior
            return -(nll_col(theta_map, phi, col_xy_fixed, col_t_fixed) * (N_col / col_xy_fixed.shape[0])) + log_prior({"phi": phi})

        def logdens_joint(params):
            lo = (nll_obs_epochs_seq(params["theta"], params.get("planes")) if cfg.transfer
                  else nll_obs(params["theta"], obs_xyt, obs_H, params.get("planes"), obs_k))
            return -(lo + nll_col(params["theta"], params["phi"], col_xy_fixed, col_t_fixed) * (N_col / col_xy_fixed.shape[0])) + log_prior(params)

        kc = jax.random.PRNGKey(cfg.seed + 1000)
        kc1, kc2, kc3 = jax.random.split(kc, 3)
        n_fix = min(cfg.batch_col * 4, dom_xy.shape[0] * cfg.n_col_slices)
        ic = jax.random.choice(kc1, dom_xy.shape[0], (n_fix,))
        col_xy_fixed = dom_xy[ic] + jax.random.uniform(kc2, (n_fix, 2), minval=-0.5, maxval=0.5) * jnp.array([dx, dy])
        col_t_fixed = jax.random.uniform(kc3, (n_fix,), minval=t0, maxval=t1)
        logdens, init = (logdens_joint, map_params) if cfg.hmc_joint else (logdens_phi, map_params["phi"])
        print(f"  [bpinn] NUTS {'joint' if cfg.hmc_joint else 'melt-net'}: warmup {cfg.hmc_warmup}, samples {cfg.hmc_samples}", flush=True)
        warm = blackjax.window_adaptation(blackjax.nuts, logdens)
        (state, tuned), _ = warm.run(jax.random.PRNGKey(cfg.seed + 2000), init, num_steps=cfg.hmc_warmup)
        kernel = jax.jit(blackjax.nuts(logdens, **tuned).step)
        kk = jax.random.PRNGKey(cfg.seed + 3000)
        for s in range(cfg.hmc_samples):
            kk, sub = jax.random.split(kk)
            state, info = kernel(sub, state)
            phi_s = state.position["phi"] if cfg.hmc_joint else state.position
            samples.append(np.asarray(eval_b(phi_s)))
            if s % max(cfg.hmc_samples // 10, 1) == 0:
                print(f"    nuts sample {s}/{cfg.hmc_samples} logdens {float(state.logdensity):.3e} accept {float(info.acceptance_rate):.2f}", flush=True)

    Hg0 = np.asarray(eval_H(map_params["theta"], jnp.stack([gx, gy, jnp.full_like(gx, t0)], 1))).reshape(ny, nx)
    Hg1 = np.asarray(eval_H(map_params["theta"], jnp.stack([gx, gy, jnp.full_like(gx, t1)], 1))).reshape(ny, nx)
    trend = (Hg1 - Hg0) / max(t1 - t0, 1e-9)
    print(f"  [bpinn] surrogate time trend dH/dt on the domain: median {np.nanmedian(trend[data.domain]):+.2f} m/yr, "
          f"p10/p90 {np.nanpercentile(trend[data.domain], 10):+.1f}/{np.nanpercentile(trend[data.domain], 90):+.1f}", flush=True)
    S = np.stack(samples)
    if cfg.transfer:
        pk = np.asarray(jax.jit(apparent_many)(map_params["theta"], t_epochs))   # (K, ny, nx), sequential
        kk, jj, ii = np.nonzero(np.isfinite(data.H_obs) & data.domain[None])
        pred = pk[kk, jj, ii]
    else:
        pred = np.asarray(eval_H(map_params["theta"], obs_xyt))
    if "planes" in map_params:
        pl = map_params["planes"]
        pred = pred + np.asarray(plane(pl, obs_k, obs_xyt[:, 0], obs_xyt[:, 1]))
        print(f"  [bpinn] per-epoch planes: |αz| median {float(jnp.median(jnp.abs(pl['z']))):.2f} m, "
              f"|αx|,|αy| median {float(jnp.median(jnp.abs(pl['x']))):.3f}/{float(jnp.median(jnp.abs(pl['y']))):.3f} m/km", flush=True)
    obs_rms = float(np.sqrt(np.mean((pred - np.asarray(obs_H)) ** 2)))
    print(f"  [bpinn] MAP surrogate fits the observations at rms {obs_rms:.2f} m; {S.shape[0]} posterior samples", flush=True)
    return BPINNResult(S.mean(0), S.std(0) if S.shape[0] > 1 else np.zeros_like(map_melt), S, map_melt,
                       obs_rms, map_hist, cfg, {"n_obs": int(N_obs), "n_col_nominal": int(N_col), "trend": trend})
