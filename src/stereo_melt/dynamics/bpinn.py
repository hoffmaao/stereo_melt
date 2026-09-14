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
use; ``eta_bar``, ``alpha_scale``, ``H_ref_m``, ``ux_ref_myr``/``uy_ref_myr``)
and ``L`` a
Gaussian low-pass at ``transfer_bg_sigma_H`` ice thicknesses that keeps the
operator at unity on the long wavelengths where ``T`` is unity anyway — then
cropped and compared to the hydrostatic thickness observations. Data
mini-batches are then whole epochs (``batch_epochs``).

``T`` is strongly anisotropic about the flow axis, so the reference velocity is
passed as COMPONENTS, defaulting to the tile mean of ``data.vx``/``data.vy``
over ``data.domain`` (time-averaged first for a time-varying velocity) — the
same ``_mean_component`` convention, and the same positive ``dx``/``dy``
spacings, that the restored-budget and inverse call sites use. An earlier
default took only the mean SPEED and placed the flow along +x; on a basin whose
flow is not along the grid x axis that mis-orients the operator by the flow
angle, which on the PIG trunk (about -107°) swaps the along- and across-flow
damping at :math:`\lambda = 2H` (|M| 0.31 vs 0.14 along flow, 0.15 vs 0.37
across), converging to within 3 % only by :math:`4H`.

The y component is mirrored on the way in. ``y_km`` descends, so ``fft2`` of the
row-indexed grid places a physical :math:`(k_x, k_y)` at lattice
:math:`(k_x, -k_y)`; since :math:`|T|` depends on :math:`\mathbf k` through
:math:`|\mathbf k|` and :math:`(\boldsymbol\alpha\!\cdot\!\mathbf k)^2`, feeding
the operator the PHYSICAL :math:`u_y` on that lattice would reflect its
anisotropy axis to :math:`-\theta` — 33° of misorientation on the PIG trunk,
leaving across-flow 2H structure 1.6x over-amplified. The production
restored-budget and inverse call sites pass the physical :math:`u_y` with a
positive ``dy`` and so carry that mirror; this module does not, because
``prepare_bpinn_data`` differences ``divu`` with a negative ``dy`` and ``_pix``
and ``residual`` are in physical y, so the operator must be too. Building it
runs two guards. The binding one rebuilds the transfer from the PHYSICAL
components and requires the mirrored operator to agree with it at the along- and
across-flow wavevectors, reading the mirrored value by exact index arithmetic
rather than by re-snapping a negated wavevector (the ``fftfreq`` lattice is
antisymmetric only away from its Nyquist bin): an outside reference is needed
because flipping the y sign in both the operator and a probe leaves :math:`|T|`
unchanged for any flow within 22.5° of a grid axis or diagonal — the PIG trunk
azimuth among them — so probes sharing the operator's convention cannot detect a
mirror error at all. The second is a physical sanity check: a unit plane wave
laid out on the real grid coordinates must come through no stronger along flow
than across it. It is asserted only where the operator is anisotropic enough for
the comparison to mean anything — the transfer with advection is divided by the
same transfer with ``alpha_scale = 0`` at one shared lattice point, and the
assertion stands down below a 5 % contrast. Gating on speed instead would be
wrong: the anisotropy is set by ``alpha_scale * u_ref * t_r / H``, so a fast
shelf with ``alpha_scale = 0`` (or a low ``eta_bar``) has an exactly isotropic
transfer, and the two probes then differ only by lattice snapping and
reflect-pad leakage — which used to raise on a perfectly oriented operator.
Both probe at :math:`2H`, or at four pixels of the coarser axis when :math:`2H`
is finer than that — a shelf thinner than its own pixel cannot carry a
resolvable 2H probe. The probe must also stay under a quarter of the shorter
domain side, above which reflect-pad leakage swamps the projected gain and the
ordering is decided by the window rather than by the flow. Outside that band the
physics check only logs. On the packaged PIG trunk window (164 rows x 154 cols
at 250 m, :math:`H` 560 m) the usable band is 1000-9625 m and the probe sits at
1120 m, 8.6x under the ceiling -- the ceiling is a quarter of the 38.5 km
shorter side, not the side itself.

Without the transfer
(the default) a hydrostatic budget can only see :math:`|T|` of channel-scale
melt (0.06 at 2H, 0.34 at 3H on the PIGREAL twin, 2026-09-12), which is why
it matched the Eulerian ceiling there.

JAX, optax and blackjax are imported lazily; the module imports without them.
Validate on the DEM-stack twins (truth known) before any real basin: see
``examples/elmer_synth/scripts/run_bpinn_twin.py``, the twin validation driver
committed alongside this module.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = ["BPINNConfig", "BPINNData", "BPINNResult", "prepare_bpinn_data", "fit_bpinn"]

# Epochs a pixel needs before its observed dH/dt is trusted as a collapse yardstick.
MIN_TREND_EPOCHS = 5
# Observed thinning a stack must show before a near-zero fitted trend means anything.
# A steady stack -- the DEM-stack twins are steady by construction -- has an observed
# median at the noise level, and a ratio against that would call a perfect fit a collapse.
STEADY_TREND_MYR = 0.5


@dataclass
class BPINNConfig:
    """Model, optimiser and observation-operator settings for :func:`fit_bpinn`.

    ``sigma_r_myr`` and ``n_col_slices`` set how hard the physics residual pulls
    against the data. The defaults (20 m/yr, 4 slices) are the pair validated on
    the PIG trunk. A steady twin tolerates far more physics weight -- 1 m/yr and
    24 slices fit it well -- but on a real stack that pair silently collapses the
    surrogate to a static field: the fit still returns a plausible-looking melt
    map, which is just the steady budget of the median base field. Two checks
    catch it, and :func:`fit_bpinn` runs the first for you:

    1. the fitted surrogate ``dH/dt`` must be of the order of the observed
       thinning (about -3 to -6 m/yr on the PIG trunk; near 0 means collapse).
       ``fit_bpinn`` prints both trends and raises a ``RuntimeWarning`` when the
       fitted one falls under 20 % of the observed one. The ratio needs a stack
       that is actually thinning, so the check stands down (and says so) when the
       observed median is under ``STEADY_TREND_MYR``, as on the steady twins;
    2. ``transfer=False`` must change the answer. If it does not, the observation
       operator is not informing the fit.
    """

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
    sigma_r_myr: float = 20.0
    n_col_slices: int = 4
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
    ux_ref_myr: float | None = None
    uy_ref_myr: float | None = None


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
    if dy >= 0:
        raise ValueError("y must be descending (north-up); the collocation sampler and the "
                         "bridging observation operator both assume that orientation")
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


def _observed_trend(data: BPINNData, min_epochs: int = MIN_TREND_EPOCHS) -> np.ndarray:
    """Per-pixel least-squares dH/dt (m/yr) of the observations themselves.

    The independent yardstick for the surrogate's fitted trend: a collapsed
    surrogate reports a trend near zero while the stack it was fitted to is
    thinning. NaN where a pixel carries fewer than ``min_epochs`` finite epochs
    or sits outside ``data.domain``.
    """
    fin = np.isfinite(data.H_obs) & data.domain[None]
    t = np.asarray(data.t_yr, float)[:, None, None]
    w = fin.astype(float)
    Hf = np.where(fin, data.H_obs, 0.0)
    n = w.sum(axis=0)
    st, stt = (w * t).sum(axis=0), (w * t * t).sum(axis=0)
    sh, sth = Hf.sum(axis=0), (Hf * t).sum(axis=0)
    den = n * stt - st ** 2
    ok = (fin.sum(axis=0) >= min_epochs) & (den > 0)
    return np.where(ok, (n * sth - st * sh) / np.where(ok, den, 1.0), np.nan)


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
        """Every term of the thickness equation at points (km, km, yr), unsummed."""
        H_scalar = lambda xx, yy, tt: H_net(theta, xx, yy, tt)  # noqa: E731
        H = H_scalar(x, y, t)
        Hx = jax.vmap(jax.grad(H_scalar, 0))(x, y, t)          # m/km
        Hy = jax.vmap(jax.grad(H_scalar, 1))(x, y, t)
        Ht = jax.vmap(jax.grad(H_scalar, 2))(x, y, t)          # m/yr
        u, v = _sample3(vx_g, x, y, t), _sample3(vy_g, x, y, t)
        return dict(H=H, Hx=Hx, Hy=Hy, Ht=Ht, u=u, v=v, divu=_sample3(divu_g, x, y, t), a=_sample2(a_g, x, y), b=b_net(phi, x, y))

    def residual(theta, phi, x, y, t):
        """Thickness-equation residual at points (km, km, yr) → m/yr."""
        p = residual_parts(theta, phi, x, y, t)
        return p["Ht"] + p["H"] * p["divu"] + p["u"] * p["Hx"] + p["v"] * p["Hy"] - p["a"] - p["b"]

    Xg, Yg = np.meshgrid(data.x_km, data.y_km)
    Xg_j, Yg_j = jnp.asarray(Xg), jnp.asarray(Yg)
    gx, gy = Xg_j.ravel(), Yg_j.ravel()

    # ---- bridging transfer as the observation operator (optional) ----
    if cfg.transfer:
        from .bridging_restoration import _bridging_transfer
        H_ref = cfg.H_ref_m if cfg.H_ref_m is not None else float(data.H0)
        if cfg.ux_ref_myr is None or cfg.uy_ref_myr is None:
            if not data.domain.any():
                raise ValueError("empty domain: cannot default the reference velocity, "
                                 "pass BPINNConfig(ux_ref_myr=..., uy_ref_myr=...)")
            ux_d = float(data.vx.mean(axis=0)[data.domain].mean() * 1e3)
            uy_d = float(data.vy.mean(axis=0)[data.domain].mean() * 1e3)
            print(f"  [bpinn] reference velocity default: tile mean over {int(data.domain.sum())} "
                  f"domain px = ({ux_d:+.0f}, {uy_d:+.0f}) m/yr", flush=True)
        else:
            ux_d = uy_d = 0.0
        ux_ref = float(cfg.ux_ref_myr) if cfg.ux_ref_myr is not None else ux_d
        uy_ref = float(cfg.uy_ref_myr) if cfg.uy_ref_myr is not None else uy_d
        u_ref = float(np.hypot(ux_ref, uy_ref))
        flow_deg = float(np.degrees(np.arctan2(uy_ref, ux_ref)))
        # y_km descends, so fft2 of the row-indexed grid puts a physical (kx, ky)
        # at lattice (kx, -ky): the flow must enter the operator y-mirrored for its
        # anisotropy axis to land on the physical flow axis.
        uy_lat = -uy_ref
        flow_hat = (np.array([1.0, 0.0]) if u_ref <= 0
                    else np.array([ux_ref, uy_lat]) / u_ref)
        along_hat = (np.array([1.0, 0.0]) if u_ref <= 0
                     else np.array([ux_ref, uy_ref]) / u_ref)
        across_hat = np.array([-along_hat[1], along_hat[0]])
        Py, Px = 2 * ny, 2 * nx
        T_np = _bridging_transfer(Py, Px, dx * 1e3, dy * 1e3, H_ref, ux_ref, uy_lat, cfg.eta_bar, cfg.alpha_scale,
                                  data.extras.get("rho_i", 917.0), data.extras.get("rho_w", 1027.0), 9.81, 0.0, 0.0)[0]
        kxp = 2 * np.pi * np.fft.fftfreq(Px, dx * 1e3)
        kyp = 2 * np.pi * np.fft.fftfreq(Py, dy * 1e3)
        K2 = kxp[None, :] ** 2 + kyp[:, None] ** 2
        sig_bg = cfg.transfer_bg_sigma_H * H_ref
        L_np = np.exp(-0.5 * K2 * sig_bg ** 2)
        M_np = 1.0 + (T_np - 1.0) * (1.0 - L_np)
        M_op = jnp.asarray(M_np)
        Xd = np.stack([np.ones(ny * nx), (Xg.ravel() - xc), (Yg.ravel() - yc)], 1)
        P_pinv = jnp.asarray(np.linalg.pinv(Xd))          # (3, ny*nx)
        Xd_j = jnp.asarray(Xd)

        # Probes must stay well inside the padded grid's Nyquist or they say nothing about
        # the operator: snapping wraps, and the +ky/-ky bins stop being mirrors. Four pixels
        # of the coarser axis is the floor; a shelf thinner than its own pixel is probed
        # there instead of at 2H, and the physics assertion downgrades to a log line.
        # There is an upper bound too: once the probe is an appreciable fraction of the tile,
        # reflect-pad leakage swamps the projected gain and the along/across ordering is
        # decided by the window rather than by the flow. A quarter of the shorter domain side
        # is the ceiling; the packaged PIG trunk (164 x 154 at 250 m) probes 1120 m in a
        # 1000-9625 m band, 8.6x under it.
        lam_min = 4.0 * max(dx, dy) * 1e3
        lam_max = 0.25 * min(nx * dx, ny * dy) * 1e3
        lam_probe = max(2.0 * H_ref, lam_min)
        unresolved = 2.0 * H_ref < lam_min or lam_probe > lam_max

        def _lattice_idx(khat, lam):
            kx_w, ky_w = (2 * np.pi / lam) * np.asarray(khat)
            return (int(np.argmin(np.abs(kyp - ky_w))), int(np.argmin(np.abs(kxp - kx_w))))

        def _mirror_idx(idx):
            """Lattice index of the y-mirrored wavevector, by exact index arithmetic."""
            m, n = idx
            return ((Py - m) % Py, n)

        def _T_along_flow(lam):
            """|T| at the padded-lattice point nearest the along-flow wavevector of wavelength lam."""
            return abs(T_np[_lattice_idx(flow_hat, lam)])

        def _probe_gain(khat, lam):
            """Gain the built operator applies to a unit plane wave of wavelength lam on PHYSICAL
            direction khat: laid out on the real (x_km, y_km) coordinates, pushed through the
            same path apparent_epoch uses, then projected back onto itself so that the
            reflect-pad's spectral leakage (which swamps a peak-amplitude estimate) does not
            bias it."""
            k = (2 * np.pi / lam) * np.asarray(khat)
            f = np.cos(k[0] * Xg * 1e3 + k[1] * Yg * 1e3)
            ap = np.pad(f, ((0, ny), (0, nx)), mode="reflect")
            out = np.real(np.fft.ifft2(np.fft.fft2(ap) * M_np))[:ny, :nx]
            return float(abs(np.sum(out * f) / np.sum(f * f)))

        # Reference transfers on the PHYSICAL lattice: one with the real advection, one with
        # it switched off. The first is the orientation guard's independent reference. The
        # ratio of the two AT THE SAME LATTICE POINT is how much anisotropy the operator
        # actually carries at the probe wavelength, which is what the physics assertion needs
        # to be gated on -- the anisotropy is set by alpha_scale * u_ref * t_r / H, not by
        # u_ref, so at alpha_scale = 0 the transfer is exactly isotropic at any speed and the
        # two probes then differ only by lattice snapping and reflect-pad leakage. Reading
        # both at one index is what makes this exact: an along/across pair snapped separately
        # sits at different |k| and reports up to 187 % anisotropy on a small grid at
        # alpha_scale = 0, which would re-enable the assertion in precisely the isotropic case
        # it has to stand down for.
        def _phys_transfer(alpha_scale):
            return _bridging_transfer(Py, Px, dx * 1e3, dy * 1e3, H_ref, ux_ref, uy_ref,
                                      cfg.eta_bar, alpha_scale, data.extras.get("rho_i", 917.0),
                                      data.extras.get("rho_w", 1027.0), 9.81, 0.0, 0.0)[0]

        T_phys = _phys_transfer(cfg.alpha_scale)
        idx_along = _lattice_idx(along_hat, lam_probe)
        t_iso = abs(_phys_transfer(0.0)[idx_along])
        aniso = abs(1.0 - abs(T_phys[idx_along]) / t_iso) if t_iso > 0 else 0.0
        aniso_margin = 0.05

        a_along, a_across = _probe_gain(along_hat, lam_probe), _probe_gain(across_hat, lam_probe)
        assert_aniso = not unresolved and aniso > aniso_margin
        if unresolved and 2.0 * H_ref < lam_min:
            gate_note = (f" (2H = {2 * H_ref:.0f} m is below this grid's {lam_min:.0f} m probe "
                         f"floor, so the probe was raised to it: logged, not asserted)")
        elif unresolved:
            gate_note = (f" ({lam_probe:.0f} m probe is above this grid's {lam_max:.0f} m "
                         f"ceiling: logged, not asserted)")
        elif not assert_aniso:
            gate_note = (f" (predicted anisotropy {aniso * 100:.1f}% is under the "
                         f"{aniso_margin * 100:.0f}% margin: logged, not asserted)")
        else:
            gate_note = ""
        print(f"  [bpinn] transfer ON: H_ref {H_ref:.0f} m, flow {u_ref:.0f} m/yr at {flow_deg:+.0f}° "
              f"(ux {ux_ref:+.0f}, uy {uy_ref:+.0f} m/yr), eta {cfg.eta_bar:.1e}, alpha {cfg.alpha_scale}; "
              f"|T| along-flow at {lam_probe:.0f}/{1.5 * lam_probe:.0f} m = "
              f"{_T_along_flow(lam_probe):.2f}/{_T_along_flow(1.5 * lam_probe):.2f}; "
              f"predicted anisotropy {aniso * 100:.0f}%; "
              f"{lam_probe:.0f} m gain along/across flow = {a_along:.3f}/{a_across:.3f}{gate_note}; "
              f"background low-pass sigma {sig_bg/1e3:.1f} km; padded FFT {Py}x{Px}", flush=True)
        # Orientation guard, against an INDEPENDENT reference: require the operator actually
        # built (physical flow, y-mirrored into the lattice) to agree with T_phys at the same
        # physical wavevectors. Comparing the two probes to each other cannot establish this
        # -- flipping the y sign in both the operator and the probe leaves |T| untouched for
        # any flow within 22.5 deg of a grid axis or diagonal, the PIG trunk azimuth among
        # them -- so a shared-convention error is only caught outside that convention.
        for name, khat in (("along-flow", along_hat), ("across-flow", across_hat)):
            idx = _lattice_idx(khat, lam_probe)
            built = abs(T_np[_mirror_idx(idx)])
            want = abs(T_phys[idx])
            if not np.isclose(built, want, rtol=1e-6, atol=1e-12):
                raise ValueError(
                    f"bridging operator is misoriented: at the {lam_probe:.0f} m {name} wavevector "
                    f"it applies |T| {built:.4f}, but the transfer built from the physical flow "
                    f"({ux_ref:+.0f}, {uy_ref:+.0f}) m/yr gives {want:.4f}. y_km descends, so "
                    f"the flow's y component must be mirrored into the FFT lattice.")
        if assert_aniso and a_along > a_across:
            raise ValueError(
                f"bridging operator does not damp along-flow structure: a {lam_probe:.0f} m plane "
                f"wave along the flow ({flow_deg:+.0f}°) comes through at {a_along:.3f} but "
                f"across-flow at {a_across:.3f}, where the transfer predicts {aniso * 100:.0f}% "
                f"anisotropy. Check the reference velocity components.")
        H_dense = jnp.asarray(np.nan_to_num(data.H_obs))
        M_dense = jnp.asarray(np.isfinite(data.H_obs) & data.domain[None])
        t_epochs = jnp.asarray(data.t_yr)

        def apparent_epoch(theta, tk):
            """Surrogate on the grid at epoch time tk → hydrostatic-apparent thickness (ny, nx)."""
            h = H_net(theta, gx, gy, jnp.full_like(gx, tk))          # (ny*nx,)
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

    eval_b = jax.jit(lambda phi: b_net(phi, gx, gy).reshape(ny, nx))
    eval_H = jax.jit(lambda theta, xyt: H_net(theta, xyt[:, 0], xyt[:, 1], xyt[:, 2]))

    if cfg.n_steps <= 0:   # debug hook: residual parts at init
        kd = jax.random.PRNGKey(0)
        params = {"theta": _init_mlp(kd, in_H, cfg.hidden, cfg.layers, 1), "phi": _init_mlp(kd, in_b, cfg.melt_hidden, cfg.melt_layers, 1)}
        xy = dom_xy[:512]
        parts = residual_parts(params["theta"], params["phi"], xy[:, 0], xy[:, 1],
                               jnp.full((xy.shape[0],), tc))
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
    obs_trend = _observed_trend(data)
    # The observed trend only exists where a pixel has enough epochs, and those pixels are
    # not a random sample of the domain -- they cluster where strips repeatedly overlap, which
    # on a trunk is also where thinning is fastest. Comparing a domain-wide fitted median
    # against that subset's observed median would flag healthy fits as collapsed, so the
    # ratio is taken over the subset on both sides.
    cmp_px = data.domain & np.isfinite(obs_trend)
    n_cmp = int(cmp_px.sum())
    dom_med = float(np.nanmedian(trend[data.domain]))
    fit_med = float(np.nanmedian(trend[cmp_px])) if n_cmp else float("nan")
    obs_med = float(np.nanmedian(obs_trend[cmp_px])) if n_cmp else float("nan")
    print(f"  [bpinn] surrogate time trend dH/dt on the domain: median {dom_med:+.2f} m/yr, "
          f"p10/p90 {np.nanpercentile(trend[data.domain], 10):+.1f}/{np.nanpercentile(trend[data.domain], 90):+.1f}"
          f"; over the {n_cmp} px with >= {MIN_TREND_EPOCHS} epochs: fitted {fit_med:+.2f} vs "
          f"observed {obs_med:+.2f} m/yr", flush=True)
    steady = np.isfinite(obs_med) and abs(obs_med) < STEADY_TREND_MYR
    if steady:
        print(f"  [bpinn] collapse check stood down: the observed trend median {obs_med:+.3f} m/yr is "
              f"under the {STEADY_TREND_MYR:g} m/yr floor, so this stack is steady and a near-zero "
              f"fitted trend {fit_med:+.3f} m/yr is not evidence of collapse", flush=True)
    if not steady and np.isfinite(obs_med) and np.isfinite(fit_med) and abs(fit_med) < 0.2 * abs(obs_med):
        warnings.warn(
            f"B-PINN surrogate may have collapsed to a static field: over the {n_cmp} domain px "
            f"with >= {MIN_TREND_EPOCHS} epochs, the fitted dH/dt median {fit_med:+.3f} m/yr is "
            f"under 20% of the observed {obs_med:+.3f} m/yr. The physics residual is probably "
            f"over-weighted -- raise sigma_r_myr (currently {cfg.sigma_r_myr:g}) or lower "
            f"n_col_slices (currently {cfg.n_col_slices}). The melt map returned is then just "
            f"the steady budget of the base field.",
            RuntimeWarning, stacklevel=2)
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
                       obs_rms, map_hist, cfg,
                       {"n_obs": int(N_obs), "n_col_nominal": int(N_col), "trend": trend,
                        "obs_trend": obs_trend, "trend_median_myr": fit_med,
                        "obs_trend_median_myr": obs_med,
                        "trend_median_domain_myr": dom_med, "n_trend_cmp_px": n_cmp,
                        "collapse_check_stood_down": bool(steady)})
