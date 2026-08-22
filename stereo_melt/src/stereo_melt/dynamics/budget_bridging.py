r"""Monolithic melt inverse: mass budget **and** viscous bridging in one fit.

Motivation
----------
The workspace has carried two disjoint solver families. The mass-budget family
(:func:`~stereo_melt.melt.eulerian_melt_rate` and friends) conserves mass and so
owns the absolute melt level, but it converts thickness to surface
*hydrostatically at every wavelength*. The Stubblefield family
(:mod:`~stereo_melt.dynamics.stubblefield_forward`) supplies the missing
mechanics — a shelf cannot sag hydrostatically into a narrow thin spot, because
viscous stresses bridge across it — but it was only ever applied to a
high-passed *anomaly*, which made it DC-blind and forced a fusion step to glue
the two answers back together in wavenumber bands.

They are not rivals, and the seam between them is the **observation operator**,
not a dynamical one.

What the DEM stack actually measures
------------------------------------
A DEM gives surface elevation :math:`z_s`; every hydrostatic inverse turns that
into thickness by dividing by the freeboard factor
:math:`f_b = 1 - \rho_i/\rho_w`. Full-Stokes ice does not float pointwise, so
the thickness we *infer* is not the thickness the shelf *has*:

.. math::
    \hat H_f(\mathbf k) \;=\; \frac{\hat z_s}{f_b}
      \;=\; T(\mathbf k)\,\hat H_{\rm true}(\mathbf k),
    \qquad
    T \;=\; \frac{G_h}{f_b\,(G_h - G_s)}
      \;=\; \frac{(1+\delta)B}{\delta B + R + q},

with :math:`G_h, G_s` the steady Stubblefield surface/basal Green's functions.
:math:`T` is the **flotation departure**: the surface response divided by the
hydrostatic response to the same thickness change. It is 1 wherever the shelf
floats freely and falls below 1 where bridging stresses hold the surface up over
a thin spot — 0.375 / 0.703 / 0.863 at :math:`\lambda = 2H / 3H / 4H`
(:math:`\alpha = 0`, the Stubblefield headline: 62% / 30% damping). With
:math:`\alpha = 0` it depends on **nothing but** :math:`\lambda/H` and the
density ratio: no viscosity, no velocity, no calibration. That closed form is
gated in ``tests/gate_budget_bridging_identity.py`` (rung G4).

The forward model
-----------------
True thickness obeys the mass budget exactly (Shean sign: negative
:math:`\dot m` = melt),

.. math:: \partial H_{\rm true}/\partial t = \dot m + \dot a
          - \nabla\!\cdot(H_{\rm true} u),

and :math:`T` is a fixed spatial multiplier, so applying it to the whole
equation and using :math:`T\{\nabla\!\cdot(H_{\rm true} u)\}
= \nabla\!\cdot(T\{H_{\rm true}\}\,u) = \nabla\!\cdot(H_f u)` — exact for uniform
:math:`u`, because :math:`\nabla\!\cdot(\cdot\,u)` is itself the Fourier
multiplier :math:`i\mathbf k\!\cdot\!u` and multipliers commute — gives

.. math::
    \left(\partial H_f/\partial t\right)_{\rm obs}
        = T\{\dot m + \dot a\} - \nabla\!\cdot(H_f u).

**The flux divergence sits outside the operator**, and it is built from the
*observed* mean thickness, which is already the bridged field. Putting it inside
(as this module did before 2026-08-20) filters it a second time and double-counts
the advection — the failure mode :mod:`.stubblefield_forward` warns about for the
post-hoc deconvolution route. In the steady limit it is worse than a bias: with
the flux inside, the fit reduces to :math:`D\{\dot m + \dot a - \nabla\!\cdot
(H_f u)\} = 0`, which for any invertible :math:`D` is the *hydrostatic* answer,
so the bridging correction silently vanishes exactly where the shelf is best
observed.

Why the 513-epoch timeseries costs nothing
------------------------------------------
With :math:`\dot m` steady over the window and the flux divergence linearised
about the observed mean thickness — the assumption every existing solver already
makes — the model is affine in :math:`t` with a slope field linear in
:math:`\dot m`. Profiling out a free per-pixel intercept, the per-epoch sum of
squares collapses **exactly** to the centred temporal moments,

.. math::
    \sum_i (H_i - \alpha - t_i B)^2 = S_{tt}\,(B - \widehat{\partial_t H})^2
                                      + \text{const},

so the whole-stack fit reduces to one weighted least-squares term per pixel
against the ordinary per-pixel trend, with weight :math:`S_{tt}` — the temporal
leverage, i.e. the inverse variance of that pixel's slope estimate up to a
common factor. **The optimiser never loops over epochs**; a 513-epoch fit costs
one FFT pair per iteration, the same as a single-snapshot fit.

The objective is therefore

.. math::
    L(\dot m) = \sum_x S_{tt}(x)\,
        \bigl( T\{\dot m + \dot a\}(x) - \nabla\!\cdot(H_f u)(x)
               - \widehat{\partial_t H_f}(x) \bigr)^2
        + \lambda \lVert \nabla \dot m \rVert^2 .

Consequences worth stating
--------------------------
* **No high-pass, no anomaly, no prior, no fusion.** The reference state never
  enters: a free per-pixel intercept was profiled out analytically.
* **The mean is identified.** :math:`T(0) = 1` *analytically* (the :math:`k\to0`
  limits :math:`G_h \to -2`, :math:`G_s \to 2/\delta` give
  :math:`\delta/(f_b(\delta+1)) = 1`), so the melt's spatial mean is fixed by
  the budget exactly as in the Eulerian solver — no plateau hunt, no hand-pinned
  DC bin.
* **``bridging=False`` reproduces the Eulerian solver exactly.** With
  :math:`T = I` the minimiser is pixelwise
  :math:`\dot m = \widehat{\partial_t H_f} + \nabla\!\cdot(H_f u) - \dot a`,
  which is Shean Eq. 10 — for any positive weights. Gate:
  ``tests/gate_budget_bridging_identity.py``.
* **The ill-conditioning is at SHORT wavelengths**, where :math:`|T|` falls to
  3% by :math:`\lambda = H`, so :math:`T^{-1}` amplifies ~30x. That is what
  ``lam``/``ridge`` regularise, and it is the honest location of the difficulty —
  the long-wavelength band the legacy high-pass removed was never the problem.
* **What this operator does NOT model.** :math:`T` is a filter of *thickness*.
  The E2a Elmer twin also carries a flotation departure proportional to the melt
  *rate* itself (roughly 0.5 m per m/yr, wavelength-independent), which no
  thickness filter can represent; in a budget inverse it appears as a
  :math:`u\!\cdot\!\nabla\dot m` dipole and is the known across-flow-channel
  failure. See ``elmer_synth/scripts/score_bridging_operator.py``.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import dh_dt, flux_divergence

SECONDS_PER_YEAR = 86400.0 * 365.25

__all__ = ["budget_bridging_melt_rate", "bridging_transfer_multiplier",
           "normalized_bridging_multiplier"]


def bridging_transfer_multiplier(
    ny: int,
    nx: int,
    dx: float,
    dy: float,
    H: float,
    ux_myr: float,
    uy_myr: float,
    *,
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    gamma: float = 0.0,
    theta: float = 1e-14,
) -> np.ndarray:
    r"""Return the flotation departure :math:`T(k) = G_h/(f_b (G_h - G_s))`.

    The map from **true** thickness to the thickness a hydrostatic inversion of
    the DEM infers, on the ``(ny, nx)`` FFT wavenumber grid (pass the *padded*
    shape when the field is mirror-doubled before transforming). Identical
    object to :func:`~stereo_melt.dynamics.bridging_restoration._bridging_transfer`
    — one transfer, shared by the spectral post-filter, the Wiener inverse and
    this monolithic fit, so they cannot drift apart.

    ``T[0, 0]`` is set to **1**, which is the analytic limit rather than a
    convention: :math:`G_h \to -2` and :math:`G_s \to 2/\delta` as
    :math:`k \to 0`, so :math:`T \to \delta/(f_b(\delta+1)) = 1` exactly. The
    :math:`k=0` bin only needs setting because
    :meth:`~stereo_melt.dynamics.linear_perturbation.LinearPerturbation.steady_state_kernel`
    hard-zeros DC to keep perturbation operators mean-free.

    With ``alpha_scale=0`` the result is real, isotropic and depends on nothing
    but :math:`\lambda/H` and :math:`\rho_i/\rho_w` — no viscosity, no velocity.
    ``alpha_scale != 0`` adds the advective term :math:`q = i\mathbf k\!\cdot\!u
    \,t_r`, which damps and phase-shifts along-flow wavenumbers; on the E2a
    geometry that is a <= 3% amplitude effect over the resolvable band, so the
    choice is not load-bearing here (it is on fast, thick ice).
    """
    from .bridging_restoration import _bridging_transfer
    from .linear_perturbation import G_GRAVITY
    from ..backend import to_numpy

    T, _kx, _ky, _fb, _u = _bridging_transfer(
        ny, nx, dx, dy, H, ux_myr, uy_myr, eta_bar, alpha_scale,
        rho_i, rho_w, G_GRAVITY, gamma, theta)
    T = np.ascontiguousarray(to_numpy(T)).astype(np.complex128)
    if not np.isfinite(T[1:, 1:]).any():
        raise ValueError("bridging transfer is entirely non-finite")
    T[0, 0] = 1.0 + 0.0j
    return T


def normalized_bridging_multiplier(
    ny: int,
    nx: int,
    dx: float,
    dy: float,
    H: float,
    ux_myr: float,
    uy_myr: float,
    *,
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    rho_i: float = rhoi,
    rho_w: float = rhow,
) -> np.ndarray:
    r"""Return :math:`D(k) = M_h(k)/|M_h|_{\max}` with ``D[0, 0] = 1``.

    ``M_h`` is the Stubblefield steady multiplier. Dividing by its long-
    wavelength plateau strips the hydrostatic conversion and any relaxation-time
    scale, leaving a dimensionless *relative* damping that tends to unity where
    the shelf floats hydrostatically. The complex phase is kept: it is the
    advective downstream lag of the surface expression.

    The plateau is the **signed complex** value of ``M_h`` at its modulus
    maximum, not the modulus itself — see the inline note; using ``max|M_h|``
    inverted the sign of every non-zero wavenumber.

    The ``k = 0`` bin is set to **1**, overriding the hand-zeroing in
    :mod:`.linear_perturbation` — appropriate for an anomaly operator, wrong for
    a relative damping, and the difference is exactly what makes the melt mean
    identifiable here.
    """
    from .stubblefield_forward import stubblefield_forward_multiplier

    M = stubblefield_forward_multiplier(
        ny, nx, dx, dy, H=H, ux_myr=ux_myr, uy_myr=uy_myr,
        eta_bar=eta_bar, alpha_scale=alpha_scale, rho_i=rho_i, rho_w=rho_w)
    # Normalise by the SIGNED COMPLEX plateau, not its modulus. M_h maps melt
    # to surface elevation in the Stubblefield convention (m > 0 = melt), so
    # its long-wavelength limit is NEGATIVE REAL (-3.66 on E2a geometry):
    # melting thins the shelf. Dividing by max|M| discarded that minus sign
    # and produced D = -1 at every k != 0 while D[0, 0] was hand-set to +1 --
    # the domain mean and the rest of the spectrum in opposite signs. Taking
    # the plateau bin's complex value instead leaves |D| untouched and sends
    # D -> +1 in the hydrostatic limit, which is what D[0, 0] = 1 asserts.
    flat = np.abs(M).ravel()
    if not np.isfinite(flat).any():
        raise ValueError("bridging multiplier is entirely non-finite")
    plateau = M.ravel()[int(np.nanargmax(flat))]
    if not np.isfinite(plateau) or abs(plateau) <= 0:
        raise ValueError(f"degenerate bridging multiplier (plateau={plateau})")
    D = M / plateau
    D[0, 0] = 1.0 + 0.0j
    return D


def _temporal_leverage(stack: xr.DataArray) -> xr.DataArray:
    r"""Per-pixel :math:`S_{tt} = \sum_i (t_i - \bar t)^2` over finite samples.

    The inverse variance of the per-pixel slope, up to the (unknown, common)
    residual variance — so it is the correct relative weight for the collapsed
    least-squares term. Pixels seen twice a decade apart outweigh pixels seen
    twice a month apart, which the unweighted chain ignores entirely.
    """
    t = stack["time"].values.astype("datetime64[ns]").astype("float64")
    t = (t - t[0]) / (1e9 * SECONDS_PER_YEAR)          # years since first epoch
    ok = np.isfinite(stack.values)                      # (time, y, x)
    n = ok.sum(0).astype(float)
    tt = np.where(ok, t[:, None, None], 0.0)
    sum_t = tt.sum(0)
    mean_t = np.divide(sum_t, n, out=np.zeros_like(sum_t), where=n > 0)
    stt = (np.where(ok, (t[:, None, None] - mean_t[None]) ** 2, 0.0)).sum(0)
    stt = np.where(n >= 2, stt, 0.0)
    return xr.DataArray(stt, dims=("y", "x"),
                        coords={"y": stack.y.values, "x": stack.x.values})


def budget_bridging_melt_rate(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    *,
    a_dot: xr.DataArray | float = 0.0,
    d: xr.DataArray | float = 0.0,
    floating_mask: xr.DataArray | None = None,
    bridging: bool = True,
    transfer: str = "flotation",
    flux_in_operator: bool = False,
    eta_bar: float = 1e14,
    alpha_scale: float = 0.34,
    eta_field: xr.DataArray | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    iters: int = 300,
    lr: float = 0.0,
    weight: str = "leverage",
    converge_tol: float = 1e-16,
    min_count: int = 3,
    robust_dh_dt: bool = False,
    estimator=None,
    rho_w: float = rhow,
    rho_i: float = rhoi,
    log_every: int = 0,
) -> xr.Dataset:
    """Melt rate from one joint budget + bridging fit to the whole stack.

    Parameters
    ----------
    h_stack, vx, vy
        Surface-elevation stack ``(time, y, x)`` and velocity (m/yr).
    a_dot, d
        SMB (m ice/yr) and firn air content (m), as in
        :func:`~stereo_melt.melt.eulerian_melt_rate`.
    floating_mask
        Cells to fit. Non-floating cells get zero weight and NaN melt.
    bridging
        ``False`` sets the operator to the identity, which reproduces the
        Eulerian solver exactly and is the correctness gate. ``True`` applies
        the bridging transfer selected by ``transfer``.
    transfer
        ``"flotation"`` (default, correct) — :math:`T = G_h/(f_b(G_h - G_s))`,
        the flotation departure, the map from true to hydrostatically-inferred
        thickness. ``"normalized"`` — the pre-2026-08-20
        :math:`D = M_h(k)/M_h(\\text{plateau})`, kept **only** so the operator
        audit can score the shipped answer against the corrected one; it drops
        :math:`G_s` and retains the advective factor that cancels in the true
        ratio, so it over-lifts along-flow wavenumbers by up to 15x on E2a
        geometry. Do not use it for science.
    flux_in_operator
        ``False`` (default, correct) puts :math:`\\nabla\\!\\cdot(H_f u)` on the
        observation side, unfiltered, as the commutation identity requires.
        ``True`` reproduces the pre-2026-08-20 double count, again for the A/B
        only.
    eta_bar, alpha_scale, eta_field
        Viscosity for the bridging operator; ``eta_field`` (a map, e.g. from the
        momentum-balance inversion) overrides the scalar via its masked median.
    lam
        Tikhonov weight on :math:`\\lVert\\nabla \\dot m\\rVert^2`.
    ridge
        Wiener weight on :math:`\\lVert\\dot m\\rVert^2`. With uniform weights the
        minimiser is :math:`D^*/(|D|^2 + \\text{ridge})`, so the deconvolution
        gain is bounded by :math:`1/(2\\sqrt{\\text{ridge}})` exactly where the
        operator is blind. This, not ``lam``, is the principled control on the
        short-wavelength null space.
    iters
        Maximum conjugate-gradient iterations (each costs one forward+backward,
        i.e. two FFT pairs). CG on this quadratic typically converges in tens.
    lr
        Unused; retained so existing call sites do not break.
    weight
        ``"leverage"`` (default, :math:`S_{tt}`), ``"count"``, or ``"none"``.
    converge_tol
        Stop once the CG relative residual falls to this value. Guards the
        degenerate case where the objective is satisfied exactly (see the note
        at the optimiser loop); never reached on real data.
    log_every
        Narrate the fit every N iterations (0 = silent).

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` (Shean sign), plus ``dHdt_obs`` / ``dHdt_fit`` (the fitted
        bridged thickness rate) and ``H_f_mean`` / ``flux_div`` / ``weight`` for
        diagnosis, and a ``fit_var_explained`` attr computed **in rate space
        against real observations** — not a contract-dependent kept-band score.
    """
    import torch

    # ---- observed side: identical construction to eulerian_melt_rate, so the
    # bridging=False identity gate holds for the right reason, not by luck.
    H_f_stack = freeboard_to_thickness(h_stack, d=d, rho_w=rho_w, rho_i=rho_i)
    reg = dh_dt(H_f_stack, min_count=min_count, robust=robust_dh_dt)
    dHdt_obs = reg["slope"] * SECONDS_PER_YEAR
    H_f_mean = H_f_stack.mean("time", skipna=True)

    vxm = vx.mean("time", skipna=True) if "time" in vx.dims else vx
    vym = vy.mean("time", skipna=True) if "time" in vy.dims else vy
    fd = flux_divergence(H_f_mean, vxm, vym, estimator=estimator)

    a_field = a_dot if isinstance(a_dot, xr.DataArray) else xr.DataArray(a_dot)
    a_field = a_field.broadcast_like(H_f_mean)

    # ---- weights
    if weight == "leverage":
        w = _temporal_leverage(H_f_stack)
    elif weight == "count":
        w = reg["count"].astype(float)
    elif weight == "none":
        w = xr.ones_like(H_f_mean)
    else:
        raise ValueError(f"weight must be leverage|count|none, got {weight!r}")

    fit = np.isfinite(dHdt_obs.values) & np.isfinite(fd.values) & (w.values > 0)
    if floating_mask is not None:
        fit &= np.asarray(floating_mask.values, bool)
    if not fit.any():
        raise ValueError("no cells satisfy the fit mask")

    ny, nx = H_f_mean.sizes["y"], H_f_mean.sizes["x"]
    dx = float(abs(H_f_mean.x.values[1] - H_f_mean.x.values[0]))
    dy = float(abs(H_f_mean.y.values[1] - H_f_mean.y.values[0]))

    a_np = np.nan_to_num(a_field.values)
    fd_np = np.nan_to_num(fd.values)
    # The flux divergence is built from the OBSERVED (already bridged) mean
    # thickness, so it must not pass through the operator a second time; it goes
    # on the observation side. `flux_in_operator=True` restores the pre-fix
    # double count for the audit A/B only.
    if flux_in_operator:
        known_in = np.where(fit, a_np - fd_np, 0.0)
        known_out = np.zeros_like(known_in)
    else:
        known_in = np.where(fit, a_np, 0.0)
        known_out = np.where(fit, -fd_np, 0.0)
    obs = np.where(fit, np.nan_to_num(dHdt_obs.values), 0.0)
    wv = np.where(fit, w.values, 0.0)
    wv = wv / max(float(wv.max()), 1e-30)

    # ---- the operator
    if transfer not in ("flotation", "normalized"):
        raise ValueError(
            f"transfer must be 'flotation' or 'normalized', got {transfer!r}")
    H_ref = float(np.nanmedian(H_f_mean.values[fit]))
    eb = eta_bar
    if bridging:
        u0x = float(np.nanmedian(vxm.values[fit]))
        u0y = float(np.nanmedian(vym.values[fit]))
        if eta_field is not None:
            ef = np.asarray(eta_field.broadcast_like(H_f_mean).values, float)
            fin = np.isfinite(ef) & fit
            if fin.any():
                eb = float(np.nanmedian(ef[fin]))
        # padded grid: the operator's downstream footprint is long, so the
        # field is mirror-doubled before transforming (as StubblefieldForward).
        build = (bridging_transfer_multiplier if transfer == "flotation"
                 else normalized_bridging_multiplier)
        D = build(2 * ny, 2 * nx, dx, dy, H=H_ref, ux_myr=u0x, uy_myr=u0y,
                  eta_bar=eb, alpha_scale=alpha_scale, rho_i=rho_i,
                  rho_w=rho_w)
        if log_every:
            print(f"    bridging operator [{transfer}]: H_ref={H_ref:.0f} m  "
                  f"u0=({u0x:.0f}, {u0y:.0f}) m/yr  eta_bar={eb:.2e} Pa s  "
                  f"alpha_scale={alpha_scale:g}  "
                  f"flux_{'inside' if flux_in_operator else 'outside'}",
                  flush=True)
    else:
        D = None
        u0x = u0y = 0.0

    # NB: there is deliberately no melt-RATE term here -- no `-tau*div(m u)`,
    # no advective shift of the recovered melt. The E2a departure from flotation
    # that motivated one (`eps = -f_b*tau*mdot`, tau ~ 4.9 a, 2026-08-20) is not
    # shelf physics: it is the Elmer twin's explicit Stokes<->free-surface
    # splitting, `eps = 0.996*M*dt`, first order in the TIMESTEP and necessary
    # and sufficient for the whole across-flow-channel dipole (2026-08-21,
    # `elmer_synth/scripts/diagnose_dt_splitting.py`). A real shelf has no dt.
    # The only physical displacement of the surface expression is the advective
    # phase already inside T(k) via `alpha`, which is sub-metre over the
    # resolvable band. Do not re-add a fitted lag.

    t_known = torch.from_numpy(np.ascontiguousarray(known_in, dtype=np.float64))
    t_kout = torch.from_numpy(np.ascontiguousarray(known_out, dtype=np.float64))
    t_obs = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float64))
    t_w = torch.from_numpy(np.ascontiguousarray(wv, dtype=np.float64))
    t_fit = torch.from_numpy(np.ascontiguousarray(fit))
    t_D = None if D is None else torch.from_numpy(np.ascontiguousarray(D))

    def _pad2x(a):
        a = torch.cat([a, torch.flip(a, dims=[0])], dim=0)
        return torch.cat([a, torch.flip(a, dims=[1])], dim=1)

    def bridge(field):
        if t_D is None:
            return field
        F = torch.fft.fft2(_pad2x(field).to(torch.complex128))
        return torch.fft.ifft2(t_D * F).real[:ny, :nx]

    # Warm start at the HYDROSTATIC inverse (m = dHdt_obs + div - a, i.e. the
    # Eulerian answer). The bridging fit is a correction to hydrostatics, not a
    # search from nothing: this starts the optimiser inside the right basin,
    # and it makes the `bridging=False` case exact at iteration 0 (the residual
    # is identically zero there) instead of merely converging toward it.
    m0 = np.where(fit, obs + fd_np - a_np, 0.0)
    wsum = float(t_w.sum())

    # --- The objective is QUADRATIC in m, so solve it as one.
    #
    # L(m) = sum_x w (T{m+a} - div(H_f u) - obs)^2 / sum(w) + lam*||grad m||^2
    #                                           +  ridge*||m||^2
    #
    # Adam was the wrong tool here: on a quadratic whose condition number is set
    # by 1/|D| (|D| ~ 0.004 near Nyquist, so ~250x amplification), a fixed step
    # either crawls or oscillates in exactly the worst-conditioned modes -- which
    # is what drove the E2a `clean` run to a rate-space var_explained of -639.
    # Conjugate gradients solves it exactly instead, and `ridge` is the Wiener
    # term: with uniform weights the minimiser is D*/(|D|^2 + ridge), bounding
    # the deconvolution gain at 1/(2*sqrt(ridge)) where the operator is blind.
    #
    # The gradient of a quadratic is affine, so the Hessian-vector product is
    # H v = g(v) - g(0) exactly -- no hand-derived adjoint of the mirror-padded
    # spectral operator (the easy thing to get wrong) is needed.
    def _grad(vec: torch.Tensor) -> torch.Tensor:
        mv = vec.detach().clone().requires_grad_(True)
        S = torch.where(t_fit, mv + t_known, torch.zeros_like(mv))
        resid = torch.where(t_fit, bridge(S) + t_kout - t_obs,
                            torch.zeros_like(mv))
        loss = (t_w * resid ** 2).sum() / wsum
        if lam:
            gx = mv[:, 1:] - mv[:, :-1]
            gy = mv[1:, :] - mv[:-1, :]
            loss = loss + lam * ((gx ** 2).mean() + (gy ** 2).mean())
        if ridge:
            loss = loss + ridge * (mv ** 2).mean()
        (g,) = torch.autograd.grad(loss, mv)
        return g

    zero = torch.zeros((ny, nx), dtype=torch.float64)
    g0 = _grad(zero)                 # g(0) = -rhs
    rhs = -g0

    def _hess(v: torch.Tensor) -> torch.Tensor:
        return _grad(v) - g0

    m = torch.from_numpy(np.ascontiguousarray(m0, dtype=np.float64))
    r = rhs - _hess(m)
    p = r.clone()
    rs = float((r * r).sum())
    rhs_norm = float((rhs * rhs).sum()) or 1.0
    hist = [rs / rhs_norm]
    n_cg = 0
    for it in range(int(iters)):
        if rs / rhs_norm <= converge_tol:
            break
        Hp = _hess(p)
        pHp = float((p * Hp).sum())
        if pHp <= 0:                 # numerically indefinite: stop, keep m
            break
        alpha = rs / pHp
        m = m + alpha * p
        r = r - alpha * Hp
        rs_new = float((r * r).sum())
        p = r + (rs_new / rs) * p
        rs = rs_new
        n_cg = it + 1
        hist.append(rs / rhs_norm)
        if log_every and (it % log_every == 0):
            print(f"    cg {it:4d}  ||r||^2/||b||^2 {rs / rhs_norm:.4e}",
                  flush=True)
    if log_every:
        print(f"    cg done: {n_cg} iters, rel resid {rs / rhs_norm:.3e}",
              flush=True)

    with torch.no_grad():
        S = torch.where(t_fit, m + t_known, torch.zeros_like(m))
        fit_rate = (bridge(S) + t_kout).numpy()
        m_out = m.numpy()

    melt = np.where(fit, m_out, np.nan)
    fit_rate = np.where(fit, fit_rate, np.nan)

    o = obs[fit]
    r = fit_rate[fit] - o
    ve = float(1.0 - np.nanvar(r) / np.nanvar(o)) if np.nanvar(o) > 0 else np.nan

    coords = {"y": H_f_mean.y.values, "x": H_f_mean.x.values}
    ds = xr.Dataset(
        {
            "melt_rate": xr.DataArray(melt, dims=("y", "x"), coords=coords),
            "dHdt_obs": dHdt_obs,
            "dHdt_fit": xr.DataArray(fit_rate, dims=("y", "x"), coords=coords),
            "H_f_mean": H_f_mean,
            "flux_div": fd,
            "weight": xr.DataArray(np.where(fit, wv, np.nan), dims=("y", "x"),
                                   coords=coords),
        },
        attrs={
            "method": "budget+bridging monolithic fit",
            "bridging": int(bool(bridging)),
            "transfer": transfer if bridging else "identity",
            "flux_in_operator": int(bool(flux_in_operator)),
            "H_ref_m": H_ref,
            "u0x_myr": u0x,
            "u0y_myr": u0y,
            "eta_bar": eb,
            "alpha_scale": alpha_scale,
            "lam": lam,
            "ridge": ridge,
            "cg_iters": n_cg,
            "cg_rel_resid": hist[-1] if hist else np.nan,
            "iters": int(iters),
            "weight_kind": weight,
            "fit_n_cells": int(fit.sum()),
            "fit_var_explained": ve,
            "fit_rms_resid_m_yr": float(np.sqrt(np.nanmean(r ** 2))),
            "fit_rms_obs_m_yr": float(np.sqrt(np.nanmean(o ** 2))),
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt",
        },
    )
    return ds
