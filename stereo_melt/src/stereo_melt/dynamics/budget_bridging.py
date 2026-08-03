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

They are not rivals. Measured on the PIG production geometry, the Stubblefield
multiplier ``|M_h(k)|`` is **flat** for every wavelength above ~10 km (0.2947 at
20 km, 0.2966 at 40 km, 0.2972 at 170 km) and rolls off only at short scales
(0.104 at 3H = 1.4 km). That plateau *is* the hydrostatic response, and the only
degenerate mode is the exact ``k = 0`` bin, which
:mod:`~stereo_melt.dynamics.linear_perturbation` zeroes by hand so that
perturbations carry no DC offset. So the transfer factorises as

.. math:: T(k) = (\text{hydrostatic}) \times D(k), \qquad D(k\to 0) = 1

and :math:`D` — the *normalised* multiplier — is a pure correction to
hydrostatics. That is the seam this module exploits.

The forward model
-----------------
Thickness responds to the mass budget,

.. math:: S = \dot m + \dot a - \nabla\!\cdot(H_f u)

(Shean sign: negative :math:`\dot m` = melt), and the thickness we *observe* —
by hydrostatic inversion of the DEM stack — is the bridged version of it:

.. math:: \left(\partial H_f/\partial t\right)_{\rm obs} = D\{S\}.

Everything is in **thickness space**, so no hydrostatic factor appears; it has
already been divided out of :math:`D`.

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
        \bigl( D\{\dot m + \dot a - \nabla\!\cdot(H_f u)\}(x)
               - \widehat{\partial_t H_f}(x) \bigr)^2
        + \lambda \lVert \nabla \dot m \rVert^2 .

Consequences worth stating
--------------------------
* **No high-pass, no anomaly, no prior, no fusion.** The reference state never
  enters: a free per-pixel intercept was profiled out analytically.
* **The mean is identified.** :math:`D(0)` is set to **1** here (not 0 as in the
  anomaly operator) because at :math:`k\to0` there is no damping; the melt's
  spatial mean is then fixed by the budget exactly as in the Eulerian solver.
* **``bridging=False`` reproduces the Eulerian solver exactly.** With
  :math:`D = I` the minimiser is pixelwise
  :math:`\dot m = \widehat{\partial_t H_f} + \nabla\!\cdot(H_f u) - \dot a`,
  which is Shean Eq. 10 — for any positive weights. Gate:
  ``tests/gate_budget_bridging_identity.py``.
* **The ill-conditioning is at SHORT wavelengths**, where :math:`|D|` falls to
  ~1.4% by :math:`\lambda = 600` m, so :math:`D^{-1}` amplifies ~70x. That is
  what ``lam`` regularises, and it is the honest location of the difficulty —
  the long-wavelength band the legacy high-pass removed was never the problem.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import dh_dt, flux_divergence

SECONDS_PER_YEAR = 86400.0 * 365.25

__all__ = ["budget_bridging_melt_rate", "normalized_bridging_multiplier"]


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
        ``False`` sets :math:`D = I`, which reproduces the Eulerian solver
        exactly and is the correctness gate. ``True`` applies the normalised
        Stubblefield damping.
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

    known = np.where(fit, np.nan_to_num(a_field.values) - np.nan_to_num(fd.values), 0.0)
    obs = np.where(fit, np.nan_to_num(dHdt_obs.values), 0.0)
    wv = np.where(fit, w.values, 0.0)
    wv = wv / max(float(wv.max()), 1e-30)

    # ---- the operator
    if bridging:
        H_ref = float(np.nanmedian(H_f_mean.values[fit]))
        u0x = float(np.nanmedian(vxm.values[fit]))
        u0y = float(np.nanmedian(vym.values[fit]))
        eb = eta_bar
        if eta_field is not None:
            ef = np.asarray(eta_field.broadcast_like(H_f_mean).values, float)
            fin = np.isfinite(ef) & fit
            if fin.any():
                eb = float(np.nanmedian(ef[fin]))
        # padded grid: the operator's downstream footprint is long, so the
        # field is mirror-doubled before transforming (as StubblefieldForward).
        D = normalized_bridging_multiplier(
            2 * ny, 2 * nx, dx, dy, H=H_ref, ux_myr=u0x, uy_myr=u0y,
            eta_bar=eb, alpha_scale=alpha_scale, rho_i=rho_i, rho_w=rho_w)
        if log_every:
            print(f"    bridging operator: H_ref={H_ref:.0f} m  "
                  f"u0=({u0x:.0f}, {u0y:.0f}) m/yr  eta_bar={eb:.2e} Pa s",
                  flush=True)
    else:
        D = None
        H_ref = float(np.nanmedian(H_f_mean.values[fit]))
        u0x = u0y = 0.0
        eb = eta_bar

    t_known = torch.from_numpy(np.ascontiguousarray(known, dtype=np.float64))
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
    m0 = np.where(fit, obs - known, 0.0)
    wsum = float(t_w.sum())

    # --- The objective is QUADRATIC in m, so solve it as one.
    #
    # L(m) = sum_x w (D{m+c} - obs)^2 / sum(w)  +  lam*||grad m||^2
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
        resid = torch.where(t_fit, bridge(S) - t_obs, torch.zeros_like(mv))
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
        fit_rate = bridge(S).numpy()
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
