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

import warnings

import numpy as np
import xarray as xr

from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import dh_dt, flux_divergence

SECONDS_PER_YEAR = 86400.0 * 365.25

__all__ = ["budget_bridging_melt_rate", "bridging_transfer_multiplier",
           "normalized_bridging_multiplier", "strip_mode_design"]


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
    :math:`k \to 0`, so :math:`T \to \delta/(f_b(\delta+1)) = 1` exactly.
    :meth:`~stereo_melt.dynamics.linear_perturbation.LinearPerturbation.steady_state_kernel`
    now carries that limit itself, so the pin is a redundant defensive
    assertion: a finite DC bin that disagrees with 1 raises rather than being
    silently overwritten. It stays non-optional because the kernel's DC value
    is :math:`\pm\infty` at the single degenerate :math:`\lambda_0 = 0`,
    where the ratio is undefined even though its limit is still 1
    (:math:`\lambda_0` cancels).

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
    dc = complex(T[0, 0])
    if np.isfinite(dc) and abs(dc - 1.0) > 1e-8:
        raise ValueError(
            f"steady_state_kernel's DC flotation departure is {dc!r}, not the "
            "analytic 1: the k=0 limit of G_h/(f_b (G_h - G_s)) disagrees with "
            "the transfer this operator is built from")
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
    plateau_rtol: float = 0.05,
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

    The ``k = 0`` bin is set to **1**: a relative damping is unity where the
    shelf floats hydrostatically, and pinning it is what makes the melt mean
    identifiable here. The plateau it is normalised against is measured over
    the **resolved** (non-DC) bins, so this stays a statement about what the
    grid can actually see rather than about
    :mod:`.linear_perturbation`'s analytic :math:`k=0` limit.

    That pin is only self-consistent if :math:`|M_h|` actually plateaus at
    the longest resolved wavelengths. Whether it does depends on ``eta_bar``
    as much as on geometry (flat at :math:`10^{13}` Pa s, **not** at the
    :math:`10^{14}` default on PIG or E2a geometry, where :math:`|D|` at
    :math:`\lambda = 40` km is already 0.77–0.85): if the lowest non-zero
    wavenumber bins fall more than ``plateau_rtol`` below the modulus maximum,
    a ``ValueError`` is raised rather than a step at :math:`k = 0` silently
    biasing the melt mean.
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
    flat = np.abs(M).ravel().copy()
    flat[0] = np.nan                     # DC is pinned below, not a resolved bin
    if not np.isfinite(flat).any():
        raise ValueError("bridging multiplier is entirely non-finite")
    plateau = M.ravel()[int(np.nanargmax(flat))]
    if not np.isfinite(plateau) or abs(plateau) <= 0:
        raise ValueError(f"degenerate bridging multiplier (plateau={plateau})")
    low_k = ([M[0, 1], M[0, -1]] if nx > 1 else []) \
        + ([M[1, 0], M[-1, 0]] if ny > 1 else [])
    low_ratio = float(np.nanmin(np.abs(low_k))) / abs(plateau)
    if not low_ratio >= 1.0 - plateau_rtol:
        raise ValueError(
            "bridging multiplier has no long-wavelength plateau: |M_h| at the "
            f"lowest resolved wavenumbers is {low_ratio:.3f} of its maximum "
            f"(tolerance {plateau_rtol:g}), so D[0, 0] = 1 would bias the melt "
            f"mean; lower eta_bar (={eta_bar:.2e} Pa s) or use "
            "transfer='flotation'")
    D = M / plateau
    D[0, 0] = 1.0 + 0.0j
    return D


def strip_mode_design(
    stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    *,
    modes: tuple = ("offset", "tilt"),
    min_count: int = 3,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    estimator=None,
):
    r"""Rate-space design fields of the per-strip survey errors — the
    **coloured noise model** of the budget inverse.

    Every DEM strip :math:`k` leaves a residual offset :math:`c_k` and plane
    :math:`a_k (x - x_k) + b_k (y - y_k)` over its footprint after the tilt
    fit. Through the per-pixel OLS slope weights
    :math:`\omega_{ik} = (t_k - \bar t_i)/S_{tt,i}` they enter the observed
    thickness rate, and through the mean-thickness weights :math:`1/n_i` (then
    the flux divergence) they enter :math:`\nabla\cdot(H_f u)` — a coherent,
    strip-shaped error with spikes at footprint edges that no white-noise
    model describes. Per strip and component :math:`j` the field

    .. math::
        G_{kj}(x) = \frac{1}{f_b}\Bigl[\omega_k(x)\,\phi_{kj}(x)
                    + \nabla\cdot\bigl(\tfrac{\phi_{kj}(x)}{n(x)}\,u\bigr)\Bigr]

    maps a unit error of that component (freeboard units: m for the offset,
    m m⁻¹ for the tilts) onto the observation side ``dHdt_obs + div(H_f u)``
    (thickness rate). With these as nuisance columns the data model becomes
    :math:`d = A m + G\theta + \varepsilon`; estimating :math:`\theta`
    jointly with the melt under a prior :math:`\theta\sim N(0, \Sigma)` is
    exactly the coloured covariance :math:`\sigma^2 W^{-1} + G\Sigma G^T`,
    and conditioned on :math:`\theta` the residual is white again.

    Returns ``(G, strip_index, component)``: ``G`` of shape ``(n_modes, ny,
    nx)``, and per-mode labels (strip index into ``stack.time``, component
    name in ``offset, tilt_x, tilt_y``). The offset modes are degenerate with
    a uniform thickness trend through offsets linear in strip time — the
    prior on :math:`\theta` is what anchors them (as control anchors the tilt
    fit); ``modes=("tilt",)`` drops them.
    """
    fb = 1.0 - rho_i / rho_w
    v = np.isfinite(stack.values)
    t = ((stack.time.values - stack.time.values[0]) / np.timedelta64(1, "D")
         / 365.25).astype(float)
    n = v.sum(0)
    ok = n >= min_count
    s1, s2 = np.tensordot(np.stack([t, t * t]), v.reshape(len(t), -1),
                          axes=1).reshape(2, *n.shape)
    tbar = np.where(ok, s1 / np.maximum(n, 1), 0.0)
    stt = np.maximum(s2 - 2.0 * tbar * s1 + n * tbar ** 2, 0.0)
    inv_stt = np.where(ok & (stt > 0), 1.0 / np.maximum(stt, 1e-30), 0.0)
    inv_n = np.where(ok, 1.0 / np.maximum(n, 1), 0.0)
    x2d, y2d = np.meshgrid(stack.x.values.astype(float), stack.y.values.astype(float))
    comps = []
    if "offset" in modes:
        comps.append("offset")
    if "tilt" in modes:
        comps += ["tilt_x", "tilt_y"]
    G, sidx, cname = [], [], []
    for k in range(stack.sizes["time"]):
        F = v[k] & ok
        if not F.any():
            continue
        xc, yc = float(x2d[F].mean()), float(y2d[F].mean())
        omega = np.where(F, (t[k] - tbar) * inv_stt, 0.0)
        for c in comps:
            phi = (F.astype(float) if c == "offset" else
                   F * (x2d - xc) if c == "tilt_x" else F * (y2d - yc))
            obs_part = omega * phi
            Herr = xr.DataArray(phi * inv_n, dims=("y", "x"),
                                coords={"y": stack.y.values, "x": stack.x.values})
            div_part = np.nan_to_num(flux_divergence(Herr, vx, vy, estimator=estimator).values)
            G.append((obs_part + div_part) / fb)
            sidx.append(k)
            cname.append(c)
    return np.stack(G), np.array(sidx), np.array(cname)


#: Coefficient in the truth-free rule ``lam = LAM_SIGMA2_COEF * sigma2_est``.
#:
#: **lam IS DIMENSIONLESS, and this rule is an EMPIRICAL calibration, not a
#: dimensional identity.** An earlier version of this note claimed lam "carries
#: the same units as the noise variance" so that "the natural scale is sigma^2
#: itself". That was wrong. The loss is
#: ``mean_w(resid^2) + lam*(mean(gx^2) + mean(gy^2))``; the residual is a
#: thickness rate in m/yr, and gx/gy are differences of the melt field, also in
#: m/yr. Both terms are (m/yr)^2, so lam is a pure number and cannot carry the
#: units of a variance. What the 2026-08-22 oracle study actually established is
#: that ``lam = sigma2_est`` landed within 6 % of the oracle lam on the
#: mixed-field rungs AT THAT STUDY'S POSTING AND NOISE LEVEL -- a fit, not a
#: derivation. The 08-29 pigreal tier is consistent with a scaling but not with
#: THIS one: raising the noise variance ~25x moved the optimal lam by ~30-300x,
#: which is not the linear relation the rule assumes.
#:
#: **lam is POSTING-SPECIFIC, and this matters in practice.** gx/gy are bare
#: per-pixel differences with no dx or dy, so the regulariser penalises
#: curvature PER PIXEL, not per metre. The same physical melt field differenced
#: on a 500 m posting gives gx twice the 250 m value, so the regularisation term
#: scales as res^2 while sigma2_est does not track it (its slope part
#: rmse^2/S_tt is posting-independent; only the divergence part
#: u^2 rmse^2/(2 n dx^2) carries 1/dx^2). A lam tuned at 250 m therefore does
#: NOT transfer to 125 m or 500 m, and every published lam on this project is
#: specific to the posting it was tuned at. Re-tune when you change resolution.
#:
#: The posting-invariant form would divide the differences by dx and dy, making
#: the regulariser a true squared gradient in (m/yr)/m and lam carry m^2. That
#: is deliberately NOT done here: it would change every solved field and
#: invalidate every calibrated lam on record. Kept as a named constant so the
#: calibration is one edit, not a magic number scattered across drivers.
LAM_SIGMA2_COEF = 1.0

#: Sentinel for the REQUIRED ``lam`` argument of
#: :func:`budget_bridging_melt_rate`. There is no defensible universal default:
#: a fixed lam is not safe across noise levels, and the ``"auto"`` rule above is
#: near-oracle only under WHITE noise -- which PIG's ~4 km correlated strip
#: error is not. Omitting lam must therefore be an error, not a silent choice.
_LAM_REQUIRED = object()


def _estimate_sigma2_white(reg, w, weight, H_f_stack, vxm, vym, dx, wv, fit):
    r"""Truth-free white-noise variance of the thickness-rate observation.

    Two independent contributions at a unit-weight pixel, both built from the
    per-pixel regression rmse so no truth is used:

    * the slope variance ``rmse^2 / S_tt`` (temporal leverage of the fit), and
    * the divergence of the mean-thickness noise,
      ``|u|^2 rmse^2 / (2 n dx^2)``, which is what the flux term propagates.

    Returned in (thickness-rate)^2, averaged over the fit cells with the
    solver's own weights -- i.e. directly comparable to the ``lam`` that
    multiplies ``mean(|grad m|^2)`` in the same loss.
    """
    rm = np.nan_to_num(reg["rmse"].values)
    cnt = np.nan_to_num(reg["count"].values).astype(float)
    stt = np.nan_to_num((w if weight == "leverage"
                         else _temporal_leverage(H_f_stack)).values)
    u2 = np.nan_to_num(vxm.values ** 2 + vym.values ** 2)
    var_i = np.where(stt > 0, rm ** 2 / np.maximum(stt, 1e-30), 0.0) \
        + u2 * rm ** 2 / (2.0 * np.maximum(cnt, 1) * dx ** 2)
    return float(np.mean((wv * var_i)[fit]))


def strip_prior_from_residual_planes(
    planes,
    dem_ids,
    strip_index: np.ndarray,
    component: np.ndarray,
    *,
    offset_var=None,
    per_strip: bool = False,
    floor_frac: float = 0.05,
    robust: bool = True,
    max_sd: float | None = 5.0,
    min_n: int = 100,
    return_summary: bool = False,
):
    r"""``tau^2`` per strip mode from per-strip control residual planes.

    The truth-free source of the coloured-noise prior :math:`\Sigma` that
    :func:`strip_mode_design` needs. ``planes`` is the table from
    :func:`stereo_melt.coregister.alignment_quality.residual_planes_for_strips`
    (``dem_id, ax, ay, se_ax, se_ay, ...``): each strip's signed
    ``aligned DEM - control`` plane, fitted against the altimetry pc_align was
    fed. Against the processing twin these planes recover the true residual
    tilt per strip at corr 0.93 / 0.75 and its across-strip variance to
    1.08x / 1.6x, where every estimator built on the DEM stack alone is 2x-167x
    low or diverges (the per-strip tilt is not identifiable from the stack:
    loosening the tilt prior grows the estimates without bound while their
    correlation with truth falls). Independent control has no such degeneracy.

    Population rule per tilt component (default, ``robust=True``)::

        tau2_c = max( scale(est_c)^2 - median(se_c^2),  floor_frac * scale^2 )

    with ``scale = 1.4826 * MAD`` -- the method-of-moments variance component
    with the fit's own estimator variance removed, well-posed here precisely
    because ``est`` is a genuine ML fit against independent data.
    ``robust=False`` uses ``var`` and ``mean(se^2)`` (identical on Gaussian
    strips, e.g. the twin). Robust is the default because real ASP roots
    contain ALIGNMENT FAILURES: on PIG's 513-strip canon ~15 % of strips carry
    residual offsets of 60-207 m and fit scatter of 68-175 m, none of them in
    ``BAD_STRIPS``, and they carried 100 % of the non-robust variance (rms
    2.3e-3 m/m against a robust 2.0e-6). Strips failing either QC gate are
    excluded from the tau^2 population and get the population value, but the
    two gates mean DIFFERENT THINGS and the summary keeps them apart:

    ``qc_alignment_failures`` / ``n_qc_alignment_failures``
        ``sd > max_sd``: a residual scatter against control of metres means
        the alignment did not converge. **This is the only list that is a
        ``BAD_STRIPS`` candidate** -- a genuinely broken strip the stack-side
        screens do not catch.
    ``qc_low_control`` / ``n_qc_low_control``
        ``n < min_n`` and NOT an alignment failure: a well-aligned strip whose
        control cloud merely clips its footprint. Its plane is too weakly
        determined to vote in the population, but the STRIP is fine. Do NOT
        put these in ``BAD_STRIPS``; dropping them discards good epochs. On
        PIG's canon the two sets were 15 and 15 -- the 15 failures were added
        to ``BAD_STRIPS``, the 15 low-control strips (scatter 0.36 m, pc_align
        ``end_p50`` 0.25 m) were deliberately KEPT.

    The two lists are disjoint and their counts sum to ``n_qc_dropped``;
    ``qc_dropped`` remains as the pooled "excluded from the tau^2 population"
    view and is NOT a bad-strip list. ``per_strip=True`` instead uses
    ``max(est_k^2 - se_k^2, floor)`` for each QC-passing strip (noisier).

    ``offset`` modes are NOT filled from control: the control lives on the
    static apron where pc_align pins the offset, while the shelf carries
    datum-stage residual (tide / IBE / MDT) the control never sees -- on the
    twin the control-derived offset variance is 0.07x the true one. Pass
    ``offset_var`` (a float, or one value per epoch of ``dem_ids``) from the
    datum error budget; it is an error to request offset modes without it.

    ``dem_ids`` is the stack's per-epoch ``dem_id`` coordinate, so
    ``strip_index`` (into the stack's time axis) resolves to a row of ``planes``.

    The planes are the residual AFTER ALIGNMENT; the stack the inverse sees is
    the one after the tilt fit, which removes only a little of them (twin: 17 %
    of the x-plane variance, 3 % of y, 93 % of the offset), so ``tau2`` is
    conservative by that margin. Validated end-to-end on the twin's
    tilt-corrected stack: the control-derived prior tracks the oracle
    (true-residual) prior within ~2 % on every melt metric at every ``lam``.
    """
    strip_index = np.asarray(strip_index, int)
    component = np.asarray(component)
    dem_ids = np.asarray([str(d) for d in np.asarray(dem_ids)])
    pl = planes.set_index("dem_id") if "dem_id" in getattr(planes, "columns", ()) else planes
    est = {"tilt_x": ("ax", "se_ax"), "tilt_y": ("ay", "se_ay")}
    # QC: a plane fitted through metres of scatter, or through too few
    # control points, is an alignment failure, not a survey error statistic.
    fitted = np.isfinite(pl["ax"].to_numpy(float))
    failed_align = np.zeros(fitted.shape, bool)
    if max_sd is not None and "sd" in pl.columns:
        failed_align = pl["sd"].to_numpy(float) > max_sd
    low_ctl = np.zeros(fitted.shape, bool)
    if min_n and "n" in pl.columns:
        low_ctl = pl["n"].to_numpy(float) < min_n
    qc = fitted & ~failed_align & ~low_ctl
    align_ids = [str(d) for d in pl.index[fitted & failed_align]]
    lowctl_ids = [str(d) for d in pl.index[fitted & low_ctl & ~failed_align]]
    dropped = [str(d) for d in pl.index[fitted & ~qc]]
    pop, summary = {}, {}
    for comp, (col, secol) in est.items():
        v = pl[col].to_numpy(float)
        se2 = pl[secol].to_numpy(float) ** 2
        ok = qc & np.isfinite(v) & np.isfinite(se2)
        if ok.sum() < 3:
            raise ValueError(f"need >=3 QC-passing strips with a fitted {col} plane, got {int(ok.sum())}")
        if robust:
            scale = 1.4826 * float(np.median(np.abs(v[ok] - np.median(v[ok]))))
            raw, corr = scale ** 2, float(np.median(se2[ok]))
        else:
            raw, corr = float(np.var(v[ok])), float(np.mean(se2[ok]))
        pop[comp] = max(raw - corr, floor_frac * raw)
        summary[comp] = dict(var_raw=raw, mean_se2=corr, tau2=pop[comp], n=int(ok.sum()),
                             var_nonrobust=float(np.var(v[ok])))
    summary["n_qc_dropped"] = len(dropped)
    summary["qc_dropped"] = dropped
    summary["n_qc_alignment_failures"] = len(align_ids)
    summary["qc_alignment_failures"] = align_ids
    summary["n_qc_low_control"] = len(lowctl_ids)
    summary["qc_low_control"] = lowctl_ids
    tau2 = np.empty(component.size, float)
    for m, (k, comp) in enumerate(zip(strip_index, component)):
        if comp == "offset":
            if offset_var is None:
                raise ValueError(
                    "offset strip modes need offset_var: control cannot see the shelf "
                    "datum residual (twin: control-derived offset variance = 0.07x true). "
                    "Pass the datum error budget, or use modes=('tilt',).")
            ov = np.broadcast_to(np.asarray(offset_var, float), (dem_ids.size,))
            tau2[m] = float(ov[k])
            continue
        col, secol = est[comp]
        val = pop[comp]
        if per_strip and dem_ids[k] in pl.index and dem_ids[k] not in dropped:
            e = float(pl.at[dem_ids[k], col])
            s2 = float(pl.at[dem_ids[k], secol]) ** 2
            if np.isfinite(e) and np.isfinite(s2):
                val = max(e * e - s2, floor_frac * pop[comp])
        tau2[m] = val
    return (tau2, summary) if return_summary else tau2


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
    # S_tt = sum t^2 ok - 2 mean_t sum t ok + mean_t^2 n: one (2, time) x
    # (time, pixels) product over the boolean mask, no (time, y, x) float
    # temporaries (the 513-epoch stack would need several GB of them)
    s1, s2 = np.tensordot(np.stack([t, t * t]), ok.reshape(len(t), -1),
                          axes=1).reshape(2, *n.shape)
    mean_t = np.divide(s1, n, out=np.zeros_like(s1), where=n > 0)
    stt = np.maximum(s2 - 2.0 * mean_t * s1 + n * mean_t ** 2, 0.0)
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
    n_bins: int = 1,
    blend_px: float = 8.0,
    flux_restored: bool = False,
    restore_kwargs: dict | None = None,
    strip_modes: np.ndarray | None = None,
    strip_prior: np.ndarray | float | None = None,
    sigma2: float | str = "auto",
    lam: float | str = _LAM_REQUIRED,
    ridge: float = 0.0,
    iters: int = 300,
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
    flux_restored, restore_kwargs
        **The corrected forward model (2026-08-24, "monolithic v2").** The
        historical model puts the flux divergence of the OBSERVED (bridged)
        mean thickness outside the operator; but the exact relation is
        ``dH_f/dt = T{m + a - div(H u)}`` with ``H = T^{-1}{H_f}`` INSIDE the
        flux term — its unregularised solution is precisely the
        restore-then-budget solver, and the historical placement is what
        over-reads along-flow channels (lifting the velocity-carried term and
        the transfer gradient). ``flux_restored=True`` computes the flux
        divergence from the RESTORATION-FILTERED mean thickness (the bounded,
        flow-projected 1/T of :func:`.bridging_restoration.bridging_restoration_filter`,
        options via ``restore_kwargs``: ``lift_cap``, ``band_lam_min``,
        ``lift_umax_myr``) and places it inside ``T`` — so the fit shares
        restore-then-budget's physics while lam regularises the
        short-wavelength deconvolution properly instead of a hard band limit.
        The restoration is PER BIN, exactly as
        :func:`~.bridging_restoration.restored_budget_melt_rate`: one filter
        per operator bin from that bin's ``(H, u_x, u_y, eta)`` centroid
        (so it is the bounded inverse of the very transfer the operator
        applies there), blended with the same partition-of-unity weights,
        and the ``lift_umax_myr`` trunk guard gives bins whose centroid speed
        exceeds it the identity filter (no lift over the fast crevassed
        trunk). Attrs record ``flux_restored_bins`` and
        ``flux_restored_guarded_bins``.
    n_bins, blend_px
        **Local operator.** With ``n_bins > 1`` the fit cells are clustered by
        ``(H, u_x, u_y[, eta])`` (:func:`~.stubblefield_forward._kmeans_geometry`,
        deterministic), one transfer is built per bin from the bin centroid —
        its own thickness, advection and (with ``eta_field``) its own secant
        viscosity — each is applied to the WHOLE padded domain (the operator is
        not spatially compact: long-wavelength modes advect far before they
        damp), and the responses are recombined with Gaussian-smoothed
        (``blend_px``) partition-of-unity weights, exactly as
        :class:`~.stubblefield_forward.BlendedStubblefieldForward`. ``n_bins=1``
        (default) is the single global multiplier at the median geometry.
        This is the answer to "T depends on the local H" (λ = 2H is 5
        elements per wavelength at H = 333 m but 3 at 460 m) and to a
        shelf-wide η spanning a decade.
    strip_modes, strip_prior, sigma2
        **Coloured noise model.** ``strip_modes`` is the ``(n_modes, ny, nx)``
        design from :func:`strip_mode_design`; the per-strip error amplitudes
        ``theta`` are estimated JOINTLY with the melt under the prior
        ``theta_j ~ N(0, strip_prior_j)`` (variances in freeboard units —
        m² for offsets, (m/m)² for tilts; a scalar applies to every mode),
        with the white-noise variance ``sigma2`` (thickness-rate², at a
        unit-weight pixel) setting the prior's weight against the data term:
        ``"auto"`` estimates it from the per-pixel regression residuals and
        their flux divergence (it still contains the strip errors, so it is an
        upper bound). Returns ``theta`` and the fitted coherent-error field
        ``strip_error_rate`` = G theta on the observation side.
    eta_bar, alpha_scale, eta_field
        Viscosity for the bridging operator; ``eta_field`` (a map, e.g. from the
        momentum-balance inversion) overrides the scalar via its masked median.
    lam
        Tikhonov weight on :math:`\\lVert\\nabla \\dot m\\rVert^2`. **Required**
        -- there is no safe default, so omitting it raises rather than
        silently picking one. Two valid choices: a float, to reproduce a
        specific published run, or ``"auto"``, which sets it from the data as
        ``LAM_SIGMA2_COEF * sigma2_est`` using the truth-free noise estimate in
        :func:`_estimate_sigma2_white` (the resolved float comes back as the
        ``lam`` attr and the estimate as ``sigma2_est``). Neither is
        universally right. A fixed lam is NOT safe across noise levels: on the
        08-29 pigreal (correlated-error) tier the same operator scores nrmse
        3.76 at lam 1e-3 -- worse than Eulerian, flux x2.58 -- and 0.95-1.09 at
        lam 0.032-0.32, which beats Eulerian on every metric. ``"auto"`` is
        near-oracle under WHITE noise only: it is built on a white variance
        estimate, so spatially correlated error (PIG's strip residual is
        coherent at ~4 km) inflates it and over-damps the melt by 300-3000x on
        that same tier. It warns when it can detect that, but the check is
        one-sided. And ``"auto"`` is an EMPIRICAL calibration, not a
        dimensional identity -- lam is dimensionless (see
        :data:`LAM_SIGMA2_COEF`).

        **Any lam, fixed or auto, is specific to the POSTING it was tuned at.**
        The smoothness term differences the melt field per pixel with no
        ``dx``, so it measures curvature per pixel: the same physical field on
        a 500 m grid produces twice the gradient it does on 250 m, and the
        regularisation term scales as ``res^2``. A lam carried across a
        resolution change silently changes how much the melt is smoothed.
    ridge
        Wiener weight on :math:`\\lVert\\dot m\\rVert^2`. With uniform weights the
        minimiser is :math:`D^*/(|D|^2 + \\text{ridge})`, so the deconvolution
        gain is bounded by :math:`1/(2\\sqrt{\\text{ridge}})` exactly where the
        operator is blind. This, not ``lam``, is the principled control on the
        short-wavelength null space.
    iters
        Maximum conjugate-gradient iterations (each costs one forward+backward,
        i.e. two FFT pairs). CG on this quadratic typically converges in tens.
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

    if lam is _LAM_REQUIRED:
        raise TypeError(
            "budget_bridging_melt_rate() requires an explicit lam: there is no "
            "safe default. Pass a float (to reproduce a published run -- a "
            "fixed lam is not transferable across noise levels), or "
            '"auto" to derive it as LAM_SIGMA2_COEF * sigma2_est, which is '
            "near-oracle under WHITE noise but over-damps the melt when the "
            "observation error is spatially correlated (PIG: ~4 km). See the "
            "lam entry in the docstring.")

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
    obs = np.where(fit, np.nan_to_num(dHdt_obs.values), 0.0)
    wv = np.where(fit, w.values, 0.0)
    wv = wv / max(float(wv.max()), 1e-30)

    # Truth-free noise variance of the observation. Computed unconditionally
    # (it used to live inside the strip-mode branch, so it was unavailable to
    # anything else) because it is what both the coloured-noise prior AND the
    # data-driven lam are scaled by.
    sigma2_est = _estimate_sigma2_white(reg, w, weight, H_f_stack,
                                        vxm, vym, dx, wv, fit)

    lam_auto = isinstance(lam, str)
    if lam_auto:
        if lam != "auto":
            raise ValueError(f'lam must be a float or "auto", got {lam!r}')
        lam_val = LAM_SIGMA2_COEF * sigma2_est
    else:
        lam_val = float(lam)
    # Self-consistency guard on the WHITE estimate. sigma2_est is built from
    # the per-pixel regression rmse, so spatially CORRELATED strip error (real
    # DEM stacks: coherent at 1-4 km) inflates it without inflating the part of
    # the misfit a smoothness prior should suppress. If the claimed noise
    # variance reaches the variance of the observation itself, the white model
    # is self-evidently wrong -- and an auto lam built on it nulls the melt.
    # Measured on the pigreal twin: sigma2_est 97.0 on the correlated-error
    # tier vs 3.2e-6 on the clean tier (7 decades) while the useful lam moves
    # only ~4, so auto over-damps there by ~300-3000x (gain 0.002 vs 0.311).
    obs_var = float(np.var(obs[fit])) if fit.any() else float("nan")
    sigma2_inflated = np.isfinite(obs_var) and sigma2_est > obs_var
    if log_every:
        _how = f"auto = {LAM_SIGMA2_COEF:g}*sigma2_est" if lam_auto else "fixed"
        print(f"    lam {lam_val:.4g} ({_how})  sigma2_est {sigma2_est:.4g}"
              f"  var(obs) {obs_var:.4g}", flush=True)
    if lam_auto and sigma2_inflated:
        warnings.warn(
            f"auto lam: white sigma2_est ({sigma2_est:.3g}) exceeds the "
            f"observation variance ({obs_var:.3g}), which means the noise is "
            "spatially correlated rather than white and the estimate is "
            "inflated. The resulting lam will over-damp the melt (on the "
            "pigreal twin: gain 0.002 vs 0.311 at a hand-picked lam). Pass an "
            "explicit lam until a coloured-noise estimate is available.",
            RuntimeWarning, stacklevel=2)

    # ---- coloured noise model: per-strip mode columns + their prior
    n_modes = 0 if strip_modes is None else int(strip_modes.shape[0])
    sigma2_val = float("nan")
    if n_modes:
        if strip_prior is None:
            raise ValueError("strip_modes needs strip_prior (variances per mode)")
        tau2 = np.broadcast_to(np.asarray(strip_prior, float), (n_modes,)).copy()
        sigma2_val = sigma2_est if sigma2 == "auto" else float(sigma2)
        # standardise the amplitudes (theta' = theta / tau, G' = G tau): the
        # tilt modes carry (x - x_c) ~ 1e4 m against theta ~ 1e-4 m/m, and
        # without this the theta block of the Hessian is ~1e8 worse
        # conditioned than the melt's and CG never converges it
        tau = np.sqrt(tau2)
        G_np = np.where(fit[None], np.nan_to_num(strip_modes, nan=0.0), 0.0) \
            * tau[:, None, None]

    # ---- the operator
    if transfer not in ("flotation", "normalized"):
        raise ValueError(
            f"transfer must be 'flotation' or 'normalized', got {transfer!r}")
    H_ref = float(np.nanmedian(H_f_mean.values[fit]))
    eb = eta_bar
    bins_w = None          # (n_bins, ny, nx) partition-of-unity weights
    bin_geom = []          # per-bin (H, ux, uy[, eta]) centroids
    if bridging:
        u0x = float(np.nanmedian(vxm.values[fit]))
        u0y = float(np.nanmedian(vym.values[fit]))
        ef = None
        if eta_field is not None:
            ef = np.asarray(eta_field.broadcast_like(H_f_mean).values, float)
            fin = np.isfinite(ef) & fit
            if fin.any():
                eb = float(np.nanmedian(ef[fin]))
        # padded grid: the operator's downstream footprint is long, so the
        # field is mirror-doubled before transforming (as StubblefieldForward).
        build = (bridging_transfer_multiplier if transfer == "flotation"
                 else normalized_bridging_multiplier)

        def _mult(Hb, uxb, uyb, etab):
            return build(2 * ny, 2 * nx, dx, dy, H=Hb, ux_myr=uxb, uy_myr=uyb,
                         eta_bar=etab, alpha_scale=alpha_scale, rho_i=rho_i,
                         rho_w=rho_w)

        if int(n_bins) <= 1:
            D = _mult(H_ref, u0x, u0y, eb)
            bin_geom = [(H_ref, u0x, u0y, eb)]
        else:
            from scipy.ndimage import gaussian_filter
            from .stubblefield_forward import _kmeans_geometry
            cols = [H_f_mean.values, vxm.values, vym.values]
            valid = fit & np.isfinite(cols[0]) & np.isfinite(cols[1]) \
                & np.isfinite(cols[2]) & (cols[0] > 0)
            if ef is not None:
                cols.append(ef)
                valid &= np.isfinite(ef) & (ef > 0)
            feats = np.stack([np.asarray(c, float)[valid] for c in cols], axis=1)
            # uniform geometry (to 1e-6 relative) must collapse to ONE bin so
            # the global multiplier is reproduced exactly, not to round-off
            scale = np.maximum(np.abs(feats).max(0), 1e-30)
            n_uniq = len(np.unique(np.round(feats / scale, 6), axis=0))
            nb = max(1, min(int(n_bins), n_uniq))
            if nb == 1:
                lab_v = np.zeros(len(feats), int)
                cent = feats.mean(0, keepdims=True)
            else:
                lab_v, cent = _kmeans_geometry(feats, nb)
            mults, w = [], np.zeros((nb, ny, nx))
            for b in range(nb):
                row = [float(c) for c in cent[b]]
                etab = row[3] if ef is not None else eb
                bin_geom.append((row[0], row[1], row[2], etab))
                mults.append(_mult(row[0], row[1], row[2], etab))
                ind = np.zeros((ny, nx))
                ind[valid] = (lab_v == b)
                w[b] = gaussian_filter(ind, blend_px, mode="nearest")
            tot = w.sum(0)
            w[0][tot < 1e-8] = 1.0
            bins_w = w / np.maximum(w.sum(0), 1e-30)
            D = np.stack(mults)
        if log_every:
            print(f"    bridging operator [{transfer}]: H_ref={H_ref:.0f} m  "
                  f"u0=({u0x:.0f}, {u0y:.0f}) m/yr  eta_bar={eb:.2e} Pa s  "
                  f"alpha_scale={alpha_scale:g}  "
                  f"flux_{'restored' if flux_restored else 'inside' if flux_in_operator else 'outside'}  "
                  f"bins={len(bin_geom)}", flush=True)
            if len(bin_geom) > 1:
                for b, (Hb, uxb, uyb, etab) in enumerate(bin_geom):
                    print(f"      bin {b}: H={Hb:.0f} m  u=({uxb:.0f}, {uyb:.0f}) "
                          f"m/yr  eta={etab:.2e}  area {bins_w[b].mean():.2f}",
                          flush=True)
    else:
        D = None
        u0x = u0y = 0.0

    # ---- the flux divergence: of the observed mean thickness on the
    # observation side (default), or, with `flux_restored`, of the PER-BIN
    # restored thickness inside the operator
    umax = None
    n_restore_guarded = 0
    if flux_restored and bridging:
        from ..backend import to_numpy
        from .bridging_restoration import bridging_restoration_filter
        rk = dict(restore_kwargs or {})
        umax = rk.pop("lift_umax_myr", None)
        hv = H_f_mean.values.astype(float)
        finh = np.isfinite(hv)
        pad = np.pad(np.where(finh, hv, float(hv[finh].mean())),
                     ((0, ny), (0, nx)), mode="symmetric")
        spec = np.fft.fft2(pad)
        hr = np.zeros((ny, nx))
        for b, (Hb, uxb, uyb, etab) in enumerate(bin_geom):
            if umax is not None and float(np.hypot(uxb, uyb)) > umax:
                n_restore_guarded += 1
                resp = pad[:ny, :nx]
            else:
                Fb, _ = bridging_restoration_filter(
                    2 * ny, 2 * nx, dx, dy, Hb, uxb, uyb, eta_bar=etab,
                    alpha_scale=alpha_scale, rho_i=rho_i, rho_w=rho_w, **rk)
                resp = np.real(np.fft.ifft2(to_numpy(Fb) * spec))[:ny, :nx]
            hr += resp if bins_w is None else bins_w[b] * resp
        H_rest = xr.DataArray(np.where(finh, hr, np.nan), dims=("y", "x"),
                              coords={"y": H_f_mean.y.values,
                                      "x": H_f_mean.x.values})
        fd = flux_divergence(H_rest, vxm, vym, estimator=estimator)
        if log_every:
            print(f"    flux_restored: {len(bin_geom)} restoration filter(s), "
                  f"{n_restore_guarded} trunk-guarded (lift_umax_myr={umax})",
                  flush=True)
    fd_np = np.nan_to_num(fd.values)
    # The flux divergence of the OBSERVED (already bridged) mean thickness must
    # not pass through the operator a second time, so it goes on the
    # observation side; `flux_in_operator=True` restores the pre-fix double
    # count for the audit A/B only. With `flux_restored` it is the divergence
    # of the RESTORED thickness and belongs inside T -- the residual is
    # T{m + a - div(H_rest u)} - dHdt_obs, dHdt_obs staying the observed
    # (bridged) rate.
    if flux_restored and bridging:
        known_in = np.where(fit, a_np - fd_np, 0.0)
        known_out = np.zeros_like(known_in)
    elif flux_in_operator:
        known_in = np.where(fit, a_np - fd_np, 0.0)
        known_out = np.zeros_like(known_in)
    else:
        known_in = np.where(fit, a_np, 0.0)
        known_out = np.where(fit, -fd_np, 0.0)

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
    t_G = (None if not n_modes else
           torch.from_numpy(np.ascontiguousarray(G_np.reshape(n_modes, -1), dtype=np.float64)))
    t_bw = (None if bins_w is None
            else torch.from_numpy(np.ascontiguousarray(bins_w, dtype=np.float64)))

    def _pad2x(a):
        a = torch.cat([a, torch.flip(a, dims=[0])], dim=0)
        return torch.cat([a, torch.flip(a, dims=[1])], dim=1)

    def bridge(field):
        if t_D is None:
            return field
        F = torch.fft.fft2(_pad2x(field).to(torch.complex128))
        if t_bw is None:
            return torch.fft.ifft2(t_D * F).real[:ny, :nx]
        # local operator: every bin's multiplier on the whole domain, blended
        resp = torch.fft.ifft2(t_D * F[None]).real[:, :ny, :nx]
        return (t_bw * resp).sum(0)

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
    # The unknown vector is [m (ny*nx) | theta (n_modes)]: the strip-error
    # amplitudes ride along in the same CG, so the coloured covariance is
    # handled exactly (marginalising theta) at the cost of n_modes extra dofs.
    n_m = ny * nx

    def _split(vec):
        return vec[:n_m].reshape(ny, nx), (vec[n_m:] if n_modes else None)

    def _grad(vec: torch.Tensor) -> torch.Tensor:
        vv = vec.detach().clone().requires_grad_(True)
        mv, th = _split(vv)
        S = torch.where(t_fit, mv + t_known, torch.zeros_like(mv))
        resid = bridge(S) + t_kout - t_obs
        if n_modes:
            resid = resid + (th[:, None] * t_G).sum(0).reshape(ny, nx)
        resid = torch.where(t_fit, resid, torch.zeros_like(mv))
        loss = (t_w * resid ** 2).sum() / wsum
        if lam_val:
            gx = mv[:, 1:] - mv[:, :-1]
            gy = mv[1:, :] - mv[:-1, :]
            loss = loss + lam_val * ((gx ** 2).mean() + (gy ** 2).mean())
        if ridge:
            loss = loss + ridge * (mv ** 2).mean()
        if n_modes:
            loss = loss + (sigma2_val / wsum) * (th ** 2).sum()   # unit-variance prior
        (g,) = torch.autograd.grad(loss, vv)
        return g

    zero = torch.zeros(n_m + n_modes, dtype=torch.float64)
    g0 = _grad(zero)                 # g(0) = -rhs
    rhs = -g0

    def _hess(v: torch.Tensor) -> torch.Tensor:
        return _grad(v) - g0

    m = torch.cat([torch.from_numpy(np.ascontiguousarray(m0, dtype=np.float64)).reshape(-1),
                   torch.zeros(n_modes, dtype=torch.float64)])
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
        mv, th = _split(m)
        S = torch.where(t_fit, mv + t_known, torch.zeros_like(mv))
        strip_rate = (np.zeros((ny, nx)) if not n_modes else
                      (th[:, None] * t_G).sum(0).reshape(ny, nx).numpy())
        # the fitted rate INCLUDING the estimated strip error, so the residual
        # reported below is the whitened one
        fit_rate = (bridge(S) + t_kout).numpy() + strip_rate
        m_out = mv.numpy()
        theta = None if not n_modes else th.numpy() * tau   # back to freeboard units

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
            "strip_error_rate": xr.DataArray(np.where(fit, strip_rate, np.nan),
                                             dims=("y", "x"), coords=coords),
            "weight": xr.DataArray(np.where(fit, wv, np.nan), dims=("y", "x"),
                                   coords=coords),
        },
        attrs={
            "method": "budget+bridging monolithic fit",
            "bridging": int(bool(bridging)),
            "transfer": transfer if bridging else "identity",
            "flux_in_operator": int(bool(flux_in_operator)),
            "flux_restored": int(bool(flux_restored)),
            "flux_restored_bins": len(bin_geom) if (flux_restored and bridging) else 0,
            "flux_restored_guarded_bins": n_restore_guarded,
            "lift_umax_myr": float(umax) if umax is not None else float("nan"),
            "H_ref_m": H_ref,
            "u0x_myr": u0x,
            "u0y_myr": u0y,
            "eta_bar": eb,
            "alpha_scale": alpha_scale,
            "n_bins": len(bin_geom),
            "n_strip_modes": n_modes,
            "sigma2_white": sigma2_val,
            "bin_geometry": ";".join(f"{Hb:.1f},{uxb:.1f},{uyb:.1f},{etab:.3e}"
                                     for Hb, uxb, uyb, etab in bin_geom),
            "lam": lam_val,
            "lam_mode": "auto" if lam_auto else "fixed",
            "sigma2_est": sigma2_est,
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
    if n_modes:
        ds["theta"] = xr.DataArray(theta, dims=("mode",))
        ds["theta_prior_var"] = xr.DataArray(tau2, dims=("mode",))
    return ds
