# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Budget-corrected, pair-banded Stubblefield inverse that matches the
Lagrangian path solver by construction.

Why the older linear-inverse entry points cannot match
:func:`stereo_melt.melt.lagrangian_melt_rate`
(``literature/plan_match_linear_inverse.md``):

1. They feed the raw surface anomaly to the kernel, so the spatially-
   varying strain thinning :math:`H_f\,\nabla\!\cdot u(x,y)` and the SMB
   pattern :math:`\dot a(x,y)` — first-order terms of the Shean budget on a
   fast shelf — are attributed to melt. Only their *tile means* were ever
   handled (uniform :math:`\gamma`, scalar DC splice).
2. They anchor one Lagrangian frame at the start of the whole record, so
   the recovered field is trajectory-averaged and seed-attributed — the
   attribution that lost GL melt and coverage in the parcel-LSQ A/B.

This module fixes both by keeping the production path solver's *sampling
structure* and adding the kernel physics on top:

- **Pair fans, Shean banding.** For every start epoch, partners within the
  1.5–2.5 yr baseline band are warped into a Lagrangian frame anchored at
  the start; each pair gives a per-cell slope sample; fans reduce by
  median, then cells reduce by median across fans (the two-level
  ``pair_median`` mosaic).
- **Budget correction.** The non-melt freeboard rate

  .. math::
      c(x, y) = \delta_f\,(\dot a - H_f\,\nabla\!\cdot u) + u\cdot\nabla d,
      \qquad \delta_f = (\rho_w - \rho_i)/\rho_w,

  is sampled along the same trajectories (with :math:`H_f` from the
  per-pixel record trend at the fan mid-time, since the shelf thins across
  the record; ``fan_H="fan-mean"`` keeps the older behaviour) and its
  running time-integral is removed before the pair differencing. The
  aggregated slope is then :math:`\delta_f\,\dot b` in the hydrostatic
  limit. Time-varying velocity (a ``time`` dim on ``vx``/``vy``) is sampled
  per sub-step for both the trajectories and the strain divergence — on
  nonstationary shelves a time-mean velocity is a first-order error (PIG:
  ~1.5x IQR stretch, +40% GL flux; ``pig/diag_path_meanvel.py``). Mosaic
  data gaps are tolerated per sub-step: NaN cells of the c-field fall back
  to the time-mean-velocity field (residual holes to SMB-only) — without
  this, one NaN sample poisons the parcel's ``∫c dτ`` and kills all its
  pairs (seams along gap boundaries; missing front/margin melt band —
  2026-07-03 PIG diagnosis).
- **Hydrostatic/non-hydrostatic split.** The recovered melt is the exact
  pointwise hydrostatic estimate :math:`R\cdot s` — algebraically the path
  solver's :math:`R\,Dh/Dt + H_f\nabla\!\cdot u - \dot a`, immune to
  transform ringing, with its DC inherent — plus the band-limited
  non-hydrostatic correction ``inverse_dhdt(s_filled) − R·s_filled``
  evaluated on a smoothly infilled field. On full coverage the sum equals
  the kernel deconvolution exactly; under strip gaps the fill artefacts are
  quarantined to the (small, ~sub-5H) correction term.

Agreement with the path solver at ≥2 km scales is therefore a construction
property; the kernel contributes only where bridging stresses make the
hydrostatic inversion wrong (channels ≲ 5 ice thicknesses). Validated in
``tests/gate_match_lagrangian.py``.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import asarray, map_coordinates, to_numpy, xp
from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import SECONDS_PER_YEAR, divergence, flux_divergence
from .lagrangian_inverse import _nan_aware_gaussian_infill_2d
from .linear_perturbation import (
    LinearPerturbation,
    _dct_wavenumber_grids,
    _dctn,
    _idctn,
    inverse_dhdt,
)

__all__ = [
    "linear_inverse_budget_melt_rate",
    "linear_inverse_eulerian_budget_melt_rate",
]

G_GRAVITY = 9.81


def _times_to_years(time_values) -> np.ndarray:
    times = pd.to_datetime(np.asarray(time_values))
    t0 = times[0]
    return np.array(
        [(t - t0).total_seconds() / SECONDS_PER_YEAR for t in times],
        dtype=np.float64,
    )


class _VelocityProvider:
    r"""Velocity (and its divergence) fields at an absolute time, on the backend.

    Static mode (``vx_tv is None``): always returns the same arrays — the
    original time-mean behaviour. Time-varying mode mirrors
    :func:`stereo_melt.melt.lagrangian_melt_rate` EXACTLY, including its
    clamp semantics (melt.py L551-556): beyond the first/last velocity
    epoch it samples a SINGLE quarter (``k0 == k1``) — it does NOT blend.
    A ``(1-w)·f[lo] + w·f[hi]`` blend with ``w = 0`` is NOT equivalent:
    ``0·NaN = NaN``, so the unused quarter's data gaps contaminate the
    sample. That bug killed every pre-2015 PIG fan wherever 2015-Q2 has
    holes (the production solver samples 2015-Q1 alone there) and was the
    root of the residual gap-zone coverage asymmetry (2026-07-03/04
    rewrite). Interior times blend two quarters, either-NaN → NaN,
    matching melt.py's sample-level blend. Results are cached per query
    time so the several lookups within one sub-step pay once.
    """

    def __init__(self, vx_mean, vy_mean, div_mean,
                 vx_tv=None, vy_tv=None, div_tv=None, v_t_years=None):
        self.vx_mean, self.vy_mean, self.div_mean = vx_mean, vy_mean, div_mean
        self.vx_tv, self.vy_tv, self.div_tv = vx_tv, vy_tv, div_tv
        self.v_t = np.asarray(v_t_years, dtype=np.float64) if v_t_years is not None else None
        self._cache_t = None
        self._cache = None

    @property
    def time_varying(self) -> bool:
        return self.vx_tv is not None and self.v_t is not None and len(self.v_t) > 1

    def _bracket(self, t_abs_yr: float):
        """melt.py-parity bracketing: (k0, k1, w) with single-quarter clamps."""
        v_t = self.v_t
        if t_abs_yr <= v_t[0]:
            return 0, 0, 0.0
        if t_abs_yr >= v_t[-1]:
            n = len(v_t) - 1
            return n, n, 0.0
        k1 = int(np.searchsorted(v_t, t_abs_yr))
        k0 = k1 - 1
        w = float((t_abs_yr - v_t[k0]) / (v_t[k1] - v_t[k0]))
        return k0, k1, w

    def fields(self, t_abs_yr: float):
        """Return ``(vx, vy, div)`` backend arrays at absolute year ``t_abs_yr``."""
        if not self.time_varying:
            return self.vx_mean, self.vy_mean, self.div_mean
        if self._cache_t is not None and t_abs_yr == self._cache_t:
            return self._cache
        k0, k1, w = self._bracket(t_abs_yr)
        if k0 == k1:
            out = (
                self.vx_tv[k0], self.vy_tv[k0],
                self.div_tv[k0] if self.div_tv is not None else self.div_mean,
            )
        else:
            out = (
                self.vx_tv[k0] * (1.0 - w) + self.vx_tv[k1] * w,
                self.vy_tv[k0] * (1.0 - w) + self.vy_tv[k1] * w,
                (self.div_tv[k0] * (1.0 - w) + self.div_tv[k1] * w)
                if self.div_tv is not None else self.div_mean,
            )
        self._cache_t, self._cache = t_abs_yr, out
        return out



def _fan_map_walk(
    h_sub_vals: np.ndarray,
    budget,
    vel: _VelocityProvider,
    t_rel: np.ndarray,
    t_abs0_yr: float,
    dt_yr: float,
    res_x: float,
    res_y: float,
    attribution: str,
    return_pairs: bool = False,
):
    r"""One fan's slope map via a structural translation of melt.py's walk.

    Rewrite (2026-07-03 evening) after two rounds of gap-zone artefacts: the
    original ``_warp_fan`` re-derived the trajectory machinery and deviated
    from :func:`stereo_melt.melt.lagrangian_melt_rate` in several
    individually-small but jointly load-bearing ways. This version mirrors
    the production loop line-for-line where it matters:

    - **Fixed ``dt_yr`` grid** from the anchor to the farthest partner
      (melt.py L471); partner endpoints at the NEAREST step
      (``h_idx = round(dt/dt_yr)``, L614) with the ACTUAL baseline in the
      slope denominator.
    - **Velocity clamp semantics** via ``_VelocityProvider`` (single-quarter
      sampling beyond the ends — no ``0·NaN`` contamination from the unused
      quarter; melt.py L551-556).
    - **Freeze, don't NaN** (melt.py L586-598): a NaN velocity sample zeros
      the displacement, keeps positions finite, and drops the parcel from
      ``valid`` cumulatively; deposits and endpoints are gated on the
      cumulative validity at their step.
    - **Post-advection nearest-cell recording** for deposits (L600-602) and
      post-advection float positions for endpoint sampling (L615-621).

    The only graft: the non-melt rate ``c = δ_f(ȧ − H_f∇·u)`` is sampled
    per step at the pre-advection position from a per-step field (NaN cells
    filled from the static fallback — the integral-form analog of melt.py's
    per-step ``vdiv NaN → skip``), accumulated into ``∫c dτ``, and
    re-localized into path deposits ``s_dep = s_pair + ⟨c⟩ − c(x(τ))``.
    The firn-advection part of the freeboard budget is NOT in ``c``: for
    static ``d`` the path integral ``∫u·∇d dτ`` telescopes exactly to
    ``d(x_end) − d(x_start)``, so it is applied per pair from two endpoint
    samples. Never integrate ``u·∇d`` per step: BedMachine FAC is a
    bilinear resample of a coarse firn model, so ``∇d`` is a
    piecewise-constant comb on the model's ~5–10 km cell lattice, and the
    quadrature error rides the 1/δ_f ≈ 9.4 hydrostatic gain into ±2–8 m/yr
    rectangular bands (PIG ``tv_path`` tiles, 2026-07-04).

    Aggregation: ``attribution="seed"`` assigns each pair's value at the
    seed; ``"path"`` deposits along visited cells with per-pair per-cell
    means (equal weights — the grid is uniform). Either way the fan reduces
    by per-cell median over its pairs; the caller medians across fans.
    Returns the fan map in slope units (m/yr), float32 ``(ny, nx)``.
    """
    n_sub = h_sub_vals.shape[0]
    ny, nx = h_sub_vals.shape[1:]
    n_cells = ny * nx
    h_arr = asarray(h_sub_vals)
    h0 = to_numpy(h_arr[0]).ravel()

    dt_far = float(t_rel[-1])
    nsf = max(1, int(np.ceil(dt_far / dt_yr)))
    h_idx = [
        max(1, min(nsf, int(round(float(t_rel[k]) / dt_yr))))
        for k in range(1, n_sub)
    ]
    end_at_step: dict[int, list[int]] = {}
    for kk, hi in enumerate(h_idx, start=1):
        end_at_step.setdefault(hi - 1, []).append(kk)

    ji, ii = xp.mgrid[0:ny, 0:nx]
    y_idx = ji.astype(xp.float64).ravel()
    x_idx = ii.astype(xp.float64).ravel()
    valid = xp.ones(n_cells, dtype=bool)

    d_fac = None
    if budget is not None:
        a_term, H_fan, d_fac, delta_frac, c_fallback = budget

    collect_path = attribution == "path"
    cell_hist = (
        np.zeros((nsf, n_cells), dtype=np.int32) if collect_path else None
    )
    c_hist = (
        np.zeros((nsf, n_cells), dtype=np.float32) if collect_path else None
    )
    valid_hist = np.zeros((nsf, n_cells), dtype=bool)
    c_cum = xp.zeros(n_cells, dtype=xp.float64)
    h_end = np.full((n_sub, n_cells), np.nan, dtype=np.float64)
    c_cum_end = np.zeros((n_sub, n_cells), dtype=np.float64)
    if d_fac is not None:
        # Endpoint firn samples for the exact ∫u·∇d dτ = d_end − d_seed.
        d0 = to_numpy(d_fac).ravel()
        d_end = np.full((n_sub, n_cells), np.nan, dtype=np.float64)

    for s in range(nsf):
        vx_f, vy_f, div_f = vel.fields(t_abs0_yr + s * dt_yr)
        vx_t = map_coordinates(vx_f, [y_idx, x_idx], order=1, mode="nearest")
        vy_t = map_coordinates(vy_f, [y_idx, x_idx], order=1, mode="nearest")
        if budget is not None:
            c_field = a_term - delta_frac * H_fan * div_f
            c_field = xp.where(xp.isfinite(c_field), c_field, c_fallback)
            c_t = map_coordinates(c_field, [y_idx, x_idx], order=1, mode="nearest")
            c_cum = c_cum + c_t * dt_yr
            if collect_path:
                c_hist[s] = to_numpy(c_t).astype(np.float32)
        # melt.py L586-598: freeze at NaN velocity, invalidate cumulatively.
        finite_v = xp.isfinite(vx_t) & xp.isfinite(vy_t)
        vx_t = xp.where(finite_v, vx_t, 0.0)
        vy_t = xp.where(finite_v, vy_t, 0.0)
        x_idx = x_idx + vx_t * dt_yr / res_x
        y_idx = y_idx - vy_t * dt_yr / res_y
        in_b = (x_idx >= 0) & (x_idx <= nx - 1) & (y_idx >= 0) & (y_idx <= ny - 1)
        valid = valid & in_b & finite_v
        valid_hist[s] = to_numpy(valid)
        if collect_path:
            xi = xp.clip(xp.round(x_idx).astype(xp.int64), 0, nx - 1)
            yi = xp.clip(xp.round(y_idx).astype(xp.int64), 0, ny - 1)
            cell_hist[s] = to_numpy(yi * nx + xi).astype(np.int32)
        for kk in end_at_step.get(s, ()):
            he = map_coordinates(
                h_arr[kk], [y_idx, x_idx], order=1,
                mode="constant", cval=float("nan"),
            )
            h_end[kk] = to_numpy(he)
            c_cum_end[kk] = to_numpy(c_cum)
            if d_fac is not None:
                d_end[kk] = to_numpy(map_coordinates(
                    d_fac, [y_idx, x_idx], order=1,
                    mode="constant", cval=float("nan"),
                ))

    pair_maps = np.full((n_sub - 1, n_cells), np.nan, dtype=np.float32)
    for kk in range(1, n_sub):
        hi = h_idx[kk - 1]
        dt_act = float(t_rel[kk])
        c_mean = c_cum_end[kk] / (hi * dt_yr)
        s_pair = (h_end[kk] - h0) / dt_act - c_mean
        if d_fac is not None:
            # Exact firn advection: ∫u·∇d dτ telescopes for static d.
            s_pair = s_pair - (d_end[kk] - d0) / dt_act
        p_ok = np.isfinite(s_pair) & valid_hist[hi - 1]
        if not p_ok.any():
            continue
        if not collect_path:
            pair_maps[kk - 1] = np.where(p_ok, s_pair, np.nan).astype(np.float32)
            continue
        psum = np.zeros(n_cells, dtype=np.float64)
        pcnt = np.zeros(n_cells, dtype=np.float64)
        base = s_pair + c_mean
        for s in range(hi):
            m = valid_hist[s] & p_ok
            if not m.any():
                continue
            vals = base - c_hist[s]
            m &= np.isfinite(vals)
            idx = cell_hist[s][m]
            psum += np.bincount(idx, weights=vals[m], minlength=n_cells)
            pcnt += np.bincount(idx, minlength=n_cells)
        pair_maps[kk - 1] = np.where(
            pcnt > 0, psum / np.maximum(pcnt, 1e-30), np.nan
        ).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fan_map = np.nanmedian(pair_maps.astype(np.float64), axis=0)
    fan_map = fan_map.reshape(ny, nx).astype(np.float32)
    if not return_pairs:
        return fan_map
    # Sparse per-pair contributions (finite cells only) for the global
    # cross-pair pooled median. Each row of pair_maps is already level 1
    # ("one mean value per pair per cell").
    pairs_sparse = []
    for row in pair_maps:
        idx = np.flatnonzero(np.isfinite(row))
        if idx.size:
            pairs_sparse.append((idx.astype(np.int32), row[idx].astype(np.float32)))
    return fan_map, pairs_sparse

def _build_ols_kernel_dct(
    H_ref: float, eta_bar: float, rho_i: float, rho_w: float, g: float,
    gamma: float, theta: float, t_secs: np.ndarray,
    ny: int, nx: int, dx: float, dy: float,
):
    """OLS-slope kernel ``K_dot(k)`` on the DCT grid (mirrors inverse_dhdt).

    Returns ``(K, k_rms)`` as backend arrays/float. ``K`` maps the internal
    Stubblefield-convention stationary melt (m ice/s, positive = melt) to the
    OLS dh/dt response (m/s) over the epochs ``t_secs`` — diagonal in the
    orthonormal DCT basis, so the forward operator is exactly symmetric.
    """
    model = LinearPerturbation(
        H=H_ref, eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
        alpha=0.0, alpha_y=0.0, gamma=gamma, theta=theta,
    )
    kx, ky = _dct_wavenumber_grids(nx, ny, dx, dy)
    t_centered = np.asarray(t_secs, dtype=np.float64)
    t_centered = t_centered - t_centered.mean()
    denom = float((t_centered**2).sum())
    K = xp.zeros_like(kx, dtype=xp.complex128)
    for i, t_s in enumerate(t_secs):
        I_h, _ = model.kernel_time_integral_stationary(kx, ky, float(t_s) / model.tr)
        K = K + float(t_centered[i]) * (model.tr * I_h)
    K = xp.real(K) / denom
    k_rms = float(to_numpy(xp.sqrt((K**2).mean())))
    kmag = xp.sqrt(kx**2 + ky**2)
    return K, k_rms, kmag


def _irls_ensemble_kernel_solve(
    fan_maps: np.ndarray,
    K,
    lam2_k,
    tukey_c: float = 4.685,
    n_irls: int = 6,
    cg_tol: float = 1e-5,
    cg_maxiter: int = 300,
    progress: bool = False,
) -> tuple[np.ndarray, dict]:
    r"""Robust (Tukey-IRLS) masked ensemble inversion through the OLS kernel.

    The path solver's per-cell **median across pairs** is, exactly, the L1
    fit of a per-pixel constant to the pair samples. This generalizes that
    filter through the linear kernel: solve

    .. math::
        \min_m \sum_f \sum_x W_f\,w_f\,(s_f - F m)^2
             + \|\Lambda^{1/2} P m\|^2,
        \qquad F = P^T K P,

    where ``s_f`` are the per-fan budget-corrected slope maps (m/s, only on
    their own observation footprints ``W_f`` — coverage enters as weights,
    so NO gap infill anywhere), ``P`` is the orthonormal DCT, and the Tukey
    biweights ``w_f`` are re-derived from the residuals each outer
    iteration (IRLS). In the identity-kernel / zero-reg limit this IS the
    per-pixel median; with the kernel it is "the median in the kernel's
    data space". Normal equations per iteration are solved by CG; the
    normal operator needs 4-6 DCTs per CG step regardless of fan count
    because the weighted fans collapse to one field.

    ``lam2_k`` is the regularization SPECTRUM (scalar or k-grid array):
    a flat value is plain Tikhonov; a ``(1 + (kσ)²)²``-shaped spectrum is
    a roughness (bi-Laplacian) penalty — the in-objective analog of the
    split path's Gaussian pre-filter, and necessary under strip gaps
    (flat Tikhonov lets the deconvolution amplify per-fan slope noise at
    high k where the masked data cannot vote it down).

    ``K`` comes from ``kernel_time_integral_stationary``, so its ``k = 0``
    bin is finite and non-zero -- the kernel is NOT degenerate there. The
    caller owns the DC level anyway: every path here splices it from the
    hydrostatic budget rather than trusting the kernel's DC response.

    Returns ``(m, diag)`` with ``m`` in the INTERNAL Stubblefield
    convention (m ice/s, positive = melt), numpy (ny, nx).
    """
    n_f, ny, nx = fan_maps.shape
    valid = np.isfinite(fan_maps)
    if not valid.any():
        raise ValueError("no finite fan samples for the IRLS ensemble solve")
    # float32 for the (F, ny, nx) stacks (values ~1e-8 m/s, rel precision
    # 1e-7 is ample); collapse to float64 2-D fields for the CG core.
    s_np = np.where(valid, fan_maps, 0.0).astype(np.float32)
    w_np = valid.astype(np.float32)

    def F_apply(v):
        return _idctn(K * _dctn(v))

    m = xp.zeros((ny, nx), dtype=xp.float64)
    diag: dict = {}
    t0 = time.time()
    for it in range(n_irls):
        omega = asarray(w_np.sum(axis=0, dtype=np.float64))
        rhs = F_apply(asarray((w_np * s_np).sum(axis=0, dtype=np.float64)))

        def A_apply(v):
            return F_apply(omega * F_apply(v)) + _idctn(lam2_k * _dctn(v))

        # CG, warm-started from the previous outer iterate.
        r = rhs - A_apply(m)
        p = r.copy()
        rs = float(to_numpy((r * r).sum()))
        rhs_norm = float(to_numpy((rhs * rhs).sum()))
        n_cg = 0
        for n_cg in range(1, cg_maxiter + 1):
            Ap = A_apply(p)
            alpha_cg = rs / float(to_numpy((p * Ap).sum()))
            m = m + alpha_cg * p
            r = r - alpha_cg * Ap
            rs_new = float(to_numpy((r * r).sum()))
            if rs_new <= cg_tol**2 * max(rhs_norm, 1e-300):
                rs = rs_new
                break
            p = r + (rs_new / rs) * p
            rs = rs_new

        # Residuals per fan -> global robust scale -> Tukey biweights.
        Fm = to_numpy(F_apply(m)).astype(np.float32)
        resid = np.where(valid, s_np - Fm[None, :, :], np.nan)
        rv = resid[valid]
        med_r = float(np.median(rv))
        sigma = 1.4826 * float(np.median(np.abs(rv - med_r)))
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = float(np.std(rv)) or 1.0
        u = (resid - np.float32(med_r)) / np.float32(tukey_c * sigma)
        w_new = np.where(
            valid & (np.abs(u) < 1.0), (1.0 - u**2) ** 2, np.float32(0.0)
        ).astype(np.float32)
        w_frac = float(w_new[valid].mean())
        dw = float(np.abs(w_new - w_np)[valid].mean())
        w_np = w_new
        if progress:
            print(
                f"    IRLS {it + 1}/{n_irls}: cg_iters={n_cg}  "
                f"sigma={sigma * SECONDS_PER_YEAR:.3f} m/yr  "
                f"mean_weight={w_frac:.3f}  d_weight={dw:.4f}  "
                f"[{time.time() - t0:.0f}s]",
                flush=True,
            )
        diag = {
            "irls_iters": it + 1, "cg_iters_last": n_cg,
            "sigma_m_yr": sigma * SECONDS_PER_YEAR,
            "mean_weight": w_frac,
        }
        if dw < 1e-3 and it >= 1:
            break
    return to_numpy(m), diag


def _pool_pairs(idx_chunks, val_chunks, n_cells, ny, nx, R_hydro):
    """Cross-pair pooled median — ``melt.py`` ``pair_median`` level 2.

    The per-fan two-level median (median WITHIN each start's fan, then median
    ACROSS fans) makes a cell's estimate depend on how many *fans* reached it,
    and fan coverage is flow-banded (each start's seed footprint advects into a
    flow-aligned swath), so median-of-fan-medians imprints flow-aligned streaks
    wherever the per-fan count is thin (PIG ``tv_path`` diagnosis 2026-07-04:
    streaks track low fan count, ``min_fan_count=1`` admits single-fan cells).

    Pooling instead collects EVERY pair's per-cell mean into one global bucket
    and takes the per-cell median across all pairs — the flat two-level median
    the production path solver uses, which has no fan-count banding by
    construction (a cell crossed by 20 pairs gets 20 samples regardless of how
    many distinct starts they came from). Same bucket size / memory profile as
    ``melt.py``'s ``pair_median`` (proven at PIG 250 m).

    Returns ``(slope, count, spread)`` on the ``(ny, nx)`` grid: pooled median
    slope (m/yr), per-cell pair count, and dispersion ``std * R_hydro`` (m/yr,
    melt units — the same rmse diagnostic the fan path reports).
    """
    if not idx_chunks:
        nan2d = np.full((ny, nx), np.nan)
        return nan2d, np.zeros((ny, nx), np.int32), nan2d.copy()
    all_idx = np.concatenate(idx_chunks)
    all_val = np.concatenate(val_chunks)
    med = (
        pd.DataFrame({"i": all_idx, "v": all_val})
        .groupby("i", sort=True)["v"]
        .median()
    )
    slope = np.full(n_cells, np.nan, dtype=np.float64)
    slope[med.index.values] = med.values.astype(np.float64)
    cnt = np.bincount(all_idx, minlength=n_cells).astype(np.int32)
    vv = all_val.astype(np.float64)
    s1 = np.bincount(all_idx, weights=vv, minlength=n_cells)
    s2 = np.bincount(all_idx, weights=vv * vv, minlength=n_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mean**2, np.nan)
    spread = np.sqrt(np.maximum(var, 0.0)) * R_hydro
    return slope.reshape(ny, nx), cnt.reshape(ny, nx), spread.reshape(ny, nx)


def _pooled_kernel_correction(
    slope_agg: np.ndarray,
    finite_s: np.ndarray,
    *,
    coords: dict,
    res_m: float,
    sigma_pix: float,
    sigma_corr_m: float,
    H_ref_val: float,
    t_secs_kernel: np.ndarray,
    eta_bar: float,
    rho_i: float,
    rho_w: float,
    g: float,
    gamma_val: float,
    theta: float,
    reg: float,
    transform: str,
    R_hydro: float,
) -> np.ndarray:
    """Median-neutralized non-hydrostatic kernel correction from one pooled
    slope field (m/yr): ``inverse_dhdt(s_filled) − R·s_filled`` on a smoothly
    infilled, pre-smoothed copy. Shared by the pair-banded and Eulerian
    budget inverses; fill artefacts stay quarantined to this band-limited
    correction term while the hydrostatic channel keeps the exact DC."""
    med_s = float(np.median(slope_agg[finite_s]))
    slope_filled = _nan_aware_gaussian_infill_2d(
        slope_agg - med_s, (sigma_pix, sigma_pix), max_iters=5,
    ) + med_s
    if sigma_corr_m > 0:
        from scipy.ndimage import gaussian_filter

        slope_filled = gaussian_filter(
            slope_filled, sigma=sigma_corr_m / res_m, mode="nearest"
        )
    slope_da = xr.DataArray(
        slope_filled / SECONDS_PER_YEAR,  # m/s for inverse_dhdt
        dims=("y", "x"), coords=coords, name="dh_dt",
    )
    m_kernel = inverse_dhdt(
        slope_da, H=H_ref_val, t_secs=t_secs_kernel,
        eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
        alpha=0.0, alpha_y=0.0, gamma=gamma_val, theta=theta,
        reg=reg, transform=transform,
        recover_dc=False, a_dot_dc=0.0,
    )
    nonhydro_corr = m_kernel.values - R_hydro * slope_filled
    # Median-neutralize the correction: it redistributes melt at sub-5H
    # scales, so its bulk level over the mask carries no channel signal —
    # only the (small, ~5-9%) viscous-relaxation amplification, which is
    # eta_bar-degenerate, plus deconvolution noise skew. Zeroing the median
    # keeps the hydrostatic part's budget-exact level authoritative.
    return nonhydro_corr - float(np.nanmedian(nonhydro_corr[finite_s]))


def linear_inverse_budget_melt_rate(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    a_dot: xr.DataArray | float = 0.0,
    d: xr.DataArray | float = 0.0,
    floating_mask: xr.DataArray | None = None,
    H_ref: float | None = None,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    gamma: float | None = None,
    theta: float = 1e-14,
    reg: float = 1e-1,
    transform: str = "dct",
    min_pair_dt_yr: float = 1.5,
    max_pair_dt_yr: float = 2.5,
    dt_yr: float = 0.05,
    budget_correction: bool = True,
    corr_prefilter_sigma_m: float | None = None,
    fan_H: str = "trend",
    attribution: str = "seed",
    estimator: str = "split",
    corr_aggregate: str = "pooled",
    slope_aggregate: str = "pooled-pairs",
    irls_iters: int = 6,
    irls_tukey_c: float = 4.685,
    min_fan_count: int = 1,
    fan_cache: str | None = None,
    progress: bool = False,
    progress_interval_s: float = 30.0,
) -> xr.Dataset:
    r"""Basal melt rate via the budget-corrected, pair-banded Stubblefield inverse.

    Sampling mirrors :func:`stereo_melt.melt.lagrangian_melt_rate`
    (``output="path"``, ``aggregator="pair_median"``, ``pairs="all"`` with
    the same dt band): per start epoch, a fan of banded partners is warped
    into the start-anchored Lagrangian frame; budget-corrected pair slopes
    reduce by median within the fan and across fans. One closed-form kernel
    step at the end splits the estimate into the exact pointwise
    hydrostatic part (``R·s`` — the path solver's quantity) plus the
    non-hydrostatic channel correction.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Geoid-referenced, corrected surface elevation stack, meters.
        NaN = missing.
    vx, vy : xarray.DataArray
        Velocity, m/yr. With a >1-entry ``time`` dimension the advection and
        the divergence in the budget correction are time-resolved (per
        sub-step linear interpolation between bracketing velocity epochs,
        clamped at the ends — mirrors
        :func:`stereo_melt.melt.lagrangian_melt_rate`). A single-entry or
        absent ``time`` dim gives the static path. Time-mean velocity on a
        nonstationary shelf is a first-order error: it stretched the PIG
        estimate's IQR ~1.5x and inflated its GL flux by ~40% (2026-07-03
        diagnosis, ``pig/diag_path_meanvel.py``).
    a_dot : xarray.DataArray or float
        Surface mass balance, m ice/yr (the spatial pattern is used — that
        is the point of the budget correction).
    d : xarray.DataArray or float
        Firn air content, m (static). Enters the thickness conversion and
        the per-pair firn-advection correction ``(d(x_end) − d(x_start)) /
        Δt`` (the exact telescoped form of ``∫u·∇d dτ``; never sampled as
        a per-step gradient — see the walk docstring).
    floating_mask : xarray.DataArray, optional
        Boolean floating-ice mask; masks outputs and anchors tile means.
    H_ref : float, optional
        Kernel reference thickness, m. Default: mean hydrostatic thickness
        over the floating mask.
    eta_bar : float
        Column viscosity, Pa s. The default 1e14 keeps ~2-yr baselines
        quasi-hydrostatic (small-k relaxation time ≈ 19·t_r), so the kernel
        only reshapes sub-5H scales.
    gamma : float, optional
        Dimensionless tile-mean extension ``⟨∇·u⟩·t_r``; derived from the
        velocity if None.
    reg : float
        Dimensionless Tikhonov scale for :func:`inverse_dhdt`.
    transform : {"dct", "fft"}
        Spatial basis for the correction step (DCT = reflective).
    min_pair_dt_yr, max_pair_dt_yr : float
        Shean's pair-baseline band (1.5–2.5 yr production default). The
        floor caps noise/Δt amplification; the cap bounds trajectory drift.
    dt_yr : float
        Trajectory integration sub-step, years.
    budget_correction : bool
        If False, skip the strain/SMB/firn correction (ablation; reduces to
        a pair-banded version of the old dhdt wrapper).
    corr_prefilter_sigma_m : float, optional
        Gaussian pre-smooth (meters) of the slope field entering the
        non-hydrostatic correction step ONLY — the pointwise hydrostatic
        part is never smoothed. The deconvolution amplifies pixel-scale
        slope noise catastrophically (±15 m/yr per cell at 0.3 m epoch
        noise in the gate), and that noise shifts the median of the skewed
        melt field. Default H_ref (keeps ~60% of a 2πH-wavelength
        channel's correction; the median shift scales ~quadratically with
        the residual noise). On well-sampled stacks (many pairs per cell,
        so slope noise is small) H/2 recovers sharper channels — pass it
        explicitly. Pass 0 to disable.
    fan_H : {"trend", "fan-mean"}
        Thickness entering each fan's strain term δ_f·H·∇·u. ``"trend"``
        (default) evaluates the per-pixel record trend of observed H_f at
        the fan's pair-weighted mid-time, clipped to the observed range —
        removes the thinning-shelf bias of ``"fan-mean"`` (mean over fan
        epochs 0..2.5 yr while pair integrals run 0..t_k), worth ~10% of
        domain flux on the fast synthetic (tests/gate_flux_amplitude.py G0).
    attribution : {"seed", "path"}
        Where each pair's budget-corrected estimate is placed. ``"seed"``
        (default): the whole pair value at the trajectory's start cell —
        compact features, speckle noise, slight upstream shift.
        ``"path"``: mirror the production path solver's deposition — the
        pair value is scattered into every cell the parcel visits over the
        baseline, with the strain/SMB terms re-localized per sub-step
        (``s_pair + ⟨c⟩ − c(x(τ),τ)``, melt.py's ``DhDt + h_t∇·u − ȧ`` in
        slope space; firn advection stays pair-level — endpoint-exact,
        see the walk docstring); per-pair per-cell duration-weighted mean, then the
        two-level median. Along-flow-averaged texture matching
        ``lagrangian_melt_rate(output="path")``, denser coverage (any
        visiting parcel covers a cell), at the cost of the known deep-bin
        dilution of path deposition. Costs roughly one extra deposition
        pass per fan on top of the walk.
    estimator : {"split", "irls"}
        How the kernel physics is applied to the fan ensemble. ``"split"``
        (default): the production decomposition — pointwise hydrostatic
        ``R·s`` on the two-level-median slope plus one band-limited kernel
        correction. ``"irls"``: robust masked ensemble inversion — a single
        melt field is fit to ALL per-fan slope maps through the OLS kernel
        under a Tukey loss (IRLS + CG). This is the path solver's
        per-pixel median-across-pairs generalized THROUGH the kernel: the
        median is the L1 fit of a per-pixel constant, and Tukey-IRLS is
        its iterated-reweighting analog, so in the identity-kernel limit
        the mode reproduces the median exactly. Fan footprints enter as
        weights (no gap infill anywhere). The DC is spliced from the
        hydrostatic two-level median (budget-exact) in preference to the
        kernel's own k=0 response, which is finite but not budget-constrained.
        Requires ``transform="dct"``.
    corr_aggregate : {"pooled", "fan-median"}
        (``estimator="split"`` only.) ``"pooled"``: one kernel correction
        from the aggregated slope field (fill artefacts quarantined to the
        best-covered field). ``"fan-median"``: the correction is computed
        per fan on that fan's own observation footprint and the per-pixel
        MEDIAN across fans is taken — the same two-level robustness the
        slope aggregation already has, applied after the linear operator.
        Cells no surviving fan covers fall back to zero correction (pure
        hydrostatic), never NaN.
    slope_aggregate : {"pooled-pairs", "fan-median"}
        How the per-pair budget slopes reduce to the hydrostatic slope field
        (``estimator="split"`` only; IRLS owns its own ensemble reduction).
        ``"pooled-pairs"`` (default): pool EVERY pair's per-cell mean into one
        global bucket and take the per-cell median across all pairs — the flat
        two-level ``pair_median`` the production path solver uses. ``"fan-
        median"`` (legacy): median within each start's fan, then median across
        fans. The nested fan reduction makes a cell's estimate depend on how
        many *fans* reached it, and fan coverage is flow-banded, so it imprints
        flow-aligned streaks where per-fan count is thin (PIG ``tv_path``
        2026-07-04); pooled pairs remove that banding by construction while
        keeping full coverage. In this mode ``count`` is the per-cell PAIR
        count (not fan count) and ``min_fan_count`` gates on it.
    irls_iters : int
        Outer IRLS iterations for ``estimator="irls"`` (converges in ~3-5;
        stops early when weights settle).
    irls_tukey_c : float
        Tukey biweight tuning constant (4.685 = 95% Gaussian efficiency,
        same as fit_tilt_stack).
    min_fan_count : int
        Minimum fans contributing for a cell to be reported.
    fan_cache : str, optional
        Path to an ``.npz`` memo of the per-fan slope maps (the expensive
        trajectory-warp product, identical across ``estimator`` /
        ``corr_aggregate`` / kernel-parameter choices). If the file exists
        and its metadata matches (grid, window, band, fan_H, budget flag,
        velocity mode), the fan loop is skipped; otherwise the loop runs
        and the memo is (re)written. Iterating on estimator modes on a
        production stack then costs minutes, not hours.
    progress, progress_interval_s :
        Narrate fan progress (start count, ETA) at this wall-clock cadence.

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` (m ice/yr, Shean convention: negative = melt),
        ``melt_rate_hydro`` (the pure hydrostatic part — the path-solver
        quantity), ``nonhydro_corr`` (the kernel's channel correction),
        ``count`` (fans per cell), ``rmse`` (std across fans), ``H_f_mean``,
        ``flux_div``, ``a_dot``. Attrs record band, kernel and correction
        parameters.
    """
    if "time" not in h_stack.dims:
        raise ValueError("linear_inverse_budget_melt_rate requires a 'time' dim")
    if h_stack.sizes["time"] < 2:
        raise ValueError("need at least two epochs")
    if transform not in ("dct", "fft"):
        raise ValueError(f"transform must be 'dct' or 'fft', got {transform!r}")
    if max_pair_dt_yr <= min_pair_dt_yr:
        raise ValueError("max_pair_dt_yr must exceed min_pair_dt_yr")

    if fan_H not in ("trend", "fan-mean"):
        raise ValueError(f"fan_H must be 'trend' or 'fan-mean', got {fan_H!r}")
    if attribution not in ("seed", "path"):
        raise ValueError(
            f"attribution must be 'seed' or 'path', got {attribution!r}"
        )
    if estimator not in ("split", "irls"):
        raise ValueError(f"estimator must be 'split' or 'irls', got {estimator!r}")
    if corr_aggregate not in ("pooled", "fan-median"):
        raise ValueError(
            f"corr_aggregate must be 'pooled' or 'fan-median', got {corr_aggregate!r}"
        )
    if slope_aggregate not in ("pooled-pairs", "fan-median"):
        raise ValueError(
            "slope_aggregate must be 'pooled-pairs' or 'fan-median', got "
            f"{slope_aggregate!r}"
        )
    if estimator == "irls" and transform != "dct":
        raise ValueError("estimator='irls' currently requires transform='dct'")

    time_varying_vel = "time" in vx.dims and vx.sizes["time"] > 1
    vx_mean = vx.mean("time", skipna=True) if "time" in vx.dims else vx
    vy_mean = vy.mean("time", skipna=True) if "time" in vy.dims else vy

    # --- static budget fields on the Eulerian grid --------------------------
    H_f_stack = freeboard_to_thickness(h_stack, d=d, rho_w=rho_w, rho_i=rho_i)
    H_f_mean = H_f_stack.mean("time", skipna=True)
    div_u = divergence(vx_mean, vy_mean)  # 1/yr
    delta_frac = (rho_w - rho_i) / rho_w
    R_hydro = rho_w / (rho_w - rho_i)

    if floating_mask is not None:
        fmask = floating_mask.astype(bool)
        H_ref_val = (
            float(H_f_mean.where(fmask).mean(skipna=True))
            if H_ref is None else float(H_ref)
        )
        div_mean = float(div_u.where(fmask).mean(skipna=True))
    else:
        fmask = None
        H_ref_val = (
            float(H_f_mean.mean(skipna=True)) if H_ref is None else float(H_ref)
        )
        div_mean = float(div_u.mean(skipna=True))
    if not np.isfinite(H_ref_val) or H_ref_val <= 0:
        raise ValueError(f"derived H_ref={H_ref_val} is not positive finite")
    if not np.isfinite(div_mean):
        div_mean = 0.0

    tr_seconds = 2.0 * eta_bar / (rho_i * g * H_ref_val)
    gamma_val = (
        (div_mean / SECONDS_PER_YEAR) * tr_seconds if gamma is None else float(gamma)
    )

    if isinstance(a_dot, xr.DataArray):
        a_dot_da = a_dot
    else:
        a_dot_da = xr.full_like(H_f_mean, float(a_dot)).rename("a_dot")

    # Firn advection is applied per pair as the EXACT endpoint difference
    # (∫u·∇d dτ = d(x_end) − d(x_start) for static d) — never as a per-step
    # u·∇d integral. BedMachine FAC is a bilinear resample of a coarse firn
    # model, so ∇d is a piecewise-constant comb on that model's ~5–10 km
    # cell lattice; per-step sampling of the comb rode the 1/δ_f ≈ 9.4
    # hydrostatic gain into ±2–8 m/yr axis-aligned bands (PIG tv_path
    # tiles, 2026-07-04).
    if isinstance(d, xr.DataArray):
        d_fac_arr = asarray(np.asarray(d.values, dtype=np.float64))
    else:
        d_fac_arr = None

    # Per-pixel record trend of H_f for the fan strain term (fan_H="trend"):
    # the shelf thins across the record, so both a record-mean H AND a
    # fan-mean OBSERVED H (mean over epochs 0..2.5 yr while pair integrals
    # run 0..t_k) bias the strain correction δ_f·H·∇·u. Evaluating the
    # record trend at each fan's pair-weighted mid-time removes the
    # systematic part (the residual per-pair term is second-order). The
    # trend is clipped to the per-pixel observed range to bound noisy fits.
    if budget_correction and fan_H == "trend":
        obs = H_f_stack.notnull()
        t_da = xr.DataArray(
            _times_to_years(h_stack["time"].values).astype(np.float32),
            dims=("time",), coords={"time": h_stack["time"].values},
        )
        n_obs = obs.sum("time")
        St = t_da.where(obs).sum("time")
        Stt = (t_da**2).where(obs).sum("time")
        Sh = H_f_stack.sum("time", skipna=True)
        Sth = (H_f_stack * t_da).sum("time", skipna=True)
        denom_fit = (n_obs * Stt - St**2).where(n_obs >= 3)
        H_slope = ((n_obs * Sth - St * Sh) / denom_fit)
        H_icept = (Sh - H_slope * St) / n_obs.where(n_obs > 0)
        H_obs_min = H_f_stack.min("time", skipna=True)
        H_obs_max = H_f_stack.max("time", skipna=True)

    def _fan_H(h_sub: xr.DataArray, t_mid_yr: float) -> xr.DataArray:
        """H_f field for the fan's strain correction, m (NaN-free)."""
        if fan_H == "trend":
            H_w = (H_icept + H_slope * t_mid_yr).clip(H_obs_min, H_obs_max)
        else:
            H_w = freeboard_to_thickness(
                h_sub, d=d, rho_w=rho_w, rho_i=rho_i
            ).mean("time", skipna=True)
        return H_w.fillna(H_f_mean).fillna(H_ref_val)

    # --- pair fans (Shean banding, two-level median) -------------------------
    t_yr = _times_to_years(h_stack["time"].values)
    n_t = len(t_yr)
    starts = []
    for i in range(n_t):
        dt_all = t_yr - t_yr[i]
        partners = np.where(
            (dt_all >= float(min_pair_dt_yr)) & (dt_all <= float(max_pair_dt_yr))
        )[0]
        if partners.size:
            starts.append((i, partners))
    if not starts:
        raise ValueError(
            f"no epoch pairs inside the {min_pair_dt_yr}-{max_pair_dt_yr} yr band"
        )

    n_pairs_total = sum(len(p) for _, p in starts)
    ny = h_stack.sizes["y"]
    nx = h_stack.sizes["x"]
    x_coords = h_stack["x"].values
    y_coords = h_stack["y"].values
    res_x = float(x_coords[1] - x_coords[0])
    res_y = float(y_coords[0] - y_coords[1])  # y descending
    vx_arr = asarray(np.asarray(vx_mean.values, dtype=np.float64))
    vy_arr = asarray(np.asarray(vy_mean.values, dtype=np.float64))
    div_arr = asarray(np.asarray(div_u.values, dtype=np.float64))

    # Velocity provider: static (time-mean) or time-varying. Time-varying
    # mirrors lagrangian_melt_rate — per-sub-step linear interpolation between
    # bracketing velocity epochs (clamped at the ends), for the trajectory AND
    # the divergence entering the budget correction. This is first-order on
    # nonstationary shelves: with time-MEAN velocity the production path
    # solver's PIG flux inflates 88.6 -> 125.2 Gt/yr and its IQR stretches
    # ~1.5x (pig/diag_path_meanvel.py, 2026-07-03) — mean-u trajectories
    # misplace 2-yr parcels across steep near-GL gradients and the skewed
    # slope errors rectify into spurious deep melt.
    if time_varying_vel:
        vx_sorted = vx.sortby("time")
        vy_sorted = vy.sortby("time")
        v_times = pd.to_datetime(vx_sorted["time"].values)
        stack_t0 = pd.to_datetime(np.asarray(h_stack["time"].values))[0]
        v_t_years = np.array(
            [(t - stack_t0).total_seconds() / SECONDS_PER_YEAR for t in v_times],
            dtype=np.float64,
        )
        vx_tv = asarray(np.asarray(vx_sorted.values, dtype=np.float64))
        vy_tv = asarray(np.asarray(vy_sorted.values, dtype=np.float64))
        div_tv = asarray(
            np.stack(
                [
                    divergence(vx_sorted.isel(time=k), vy_sorted.isel(time=k)).values
                    for k in range(len(v_times))
                ]
            ).astype(np.float64)
        )
        vel_prov = _VelocityProvider(
            vx_arr, vy_arr, div_arr, vx_tv, vy_tv, div_tv, v_t_years
        )
        if progress:
            print(
                f"  time-varying velocity: {len(v_times)} epochs "
                f"{v_times[0].date()}..{v_times[-1].date()} — per-sub-step "
                f"linear interp (clamped ends) for trajectories and div",
                flush=True,
            )
    else:
        vel_prov = _VelocityProvider(vx_arr, vy_arr, div_arr)

    a_term_arr = (
        asarray(np.asarray((delta_frac * a_dot_da).values, dtype=np.float64))
        if budget_correction else None
    )

    # The fan loop is the expensive product and is identical across
    # estimator / corr_aggregate / kernel-parameter choices — memoize it.
    cache_meta = json.dumps(
        {
            "n_starts": len(starts), "n_pairs": int(n_pairs_total),
            "n_epochs": int(n_t),
            "min_pair_dt_yr": float(min_pair_dt_yr),
            "max_pair_dt_yr": float(max_pair_dt_yr),
            "dt_yr": float(dt_yr), "fan_H": fan_H,
            "budget_correction": int(bool(budget_correction)),
            "velocity_time_varying": int(bool(time_varying_vel)),
            "ny": int(ny), "nx": int(nx),
            "res_x": float(res_x), "res_y": float(res_y),
            "t_first": str(h_stack["time"].values[0]),
            "t_last": str(h_stack["time"].values[-1]),
            "attribution": attribution,
            "c_fallback": "mean-vel-v1",
            "firn_adv": "endpoint-exact-v1",
            "walk": "meltpy-parity-v2",
            "slope_aggregate": slope_aggregate,
        },
        sort_keys=True,
    )
    pool_pairs = slope_aggregate == "pooled-pairs"
    slope_pool = count_pool = spread_pool = None
    cache_hit = False
    if fan_cache is not None and os.path.exists(fan_cache):
        z = np.load(fan_cache, allow_pickle=False)
        meta_ok = str(z["meta_json"]) == cache_meta
        pool_ok = (not pool_pairs) or ("slope_pool" in z.files)
        if meta_ok and pool_ok:
            fan_maps = z["fan_maps"]
            dt_weighted_sum = float(z["dt_weighted_sum"])
            if pool_pairs:
                # meta encodes slope_aggregate, so a matching cache carries the
                # pooled arrays (final maps, not the multi-GB bucket).
                slope_pool = z["slope_pool"]
                count_pool = z["count_pool"]
                spread_pool = z["spread_pool"]
            cache_hit = True
            if progress:
                print(f"  fan-maps cache HIT: {fan_cache}", flush=True)
        elif progress:
            print("  fan-maps cache STALE (meta mismatch) — recomputing", flush=True)

    if not cache_hit:
        fan_maps = np.full((len(starts), ny, nx), np.nan, dtype=np.float32)
        dt_weighted_sum = 0.0
        pool_idx_chunks: list = []  # per-pair cell indices (pooled median bucket)
        pool_val_chunks: list = []  # per-pair per-cell means
        t_wall = time.time()
        last_print = t_wall
        for s_idx, (i, partners) in enumerate(starts):
            sub_idx = np.concatenate(([i], partners))
            h_sub = h_stack.isel(time=sub_idx)
            t_rel = t_yr[sub_idx] - t_yr[i]

            if budget_correction:
                t_mid = float(t_yr[i] + np.mean(t_rel[1:]) / 2.0)
                H_fan_arr = asarray(
                    np.asarray(_fan_H(h_sub, t_mid).values, dtype=np.float64)
                )
                # NaN-free fallback c-field: time-mean velocity/divergence
                # (nearly gap-free), residual holes -> SMB-only (the
                # production solver's zero-strain-per-step analog). Without
                # it, one NaN div/velocity sample poisons the parcel's
                # entire ∫c dτ and kills ALL its pairs — seen as seams
                # along quarterly mosaic-gap boundaries and a missing
                # front/margin band in the first PIG path run (2026-07-03).
                c_fb = a_term_arr - delta_frac * H_fan_arr * div_arr
                c_fb = xp.where(xp.isfinite(c_fb), c_fb, a_term_arr)
                budget = (
                    a_term_arr, H_fan_arr, d_fac_arr, delta_frac, c_fb
                )
            else:
                budget = None
            if pool_pairs:
                fan_maps[s_idx], pairs_sparse = _fan_map_walk(
                    h_sub.values.astype(np.float64), budget, vel_prov,
                    t_rel, float(t_yr[i]), dt_yr, res_x, res_y, attribution,
                    return_pairs=True,
                )
                for p_idx, p_val in pairs_sparse:
                    pool_idx_chunks.append(p_idx)
                    pool_val_chunks.append(p_val)
            else:
                fan_maps[s_idx] = _fan_map_walk(
                    h_sub.values.astype(np.float64), budget, vel_prov,
                    t_rel, float(t_yr[i]), dt_yr, res_x, res_y, attribution,
                )
            dt_weighted_sum += float(t_rel[1:].sum())

            if progress and (
                time.time() - last_print >= progress_interval_s
                or s_idx == len(starts) - 1
            ):
                last_print = time.time()
                el = last_print - t_wall
                print(
                    f"  fan {s_idx + 1}/{len(starts)} (start t={t_yr[i]:.2f} yr, "
                    f"{len(partners)} partners)  elapsed {el:.0f}s  "
                    f"ETA {el / (s_idx + 1) * (len(starts) - s_idx - 1):.0f}s",
                    flush=True,
                )
        if pool_pairs:
            if progress:
                n_ent = sum(c.size for c in pool_idx_chunks)
                print(
                    f"  pooling {len(pool_idx_chunks):,} pair maps "
                    f"({n_ent:,} cell-values) into cross-pair median...",
                    flush=True,
                )
            slope_pool, count_pool, spread_pool = _pool_pairs(
                pool_idx_chunks, pool_val_chunks, ny * nx, ny, nx, R_hydro
            )
            del pool_idx_chunks, pool_val_chunks
        if fan_cache is not None:
            extra = (
                dict(slope_pool=slope_pool, count_pool=count_pool,
                     spread_pool=spread_pool)
                if pool_pairs else {}
            )
            np.savez_compressed(
                fan_cache, fan_maps=fan_maps,
                dt_weighted_sum=np.float64(dt_weighted_sum),
                meta_json=np.array(cache_meta), **extra,
            )
            if progress:
                print(f"  fan-maps cache saved: {fan_cache}", flush=True)

    if pool_pairs:
        # Flat cross-pair pooled median (REF/melt.py pair_median parity): no
        # fan-count banding. `count` is the per-cell PAIR count.
        slope_agg = np.asarray(slope_pool, dtype=np.float64)
        spread = np.asarray(spread_pool, dtype=np.float64)
        count = np.asarray(count_pool, dtype=np.int32)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            slope_agg = np.nanmedian(fan_maps, axis=0).astype(np.float64)  # m/yr
            spread = np.nanstd(fan_maps.astype(np.float64), axis=0) * R_hydro
        count = np.isfinite(fan_maps).sum(axis=0).astype(np.int32)
    slope_agg = np.where(count >= int(min_fan_count), slope_agg, np.nan)
    if fmask is not None:
        slope_agg = np.where(fmask.values, slope_agg, np.nan)
    if not np.isfinite(slope_agg).any():
        raise ValueError("no cell survived the pair band / fan-count gates")

    # --- hydrostatic part (owns the budget-exact DC) -------------------------
    m_hydro = R_hydro * slope_agg  # m ice/yr, Shean sign preserved

    finite_s = np.isfinite(slope_agg)
    res_m = float(abs(h_stack["x"].values[1] - h_stack["x"].values[0]))
    sigma_pix = max(H_ref_val / res_m, 1.0)
    sigma_corr_m = (
        H_ref_val if corr_prefilter_sigma_m is None
        else float(corr_prefilter_sigma_m)
    )
    coords = {"y": h_stack["y"].values, "x": h_stack["x"].values}

    # Two-epoch kernel at the count-weighted mean pair baseline: the OLS-
    # slope kernel over {0, Δt̄} is exactly the pair-difference response.
    mean_pair_dt_yr = dt_weighted_sum / max(n_pairs_total, 1)
    t_secs_kernel = np.array([0.0, mean_pair_dt_yr * SECONDS_PER_YEAR])

    irls_diag: dict = {}
    if estimator == "irls":
        # Robust masked ensemble inversion: the two-level median's
        # robustness moved INSIDE the inverse (Tukey-IRLS over all per-fan
        # slope maps through the kernel; footprints as weights, no infill).
        if progress:
            print(
                f"  IRLS ensemble kernel solve: {fan_maps.shape[0]} fan maps, "
                f"Tukey c={irls_tukey_c}, {irls_iters} outer iters max",
                flush=True,
            )
        K_dot, k_rms, kmag = _build_ols_kernel_dct(
            H_ref_val, eta_bar, rho_i, rho_w, g, gamma_val, theta,
            t_secs_kernel, ny, nx, res_x, res_y,
        )
        # Roughness (bi-Laplacian) regularization spectrum at the same
        # scale the split path pre-filters at: flat Tikhonov alone lets
        # the masked deconvolution amplify per-fan slope noise at high k
        # (gate_flux_amplitude G2-G4: q25 blew out ~3x, flux -45%).
        lam2_k = (reg * k_rms) ** 2 * (
            (1.0 + (kmag * sigma_corr_m) ** 2) ** 2 if sigma_corr_m > 0 else 1.0
        )
        m_int, irls_diag = _irls_ensemble_kernel_solve(
            fan_maps / np.float32(SECONDS_PER_YEAR),  # m/yr -> m/s
            K_dot, lam2_k,
            tukey_c=irls_tukey_c, n_irls=irls_iters, progress=progress,
        )
        melt_irls = -m_int * SECONDS_PER_YEAR  # Shean convention, zero-mean
        # The kernel's k=0 bin is finite (kernel_time_integral_stationary),
        # but the hydrostatic two-level median is budget-exact, so it owns
        # the level here.
        dc_splice = float(
            np.nanmedian(m_hydro[finite_s]) - np.nanmedian(melt_irls[finite_s])
        )
        melt = np.where(finite_s, melt_irls + dc_splice, np.nan)
        nonhydro_corr = melt - np.where(finite_s, m_hydro, np.nan)  # diagnostic
    elif corr_aggregate == "pooled":
        nonhydro_corr = _pooled_kernel_correction(
            slope_agg, finite_s, coords=coords, res_m=res_m,
            sigma_pix=sigma_pix, sigma_corr_m=sigma_corr_m,
            H_ref_val=H_ref_val, t_secs_kernel=t_secs_kernel,
            eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
            gamma_val=gamma_val, theta=theta, reg=reg, transform=transform,
            R_hydro=R_hydro,
        )
        melt = np.where(finite_s, m_hydro + nonhydro_corr, np.nan)
    else:
        # corr_aggregate == "fan-median": the kernel correction gets the
        # same across-fan median robustness the slope already has — computed
        # per fan on that fan's own footprint (honest coverage), median
        # across contributing fans, zero where no fan survives.
        from scipy.ndimage import gaussian_filter

        corr_stack = np.full(fan_maps.shape, np.nan, dtype=np.float32)
        t_c0 = time.time()
        for f_idx in range(fan_maps.shape[0]):
            s_f = fan_maps[f_idx].astype(np.float64)
            W_f = np.isfinite(s_f)
            if W_f.sum() < 500:
                continue
            med_f = float(np.median(s_f[W_f]))
            filled = _nan_aware_gaussian_infill_2d(
                s_f - med_f, (sigma_pix, sigma_pix), max_iters=5,
            ) + med_f
            if sigma_corr_m > 0:
                filled = gaussian_filter(
                    filled, sigma=sigma_corr_m / res_m, mode="nearest"
                )
            s_da = xr.DataArray(
                filled / SECONDS_PER_YEAR, dims=("y", "x"), coords=coords,
                name="dh_dt",
            )
            mk = inverse_dhdt(
                s_da, H=H_ref_val, t_secs=t_secs_kernel,
                eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
                alpha=0.0, alpha_y=0.0, gamma=gamma_val, theta=theta,
                reg=reg, transform=transform,
                recover_dc=False, a_dot_dc=0.0,
            )
            corr_stack[f_idx] = np.where(
                W_f, mk.values - R_hydro * filled, np.nan
            ).astype(np.float32)
            if progress and (f_idx + 1) % 50 == 0:
                print(
                    f"    fan-median corr {f_idx + 1}/{fan_maps.shape[0]}  "
                    f"[{time.time() - t_c0:.0f}s]",
                    flush=True,
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr_med = np.nanmedian(corr_stack.astype(np.float64), axis=0)
        del corr_stack
        nonhydro_corr = np.where(np.isfinite(corr_med), corr_med, 0.0)
        nonhydro_corr = nonhydro_corr - float(np.nanmedian(nonhydro_corr[finite_s]))
        melt = np.where(finite_s, m_hydro + nonhydro_corr, np.nan)

    melt_da = xr.DataArray(melt, dims=("y", "x"), coords=coords, name="melt_rate")
    hydro_da = xr.DataArray(
        np.where(finite_s, m_hydro, np.nan), dims=("y", "x"), coords=coords,
        name="melt_rate_hydro",
    )
    corr_da = xr.DataArray(
        np.where(finite_s, nonhydro_corr, np.nan), dims=("y", "x"), coords=coords,
        name="nonhydro_corr",
    )
    if fmask is not None:
        melt_da = melt_da.where(fmask)
        hydro_da = hydro_da.where(fmask)
        corr_da = corr_da.where(fmask)
        H_f_mean = H_f_mean.where(fmask)

    fd = flux_divergence(H_f_mean, vx_mean, vy_mean)
    return xr.Dataset(
        {
            "melt_rate": melt_da,
            "melt_rate_hydro": hydro_da,
            "nonhydro_corr": corr_da,
            "count": xr.DataArray(count, dims=("y", "x"), coords=coords),
            "rmse": xr.DataArray(spread, dims=("y", "x"), coords=coords),
            "H_f_mean": H_f_mean,
            "flux_div": fd,
            "a_dot": a_dot_da,
        },
        attrs={
            "equation": (
                "budget-corrected pair-banded Stubblefield inverse: two-level "
                "pair_median of R·(Dh/Dt − δ_f(ȧ − H_f∇·u) − Δd/Δt) along "
                "trajectories, plus kernel non-hydrostatic channel correction"
            ),
            "units": "m ice yr^-1; Shean convention: negative = melt, positive = accretion",
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref_val,
            "gamma_dimless": gamma_val,
            "tr_yr": tr_seconds / SECONDS_PER_YEAR,
            "alpha": 0.0, "theta": theta, "reg": reg,
            "transform": transform,
            "min_pair_dt_yr": float(min_pair_dt_yr),
            "max_pair_dt_yr": float(max_pair_dt_yr),
            "mean_pair_dt_yr": float(mean_pair_dt_yr),
            "velocity_time_varying": int(bool(time_varying_vel)),
            "fan_H": fan_H,
            "attribution_mode": attribution,
            "estimator": estimator,
            "corr_aggregate": corr_aggregate if estimator == "split" else "n/a",
            "slope_aggregate": slope_aggregate,
            "count_semantics": (
                "per-cell pair count (pooled cross-pair median)"
                if pool_pairs else "per-cell fan count (median of fan medians)"
            ),
            **{f"irls_{k}": v for k, v in irls_diag.items()},
            "corr_prefilter_sigma_m": float(sigma_corr_m),
            "n_starts": len(starts),
            "n_pairs": int(n_pairs_total),
            "budget_correction": int(bool(budget_correction)),
            "min_fan_count": int(min_fan_count),
            "dt_yr": float(dt_yr),
            "attribution": (
                "seed (fan-start position), one map per start fan; two-level "
                "median (pair_median mosaic)"
                if attribution == "seed" else
                "path (deposited along visited cells per sub-step, non-melt "
                "terms re-localized; melt.py output='path' parity), per-pair "
                "weighted mean -> fan median -> cross-fan median"
            ),
        },
    )


def linear_inverse_eulerian_budget_melt_rate(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    a_dot: xr.DataArray | float = 0.0,
    d: xr.DataArray | float = 0.0,
    floating_mask: xr.DataArray | None = None,
    H_ref: float | None = None,
    eta_bar: float = 1e14,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    gamma: float | None = None,
    theta: float = 1e-14,
    reg: float = 1e-1,
    transform: str = "dct",
    corr_prefilter_sigma_m: float | None = None,
    min_count: int = 3,
    robust_dh_dt: bool = False,
    vel_smooth_sigma_m: float | None = None,
    progress: bool = False,
) -> xr.Dataset:
    r"""Basal melt rate via the budget-corrected EULERIAN Stubblefield inverse.

    The Eulerian twin of :func:`linear_inverse_budget_melt_rate`. Instead of
    warping Shean-banded pair fans, the hydrostatic channel here IS the
    production Eulerian estimator — :func:`stereo_melt.melt.eulerian_melt_rate`
    called directly, so ``melt_rate_hydro`` is bit-identical to the production
    Eulerian melt rate (per-cell dh/dt regression over ALL epochs, plus the
    flux-divergence and SMB budget terms). The identity that makes the split
    exact: with :math:`s = \delta_f\,\dot b` the melt-attributable freeboard
    rate and :math:`R = \rho_w/(\rho_w-\rho_i) = 1/\delta_f`,

    .. math::
        R\,s \;=\; \dot b_\mathrm{eul}
              \;=\; \partial H_f/\partial t + \nabla\!\cdot(H_f u) - \dot a,

    so the recovered field is the exact pointwise hydrostatic estimate (DC
    inherent, no transform ringing) plus the same band-limited
    non-hydrostatic channel correction the pair-banded inverse uses
    (``inverse_dhdt(s_filled) − R·s_filled`` on a smoothly infilled,
    pre-smoothed copy), with the OLS-slope kernel built over the record's
    ACTUAL epoch times — matching the per-cell regression that produced the
    slope, rather than the two-epoch pair kernel.

    When to prefer it: the pair-banded/path inverses are constrained only by
    1.5–2.5 yr DEM pairs, so cells with thin in-band pair coverage (strip
    margins, the grid-west edge of a stack) inherit the path solver's
    under-determination. This estimator pools EVERY epoch a cell has seen
    into one (optionally Tukey-robust) trend — a better-conditioned data
    model where epochs are plentiful but banded pairs are not. The price is
    the usual Eulerian sensitivity: :math:`\nabla\!\cdot(H_f u)` is evaluated
    at fixed cells from the time-MEAN velocity and record-mean thickness, and
    a per-cell linear trend is only the material derivative up to
    flow-smearing of along-flow melt gradients.

    Parameters
    ----------
    h_stack, vx, vy, a_dot, d, floating_mask :
        As in :func:`linear_inverse_budget_melt_rate`. A ``time`` dimension
        on the velocity is reduced to its mean (the Eulerian solver's
        convention) — there is no per-sub-step sampling here.
    H_ref, eta_bar, gamma, theta, reg, transform, corr_prefilter_sigma_m :
        Kernel-correction parameters, identical semantics to
        :func:`linear_inverse_budget_melt_rate` (default pre-smooth
        ``H_ref``; pass 0 to disable).
    min_count : int
        Minimum finite epochs per cell for the dh/dt regression
        (:func:`stereo_melt.melt.eulerian_melt_rate` passthrough).
    robust_dh_dt : bool
        Tukey-IRLS per-cell regression instead of OLS (production Eulerian
        config on PIG/Nansen uses True).
    vel_smooth_sigma_m : float, optional
        Gaussian velocity pre-smooth passthrough.

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` (m ice/yr, Shean convention: negative = melt),
        ``melt_rate_hydro`` (== production Eulerian melt rate, exactly),
        ``nonhydro_corr``, ``count`` (per-cell EPOCHS in the regression —
        not pairs), ``rmse`` (regression rmse, m ice-eq), ``dHdt``,
        ``flux_div``, ``H_f_mean``, ``a_dot``.
    """
    if "time" not in h_stack.dims:
        raise ValueError(
            "linear_inverse_eulerian_budget_melt_rate requires a 'time' dim"
        )
    if h_stack.sizes["time"] < 2:
        raise ValueError("need at least two epochs")
    if transform not in ("dct", "fft"):
        raise ValueError(f"transform must be 'dct' or 'fft', got {transform!r}")

    # Lazy import: melt.py imports dynamics.lagrangian_inverse, so a
    # module-level import here would cycle through the dynamics package
    # __init__ while this module is still initializing.
    from ..melt import eulerian_melt_rate

    t0 = time.time()
    euler = eulerian_melt_rate(
        h_stack, vx, vy, a_dot=a_dot, d=d, rho_w=rho_w, rho_i=rho_i,
        min_count=min_count, robust_dh_dt=robust_dh_dt,
        vel_smooth_sigma_m=vel_smooth_sigma_m,
    )
    if progress:
        print(
            f"  eulerian hydro channel: {h_stack.sizes['time']} epochs, "
            f"robust={robust_dh_dt}, min_count={min_count}  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    delta_frac = (rho_w - rho_i) / rho_w
    R_hydro = rho_w / (rho_w - rho_i)

    vx_mean = vx.mean("time", skipna=True) if "time" in vx.dims else vx
    vy_mean = vy.mean("time", skipna=True) if "time" in vy.dims else vy
    div_u = divergence(vx_mean, vy_mean)  # 1/yr

    H_f_mean = euler["H_f_mean"]
    if floating_mask is not None:
        fmask = floating_mask.astype(bool)
        H_ref_val = (
            float(H_f_mean.where(fmask).mean(skipna=True))
            if H_ref is None else float(H_ref)
        )
        div_mean = float(div_u.where(fmask).mean(skipna=True))
    else:
        fmask = None
        H_ref_val = (
            float(H_f_mean.mean(skipna=True)) if H_ref is None else float(H_ref)
        )
        div_mean = float(div_u.mean(skipna=True))
    if not np.isfinite(H_ref_val) or H_ref_val <= 0:
        raise ValueError(f"derived H_ref={H_ref_val} is not positive finite")
    if not np.isfinite(div_mean):
        div_mean = 0.0

    tr_seconds = 2.0 * eta_bar / (rho_i * g * H_ref_val)
    gamma_val = (
        (div_mean / SECONDS_PER_YEAR) * tr_seconds if gamma is None else float(gamma)
    )

    m_hydro = np.asarray(euler["melt_rate"].values, dtype=np.float64)
    if fmask is not None:
        m_hydro = np.where(np.asarray(fmask.values, dtype=bool), m_hydro, np.nan)
    # Melt-attributable freeboard rate: R·s == the Eulerian estimate exactly.
    slope_agg = delta_frac * m_hydro  # m/yr
    finite_s = np.isfinite(slope_agg)
    if not finite_s.any():
        raise ValueError("no cell survived the Eulerian regression / mask gates")

    res_m = float(abs(h_stack["x"].values[1] - h_stack["x"].values[0]))
    sigma_pix = max(H_ref_val / res_m, 1.0)
    sigma_corr_m = (
        H_ref_val if corr_prefilter_sigma_m is None
        else float(corr_prefilter_sigma_m)
    )
    coords = {"y": h_stack["y"].values, "x": h_stack["x"].values}
    # OLS-slope kernel over the record's actual epoch times (inverse_dhdt
    # centers them internally) — the response of "OLS trend of kernel-filtered
    # topography", matching the estimator that produced the slope field.
    t_secs_kernel = _times_to_years(h_stack["time"].values) * SECONDS_PER_YEAR

    if progress:
        print(
            f"  kernel correction: H_ref={H_ref_val:.0f} m, "
            f"gamma={gamma_val:.2e}, t_r={tr_seconds / SECONDS_PER_YEAR:.2f} yr, "
            f"presmooth={sigma_corr_m:.0f} m, reg={reg:g}",
            flush=True,
        )
    nonhydro_corr = _pooled_kernel_correction(
        slope_agg, finite_s, coords=coords, res_m=res_m,
        sigma_pix=sigma_pix, sigma_corr_m=sigma_corr_m,
        H_ref_val=H_ref_val, t_secs_kernel=t_secs_kernel,
        eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
        gamma_val=gamma_val, theta=theta, reg=reg, transform=transform,
        R_hydro=R_hydro,
    )
    melt = np.where(finite_s, m_hydro + nonhydro_corr, np.nan)

    melt_da = xr.DataArray(melt, dims=("y", "x"), coords=coords, name="melt_rate")
    hydro_da = xr.DataArray(
        np.where(finite_s, m_hydro, np.nan), dims=("y", "x"), coords=coords,
        name="melt_rate_hydro",
    )
    corr_da = xr.DataArray(
        np.where(finite_s, nonhydro_corr, np.nan), dims=("y", "x"),
        coords=coords, name="nonhydro_corr",
    )
    count_da = euler["count"]
    rmse_da = euler["rmse"]
    dHdt_da = euler["dHdt"]
    fd_da = euler["flux_div"]
    a_dot_da = euler["a_dot"]
    if fmask is not None:
        melt_da = melt_da.where(fmask)
        hydro_da = hydro_da.where(fmask)
        corr_da = corr_da.where(fmask)
        H_f_mean = H_f_mean.where(fmask)
        dHdt_da = dHdt_da.where(fmask)
        fd_da = fd_da.where(fmask)

    return xr.Dataset(
        {
            "melt_rate": melt_da,
            "melt_rate_hydro": hydro_da,
            "nonhydro_corr": corr_da,
            "count": count_da,
            "rmse": rmse_da,
            "dHdt": dHdt_da,
            "flux_div": fd_da,
            "H_f_mean": H_f_mean,
            "a_dot": a_dot_da,
        },
        attrs={
            "equation": (
                "budget-corrected Eulerian Stubblefield inverse: "
                "R·s = dH_f/dt + div(H_f u) − a_dot (per-cell all-epoch "
                "regression; == eulerian_melt_rate exactly) plus kernel "
                "non-hydrostatic channel correction"
            ),
            "units": "m ice yr^-1; Shean convention: negative = melt, positive = accretion",
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "H_ref_m": H_ref_val,
            "gamma_dimless": gamma_val,
            "tr_yr": tr_seconds / SECONDS_PER_YEAR,
            "alpha": 0.0, "theta": theta, "reg": reg,
            "transform": transform,
            "estimator": "eulerian-split",
            "kernel_t_secs": "record epoch times (OLS-slope kernel)",
            "n_epochs": int(h_stack.sizes["time"]),
            "min_count": int(min_count),
            "robust_dh_dt": int(bool(robust_dh_dt)),
            "vel_smooth_sigma_m": float(vel_smooth_sigma_m or 0.0),
            "corr_prefilter_sigma_m": float(sigma_corr_m),
            "count_semantics": "per-cell epoch count (dh/dt regression)",
            "hydro_channel": (
                "melt_rate_hydro == stereo_melt.melt.eulerian_melt_rate "
                "output (bit-identical; same call)"
            ),
        },
    )
