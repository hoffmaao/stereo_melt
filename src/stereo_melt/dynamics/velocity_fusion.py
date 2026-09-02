# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Spatiotemporal velocity-field fusion (harmonic infill + EOF + Gaussian process).

The time-varying Lagrangian melt solver (:func:`stereo_melt.melt.lagrangian_melt_rate`)
advects parcels through a stack of velocity epochs. Real velocity products are
(a) spatially gappy (decorrelation near shear margins / calving fronts leaves
NaN holes that inject spurious structure into ``div(u)``), (b) noisy, and
(c) heterogeneous in cadence and resolution — e.g. a sub-annual 250 m mosaic
record that stops mid-period plus coarser annual mosaics that extend past it.

This module fuses any such collection of regridded velocity observations into a
single continuous, gap-free, denoised field evaluated on a caller-chosen target
time axis. The pipeline:

1. **Harmonic spatial infill** (:func:`harmonic_infill_2d`) — fill each epoch's
   NaN gaps by solving Laplace's equation on the gap cells with the surrounding
   finite cells as Dirichlet boundary data. The minimum-curvature (harmonic)
   fill adds no spurious divergence structure, unlike zero-fill or a wide
   Gaussian blur. Cells outside the observed ice domain stay NaN.
2. **EOF / truncated SVD** (:func:`velocity_eof`) — decompose the gap-filled
   ``[n_epoch, 2*n_pixel]`` vector matrix (``vx`` then ``vy`` columns) into a
   few spatial modes and their principal-component time series. Truncation
   denoises and exposes the low-rank structure of ice-shelf flow (a dominant
   mean field plus a slowly evolving acceleration pattern).
3. **GP temporal model** (:func:`gp_fuse_series`) — fit a Gaussian process
   (smooth trend + optional annual periodic kernel) to each retained PC time
   series, then predict it on the target axis. The GP *fuses* observations of
   different cadence/precision (weighted by ``obs_weight``) and yields a smooth,
   denoised interpolant with uncertainty.
4. **Reconstruct** — recombine modes × predicted coefficients + the temporal
   mean, split back into ``vx``/``vy`` on the target time axis.

**Anchor, do not extrapolate.** The intended use is to *interpolate/fuse* across
the span bracketed by real observations — NOT to extrapolate a pre-gap trend
into an unobserved window. For Pine Island Glacier specifically, ice-front
retreat *caused* the flow speedup (Joughin et al. 2021, Sci. Adv.), so a trend
extrapolated from the pre-calving record would miss the calving-driven change.
Always include real observations spanning the target window in the input;
the GP then anchors on them. Vector (rather than speed/direction) EOF is the
default for the same reason — a stationary-direction prior would suppress any
real post-calving change in flow direction that the observations carry.

The decomposition is small (tens of epochs, a handful of modes) and runs on the
CPU with numpy/scipy/sklearn; it is not a backend-toggled array stage.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from scipy.sparse.linalg import spsolve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    ExpSineSquared,
    RBF,
    WhiteKernel,
)


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"  [velocity_fusion] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# 1. Harmonic (Laplace) spatial infill
# --------------------------------------------------------------------------- #
def harmonic_infill_2d(
    field: np.ndarray,
    domain_mask: np.ndarray,
    max_unknowns: int = 600_000,
    verbose: bool = False,
) -> np.ndarray:
    r"""Fill NaN gaps inside ``domain_mask`` by solving Laplace's equation.

    Gap cells (``domain_mask & ~isfinite``) become the unknowns; finite cells
    are Dirichlet boundary values. Neighbours outside ``domain_mask`` are
    dropped (Neumann / no-flux at the domain edge) rather than pulled toward
    zero. The result is the minimum-curvature interpolant on the gap, which —
    unlike a zero-fill or Gaussian blur — introduces no spurious ``div(u)``.

    Cells outside ``domain_mask`` are returned as NaN. Any gap component with
    no finite Dirichlet support (fully edge-bounded) is left to a small
    Tikhonov term + domain-mean fallback so the solve never goes singular.

    Parameters
    ----------
    field : ndarray, shape (ny, nx)
        2-D field with NaN gaps.
    domain_mask : ndarray of bool, shape (ny, nx)
        Cells to fill / keep. Typically pixels observed in enough epochs.
    max_unknowns : int
        Above this gap size the direct sparse solve is skipped in favour of a
        nan-aware iterative Gaussian fill (guards pathological whole-grid gaps).
    """
    field = np.asarray(field, dtype=np.float64)
    domain = np.asarray(domain_mask, dtype=bool)
    out = np.where(domain, field, np.nan).astype(np.float64)

    finite = np.isfinite(out) & domain
    gap = domain & ~finite
    n_unk = int(gap.sum())
    if n_unk == 0:
        return out
    if not finite.any():
        return out  # nothing to anchor on; leave NaN

    if n_unk > max_unknowns:
        _log(f"gap {n_unk} > max_unknowns; Gaussian fallback", verbose)
        return _gaussian_fallback(out, domain)

    ny, nx = out.shape
    lin = -np.ones((ny, nx), dtype=np.int64)
    gj, gi = np.where(gap)
    lin[gj, gi] = np.arange(n_unk)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(n_unk, dtype=np.float64)

    for k in range(n_unk):
        r0, c0 = int(gj[k]), int(gi[k])
        diag = 0.0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r0 + dr, c0 + dc
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            if not domain[rr, cc]:
                continue  # Neumann at domain edge
            diag += 1.0
            if gap[rr, cc]:
                rows.append(k)
                cols.append(int(lin[rr, cc]))
                data.append(-1.0)
            else:
                b[k] += out[rr, cc]  # Dirichlet
        rows.append(k)
        cols.append(k)
        data.append(diag if diag > 0 else 1.0)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n_unk, n_unk))
    A = A + 1e-8 * sp.identity(n_unk, format="csr")  # nonsingular for islands
    sol = spsolve(A, b)
    out[gj, gi] = sol

    if not np.isfinite(out[domain]).all():
        dmean = float(np.nanmean(out[domain]))
        bad = domain & ~np.isfinite(out)
        out[bad] = dmean
        _log(f"{int(bad.sum())} isolated cells set to domain mean", verbose)
    return out


def _gaussian_fallback(field: np.ndarray, domain: np.ndarray) -> np.ndarray:
    r"""Nan-aware iterative Gaussian infill restricted to ``domain`` (fallback)."""
    from .lagrangian_inverse import _nan_aware_gaussian_infill_2d

    filled = _nan_aware_gaussian_infill_2d(
        np.where(domain, field, np.nan), sigma_pix=(4.0, 4.0), max_iters=8
    )
    return np.where(domain, filled, np.nan)


def build_domain_mask(finite_stack: np.ndarray, coverage_frac: float) -> np.ndarray:
    r"""Domain = pixels finite in at least ``coverage_frac`` of epochs.

    ``finite_stack`` is a boolean ``(n_epoch, ny, nx)`` array. Ocean / rock /
    permanently-void cells (finite in no or few epochs) are excluded, so the
    EOF operates over a consistent, physically-observed footprint.
    """
    frac = finite_stack.mean(axis=0)
    return frac >= float(coverage_frac)


# --------------------------------------------------------------------------- #
# 2. EOF / truncated SVD
# --------------------------------------------------------------------------- #
def velocity_eof(matrix: np.ndarray, n_modes: int) -> dict:
    r"""Truncated-SVD (EOF) decomposition of a centred space-time matrix.

    Parameters
    ----------
    matrix : ndarray, shape (n_epoch, n_feature)
        Gap-free vector matrix (``vx`` columns then ``vy`` columns).
    n_modes : int
        Maximum number of modes to retain (capped at ``n_epoch - 1``).

    Returns
    -------
    dict with keys
        ``mean`` (n_feature,) temporal mean removed before the SVD;
        ``modes`` (k, n_feature) spatial EOFs (rows of ``Vt``);
        ``coeffs`` (n_epoch, k) principal-component time series (``U*S``);
        ``var_explained`` (k,) fraction of total variance per mode;
        ``k`` the number retained.
    """
    n_epoch = matrix.shape[0]
    k = int(min(n_modes, n_epoch - 1))
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    total = float((S**2).sum()) or 1.0
    return {
        "mean": mean,
        "modes": Vt[:k],
        "coeffs": U[:, :k] * S[:k],
        "var_explained": (S[:k] ** 2) / total,
        "singular_values": S[:k],
        "k": k,
    }


# --------------------------------------------------------------------------- #
# 3. GP temporal model
# --------------------------------------------------------------------------- #
def gp_fuse_series(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    t_pred: np.ndarray,
    obs_alpha: np.ndarray | float = 1e-3,
    seasonal: bool = True,
    trend_length_yr: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, object]:
    r"""Gaussian-process fit of one coefficient series, predicted on ``t_pred``.

    Times are in decimal years. The kernel is a smooth trend (``RBF``) plus an
    optional fixed-period annual term (``ExpSineSquared``, period 1 yr) plus a
    ``WhiteKernel`` learned noise floor. ``obs_alpha`` adds per-observation
    noise variance (use a per-epoch array to down-weight noisier sources).

    Returns ``(mean_pred, std_pred, fitted_gp)``.
    """
    t_obs = np.asarray(t_obs, dtype=float).reshape(-1, 1)
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    t_pred = np.asarray(t_pred, dtype=float).reshape(-1, 1)

    span = float(t_obs.max() - t_obs.min()) or 1.0
    y_scale = float(np.std(y_obs)) or 1.0
    kernel = ConstantKernel(y_scale**2, (1e-3 * y_scale**2, 1e3 * y_scale**2)) * RBF(
        length_scale=trend_length_yr, length_scale_bounds=(0.5, max(2.0, 3 * span))
    )
    if seasonal:
        kernel = kernel + ConstantKernel(
            0.1 * y_scale**2, (1e-4 * y_scale**2, 1e2 * y_scale**2)
        ) * ExpSineSquared(
            length_scale=1.0,
            periodicity=1.0,
            periodicity_bounds="fixed",
            length_scale_bounds=(0.2, 10.0),
        )
    kernel = kernel + WhiteKernel(
        noise_level=(0.05 * y_scale) ** 2,
        noise_level_bounds=(1e-6 * y_scale**2 + 1e-12, y_scale**2 + 1e-9),
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=obs_alpha,
        normalize_y=True,
        n_restarts_optimizer=2,
    )
    gp.fit(t_obs, y_obs)
    mean, std = gp.predict(t_pred, return_std=True)
    return mean, std, gp


def kalman_fuse_series(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    t_pred: np.ndarray,
    obs_alpha: np.ndarray | float = 1e-3,
    seasonal: bool = True,
    trend_length_yr: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    r"""State-space (Kalman filter + RTS smoother) fit of one coefficient series.

    The O(n) state-space dual of :func:`gp_fuse_series`. State is a local-linear-
    trend (integrated random walk) ``[level, slope]`` -- the SDE form of a
    cubic-spline / Matern-3/2 smoother -- optionally augmented with an annual
    harmonic ``[cos, sin]`` pair. A forward Kalman filter then a
    Rauch-Tung-Striebel backward smooth run over the sorted union of observation
    and prediction times, so the series is interpolated AND denoised in a single
    pass with per-observation noise fusion and a returned 1-sigma uncertainty.

    Unlike the stationary RBF kernel in :func:`gp_fuse_series`, the explicit
    ``slope`` state follows a non-stationary (e.g. calving-driven) acceleration
    without a global-trend prior -- the regime the module docstring warns a
    stationary kernel mishandles. Times are decimal years; ``obs_alpha`` is the
    per-observation *relative* measurement variance (scalar or per-epoch array;
    larger = down-weighted), scaled internally by the series variance with a
    noise floor so the smoother denoises rather than chasing per-epoch noise.

    Returns ``(mean_pred, std_pred, info)`` matching the :func:`gp_fuse_series`
    signature (third element is a descriptive string rather than a GP object).
    """
    t_obs = np.asarray(t_obs, float).ravel()
    y_obs = np.asarray(y_obs, float).ravel()
    t_pred = np.asarray(t_pred, float).ravel()
    R_rel = np.asarray(obs_alpha, float)
    if R_rel.ndim == 0:
        R_rel = np.full(t_obs.shape, float(R_rel))

    srt = np.argsort(t_obs, kind="stable")
    t_obs, y_obs, R_rel = t_obs[srt], y_obs[srt], R_rel[srt]
    y_mean = float(np.mean(y_obs))
    y_scale = float(np.std(y_obs)) or 1.0

    # Measurement variance: relative alpha scaled to series variance, floored so
    # the smoother denoises rather than interpolating per-epoch noise (the role
    # the GP's learned WhiteKernel plays in gp_fuse_series).
    R = np.maximum(R_rel * y_scale**2, (0.05 * y_scale) ** 2)
    # Integrated-random-walk slope diffusion: ~y_scale drift over trend_length.
    L = max(float(trend_length_yr), 0.25)
    q = (y_scale**2) / (L**3)
    seasonal = bool(seasonal)
    ns = 4 if seasonal else 2
    wf = 2.0 * np.pi  # annual angular frequency (rad / yr)
    qh = (0.05 * y_scale) ** 2  # harmonic process-noise floor

    H = np.zeros((1, ns))
    H[0, 0] = 1.0
    if seasonal:
        H[0, 2] = 1.0  # observed = level + cosine component

    def _FQ(dt: float):
        F = np.eye(ns)
        F[0, 1] = dt
        Q = np.zeros((ns, ns))
        Q[0, 0] = q * dt**3 / 3.0
        Q[0, 1] = Q[1, 0] = q * dt**2 / 2.0
        Q[1, 1] = q * dt
        if seasonal:
            th = wf * dt
            c, s = np.cos(th), np.sin(th)
            F[2, 2], F[2, 3] = c, s
            F[3, 2], F[3, 3] = -s, c
            Q[2, 2] = Q[3, 3] = qh * dt
        return F, Q

    t_all = np.unique(np.concatenate([t_obs, t_pred]))
    n = t_all.size
    if n == 1:  # degenerate: single time
        return np.full(t_pred.shape, y_mean), np.full(t_pred.shape, y_scale), "kalman(degenerate)"
    obs_at: dict[int, list[int]] = {}
    for j, i in enumerate(np.searchsorted(t_all, t_obs)):
        obs_at.setdefault(int(i), []).append(j)

    x = np.zeros(ns)
    x[0] = y_mean
    P = np.eye(ns) * (10.0 * y_scale) ** 2
    xf = np.zeros((n, ns)); Pf = np.zeros((n, ns, ns))
    xp = np.zeros((n, ns)); Pp = np.zeros((n, ns, ns)); Fs = np.zeros((n, ns, ns))
    for i in range(n):
        dt = float(t_all[i] - t_all[i - 1]) if i > 0 else 0.0
        F, Q = _FQ(dt)
        x = F @ x
        P = F @ P @ F.T + Q
        xp[i], Pp[i], Fs[i] = x, P, F
        for j in obs_at.get(i, ()):  # measurement update(s) at this time
            innov = y_obs[j] - float(H @ x)
            S = float(H @ P @ H.T) + R[j]
            K = (P @ H.T)[:, 0] / S
            x = x + K * innov
            P = (np.eye(ns) - np.outer(K, H[0])) @ P
        xf[i], Pf[i] = x, P

    xs, Ps = xf.copy(), Pf.copy()
    for i in range(n - 2, -1, -1):
        Ppi = Pp[i + 1]
        inv = np.linalg.pinv(Ppi)
        C = Pf[i] @ Fs[i + 1].T @ inv
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T

    ip = np.searchsorted(t_all, t_pred)
    mean_pred = np.array([float(H @ xs[k]) for k in ip])
    std_pred = np.array([np.sqrt(max(float(H @ Ps[k] @ H.T), 0.0)) for k in ip])
    return mean_pred, std_pred, f"kalman(L={L:.1f}yr,seasonal={seasonal})"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _nan_aware_gaussian(field2d: np.ndarray, domain: np.ndarray, sigma_pix: float) -> np.ndarray:
    r"""Normalized-convolution Gaussian smooth restricted to ``domain``.

    Smooths only over finite, in-domain cells (``filtered = G*(f·m)/G*m``) so
    NaN gaps and the domain edge do not bleed zeros into the result. Cells
    outside ``domain`` stay NaN. Used to apply a modest (<= grid-scale) spatial
    Gaussian to the *velocity* field for a less-noisy ``div(u)`` WITHOUT coarsening
    the melt product -- the smoothing length is a caller-set parameter.
    """
    from scipy.ndimage import gaussian_filter

    if not (sigma_pix and sigma_pix > 0):
        return np.where(domain, field2d, np.nan)
    m = domain & np.isfinite(field2d)
    a = np.where(m, field2d, 0.0)
    num = gaussian_filter(a, sigma_pix, mode="nearest")
    den = gaussian_filter(m.astype(float), sigma_pix, mode="nearest")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.nan)
    return np.where(domain, out, np.nan)
def _decimal_year(times) -> np.ndarray:
    r"""Convert datetime-like values to decimal years."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
    year_start = pd.to_datetime(idx.year.astype(str) + "-01-01")
    next_year = pd.to_datetime((idx.year + 1).astype(str) + "-01-01")
    frac = (idx - year_start) / (next_year - year_start)
    return idx.year.to_numpy() + frac.to_numpy()


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def fuse_velocity_field(
    vx: xr.DataArray,
    vy: xr.DataArray,
    target_times,
    n_modes: int = 5,
    coverage_frac: float = 0.5,
    direction_locked: bool = False,
    obs_weight: np.ndarray | None = None,
    seasonal: bool = True,
    var_target: float = 0.999,
    temporal_model: str = "gp",
    spatial_smooth_m: float = 0.0,
    verbose: bool = True,
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    r"""Fuse a gappy, multi-cadence velocity stack into a continuous field.

    Parameters
    ----------
    vx, vy : xr.DataArray, dims (time, y, x)
        Observed velocity epochs on a common grid (m/yr). NaN = no observation.
        Different sources (e.g. quarterly + annual mosaics) must already be
        regridded onto this shared grid and concatenated on ``time``.
    target_times : array-like of datetime64
        Output time axis (e.g. the DEM-stack epochs). May extend between — but
        for physical validity should not extend far beyond — the observed span.
    n_modes : int
        Maximum EOF modes to retain (further capped by ``var_target`` and
        ``n_epoch - 1``).
    coverage_frac : float
        A pixel joins the EOF domain if finite in at least this fraction of
        input epochs.
    direction_locked : bool
        If True, EOF the speed projected on the time-mean flow direction and
        hold direction fixed (a stationary-direction prior). Default False
        (vector EOF) — preferred when real obs span the target window, so any
        real change in flow direction is retained rather than suppressed.
    obs_weight : ndarray, shape (n_epoch,), optional
        Relative reliability per input epoch (larger = more trusted). Mapped to
        per-observation GP noise ``alpha ∝ 1/weight``. ``None`` = equal weight.
    seasonal : bool
        Include an annual periodic GP term.
    var_target : float
        Stop adding modes once cumulative variance explained reaches this.

    Returns
    -------
    (vx_fused, vy_fused, diagnostics)
        DataArrays on ``(target_time, y, x)`` and a dict of EOF/GP diagnostics
        (variance explained, retained modes, per-mode GP kernels, PC series,
        prediction std collapsed to a scalar uncertainty proxy).
    """
    if vx.dims != ("time", "y", "x") or vy.dims != ("time", "y", "x"):
        raise ValueError(f"vx/vy must be (time, y, x); got {vx.dims} / {vy.dims}")

    vx_v = np.asarray(vx.values, dtype=np.float64)
    vy_v = np.asarray(vy.values, dtype=np.float64)
    ny, nx = vx_v.shape[1:]
    n_epoch = vx_v.shape[0]
    t_obs = _decimal_year(vx["time"].values)
    t_pred = _decimal_year(np.asarray(target_times))
    _log(
        f"{n_epoch} obs epochs {t_obs.min():.2f}..{t_obs.max():.2f} -> "
        f"{len(t_pred)} target epochs {t_pred.min():.2f}..{t_pred.max():.2f}",
        verbose,
    )
    if t_pred.max() > t_obs.max() + 0.5 or t_pred.min() < t_obs.min() - 0.5:
        _log(
            "WARNING: target span exceeds observed span by >0.5 yr — fusion is "
            "extrapolating; results outside the observed window are low-confidence.",
            verbose,
        )

    # Domain from joint finite coverage (vx and vy).
    finite_stack = np.isfinite(vx_v) & np.isfinite(vy_v)
    domain = build_domain_mask(finite_stack, coverage_frac)
    dj, di = np.where(domain)
    n_pix = dj.size
    if n_pix == 0:
        raise ValueError("empty velocity domain — lower coverage_frac")
    _log(
        f"domain {n_pix} px (coverage>={coverage_frac}); "
        f"mean gap frac {1 - finite_stack[:, dj, di].mean():.3f}",
        verbose,
    )

    # 1. Harmonic infill each epoch over the domain.
    vx_fill = np.empty((n_epoch, n_pix))
    vy_fill = np.empty((n_epoch, n_pix))
    for e in range(n_epoch):
        fx = harmonic_infill_2d(vx_v[e], domain, verbose=False)
        fy = harmonic_infill_2d(vy_v[e], domain, verbose=False)
        vx_fill[e] = fx[dj, di]
        vy_fill[e] = fy[dj, di]
    _log("harmonic infill complete", verbose)

    # 2/3. EOF + GP — vector or direction-locked.
    if direction_locked:
        vmean_x = vx_fill.mean(axis=0)
        vmean_y = vy_fill.mean(axis=0)
        mag = np.hypot(vmean_x, vmean_y)
        mag[mag == 0] = 1.0
        dir_x, dir_y = vmean_x / mag, vmean_y / mag
        speed = vx_fill * dir_x[None, :] + vy_fill * dir_y[None, :]
        eof = velocity_eof(speed, n_modes)
    else:
        matrix = np.concatenate([vx_fill, vy_fill], axis=1)  # (n_epoch, 2*n_pix)
        eof = velocity_eof(matrix, n_modes)

    cumvar = np.cumsum(eof["var_explained"])
    k_keep = int(np.searchsorted(cumvar, var_target) + 1)
    k_keep = max(1, min(k_keep, eof["k"]))
    _log(
        f"EOF: retaining {k_keep}/{eof['k']} modes "
        f"(var {cumvar[k_keep - 1]:.4f}); per-mode {np.round(eof['var_explained'][:k_keep], 3)}",
        verbose,
    )

    if obs_weight is None:
        obs_alpha: np.ndarray | float = 1e-3
    else:
        w = np.asarray(obs_weight, dtype=float).ravel()
        obs_alpha = 1e-3 / np.clip(w / np.median(w), 1e-2, 1e2)

    coeffs_pred = np.zeros((len(t_pred), k_keep))
    std_pred = np.zeros((len(t_pred), k_keep))
    gp_info = []
    if temporal_model not in ("gp", "kalman"):
        raise ValueError(f"temporal_model must be 'gp' or 'kalman'; got {temporal_model!r}")
    for m in range(k_keep):
        if temporal_model == "kalman":
            mean_m, std_m, model = kalman_fuse_series(
                t_obs, eof["coeffs"][:, m], t_pred, obs_alpha=obs_alpha, seasonal=seasonal
            )
            desc = model
        else:
            mean_m, std_m, model = gp_fuse_series(
                t_obs, eof["coeffs"][:, m], t_pred, obs_alpha=obs_alpha, seasonal=seasonal
            )
            desc = str(model.kernel_)
        coeffs_pred[:, m] = mean_m
        std_pred[:, m] = std_m
        gp_info.append(desc)
        _log(f"  mode {m}: {temporal_model} {desc}", verbose)

    # 4. Reconstruct.
    modes = eof["modes"][:k_keep]  # (k, n_feat)
    recon = eof["mean"][None, :] + coeffs_pred @ modes  # (n_pred, n_feat)

    vx_out = np.full((len(t_pred), ny, nx), np.nan)
    vy_out = np.full((len(t_pred), ny, nx), np.nan)
    if direction_locked:
        for ti in range(len(t_pred)):
            s = recon[ti]
            vx_out[ti][domain] = s * dir_x
            vy_out[ti][domain] = s * dir_y
    else:
        for ti in range(len(t_pred)):
            vx_out[ti][domain] = recon[ti, :n_pix]
            vy_out[ti][domain] = recon[ti, n_pix:]

    # Optional modest spatial Gaussian on the fused velocity (NOT the melt grid):
    # smooths strain rates for a less-noisy div(u) while leaving the 250 m product
    # resolution intact. sigma is in METERS, converted via the grid spacing.
    sigma_pix = 0.0
    if spatial_smooth_m and spatial_smooth_m > 0:
        xv = np.asarray(vx["x"].values, dtype=float)
        yv = np.asarray(vx["y"].values, dtype=float)
        dx = abs(float(xv[1] - xv[0])) if xv.size > 1 else 1.0
        dy = abs(float(yv[1] - yv[0])) if yv.size > 1 else dx
        sigma_pix = float(spatial_smooth_m) / float(0.5 * (dx + dy))
        for ti in range(len(t_pred)):
            vx_out[ti] = _nan_aware_gaussian(vx_out[ti], domain, sigma_pix)
            vy_out[ti] = _nan_aware_gaussian(vy_out[ti], domain, sigma_pix)
        _log(
            f"spatial Gaussian sigma={spatial_smooth_m:.0f} m "
            f"({sigma_pix:.2f} px; FWHM~{2.355 * spatial_smooth_m:.0f} m)",
            verbose,
        )

    coords = {"time": pd.DatetimeIndex(pd.to_datetime(np.asarray(target_times))),
              "y": vx["y"].values, "x": vx["x"].values}
    vx_da = xr.DataArray(vx_out, dims=("time", "y", "x"), coords=coords, name="vx")
    vy_da = xr.DataArray(vy_out, dims=("time", "y", "x"), coords=coords, name="vy")
    vx_da.attrs.update(vx.attrs)
    vy_da.attrs.update(vy.attrs)

    diagnostics = {
        "n_obs_epochs": n_epoch,
        "n_domain_px": n_pix,
        "domain_mask": domain,
        "var_explained": eof["var_explained"],
        "k_retained": k_keep,
        "cum_var": float(cumvar[k_keep - 1]),
        "pc_obs": eof["coeffs"][:, :k_keep],
        "pc_pred": coeffs_pred,
        "pc_pred_std": std_pred,
        "t_obs": t_obs,
        "t_pred": t_pred,
        "gp_kernels": gp_info,
        "direction_locked": direction_locked,
        "temporal_model": temporal_model,
        "spatial_smooth_m": float(spatial_smooth_m),
        "spatial_smooth_sigma_px": sigma_pix,
    }
    return vx_da, vy_da, diagnostics
