# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Nuth & Kääb (2011) sub-pixel translation, point-cloud against raster.

**Why this exists.** ASP ``pc_align`` (6-DOF point-to-plane ICP) is our primary
coregistration, and it stays that way -- it solves rotation, which nothing here
does. But ICP's weakness is exactly sub-pixel *horizontal* accuracy, and the
community convention (xDEM's documented recommendation) is to **finish** an ICP
or deramp pipeline with Nuth & Kääb rather than to use either alone. Our chain
had no such refinement step; this is it. It does not replace ``pc_align``.

**The method.** A DEM displaced horizontally by :math:`\mathbf{s}=(s_x,s_y)`
produces an elevation difference against a truth surface that is, to first
order, the directional derivative along the displacement:

.. math::
    dh(\mathbf{x}) = z_{\rm dem}(\mathbf{x}) - z_{\rm ref}(\mathbf{x})
                   \approx -\nabla z\cdot\mathbf{s} + \Delta z .

Normalising by the slope turns that into the cosine relation Nuth & Kääb fit,

.. math::
    \frac{dh}{\tan\alpha} = a\,\cos(b-\psi) + c ,

with :math:`\alpha` slope, :math:`\psi` aspect (clockwise from north, pointing
downhill), :math:`a=\lVert\mathbf{s}\rVert`, :math:`b` the shift direction and
:math:`c=\Delta z/\overline{\tan\alpha}`. Expanding the cosine makes the fit
**linear** in :math:`(s_x, s_y, c)` -- no nonlinear optimiser, no starting
guess -- since :math:`a\cos(b-\psi) = s_x\sin\psi + s_y\cos\psi` once
:math:`\psi` is measured clockwise from north.

Two ``method`` options, same first-order model:

``"nuth_kaab"``
    The published form above: divide by :math:`\tan\alpha`, fit
    :math:`[\sin\psi, \cos\psi, 1]`, recover :math:`\Delta z = c\,
    \overline{\tan\alpha}`. Use it when comparability with xDEM /
    the literature is the point. Needs the ``min_slope_deg`` guard because
    :math:`dh/\tan\alpha` blows up as the terrain flattens.
``"gradient"`` (default)
    Fit :math:`dh = -g_x s_x - g_y s_y + \Delta z` directly on
    :math:`[-g_x, -g_y, 1]`. Algebraically the same model, but it never
    divides by the slope, so near-flat cells simply carry little weight
    instead of exploding, and :math:`\Delta z` is a free parameter rather than
    the ``c \overline{\tan\alpha}`` approximation. Better conditioned on ice.

**Sign convention**, the classic source of bugs here: the returned
``(dx, dy, dz)`` is the DEM's estimated *position error* -- the raster's
content sits at ``x - dx``. Correct it by **sampling the DEM at**
``(x + dx, y + dy)`` and **adding** ``-dz`` to the elevations, which is what
:func:`apply_shift_to_points` and the iteration below do. Round-tripped in
``tests/gate_nuth_kaab.py``.

**MEASURED ON OUR DATA (2026-09-02) -- do not wire this into the production
align path without re-measuring. The NMAD pairs below were measured while
``nmad_after`` was still taken over every finite control point rather than
over the slope-gated set ``nmad_before`` uses (fixed 2026-09-03); they are
kept as the figures the decision was actually made on, and the like-for-like
pair has not been re-measured on the strips.** Run as xDEM recommends, i.e. as
a finish after ``pc_align``, on the 173 uncorrupted processing-twin strips
against the same control ``pc_align`` was fed, it makes the residual WORSE:
NMAD 0.756 -> 0.774 m, residual-offset rms 0.181 -> 0.285 m, residual tilt
essentially unchanged (it models translation, not tilt). It reports shifts of
median 5.1 m (p90 17 m) -- larger than the 2.8 m a-priori shift ``pc_align``
had already removed -- and 109/160 of them exceed twice their own standard
error, so they look "significant" while degrading the fit: the classic
signature of a mis-specified model. Only part of that is residual tilt
aliasing into an apparent shift (corr(dy, residual ay) = -0.23; in x it is
~0). Removing the per-strip plane FIRST cuts the spurious shift by a third
(6.3 -> 4.0 m) and flips the method to helping slightly (NMAD 0.558 ->
0.493 m), so **order matters: a translation refinement is only meaningful
after the ramp is gone**, and in this chain the ramp comes out later and
stack-wide in :func:`~stereo_melt.coregister.tilt.fit_tilt_stack`, not per
strip. The deeper limit is control geometry: our control lives on the static
apron, whose median 90th-percentile slope is 5 deg, giving a per-strip
horizontal standard error of ~1.3 m. Kept as a validated diagnostic; the
production path is unchanged.

Flat surfaces carry no horizontal information at all -- on a floating shelf the
horizontal shift is unidentifiable, and this returns it as such (large
``se_dx``/``se_dy``, small ``slope_p90``) rather than inventing a number. If the
slope gate leaves too few usable points to fit at all, the shift comes back as
**NaN** rather than as an exact ``(0, 0, 0)`` that a caller could not tell from
a converged "no shift needed". Run it against the static apron control
(rock + slow ice), which is where the relief is.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "terrain_slope_aspect",
    "sample_bilinear",
    "apply_shift_to_points",
    "nuth_kaab_point_raster",
]


def terrain_slope_aspect(dem: np.ndarray, dx: float, dy: float):
    r"""``(slope_rad, aspect_rad, gx, gy)`` for a north-up DEM array.

    ``dem`` is indexed ``[row, col]`` with rows running **north to south**
    (the workspace EPSG:3031 convention, y descending) and columns west to
    east. ``dx``/``dy`` are the positive pixel sizes in metres.

    ``gx``/``gy`` are :math:`\partial z/\partial\mathrm{east}` and
    :math:`\partial z/\partial\mathrm{north}` -- note the row axis is negated
    so that ``gy`` is a true northward derivative. ``aspect`` is measured
    clockwise from north and points **downhill**, matching the GDAL/xDEM
    convention.
    """
    dz_drow, dz_dcol = np.gradient(dem, dy, dx)
    gx = dz_dcol
    gy = -dz_drow                      # rows increase southward
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, -gy) % (2.0 * np.pi)   # downhill, cw from north
    return slope, aspect, gx, gy


def _grid_sampler(values: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Bilinear interpolator over a north-up grid (y descending)."""
    from scipy.interpolate import RegularGridInterpolator

    y_asc = y[::-1]
    v_asc = values[::-1, :]
    return RegularGridInterpolator((y_asc, x), v_asc, method="linear",
                                   bounds_error=False, fill_value=np.nan)


def sample_bilinear(values: np.ndarray, x: np.ndarray, y: np.ndarray,
                    east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Bilinearly sample a north-up grid at scattered points."""
    itp = _grid_sampler(values, x, y)
    return itp(np.column_stack([np.asarray(north, float), np.asarray(east, float)]))


def apply_shift_to_points(east: np.ndarray, north: np.ndarray,
                          dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Move sample locations to undo a DEM position error ``(dx, dy)``."""
    return np.asarray(east, float) + dx, np.asarray(north, float) + dy


def _robust_lstsq(A: np.ndarray, b: np.ndarray, *, max_iter: int = 5,
                  c_tukey: float = 4.685):
    """Tukey-biweight IRLS; returns ``(coef, se, scale, weights)``."""
    w = np.ones_like(b)
    coef = np.zeros(A.shape[1])
    scale = np.nan
    for _ in range(max_iter):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        res = b - A @ coef
        mad = float(np.median(np.abs(res - np.median(res)))) + 1e-12
        scale = 1.4826 * mad
        u = res / (c_tukey * scale)
        w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if w.sum() < A.shape[1] + 1:
            w = np.ones_like(b)
            break
    # Standard errors via the SVD of A^T W A rather than a plain inverse: on a
    # flat surface the horizontal columns are (near-)null, and the honest
    # answer for those parameters is INFINITE uncertainty -- an unidentifiable
    # shift, thresholdable by the caller. A raw inv() raises there and a NaN
    # cannot be told apart from a crash.
    AtWA = (A * w[:, None]).T @ A
    sv, V = np.linalg.eigh(AtWA)                     # symmetric PSD
    tol = float(np.max(np.abs(sv))) * max(AtWA.shape) * np.finfo(float).eps
    good = sv > tol
    # Sum only the well-determined directions (a plain (V**2) @ inv_sv would
    # hit 0*inf = NaN wherever a parameter has no support in a null direction).
    var = (V[:, good] ** 2) @ (1.0 / sv[good]) if good.any() else np.zeros(A.shape[1])
    if not good.all():                               # exactly-null direction
        var = np.where((V[:, ~good] ** 2).sum(axis=1) > 1e-12, np.inf, var)
    se = np.sqrt(scale ** 2 * var)
    return coef, se, scale, w


def nuth_kaab_point_raster(
    dem: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    z_ref: np.ndarray,
    *,
    method: str = "gradient",
    max_iter: int = 10,
    tol_m: float = 0.01,
    min_slope_deg: float = 3.0,
    max_slope_deg: float = 60.0,
    robust: bool = True,
    min_points: int = 30,
    verbose: bool = False,
) -> dict:
    r"""Estimate a DEM's ``(dx, dy, dz)`` position error against control points.

    Parameters
    ----------
    dem, x, y
        North-up raster and its coordinate axes (``y`` descending, metres).
    east, north, z_ref
        Control points and their elevations, on the DEM's datum.
    method
        ``"gradient"`` (default) or ``"nuth_kaab"`` -- see the module
        docstring. ``min_slope_deg`` gates the fit cells in both, but only
        ``"nuth_kaab"`` actually needs it.
    max_iter, tol_m
        Stop when an iteration's incremental horizontal shift is under
        ``tol_m``, or after ``max_iter`` iterations.

    Returns
    -------
    dict
        ``dx``, ``dy``, ``dz`` (metres, DEM position error -- see the module
        docstring for the sign), ``se_dx``/``se_dy``/``se_dz``, ``n_iter``,
        ``converged``, ``n_points`` used, ``nmad_before``/``nmad_after`` of the
        residual over the SAME control points (the slope-gated set selected on
        the first iteration, so the pair is like-for-like -- an ungated
        ``nmad_after`` would mix in the quieter sub-gate points and overstate
        the improvement), ``slope_p90`` (degrees; low means the horizontal solve is
        weakly constrained) and the per-iteration ``history``. ``dx``/``dy``/
        ``dz`` stay NaN if no iteration ever had ``min_points`` usable points;
        ``n_points`` then reports how many the slope gate did leave, which is
        not necessarily zero.
    """
    if method not in ("gradient", "nuth_kaab"):
        raise ValueError(f'method must be "gradient" or "nuth_kaab", got {method!r}')
    east = np.asarray(east, float)
    north = np.asarray(north, float)
    z_ref = np.asarray(z_ref, float)

    slope, aspect, gx, gy = terrain_slope_aspect(dem, abs(float(x[1] - x[0])),
                                                 abs(float(y[1] - y[0])))
    s_dem = _grid_sampler(dem, x, y)
    s_slope = _grid_sampler(slope, x, y)
    s_aspect_sin = _grid_sampler(np.sin(aspect), x, y)
    s_aspect_cos = _grid_sampler(np.cos(aspect), x, y)
    s_gx = _grid_sampler(gx, x, y)
    s_gy = _grid_sampler(gy, x, y)

    def _at(pe, pn, itp):
        return itp(np.column_stack([pn, pe]))

    out = dict(dx=np.nan, dy=np.nan, dz=np.nan, se_dx=np.nan, se_dy=np.nan,
               se_dz=np.nan, n_iter=0, converged=False, n_points=0,
               nmad_before=np.nan, nmad_after=np.nan, slope_p90=np.nan,
               history=[], method=method)

    tot = np.zeros(3)          # cumulative (dx, dy, dz)
    ok0 = None                 # the gated point set nmad_before was measured on
    lo, hi = np.radians(min_slope_deg), np.radians(max_slope_deg)
    for it in range(max_iter):
        pe, pn = apply_shift_to_points(east, north, tot[0], tot[1])
        zd = _at(pe, pn, s_dem)
        dh = zd - z_ref - tot[2]              # remove the vertical bias applied so far
        sl = _at(pe, pn, s_slope)
        ok = np.isfinite(dh) & np.isfinite(sl) & (sl >= lo) & (sl <= hi)
        if ok.sum() < min_points:
            if out["n_iter"] == 0:
                out["n_points"] = int(ok.sum())
            out["history"].append(dict(iter=it, n=int(ok.sum()), note="too few usable points"))
            break
        if it == 0:
            ok0 = ok
            out["nmad_before"] = float(1.4826 * np.median(np.abs(dh[ok] - np.median(dh[ok]))))
            out["slope_p90"] = float(np.degrees(np.percentile(sl[ok], 90)))

        if method == "nuth_kaab":
            sa = _at(pe, pn, s_aspect_sin)[ok]
            ca = _at(pe, pn, s_aspect_cos)[ok]
            tan_a = np.tan(sl[ok])
            A = np.column_stack([sa, ca, np.ones_like(sa)])
            b = dh[ok] / tan_a
            coef, se, scale, _ = (_robust_lstsq(A, b) if robust
                                  else (np.linalg.lstsq(A, b, rcond=None)[0],
                                        np.full(3, np.nan), np.nan, None))
            step = np.array([coef[0], coef[1], coef[2] * float(np.mean(tan_a))])
            se_step = np.array([se[0], se[1], se[2] * float(np.mean(tan_a))])
        else:
            A = np.column_stack([-_at(pe, pn, s_gx)[ok], -_at(pe, pn, s_gy)[ok],
                                 np.ones(int(ok.sum()))])
            b = dh[ok]
            coef, se, scale, _ = (_robust_lstsq(A, b) if robust
                                  else (np.linalg.lstsq(A, b, rcond=None)[0],
                                        np.full(3, np.nan), np.nan, None))
            step, se_step = coef.copy(), se.copy()

        tot = tot + step
        out.update(n_iter=it + 1, n_points=int(ok.sum()),
                   se_dx=float(se_step[0]), se_dy=float(se_step[1]),
                   se_dz=float(se_step[2]))
        out["history"].append(dict(iter=it, n=int(ok.sum()),
                                   step=[float(v) for v in step],
                                   total=[float(v) for v in tot],
                                   resid_scale=float(scale) if scale == scale else np.nan))
        if verbose:
            print(f"    NK[{method}] iter {it}: step ({step[0]:+.3f}, {step[1]:+.3f}, "
                  f"{step[2]:+.3f}) m  total ({tot[0]:+.3f}, {tot[1]:+.3f}, {tot[2]:+.3f})  "
                  f"n={int(ok.sum())}", flush=True)
        if float(np.hypot(step[0], step[1])) < tol_m:
            out["converged"] = True
            break

    # Only claim a shift if an iteration actually ran: bailing on the first
    # pass (a strip whose control is all below min_slope_deg) would otherwise
    # report an exact (0, 0, 0), indistinguishable from a converged "no shift
    # needed" and folded into any per-strip aggregate as a real measurement.
    if out["n_iter"] == 0:
        return out
    out.update(dx=float(tot[0]), dy=float(tot[1]), dz=float(tot[2]))
    pe, pn = apply_shift_to_points(east, north, tot[0], tot[1])
    dh_f = _at(pe, pn, s_dem) - z_ref - tot[2]
    # Same points as nmad_before, not every finite one: dh scatter grows with
    # slope, so scoring "after" on the ungated set measures a quieter sample
    # and reports an improvement the shift did not make.
    f = ok0 & np.isfinite(dh_f)
    if f.any():
        out["nmad_after"] = float(1.4826 * np.median(np.abs(dh_f[f] - np.median(dh_f[f]))))
    return out
