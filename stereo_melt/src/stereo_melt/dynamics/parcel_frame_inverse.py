# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Windowed parcel-frame linear inverse (Level 1 of ``plan_lagrangian.md``).

The linear-inverse reimplementation that treats **Lagrangian motion and parcel
strain properly** (user directive 2026-07-02; audit + design in
``literature/plan_lagrangian.md`` "Implementation status + v1"):

- **Motion**: observations are advected into short-window co-moving frames
  using the TIME-VARYING velocity, one frame per anchor epoch. Short windows
  (``window_yr``) bound the accumulated frame deformation to the ~10-30 %
  regime where the ξ-grid FFT remains meaningful, instead of the whole-record
  single-anchor frame whose deformation reaches 40-150 % on a 14-yr PIG
  window.
- **Strain**: the parcel's own along-path divergence enters POINTWISE.
  ``γ(ξ) = ⟨∇·u⟩_path(ξ) · t_r`` is a field, not a domain scalar; the kernel
  is expanded to first order in both ``ΔH(ξ)`` and ``Δγ(ξ)`` around window
  means (finite-difference kernels, same construction as
  :class:`~stereo_melt.dynamics.pseudospectral.LinearizedHForwardOp` but with
  the extra γ term).
- **Coverage structure**: each window is inverted independently with the
  honest NaN ``W``-mask CG (:func:`cg_invert_stationary`), then windows are
  median-combined per Eulerian pixel (each window's anchor grid *is* the
  Eulerian grid at its anchor epoch). A window only needs internal coverage,
  and ~10 windows vote — the structural mitigation for the single-anchor
  gap-collapse mode diagnosed 2026-06-13.

The product remains a zero-mean **anomaly / correction layer** (the k=0 mode
is not constrained by the kernel): pair it with the mass-budget Lagrangian
melt for absolute rates. Shean sign convention on output (negative = melt).
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd
import xarray as xr

from ..backend import asarray, map_coordinates, to_numpy, xp
from ..constants import rhoi, rhow
from ..freeboard import freeboard_to_thickness
from ..kinematics import SECONDS_PER_YEAR, divergence, flux_divergence, gaussian_smooth_nan
from .pseudospectral import PerturbationForwardOp, cg_invert_stationary

__all__ = [
    "HGammaWindowOp",
    "lagrangian_frame_window",
    "windowed_parcel_frame_inverse",
]

G_GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Window frame transform with time-varying velocity + along-path backgrounds
# ---------------------------------------------------------------------------


def _prep_velocity(vx, vy, smooth_m, vdiv_clip):
    """Smooth (NaN-aware) velocity per time slice; return arrays + divergence.

    Returns ``(vx_tv, vy_tv, vdiv_tv, v_t_years_or_None)`` where the arrays are
    backend arrays with a leading time axis (length 1 for static fields) and
    ``v_t_years`` is the velocity time axis in years since ``t0_ref`` (None for
    static). ``t0_ref`` is supplied by the caller via the ``times`` argument.
    """

    def smo(v2d):
        if smooth_m and smooth_m > 0:
            res = abs(float(v2d["x"].values[1] - v2d["x"].values[0]))
            return gaussian_smooth_nan(v2d, float(smooth_m) / res)
        return v2d

    if "time" in vx.dims and vx.sizes["time"] > 1:
        vxs = vx.sortby("time")
        vys = vy.sortby("time")
        slices_x = [smo(vxs.isel(time=i)) for i in range(vxs.sizes["time"])]
        slices_y = [smo(vys.isel(time=i)) for i in range(vys.sizes["time"])]
        vdivs = [divergence(sx, sy) for sx, sy in zip(slices_x, slices_y)]
        vx_tv = asarray(np.stack([s.values for s in slices_x]).astype(np.float64))
        vy_tv = asarray(np.stack([s.values for s in slices_y]).astype(np.float64))
        vd_tv = asarray(np.stack([s.values for s in vdivs]).astype(np.float64))
        v_times = pd.to_datetime(vxs["time"].values)
    else:
        vx2 = vx.isel(time=0) if "time" in vx.dims else vx
        vy2 = vy.isel(time=0) if "time" in vy.dims else vy
        vx2 = smo(vx2)
        vy2 = smo(vy2)
        vd = divergence(vx2, vy2)
        vx_tv = asarray(vx2.values.astype(np.float64))[None]
        vy_tv = asarray(vy2.values.astype(np.float64))[None]
        vd_tv = asarray(vd.values.astype(np.float64))[None]
        v_times = None
    if vdiv_clip is not None and vdiv_clip > 0:
        vd_tv = xp.clip(vd_tv, -float(vdiv_clip), float(vdiv_clip))
    return vx_tv, vy_tv, vd_tv, v_times


def _bracket_weights(v_t_years, t_abs):
    """(k0, k1, w) for linear time interpolation with constant end-extrapolation."""
    nv = v_t_years.size
    if t_abs <= v_t_years[0]:
        return 0, 0, 0.0
    if t_abs >= v_t_years[-1]:
        return nv - 1, nv - 1, 0.0
    k1 = int(np.searchsorted(v_t_years, t_abs))
    k0 = k1 - 1
    w = float((t_abs - v_t_years[k0]) / (v_t_years[k1] - v_t_years[k0]))
    return k0, k1, w


def lagrangian_frame_window(
    h_stack: xr.DataArray,
    epoch_idx: np.ndarray,
    vx_tv,
    vy_tv,
    vdiv_tv,
    v_t_years: np.ndarray | None,
    H_field: np.ndarray,
    dt_yr: float = 0.05,
):
    r"""Advect one window of epochs into the anchor's co-moving frame.

    Parcels are seeded at every pixel at the window's FIRST epoch and marched
    forward through the (optionally time-varying) velocity. Each epoch in the
    window samples the surface at the parcel positions. Along the march the
    parcel accumulates its own ``∇·u`` and background-``H`` samples, whose
    path means become the pointwise strain / thickness fields of the window's
    kernel expansion.

    Parameters
    ----------
    h_stack : (time, y, x) DataArray — full stack (window selected by index).
    epoch_idx : int array — indices of the window's epochs, ascending; the
        first entry is the anchor.
    vx_tv, vy_tv, vdiv_tv : backend arrays (n_v, ny, nx) from
        :func:`_prep_velocity` (n_v == 1 for a static field).
    v_t_years : velocity time axis in years since the STACK's first epoch,
        or None for static.
    H_field : (ny, nx) float array — background total thickness, m.
    dt_yr : march sub-step, years.

    Returns
    -------
    h_lag : (n_e, ny, nx) float64 — window stack in the anchor frame.
    divu_path : (ny, nx) — along-path mean ∇·u (1/yr) per parcel.
    H_path : (ny, nx) — along-path mean background H (m) per parcel.
    """
    times = pd.to_datetime(h_stack["time"].values)
    t_years_all = np.array(
        [(t - times[0]).total_seconds() / SECONDS_PER_YEAR for t in times], dtype=np.float64
    )
    e_idx = np.asarray(epoch_idx, dtype=np.int64)
    t_win = t_years_all[e_idx]
    t_anchor = float(t_win[0])
    n_e = int(e_idx.size)
    ny = h_stack.sizes["y"]
    nx = h_stack.sizes["x"]

    n_steps = max(1, int(np.ceil((t_win[-1] - t_anchor) / dt_yr)))
    e_step = np.clip(np.round((t_win - t_anchor) / dt_yr).astype(np.int64), 0, n_steps)

    H_b = asarray(np.asarray(H_field, dtype=np.float64))
    static_vel = v_t_years is None

    ji, ii = xp.mgrid[0:ny, 0:nx]
    y_idx = ji.astype(xp.float64).ravel()
    x_idx = ii.astype(xp.float64).ravel()
    valid = xp.ones(y_idx.size, dtype=bool)

    div_sum = xp.zeros(y_idx.size, dtype=xp.float64)
    H_sum = xp.zeros(y_idx.size, dtype=xp.float64)
    n_samp = 0

    h_lag = np.full((n_e, ny, nx), np.nan, dtype=np.float64)

    def _snapshot(e_pos: int) -> None:
        h_e = map_coordinates(
            asarray(np.asarray(h_stack.values[e_idx[e_pos]], dtype=np.float64)),
            [y_idx, x_idx], order=1, mode="constant", cval=float("nan"),
        )
        h_e = xp.where(valid, h_e, xp.nan)
        h_lag[e_pos] = to_numpy(h_e).reshape(ny, nx)

    next_e = 0
    for k in range(n_steps + 1):
        while next_e < n_e and e_step[next_e] == k:
            _snapshot(next_e)
            next_e += 1
        if k == n_steps:
            break
        t_abs = t_anchor + k * dt_yr
        if static_vel:
            vx_t = map_coordinates(vx_tv[0], [y_idx, x_idx], order=1, mode="nearest")
            vy_t = map_coordinates(vy_tv[0], [y_idx, x_idx], order=1, mode="nearest")
            dv_t = map_coordinates(vdiv_tv[0], [y_idx, x_idx], order=1, mode="nearest")
        else:
            k0, k1, w = _bracket_weights(v_t_years, t_abs)
            if k0 == k1:
                vx_t = map_coordinates(vx_tv[k0], [y_idx, x_idx], order=1, mode="nearest")
                vy_t = map_coordinates(vy_tv[k0], [y_idx, x_idx], order=1, mode="nearest")
                dv_t = map_coordinates(vdiv_tv[k0], [y_idx, x_idx], order=1, mode="nearest")
            else:
                vx_t = (1 - w) * map_coordinates(
                    vx_tv[k0], [y_idx, x_idx], order=1, mode="nearest"
                ) + w * map_coordinates(vx_tv[k1], [y_idx, x_idx], order=1, mode="nearest")
                vy_t = (1 - w) * map_coordinates(
                    vy_tv[k0], [y_idx, x_idx], order=1, mode="nearest"
                ) + w * map_coordinates(vy_tv[k1], [y_idx, x_idx], order=1, mode="nearest")
                dv_t = (1 - w) * map_coordinates(
                    vdiv_tv[k0], [y_idx, x_idx], order=1, mode="nearest"
                ) + w * map_coordinates(vdiv_tv[k1], [y_idx, x_idx], order=1, mode="nearest")

        finite_v = xp.isfinite(vx_t) & xp.isfinite(vy_t)
        vx_t = xp.where(finite_v, vx_t, 0.0)
        vy_t = xp.where(finite_v, vy_t, 0.0)
        div_sum = div_sum + xp.where(xp.isfinite(dv_t), dv_t, 0.0)
        H_t = map_coordinates(H_b, [y_idx, x_idx], order=1, mode="nearest")
        H_sum = H_sum + xp.where(xp.isfinite(H_t), H_t, 0.0)
        n_samp += 1

        res_x = float(h_stack["x"].values[1] - h_stack["x"].values[0])
        res_y = float(h_stack["y"].values[0] - h_stack["y"].values[1])
        x_idx = x_idx + vx_t * dt_yr / res_x
        y_idx = y_idx - vy_t * dt_yr / res_y
        in_bounds = (x_idx >= 0) & (x_idx <= nx - 1) & (y_idx >= 0) & (y_idx <= ny - 1)
        valid = valid & in_bounds & finite_v

    divu_path = to_numpy(div_sum).reshape(ny, nx) / max(n_samp, 1)
    H_path = to_numpy(H_sum).reshape(ny, nx) / max(n_samp, 1)
    return h_lag, divu_path, H_path


# ---------------------------------------------------------------------------
# Kernel expanded to first order in ΔH(ξ) and Δγ(ξ)
# ---------------------------------------------------------------------------


class HGammaWindowOp:
    r"""Stationary forward/adjoint with pointwise ``ΔH`` and ``Δγ`` corrections.

    ``h(t_n) = IFFT[K0_n m̂] + ΔH·IFFT[KH_n m̂] + Δγ·IFFT[Kγ_n m̂]``

    where ``K0`` is the cumulative Stubblefield kernel at the window means
    ``(H_ref, γ_ref)`` and ``KH = ∂K/∂H``, ``Kγ = ∂K/∂γ`` are centered finite
    differences built from :class:`PerturbationForwardOp` instances. The
    cumulative kernels are assembled with an O(n_t) exponential recurrence
    (prefix sums of ``e^{λΔt}``, all exponents ≤ 0) instead of the O(n_t²)
    direct sum, so ~100-epoch windows stay cheap.

    Duck-types the :func:`cg_invert_stationary` operator interface
    (``n_t, ny, nx, kx, ky, forward_stationary, adjoint_stationary,
    _kernel_for_reference``).
    """

    def __init__(
        self,
        H_ref: float,
        gamma_ref: float,
        div_u_ref_si: float,
        dH_field: np.ndarray,
        dgamma_field: np.ndarray,
        times,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        eta_bar: float = 1e14,
        rho_i: float = rhoi,
        rho_w: float = rhow,
        g: float = G_GRAVITY,
        theta: float = 1e-14,
        eps_H: float = 0.05,
        eps_g: float = 0.01,
    ):
        self.H_ref = float(H_ref)
        self.eps_H = float(eps_H)
        self.eps_g = float(eps_g)
        self.dH = asarray(np.nan_to_num(np.asarray(dH_field, dtype=np.float64)))
        self.dg = asarray(np.nan_to_num(np.asarray(dgamma_field, dtype=np.float64)))

        def tr_of(Hv):
            return 2.0 * eta_bar / (rho_i * g * Hv)

        # γ co-varies with H through t_r: at the perturbed thickness the SAME
        # dimensional ⟨∇·u⟩ maps to a different dimensionless γ.
        ops = {}
        specs = {
            "0": (self.H_ref, gamma_ref),
            "H+": (self.H_ref * (1 + eps_H), div_u_ref_si * tr_of(self.H_ref * (1 + eps_H))),
            "H-": (self.H_ref * (1 - eps_H), div_u_ref_si * tr_of(self.H_ref * (1 - eps_H))),
            "g+": (self.H_ref, gamma_ref + eps_g),
            "g-": (self.H_ref, gamma_ref - eps_g),
        }
        for key, (Hv, gv) in specs.items():
            ops[key] = PerturbationForwardOp(
                H=Hv, times=times, nx=nx, ny=ny, dx=dx, dy=dy,
                eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g,
                alpha=0.0, gamma=gv, theta=theta,
            )
        self._ops = ops
        op0 = ops["0"]
        self.n_t = op0.n_t
        self.ny = int(ny)
        self.nx = int(nx)
        self.kx = op0.kx
        self.ky = op0.ky
        self.tr = op0.tr
        self.model = op0.model
        self.t_secs = op0.t_secs
        self.dt_weights = op0.dt_weights
        self._cache: dict = {}

    # -- O(n_t) cumulative kernel via the exponential prefix recurrence -----
    @staticmethod
    def _k_cum_fast(op: PerturbationForwardOp):
        pref = -op.model.delta * op.B / op.mu_safe
        w = op.dt_weights
        t = op.t_secs
        n_t = op.n_t
        shape = op.B.shape
        A = xp.zeros(shape, dtype=xp.complex128)
        Bacc = xp.zeros(shape, dtype=xp.complex128)
        K_cum = xp.zeros((n_t,) + shape, dtype=xp.complex128)
        for n in range(n_t):
            if n == 0:
                dt_nd = 0.0
            else:
                dt_nd = (t[n] - t[n - 1]) / op.tr
            if n > 0:
                A = A * xp.exp(op.lp * dt_nd)
                Bacc = Bacc * xp.exp(op.lm * dt_nd)
            # add this epoch's own (zero-lag) contribution
            A = A + w[n]
            Bacc = Bacc + w[n]
            K_cum[n] = pref * (A - Bacc)
        return K_cum

    def _kernels(self, reference: str):
        key = ("K", reference)
        if key in self._cache:
            return self._cache[key]
        Ks = {}
        for name, op in self._ops.items():
            Kc = self._k_cum_fast(op)
            if reference == "time_mean":
                Kc = Kc - Kc.mean(axis=0, keepdims=True)
            elif reference != "anchor_first":
                raise ValueError(f"reference must be 'anchor_first' or 'time_mean', got {reference!r}")
            Ks[name] = Kc
        K0 = Ks["0"]
        KH = (Ks["H+"] - Ks["H-"]) / (2.0 * self.eps_H * self.H_ref)
        Kg = (Ks["g+"] - Ks["g-"]) / (2.0 * self.eps_g)
        self._cache[key] = (K0, KH, Kg)
        return self._cache[key]

    def _kernel_for_reference(self, reference: str):
        # Used by cg_invert_stationary only for the Tikhonov scale.
        return self._kernels(reference)[0]

    def forward_stationary(self, m_2d: np.ndarray, reference: str = "time_mean") -> np.ndarray:
        m_2d = np.asarray(m_2d, dtype=np.float64)
        if m_2d.shape != (self.ny, self.nx):
            raise ValueError(f"m_2d shape {m_2d.shape} != ({self.ny}, {self.nx})")
        K0, KH, Kg = self._kernels(reference)
        m_hat = xp.fft.fft2(asarray(m_2d / SECONDS_PER_YEAR))
        h_out = np.empty((self.n_t, self.ny, self.nx), dtype=np.float64)
        for n in range(self.n_t):
            h0 = xp.fft.ifft2(K0[n] * m_hat).real
            hH = xp.fft.ifft2(KH[n] * m_hat).real
            hg = xp.fft.ifft2(Kg[n] * m_hat).real
            h_out[n] = to_numpy(h0 + self.dH * hH + self.dg * hg)
        return h_out

    def adjoint_stationary(self, h: np.ndarray, reference: str = "time_mean") -> np.ndarray:
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (self.n_t, self.ny, self.nx):
            raise ValueError(f"h shape {h.shape} != ({self.n_t}, {self.ny}, {self.nx})")
        K0, KH, Kg = self._kernels(reference)
        m_hat_adj = xp.zeros((self.ny, self.nx), dtype=xp.complex128)
        for n in range(self.n_t):
            h_n = asarray(h[n])
            m_hat_adj = m_hat_adj + xp.conj(K0[n]) * xp.fft.fft2(h_n)
            m_hat_adj = m_hat_adj + xp.conj(KH[n]) * xp.fft.fft2(self.dH * h_n)
            m_hat_adj = m_hat_adj + xp.conj(Kg[n]) * xp.fft.fft2(self.dg * h_n)
        return to_numpy(xp.fft.ifft2(m_hat_adj).real) / SECONDS_PER_YEAR


# ---------------------------------------------------------------------------
# Windowed driver
# ---------------------------------------------------------------------------


def windowed_parcel_frame_inverse(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    floating_mask: xr.DataArray | None = None,
    H_total_field: xr.DataArray | None = None,
    d: xr.DataArray | float = 0.0,
    window_yr: float = 3.0,
    stride_yr: float = 1.5,
    min_window_epochs: int = 8,
    eta_bar: float = 1e13,
    rho_i: float = rhoi,
    rho_w: float = rhow,
    g: float = G_GRAVITY,
    theta: float = 1e-14,
    tikhonov: float = 1e-2,
    length_scale_m: float = 500.0,
    max_iter: int = 60,
    cg_tol: float = 1e-6,
    dt_yr: float = 0.05,
    vel_smooth_m: float = 3000.0,
    vdiv_clip: float = 0.2,
    dH_clip_rel: float = 0.3,
    dgamma_clip: float = 0.05,
    localize_ring_frac: float = 0.8,
    verbose: bool = True,
) -> xr.Dataset:
    r"""Windowed parcel-frame linear inverse — the Level-1 reimplementation.

    See the module docstring for the design. Returns per-pixel across-window
    ``melt_rate`` (median, Shean sign), ``window_count``, ``melt_spread``
    (across-window NMAD — the quality gate), and diagnostics.
    """
    if "time" not in h_stack.dims or h_stack.sizes["time"] < min_window_epochs:
        raise ValueError("h_stack needs a time dim with enough epochs")

    times = pd.to_datetime(h_stack["time"].values)
    t_years = np.array(
        [(t - times[0]).total_seconds() / SECONDS_PER_YEAR for t in times], dtype=np.float64
    )
    T = len(times)
    ny = h_stack.sizes["y"]
    nx = h_stack.sizes["x"]
    dxm = float(abs(h_stack["x"].values[1] - h_stack["x"].values[0]))
    dym = float(abs(h_stack["y"].values[0] - h_stack["y"].values[1]))

    # Velocity prep: NaN-aware smoothing per slice, divergence, clip.
    vx_tv, vy_tv, vdiv_tv, v_times = _prep_velocity(vx, vy, vel_smooth_m, vdiv_clip)
    if v_times is not None:
        v_t_years = np.array(
            [(t - times[0]).total_seconds() / SECONDS_PER_YEAR for t in v_times],
            dtype=np.float64,
        )
    else:
        v_t_years = None

    # Background thickness field for the kernel expansion.
    H_f_stack = freeboard_to_thickness(h_stack, d=d, rho_w=rho_w, rho_i=rho_i)
    H_f_mean = H_f_stack.mean("time", skipna=True)
    if H_total_field is not None:
        H_bg = np.asarray(H_total_field.values, dtype=np.float64)
    else:
        H_bg = np.asarray(H_f_mean.values, dtype=np.float64)
    fmask = floating_mask.values.astype(bool) if floating_mask is not None else np.isfinite(H_bg)
    H_fill = float(np.nanmean(np.where(fmask, H_bg, np.nan)))
    H_bg = np.where(np.isfinite(H_bg), H_bg, H_fill)

    # Window anchors: greedy stride over the epoch axis.
    anchors = []
    next_t = t_years[0]
    for i in range(T):
        if t_years[i] + 1e-9 >= next_t:
            n_in = int(np.sum((t_years >= t_years[i]) & (t_years <= t_years[i] + window_yr)))
            if n_in >= min_window_epochs:
                anchors.append(i)
                next_t = t_years[i] + stride_yr
    if not anchors:
        raise ValueError("no window has enough epochs; loosen window_yr/min_window_epochs")

    if verbose:
        print(
            f"  parcel-frame inverse: {len(anchors)} windows "
            f"(window={window_yr} yr, stride={stride_yr} yr, "
            f"min_epochs={min_window_epochs}); grid {ny}x{nx} @ {dxm:.0f} m; "
            f"vel smooth={vel_smooth_m:.0f} m, vdiv clip={vdiv_clip}/yr",
            flush=True,
        )

    m_windows = np.full((len(anchors), ny, nx), np.nan, dtype=np.float64)
    window_meta = []
    t0 = _time.time()
    for wi, a in enumerate(anchors):
        in_win = np.where((t_years >= t_years[a]) & (t_years <= t_years[a] + window_yr))[0]
        h_lag, divu_path, H_path = lagrangian_frame_window(
            h_stack, in_win, vx_tv, vy_tv, vdiv_tv, v_t_years, H_bg, dt_yr=dt_yr
        )

        # Per-pixel time-mean anomaly in the window frame (reference="time_mean").
        with np.errstate(invalid="ignore"):
            h_anom = h_lag - np.nanmean(h_lag, axis=0, keepdims=True)

        # localize: subtract the per-epoch outer-ring mean (kills uniform
        # per-epoch offsets that alias into low-k melt).
        if floating_mask is not None and fmask.any():
            yy, xx2 = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
            y_c = float(yy[fmask].mean())
            x_c = float(xx2[fmask].mean())
            r = np.sqrt((yy - y_c) ** 2 + (xx2 - x_c) ** 2)
            ring = (r >= localize_ring_frac * float(r[fmask].max())) & fmask
            if ring.any():
                with np.errstate(invalid="ignore"):
                    far = np.nanmean(np.where(ring[None], h_anom, np.nan), axis=(1, 2))
                h_anom = h_anom - np.nan_to_num(far)[:, None, None]

        # Window-mean background + pointwise corrections from the parcels.
        H_ref_w = float(np.nanmean(np.where(fmask, H_path, np.nan)))
        tr_w = 2.0 * eta_bar / (rho_i * g * H_ref_w)
        div_ref = float(np.nanmean(np.where(fmask, divu_path, np.nan))) / SECONDS_PER_YEAR
        gamma_ref = div_ref * tr_w
        dH = np.clip(H_path - H_ref_w, -dH_clip_rel * H_ref_w, dH_clip_rel * H_ref_w)
        dgam = np.clip(
            (divu_path / SECONDS_PER_YEAR) * tr_w - gamma_ref, -dgamma_clip, dgamma_clip
        )

        op = HGammaWindowOp(
            H_ref=H_ref_w, gamma_ref=gamma_ref, div_u_ref_si=div_ref,
            dH_field=dH, dgamma_field=dgam,
            times=h_stack["time"].values[in_win],
            nx=nx, ny=ny, dx=dxm, dy=dym,
            eta_bar=eta_bar, rho_i=rho_i, rho_w=rho_w, g=g, theta=theta,
        )
        m_w, info = cg_invert_stationary(
            op, h_anom,
            tikhonov=tikhonov, length_scale_m=length_scale_m,
            max_iter=max_iter, tol=cg_tol, reference="time_mean",
        )
        m_windows[wi] = -m_w  # Shean sign
        window_meta.append(
            dict(anchor=str(times[a].date()), n_epochs=int(in_win.size),
                 H_ref=H_ref_w, gamma_ref=gamma_ref,
                 cg_iter=int(info["n_iter"]), converged=bool(info["converged"]))
        )
        if verbose:
            el = _time.time() - t0
            print(
                f"    window {wi + 1}/{len(anchors)} anchor={times[a].date()} "
                f"epochs={in_win.size} H_ref={H_ref_w:.0f}m gamma={gamma_ref:+.3f} "
                f"CG {info['n_iter']} it (conv={info['converged']})  "
                f"elapsed={el / 60:.1f} min",
                flush=True,
            )

    with np.errstate(invalid="ignore"):
        m_med = np.nanmedian(m_windows, axis=0)
        n_win = np.isfinite(m_windows).sum(axis=0)
        m_mad = 1.4826 * np.nanmedian(
            np.abs(m_windows - m_med[None]), axis=0
        )

    coords = {"y": h_stack["y"].values, "x": h_stack["x"].values}
    dims = ("y", "x")
    m_da = xr.DataArray(m_med, dims=dims, coords=coords, name="melt_rate")
    if floating_mask is not None:
        m_da = m_da.where(floating_mask)
        H_f_mean = H_f_mean.where(floating_mask)

    vx_mean2 = vx.mean("time", skipna=True) if "time" in vx.dims else vx
    vy_mean2 = vy.mean("time", skipna=True) if "time" in vy.dims else vy
    fd = flux_divergence(H_f_mean, vx_mean2, vy_mean2)

    return xr.Dataset(
        {
            "melt_rate": m_da,
            "melt_spread": xr.DataArray(m_mad, dims=dims, coords=coords),
            "window_count": xr.DataArray(n_win.astype(np.int32), dims=dims, coords=coords),
            "H_f_mean": H_f_mean,
            "flux_div": fd,
            "a_dot": xr.zeros_like(H_f_mean).rename("a_dot"),
        },
        attrs={
            "equation": (
                "windowed parcel-frame Stubblefield inverse (Level 1, "
                "plan_lagrangian.md): per-anchor co-moving frames w/ "
                "time-varying velocity; pointwise dH/dgamma kernel expansion; "
                "per-window masked CG; cross-window median"
            ),
            "units": (
                "m ice yr^-1 ANOMALY (k=0 unconstrained); Shean convention: "
                "negative = melt, positive = accretion"
            ),
            "rho_w": rho_w, "rho_i": rho_i, "eta_bar": eta_bar,
            "window_yr": window_yr, "stride_yr": stride_yr,
            "n_windows": len(anchors),
            "vel_smooth_m": vel_smooth_m, "vdiv_clip": vdiv_clip,
            "tikhonov": tikhonov, "length_scale_m": length_scale_m,
            "dH_clip_rel": dH_clip_rel, "dgamma_clip": dgamma_clip,
            "windows": str(window_meta),
        },
    )
