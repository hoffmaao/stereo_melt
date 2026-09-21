# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""Basal melt-rate solvers: Eulerian and Lagrangian mass-budget forms.

Mass conservation for an ice column of ice-equivalent thickness :math:`H`
with column-averaged velocity :math:`u`, surface mass balance
:math:`\dot a`, and basal mass balance :math:`\dot b` (**positive =
accretion, negative = melt**, as in Shean et al. 2019) is

.. math::
    \frac{\partial H}{\partial t} = -\nabla\!\cdot(H u) + \dot a + \dot b

(Shean et al. 2019, Eq. 4). ``melt_rate`` carries this basal-mass-balance
sign; **negate when comparing to
datasets that publish a positive=melt convention** (e.g. Davison 2023,
Adusumilli 2020, Paolo 2024).

The **Eulerian** form (their Eq. 10), implemented by
:func:`eulerian_melt_rate`:

.. math::
    \dot b = \frac{\partial H_f}{\partial t}
             + \nabla\!\cdot(H_f u) - \dot a

fits :math:`\partial H_f/\partial t` by per-pixel regression of the
stack against time and evaluates :math:`\nabla\!\cdot(H_f u)` on the
time-mean field (or, with ``common_epoch=True``, on that mean referred to
one epoch by :func:`~stereo_melt.kinematics.common_epoch_mean`).

The **Lagrangian** form (their Eq. 7), implemented by
:func:`lagrangian_melt_rate`:

.. math::
    \dot b = \frac{D H_f}{D t}
             + H_f\,\nabla\!\cdot u - \dot a

advects particles through the velocity field and replaces the fixed-grid
time derivative with the material derivative :math:`D H_f/D t` along
each trajectory. This is the correct form for strongly-advective flow
(e.g. Pine Island, Thwaites); the Eulerian form is adequate when the
advection term is small relative to local thinning.

Units
-----
Surface :math:`h` and thickness :math:`H_f` are in meters; velocities
are in meters per year; surface mass balance and melt rate are in
meters of ice-equivalent per year. The Eulerian ``dh/dt`` regression
returns a slope per second and is converted to per year internally.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xarray as xr

from .backend import asarray, map_coordinates, scatter_add, to_numpy, xp
from .constants import rhoi, rhow
from .dynamics.lagrangian_inverse import (
    linear_inverse_dhdt_lagrangian_melt_rate,
    linear_inverse_lagrangian_melt_rate,
)
from .freeboard import freeboard_to_thickness
from .kinematics import (
    SECONDS_PER_YEAR,
    DivergenceEstimator,
    common_epoch_mean,
    dh_dt,
    divergence,
    flux_divergence,
    gaussian_smooth_nan,
)

__all__ = [
    "eulerian_melt_rate",
    "lagrangian_melt_rate",
    "lagrangian_parcel_lsq_melt_rate",
    "linear_inverse_dhdt_lagrangian_melt_rate",
    "linear_inverse_lagrangian_melt_rate",
]


def _smooth_velocity_da(v: xr.DataArray, sigma_m: float | None) -> xr.DataArray:
    """NaN-aware Gaussian smoothing of a velocity component before differencing.

    Unsmoothed mosaics put grid-scale noise into :math:`\\nabla\\cdot u`,
    which the solvers multiply by ``H``. ``sigma_m`` in metres;
    ``None``/``<=0`` is a no-op. Time-sliced when a ``time`` dim is present.
    """
    if sigma_m is None or sigma_m <= 0:
        return v
    res = abs(float(v["x"].values[1] - v["x"].values[0]))
    sig = float(sigma_m) / res
    if "time" in v.dims:
        slices = [gaussian_smooth_nan(v.isel(time=i), sig) for i in range(v.sizes["time"])]
        return xr.concat(slices, dim="time").assign_coords(time=v["time"])
    return gaussian_smooth_nan(v, sig)


def eulerian_melt_rate(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    a_dot: xr.DataArray | float = 0.0,
    d: xr.DataArray | float = 0.0,
    rho_w: float = rhow,
    rho_i: float = rhoi,
    estimator: DivergenceEstimator | None = None,
    min_count: int = 3,
    robust_dh_dt: bool = False,
    vel_smooth_sigma_m: float | None = None,
    common_epoch: bool = False,
    epoch_rate_sigma_px: float = 2.0,
) -> xr.Dataset:
    r"""Return the basal melt-rate field from a repeat-DEM stack (Eulerian).

    Solves (positive = accretion)

    .. math::
        \dot b = \frac{\partial H_f}{\partial t}
                 + \nabla\!\cdot(H_f u) - \dot a

    where :math:`H_f` is obtained by hydrostatic inversion of the surface
    elevation stack, :math:`\partial H_f/\partial t` by per-pixel linear
    regression, and :math:`\nabla\!\cdot(H_f u)` by the supplied flux-
    divergence estimator.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Geoid-referenced, tidally- and IBE-corrected surface elevation
        stack on the target EPSG:3031 grid, meters.
    vx, vy : xarray.DataArray
        Column-averaged velocity on the ``(y, x)`` grid in m/yr. May
        carry a ``time`` dimension; in that case the time-mean is used.
    a_dot : xarray.DataArray or float, optional
        Surface mass balance rate in m ice / yr. Scalar or field
        broadcastable against the target grid. Defaults to 0.
    d : xarray.DataArray or float, optional
        Firn air content in meters. Defaults to 0 (pure ice).
    rho_w, rho_i : float
        Seawater and ice densities in kg m\ :sup:`-3`.
    estimator : DivergenceEstimator, optional
        Alternative flux-divergence estimator. Defaults to central
        finite differences.
    common_epoch : bool, optional
        Refer the mean thickness entering the flux divergence to one
        epoch with :func:`~stereo_melt.kinematics.common_epoch_mean`,
        instead of using the per-pixel time-mean. The time-mean is
        evaluated at whatever epochs each pixel's strips supply, so on a
        changing surface it carries a strip-shaped sampling artifact that
        the divergence differentiates.
    epoch_rate_sigma_px : float, optional
        Smoothing scale of the rate field used for that correction.
    min_count : int
        Minimum finite samples per pixel required by the dh/dt
        regression; pixels with fewer valid samples are NaN.

    Returns
    -------
    xarray.Dataset
        See :func:`lagrangian_melt_rate` for the matching variable set.
        With ``common_epoch=True``, ``H_f_mean`` is the common-epoch mean
        rather than the time-mean; the choice is stamped in the
        ``common_epoch`` attr.
    """
    vx = _smooth_velocity_da(vx, vel_smooth_sigma_m)
    vy = _smooth_velocity_da(vy, vel_smooth_sigma_m)

    H_f_stack = freeboard_to_thickness(h_stack, d=d, rho_w=rho_w, rho_i=rho_i)

    reg = dh_dt(H_f_stack, min_count=min_count, robust=robust_dh_dt)
    dHdt = reg["slope"] * SECONDS_PER_YEAR

    if common_epoch:
        H_f_mean = common_epoch_mean(H_f_stack, reg["slope"],
                                     sigma_px=epoch_rate_sigma_px)
    else:
        H_f_mean = H_f_stack.mean("time", skipna=True)

    if "time" in vx.dims:
        vx = vx.mean("time", skipna=True)
    if "time" in vy.dims:
        vy = vy.mean("time", skipna=True)

    fd = flux_divergence(H_f_mean, vx, vy, estimator=estimator)

    a_dot_field = xr.DataArray(a_dot) if not isinstance(a_dot, xr.DataArray) else a_dot
    a_dot_field = a_dot_field.broadcast_like(H_f_mean)

    melt_rate = dHdt + fd - a_dot_field

    return xr.Dataset(
        {
            "melt_rate": melt_rate,
            "H_f_mean": H_f_mean,
            "dHdt": dHdt,
            "flux_div": fd,
            "a_dot": a_dot_field,
            "count": reg["count"],
            "rmse": reg["rmse"],
        },
        attrs={
            "equation": "b_dot = dH/dt + div(H u) - a_dot  (Shean 2019 Eq. 10)",
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "rho_w": rho_w,
            "rho_i": rho_i,
            "vel_smooth_sigma_m": float(vel_smooth_sigma_m or 0.0),
            "common_epoch": int(bool(common_epoch)),
            "epoch_rate_sigma_px": float(epoch_rate_sigma_px),
        },
    )


def _diag_region_stats(
    diag_flat: np.ndarray, idx_np: np.ndarray, val_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-region (count, mean, median) of one epoch pair's per-cell values."""
    n_regions = diag_flat.shape[0]
    cnt = np.zeros(n_regions, dtype=np.int64)
    mean = np.full(n_regions, np.nan)
    med = np.full(n_regions, np.nan)
    for r in range(n_regions):
        sel = diag_flat[r][idx_np]
        n = int(sel.sum())
        cnt[r] = n
        if n:
            v = val_np[sel]
            mean[r] = float(v.mean())
            med[r] = float(np.median(v))
    return cnt, mean, med


def lagrangian_melt_rate(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    a_dot: xr.DataArray | float = 0.0,
    d: xr.DataArray | float = 0.0,
    rho_w: float = rhow,
    rho_i: float = rhoi,
    dt_yr: float = 0.05,
    aggregator: str = "mean",
    pairs: str = "consecutive",
    min_dt_yr: float = 0.0,
    max_dt_yr: float | None = None,
    seed_stride: int = 1,
    output: str = "path",
    progress_interval_s: float = 30.0,
    vel_smooth_sigma_m: float | None = None,
    vdiv_clip: float | None = None,
    pair_diag_masks: np.ndarray | None = None,
) -> xr.Dataset:
    r"""Return the basal melt-rate field via Lagrangian path integration.

    Scalar-endpoint scheme. For each epoch pair :math:`(i, j)`:

    1. Seed a particle at every pixel (or every ``seed_stride``-th
       pixel) of the earlier DEM.
    2. Advect particles with forward-Euler integration at sub-steps of
       ``dt_yr`` years, bilinearly sampling ``(vx, vy)`` at each step.
    3. Sample the later DEM at the endpoint positions and assign a
       scalar material derivative :math:`D H_f / D t = \Delta H_f /
       \Delta t` to each trajectory.
    4. At every step, contribute (positive = accretion)
       :math:`D H_f/D t + H_f(t)\,\nabla\!\cdot u - \dot a` to the
       visited cell, where :math:`H_f(t)` evolves linearly along the
       path.

    Per-cell melt rate is the mean (or median) of path contributions.
    Runs fully on the active array backend (numpy or cupy);
    ``aggregator="median"`` offloads the final gather to CPU.

    Parameters
    ----------
    h_stack : xarray.DataArray, dims ``(time, y, x)``
        Geoid-referenced, corrected surface elevation stack, meters.
    vx, vy : xarray.DataArray
        Column-averaged velocity on the ``(y, x)`` grid in m/yr. A
        ``time`` dimension with a single entry (or no ``time`` dim) is
        used as a static field. With >1 ``time`` entry the advection is
        time-resolved: at each integration sub-step the parcel samples
        velocity (and its divergence) linearly interpolated between the
        two bracketing velocity epochs, held constant beyond the
        first/last epoch. The diagnostic ``flux_div`` always uses the
        time-mean. NaN velocity (data gaps) is handled: a parcel that
        samples a gap is dropped from that step on (same as leaving the
        domain), so gaps may be passed through rather than zero-filled;
        NaN divergence in the gap halo masks only the affected step.
    a_dot : xarray.DataArray or float, optional
        Surface mass balance in m ice / yr.
    d : xarray.DataArray or float, optional
        Firn air content, meters.
    rho_w, rho_i : float
        Seawater and ice densities in kg m\ :sup:`-3`.
    dt_yr : float
        Forward-Euler integration sub-step in years. Smaller values
        reduce integration error for strongly-curved trajectories.
    aggregator : {"mean", "median", "pair_median"}
        Per-cell reduction. ``"mean"`` and ``"median"`` pool every path
        contribution (in ``"path"`` mode, every visited step of every parcel);
        ``"median"`` gathers to CPU via :func:`pandas.DataFrame.groupby`.
        ``"pair_median"`` is a two-level mosaic: reduce each epoch
        pair to one mean value per cell, then take the median across pairs --
        robust to per-pair / per-strip outliers, and memory-bounded (one value
        per pair-cell, not per step, so it is feasible for dense ``"path"`` runs
        where pooled ``"median"`` would not be). In ``"origin"`` mode each
        parcel already yields one value per pair-cell, so ``"median"`` and
        ``"pair_median"`` coincide.
    pairs : {"consecutive", "all"}
        Which epoch pairs to integrate. ``"consecutive"`` uses
        :math:`(0,1), (1,2), \dots, (T-2, T-1)`; ``"all"`` uses every
        :math:`i<j`.
    min_dt_yr : float
        Minimum pair baseline. Pairs with :math:`t_j - t_i < \text{min\_dt\_yr}`
        are skipped. Short baselines amplify per-epoch coregistration
        residuals into bogus :math:`\partial h/\partial t` (noise / dt);
        the basin pipelines use a 1.5 yr floor. Set to a value comparable to the
        expected coregistration-error / melt-signal ratio.
    max_dt_yr : float or None
        Maximum pair baseline. Pairs with :math:`t_j - t_i > \text{max\_dt\_yr}`
        are skipped (``None`` = no cap). The companion upper bound to
        ``min_dt_yr``: on fast-flowing shelves long baselines advect particles
        tens of km on a *time-mean* velocity field, accumulating trajectory
        error and walking seeds out of the domain (production: 2.5 yr). Without it,
        ``pairs="all"`` on a multi-year stack is O(T^2) in epoch count and the
        step budget is dominated by long, low-quality trajectories.
    seed_stride : int
        Pixel stride for particle seeding. ``1`` seeds every pixel.
        ``output="origin"`` requires ``1`` (one value per seed cell).
    output : {"path", "origin"}
        Where each trajectory's melt is deposited. ``"path"`` (default)
        scatters every step's contribution into the cell visited at that
        step (distributed product); with sparse ``seed_stride`` it
        under-samples slow ice into a grid-scale checkerboard. ``"origin"``
        averages the
        contribution along the trajectory into one value, assign it to the
        parcel's seed (origin) cell, and drop parcels that never leave that
        cell. With dense seeding this is checkerboard-free — the published
        PIG scheme.
    progress_interval_s : float
        Wall-clock seconds between progress prints during the epoch-pair
        integration. Each line reports pairs done, cumulative integration
        steps vs. the precomputed budget, elapsed time, step rate, and ETA.
        Set to ``0`` to silence. ``pairs="all"`` is O(T^2) in epoch count
        and this loop dominates runtime, so it narrates by default.
    pair_diag_masks : numpy.ndarray or None
        Opt-in per-pair diagnostic regions, shape ``(R, ny, nx)`` bool
        (regions may overlap). For every dt-band epoch pair, records
        count / mean / median of that pair's per-cell melt values inside
        each region — the melt-space strip-QC hook: a strip with a
        vertical-datum or tilt residual biases every pair it enters, with
        opposite sign as earlier vs. later epoch, so per-pair region stats
        attribute a localized artefact to epochs. Requires the per-pair
        reduction, i.e. ``aggregator="pair_median"`` (path mode) or
        ``output="origin"`` (origin deposits are per-pair by construction;
        stats are then over each parcel's SEED cell region). Adds
        ``pair_i/pair_j/pair_dt`` (pair,) and
        ``pair_diag_count/mean/median`` (pair, region) to the output.

    Returns
    -------
    xarray.Dataset
        Variables on the ``(y, x)`` grid:

        - ``melt_rate`` — basal mass balance, m ice yr\ :sup:`-1`
          (negative = melt, positive = accretion)
        - ``H_f_mean`` — time-mean ice-equivalent thickness, m
        - ``dHdt`` — Lagrangian :math:`D H_f/D t`, m ice yr\ :sup:`-1`
        - ``flux_div`` — :math:`\nabla\!\cdot(H_f u)` on the time-mean
          thickness, for diagnostic parity with the Eulerian output
        - ``a_dot`` — broadcast SMB rate, m ice yr\ :sup:`-1`
        - ``count`` — number of path contributions per cell
        - ``rmse`` — standard deviation of path contributions per cell
    """
    if aggregator not in ("mean", "median", "pair_median"):
        raise ValueError(
            f"aggregator must be 'mean', 'median' or 'pair_median', got {aggregator!r}"
        )
    if pairs not in ("consecutive", "all"):
        raise ValueError(f"pairs must be 'consecutive' or 'all', got {pairs!r}")
    if seed_stride < 1:
        raise ValueError(f"seed_stride must be >= 1, got {seed_stride}")
    if output not in ("path", "origin"):
        raise ValueError(f"output must be 'path' or 'origin', got {output!r}")
    if output == "origin" and seed_stride != 1:
        raise ValueError(
            "output='origin' assigns one value per seed cell, so it needs dense "
            f"seeding; got seed_stride={seed_stride} (use seed_stride=1)"
        )
    if dt_yr <= 0:
        raise ValueError(f"dt_yr must be > 0, got {dt_yr}")
    if min_dt_yr < 0:
        raise ValueError(f"min_dt_yr must be >= 0, got {min_dt_yr}")
    if max_dt_yr is not None and max_dt_yr <= min_dt_yr:
        raise ValueError(
            f"max_dt_yr must be > min_dt_yr; got max={max_dt_yr}, min={min_dt_yr}"
        )
    if pair_diag_masks is not None and output == "path" and aggregator != "pair_median":
        raise ValueError(
            "pair_diag_masks needs per-pair values: use aggregator='pair_median' "
            f"(path mode) or output='origin'; got output={output!r}, "
            f"aggregator={aggregator!r}"
        )

    # Input hygiene: NaN-aware
    # Gaussian velocity smoothing before any differencing, and a hard clip
    # on the divergence before it multiplies H. Both default OFF for
    # back-compat with existing production runs.
    vx = _smooth_velocity_da(vx, vel_smooth_sigma_m)
    vy = _smooth_velocity_da(vy, vel_smooth_sigma_m)

    # Hydrostatic inversion
    H_f_stack_xr = freeboard_to_thickness(h_stack, d=d, rho_w=rho_w, rho_i=rho_i)
    H_f_mean_xr = H_f_stack_xr.mean("time", skipna=True)

    # Velocity for the integration. A time-mean field is always built (used by
    # the diagnostic flux_div and the static advection path). If vx/vy carry a
    # time dimension with >1 entry, the advection is time-resolved: each parcel
    # samples velocity at its absolute time along the trajectory (built below).
    time_varying_vel = "time" in vx.dims and vx.sizes["time"] > 1
    vx_mean_xr = vx.mean("time", skipna=True) if "time" in vx.dims else vx
    vy_mean_xr = vy.mean("time", skipna=True) if "time" in vy.dims else vy
    vdiv_xr = divergence(vx_mean_xr, vy_mean_xr)

    # Grid geometry (EPSG:3031: x ascends, y descends)
    x_coords = H_f_stack_xr["x"].values
    y_coords = H_f_stack_xr["y"].values
    ny = H_f_stack_xr.sizes["y"]
    nx = H_f_stack_xr.sizes["x"]
    res_x = float(x_coords[1] - x_coords[0])
    res_y = float(y_coords[0] - y_coords[1])

    # Move arrays to active backend
    H_f_stack = asarray(np.asarray(H_f_stack_xr.values, dtype=np.float64))
    vx_arr = asarray(np.asarray(vx_mean_xr.values, dtype=np.float64))
    vy_arr = asarray(np.asarray(vy_mean_xr.values, dtype=np.float64))
    vdiv_arr = asarray(np.asarray(vdiv_xr.values, dtype=np.float64))
    if vdiv_clip is not None:
        # Clip implausible divergence (e.g. ±0.2 /yr). NaN propagates.
        vdiv_arr = xp.clip(vdiv_arr, -float(vdiv_clip), float(vdiv_clip))

    if isinstance(a_dot, (int, float)):
        a_dot_arr = xp.full((ny, nx), float(a_dot), dtype=xp.float64)
    else:
        a_dot_arr = asarray(np.asarray(a_dot.broadcast_like(H_f_mean_xr).values, dtype=np.float64))

    # Time axis in years since epoch 0
    times = pd.to_datetime(H_f_stack_xr["time"].values)
    t_years = np.array(
        [(t - times[0]).total_seconds() / SECONDS_PER_YEAR for t in times], dtype=np.float64
    )

    # Time-varying velocity stack (optional). Sorted by time, with the velocity
    # time axis in the SAME "years since times[0]" frame as the trajectories, so
    # a parcel at absolute time t samples between the two bracketing velocity
    # epochs. v_t_years stays numpy (scalar per-step lookups in the hot loop).
    if time_varying_vel:
        vx_sorted = vx.sortby("time")
        vy_sorted = vy.sortby("time")
        v_times = pd.to_datetime(vx_sorted["time"].values)
        v_t_years = np.array(
            [(t - times[0]).total_seconds() / SECONDS_PER_YEAR for t in v_times],
            dtype=np.float64,
        )
        nv = int(v_t_years.size)
        vx_tv = asarray(np.asarray(vx_sorted.values, dtype=np.float64))
        vy_tv = asarray(np.asarray(vy_sorted.values, dtype=np.float64))
        vdiv_tv = asarray(
            np.stack(
                [
                    divergence(vx_sorted.isel(time=k), vy_sorted.isel(time=k)).values
                    for k in range(nv)
                ]
            ).astype(np.float64)
        )
        if vdiv_clip is not None:
            vdiv_tv = xp.clip(vdiv_tv, -float(vdiv_clip), float(vdiv_clip))
        if progress_interval_s > 0:
            print(
                f"  time-varying velocity: {nv} epochs at "
                f"{[str(t.date()) for t in v_times]} "
                f"(t_years={np.round(v_t_years, 2).tolist()}); linear-interp "
                f"along trajectory, held constant beyond first/last epoch",
                flush=True,
            )

    # Epoch pairs
    T = len(times)
    if pairs == "consecutive":
        epoch_pairs = [(i, i + 1) for i in range(T - 1)]
    else:
        epoch_pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]

    # Group dt-valid epoch pairs by START epoch: seed each start DEM once, integrate ONE
    # trajectory out to its farthest partner, and reuse that trajectory for every
    # partner DEM (they share all but the tail). That is ~T fewer integrations
    # than a per-(i,j)-pair loop, so the step budget is the sum over starts of the
    # farthest-partner length, not the sum over pairs.
    starts: dict[int, list] = {}
    for i, j in epoch_pairs:
        dt_total = t_years[j] - t_years[i]
        if dt_total <= 0 or dt_total < min_dt_yr:
            continue
        if max_dt_yr is not None and dt_total > max_dt_yr:
            continue
        starts.setdefault(i, []).append((j, dt_total))
    for i in starts:
        starts[i].sort(key=lambda jt: jt[0])  # partners ascending in epoch -> dt
    start_list = sorted(starts)

    # Integration on a fixed dt_yr grid. Budget = sum over starts of
    # the steps to that start's farthest partner.
    n_steps_far = {i: max(1, int(np.ceil(starts[i][-1][1] / dt_yr))) for i in start_list}
    total_steps = sum(n_steps_far.values())
    n_starts = len(start_list)
    n_pairs = sum(len(v) for v in starts.values())

    # Strided seed-candidate grid. Per start we keep only the finite-thickness
    # pixels of THAT start DEM --
    # dense per-pixel seeding, with each start's trajectory-history arrays sized
    # to one DEM's live cells rather than the full grid.
    y_cand, x_cand = xp.mgrid[0:ny:seed_stride, 0:nx:seed_stride]
    y_cand = y_cand.ravel().astype(xp.int64)
    x_cand = x_cand.ravel().astype(xp.int64)

    # Cell-wise accumulators (flat indexing into (ny, nx)), summed across every
    # start/partner deposit.
    n_cells = ny * nx
    sum_bdot = xp.zeros(n_cells, dtype=xp.float64)
    sum_sq = xp.zeros(n_cells, dtype=xp.float64)
    sum_DhDt = xp.zeros(n_cells, dtype=xp.float64)
    count = xp.zeros(n_cells, dtype=xp.int32)

    # Optional CPU gather for median / pair_median aggregators
    median_bucket_idx: list = []
    median_bucket_val: list = []
    # Per-pair scratch for aggregator="pair_median" (two-level mosaic:
    # mean within a pair -> one value per pair per cell -> median across pairs).
    # Reused across pairs; reset only at each pair's touched cells.
    pair_sum = xp.zeros(n_cells, dtype=xp.float64) if aggregator == "pair_median" else None
    pair_cnt = xp.zeros(n_cells, dtype=xp.float64) if aggregator == "pair_median" else None

    # Opt-in per-pair region diagnostics (numpy-side; values are gathered to
    # CPU for the median bucket anyway, so this adds no backend traffic).
    diag_flat = None
    diag_rows: list | None = None
    if pair_diag_masks is not None:
        masks_np = np.asarray(pair_diag_masks, dtype=bool)
        if masks_np.ndim == 2:
            masks_np = masks_np[None]
        if masks_np.shape[1:] != (ny, nx):
            raise ValueError(
                f"pair_diag_masks shape {masks_np.shape} does not match grid "
                f"(R, {ny}, {nx})"
            )
        diag_flat = masks_np.reshape(masks_np.shape[0], n_cells)
        diag_rows = []
    if progress_interval_s > 0:
        cap_txt = f", dt<={max_dt_yr:.3f}yr" if max_dt_yr is not None else ""
        print(
            f"  Lagrangian path integration (output={output!r}, pairs={pairs!r}, "
            f"aggregator={aggregator!r}, per-start): {n_starts:,} start epochs, "
            f"{n_pairs:,} partner pairs after dt>={min_dt_yr:.3f}yr{cap_txt} filter "
            f"(of {len(epoch_pairs):,} i<j), {total_steps:,} integration steps "
            f"(seed_stride={seed_stride})",
            flush=True,
        )

    loop_t0 = time.time()
    last_print = loop_t0
    steps_done = 0
    for s_idx, i in enumerate(start_list):
        partners = starts[i]
        nsf = n_steps_far[i]
        # Charge the budget up front so empty-seed starts still carry the meter.
        steps_done += nsf

        # Seed this start DEM's finite-thickness pixels.
        keep = xp.isfinite(H_f_stack[i])[y_cand, x_cand]
        y_seed_flat = y_cand[keep]
        x_seed_flat = x_cand[keep]
        n_seeds = int(y_seed_flat.size)
        if n_seeds == 0:
            continue
        H_ref_at_seeds = H_f_stack[i][y_seed_flat, x_seed_flat]
        a_dot_at_seeds = a_dot_arr[y_seed_flat, x_seed_flat]
        origin_flat = (y_seed_flat * nx + x_seed_flat).astype(xp.int64)

        x_idx = x_seed_flat.astype(xp.float64).copy()
        y_idx = y_seed_flat.astype(xp.float64).copy()
        valid = xp.ones(n_seeds, dtype=bool)

        # One trajectory integrated out to the farthest partner. Float positions
        # are kept so a nearer partner can sample its endpoint at its own step.
        cell_flat_hist = xp.zeros((nsf, n_seeds), dtype=xp.int64)
        vdiv_hist = xp.zeros((nsf, n_seeds), dtype=xp.float64)
        valid_hist = xp.zeros((nsf, n_seeds), dtype=bool)        # cumulative-valid AND finite-div
        valid_cum_hist = xp.zeros((nsf, n_seeds), dtype=bool)    # cumulative on-grid validity
        posx_hist = xp.zeros((nsf, n_seeds), dtype=xp.float64)
        posy_hist = xp.zeros((nsf, n_seeds), dtype=xp.float64)

        for k in range(nsf):
            if time_varying_vel:
                # Parcel absolute time at the start of this sub-step, then the
                # bracketing velocity epochs (constant extrapolation at ends).
                t_abs = t_years[i] + k * dt_yr
                if t_abs <= v_t_years[0]:
                    k0 = k1 = 0
                    w = 0.0
                elif t_abs >= v_t_years[-1]:
                    k0 = k1 = nv - 1
                    w = 0.0
                else:
                    k1 = int(np.searchsorted(v_t_years, t_abs))
                    k0 = k1 - 1
                    w = float((t_abs - v_t_years[k0]) / (v_t_years[k1] - v_t_years[k0]))
                if k0 == k1:
                    vx_t = map_coordinates(vx_tv[k0], [y_idx, x_idx], order=1, mode="nearest")
                    vy_t = map_coordinates(vy_tv[k0], [y_idx, x_idx], order=1, mode="nearest")
                    vdiv_t = map_coordinates(vdiv_tv[k0], [y_idx, x_idx], order=1, mode="nearest")
                else:
                    vx_t = (1.0 - w) * map_coordinates(
                        vx_tv[k0], [y_idx, x_idx], order=1, mode="nearest"
                    ) + w * map_coordinates(vx_tv[k1], [y_idx, x_idx], order=1, mode="nearest")
                    vy_t = (1.0 - w) * map_coordinates(
                        vy_tv[k0], [y_idx, x_idx], order=1, mode="nearest"
                    ) + w * map_coordinates(vy_tv[k1], [y_idx, x_idx], order=1, mode="nearest")
                    vdiv_t = (1.0 - w) * map_coordinates(
                        vdiv_tv[k0], [y_idx, x_idx], order=1, mode="nearest"
                    ) + w * map_coordinates(vdiv_tv[k1], [y_idx, x_idx], order=1, mode="nearest")
            else:
                vx_t = map_coordinates(vx_arr, [y_idx, x_idx], order=1, mode="nearest")
                vy_t = map_coordinates(vy_arr, [y_idx, x_idx], order=1, mode="nearest")
                vdiv_t = map_coordinates(vdiv_arr, [y_idx, x_idx], order=1, mode="nearest")

            # NaN-robust sampling: a velocity gap (NaN -- e.g. a data void in the
            # mosaic) can't advect a parcel, so freeze it (0 displacement keeps
            # x_idx/y_idx finite, avoiding a NaN->int cast) and invalidate it
            # from this step on -- the same treatment as leaving the domain. A
            # NaN divergence sample (the ~1-px halo divergence() spreads around a
            # gap) masks only that step's contribution, not the whole trajectory.
            finite_v = xp.isfinite(vx_t) & xp.isfinite(vy_t)
            vx_t = xp.where(finite_v, vx_t, 0.0)
            vy_t = xp.where(finite_v, vy_t, 0.0)

            # Advect: vx (m/yr) over the fixed dt_yr step -> meters -> index space.
            # EPSG:3031 convention: x ascends, y descends, so dy_idx = -vy*dt/res_y.
            dx_idx = vx_t * dt_yr / res_x
            dy_idx = -vy_t * dt_yr / res_y
            x_idx = x_idx + dx_idx
            y_idx = y_idx + dy_idx

            in_bounds = (x_idx >= 0) & (x_idx <= nx - 1) & (y_idx >= 0) & (y_idx <= ny - 1)
            valid = valid & in_bounds & finite_v

            xi = xp.clip(xp.round(x_idx).astype(xp.int64), 0, nx - 1)
            yi = xp.clip(xp.round(y_idx).astype(xp.int64), 0, ny - 1)
            cell_flat_hist[k] = yi * nx + xi
            finite_div = xp.isfinite(vdiv_t)
            vdiv_hist[k] = xp.where(finite_div, vdiv_t, 0.0)
            valid_cum_hist[k] = valid
            valid_hist[k] = valid & finite_div
            posx_hist[k] = x_idx
            posy_hist[k] = y_idx

        # Each partner DEM reuses the shared trajectory, sliced at its own step.
        for j, dt_total in partners:
            # DEM times round to the fixed dt grid: the endpoint is sampled at the nearest step, DhDt uses the actual dt.
            h_idx = max(1, min(nsf, int(round(dt_total / dt_yr))))
            H_end = map_coordinates(
                H_f_stack[j],
                [posy_hist[h_idx - 1], posx_hist[h_idx - 1]],
                order=1,
                mode="constant",
                cval=float("nan"),
            )
            DhDt = (H_end - H_ref_at_seeds) / dt_total
            valid_end = xp.isfinite(DhDt) & valid_cum_hist[h_idx - 1]

            # Deposit the trajectory's melt contribution(s). Two schemes:
            #   "path"   — scatter each step's bdot into the cell visited at that
            #              step (per-cell mean); never-moving parcels are kept
            #              (valid bilinear endpoint DH/Dt).
            #   "origin" — average bdot along the
            #              trajectory into ONE value at the parcel's ORIGIN (seed)
            #              cell, dropping never-moving parcels (degenerate Eulerian
            #              dh/dt). Dense seeding keeps coverage complete, so there
            #              is no sparse-seed/scatter checkerboard.
            if output == "path":
                pair_chunks = [] if aggregator == "pair_median" else None
                for k in range(h_idx):
                    t_k = (k + 1) * dt_yr
                    h_t = H_ref_at_seeds + DhDt * t_k
                    bdot_k = DhDt + h_t * vdiv_hist[k] - a_dot_at_seeds
                    mask_k = valid_hist[k] & valid_end

                    idx = cell_flat_hist[k][mask_k]
                    if idx.size == 0:
                        continue
                    vals = bdot_k[mask_k]

                    scatter_add(sum_bdot, idx, vals)
                    scatter_add(sum_sq, idx, vals * vals)
                    scatter_add(sum_DhDt, idx, DhDt[mask_k])
                    scatter_add(count, idx, xp.ones(idx.shape, dtype=xp.int32))

                    if aggregator == "median":
                        median_bucket_idx.append(to_numpy(idx))
                        median_bucket_val.append(to_numpy(vals))
                    elif aggregator == "pair_median":
                        # Accumulate this pair's per-cell sum/count; collapse to
                        # one mean value per visited cell after the step loop.
                        scatter_add(pair_sum, idx, vals)
                        scatter_add(pair_cnt, idx, xp.ones(idx.shape, dtype=xp.float64))
                        pair_chunks.append(idx)
                if aggregator == "pair_median" and pair_chunks:
                    tidx = xp.unique(xp.concatenate(pair_chunks))
                    tidx_np = to_numpy(tidx)
                    pvals_np = to_numpy(pair_sum[tidx] / pair_cnt[tidx])
                    median_bucket_idx.append(tidx_np)
                    median_bucket_val.append(pvals_np)
                    pair_sum[tidx] = 0.0  # reset only this pair's touched cells
                    pair_cnt[tidx] = 0.0
                    if diag_rows is not None:
                        diag_rows.append(
                            (i, j, dt_total)
                            + _diag_region_stats(diag_flat, tidx_np, pvals_np)
                        )
                elif diag_rows is not None:
                    diag_rows.append(
                        (i, j, dt_total)
                        + _diag_region_stats(
                            diag_flat, np.empty(0, np.int64), np.empty(0)
                        )
                    )
            else:  # output == "origin"
                psum = xp.zeros(n_seeds, dtype=xp.float64)
                pcnt = xp.zeros(n_seeds, dtype=xp.float64)
                moved = xp.zeros(n_seeds, dtype=bool)
                for k in range(h_idx):
                    t_k = (k + 1) * dt_yr
                    h_t = H_ref_at_seeds + DhDt * t_k
                    bdot_k = DhDt + h_t * vdiv_hist[k] - a_dot_at_seeds
                    mk = valid_hist[k]
                    psum = psum + xp.where(mk, bdot_k, 0.0)
                    pcnt = pcnt + mk.astype(xp.float64)
                    # "moved" iff the rounded cell ever differs from the origin cell
                    moved = moved | (mk & (cell_flat_hist[k] != origin_flat))

                good = valid_end & (pcnt > 0) & moved
                if bool(good.any()):
                    parcel_mb = psum / xp.maximum(pcnt, 1.0)
                    idx = origin_flat[good]
                    vals = parcel_mb[good]
                    scatter_add(sum_bdot, idx, vals)
                    scatter_add(sum_sq, idx, vals * vals)
                    scatter_add(sum_DhDt, idx, DhDt[good])
                    scatter_add(count, idx, xp.ones(idx.shape, dtype=xp.int32))

                    # Origin deposits one value per parcel == one per pair per
                    # cell, so "median" here already IS the cross-pair median;
                    # "pair_median" routes to the same bucket.
                    if aggregator in ("median", "pair_median") or diag_rows is not None:
                        idx_np = to_numpy(idx)
                        vals_np = to_numpy(vals)
                        if aggregator in ("median", "pair_median"):
                            median_bucket_idx.append(idx_np)
                            median_bucket_val.append(vals_np)
                        if diag_rows is not None:
                            diag_rows.append(
                                (i, j, dt_total)
                                + _diag_region_stats(diag_flat, idx_np, vals_np)
                            )
                elif diag_rows is not None:
                    diag_rows.append(
                        (i, j, dt_total)
                        + _diag_region_stats(
                            diag_flat, np.empty(0, np.int64), np.empty(0)
                        )
                    )

        if progress_interval_s > 0:
            now = time.time()
            if now - last_print >= progress_interval_s:
                elapsed = now - loop_t0
                rate = steps_done / elapsed if elapsed > 0 else 0.0
                eta = (total_steps - steps_done) / rate if rate > 0 else float("nan")
                print(
                    f"    starts {s_idx + 1:,}/{n_starts:,}  "
                    f"steps {steps_done:,}/{total_steps:,} "
                    f"({100 * steps_done / total_steps:.1f}%)  "
                    f"elapsed={elapsed / 60:.1f}min  rate={rate:,.0f} steps/s  "
                    f"eta={eta / 60:.1f}min",
                    flush=True,
                )
                last_print = now

    if progress_interval_s > 0:
        total_elapsed = time.time() - loop_t0
        done_rate = total_steps / total_elapsed if total_elapsed > 0 else 0.0
        print(
            f"  Lagrangian integration done: {n_starts:,} starts, {n_pairs:,} pairs, "
            f"{total_steps:,} steps in {total_elapsed / 60:.1f}min ({done_rate:,.0f} steps/s)",
            flush=True,
        )

    # Reduce
    sum_bdot_np = to_numpy(sum_bdot)
    sum_sq_np = to_numpy(sum_sq)
    sum_DhDt_np = to_numpy(sum_DhDt)
    count_np = to_numpy(count)
    ok = count_np > 0

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bdot = np.where(ok, sum_bdot_np / np.maximum(count_np, 1), np.nan)
        mean_DhDt = np.where(ok, sum_DhDt_np / np.maximum(count_np, 1), np.nan)
        mean_sq = np.where(ok, sum_sq_np / np.maximum(count_np, 1), np.nan)
        rmse = np.sqrt(np.maximum(mean_sq - mean_bdot**2, 0.0))

    if aggregator in ("median", "pair_median"):
        if not median_bucket_idx:
            final_bdot = mean_bdot
        else:
            all_idx = np.concatenate(median_bucket_idx)
            all_vals = np.concatenate(median_bucket_val)
            med = pd.DataFrame({"idx": all_idx, "val": all_vals}).groupby("idx")["val"].median()
            final_bdot = np.full(n_cells, np.nan, dtype=np.float64)
            final_bdot[med.index.values] = med.values
    else:
        final_bdot = mean_bdot

    final_bdot_2d = final_bdot.reshape(ny, nx)
    mean_DhDt_2d = mean_DhDt.reshape(ny, nx)
    count_2d = count_np.reshape(ny, nx)
    rmse_2d = rmse.reshape(ny, nx)

    coords = {"y": y_coords, "x": x_coords}
    dims = ("y", "x")

    # Diagnostic flux_div on the time-mean field, for parity with the Eulerian output
    fd = flux_divergence(H_f_mean_xr, vx_mean_xr, vy_mean_xr)
    a_dot_field_xr = xr.DataArray(a_dot) if not isinstance(a_dot, xr.DataArray) else a_dot
    a_dot_field_xr = a_dot_field_xr.broadcast_like(H_f_mean_xr)

    ds = xr.Dataset(
        {
            "melt_rate": (dims, final_bdot_2d),
            "H_f_mean": H_f_mean_xr,
            "dHdt": (dims, mean_DhDt_2d),
            "flux_div": fd,
            "a_dot": a_dot_field_xr,
            "count": (dims, count_2d),
            "rmse": (dims, rmse_2d),
        },
        coords=coords,
        attrs={
            "equation": "b_dot = DH/Dt + H div(u) - a_dot  (Shean 2019 Eq. 7 / Eq. 10 Lagrangian)",
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "rho_w": rho_w,
            "rho_i": rho_i,
            "integrator": "forward-Euler",
            "output": output,
            "dt_yr": dt_yr,
            "aggregator": aggregator,
            "pairs": pairs,
            "min_dt_yr": min_dt_yr,
            "seed_stride": seed_stride,
            "velocity_time_varying": int(time_varying_vel),
            "velocity_n_epochs": int(nv) if time_varying_vel else 1,
            "vel_smooth_sigma_m": float(vel_smooth_sigma_m or 0.0),
            "vdiv_clip": float(vdiv_clip) if vdiv_clip is not None else 0.0,
        },
    )

    if diag_rows is not None:
        n_regions = diag_flat.shape[0]
        if diag_rows:
            pair_i_np = np.array([r[0] for r in diag_rows], dtype=np.int64)
            pair_j_np = np.array([r[1] for r in diag_rows], dtype=np.int64)
            pair_dt_np = np.array([r[2] for r in diag_rows], dtype=np.float64)
            dcnt = np.stack([r[3] for r in diag_rows])
            dmean = np.stack([r[4] for r in diag_rows])
            dmed = np.stack([r[5] for r in diag_rows])
        else:
            pair_i_np = np.zeros(0, dtype=np.int64)
            pair_j_np = np.zeros(0, dtype=np.int64)
            pair_dt_np = np.zeros(0, dtype=np.float64)
            dcnt = np.zeros((0, n_regions), dtype=np.int64)
            dmean = np.zeros((0, n_regions))
            dmed = np.zeros((0, n_regions))
        ds["pair_i"] = ("pair", pair_i_np)
        ds["pair_j"] = ("pair", pair_j_np)
        ds["pair_dt_yr"] = ("pair", pair_dt_np)
        ds["pair_diag_count"] = (("pair", "region"), dcnt)
        ds["pair_diag_mean"] = (("pair", "region"), dmean)
        ds["pair_diag_median"] = (("pair", "region"), dmed)

    return ds


def lagrangian_parcel_lsq_melt_rate(
    h_stack: xr.DataArray,
    vx: xr.DataArray,
    vy: xr.DataArray,
    a_dot: xr.DataArray | float = 0.0,
    d: xr.DataArray | float = 0.0,
    rho_w: float = rhow,
    rho_i: float = rhoi,
    dt_yr: float = 0.05,
    vel_smooth_sigma_m: float | None = 3000.0,
    vdiv_clip: float | None = 0.2,
    min_epochs: int = 4,
    min_span_yr: float = 1.0,
    min_thickness_m: float = 1.0,
    robust: bool = True,
    robust_c: float = 4.685,
    robust_max_iter: int = 8,
    robust_tol: float = 1e-3,
    seed_stride: int = 1,
    seed_block: int = 200_000,
    progress_interval_s: float = 30.0,
) -> xr.Dataset:
    r"""Basal melt rate via a per-parcel, strain-exact time-series LSQ.

    For a parcel with footprint stretch
    :math:`s(t) = \exp\int \nabla\!\cdot u\,dt'` along its trajectory,
    column mass conservation gives

    .. math::
        \frac{d}{dt}\bigl(H s\bigr) = s\,(\dot a + \dot b)

    so the stretch-weighted thickness :math:`y = H s` minus the
    accumulated stretched SMB :math:`A(t) = \int \dot a\,s\,dt'` is
    **linear** in the stretched time :math:`\tau(t) = \int s\,dt'` with
    slope exactly :math:`\dot b` (constant-rate assumption). One parcel =
    one robust regression over *every* DEM epoch its trajectory crosses —
    strain is handled multiplicatively (no small-strain linearization),
    and per-strip elevation blunders are downweighted by Tukey IRLS
    instead of entering endpoint pair differences.

    Regresses along the path instead of the per-pair endpoint
    differencing of :func:`lagrangian_melt_rate`. The recovered
    :math:`\dot b` is attributed to the parcel's **seed cell at the window
    start** (origin product; no path smearing).

    Parameters
    ----------
    h_stack, vx, vy, a_dot, d, rho_w, rho_i
        As in :func:`lagrangian_melt_rate`. Time-varying velocity (>1
        ``time`` entries) is sampled along the trajectory.
    dt_yr : float
        Trajectory integration sub-step, years.
    vel_smooth_sigma_m : float or None
        NaN-aware Gaussian smoothing of ``vx, vy`` before divergence
        Default 3000 m; ``None``/``<=0`` disables.
    vdiv_clip : float or None
        Hard clip on :math:`\nabla\cdot u` (e.g. ±0.2 /yr).
    min_epochs : int
        Minimum finite thickness observations along the trajectory.
    min_span_yr : float
        Minimum time span (first-to-last valid observation).
    min_thickness_m : float
        Observations with :math:`H \le` this are masked (open water /
        blunders).
    robust : bool
        Tukey-biweight IRLS on the per-parcel regression residuals.
    seed_stride, seed_block : int
        Seed decimation and per-block seed count (memory bound).
    progress_interval_s : float
        Narration cadence (this is a long-running stage; it narrates).

    Returns
    -------
    xarray.Dataset
        ``melt_rate`` (m ice/yr, negative = melt), ``count`` (epochs used),
        ``span_yr``, ``rmse`` (residual std, m), ``stderr`` (slope
        standard error, m ice/yr — a principled per-pixel quality gate),
        ``H_f_mean``, ``flux_div``, ``a_dot``.
    """
    if dt_yr <= 0:
        raise ValueError(f"dt_yr must be > 0, got {dt_yr}")
    if seed_stride < 1:
        raise ValueError(f"seed_stride must be >= 1, got {seed_stride}")
    if min_epochs < 3:
        raise ValueError(f"min_epochs must be >= 3, got {min_epochs}")

    # --- input hygiene (see design note literature/plan_lagrangian_parcel_lsq.md)
    vx = _smooth_velocity_da(vx, vel_smooth_sigma_m)
    vy = _smooth_velocity_da(vy, vel_smooth_sigma_m)

    # Hydrostatic inversion at the observation location: H = gamma (h - d).
    H_f_stack_xr = freeboard_to_thickness(h_stack, d=d, rho_w=rho_w, rho_i=rho_i)
    H_f_mean_xr = H_f_stack_xr.mean("time", skipna=True)

    time_varying_vel = "time" in vx.dims and vx.sizes["time"] > 1
    vx_mean_xr = vx.mean("time", skipna=True) if "time" in vx.dims else vx
    vy_mean_xr = vy.mean("time", skipna=True) if "time" in vy.dims else vy
    vdiv_xr = divergence(vx_mean_xr, vy_mean_xr)

    x_coords = H_f_stack_xr["x"].values
    y_coords = H_f_stack_xr["y"].values
    ny = H_f_stack_xr.sizes["y"]
    nx = H_f_stack_xr.sizes["x"]
    res_x = float(x_coords[1] - x_coords[0])
    res_y = float(y_coords[0] - y_coords[1])

    H_f_stack = asarray(np.asarray(H_f_stack_xr.values, dtype=np.float64))
    vx_arr = asarray(np.asarray(vx_mean_xr.values, dtype=np.float64))
    vy_arr = asarray(np.asarray(vy_mean_xr.values, dtype=np.float64))
    vdiv_arr = asarray(np.asarray(vdiv_xr.values, dtype=np.float64))
    if vdiv_clip is not None:
        vdiv_arr = xp.clip(vdiv_arr, -float(vdiv_clip), float(vdiv_clip))

    if isinstance(a_dot, (int, float)):
        a_dot_arr = xp.full((ny, nx), float(a_dot), dtype=xp.float64)
    else:
        a_dot_arr = asarray(
            np.asarray(a_dot.broadcast_like(H_f_mean_xr).values, dtype=np.float64)
        )

    times = pd.to_datetime(H_f_stack_xr["time"].values)
    t_years = np.array(
        [(t - times[0]).total_seconds() / SECONDS_PER_YEAR for t in times],
        dtype=np.float64,
    )
    T = len(times)

    if time_varying_vel:
        vx_sorted = vx.sortby("time")
        vy_sorted = vy.sortby("time")
        v_times = pd.to_datetime(vx_sorted["time"].values)
        v_t_years = np.array(
            [(t - times[0]).total_seconds() / SECONDS_PER_YEAR for t in v_times],
            dtype=np.float64,
        )
        nv = int(v_t_years.size)
        vx_tv = asarray(np.asarray(vx_sorted.values, dtype=np.float64))
        vy_tv = asarray(np.asarray(vy_sorted.values, dtype=np.float64))
        vdiv_tv = asarray(
            np.stack(
                [
                    divergence(vx_sorted.isel(time=k), vy_sorted.isel(time=k)).values
                    for k in range(nv)
                ]
            ).astype(np.float64)
        )
        if vdiv_clip is not None:
            vdiv_tv = xp.clip(vdiv_tv, -float(vdiv_clip), float(vdiv_clip))

    # March grid: one fixed-dt trajectory per seed spanning the full window.
    # Each DEM epoch snapshots the trajectory state at its nearest step.
    K = max(1, int(np.ceil((t_years[-1] - t_years[0]) / dt_yr)))
    e_step = np.clip(np.round((t_years - t_years[0]) / dt_yr).astype(np.int64), 0, K)

    # Seeds: any-time-coverage cells (parcel products are attributed here).
    finite_any = np.isfinite(to_numpy(H_f_stack)).any(axis=0)
    yy, xx = np.mgrid[0:ny:seed_stride, 0:nx:seed_stride]
    seed_mask = finite_any[yy, xx].ravel()
    y_seed_all = yy.ravel()[seed_mask].astype(np.int64)
    x_seed_all = xx.ravel()[seed_mask].astype(np.int64)
    n_seed_all = int(y_seed_all.size)

    n_cells = ny * nx
    out_slope = np.full(n_cells, np.nan)
    out_stderr = np.full(n_cells, np.nan)
    out_rmse = np.full(n_cells, np.nan)
    out_count = np.zeros(n_cells, dtype=np.int32)
    out_span = np.full(n_cells, np.nan)

    n_blocks = (n_seed_all + seed_block - 1) // seed_block
    if progress_interval_s > 0:
        print(
            f"  parcel-LSQ: {n_seed_all:,} seeds in {n_blocks} block(s), "
            f"{T} epochs over {t_years[-1]:.2f} yr, K={K} steps at dt={dt_yr} yr "
            f"(smooth={vel_smooth_sigma_m or 0} m, vdiv_clip={vdiv_clip}, "
            f"robust={robust})",
            flush=True,
        )

    loop_t0 = time.time()
    last_print = loop_t0
    for b in range(n_blocks):
        sl = slice(b * seed_block, min((b + 1) * seed_block, n_seed_all))
        y_seed = asarray(y_seed_all[sl])
        x_seed = asarray(x_seed_all[sl])
        n_s = int(y_seed.size)

        x_idx = x_seed.astype(xp.float64).copy()
        y_idx = y_seed.astype(xp.float64).copy()
        valid = xp.ones(n_s, dtype=bool)
        ln_s = xp.zeros(n_s, dtype=xp.float64)
        tau = xp.zeros(n_s, dtype=xp.float64)
        A_acc = xp.zeros(n_s, dtype=xp.float64)

        # Per-epoch snapshots (numpy, gathered as the march passes each epoch)
        obs_y = np.full((T, n_s), np.nan)   # H*s - A
        obs_tau = np.full((T, n_s), np.nan)
        obs_ok = np.zeros((T, n_s), dtype=bool)

        def _snapshot(e: int) -> None:
            s_now = xp.exp(ln_s)
            H_e = map_coordinates(
                H_f_stack[e], [y_idx, x_idx], order=1, mode="constant",
                cval=float("nan"),
            )
            ok = valid & xp.isfinite(H_e) & (H_e > float(min_thickness_m))
            y_e = H_e * s_now - A_acc
            obs_y[e] = to_numpy(xp.where(ok, y_e, xp.nan))
            obs_tau[e] = to_numpy(tau)
            obs_ok[e] = to_numpy(ok)

        next_e = 0
        for k in range(K + 1):
            while next_e < T and e_step[next_e] == k:
                _snapshot(next_e)
                next_e += 1
            if k == K:
                break

            # Velocity + divergence at the parcel positions and current time
            if time_varying_vel:
                t_abs = t_years[0] + k * dt_yr
                if t_abs <= v_t_years[0]:
                    k0 = k1 = 0
                    w = 0.0
                elif t_abs >= v_t_years[-1]:
                    k0 = k1 = nv - 1
                    w = 0.0
                else:
                    k1 = int(np.searchsorted(v_t_years, t_abs))
                    k0 = k1 - 1
                    w = float((t_abs - v_t_years[k0]) / (v_t_years[k1] - v_t_years[k0]))
                if k0 == k1:
                    vx_t = map_coordinates(vx_tv[k0], [y_idx, x_idx], order=1, mode="nearest")
                    vy_t = map_coordinates(vy_tv[k0], [y_idx, x_idx], order=1, mode="nearest")
                    dv_t = map_coordinates(vdiv_tv[k0], [y_idx, x_idx], order=1, mode="nearest")
                else:
                    vx_t = (1.0 - w) * map_coordinates(
                        vx_tv[k0], [y_idx, x_idx], order=1, mode="nearest"
                    ) + w * map_coordinates(vx_tv[k1], [y_idx, x_idx], order=1, mode="nearest")
                    vy_t = (1.0 - w) * map_coordinates(
                        vy_tv[k0], [y_idx, x_idx], order=1, mode="nearest"
                    ) + w * map_coordinates(vy_tv[k1], [y_idx, x_idx], order=1, mode="nearest")
                    dv_t = (1.0 - w) * map_coordinates(
                        vdiv_tv[k0], [y_idx, x_idx], order=1, mode="nearest"
                    ) + w * map_coordinates(vdiv_tv[k1], [y_idx, x_idx], order=1, mode="nearest")
            else:
                vx_t = map_coordinates(vx_arr, [y_idx, x_idx], order=1, mode="nearest")
                vy_t = map_coordinates(vy_arr, [y_idx, x_idx], order=1, mode="nearest")
                dv_t = map_coordinates(vdiv_arr, [y_idx, x_idx], order=1, mode="nearest")

            finite_v = xp.isfinite(vx_t) & xp.isfinite(vy_t)
            vx_t = xp.where(finite_v, vx_t, 0.0)
            vy_t = xp.where(finite_v, vy_t, 0.0)
            dv_t = xp.where(xp.isfinite(dv_t), dv_t, 0.0)
            a_t = map_coordinates(a_dot_arr, [y_idx, x_idx], order=1, mode="nearest")
            a_t = xp.where(xp.isfinite(a_t), a_t, 0.0)

            # Accumulate the stretched-time quadrature BEFORE stepping
            # (rectangle rule consistent with forward-Euler advection).
            s_now = xp.exp(ln_s)
            tau = tau + s_now * dt_yr
            A_acc = A_acc + a_t * s_now * dt_yr
            ln_s = ln_s + dv_t * dt_yr

            x_idx = x_idx + vx_t * dt_yr / res_x
            y_idx = y_idx - vy_t * dt_yr / res_y
            in_bounds = (x_idx >= 0) & (x_idx <= nx - 1) & (y_idx >= 0) & (y_idx <= ny - 1)
            valid = valid & in_bounds & finite_v

            if progress_interval_s > 0:
                now = time.time()
                if now - last_print >= progress_interval_s:
                    frac = (b + (k + 1) / (K + 1)) / n_blocks
                    elapsed = now - loop_t0
                    eta = elapsed * (1 - frac) / max(frac, 1e-9)
                    print(
                        f"    block {b + 1}/{n_blocks}  step {k + 1}/{K}  "
                        f"({100 * frac:.1f}%)  elapsed={elapsed / 60:.1f}min  "
                        f"eta={eta / 60:.1f}min",
                        flush=True,
                    )
                    last_print = now
        while next_e < T:
            _snapshot(next_e)
            next_e += 1

        # ---- vectorized robust regression: y = b0 + b1 * tau, per seed ----
        # (all-NaN slices are expected for seeds with no valid observations;
        # they resolve to NaN outputs, so the RuntimeWarnings are suppressed)
        import warnings

        warn_ctx = warnings.catch_warnings()
        warn_ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        m = obs_ok
        n_obs = m.sum(axis=0)
        t_first = np.where(m.any(axis=0), np.nanmin(np.where(m, t_years[:, None], np.nan), axis=0), np.nan)
        t_last = np.where(m.any(axis=0), np.nanmax(np.where(m, t_years[:, None], np.nan), axis=0), np.nan)
        span = t_last - t_first
        fit_ok = (n_obs >= min_epochs) & (span >= min_span_yr)

        w = m.astype(np.float64)
        b1 = np.full(n_s, np.nan)
        b0 = np.full(n_s, np.nan)
        yv = np.where(m, obs_y, 0.0)
        xv = np.where(m, obs_tau, 0.0)
        n_iter = robust_max_iter if robust else 0
        for it in range(n_iter + 1):
            W = w.sum(axis=0)
            W_safe = np.maximum(W, 1e-12)
            xbar = (w * xv).sum(axis=0) / W_safe
            ybar = (w * yv).sum(axis=0) / W_safe
            dxv = xv - xbar
            Sxx = (w * dxv * dxv).sum(axis=0)
            Sxy = (w * dxv * (yv - ybar)).sum(axis=0)
            b1_new = np.where(Sxx > 1e-12, Sxy / np.maximum(Sxx, 1e-12), np.nan)
            b0_new = ybar - b1_new * xbar
            if it > 0 and np.nanmax(
                np.abs(b1_new - b1) / np.maximum(np.abs(b1), 1e-9)
            ) < robust_tol:
                b1, b0 = b1_new, b0_new
                break
            b1, b0 = b1_new, b0_new
            if it == n_iter:
                break
            # Tukey biweight on residuals, scale = per-seed normal MAD
            r = np.where(m, obs_y - (b0 + b1 * obs_tau), np.nan)
            scale = 1.4826 * np.nanmedian(np.abs(r), axis=0)
            scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, np.nan)
            u = r / (robust_c * scale)
            w = np.where(
                m & np.isfinite(u) & (np.abs(u) < 1.0),
                (1.0 - u**2) ** 2,
                0.0,
            )
            # Seeds whose scale collapsed (perfect fit) keep OLS weights
            w = np.where(np.isfinite(scale)[None, :], w, m.astype(np.float64))

        # Residual diagnostics at the final weights
        r = np.where(m, obs_y - (b0 + b1 * obs_tau), np.nan)
        W = w.sum(axis=0)
        wr2 = np.nansum(w * np.where(m, r, 0.0) ** 2, axis=0)
        n_eff = np.maximum(W, 1e-12)
        rmse_fit = np.sqrt(wr2 / n_eff)
        dxv = np.where(m, obs_tau, 0.0) - (
            (w * np.where(m, obs_tau, 0.0)).sum(axis=0) / n_eff
        )
        Sxx = (w * dxv * dxv).sum(axis=0)
        dof = np.maximum(n_obs - 2, 1)
        stderr = np.sqrt((wr2 / dof) / np.maximum(Sxx, 1e-12))

        b1 = np.where(fit_ok, b1, np.nan)
        warn_ctx.__exit__(None, None, None)
        flat = to_numpy(y_seed).astype(np.int64) * nx + to_numpy(x_seed).astype(np.int64)
        out_slope[flat] = b1
        out_stderr[flat] = np.where(fit_ok, stderr, np.nan)
        out_rmse[flat] = np.where(fit_ok, rmse_fit, np.nan)
        out_count[flat] = n_obs.astype(np.int32)
        out_span[flat] = span

    if progress_interval_s > 0:
        print(
            f"  parcel-LSQ done: {n_seed_all:,} seeds in "
            f"{(time.time() - loop_t0) / 60:.1f} min; "
            f"{int(np.isfinite(out_slope).sum()):,} cells fit",
            flush=True,
        )

    dims = ("y", "x")
    coords = {"y": y_coords, "x": x_coords}
    fd = flux_divergence(H_f_mean_xr, vx_mean_xr, vy_mean_xr)
    a_dot_field_xr = xr.DataArray(a_dot) if not isinstance(a_dot, xr.DataArray) else a_dot
    a_dot_field_xr = a_dot_field_xr.broadcast_like(H_f_mean_xr)

    return xr.Dataset(
        {
            "melt_rate": (dims, out_slope.reshape(ny, nx)),
            "stderr": (dims, out_stderr.reshape(ny, nx)),
            "rmse": (dims, out_rmse.reshape(ny, nx)),
            "count": (dims, out_count.reshape(ny, nx)),
            "span_yr": (dims, out_span.reshape(ny, nx)),
            "H_f_mean": H_f_mean_xr,
            "flux_div": fd,
            "a_dot": a_dot_field_xr,
        },
        coords=coords,
        attrs={
            "equation": (
                "d(H s)/dt = s (a_dot + b_dot); regress y=H*s - int(a s dt) "
                "on tau=int(s dt) per parcel; slope = b_dot"
            ),
            "units": "m ice yr^-1; Shean convention: negative melt_rate = melt, positive = accretion",
            "rho_w": rho_w,
            "rho_i": rho_i,
            "integrator": "forward-Euler",
            "dt_yr": dt_yr,
            "vel_smooth_sigma_m": float(vel_smooth_sigma_m or 0.0),
            "vdiv_clip": float(vdiv_clip) if vdiv_clip is not None else 0.0,
            "min_epochs": min_epochs,
            "min_span_yr": min_span_yr,
            "robust": int(robust),
            "seed_stride": seed_stride,
            "velocity_time_varying": int(time_varying_vel),
        },
    )
