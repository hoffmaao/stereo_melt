# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.

r"""Leading-edge maximum-gradient (LMG) retracker for CryoSat-2 SARIn.

Implements the waveform-range step of Nilsson et al. 2016 (TC 10:2953;
see ``reference_nilsson2016_retracker``) — the retracker Zinck/BURGEE and
the JPL/captoolkit lineage use to turn L1B SARIn waveforms into surface
ranges. This module does **only** the retracking (range), not the
interferometric POCA relocation (geolocation); that is the deliberate
scope of the "floor test" (see ``project`` notes): isolate the
retracker's contribution to the CS2-vs-IS2 elevation agreement while
holding ESA's geolocation + all geophysical corrections fixed.

Two retrackers are provided:

- :func:`retrack_lmg` — the Nilsson LMG: oversample the leading edge with
  a monotone (PCHIP) spline, take the gate of **maximum gradient**. Per
  Wingham 2006a the mean-surface return on a delay/Doppler waveform sits
  near the waveform maximum, not the half-power point, so the steepest
  point of the leading edge is the physically-motivated retrack gate.
- :func:`retrack_threshold` — a TFMRA-style fixed-fraction retracker,
  used as the calibration cross-check against ESA's operational range.

Range model
-----------
Range to waveform gate ``g`` (one-way, metres)::

    range(g) = 0.5 * c * window_del + (g - g_ref) * bin_range + range_cor

where ``g_ref`` is the gate the window delay references (window centre;
512 for the 1024-bin Baseline-D SARIn echo) and ``bin_range`` is the
range increment per gate. ``bin_range`` is **calibrated empirically**
from matched ESA ranges (see :func:`calibrate_bin_range`) rather than
trusted from spec, so the floor test does not hinge on a documented
constant.

The floor-test elevation anchors on ESA's own fully-corrected product::

    h_lmg = height_1 + (range_1 - range_lmg)

so altitude, tropo/iono/tide/geoid corrections, and the POCA relocation
all cancel — only the retracker gate difference remains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

#: Speed of light (m/s).
C_LIGHT = 299_792_458.0
#: Default reference gate for the 1024-bin Baseline-D SARIn echo (window
#: centre). Only enters as an additive constant in the range model; a
#: half-gate error is ~6 cm and is absorbed by the empirical calibration.
DEFAULT_G_REF = 512.0


# ---------------------------------------------------------------------------
# Waveform preconditioning
# ---------------------------------------------------------------------------


def _noise_floor(wf: np.ndarray, n_early: int) -> float:
    r"""Robust noise estimate: mean of the lowest-power early gates.

    Uses the lower half (by value) of the first ``n_early`` gates so a
    stray bright gate near window start doesn't inflate the floor.
    """
    head = wf[:n_early]
    thr = np.median(head)
    low = head[head <= thr]
    return float(low.mean()) if low.size else float(head.mean())


def _leading_edge_bounds(
    wf: np.ndarray, noise: float, peak_idx: int, start_frac: float
) -> tuple[int, int]:
    r"""Index bounds ``[le_start, peak_idx]`` of the first leading edge.

    Walk back from the first-return peak to the last gate at or below
    ``noise + start_frac * (peak - noise)``.
    """
    amp = wf[peak_idx] - noise
    thr = noise + start_frac * amp
    le_start = peak_idx
    while le_start > 0 and wf[le_start - 1] > thr:
        le_start -= 1
    return max(le_start - 1, 0), peak_idx


def _first_return_peak(wf: np.ndarray, noise: float, peak_frac: float) -> int:
    r"""Index of the first significant local maximum (the surface return).

    The global max can fall on a volume-scattering tail; the *first*
    return is the surface. Returns the first local maximum whose
    amplitude exceeds ``peak_frac`` of the global amplitude, else the
    global argmax.
    """
    gmax = int(np.argmax(wf))
    amp = wf[gmax] - noise
    if amp <= 0:
        return gmax
    thr = noise + peak_frac * amp
    rising = wf[1:] < wf[:-1]  # wf[i] is a peak if wf[i] >= wf[i+1]
    for i in range(1, wf.size - 1):
        if wf[i] >= thr and wf[i] >= wf[i - 1] and wf[i] >= wf[i + 1]:
            return i
    return gmax


# ---------------------------------------------------------------------------
# Retrackers
# ---------------------------------------------------------------------------


@dataclass
class RetrackResult:
    gate: float          # sub-gate retrack index, NaN if rejected
    snr: float           # peak / noise
    le_start: float      # leading-edge start gate (diagnostic)
    peak: float          # first-return peak gate (diagnostic)
    status: str          # "ok" or rejection reason


def retrack_lmg(
    wf: np.ndarray,
    *,
    oversample: int = 10,
    n_early: int = 60,
    snr_min: float = 6.0,
    gate_lo: int = 40,
    gate_hi: int | None = None,
    le_start_frac: float = 0.05,
    peak_frac: float = 0.30,
    smooth: int = 3,
) -> RetrackResult:
    r"""Leading-edge maximum-gradient retrack of one SARIn power waveform.

    Parameters
    ----------
    wf
        1-D power waveform (counts or watts; scale-invariant).
    oversample
        PCHIP oversampling factor on the leading edge.
    n_early
        Number of leading gates used for the noise-floor estimate.
    snr_min
        Reject waveforms with ``peak/noise`` below this.
    gate_lo, gate_hi
        Reject if the first-return peak falls outside this gate window
        (off-window / edge returns). ``gate_hi`` defaults to ``N - 40``.
    le_start_frac
        Leading edge starts at ``noise + le_start_frac*(peak-noise)``.
    peak_frac
        First-return detection threshold (fraction of amplitude).
    smooth
        Boxcar smoothing width applied before retracking (gates).

    Returns
    -------
    RetrackResult
        ``gate`` is the maximum-gradient gate (sub-gate), NaN if rejected.
    """
    n = wf.size
    if gate_hi is None:
        gate_hi = n - 40

    w = np.asarray(wf, dtype=np.float64)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        w = np.convolve(w, k, mode="same")

    noise = _noise_floor(w, n_early)
    peak_idx = _first_return_peak(w, noise, peak_frac)
    peak = w[peak_idx]
    amp = peak - noise
    if amp <= 0:
        return RetrackResult(np.nan, 0.0, np.nan, float(peak_idx), "flat")
    snr = peak / max(noise, 1e-6)
    if peak_idx < gate_lo or peak_idx > gate_hi:
        return RetrackResult(np.nan, snr, np.nan, float(peak_idx), "peak_oob")
    if snr < snr_min:
        return RetrackResult(np.nan, snr, np.nan, float(peak_idx), "low_snr")

    le_start, le_end = _leading_edge_bounds(w, noise, peak_idx, le_start_frac)
    if le_end - le_start < 2:
        return RetrackResult(np.nan, snr, float(le_start), float(peak_idx), "short_le")

    x = np.arange(le_start, le_end + 1, dtype=np.float64)
    seg = w[le_start:le_end + 1]
    xf = np.linspace(le_start, le_end, (seg.size - 1) * oversample + 1)
    yf = PchipInterpolator(x, seg)(xf)
    grad = np.gradient(yf, xf)
    gate = float(xf[int(np.argmax(grad))])
    return RetrackResult(gate, snr, float(le_start), float(peak_idx), "ok")


def retrack_threshold(
    wf: np.ndarray,
    *,
    thresh: float = 0.5,
    oversample: int = 10,
    n_early: int = 60,
    snr_min: float = 6.0,
    gate_lo: int = 40,
    gate_hi: int | None = None,
    peak_frac: float = 0.30,
    smooth: int = 3,
) -> RetrackResult:
    r"""TFMRA-style fixed-fraction leading-edge retracker (calibration ref).

    Retrack gate = first crossing of ``noise + thresh*(peak-noise)`` on
    the oversampled leading edge. Used to (a) empirically calibrate the
    range bin size against ESA's ``range_1`` and (b) cross-check the LMG
    gate. ``thresh=0.5`` is the classic half-power point.
    """
    n = wf.size
    if gate_hi is None:
        gate_hi = n - 40

    w = np.asarray(wf, dtype=np.float64)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        w = np.convolve(w, k, mode="same")

    noise = _noise_floor(w, n_early)
    peak_idx = _first_return_peak(w, noise, peak_frac)
    peak = w[peak_idx]
    amp = peak - noise
    if amp <= 0:
        return RetrackResult(np.nan, 0.0, np.nan, float(peak_idx), "flat")
    snr = peak / max(noise, 1e-6)
    if peak_idx < gate_lo or peak_idx > gate_hi:
        return RetrackResult(np.nan, snr, np.nan, float(peak_idx), "peak_oob")
    if snr < snr_min:
        return RetrackResult(np.nan, snr, np.nan, float(peak_idx), "low_snr")

    le_start, le_end = _leading_edge_bounds(w, noise, peak_idx, 0.01)
    if le_end - le_start < 2:
        return RetrackResult(np.nan, snr, float(le_start), float(peak_idx), "short_le")

    level = noise + thresh * amp
    x = np.arange(le_start, le_end + 1, dtype=np.float64)
    seg = w[le_start:le_end + 1]
    xf = np.linspace(le_start, le_end, (seg.size - 1) * oversample + 1)
    yf = PchipInterpolator(x, seg)(xf)
    above = np.where(yf >= level)[0]
    if above.size == 0:
        return RetrackResult(np.nan, snr, float(le_start), float(peak_idx), "no_cross")
    gate = float(xf[above[0]])
    return RetrackResult(gate, snr, float(le_start), float(peak_idx), "ok")


def retrack_ocog(
    wf: np.ndarray,
    *,
    n_early: int = 60,
    snr_min: float = 6.0,
    gate_lo: int = 40,
    gate_hi: int | None = None,
    smooth: int = 3,
    coh: np.ndarray | None = None,  # unused; uniform registry signature
) -> RetrackResult:
    r"""Offset-Centre-of-Gravity retracker (Wingham et al. 1986).

    Amplitude-and-width statistics over the whole (noise-subtracted) echo,
    with the retrack point at the OCOG leading edge ``COG - W/2``. Uses no
    leading-edge peak detection, so it is robust to the multi-peak / wrong-
    peak failures that trip the threshold and LMG retrackers on rough
    grounded terrain — at the cost of being pulled by trailing energy.
    """
    n = wf.size
    if gate_hi is None:
        gate_hi = n - 40
    w = np.asarray(wf, dtype=np.float64)
    if smooth > 1:
        w = np.convolve(w, np.ones(smooth) / smooth, mode="same")
    noise = _noise_floor(w, n_early)
    p = np.clip(w - noise, 0.0, None)
    idx = np.arange(n, dtype=np.float64)
    win = (idx >= gate_lo) & (idx <= gate_hi)
    p2 = p[win] ** 2
    p4 = p[win] ** 4
    gg = idx[win]
    s2 = float(p2.sum())
    s4 = float(p4.sum())
    if s2 <= 0.0 or s4 <= 0.0:
        return RetrackResult(np.nan, 0.0, np.nan, np.nan, "flat")
    amp = np.sqrt(s4 / s2)              # OCOG amplitude
    width = (s2 * s2) / s4              # OCOG width (gates)
    cog = float((gg * p2).sum() / s2)   # centre of gravity
    snr = (amp + noise) / max(noise, 1e-6)
    if snr < snr_min:
        return RetrackResult(np.nan, snr, np.nan, cog, "low_snr")
    lep = cog - 0.5 * width            # leading-edge position
    if lep < gate_lo or lep > gate_hi:
        return RetrackResult(np.nan, snr, float(lep), cog, "peak_oob")
    return RetrackResult(float(lep), snr, float(lep), cog, "ok")


def _first_coherent_peak(
    w: np.ndarray, coh: np.ndarray, noise: float, peak_frac: float,
    coh_min: float, gate_lo: int, gate_hi: int, run: int,
) -> int:
    r"""First local maximum that is a *coherent* surface return.

    Unlike :func:`_first_return_peak` (amplitude-only, which skips an early
    moderate return when a later facet is brighter — the wrong-peak failure
    mode on rough grounded ice), this requires sustained interferometric
    coherence ``> coh_min`` around the peak. The first such return is the
    POCA-consistent surface echo regardless of a brighter later peak.
    """
    amp = w[int(np.argmax(w))] - noise
    if amp <= 0:
        return int(np.argmax(w))
    thr = noise + peak_frac * amp
    for i in range(max(gate_lo, 1), min(gate_hi, w.size - 1)):
        if w[i] >= thr and w[i] >= w[i - 1] and w[i] >= w[i + 1]:
            lo = max(0, i - run)
            hi = min(w.size, i + run + 1)
            if np.median(coh[lo:hi]) >= coh_min:
                return i
    return int(np.argmax(w))


def retrack_lmg_coherent(
    wf: np.ndarray,
    coh: np.ndarray,
    *,
    oversample: int = 10,
    n_early: int = 60,
    snr_min: float = 6.0,
    coh_min: float = 0.7,
    gate_lo: int = 40,
    gate_hi: int | None = None,
    le_start_frac: float = 0.05,
    peak_frac: float = 0.15,
    smooth: int = 3,
    run: int = 3,
) -> RetrackResult:
    r"""Coherence-guided leading-edge max-gradient retrack of a SARIn echo.

    Same max-gradient retrack as :func:`retrack_lmg`, but the first-return
    detection is gated on interferometric **coherence** (which the plain
    LMG ignores). On rough grounded ice the plain LMG can latch onto a
    later, brighter off-nadir facet; restricting to the first *coherent*
    return picks the POCA-consistent surface echo, and rejecting low-
    coherence retrack gates screens ambiguous waveforms outright.
    """
    n = wf.size
    if gate_hi is None:
        gate_hi = n - 40
    w = np.asarray(wf, dtype=np.float64)
    c = np.asarray(coh, dtype=np.float64)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        w = np.convolve(w, k, mode="same")
        c = np.convolve(c, k, mode="same")
    noise = _noise_floor(w, n_early)
    peak_idx = _first_coherent_peak(w, c, noise, peak_frac, coh_min,
                                    gate_lo, gate_hi, run)
    peak = w[peak_idx]
    amp = peak - noise
    if amp <= 0:
        return RetrackResult(np.nan, 0.0, np.nan, float(peak_idx), "flat")
    snr = peak / max(noise, 1e-6)
    if peak_idx < gate_lo or peak_idx > gate_hi:
        return RetrackResult(np.nan, snr, np.nan, float(peak_idx), "peak_oob")
    if snr < snr_min:
        return RetrackResult(np.nan, snr, np.nan, float(peak_idx), "low_snr")
    le_start, le_end = _leading_edge_bounds(w, noise, peak_idx, le_start_frac)
    if le_end - le_start < 2:
        return RetrackResult(np.nan, snr, float(le_start), float(peak_idx), "short_le")
    x = np.arange(le_start, le_end + 1, dtype=np.float64)
    seg = w[le_start:le_end + 1]
    xf = np.linspace(le_start, le_end, (seg.size - 1) * oversample + 1)
    yf = PchipInterpolator(x, seg)(xf)
    grad = np.gradient(yf, xf)
    gate = float(xf[int(np.argmax(grad))])
    if c[int(round(gate))] < coh_min:
        return RetrackResult(np.nan, snr, float(le_start), float(peak_idx), "low_coh")
    return RetrackResult(gate, snr, float(le_start), float(peak_idx), "ok")


# ---------------------------------------------------------------------------
# Range / elevation model
# ---------------------------------------------------------------------------


def range_from_gate(
    window_del: np.ndarray,
    gate: np.ndarray,
    range_cor: np.ndarray,
    *,
    bin_range_m: float,
    g_ref: float = DEFAULT_G_REF,
    intercept_m: float = 0.0,
) -> np.ndarray:
    r"""One-way range (m) to retrack ``gate`` from the L1B window delay.

    ``intercept_m`` is the empirically-calibrated constant (see
    :func:`calibrate_bin_range`) that aligns this range model's absolute
    datum to ESA's ``range_1``. It absorbs the difference between the
    assumed reference gate ``g_ref`` (window centre) and the gate ESA's
    window delay actually references — a ~36-gate / ~6 m convention offset
    for Baseline-D SARIn, *not* a retracker effect. Pass ``cal.intercept_m``
    so the LMG and threshold ranges sit on ESA's scale; only the gate
    *difference* between the two retrackers then survives into the elevation.
    """
    return (0.5 * C_LIGHT * np.asarray(window_del)
            + (np.asarray(gate) - g_ref) * bin_range_m
            + np.asarray(range_cor)
            + intercept_m)


@dataclass
class BinRangeCalibration:
    bin_range_m: float       # fitted range per gate (m)
    intercept_m: float       # ESA-vs-threshold gate offset folded in (m)
    resid_mad_m: float       # robustness of the linear fit (m)
    n: int                   # records used


def calibrate_bin_range(
    range_1: np.ndarray,
    window_del: np.ndarray,
    range_cor: np.ndarray,
    gate_le: np.ndarray,
    *,
    g_ref: float = DEFAULT_G_REF,
) -> BinRangeCalibration:
    r"""Empirically fit the range bin size from matched ESA ranges.

    For each matched record the ESA range satisfies
    ``range_1 - 0.5 c window_del - range_cor = bin_range*(g_esa - g_ref)``.
    With ``g_esa`` tracking the detected leading-edge gate ``gate_le`` up
    to a roughly-constant retracker offset, a robust line through
    ``(gate_le - g_ref, LHS)`` recovers ``bin_range`` (slope) — no spec
    constant trusted. The intercept absorbs the ESA-vs-threshold offset.
    """
    lhs = (np.asarray(range_1, dtype=np.float64)
           - 0.5 * C_LIGHT * np.asarray(window_del, dtype=np.float64)
           - np.asarray(range_cor, dtype=np.float64))
    xg = np.asarray(gate_le, dtype=np.float64) - g_ref
    good = np.isfinite(lhs) & np.isfinite(xg)
    lhs, xg = lhs[good], xg[good]
    if lhs.size < 10:
        return BinRangeCalibration(np.nan, np.nan, np.nan, int(lhs.size))
    # Robust-ish: ordinary LS then one MAD-clipped refit.
    A = np.column_stack([xg, np.ones_like(xg)])
    slope, intercept = np.linalg.lstsq(A, lhs, rcond=None)[0]
    resid = lhs - (slope * xg + intercept)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    keep = np.abs(resid - np.median(resid)) <= 5.0 * (mad + 1e-9)
    if keep.sum() >= 10:
        slope, intercept = np.linalg.lstsq(A[keep], lhs[keep], rcond=None)[0]
        resid = lhs[keep] - (slope * xg[keep] + intercept)
        mad = float(np.median(np.abs(resid - np.median(resid))))
    return BinRangeCalibration(float(slope), float(intercept), mad, int(keep.sum()))


def h_lmg_from_anchor(
    height_1: np.ndarray,
    range_1: np.ndarray,
    range_lmg: np.ndarray,
) -> np.ndarray:
    r"""Floor-test elevation: shift ESA's height by the range difference.

    ``h_lmg = height_1 + (range_1 - range_lmg)``. Everything ESA applied
    (altitude, geophysical corrections, POCA relocation) cancels; only the
    retracker gate difference moves the surface.
    """
    return (np.asarray(height_1, dtype=np.float64)
            + (np.asarray(range_1, dtype=np.float64)
               - np.asarray(range_lmg, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Application: match L1B waveforms to L2 POCA points and re-retrack
# ---------------------------------------------------------------------------


@dataclass
class LmgElevationResult:
    gate_lmg: np.ndarray     # LMG retrack gate per L2 point (NaN if unmatched/bad)
    gate_thr: np.ndarray     # 50% threshold gate (calibration reference)
    snr: np.ndarray
    status: np.ndarray       # per-point retrack/match status
    window_del: np.ndarray
    range_cor: np.ndarray
    range_lmg: np.ndarray
    range_thr: np.ndarray
    h_lmg: np.ndarray        # LMG floor-test elevation (anchored on height_1)
    h_thr: np.ndarray        # threshold elevation (validation: ~= height_1)
    calibration: BinRangeCalibration
    n_matched: int


def build_lmg_elevations(
    t_tai: np.ndarray,
    range_esa: np.ndarray,
    h_esa: np.ndarray,
    l1b_paths,
    *,
    time_tol_s: float = 0.01,
    g_ref: float = DEFAULT_G_REF,
    thresh: float = 0.5,
    verbose: bool = True,
) -> LmgElevationResult:
    r"""Re-retrack L2 POCA points with LMG by matching their L1B waveforms.

    For each L2 POCA point (given its TAI timestamp ``t_tai``, ESA range
    ``range_esa``, and ESA height ``h_esa``), find the L1B record with the
    same timestamp, retrack the waveform with both the LMG and a 50%
    threshold retracker, then form the floor-test elevation
    ``h_lmg = h_esa + (range_esa - range_lmg)``.

    Streams over ``l1b_paths`` one granule at a time (waveforms are large)
    and matches by timestamp via :func:`numpy.searchsorted`. The range bin
    size is calibrated empirically (slope of ``range_esa`` vs the threshold
    gate; see :func:`calibrate_bin_range`) and the window reference gate
    ``g_ref`` is held at the 1024-bin centre.

    Parameters
    ----------
    t_tai, range_esa, h_esa
        Per-L2-point TAI seconds (since 2000), ESA retracker-1 range (m),
        and ESA height_1 (m). Same length, the output axis.
    l1b_paths
        Iterable of L1B SARIn granule paths covering the point timestamps.
    time_tol_s
        Max ``|Δt|`` for an L1B↔L2 record match (20 Hz spacing is 0.05 s;
        the two products share a clock so the true match is ~0).

    Returns
    -------
    LmgElevationResult
    """
    t_tai = np.asarray(t_tai, dtype=np.float64)
    range_esa = np.asarray(range_esa, dtype=np.float64)
    h_esa = np.asarray(h_esa, dtype=np.float64)
    n = t_tai.size

    gate_lmg = np.full(n, np.nan)
    gate_thr = np.full(n, np.nan)
    snr = np.full(n, np.nan)
    window_del = np.full(n, np.nan)
    range_cor = np.full(n, np.nan)
    status = np.full(n, "unmatched", dtype=object)

    order = np.argsort(t_tai)
    ts = t_tai[order]

    paths = list(l1b_paths)
    n_matched = 0
    for gi, path in enumerate(paths):
        from ..io.cs2_l1b import read_cs2_l1b_sarin_granule
        g = read_cs2_l1b_sarin_granule(path)
        if g is None or g["time"].size == 0:
            continue
        lt = g["time"]
        lo = np.searchsorted(ts, lt.min() - time_tol_s, side="left")
        hi = np.searchsorted(ts, lt.max() + time_tol_s, side="right")
        if hi <= lo:
            continue
        lorder = np.argsort(lt)
        lts = lt[lorder]
        for si in range(lo, hi):
            oi = order[si]
            tt = ts[si]
            k = np.searchsorted(lts, tt)
            best_li = -1
            best_dt = time_tol_s
            for kk in (k - 1, k):
                if 0 <= kk < lts.size:
                    dt = abs(lts[kk] - tt)
                    if dt < best_dt:
                        best_dt = dt
                        best_li = int(lorder[kk])
            if best_li < 0:
                continue
            wf = g["pwr"][best_li]
            rl = retrack_lmg(wf)
            rt = retrack_threshold(wf, thresh=thresh)
            gate_lmg[oi] = rl.gate
            gate_thr[oi] = rt.gate
            snr[oi] = rl.snr
            window_del[oi] = g["window_del"][best_li]
            range_cor[oi] = g["range_cor"][best_li]
            status[oi] = rl.status
            n_matched += 1
        if verbose and (gi + 1) % 50 == 0:
            print(f"    … {gi + 1}/{len(paths)} L1B granules scanned, "
                  f"{n_matched} L2 points matched", flush=True)

    # Calibrate the range bin size AND the absolute-datum intercept from the
    # threshold gate (independent of LMG) so the range scale is not tuned to
    # the retracker under test. The intercept absorbs the reference-gate
    # convention offset (ESA's window delay references a tracking gate ~36
    # gates off the window centre); applying it to BOTH ranges pins the
    # threshold elevation to ESA's height_1, leaving only the LMG-vs-threshold
    # gate difference in h_lmg. Without it, a spurious ~6 m datum nuisance
    # would swamp the retracker signal in the IS2 floor test.
    cal_mask = (np.isfinite(gate_thr) & np.isfinite(window_del)
                & np.isfinite(range_esa))
    cal = calibrate_bin_range(
        range_esa[cal_mask], window_del[cal_mask],
        range_cor[cal_mask], gate_thr[cal_mask], g_ref=g_ref,
    )
    b = cal.bin_range_m
    range_lmg = range_from_gate(window_del, gate_lmg, range_cor,
                                bin_range_m=b, g_ref=g_ref,
                                intercept_m=cal.intercept_m)
    range_thr = range_from_gate(window_del, gate_thr, range_cor,
                                bin_range_m=b, g_ref=g_ref,
                                intercept_m=cal.intercept_m)
    h_lmg = h_lmg_from_anchor(h_esa, range_esa, range_lmg)
    h_thr = h_lmg_from_anchor(h_esa, range_esa, range_thr)

    if verbose:
        print(f"  matched {n_matched}/{n} L2 points to L1B waveforms; "
              f"bin_range={b:.4f} m (fit MAD {cal.resid_mad_m:.2f} m, "
              f"n={cal.n})")

    return LmgElevationResult(
        gate_lmg=gate_lmg, gate_thr=gate_thr, snr=snr, status=status,
        window_del=window_del, range_cor=range_cor,
        range_lmg=range_lmg, range_thr=range_thr,
        h_lmg=h_lmg, h_thr=h_thr, calibration=cal, n_matched=n_matched,
    )


# ---------------------------------------------------------------------------
# Retracker suite — bake-off harness
# ---------------------------------------------------------------------------

#: Registry of retrackers, each ``(wf, coh) -> RetrackResult``. Coherence is
#: passed to every retracker for a uniform signature; power-only methods
#: ignore it. The threshold family spans the leading edge (0.25 toe → 0.80
#: near-peak); ``ocog`` is amplitude/width; ``lmg`` is our max-gradient;
#: ``lmg_coh`` adds SARIn coherence gating.
RETRACKERS = {
    # Threshold family spanning the literature sweet spot. Schröder 2019 use
    # 10% for flat topography (lowest penetration sensitivity); Xiao 2017 find
    # 25% best for SARIn; Aublanc 2021 use 50% for their SARIn POCA chain. We
    # span 10-50% to find OUR optimum vs IS2; tfmra80 kept only as a high-side
    # control (the lit shies away from >50% — elevation underestimated there).
    "tfmra10": lambda wf, coh: retrack_threshold(wf, thresh=0.10),
    "tfmra25": lambda wf, coh: retrack_threshold(wf, thresh=0.25),
    "tfmra40": lambda wf, coh: retrack_threshold(wf, thresh=0.40),
    "tfmra50": lambda wf, coh: retrack_threshold(wf, thresh=0.50),
    "tfmra80": lambda wf, coh: retrack_threshold(wf, thresh=0.80),
    "lmg": lambda wf, coh: retrack_lmg(wf),
    "ocog": lambda wf, coh: retrack_ocog(wf),
    "lmg_coh": lambda wf, coh: retrack_lmg_coherent(wf, coh),
}


@dataclass
class MultiRetrackResult:
    names: list              # retracker names (incl. "ensemble" if built)
    gates: dict              # name -> (n,) sub-gate, NaN if rejected
    heights: dict            # name -> (n,) floor-test elevation (m)
    status: dict             # name -> (n,) status string
    window_del: np.ndarray
    range_cor: np.ndarray
    range_esa: np.ndarray
    calibration: BinRangeCalibration
    n_matched: int


def build_multi_retracker_elevations(
    t_tai: np.ndarray,
    range_esa: np.ndarray,
    h_esa: np.ndarray,
    l1b_paths,
    *,
    retrackers: dict | None = None,
    ref: str = "tfmra50",
    ensemble: tuple = ("tfmra50", "lmg", "ocog"),
    time_tol_s: float = 0.01,
    g_ref: float = DEFAULT_G_REF,
    verbose: bool = True,
) -> MultiRetrackResult:
    r"""Run a *suite* of retrackers over matched L1B waveforms for a bake-off.

    Like :func:`build_lmg_elevations` but evaluates every retracker in
    ``retrackers`` (default :data:`RETRACKERS`) on each matched waveform, plus
    a NaN-robust median ``ensemble`` over ``ensemble``. A single range-bin /
    datum calibration is fit from the ``ref`` retracker's gates vs ESA and
    applied to all, so every retracker's elevation sits on the same scale and
    differences are pure retracker effects. Score the returned heights against
    ICESat-2 to rank them.
    """
    if retrackers is None:
        retrackers = RETRACKERS
    names = list(retrackers)
    t_tai = np.asarray(t_tai, dtype=np.float64)
    range_esa = np.asarray(range_esa, dtype=np.float64)
    h_esa = np.asarray(h_esa, dtype=np.float64)
    n = t_tai.size

    gates = {nm: np.full(n, np.nan) for nm in names}
    status = {nm: np.full(n, "unmatched", dtype=object) for nm in names}
    window_del = np.full(n, np.nan)
    range_cor = np.full(n, np.nan)

    order = np.argsort(t_tai)
    ts = t_tai[order]
    paths = list(l1b_paths)
    n_matched = 0
    for gi, path in enumerate(paths):
        from ..io.cs2_l1b import read_cs2_l1b_sarin_granule
        g = read_cs2_l1b_sarin_granule(path)
        if g is None or g["time"].size == 0:
            continue
        lt = g["time"]
        lo = np.searchsorted(ts, lt.min() - time_tol_s, side="left")
        hi = np.searchsorted(ts, lt.max() + time_tol_s, side="right")
        if hi <= lo:
            continue
        lorder = np.argsort(lt)
        lts = lt[lorder]
        for si in range(lo, hi):
            oi = order[si]
            tt = ts[si]
            k = np.searchsorted(lts, tt)
            best_li, best_dt = -1, time_tol_s
            for kk in (k - 1, k):
                if 0 <= kk < lts.size:
                    dt = abs(lts[kk] - tt)
                    if dt < best_dt:
                        best_dt, best_li = dt, int(lorder[kk])
            if best_li < 0:
                continue
            wf = g["pwr"][best_li]
            ch = g["coherence"][best_li]
            window_del[oi] = g["window_del"][best_li]
            range_cor[oi] = g["range_cor"][best_li]
            for nm, fn in retrackers.items():
                r = fn(wf, ch)
                gates[nm][oi] = r.gate
                status[nm][oi] = r.status
            n_matched += 1
        if verbose and (gi + 1) % 50 == 0:
            print(f"    … {gi + 1}/{len(paths)} L1B granules, {n_matched} matched",
                  flush=True)

    if ensemble:
        stk = np.vstack([gates[nm] for nm in ensemble])
        ens = np.nanmedian(stk, axis=0)
        gates["ensemble"] = ens
        status["ensemble"] = np.where(np.isfinite(ens), "ok", "unmatched").astype(object)
        names = names + ["ensemble"]

    # One calibration from the reference retracker; applied to all.
    gr = gates[ref]
    cm = np.isfinite(gr) & np.isfinite(window_del) & np.isfinite(range_esa)
    cal = calibrate_bin_range(range_esa[cm], window_del[cm], range_cor[cm],
                              gr[cm], g_ref=g_ref)
    heights = {}
    for nm in names:
        rng = range_from_gate(window_del, gates[nm], range_cor,
                              bin_range_m=cal.bin_range_m, g_ref=g_ref,
                              intercept_m=cal.intercept_m)
        heights[nm] = h_lmg_from_anchor(h_esa, range_esa, rng)

    if verbose:
        print(f"  matched {n_matched}/{n}; bin={cal.bin_range_m:.4f} m "
              f"(ref={ref}); {len(names)} retrackers", flush=True)
    return MultiRetrackResult(
        names=names, gates=gates, heights=heights, status=status,
        window_del=window_del, range_cor=range_cor,
        range_esa=range_esa, calibration=cal, n_matched=n_matched,
    )
