"""Per-strip noise estimation from ASP ``geodiff`` residuals.

After ``pc_align`` finishes, ASP writes a ``<strip>-final-diff.csv`` with
``longitude, latitude, height_diff_m`` rows for every control point in
the IS-2 + rock CSV. Those residuals are the post-coregistration
elevation discrepancy between the strip and the truth: directly the
data we need to characterize per-strip uncertainty.

This module produces three quantities per strip:

- ``sigma_iso`` --- MAD-based isotropic standard deviation. One number,
  appropriate as a constant per-pixel σ in any LS-style downstream fit.
- ``bias_AT(s)`` --- smoothed along-track bias. A 1-D function of the
  along-track coordinate ``s`` (in meters). Stereo DEMs from WV1/WV2/WV3
  carry along-track-correlated jitter from the satellite's stereo
  geometry; ``bias_AT`` is the slow component of that signal as
  recovered by a smoothing-spline fit to the residuals projected onto
  the strip's principal axis. Subtracting it from the strip surface
  removes the dominant correlated error before downstream consumption.
- ``sigma_AT`` --- robust dispersion of the residuals *after* removing
  ``bias_AT(s)``. Quantifies the irreducible noise floor.

The model is the simplest version of Smith's ``est_DEM_jitter_AT.py``
approach (altimetryFit): residual = bias_AT(s) + iid noise, with a
smoothness prior on ``d bias_AT / ds`` (scipy ``UnivariateSpline``) and
MAD outlier rejection.

References
----------

- Smith, B., et al. ``est_DEM_jitter_AT.py`` in
  ``github.com/SmithB/altimetryFit/altimetryFit``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import UnivariateSpline

__all__ = [
    "parse_geodiff",
    "estimate_strip_sigma",
    "write_strip_sigma_sidecar",
    "load_strip_sigma_sidecar",
]


def parse_geodiff(diff_csv: str | Path) -> pd.DataFrame:
    r"""Parse an ASP ``geodiff`` ``-final-diff.csv``.

    Returns a DataFrame with columns ``lon`` (deg), ``lat`` (deg),
    ``dh`` (m, signed residual = control - DEM). The header contains
    summary stats (Max/Min/Mean/StdDev/Median); we ignore those and
    parse the body.
    """
    df = pd.read_csv(
        diff_csv,
        comment="#",
        header=None,
        names=["lon", "lat", "dh"],
        engine="c",
    )
    return df.dropna()


def _project_to_3031(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat -> EPSG:3031 (Antarctic Polar Stereographic) in meters."""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    return tr.transform(lon, lat)


def _along_track_axis(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(s, c, theta)`` for the principal-axis projection.

    ``s`` is the along-track coordinate (m, zero-centered), ``c`` is the
    cross-track coordinate (m), and ``theta`` is the along-track
    direction angle in radians from +x. Computed via PCA on the
    centered point cloud — the long axis of the residual distribution
    is the strip's along-track direction.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cx, cy = float(x.mean()), float(y.mean())
    xc, yc = x - cx, y - cy
    cov = np.array([[np.dot(xc, xc), np.dot(xc, yc)],
                    [np.dot(xc, yc), np.dot(yc, yc)]]) / max(len(x) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues -> last col is principal axis
    v_AT = vecs[:, -1]
    v_CT = vecs[:, 0]
    s = xc * v_AT[0] + yc * v_AT[1]
    c = xc * v_CT[0] + yc * v_CT[1]
    theta = float(np.arctan2(v_AT[1], v_AT[0]))
    return s, c, theta


def _mad_sigma(x: np.ndarray) -> float:
    """Median absolute deviation scaled to a Gaussian σ estimator."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - med)))


def _fit_along_track_bias(
    s: np.ndarray,
    dh: np.ndarray,
    smoothing_scale_m: float = 5_000.0,
    iqr_clip_factor: float = 3.0,
) -> tuple[UnivariateSpline | None, np.ndarray]:
    """Fit a smoothed `bias_AT(s)` to residuals along the along-track axis.

    Two-pass IRLS-flavored fit:

    1. Robustly remove outliers via MAD (rows beyond
       ``iqr_clip_factor * MAD`` of the median residual are dropped).
    2. ``scipy.interpolate.UnivariateSpline`` on the kept points with a
       smoothing factor proportional to ``smoothing_scale_m`` and the
       inlier RMS — yields a low-frequency along-track bias.

    Parameters
    ----------
    s, dh : 1-D arrays
        Sorted-by-``s`` along-track coordinate (m) and residual (m).
    smoothing_scale_m : float
        Approximate length scale (m) of the slowest signal we want
        ``bias_AT`` to track. Larger values flatten the spline.
    iqr_clip_factor : float
        Multiplier of MAD for outlier rejection.

    Returns
    -------
    spline : UnivariateSpline or None
        Fitted bias function, or None if too few points to fit.
    inlier_mask : ndarray of bool
        Mask matching ``s`` indicating which rows survived the cut.
    """
    order = np.argsort(s)
    s_sorted = s[order]
    dh_sorted = dh[order]

    sig0 = _mad_sigma(dh_sorted)
    if sig0 == 0 or not np.isfinite(sig0):
        return None, np.ones_like(dh_sorted, dtype=bool)
    keep = np.abs(dh_sorted - np.median(dh_sorted)) < iqr_clip_factor * sig0
    if keep.sum() < 8:
        return None, keep

    s_in = s_sorted[keep]
    dh_in = dh_sorted[keep]
    s_unique, idx_first = np.unique(s_in, return_index=True)
    dh_unique = dh_in[idx_first]
    if s_unique.size < 8:
        return None, keep
    rms = float(np.std(dh_unique))
    n = s_unique.size
    s_factor = max(rms * rms * n, 1e-6)
    try:
        spl = UnivariateSpline(s_unique, dh_unique, k=3, s=s_factor)
    except Exception:
        return None, keep

    # Reorder inlier mask back to the original (unsorted) ordering.
    inv_order = np.argsort(order)
    keep_full = keep[inv_order]
    return spl, keep_full


def estimate_strip_sigma(
    strip_id: str,
    asp_root: str | Path,
    smoothing_scale_m: float = 5_000.0,
    iqr_clip_factor: float = 3.0,
) -> dict:
    r"""Compute per-strip noise statistics from the ASP final-diff residuals.

    Parameters
    ----------
    strip_id : str
        Strip identifier, i.e. the basename without ``.tif`` suffix
        (e.g. ``SETSM_s2s041_WV01_20190120_..._2m_lsf_seg1``).
    asp_root : str or Path
        ASP output root, typically ``data/REMA/strips/ASP/``. The
        residual CSV is expected at
        ``<asp_root>/final/<strip_id>-final-diff.csv``.
    smoothing_scale_m : float
        Approximate length scale (m) for the along-track bias fit.
    iqr_clip_factor : float
        Multiplier of MAD for outlier rejection.

    Returns
    -------
    dict
        Keys:

        - ``strip_id`` --- input.
        - ``n_residuals`` --- total residuals parsed.
        - ``n_inliers`` --- residuals retained after MAD clip.
        - ``sigma_iso_m`` --- MAD-based σ over all residuals.
        - ``sigma_AT_m`` --- MAD-based σ of residuals after subtracting
          the along-track bias. ``NaN`` if the bias fit failed.
        - ``bias_AT_max_m`` --- absolute peak of ``bias_AT(s)``.
        - ``bias_AT_rms_m`` --- RMS of ``bias_AT(s)`` evaluated at the
          inlier ``s`` values.
        - ``along_track_theta_rad`` --- direction of the along-track
          axis (rad, from +x in EPSG:3031).
        - ``along_track_extent_m`` --- max(|s|) of inliers.
        - ``smoothing_scale_m`` --- echo of input.
        - ``residual_mean_m`` / ``residual_median_m`` --- bulk stats.
    """
    asp_root = Path(asp_root)
    diff_csv = asp_root / "final" / f"{strip_id}-final-diff.csv"
    if not diff_csv.exists():
        raise FileNotFoundError(f"final-diff CSV missing: {diff_csv}")

    df = parse_geodiff(diff_csv)
    if df.empty:
        raise ValueError(f"empty residual CSV: {diff_csv}")

    x, y = _project_to_3031(df["lon"].to_numpy(), df["lat"].to_numpy())
    dh = df["dh"].to_numpy(dtype=np.float64)

    sigma_iso_m = _mad_sigma(dh)

    s, _, theta = _along_track_axis(x, y)
    spl, keep = _fit_along_track_bias(
        s, dh, smoothing_scale_m=smoothing_scale_m, iqr_clip_factor=iqr_clip_factor
    )
    if spl is not None:
        bias_at_inliers = spl(s[keep])
        residual_after_bias = dh[keep] - bias_at_inliers
        sigma_AT_m = _mad_sigma(residual_after_bias)
        # Sample bias on a regular along-track grid so we can summarize.
        s_grid = np.linspace(float(s[keep].min()), float(s[keep].max()), 128)
        bias_grid = spl(s_grid)
        bias_AT_max_m = float(np.nanmax(np.abs(bias_grid)))
        bias_AT_rms_m = float(np.sqrt(np.nanmean(bias_grid * bias_grid)))
    else:
        sigma_AT_m = float("nan")
        bias_AT_max_m = float("nan")
        bias_AT_rms_m = float("nan")

    return {
        "strip_id": strip_id,
        "n_residuals": int(len(dh)),
        "n_inliers": int(keep.sum()) if spl is not None else int(len(dh)),
        "sigma_iso_m": float(sigma_iso_m),
        "sigma_AT_m": float(sigma_AT_m),
        "bias_AT_max_m": float(bias_AT_max_m),
        "bias_AT_rms_m": float(bias_AT_rms_m),
        "along_track_theta_rad": float(theta),
        "along_track_extent_m": float(np.max(np.abs(s[keep])) if spl is not None else np.max(np.abs(s))),
        "smoothing_scale_m": float(smoothing_scale_m),
        "residual_mean_m": float(np.mean(dh)),
        "residual_median_m": float(np.median(dh)),
    }


def _sidecar_path(asp_root: Path, strip_id: str) -> Path:
    return Path(asp_root) / "asp_aligned" / f"{strip_id}_sigma.json"


def write_strip_sigma_sidecar(
    strip_id: str, asp_root: str | Path, **kwargs
) -> Path:
    r"""Compute strip σ and write a sidecar JSON next to the aligned DEM.

    Returns the sidecar path. Extra ``kwargs`` are forwarded to
    :func:`estimate_strip_sigma`.
    """
    stats = estimate_strip_sigma(strip_id, asp_root, **kwargs)
    out = _sidecar_path(Path(asp_root), strip_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    return out


def load_strip_sigma_sidecar(strip_id: str, asp_root: str | Path) -> dict | None:
    r"""Load a previously-written σ sidecar; return ``None`` if missing."""
    p = _sidecar_path(Path(asp_root), strip_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())
