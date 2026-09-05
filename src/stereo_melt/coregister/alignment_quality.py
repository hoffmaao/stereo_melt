# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

"""Per-strip pc_align quality fields.

Walks each strip's pc_align outputs (``*-log-pc_align-*.txt``,
``*-beg_errors.csv``, ``*-end_errors.csv``) and returns one row per
strip with the NED translation magnitudes and the
beg/end percentiles of the Euclidean point-to-plane residual.

``end_p50`` (post-alignment residual median) is the cleanest single
quality metric for a pc_align run: large values flag strips where
pc_align either could not converge (degenerate control geometry) or
where the strip carries non-rigid distortion the 6-DOF transform
cannot remove. Per the docstring on ``score_tilt_residuals``, it
*does not* directly informer per-strip alpha_z uncertainty (that
depends on per-source coverage, not on per-control-point precision),
but it is the right quantity for a "should this strip be in the
stack at all?" decision.

Used by:
- ``tilt_qc.score_tilt_residuals`` (optional join, exposes end_p50
  as a per-epoch QC column)
- ``tilt_qc.suggest_bad_epochs`` (optional ``end_p50_threshold`` gate)
- ``scripts/fig4_alignment_diagnostics.py`` (already parses this
  inline; can be refactored to call here)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


__all__ = [
    "parse_per_strip_quality",
    "aggregate_basin_quality",
    "load_combined_reference",
    "load_control_glob",
    "sample_dem_at_points",
    "fit_residual_plane",
    "strip_residual_plane",
    "residual_planes_for_strips",
]


_DATE_RE = re.compile(r"_(\d{8})_")
_NED_RE = re.compile(
    r"Translation vector \(North-East-Down, meters\):\s*Vector3\(\s*"
    r"([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*\)"
)


def _latest_pc_align_log(aligned_dir: Path, stem: str) -> Path | None:
    logs = sorted(aligned_dir.glob(f"{stem}-log-pc_align-*.txt"))
    return logs[-1] if logs else None


def _percentiles_from_errors_csv(csv_path: Path):
    """Parse pc_align beg/end errors CSV. Col 4 = Euclidean residual."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(
            csv_path, comment="#", header=None,
            names=["easting", "northing", "height", "error"],
        )
    except (OSError, pd.errors.EmptyDataError):
        return None
    err = df["error"].to_numpy()
    err = err[np.isfinite(err)]
    if err.size == 0:
        return None
    p16, p50, p84 = np.percentile(err, [16, 50, 84])
    return float(p16), float(p50), float(p84)


def _ned_from_log(log_path: Path | None):
    """Parse latest pc_align log; return (N, E, D) translation or None."""
    if log_path is None:
        return None
    try:
        text = log_path.read_text()
    except OSError:
        return None
    m = _NED_RE.search(text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _date_from_stem(stem: str) -> "pd.Timestamp":
    m = _DATE_RE.search(stem)
    if not m:
        return pd.NaT
    try:
        return pd.Timestamp(m.group(1))
    except ValueError:
        return pd.NaT


def parse_per_strip_quality(aligned_dir: "Path | str") -> pd.DataFrame:
    """Return one row per aligned strip in ``aligned_dir``.

    Columns:
      ``dem_id``, ``date``, ``aligned_dir``, ``ned_dn``, ``ned_de``,
      ``ned_dd``, ``beg_p16``, ``beg_p50``, ``beg_p84``,
      ``end_p16``, ``end_p50``, ``end_p84``.

    Strips are enumerated by their ``*-end_errors.csv`` (the file that
    actually carries ``end_p50``), **not** by the aligned
    ``*-trans_reference-DEM.tif``: that DEM is swept to reclaim disk
    after ``point2dem`` (per-strip disk-budget policy), so keying off it
    silently drops every strip whose aligned output is already gone --
    notably the pre-IS2 / CS2-only era. Strips missing the end_errors
    CSV are skipped; a missing pc_align log only nulls the NED
    translation. Beg-errors are filled with NaN when absent.
    """
    aligned_dir = Path(aligned_dir)
    if not aligned_dir.exists():
        return pd.DataFrame()

    rows = []
    for end_csv in sorted(aligned_dir.glob("*-end_errors.csv")):
        stem = end_csv.name.replace("-end_errors.csv", "")
        end = _percentiles_from_errors_csv(end_csv)
        if end is None:
            continue
        log = _latest_pc_align_log(aligned_dir, stem)
        ned = _ned_from_log(log)  # may be None if the log was swept too
        beg = _percentiles_from_errors_csv(aligned_dir / f"{stem}-beg_errors.csv")
        rows.append({
            "dem_id": stem,
            "date": _date_from_stem(stem),
            "aligned_dir": str(aligned_dir),
            "ned_dn": ned[0] if ned is not None else np.nan,
            "ned_de": ned[1] if ned is not None else np.nan,
            "ned_dd": ned[2] if ned is not None else np.nan,
            "beg_p16": beg[0] if beg is not None else np.nan,
            "beg_p50": beg[1] if beg is not None else np.nan,
            "beg_p84": beg[2] if beg is not None else np.nan,
            "end_p16": end[0],
            "end_p50": end[1],
            "end_p84": end[2],
        })
    return pd.DataFrame(rows)


def aggregate_basin_quality(
    strip_sources: "list[tuple[Path, str]]",
) -> pd.DataFrame:
    """Aggregate across a basin's ``STRIP_SOURCES`` list.

    ``strip_sources`` is the per-basin ``config.STRIP_SOURCES`` value:
    a list of ``(aligned_dir, variant_tag)`` tuples. Beardmore has two
    entries (``ASP_is2cs2`` + ``ASP_cs2``); other basins have one.

    Returns a single concatenated DataFrame with an extra ``variant``
    column tagging which ASP root each row came from.
    """
    dfs = []
    for aligned_dir, variant in strip_sources:
        df = parse_per_strip_quality(aligned_dir)
        if df.empty:
            continue
        df["variant"] = variant
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-strip residual PLANE against independent control (2026-09-02)
# ---------------------------------------------------------------------------
# The Euclidean ``end_errors`` residual above is unsigned and its heights sit
# in the pre-transform frame, so it cannot give the sign or the plane of what
# is left after alignment. The functions below sample the ALIGNED DEM at the
# control points pc_align was fed (``reference_files/combined_reference_*.csv``;
# the twin's ``<control_dir>/<dem_id>_<source>.csv``) and fit a robust plane to
# the SIGNED residual ``DEM - control``. Against the processing twin (173
# strips) that per-strip plane recovers the true residual tilt at corr 0.93 (x)
# / 0.75 (y), regression slope ~0.95, and the across-strip variance -- the
# coloured-noise prior the melt inverse needs -- to 1.08x / 1.6x, where every
# estimator built on the DEM stack alone is 2x-167x low or diverges. The
# offset is NOT recoverable this way: control lives on the static apron, where
# pc_align pins it, while the shelf carries datum-stage residual the control
# never sees (twin: 0.07x). See dynamics.budget_bridging.strip_prior_from_residual_planes.

_XYH_COLUMNS = ("easting", "northing", "h_mean")


def _read_xyh_csv(path: Path) -> np.ndarray | None:
    """``(n, 3)`` easting/northing/height from a control CSV; header optional.

    Three outcomes the caller has to be able to tell apart, because a strip
    with no control and a strip whose control failed to parse warrant opposite
    responses:

    * **absent** -- returns ``None``. The strip simply has no control file.
    * **present but unreadable** -- raises ``ValueError`` naming the file and
      what was found. A readable file with an unexpected schema used to come
      back as ``None`` too: the header row was re-read as data, every column
      went object-dtype, ``select_dtypes("number")`` found nothing, and the
      empty result was indistinguishable from a missing file. Downstream that
      surfaced as ``qc_unfitted`` -- the bucket documented as "normally no
      control, not a bad strip" -- so a basin-wide column-name mismatch read as
      "no control anywhere" with nothing naming the real cause.
    * **present and valid** -- returns the array.

    The headerless branch is entered by TESTING row 0, not by inferring it from
    a failed name match, so a capitalised or renamed header is reported rather
    than silently consumed. Accepted layouts: the
    :data:`_XYH_COLUMNS` names in any order and case (extra columns ignored), or
    three or more numeric columns with no header at all.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        head = pd.read_csv(path, comment="#", header=None, nrows=1)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if head.empty:
        return None
    # Row 0 is a header iff it is not fully numeric.
    row0_numeric = pd.to_numeric(head.iloc[0], errors="coerce").notna().all()
    try:
        df = (pd.read_csv(path, comment="#", header=None) if row0_numeric
              else pd.read_csv(path, comment="#"))
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"control CSV {path} exists but could not be parsed: {exc}")
    if df.empty:
        return None

    if row0_numeric:
        cols = [pd.to_numeric(df[c], errors="coerce") for c in df.columns]
        cols = [c for c in cols if c.notna().any()]
        if len(cols) < 3:
            raise ValueError(
                f"control CSV {path} has no header and only {len(cols)} numeric "
                f"column(s); need at least 3 (easting, northing, height)")
        arr = np.column_stack([c.to_numpy(float) for c in cols[:3]])
    else:
        lookup = {str(c).strip().lower(): c for c in df.columns}
        missing = [c for c in _XYH_COLUMNS if c not in lookup]
        if missing:
            raise ValueError(
                f"control CSV {path} has header {list(df.columns)}, which is "
                f"missing {missing}; expected the columns {list(_XYH_COLUMNS)} "
                "(any order/case) or a headerless numeric file")
        arr = np.column_stack([
            pd.to_numeric(df[lookup[c]], errors="coerce").to_numpy(float)
            for c in _XYH_COLUMNS])

    arr = arr[np.isfinite(arr).all(1)]
    return arr if arr.size else None


def load_combined_reference(reference_dir: "Path | str", dem_id: str) -> np.ndarray | None:
    """The control cloud pc_align was fed for ``dem_id`` (production layout).

    ``<asp_root>/reference_files/combined_reference_<dem_id>.csv`` -- the
    frame the ``-trans_reference-DEM.tif`` was moved INTO, so sampling the
    aligned DEM at these points gives a signed residual with median ~0 (PIG
    2018-10-25 strip: median +0.000 m, MAD 0.370 m vs pc_align's own p50 0.380).
    """
    return _read_xyh_csv(Path(reference_dir) / f"combined_reference_{dem_id}.csv")


def load_control_glob(control_dir: "Path | str", dem_id: str,
                      pattern: str = "{dem_id}_*.csv") -> np.ndarray | None:
    """Concatenate every ``<control_dir>/<dem_id>_<source>.csv`` (twin layout)."""
    parts = [_read_xyh_csv(p) for p in sorted(Path(control_dir).glob(pattern.format(dem_id=dem_id)))]
    parts = [p for p in parts if p is not None]
    return np.concatenate(parts) if parts else None


def sample_dem_at_points(dem_path: "Path | str", easting: np.ndarray,
                         northing: np.ndarray) -> np.ndarray:
    """Aligned-DEM height at each point (NaN where nodata / outside), windowed read.

    Out-of-footprint points are masked against the raster BOUNDS, not against
    the nodata sentinel. rasterio's ``sample`` fills points outside the grid
    with ``dataset.nodata or 0``, so a DEM written without a nodata tag -- or
    with ``nodata == 0.0``, which that ``or`` collapses to the same value --
    would otherwise hand back a real 0.0 m elevation for every control point
    overhanging the strip. The bounds test is HALF-OPEN on the far edges
    (``e < right``, ``n > bottom``) to match rasterio's flooring ``rowcol``:
    a point exactly on ``bounds.right`` maps to ``col == width`` and one on
    ``bounds.bottom`` to ``row == height``, both of which the sampler rejects
    and fills, so a closed test would let those two edges through. Downstream that is a residual of ``0 - h_control``,
    tens of metres on a shelf, which inflates
    :func:`fit_residual_plane`'s ``sd`` and can make a good strip look like an
    alignment failure. (Robustness only: every PIG and twin aligned DEM
    carries ``nodata = -9999.0``, so no measured result on record went through
    the unguarded path.)
    """
    import rasterio

    e = np.asarray(easting, float)
    n = np.asarray(northing, float)
    z = np.full(e.shape, np.nan, float)
    with rasterio.open(dem_path) as src:
        b = src.bounds
        inside = (np.isfinite(e) & np.isfinite(n)
                  & (e >= b.left) & (e < b.right)
                  & (n > b.bottom) & (n <= b.top))
        if inside.any():
            vals = np.array([v[0] for v in src.sample(zip(e[inside], n[inside]))], float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
            z[inside] = vals
    return z


def fit_residual_plane(easting: np.ndarray, northing: np.ndarray, resid: np.ndarray, *,
                       robust: bool = True, max_iter: int = 6, c_tukey: float = 4.685,
                       min_points: int = 30) -> dict:
    r"""Robust plane ``c + ax (x - xc) + ay (y - yc)`` through a signed residual.

    Tukey-biweight IRLS on the MAD scale (the tilt fit's own weighting). The
    standard errors are the WLS ones, ``sd^2 (A^T W A)^{-1}``, with ``sd`` the
    robust residual scale -- they carry the control's own noise and its
    geometry (a strip with control only along one edge has a large ``se_ay``),
    which is what lets the population prior subtract estimator variance.
    Returns NaNs (and ``n``) when fewer than ``min_points`` are usable.
    """
    e = np.asarray(easting, float)
    n = np.asarray(northing, float)
    r = np.asarray(resid, float)
    ok = np.isfinite(e) & np.isfinite(n) & np.isfinite(r)
    out = dict(offset=np.nan, ax=np.nan, ay=np.nan, se_offset=np.nan, se_ax=np.nan,
               se_ay=np.nan, sd=np.nan, n=int(ok.sum()), xc=np.nan, yc=np.nan)
    if ok.sum() < min_points:
        return out
    e, n, r = e[ok], n[ok], r[ok]
    xc, yc = float(e.mean()), float(n.mean())
    A = np.column_stack([np.ones_like(e), e - xc, n - yc])
    w = np.ones_like(r)
    for _ in range(max_iter if robust else 1):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], r * sw, rcond=None)
        res = r - A @ coef
        mad = float(np.median(np.abs(res - np.median(res)))) + 1e-12
        if not robust:
            break
        u = res / (c_tukey * 1.4826 * mad)
        w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if w.sum() < min_points:  # degenerate reweighting: fall back to LSQ
            w = np.ones_like(r)
            break
    sd = 1.4826 * mad
    try:
        cov = sd ** 2 * np.linalg.inv((A * w[:, None]).T @ A)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    out.update(offset=float(coef[0]), ax=float(coef[1]), ay=float(coef[2]),
               se_offset=float(se[0]), se_ax=float(se[1]), se_ay=float(se[2]),
               sd=float(sd), xc=xc, yc=yc)
    return out


def strip_residual_plane(aligned_dem_path: "Path | str", control_xyh: np.ndarray, *,
                         max_points: int | None = 6000, seed: int = 0, **kw) -> dict:
    """Signed ``aligned DEM - control`` plane for one strip (see fit_residual_plane).

    ``max_points``: random subsample of the control before sampling the DEM.
    A plane has three parameters; 6000 points leave the standard errors
    within a few percent of the full cloud while the per-point raster read
    (the whole cost on a 2 m strip with ~30k control points) drops 5x.
    ``None`` uses every point.
    """
    xyh = np.asarray(control_xyh, float)
    if max_points is not None and len(xyh) > max_points:
        idx = np.random.default_rng(seed).choice(len(xyh), max_points, replace=False)
        xyh = xyh[np.sort(idx)]
    z = sample_dem_at_points(aligned_dem_path, xyh[:, 0], xyh[:, 1])
    return fit_residual_plane(xyh[:, 0], xyh[:, 1], z - xyh[:, 2], **kw)


def residual_planes_for_strips(items, *, log_every: int = 0, **kw) -> pd.DataFrame:
    """One row per strip from ``(dem_id, aligned_dem_path, control_xyh)`` items.

    ``control_xyh`` may be ``None`` (row of NaNs, ``n=0``) so a missing control
    file is visible in the table rather than silently dropped.
    """
    rows = []
    for i, (dem_id, dem_path, xyh) in enumerate(items):
        if xyh is None or len(xyh) == 0:
            row = dict(offset=np.nan, ax=np.nan, ay=np.nan, se_offset=np.nan, se_ax=np.nan,
                       se_ay=np.nan, sd=np.nan, n=0, xc=np.nan, yc=np.nan)
        else:
            row = strip_residual_plane(dem_path, xyh, **kw)
        row["dem_id"] = dem_id
        rows.append(row)
        if log_every and (i + 1) % log_every == 0:
            print(f"    residual planes: {i + 1} strips", flush=True)
    cols = ["dem_id", "offset", "ax", "ay", "se_offset", "se_ax", "se_ay", "sd", "n", "xc", "yc"]
    return pd.DataFrame(rows, columns=cols)
