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
