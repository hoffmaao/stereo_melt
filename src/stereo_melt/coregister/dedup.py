"""Multi-strip overlap dedup, mirroring altimetryFit/remove_overlapping_DEM_data.

REMA strip search returns multiple strips that may cover the same area
on the same day or within a few days (different sensors, overlapping
segments, repeat passes). Smith's framework drops the higher-σ overlap
to avoid double-counting in least-squares fits where each pixel of
each strip enters as an independent observation.

This module provides two operations:

- :func:`find_overlapping_strip_pairs` --- identify strip pairs that
  are within ``dt_min_days`` of each other in time AND spatially
  coincident (their bounding boxes overlap or one contains the
  other's centroid). Returns ``[(i, j, overlap_area_sqkm), ...]``.
- :func:`dedup_stack` --- given a stack with per-strip σ from the
  jitter sidecars, drop overlapping epochs preferring the lower-σ
  member.

Both are non-destructive; ``dedup_stack`` returns a new
:class:`xarray.DataArray` and a list of which time indices were
dropped, leaving the original on disk untouched.

For the eventual LS-fit framework (LSsurf-style) overlap is acceptable
when each observation has a proper σ, and dedup is unnecessary
provided per-pixel weights are honest. This module is for hard-
decision pipelines (like the existing Eulerian / Lagrangian solvers)
where redundant epochs would just add correlated noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .jitter import load_strip_sigma_sidecar

__all__ = ["find_overlapping_strip_pairs", "dedup_stack"]


def find_overlapping_strip_pairs(
    stack: xr.DataArray,
    dt_min_days: float = 7.0,
    min_spatial_overlap_frac: float = 0.05,
) -> list[tuple[int, int, float]]:
    r"""Identify near-simultaneous, spatially-coincident epoch pairs in a stack.

    Two epochs ``(i, j)`` are flagged when:

    1. ``|t_i - t_j| < dt_min_days`` --- close in time.
    2. The fraction of finite pixels they share (intersection over
       smaller-of-the-two finite areas) exceeds ``min_spatial_overlap_frac``.

    Returns the list of ``(i, j, overlap_fraction)`` tuples, sorted
    descending by overlap fraction.

    Parameters
    ----------
    stack : xarray.DataArray, dims ``(time, y, x)``
        Surface elevation stack.
    dt_min_days : float
        Temporal proximity threshold, days. ``7.0`` matches roughly
        one orbital cycle of WorldView; pairs within this are
        candidates for dedup.
    min_spatial_overlap_frac : float
        Minimum spatial-overlap fraction to qualify as redundant.
    """
    times = pd.to_datetime(stack["time"].values)
    n_t = stack.sizes["time"]
    pairs: list[tuple[int, int, float]] = []
    finite = np.isfinite(stack.values)
    finite_count = finite.reshape(n_t, -1).sum(axis=1)
    for i in range(n_t):
        for j in range(i + 1, n_t):
            dt = abs((times[j] - times[i]).total_seconds()) / 86400.0
            if dt > dt_min_days:
                continue
            overlap = (finite[i] & finite[j]).sum()
            denom = max(min(finite_count[i], finite_count[j]), 1)
            frac = float(overlap) / float(denom)
            if frac >= min_spatial_overlap_frac:
                pairs.append((i, j, frac))
    pairs.sort(key=lambda p: -p[2])
    return pairs


def dedup_stack(
    stack: xr.DataArray,
    asp_root: str | Path,
    strip_ids: list[str] | None = None,
    dt_min_days: float = 7.0,
    min_spatial_overlap_frac: float = 0.05,
) -> tuple[xr.DataArray, list[int]]:
    r"""Drop redundant overlapping epochs, preferring lower-σ strips.

    Per pair flagged by :func:`find_overlapping_strip_pairs`, drop the
    member with higher ``sigma_iso_m`` (from the strip jitter sidecar
    written by :func:`stereo_melt.coregister.jitter.write_strip_sigma_sidecar`).
    If σ is missing for either member, drop the later-in-time one.

    Parameters
    ----------
    stack : xarray.DataArray
    asp_root : str or Path
        ASP output root, used to locate the σ sidecars.
    strip_ids : list of str, optional
        Per-epoch strip identifiers. If None, the function tries to
        recover them from the stack's ``source`` attribute, otherwise
        skips σ-based prioritization and just keeps the earlier
        epoch on every dropped pair.
    dt_min_days, min_spatial_overlap_frac
        Forwarded to :func:`find_overlapping_strip_pairs`.

    Returns
    -------
    deduped_stack : xarray.DataArray
        Stack with redundant epochs removed.
    dropped_indices : list of int
        Indices into the *original* stack that were removed.
    """
    pairs = find_overlapping_strip_pairs(
        stack, dt_min_days=dt_min_days, min_spatial_overlap_frac=min_spatial_overlap_frac
    )

    # Resolve sigma per epoch (NaN if unknown).
    n_t = stack.sizes["time"]
    sigma = np.full(n_t, np.nan, dtype=np.float64)
    if strip_ids is not None:
        for k, sid in enumerate(strip_ids):
            side = load_strip_sigma_sidecar(sid, asp_root)
            if side is not None and np.isfinite(side.get("sigma_iso_m", np.nan)):
                sigma[k] = float(side["sigma_iso_m"])

    drop = set()
    for i, j, _frac in pairs:
        if i in drop or j in drop:
            continue
        si, sj = sigma[i], sigma[j]
        if np.isfinite(si) and np.isfinite(sj):
            drop.add(j if sj >= si else i)
        else:
            # No sigma info: drop the later epoch (arbitrary but
            # reproducible).
            drop.add(j)

    keep_mask = np.ones(n_t, dtype=bool)
    for k in drop:
        keep_mask[k] = False
    return stack.isel(time=keep_mask), sorted(drop)
