# Copyright (C) 2024-2026 by Andrew Hoffman <andrewohoffman@gmail.com>
#
# This file is part of stereo_melt.
#
# stereo_melt is free software: you can redistribute it and/or modify it
# under the terms of the MIT License. See the LICENSE file in the project
# root for full terms.

r"""CryoSat-2 SARIn vs ICESat-2 ATL06 cross-validation.

Stage 2 of the CS2 GCP integration plan: before adding CS2 to ASP
``pc_align`` alongside IS2, measure how close CS2 returns track IS2
returns at coincident points (within 750 m / ±30 days) and stratify the
residuals by surface class, terrain slope, and year.

Pass criteria (Stage 2 of the project plan, calibrated to Ross /
Transantarctic Mountains terrain — *not* East Antarctic plateau):

- |median bias| < 3 m on the gate class (default ``fast_grounded``)
- MAD < 8 m on the gate class
- |year-over-year median drift| < 1.5 m/yr on the gate class

The gate class is restricted to ``fast_grounded`` because that's the
slice we actually use as ASP control near the grounding zone. The
``slow_grounded`` / plateau ridges contributing through the TAM walls
have a much higher MAD floor (POCA-correction errors on steep adjacent
terrain) than published East Antarctic plateau norms suggest, and
trying to gate on them would false-fail.

Inputs / outputs are all files on disk so the harness can re-run cheaply
as more data lands. Driver lives in ``beardmore/validate_cs2.py``; this
module is study-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# pointCollection lives in vendor/ — import lazily inside loaders so this
# module's other functions still import without it.

# --- BedMachine mask values (per BedMachine v3 docs) ---
BM_OCEAN = 0
BM_ROCK = 1
BM_GROUNDED = 2
BM_FLOATING = 3
BM_LAKE = 4

SLOPE_BIN_EDGES_DEG = (0.0, 1.0, 3.0, 5.0, np.inf)
SLOPE_BIN_LABELS = ("<1°", "1-3°", "3-5°", ">5°")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_is2_cache(is2_dir: Path) -> pd.DataFrame:
    r"""Concat all per-strip IS2 ASP CSVs into one DataFrame.

    Expected schema (from :mod:`beardmore.cache_icesat2`):
    ``time, easting, northing, h_mean``. Returned with normalized columns
    ``t, x, y, h``.
    """
    # IS2 cache is HDF5 now; legacy CSVs still readable.
    files = sorted(Path(is2_dir).glob("icesat2_filtered_*.h5")) \
        + sorted(Path(is2_dir).glob("icesat2_filtered_*.csv"))
    if not files:
        raise FileNotFoundError(f"no IS2 caches (.h5 or .csv) under {is2_dir}")

    from ..io.altimetry import read_control_any
    parts = []
    for f in files:
        df = read_control_any(f)
        parts.append(df.rename(columns={
            "time": "t", "easting": "x", "northing": "y", "h_mean": "h",
        }))
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["x", "y", "h"]).reset_index(drop=True)
    return out


def load_cs2_cache(
    cs2_dir: Path,
    *,
    peakiness_min: float | None = 1.5,
    retracker_quality_min: int | None = 1,
) -> pd.DataFrame:
    r"""Concat all per-strip CS2 PointCollection HDF5 caches into one DataFrame.

    Expected per-file fields from :mod:`beardmore.cache_cryosat2`:
    ``x, y, h, t, source, quality, retracker_quality, peakiness, surface_type``.
    Returned columns: ``t, x, y, h, quality, peakiness, retracker_quality``.

    Quality filters applied at load time (defaults track Schröder 2019 et al.
    for SARIn over Antarctica):

    - ``peakiness > peakiness_min`` — peakiness ≤ 1.5 indicates diffuse /
      multipath returns; pass ``None`` to disable.
    - ``retracker_quality >= retracker_quality_min`` — 0 means the retracker
      failed to compute a quality metric (i.e. retrack failed). Pass ``None``
      to disable. The default of 1 keeps everything except outright failures.
    """
    from pointCollection.data import data as PCData

    files = sorted(Path(cs2_dir).glob("cs2_filtered_*.h5"))
    if not files:
        raise FileNotFoundError(f"no CS2 caches under {cs2_dir}")
    parts = []
    n_pre = 0
    n_post = 0
    for f in files:
        pc = PCData().from_h5(str(f))
        d = {fld: np.asarray(getattr(pc, fld)) for fld in pc.fields}
        n = d["x"].size
        n_pre += n
        peak = d.get("peakiness", np.full(n, np.nan, dtype=np.float64))
        retr = d.get("retracker_quality", np.zeros(n, dtype=np.int32))
        keep = np.ones(n, dtype=bool)
        if peakiness_min is not None:
            keep &= np.isfinite(peak) & (peak > peakiness_min)
        if retracker_quality_min is not None:
            keep &= retr >= retracker_quality_min
        n_post += int(keep.sum())
        parts.append(pd.DataFrame({
            "t": pd.to_datetime(d["t"][keep]) if "t" in d else pd.NaT,
            "x": d["x"][keep], "y": d["y"][keep], "h": d["h"][keep],
            "quality": d.get("quality", np.zeros(n, dtype=np.int32))[keep],
            "peakiness": peak[keep],
            "retracker_quality": retr[keep],
        }))
    if peakiness_min is not None or retracker_quality_min is not None:
        print(f"  CS2 quality filter "
              f"(peakiness>{peakiness_min}, retr_q>={retracker_quality_min}): "
              f"{n_post}/{n_pre} kept ({100.0 * n_post / max(n_pre, 1):.1f}%)")
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["x", "y", "h"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Spatio-temporal collocation
# ---------------------------------------------------------------------------


def collocate(
    cs2: pd.DataFrame,
    is2: pd.DataFrame,
    *,
    radius_m: float = 750.0,
    time_window_days: float = 30.0,
    min_is2_neighbors: int = 3,
) -> pd.DataFrame:
    r"""For each CS2 point, find IS2 points within ``radius_m`` and ±``time_window_days``,
    then take the IS2 median as the truth value.

    Two-pass filter for efficiency: spatial KDTree first (cheap), then
    per-CS2-point time mask. Returns one row per CS2 point that had at
    least ``min_is2_neighbors`` IS2 returns in its window. Columns:
    ``t, x, y, h_cs2, h_is2_med, n_is2, residual``.
    """
    if len(cs2) == 0 or len(is2) == 0:
        return pd.DataFrame(
            columns=["t", "x", "y", "h_cs2", "h_is2_med", "n_is2", "residual"]
        )

    is2_xy = np.column_stack([is2["x"].to_numpy(), is2["y"].to_numpy()])
    is2_t = is2["t"].to_numpy(dtype="datetime64[ns]")
    is2_h = is2["h"].to_numpy()
    tree = cKDTree(is2_xy)

    cs2_xy = np.column_stack([cs2["x"].to_numpy(), cs2["y"].to_numpy()])
    cs2_t = cs2["t"].to_numpy(dtype="datetime64[ns]")
    cs2_h = cs2["h"].to_numpy()

    half_window = np.timedelta64(int(time_window_days * 24 * 3600 * 1e9), "ns")

    rows = []
    # query_ball_point returns a list of neighbor indices per point.
    neighbor_lists = tree.query_ball_point(cs2_xy, r=radius_m, workers=-1)
    for i, nbrs in enumerate(neighbor_lists):
        if not nbrs:
            continue
        nbrs = np.asarray(nbrs, dtype=np.int64)
        dt = np.abs(is2_t[nbrs] - cs2_t[i])
        nbrs = nbrs[dt <= half_window]
        if nbrs.size < min_is2_neighbors:
            continue
        h_med = float(np.median(is2_h[nbrs]))
        rows.append((cs2_t[i], cs2_xy[i, 0], cs2_xy[i, 1],
                     cs2_h[i], h_med, int(nbrs.size), cs2_h[i] - h_med))
    return pd.DataFrame(
        rows, columns=["t", "x", "y", "h_cs2", "h_is2_med", "n_is2", "residual"],
    )


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def _sample_grid(arr_2d, x_coords, y_coords, e, n) -> np.ndarray:
    r"""Nearest-neighbor sample of a 2D grid at scattered (e, n) coords."""
    ix = np.clip(np.searchsorted(x_coords, e), 0, len(x_coords) - 1)
    if y_coords[0] > y_coords[-1]:
        iy = (len(y_coords) - 1) - np.clip(
            np.searchsorted(y_coords[::-1], n), 0, len(y_coords) - 1,
        )
    else:
        iy = np.clip(np.searchsorted(y_coords, n), 0, len(y_coords) - 1)
    return arr_2d[iy, ix]


@dataclass
class StratificationContext:
    bedmachine_path: Path
    velocity_path: Path | None = None
    velocity_var: str = "VX"  # MEaSUREs phase-map convention; see config
    velocity_var_y: str = "VY"
    fast_ice_threshold_m_per_yr: float = 10.0
    slope_smooth_sigma_m: float = 2_000.0


def stratify(df: pd.DataFrame, ctx: StratificationContext) -> pd.DataFrame:
    r"""Add ``surface_class``, ``slope_bin``, and ``year`` columns.

    ``surface_class`` is one of ``rock``, ``slow_grounded``, ``fast_grounded``,
    ``floating``, ``other``. The ``floating`` stratum is informational —
    SARIn over floating ice has tide effects we don't correct here, so its
    residuals are not held to the grounded-ice pass criteria.
    """
    if df.empty:
        for col in ("surface_class", "slope_bin", "year"):
            df[col] = pd.Series(dtype="object")
        return df

    e = df["x"].to_numpy()
    n = df["y"].to_numpy()

    # Bbox-crop the BedMachine grids before sampling — Beardmore's strip
    # extents are tiny vs the continent-wide raster.
    pad = max(ctx.slope_smooth_sigma_m * 4, 5_000.0)
    bbox = (e.min() - pad, e.max() + pad, n.min() - pad, n.max() + pad)
    bm = xr.open_dataset(ctx.bedmachine_path)
    by = bm["y"].values
    if by[0] > by[-1]:
        sub_mask = bm["mask"].sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2])).load()
        sub_surf = bm["surface"].sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2])).load()
    else:
        sub_mask = bm["mask"].sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3])).load()
        sub_surf = bm["surface"].sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3])).load()

    pt_mask = _sample_grid(sub_mask.values, sub_mask["x"].values, sub_mask["y"].values, e, n)

    dx = abs(float(sub_surf["x"].values[1] - sub_surf["x"].values[0]))
    dy = abs(float(sub_surf["y"].values[1] - sub_surf["y"].values[0]))
    s = sub_surf.values.astype(np.float64)
    s_smooth = gaussian_filter(s, sigma=(ctx.slope_smooth_sigma_m / dy,
                                         ctx.slope_smooth_sigma_m / dx))
    gy, gx = np.gradient(s_smooth, dy, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    pt_slope = _sample_grid(slope_deg, sub_surf["x"].values, sub_surf["y"].values, e, n)

    # Velocity stratification (only matters for grounded ice).
    pt_speed = np.zeros_like(e)
    if ctx.velocity_path is not None:
        vel = xr.open_dataset(ctx.velocity_path)
        vy = vel["y"].values
        if vy[0] > vy[-1]:
            sub_vx = vel[ctx.velocity_var].sel(
                x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2])).load()
            sub_vy = vel[ctx.velocity_var_y].sel(
                x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2])).load()
        else:
            sub_vx = vel[ctx.velocity_var].sel(
                x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3])).load()
            sub_vy = vel[ctx.velocity_var_y].sel(
                x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3])).load()
        # Take absolute speed; ignore sign.
        speed = np.hypot(sub_vx.values, sub_vy.values)
        pt_speed = _sample_grid(speed, sub_vx["x"].values, sub_vx["y"].values, e, n)

    surface_class = np.full(len(df), "other", dtype=object)
    surface_class[pt_mask == BM_ROCK] = "rock"
    grounded = pt_mask == BM_GROUNDED
    surface_class[grounded & (pt_speed < ctx.fast_ice_threshold_m_per_yr)] = "slow_grounded"
    surface_class[grounded & (pt_speed >= ctx.fast_ice_threshold_m_per_yr)] = "fast_grounded"
    surface_class[pt_mask == BM_FLOATING] = "floating"

    slope_bin = pd.cut(pt_slope, bins=SLOPE_BIN_EDGES_DEG,
                       labels=SLOPE_BIN_LABELS, right=False, include_lowest=True)

    out = df.copy()
    out["surface_class"] = surface_class
    out["slope_deg"] = pt_slope
    out["slope_bin"] = slope_bin
    out["speed_m_per_yr"] = pt_speed
    out["year"] = pd.to_datetime(out["t"]).dt.year
    return out


# ---------------------------------------------------------------------------
# Summaries + pass criteria
# ---------------------------------------------------------------------------


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))))


def summarize_by(df: pd.DataFrame, *by: str) -> pd.DataFrame:
    r"""Per-stratum residual summary: count, median, MAD, p16/p84.

    ``by`` is one or more column names (e.g. ``"surface_class"``,
    ``"slope_bin"``, ``"year"``).
    """
    if df.empty:
        return pd.DataFrame(
            columns=list(by) + ["n", "median", "mad", "p16", "p84"]
        )
    rows = []
    for keys, sub in df.groupby(list(by), observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        r = sub["residual"].to_numpy()
        r = r[np.isfinite(r)]
        rows.append((*keys, r.size, float(np.median(r)) if r.size else float("nan"),
                     _mad(r), float(np.percentile(r, 16)) if r.size else float("nan"),
                     float(np.percentile(r, 84)) if r.size else float("nan")))
    out = pd.DataFrame(
        rows, columns=list(by) + ["n", "median", "mad", "p16", "p84"],
    )
    return out.sort_values(list(by)).reset_index(drop=True)


@dataclass
class PassCriteria:
    """Gate thresholds for the CS2-vs-IS2 validation decision.

    Defaults are calibrated to Ross / Transantarctic Mountains terrain
    (steep, glaciated, POCA-correction-prone), *not* the East Antarctic
    plateau where SARIn over flat ice is much cleaner.
    """

    bias_max_m: float = 3.0
    mad_max_m_grounded: float = 8.0
    yoy_drift_max_m_per_yr: float = 1.5
    gate_class: str = "fast_grounded"


def apply_pass_criteria(
    pairs: pd.DataFrame,
    by_class: pd.DataFrame,
    pc: PassCriteria = PassCriteria(),
) -> dict:
    r"""Check Stage 2 pass criteria against the gate class only.

    ``pairs`` is the stratified per-CS2-point residuals dataframe;
    ``by_class`` is the per-class summary. We don't gate on the
    ``slow_grounded`` plateau ridges because their MAD floor is
    legitimately higher than published EAIS norms (steep TAM walls
    contributing POCA-correction error). Bias, MAD, and YoY drift are
    all evaluated on ``pc.gate_class`` only — the slice we'll actually
    use as ASP control near the grounding zone.
    """
    res: dict = {"criteria": pc, "fails": [], "passes_all": True}

    gate_row = by_class[by_class["surface_class"] == pc.gate_class]
    if gate_row.empty:
        res["passes_all"] = False
        res["fails"].append({"check": "gate_class_missing",
                             "gate_class": pc.gate_class})
        return res
    gate_row = gate_row.iloc[0]

    # 1. Bias on the gate class.
    if abs(float(gate_row["median"])) > pc.bias_max_m:
        res["passes_all"] = False
        res["fails"].append({"check": "bias",
                             "value_m": float(gate_row["median"])})

    # 2. MAD on the gate class.
    if float(gate_row["mad"]) > pc.mad_max_m_grounded:
        res["passes_all"] = False
        res["fails"].append({"check": "mad_grounded",
                             "value_m": float(gate_row["mad"])})

    # 3. Year-over-year drift on the gate-class subset only.
    gate_pairs = pairs[pairs["surface_class"] == pc.gate_class]
    by_year_gate = summarize_by(gate_pairs, "year")
    if len(by_year_gate) >= 2:
        sorted_y = by_year_gate.sort_values("year").reset_index(drop=True)
        diffs = sorted_y["median"].diff().abs()
        max_diff = float(diffs.max())
        res["yoy_max_diff_m"] = max_diff
        if max_diff > pc.yoy_drift_max_m_per_yr:
            res["passes_all"] = False
            res["fails"].append({"check": "yoy_drift", "max_diff_m": max_diff})

    return res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_residuals(
    df: pd.DataFrame,
    by_class: pd.DataFrame,
    by_class_slope: pd.DataFrame,
    by_year: pd.DataFrame,
    out_png: Path,
) -> None:
    r"""3-panel summary figure: histogram-by-class, slope×class heatmap of
    median residual, and year-over-year median + MAD ribbon."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: residual histograms by surface class.
    ax = axes[0]
    classes = ["rock", "slow_grounded", "fast_grounded", "floating"]
    colors = {"rock": "saddlebrown", "slow_grounded": "tab:blue",
              "fast_grounded": "tab:cyan", "floating": "tab:gray"}
    for c in classes:
        sub = df[df["surface_class"] == c]["residual"]
        if sub.empty:
            continue
        ax.hist(np.clip(sub, -10, 10), bins=60, alpha=0.55, label=f"{c} (n={len(sub)})",
                color=colors.get(c, None), density=True)
    ax.axvline(0, ls="--", color="k", lw=0.6)
    ax.set_xlabel("CS2 − IS2 residual (m)")
    ax.set_ylabel("density")
    ax.set_title("residuals by surface class")
    ax.legend(fontsize=8)

    # Panel 2: median residual by (class, slope_bin).
    ax = axes[1]
    if not by_class_slope.empty:
        pivot = by_class_slope.pivot(index="surface_class", columns="slope_bin",
                                     values="median")
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("median residual (m) by class × slope")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.set_axis_off()

    # Panel 3: year-over-year median ± MAD.
    ax = axes[2]
    if not by_year.empty:
        ax.errorbar(by_year["year"], by_year["median"], yerr=by_year["mad"],
                    marker="o", capsize=3)
        ax.axhline(0, ls="--", color="k", lw=0.6)
        ax.set_xlabel("year")
        ax.set_ylabel("median residual ± MAD (m)")
        ax.set_title("year-over-year drift")
    else:
        ax.set_axis_off()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    residuals: pd.DataFrame
    by_class: pd.DataFrame
    by_class_slope: pd.DataFrame
    by_year: pd.DataFrame
    decision: dict


def run_validation(
    is2_dir: Path,
    cs2_dir: Path,
    ctx: StratificationContext,
    *,
    radius_m: float = 750.0,
    time_window_days: float = 30.0,
    pass_criteria: PassCriteria = PassCriteria(),
    peakiness_min: float | None = 1.5,
    retracker_quality_min: int | None = 1,
) -> ValidationResult:
    r"""Run the full Stage 2 harness end-to-end.

    Loads the two caches, collocates, stratifies, summarizes, applies pass
    criteria. Caller is responsible for writing the residuals CSV and the
    figure (so the same in-memory result can be inspected interactively).

    ``peakiness_min`` and ``retracker_quality_min`` gate the CS2 cache at
    load time (see :func:`load_cs2_cache`). Pass ``None`` to disable.
    """
    is2 = load_is2_cache(is2_dir)
    cs2 = load_cs2_cache(
        cs2_dir,
        peakiness_min=peakiness_min,
        retracker_quality_min=retracker_quality_min,
    )
    print(f"  loaded {len(is2)} IS2 + {len(cs2)} CS2 points")

    pairs = collocate(cs2, is2,
                      radius_m=radius_m, time_window_days=time_window_days)
    print(f"  {len(pairs)} CS2 points collocated with ≥3 IS2 neighbors")

    pairs = stratify(pairs, ctx)
    by_class = summarize_by(pairs, "surface_class")
    by_class_slope = summarize_by(pairs, "surface_class", "slope_bin")
    by_year = summarize_by(pairs, "year")
    decision = apply_pass_criteria(pairs, by_class, pass_criteria)
    return ValidationResult(
        residuals=pairs,
        by_class=by_class,
        by_class_slope=by_class_slope,
        by_year=by_year,
        decision=decision,
    )
