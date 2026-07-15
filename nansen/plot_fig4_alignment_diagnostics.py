"""Nansen Shean 2019 Fig 4 analog: pc_align translation + DEM-altimetry error.

Walks ``<asp_root>/asp_aligned/`` for the Nansen-AOI strips, parses the
latest pc_align log per strip for the (north, east, down) translation
and the percentiles of the pre/post-alignment error CSVs
(``*-beg_errors.csv`` / ``*-end_errors.csv``, column 4 = Euclidean
point-to-plane distance), and produces the standard two-row figure:

  * row 1: three offset-vs-offset scatters (Δy vs Δx, Δz vs Δx, Δz vs Δy)
  * row 2: per-strip median DEM-altimetry residual vs time with 16-84%
    spread bars (pre = open; post = filled), basin-mean dashed lines.

Convention: x = East, y = North, z = Up (= -Down).

Nansen is fully IS2-era, so there's no CS2-only / IS2+CS2 era split —
all strips share the same control source mix.

Run:

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -m nansen.plot_fig4_alignment_diagnostics
"""

from __future__ import annotations

import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Point pyproj at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

import fiona
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box, shape

from nansen import config


_NED_RE = re.compile(
    r"Translation vector \(North-East-Down, meters\):\s*Vector3\(\s*"
    r"([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*\)"
)
_DATE_RE = re.compile(r"_(\d{8})_")


def _load_aoi():
    """Use the wider stack extent if present; fall back to narrow AOI."""
    p = config.NANSEN_AOI_SHP
    with fiona.open(p) as src:
        feat = next(iter(src))
    return shape(feat["geometry"])


def _strip_bbox(p: Path):
    try:
        with rasterio.open(p) as src:
            b = src.bounds
        return box(b.left, b.bottom, b.right, b.top)
    except Exception:
        return None


def _latest_log_for(stem: str, aligned_dir: Path) -> Path | None:
    logs = sorted(aligned_dir.glob(f"{stem}-log-pc_align-*.txt"))
    return logs[-1] if logs else None


def _parse_ned(log_path: Path):
    try:
        text = log_path.read_text()
    except OSError:
        return None
    m = _NED_RE.search(text)
    if not m:
        return None
    n, e, d = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return {"x_off": e, "y_off": n, "z_off": -d}


def _error_percentiles(csv_path: Path):
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


def _date_from_stem(stem: str):
    m = _DATE_RE.search(stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        return None


def harvest(aligned_dir: Path, aoi) -> dict[str, np.ndarray]:
    keys = ("x_off", "y_off", "z_off",
            "in_p16", "in_p50", "in_p84",
            "out_p16", "out_p50", "out_p84")
    bag: dict[str, list] = {k: [] for k in keys}
    dates: list = []

    if not aligned_dir.exists():
        raise FileNotFoundError(f"No aligned-strip dir: {aligned_dir}")

    dems = sorted(aligned_dir.glob("*-trans_reference-DEM.tif"))
    print(f"  {len(dems)} aligned DEMs in {aligned_dir}")
    kept = 0
    for dem in dems:
        bbox = _strip_bbox(dem)
        if bbox is None or not bbox.intersects(aoi):
            continue
        stem = dem.name.replace("-trans_reference-DEM.tif", "")
        log = _latest_log_for(stem, aligned_dir)
        if log is None:
            continue
        stats = _parse_ned(log)
        if stats is None:
            continue
        beg = _error_percentiles(aligned_dir / f"{stem}-beg_errors.csv")
        end = _error_percentiles(aligned_dir / f"{stem}-end_errors.csv")
        if beg is None or end is None:
            continue
        bag["x_off"].append(stats["x_off"])
        bag["y_off"].append(stats["y_off"])
        bag["z_off"].append(stats["z_off"])
        bag["in_p16"].append(beg[0])
        bag["in_p50"].append(beg[1])
        bag["in_p84"].append(beg[2])
        bag["out_p16"].append(end[0])
        bag["out_p50"].append(end[1])
        bag["out_p84"].append(end[2])
        dates.append(_date_from_stem(stem))
        kept += 1
    print(f"    kept {kept} inside Nansen AOI")

    out = {k: np.array(v) for k, v in bag.items()}
    out["dates"] = np.array(dates, dtype=object)
    return out


def main() -> None:
    config.ensure_output_dirs()
    aoi = _load_aoi()
    print(f"Nansen AOI bounds (EPSG:3031): {aoi.bounds}")

    aligned_dir = config.STRIP_ALIGNED_DIR
    d = harvest(aligned_dir, aoi)

    if d["x_off"].size == 0:
        raise SystemExit("No aligned Nansen strips found.")

    print(
        f"\nn={d['x_off'].size:3d}  "
        f"median (x,y,z)=({np.median(d['x_off']):+.2f}, "
        f"{np.median(d['y_off']):+.2f}, {np.median(d['z_off']):+.2f}) m  "
        f"mean p50: pre={d['in_p50'].mean():.3f} m  "
        f"post={d['out_p50'].mean():.3f} m"
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.05), hspace=0.32, wspace=0.28)
    ax_yx = fig.add_subplot(gs[0, 0])
    ax_zx = fig.add_subplot(gs[0, 1])
    ax_zy = fig.add_subplot(gs[0, 2])
    ax_t = fig.add_subplot(gs[1, :])

    color = "tab:green"
    label = f"Nansen (n={d['x_off'].size})"
    kw = dict(color=color, alpha=0.6, s=28, edgecolor="none")
    ax_yx.scatter(d["x_off"], d["y_off"], label=label, **kw)
    ax_zx.scatter(d["x_off"], d["z_off"], label=label, **kw)
    ax_zy.scatter(d["y_off"], d["z_off"], label=label, **kw)

    for ax, (xlab, ylab, title) in zip(
        (ax_yx, ax_zx, ax_zy),
        (
            ("x offset (East, m)", "y offset (North, m)", "y vs x offset"),
            ("x offset (East, m)", "z offset (Up, m)",    "z vs x offset"),
            ("y offset (North, m)", "z offset (Up, m)",   "z vs y offset"),
        ),
    ):
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)

    # DEM-altimetry error time series
    valid = np.array([dt is not None for dt in d["dates"]])
    ts = np.array([dt for dt in d["dates"][valid]])
    in50 = d["in_p50"][valid]
    out50 = d["out_p50"][valid]
    in_yerr = np.vstack([in50 - d["in_p16"][valid],
                         d["in_p84"][valid] - in50])
    out_yerr = np.vstack([out50 - d["out_p16"][valid],
                          d["out_p84"][valid] - out50])

    ax_t.errorbar(
        ts, in50, yerr=in_yerr,
        fmt="o", color=color, alpha=0.5, mfc="none",
        ms=6, mew=1.2, elinewidth=1.0, capsize=2.5, zorder=2,
    )
    ax_t.errorbar(
        ts, out50, yerr=out_yerr,
        fmt="o", color=color, alpha=1.0,
        ms=6, mew=0.0, elinewidth=1.2, capsize=2.5, zorder=3,
    )

    in_mean = float(in50.mean())
    out_mean = float(out50.mean())
    ax_t.axhline(in_mean, color=color, linestyle="--",
                 lw=1.2, alpha=0.55, zorder=1)
    ax_t.axhline(out_mean, color=color, linestyle="--",
                 lw=1.6, alpha=1.0, zorder=1)

    legend_handles = [
        plt.Line2D([], [], marker="o", color=color, linestyle="none",
                   label=f"Nansen (pre→post mean: {in_mean:.2f}→{out_mean:.2f} m, "
                         f"n={int(valid.sum())})"),
        plt.Line2D([], [], marker="o", color="0.3", linestyle="none",
                   mfc="none", mew=1.2, alpha=0.5, label="pre (16-84%)"),
        plt.Line2D([], [], marker="o", color="0.3", linestyle="none",
                   alpha=1.0, label="post (16-84%)"),
        plt.Line2D([], [], color="0.3", linestyle="--", lw=1.2, alpha=0.55,
                   label="mean (pre)"),
        plt.Line2D([], [], color="0.3", linestyle="--", lw=1.6, alpha=1.0,
                   label="mean (post)"),
    ]

    Y_TOP = 35.0
    ax_t.set_xlabel("Strip date")
    ax_t.set_ylabel("Median DEM-altimetry error (m, with 16-84% spread)")
    ax_t.set_title(
        f"Pre- vs post-alignment DEM error  (y-axis 0 to {Y_TOP:.0f} m; "
        "long-tail upper error bars clipped)"
    )
    ax_t.legend(handles=legend_handles, loc="upper right", fontsize=9, ncols=2)
    ax_t.xaxis.set_major_locator(mdates.YearLocator())
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax_t.get_xticklabels(), rotation=30, ha="right")
    ax_t.set_ylim(0, Y_TOP)
    ax_t.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Nansen pc_align ICP translation components + DEM error  "
        f"({config.START_TIME} → {config.END_TIME})",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()

    out_path = config.FIGURES_DIR / "fig4_alignment_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
