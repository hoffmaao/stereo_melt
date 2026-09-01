"""ICP translation components + per-strip alignment error vs time (Shean 2019 Fig 4 analog).

Walks the Beardmore aligned-strip dirs, parses the latest pc_align log
per strip for the (north, east, down) translation and pre/post
p16/p50/p84 residuals from ``beg_errors.csv``/``end_errors.csv``
(column 4 = pc_align's Euclidean point-to-plane distance). Signed
vertical bias is captured by the ``z_off`` scatter panels (col 2 and
3, signed) and by the pre→post mean shift visible in the time-series
panel; the time-series itself is always non-negative because pc_align
stores |distance|.

Beardmore is split by control source (CS2-only pre-Oct 2018, IS2+CS2
post-Oct 2018) so the era boundary is visible. Strips outside the
Beardmore stack AOI are filtered out so the shared
``data/REMA/strips/ASP_*`` dirs can be safely scanned.

Convention: x = East, y = North, z = Up (= -Down).
"""
from __future__ import annotations

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

ROOT = Path("/wd2/projects/stereo_melt")
EXAMPLES = ROOT / "examples"
OUT = EXAMPLES / "beardmore" / "figures" / "fig4_alignment_diagnostics.png"

AOI_SHP = ROOT / "data/shapefiles/beardmore_stack_extent.shp"

# Raw REMA strips live in this dir (basin-agnostic). The pre-alignment
# signed residual samples the raw strip at the reference (x, y) points.
STRIPS_DIR = ROOT / "data" / "REMA" / "strips"

# Beardmore strips are split across two control-source dirs. We split
# the population so the IS2-era and CS2-era alignments show as separate
# colours in every panel (their distributions differ by 8x in
# post-align altimetric residual and 6x in horizontal translation
# magnitude, per the wider-AOI run).
BASINS = [
    {
        "name": "Beardmore IS2+CS2 (post-Oct 2018)",
        "color": "tab:blue",
        "dirs": [ROOT / "data/REMA/strips/ASP_is2cs2/asp_aligned"],
    },
    {
        "name": "Beardmore IS2-only (post-Oct 2018, ablation)",
        "color": "tab:green",
        "dirs": [ROOT / "data/REMA/strips/ASP_is2only/asp_aligned"],
    },
    {
        "name": "Beardmore CS2-only (pre-Oct 2018)",
        "color": "tab:orange",
        "dirs": [ROOT / "data/REMA/strips/ASP_cs2/asp_aligned"],
    },
]

NED_RE = re.compile(
    r"Translation vector \(North-East-Down, meters\):\s*Vector3\(\s*"
    r"([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*\)"
)
DATE_RE = re.compile(r"_(\d{8})_")


def load_error_percentiles(csv_path: Path) -> tuple[float, float, float] | None:
    """Parse a pc_align beg/end errors CSV. Column 4 = Euclidean distance.

    Returns (p16, p50, p84) of the distance column (m), or None if the
    file is missing/empty.
    """
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, comment="#", header=None,
                         names=["easting", "northing", "height", "error"])
    except (OSError, pd.errors.EmptyDataError):
        return None
    err = df["error"].to_numpy()
    err = err[np.isfinite(err)]
    if err.size == 0:
        return None
    p16, p50, p84 = np.percentile(err, [16, 50, 84])
    return float(p16), float(p50), float(p84)


def latest_log_for(strip_stem: str, aligned_dir: Path) -> Path | None:
    logs = sorted(aligned_dir.glob(f"{strip_stem}-log-pc_align-*.txt"))
    return logs[-1] if logs else None


def parse_log(log_path: Path) -> dict | None:
    try:
        text = log_path.read_text()
    except OSError:
        return None
    ned = NED_RE.search(text)
    if not ned:
        return None
    n, e, d = float(ned.group(1)), float(ned.group(2)), float(ned.group(3))
    return {"x_off": e, "y_off": n, "z_off": -d}


def date_from_stem(stem: str) -> datetime | None:
    m = DATE_RE.search(stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        return None


def _load_aoi():
    with fiona.open(AOI_SHP) as src:
        feat = next(iter(src))
    return shape(feat["geometry"])


def _strip_bbox(p: Path):
    try:
        with rasterio.open(p) as src:
            b = src.bounds
        return box(b.left, b.bottom, b.right, b.top)
    except Exception:
        return None


def harvest(dirs: Iterable[Path], aoi) -> dict[str, np.ndarray]:
    keys = ("x_off", "y_off", "z_off",
            "in_p16", "in_p50", "in_p84",
            "out_p16", "out_p50", "out_p84")
    bag: dict[str, list] = {k: [] for k in keys}
    dates: list = []
    for d in dirs:
        if not d.exists():
            continue
        for dem in sorted(d.glob("*-trans_reference-DEM.tif")):
            bbox = _strip_bbox(dem)
            if bbox is None or not bbox.intersects(aoi):
                continue
            stem = dem.name.replace("-trans_reference-DEM.tif", "")
            log = latest_log_for(stem, d)
            if log is None:
                continue
            stats = parse_log(log)
            if stats is None:
                continue
            beg = load_error_percentiles(d / f"{stem}-beg_errors.csv")
            end = load_error_percentiles(d / f"{stem}-end_errors.csv")
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
            dates.append(date_from_stem(stem))
    out = {k: np.array(v) for k, v in bag.items()}
    out["dates"] = np.array(dates, dtype=object)
    return out


def main() -> None:
    aoi = _load_aoi()
    print(f"Beardmore stack AOI bounds (EPSG:3031): {aoi.bounds}")
    data = {b["name"]: harvest(b["dirs"], aoi) for b in BASINS}
    for b in BASINS:
        d = data[b["name"]]
        n = d["x_off"].size
        if n == 0:
            print(f"{b['name']}: n=0")
            continue
        print(
            f"{b['name']}: n={n:3d}  "
            f"median (x,y,z)=({np.median(d['x_off']):+.2f}, {np.median(d['y_off']):+.2f}, {np.median(d['z_off']):+.2f}) m  "
            f"mean p50: pre={d['in_p50'].mean():.3f} m  post={d['out_p50'].mean():.3f} m"
        )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.05), hspace=0.32, wspace=0.28)
    ax_yx = fig.add_subplot(gs[0, 0])
    ax_zx = fig.add_subplot(gs[0, 1])
    ax_zy = fig.add_subplot(gs[0, 2])
    ax_t  = fig.add_subplot(gs[1, :])

    for b in BASINS:
        d = data[b["name"]]
        if d["x_off"].size == 0:
            continue
        label = f"{b['name']} (n={d['x_off'].size})"
        kw = dict(color=b["color"], alpha=0.6, s=28, edgecolor="none")
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

    legend_handles: list = []
    for b in BASINS:
        d = data[b["name"]]
        if d["x_off"].size == 0:
            continue
        valid = np.array([dt is not None for dt in d["dates"]])
        if not valid.any():
            continue
        ts = np.array([dt for dt in d["dates"][valid]])

        in50 = d["in_p50"][valid]
        out50 = d["out_p50"][valid]
        in_yerr = np.vstack([in50 - d["in_p16"][valid], d["in_p84"][valid] - in50])
        out_yerr = np.vstack([out50 - d["out_p16"][valid], d["out_p84"][valid] - out50])

        ax_t.errorbar(
            ts, in50, yerr=in_yerr,
            fmt="o", color=b["color"], alpha=0.5, mfc="none",
            ms=6, mew=1.2, elinewidth=1.0, capsize=2.5, zorder=2,
        )
        ax_t.errorbar(
            ts, out50, yerr=out_yerr,
            fmt="o", color=b["color"], alpha=1.0,
            ms=6, mew=0.0, elinewidth=1.2, capsize=2.5, zorder=3,
        )

        in_mean = float(in50.mean())
        out_mean = float(out50.mean())
        ax_t.axhline(in_mean, color=b["color"], linestyle="--", lw=1.2, alpha=0.55, zorder=1)
        ax_t.axhline(out_mean, color=b["color"], linestyle="--", lw=1.6, alpha=1.0, zorder=1)

        legend_handles.append(plt.Line2D(
            [], [], marker="o", color=b["color"], linestyle="none",
            label=f"{b['name']} (pre→post mean: {in_mean:.2f}→{out_mean:.2f} m, n={valid.sum()})",
        ))

    legend_handles.append(plt.Line2D([], [], marker="o", color="0.3", linestyle="none",
                                     mfc="none", mew=1.2, alpha=0.5, label="pre (16–84%)"))
    legend_handles.append(plt.Line2D([], [], marker="o", color="0.3", linestyle="none",
                                     alpha=1.0, label="post (16–84%)"))
    legend_handles.append(plt.Line2D([], [], color="0.3", linestyle="--", lw=1.2, alpha=0.55,
                                     label="basin mean (pre)"))
    legend_handles.append(plt.Line2D([], [], color="0.3", linestyle="--", lw=1.6, alpha=1.0,
                                     label="basin mean (post)"))

    Y_TOP = 35.0
    ax_t.set_xlabel("Strip date")
    ax_t.set_ylabel("Median DEM-altimetry error  (m, with 16-84% spread)")
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

    fig.suptitle("Beardmore pc_align ICP translation components + DEM error", fontsize=14, y=1.00)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
