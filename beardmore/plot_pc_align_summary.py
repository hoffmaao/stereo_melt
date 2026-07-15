"""pc_align summary: per-strip translation offsets + altimetric residual.

Iterates over every Beardmore-AOI strip that has an aligned DEM in
``config.STRIP_SOURCES`` (ASP_is2cs2 + ASP_cs2). For each:

  * Parse the pc_align log
    (``*-log-pc_align-*.txt``) for the post-ICP
    ``Translation vector (North-East-Down, meters)`` line.
  * Parse the post-alignment per-control-point error CSV
    (``*-end_errors.csv``, column 4) and take the median as the
    per-strip altimetric residual.

Output is a four-panel figure (one row): Δy vs Δx, Δz vs Δx, Δy vs Δz,
and a histogram of the per-strip median altimetric residual. Points are
colour-coded by control source (cs2 vs is2cs2) so era splits stay
visible.

The companion script ``beardmore.plot_fig4_alignment_diagnostics``
follows the original Shean 2019 Fig 4 layout (3 scatters + DEM error
vs time); use that one for the manuscript figure.

Run:

    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -m beardmore.plot_pc_align_summary
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

# Point pyproj at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box, shape

from beardmore import config


_NED_RE = re.compile(
    r"Translation vector \(North-East-Down, meters\):\s*"
    r"Vector3\(([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\)"
)


def _load_aoi():
    """Beardmore wider-AOI stack polygon in EPSG:3031."""
    with fiona.open(config.BEARDMORE_STACK_AOI_SHP) as src:
        feat = next(iter(src))
    return shape(feat["geometry"])


def _strip_bbox(p: Path):
    try:
        with rasterio.open(p) as src:
            b = src.bounds
        return box(b.left, b.bottom, b.right, b.top)
    except Exception:
        return None


def _parse_ned(stem_dir: Path, stem: str):
    """Return (n, e, d) in meters from the most recent pc_align log, or None."""
    logs = sorted(glob.glob(str(stem_dir / f"{stem}-log-pc_align-*.txt")))
    if not logs:
        return None
    with open(logs[-1]) as fh:
        for line in fh:
            m = _NED_RE.search(line)
            if m:
                return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None


def _median_residual(stem_dir: Path, stem: str):
    """Median |error| (m) from end_errors.csv col 4, or NaN."""
    csv_path = stem_dir / f"{stem}-end_errors.csv"
    if not csv_path.exists():
        return float("nan")
    try:
        arr = np.loadtxt(csv_path, delimiter=",", comments="#", usecols=3)
        return float(np.nanmedian(np.abs(arr)))
    except Exception:
        return float("nan")


_DATE_RE = re.compile(r"_(\d{8})_")


def _date_from_stem(stem: str):
    m = _DATE_RE.search(stem)
    return pd.Timestamp(m.group(1)) if m else pd.NaT


def collect() -> pd.DataFrame:
    aoi = _load_aoi()
    print(f"AOI bounds (EPSG:3031): {aoi.bounds}")

    rows = []
    for aligned_dir, source_tag in config.STRIP_SOURCES:
        aligned_dir = Path(aligned_dir)
        if not aligned_dir.exists():
            print(f"  skip (missing): {aligned_dir}")
            continue
        dems = sorted(aligned_dir.glob("*-trans_reference-DEM.tif"))
        print(f"  {source_tag}: {len(dems)} aligned DEMs in {aligned_dir}")
        kept = 0
        for dem in dems:
            bbox = _strip_bbox(dem)
            if bbox is None or not bbox.intersects(aoi):
                continue
            stem = dem.name.replace("-trans_reference-DEM.tif", "")
            ned = _parse_ned(aligned_dir, stem)
            if ned is None:
                continue
            med = _median_residual(aligned_dir, stem)
            rows.append(
                {
                    "stem": stem,
                    "source": source_tag,
                    "date": _date_from_stem(stem),
                    "dn_m": ned[0],
                    "de_m": ned[1],
                    "dd_m": ned[2],
                    "mag_m": float(np.linalg.norm(ned)),
                    "median_resid_m": med,
                }
            )
            kept += 1
        print(f"    kept {kept} inside Beardmore AOI")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), constrained_layout=True)

    colors = {"cs2": "#E67E22", "is2cs2": "#3498DB"}
    sources = sorted(df["source"].unique())

    def _scatter(ax, xcol, ycol, xlabel, ylabel):
        for src in sources:
            sub = df[df["source"] == src]
            ax.scatter(
                sub[xcol], sub[ycol], s=14, alpha=0.55,
                c=colors.get(src, "#888888"), edgecolors="none",
                label=f"{src} (n={len(sub)})",
            )
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.axvline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")

    _scatter(axes[0], "de_m", "dn_m", r"$\Delta$ east (m)", r"$\Delta$ north (m)")
    axes[0].set_title("Δy vs Δx  (north vs east)")

    _scatter(axes[1], "de_m", "dd_m", r"$\Delta$ east (m)", r"$\Delta$ down (m)")
    axes[1].set_title("Δz vs Δx  (down vs east)")

    _scatter(axes[2], "dd_m", "dn_m", r"$\Delta$ down (m)", r"$\Delta$ north (m)")
    axes[2].set_title("Δy vs Δz  (north vs down)")

    ax = axes[3]
    valid = df["median_resid_m"].dropna()
    if len(valid):
        vmax = float(np.percentile(valid, 99))
        edges = np.linspace(0, max(vmax, 0.1), 40)
        for src in sources:
            sub = df[df["source"] == src]["median_resid_m"].dropna()
            ax.hist(
                sub, bins=edges, alpha=0.55,
                color=colors.get(src, "#888888"),
                label=f"{src}  med={float(sub.median()):.2f} m  n={len(sub)}",
            )
    ax.set_xlabel("per-strip median |altimetric residual| post-align (m)")
    ax.set_ylabel("strip count")
    ax.set_title("median DEM altimetric residual")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    axes[0].legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Beardmore pc_align summary  ({len(df)} strips, "
        f"{df['date'].min().date()} → {df['date'].max().date()})",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config.ensure_output_dirs()
    df = collect()
    if df.empty:
        raise SystemExit("No Beardmore-AOI pc_align logs found in STRIP_SOURCES.")

    summary_path = config.RESULTS_DIR / "pc_align_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}  ({len(df)} rows)")

    by_src = df.groupby("source")
    print("\nPer-source translation summary (median ± MAD, m):")
    for src, sub in by_src:
        for col, lbl in [("de_m", "Δeast"), ("dn_m", "Δnorth"),
                          ("dd_m", "Δdown"), ("mag_m", "|Δ|"),
                          ("median_resid_m", "alt-resid")]:
            v = sub[col].dropna().values
            if len(v) == 0:
                continue
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            print(f"  {src:7s}  {lbl:10s}  {med:+.2f} ± {mad:.2f}  (n={len(v)})")

    out_path = config.FIGURES_DIR / "pc_align_summary.png"
    plot(df, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
