"""Venable Shean 2019 Fig 4 analog: pc_align translation + post-align residual.

Adapted from ``pig.plot_fig4_alignment_diagnostics`` (single control
population — Venable is fully IS2-era, one ASP root). Row 1: the pc_align
ICP translation components (Δy vs Δx, Δz vs Δx, Δz vs Δy) parsed from each
strip's latest pc_align log. Row 2: per-strip post-alignment vertical
residual vs time, resampled from the aligned DEM at the strip's fixed
rock+altimetry control points (``reference_files/combined_reference_*.csv``),
with 16-84% spread and the basin mean dashed.

Convention: x = East, y = North, z = Up (= -Down).

Run:
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -m venable.plot_fig4_alignment_diagnostics
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

import fiona
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box, shape

from venable import config

_VARIANTS = [
    (config.STRIP_ALIGNED_DIR, "IS2 control (2019-2024)", "tab:blue"),
]
_RESAMPLE_MAX_PTS = 3000
_RESAMPLE_SEED = 0

_NED_RE = re.compile(
    r"Translation vector \(North-East-Down, meters\):\s*Vector3\(\s*"
    r"([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*\)"
)
_DATE_RE = re.compile(r"_(\d{8})_")
_CTRL_PREFIX = "combined_reference_"


def _load_aoi():
    with fiona.open(config.VENABLE_AOI_SHP) as src:
        feat = next(iter(src))
    return shape(feat["geometry"])


def _build_control_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    ref_dir = config.ASP_ROOT / "reference_files"
    if ref_dir.exists():
        for csv in ref_dir.glob(f"{_CTRL_PREFIX}*.csv"):
            stem = csv.name[len(_CTRL_PREFIX):-len(".csv")]
            idx.setdefault(stem, csv)
    return idx


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


def _resample_residual(dem_path: Path, control_csv: Path):
    if not control_csv.exists() or not dem_path.exists():
        return None
    try:
        df = pd.read_csv(control_csv)
    except (OSError, pd.errors.EmptyDataError):
        return None
    cols = {c.lower(): c for c in df.columns}
    ec, nc, hc = cols.get("easting"), cols.get("northing"), cols.get("h_mean")
    if not (ec and nc and hc) or df.empty:
        return None
    if len(df) > _RESAMPLE_MAX_PTS:
        df = df.sample(n=_RESAMPLE_MAX_PTS, random_state=_RESAMPLE_SEED)
    e = df[ec].to_numpy(dtype=float)
    n = df[nc].to_numpy(dtype=float)
    h = df[hc].to_numpy(dtype=float)
    try:
        with rasterio.open(dem_path) as src:
            nod = src.nodata if src.nodata is not None else -9999.0
            vals = np.fromiter(
                (v[0] for v in src.sample(zip(e, n))), dtype=float, count=len(e)
            )
    except Exception:
        return None
    resid = vals - h
    good = np.isfinite(resid) & (vals != nod)
    resid = resid[good]
    if resid.size < 20:
        return None
    a = np.abs(resid)
    p16, p50, p84 = np.percentile(a, [16, 50, 84])
    return float(p16), float(p50), float(p84), float(np.median(resid))


def _date_from_stem(stem: str):
    m = _DATE_RE.search(stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        return None


def harvest(aligned_dir: Path, aoi, control_index: dict[str, Path]) -> dict:
    keys = ("x_off", "y_off", "z_off", "out_p16", "out_p50", "out_p84", "out_bias")
    bag: dict[str, list] = {k: [] for k in keys}
    dates: list = []
    if not aligned_dir.exists():
        raise FileNotFoundError(f"No aligned-strip dir: {aligned_dir}")
    stems = sorted({
        re.sub(r"-log-pc_align-.*$", "", p.name)
        for p in aligned_dir.glob("*-log-pc_align-*.txt")
    })
    print(f"  {len(stems)} aligned strips (by pc_align log) in {aligned_dir}")
    kept = no_ctrl = no_resample = 0
    for stem in stems:
        if (aligned_dir / f"{stem}.bad_align").exists():
            continue
        dem = aligned_dir / f"{stem}-trans_reference-DEM.tif"
        if not dem.exists():
            continue
        bbox = _strip_bbox(dem)
        if bbox is None or not bbox.intersects(aoi):
            continue
        log = _latest_log_for(stem, aligned_dir)
        if log is None:
            continue
        stats = _parse_ned(log)
        if stats is None:
            continue
        control_csv = control_index.get(stem)
        if control_csv is None:
            no_ctrl += 1
            continue
        res = _resample_residual(dem, control_csv)
        if res is None:
            no_resample += 1
            continue
        bag["x_off"].append(stats["x_off"])
        bag["y_off"].append(stats["y_off"])
        bag["z_off"].append(stats["z_off"])
        bag["out_p16"].append(res[0])
        bag["out_p50"].append(res[1])
        bag["out_p84"].append(res[2])
        bag["out_bias"].append(res[3])
        dates.append(_date_from_stem(stem))
        kept += 1
        if kept % 50 == 0:
            print(f"    ... resampled {kept} strips", flush=True)
    print(f"    kept {kept} inside AOI ({no_ctrl} lacked control CSV, "
          f"{no_resample} failed resample)")
    out = {k: np.array(v) for k, v in bag.items()}
    out["dates"] = np.array(dates, dtype=object)
    return out


def main() -> None:
    aoi = _load_aoi()
    print(f"Venable AOI bounds (EPSG:3031): {aoi.bounds}")
    control_index = _build_control_index()
    print(f"control index: {len(control_index)} stems with a combined_reference CSV")

    populations = []
    for aligned_dir, tag, color in _VARIANTS:
        print(f"\n[{tag}]")
        d = harvest(aligned_dir, aoi, control_index)
        if d["x_off"].size == 0:
            continue
        print(
            f"  n={d['x_off'].size:3d}  "
            f"median (x,y,z)=({np.median(d['x_off']):+.2f}, "
            f"{np.median(d['y_off']):+.2f}, {np.median(d['z_off']):+.2f}) m  "
            f"post resid |p50|={np.median(d['out_p50']):.3f} m  "
            f"signed-bias median={np.median(d['out_bias']):+.3f} m"
        )
        populations.append((tag, color, d))
    if not populations:
        raise SystemExit("No populations to plot.")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.05), hspace=0.32, wspace=0.28)
    ax_yx = fig.add_subplot(gs[0, 0])
    ax_zx = fig.add_subplot(gs[0, 1])
    ax_zy = fig.add_subplot(gs[0, 2])
    ax_t = fig.add_subplot(gs[1, :])

    for tag, color, d in populations:
        label = f"{tag} (n={d['x_off'].size})"
        kw = dict(color=color, alpha=0.6, s=28, edgecolor="none")
        ax_yx.scatter(d["x_off"], d["y_off"], label=label, **kw)
        ax_zx.scatter(d["x_off"], d["z_off"], label=label, **kw)
        ax_zy.scatter(d["y_off"], d["z_off"], label=label, **kw)
    for ax, (xlab, ylab, title) in zip(
        (ax_yx, ax_zx, ax_zy),
        (
            ("x offset (East, m)", "y offset (North, m)", "y vs x offset"),
            ("x offset (East, m)", "z offset (Up, m)", "z vs x offset"),
            ("y offset (North, m)", "z offset (Up, m)", "z vs y offset"),
        ),
    ):
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)

    for tag, color, d in populations:
        valid = np.array([dt is not None for dt in d["dates"]])
        ts = np.array([dt for dt in d["dates"][valid]])
        out50 = d["out_p50"][valid]
        out_yerr = np.vstack([out50 - d["out_p16"][valid],
                              d["out_p84"][valid] - out50])
        ax_t.errorbar(ts, out50, yerr=out_yerr, fmt="o", color=color,
                      alpha=0.9, ms=6, mew=0.0, elinewidth=1.1, capsize=2.5)
        ax_t.axhline(float(out50.mean()), color=color, linestyle="--", lw=1.6)
        ax_t.set_title(
            f"post-alignment |resid| p50 vs time — mean {out50.mean():.2f} m, "
            f"signed-bias median {np.median(d['out_bias'][valid]):+.3f} m"
        )
    ax_t.set_ylabel("|DEM − control| p50 (m), 16–84% bars")
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_t.grid(True, alpha=0.25)

    out_png = config.FIGURES_DIR / "fig4_alignment_diagnostics.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
