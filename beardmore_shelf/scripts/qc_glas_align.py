"""Per-strip QC screen for an ASP align root (default ``data/ASP_glas``).

``--root <NAME>`` points the same screen at any per-basin align root that
carries the standard evidence layout (``initial/``, ``final/``,
``asp_aligned/``) — e.g. ``--root ASP_ctempoatm`` for the pre-IS2 era.
Outputs are keyed by the root name (``logs/qc_<slug>_align.csv``,
``figures/qc_<slug>_align.png``). The ``--glas-sample`` laser residual is
GLAS-root-specific and silently no-ops where no per-strip GLAS cache exists.

For every aligned strip, aggregates the on-disk alignment evidence:

- ``initial/<s>-initial-diff.csv``  -> before-median vs combined control
- ``final/<s>-final-diff.csv``      -> after-median, core-MAD, control count
- ``asp_aligned/<s>-log-pc_align-*``-> pc_align translation magnitude
- ``asp_aligned/<s>.sources.json``  -> control sources used
- optional ``--glas-sample``        -> sample the aligned DEM at the cached
  GLAS points (``ASP/glas_data/glas_filtered_<s>.csv``) for a laser-only
  residual, separating the ICESat-1 datum check from the rock control that
  dominates the combined-CSV point count.

Core-MAD is 1.4826*MAD over diffs within +-10 m of the median: the combined
control includes steep-rock points whose raw spread (tens of m at 2 m
posting on cliff faces) swamps the alignment signal.

Screening gates (Shean-style bias screen; the translation cap already ran at
align time inside ``align_strip_with_asp``):

- QUARANTINE: ``|after_med| > --max-after-med`` (default 2.0 m) or
  ``n_ctl < --min-ctl`` (default 50) — misconverged or under-determined.
- WATCH: ``--watch-med`` (1.0) < ``|after_med|`` <= max — kept; the stack
  tilt LSQ absorbs meter-level datum residuals via per-epoch alpha-z.

Default is report-only. ``--apply`` moves QUARANTINE strips to
``bad_align/<s>/`` via the same library helper the align-time cap uses
(leaves a ``.bad_align`` sentinel so re-runs skip them).

Run (report, then apply once happy):

    cd /wd2/projects/stereo_melt
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        -m beardmore_shelf.scripts.qc_glas_align --glas-sample
    ... review table/figure ...
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python \
        -m beardmore_shelf.scripts.qc_glas_align --apply
"""
from __future__ import annotations

import stereo_melt.envsetup  # noqa: F401  (PROJ_DATA fix, must precede rasterio)

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stereo_melt.coregister.asp import (
    _quarantine_alignment,
    _read_translation_magnitude,
)

from beardmore_shelf import config

MED_PAT = re.compile(r"# Median difference:\s+([-\d\.eE+]+)")


def geodiff_stats(path: Path) -> tuple[float, float, int]:
    """(median, core_mad, n) from a geodiff CSV; header median if present."""
    med_hdr = np.nan
    diffs: list[float] = []
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                m = MED_PAT.search(line)
                if m:
                    med_hdr = float(m.group(1))
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    diffs.append(float(parts[2]))
                except ValueError:
                    pass
    arr = np.asarray(diffs, float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return np.nan, np.nan, 0
    med = med_hdr if np.isfinite(med_hdr) else float(np.median(arr))
    core = arr[np.abs(arr - med) <= 10.0]
    mad = float(1.4826 * np.median(np.abs(core - np.median(core)))) if core.size else np.nan
    return med, mad, int(arr.size)


def sample_glas(dem_path: Path, csv_path: Path) -> tuple[float, float, int]:
    """Median/MAD/n of (DEM - GLAS h_mean) at the cached laser points."""
    import rasterio

    pts = pd.read_csv(csv_path)
    if not len(pts):
        return np.nan, np.nan, 0
    with rasterio.open(dem_path) as src:
        nod = src.nodata
        vals = np.array(
            [v[0] for v in src.sample(zip(pts["easting"], pts["northing"]))],
            dtype=float,
        )
    if nod is not None:
        vals[vals == nod] = np.nan
    d = vals - pts["h_mean"].to_numpy(float)
    d = d[np.isfinite(d)]
    if not d.size:
        return np.nan, np.nan, 0
    med = float(np.median(d))
    return med, float(1.4826 * np.median(np.abs(d - med))), int(d.size)


def collect(root: Path, glas_sample: bool) -> pd.DataFrame:
    glas_dir = Path(config.ASP_ROOT) / "glas_data"
    rows = []
    finals = sorted((root / "final").glob("*-final-diff.csv"))
    for k, fcsv in enumerate(finals):
        sid = fcsv.name.replace("-final-diff.csv", "")
        a_med, a_mad, a_n = geodiff_stats(fcsv)
        icsv = root / "initial" / f"{sid}-initial-diff.csv"
        b_med = geodiff_stats(icsv)[0] if icsv.exists() else np.nan
        tr = _read_translation_magnitude(str(root / "asp_aligned" / sid))
        src = ""
        sj = root / "asp_aligned" / f"{sid}.sources.json"
        if sj.exists():
            try:
                src = "+".join(json.loads(sj.read_text()).get("sources", []))
            except Exception:
                src = "?"
        g_med = g_mad = np.nan
        g_n = 0
        dem = root / "asp_aligned" / f"{sid}-trans_reference-DEM.tif"
        gcsv = glas_dir / f"glas_filtered_{sid}.csv"
        if glas_sample and dem.exists() and gcsv.exists():
            try:
                g_med, g_mad, g_n = sample_glas(dem, gcsv)
            except Exception as e:  # non-fatal: QC keeps going
                print(f"  !! glas-sample failed for {sid}: {e}")
        parts = sid.split("_")
        rows.append(dict(
            strip=sid, date=parts[3], cat1=parts[4], cat2=parts[5],
            before_med=b_med, after_med=a_med, core_mad=a_mad, n_ctl=a_n,
            translation_m=tr if tr is not None else np.nan, sources=src,
            glas_med=g_med, glas_mad=g_mad, glas_n=g_n,
            dem_on_disk=dem.exists(),
        ))
        if glas_sample and (k + 1) % 50 == 0:
            print(f"  ... {k + 1}/{len(finals)} strips collected")
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, out_png: Path, max_after: float,
                title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.hist(df.before_med.clip(-40, 40), bins=60, color="tab:gray")
    ax.set_title(f"before-median vs control (n={len(df)})")
    ax.set_xlabel("m")
    ax = axes[0, 1]
    ax.hist(df.after_med.clip(-5, 5), bins=60, color="tab:blue", label="combined")
    if df.glas_n.gt(0).any():
        ax.hist(df.loc[df.glas_n > 0, "glas_med"].clip(-5, 5), bins=60,
                color="tab:orange", alpha=0.6, label="GLAS-only")
        ax.legend(fontsize=8)
    for s in (-max_after, max_after):
        ax.axvline(s, color="r", ls="--", lw=0.8)
    n_out = int((df.after_med.abs() > 5).sum())
    ax.set_title(f"after-median (clipped +-5; {n_out} beyond)")
    ax.set_xlabel("m")
    ax = axes[1, 0]
    g = df.groupby("date").agg(n=("strip", "size"),
                               bad=("after_med", lambda s: int((s.abs() > max_after).sum())))
    g.plot.bar(ax=ax, color=["tab:blue", "tab:red"])
    ax.set_title("strips per date (red = beyond gate)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax = axes[1, 1]
    ax.scatter(df.n_ctl.clip(lower=1), df.after_med.clip(-20, 20), s=8, alpha=0.5)
    ax.set_xscale("log")
    ax.axhline(max_after, color="r", ls="--", lw=0.8)
    ax.axhline(-max_after, color="r", ls="--", lw=0.8)
    ax.set_xlabel("n control points")
    ax.set_ylabel("after-median (m)")
    ax.set_title("bias vs control count")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-after-med", type=float, default=2.0,
                   help="quarantine gate on |after-median| vs combined control, m")
    p.add_argument("--min-ctl", type=int, default=50,
                   help="quarantine gate on control-point count")
    p.add_argument("--watch-med", type=float, default=1.0,
                   help="report-only WATCH band lower edge, m")
    p.add_argument("--glas-sample", action="store_true",
                   help="also sample each aligned DEM at the cached GLAS points")
    p.add_argument("--root", type=str, default="ASP_glas",
                   help="align-root directory name under <basin>/data/ "
                        "(default ASP_glas; e.g. ASP_ctempoatm)")
    p.add_argument("--apply", action="store_true",
                   help="quarantine gated strips (default: report only)")
    args = p.parse_args()

    root = Path(config.BASIN_DIR) / "data" / args.root
    if not (root / "final").exists():
        raise SystemExit(f"no final/ geodiff dir under {root}")
    slug = args.root.lower().removeprefix("asp_") or "asp"
    df = collect(root, args.glas_sample)
    if df.empty:
        print("no strips with final geodiff under", root)
        return

    bad = (df.after_med.abs() > args.max_after_med) | (df.n_ctl < args.min_ctl)
    watch = ~bad & (df.after_med.abs() > args.watch_med)
    df["screen"] = np.where(bad, "QUARANTINE", np.where(watch, "WATCH", "ok"))

    out_csv = Path(config.BASIN_DIR) / "logs" / f"qc_{slug}_align.csv"
    df.to_csv(out_csv, index=False)
    out_png = Path(config.FIGURES_DIR) / f"qc_{slug}_align.png"
    make_figure(df, out_png, args.max_after_med,
                title=f"{args.root} align QC")

    am = df.after_med.to_numpy()
    print(f"\n=== {args.root} align QC: {len(df)} strips ===")
    print(f"after-median: med {np.nanmedian(am):+.3f}  "
          f"MAD {1.4826 * np.nanmedian(np.abs(am - np.nanmedian(am))):.3f}  "
          f"p05 {np.nanpercentile(am, 5):+.3f}  p95 {np.nanpercentile(am, 95):+.3f} m")
    print(f"|after-median| bands: <=0.5m {int((np.abs(am) <= 0.5).sum())}  "
          f"0.5-1m {int(((np.abs(am) > 0.5) & (np.abs(am) <= 1)).sum())}  "
          f"1-2m {int(((np.abs(am) > 1) & (np.abs(am) <= 2)).sum())}  "
          f">2m {int((np.abs(am) > 2).sum())}")
    if df.glas_n.gt(0).any():
        gm = df.loc[df.glas_n > 0, "glas_med"]
        print(f"GLAS-only residual ({len(gm)} strips): med {gm.median():+.3f}  "
              f"MAD {1.4826 * float(np.median(np.abs(gm - gm.median()))):.3f} m  "
              f"(per-strip MAD med {df.loc[df.glas_n > 0, 'glas_mad'].median():.2f})")
    print(f"screen: ok {int((df.screen == 'ok').sum())}  "
          f"WATCH {int(watch.sum())}  QUARANTINE {int(bad.sum())}")
    print(f"per-strip table -> {out_csv}\nfigure -> {out_png}")

    already = sorted(q.name for q in (root / "bad_align").glob("*") if q.is_dir())
    print(f"already quarantined at align time: {len(already)}")

    qdf = df[bad].sort_values("after_med", key=lambda s: s.abs(), ascending=False)
    if len(qdf):
        print("\nQUARANTINE list:")
        print(qdf[["strip", "before_med", "after_med", "n_ctl", "translation_m", "glas_med"]]
              .to_string(index=False, float_format=lambda x: f"{x: .2f}"))
    wdf = df[watch].sort_values("after_med", key=lambda s: s.abs(), ascending=False)
    if len(wdf):
        print(f"\nWATCH ({args.watch_med}-{args.max_after_med} m, kept for tilt LSQ):")
        print(wdf[["strip", "before_med", "after_med", "n_ctl", "glas_med"]]
              .head(30).to_string(index=False, float_format=lambda x: f"{x: .2f}"))

    if not args.apply:
        print("\nreport-only (pass --apply to quarantine)")
        return
    print("\napplying quarantine...")
    for _, r in qdf.iterrows():
        reason = (f"qc_glas_align screen: after_med={r.after_med:+.2f} m "
                  f"(gate {args.max_after_med}), n_ctl={int(r.n_ctl)} "
                  f"(gate {args.min_ctl})")
        qdir = _quarantine_alignment(
            str(root / "asp_aligned" / r.strip), str(root), r.strip, reason
        )
        print(f"  -> {qdir}")
    print(f"quarantined {len(qdf)} strips")


if __name__ == "__main__":
    main()
