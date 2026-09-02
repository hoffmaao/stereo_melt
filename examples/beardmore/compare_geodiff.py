"""Aggregate ASP geodiff stats across IS2 / CS2 / IS2+CS2 control variants.

Reads `*-initial-diff.csv` and `*-final-diff.csv` headers (which carry
ASP's pre-computed median/mean/stddev/max/min) and the data rows (to
compute MAD and RMSE). Emits one CSV row per (variant, strip, stage) and
prints per-variant aggregates.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

STRIPS_ROOT = Path("/wd2/projects/stereo_melt/data/REMA/strips")
VARIANTS = {
    "is2": STRIPS_ROOT / "ASP",
    "cs2": STRIPS_ROOT / "ASP_cs2",
    "is2+cs2": STRIPS_ROOT / "ASP_is2cs2",
}
OUT_PER_STRIP = Path("beardmore/results/geodiff_comparison_per_strip.csv")
OUT_SUMMARY = Path("beardmore/results/geodiff_comparison_summary.csv")

HDR_PATTERNS = {
    "max_m": re.compile(r"# Max difference:\s+([-\d\.eE+]+)"),
    "min_m": re.compile(r"# Min difference:\s+([-\d\.eE+]+)"),
    "mean_m": re.compile(r"# Mean difference:\s+([-\d\.eE+]+)"),
    "std_m": re.compile(r"# StdDev of difference:\s+([-\d\.eE+]+)"),
    "median_m": re.compile(r"# Median difference:\s+([-\d\.eE+]+)"),
}


def parse_geodiff(path: Path) -> dict:
    """Read header stats and compute MAD + RMSE from the data rows."""
    stats = {k: np.nan for k in HDR_PATTERNS}
    diffs: list[float] = []
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                for key, pat in HDR_PATTERNS.items():
                    m = pat.search(line)
                    if m:
                        stats[key] = float(m.group(1))
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    diffs.append(float(parts[2]))
                except ValueError:
                    pass
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size:
        med = stats["median_m"]
        if not np.isfinite(med):
            med = float(np.median(arr))
            stats["median_m"] = med
        stats["mad_m"] = float(np.median(np.abs(arr - med)))
        stats["rmse_m"] = float(np.sqrt(np.mean(arr * arr)))
        stats["n"] = int(arr.size)
    else:
        stats["mad_m"] = np.nan
        stats["rmse_m"] = np.nan
        stats["n"] = 0
    return stats


def strip_id(path: Path, suffix: str) -> str:
    return path.name.replace(suffix, "")


def collect() -> pd.DataFrame:
    rows = []
    for variant, root in VARIANTS.items():
        for stage, suffix in (("initial", "-initial-diff.csv"), ("final", "-final-diff.csv")):
            stage_dir = root / stage
            if not stage_dir.exists():
                continue
            for csv_path in sorted(stage_dir.glob(f"*{suffix}")):
                stats = parse_geodiff(csv_path)
                rows.append(
                    {
                        "variant": variant,
                        "stage": stage,
                        "strip": strip_id(csv_path, suffix),
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["variant", "stage"])
        .agg(
            n_strips=("strip", "nunique"),
            median_of_strip_medians=("median_m", "median"),
            mad_of_strip_medians=("median_m", lambda s: float(np.median(np.abs(s - np.median(s))))),
            mean_strip_mad=("mad_m", "mean"),
            mean_strip_rmse=("rmse_m", "mean"),
            mean_strip_std=("std_m", "mean"),
            max_abs_strip_median=("median_m", lambda s: float(np.max(np.abs(s)))),
        )
        .reset_index()
    )
    return g


def main() -> None:
    df = collect()
    df.to_csv(OUT_PER_STRIP, index=False)
    summary = summarize(df)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Wrote {len(df)} per-strip rows to {OUT_PER_STRIP}")
    print(f"Wrote {len(summary)} summary rows to {OUT_SUMMARY}")
    print()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print()
    print("Per-variant final-stage strip medians (sorted):")
    final = df[df.stage == "final"].copy()
    for v in ["is2", "cs2", "is2+cs2"]:
        sub = final[final.variant == v].sort_values("median_m")
        if not len(sub):
            continue
        print(f"\n  [{v}]  n={len(sub)}")
        print(sub[["strip", "median_m", "mad_m", "rmse_m", "n"]].to_string(index=False, float_format=lambda x: f"{x: .3f}"))


if __name__ == "__main__":
    main()
