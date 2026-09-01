"""Aggregate ASP geodiff stats for the Nansen IS2-control alignment run.

Mirrors :mod:`beardmore.compare_geodiff` but with a single variant
(``is2``) since Nansen only runs the IS2 control set.

Reads ``*-initial-diff.csv`` and ``*-final-diff.csv`` under
``<ASP_ROOT>/{initial,final}/`` (which ``align_strip_with_asp``
writes for every strip). The CSV headers carry ASP's pre-computed
median/mean/stddev/max/min; the data rows give us MAD and RMSE.

Emits one row per (variant, strip, stage) to
``nansen/results/geodiff_comparison_per_strip.csv`` and a per-variant
summary to ``geodiff_comparison_summary.csv``. ``nansen.qc_strips``
reads the per-strip table downstream.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from nansen import config

VARIANTS = {
    "is2": config.ASP_ROOT,
}
OUT_PER_STRIP = config.RESULTS_DIR / "geodiff_comparison_per_strip.csv"
OUT_SUMMARY = config.RESULTS_DIR / "geodiff_comparison_summary.csv"

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
    config.ensure_output_dirs()
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
    for v in VARIANTS:
        sub = final[final.variant == v].sort_values("median_m")
        if not len(sub):
            continue
        print(f"\n  [{v}]  n={len(sub)}")
        print(sub[["strip", "median_m", "mad_m", "rmse_m", "n"]].to_string(index=False, float_format=lambda x: f"{x: .3f}"))


if __name__ == "__main__":
    main()
