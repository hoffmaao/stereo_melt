"""Plot geodiff comparison across IS2 / CS2 / IS2+CS2 control variants.

Reads the per-strip CSV produced by `compare_geodiff.py` and produces a
two-panel figure: (left) per-strip final medians, (right) per-strip MAD.
Strips are ordered by date, color-coded by variant. Saved to
`beardmore/figures/geodiff_comparison.png`.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PER_STRIP = Path("beardmore/results/geodiff_comparison_per_strip.csv")
OUT_FIG = Path("beardmore/figures/geodiff_comparison.png")

DATE_RE = re.compile(r"_(\d{8})_")


def strip_date(s: str) -> pd.Timestamp:
    m = DATE_RE.search(s)
    return pd.Timestamp(m.group(1)) if m else pd.NaT


def main() -> None:
    df = pd.read_csv(PER_STRIP)
    df["date"] = df["strip"].map(strip_date)
    final = df[df.stage == "final"].copy().sort_values("date")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"is2": "#1f77b4", "cs2": "#d62728", "is2+cs2": "#2ca02c"}

    # Panel 1: per-strip final medians
    ax = axes[0]
    for variant, color in colors.items():
        sub = final[final.variant == variant]
        ax.scatter(sub["date"], sub["median_m"], s=40, color=color, label=variant, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("Per-strip final median (m), symlog")
    ax.set_title("Final geodiff median — alignment quality")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 2: per-strip MAD
    ax = axes[1]
    for variant, color in colors.items():
        sub = final[final.variant == variant]
        ax.scatter(sub["date"], sub["mad_m"], s=40, color=color, label=variant, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("Per-strip final MAD (m)")
    ax.set_title("Final geodiff MAD — within-strip residual scatter")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Beardmore ASP coregistration: IS2 vs CS2 vs IS2+CS2 control", fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
