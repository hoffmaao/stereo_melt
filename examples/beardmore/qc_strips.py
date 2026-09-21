"""Hard-reject DEM strips with poor pc_align residuals.

Reads ``results/geodiff_comparison_per_strip.csv`` (the per-variant
post-pc_align residual table emitted by ``compare_geodiff.py``) and
writes ``results/strips_rejected_<variant>.csv`` listing every strip
that fails one or more thresholds, with the reason and the metric/
threshold values so the cut is auditable.

Cuts (default thresholds; override on the CLI):

  - ``--min-n``        Minimum number of pc_align control points. Strips
                       with fewer cannot constrain a 6-DOF transform; the
                       residual statistics are not meaningful. Default 50.

  - ``--max-median``   Maximum |post-align median (m)|. Tags strips with
                       a residual *vertical* bias that pc_align failed
                       to remove. Default 0.5 m. Note the per-epoch αz
                       term in ``fit_tilt_stack`` will absorb a uniform
                       vertical offset cleanly, so this cut is more of a
                       sanity check than a hard requirement.

  - ``--max-rmse``     Maximum post-align RMSE (m). Catches strips with
                       residual *spatial* structure (tilt curvature,
                       cross-track distortion) beyond what αx, αy, αz
                       can capture. Default 2.0 m.

  - ``--mad-mult``     If non-zero, also reject strips whose ``rmse_m``
                       exceeds ``cohort_median + mad_mult · MAD``.
                       Cohort = strips of the same variant in the
                       ``final`` stage. Default 3.0.

Output columns: ``strip, variant, n, median_m, rmse_m, reason, threshold,
value``.

Run::

    python -m beardmore.qc_strips --variant is2
    python -m beardmore.qc_strips --variant is2 --min-n 30 --max-rmse 1.5

Then re-run ``beardmore.build_stack`` for that variant; it auto-discovers
the rejection CSV and skips matching strips.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from beardmore import config
from beardmore.compare_geodiff import VARIANTS

DEFAULT_GEODIFF_CSV = (
    Path(config.BASIN_DIR) / "results" / "geodiff_comparison_per_strip.csv"
)
DEFAULT_OUT_DIR = Path(config.BASIN_DIR) / "results"


def _cohort_mad_threshold(values: np.ndarray, mult: float) -> float:
    """Return ``median(values) + mult · 1.4826 · MAD(values)`` (NaN-safe)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("inf")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med + mult * 1.4826 * mad


def reject_strips(
    df: pd.DataFrame,
    variant: str,
    min_n: int,
    max_median: float,
    max_rmse: float,
    mad_mult: float,
) -> pd.DataFrame:
    """Apply thresholds; return one row per (strip, reason) rejection.

    A strip can be rejected by multiple criteria — each one becomes a
    separate row so the ``reason`` column never aggregates.
    """
    sub = df[(df["variant"] == variant) & (df["stage"] == "final")].copy()
    if sub.empty:
        raise SystemExit(
            f"no rows for variant={variant!r} stage='final' in geodiff CSV"
        )

    rmse_thresh = max_rmse
    if mad_mult > 0:
        cohort = _cohort_mad_threshold(sub["rmse_m"].to_numpy(), mad_mult)
        rmse_thresh = min(rmse_thresh, cohort)

    rejections: list[dict] = []
    for _, r in sub.iterrows():
        strip = str(r["strip"])
        n = int(r.get("n", 0)) if pd.notna(r.get("n")) else 0
        med = float(r["median_m"]) if pd.notna(r["median_m"]) else float("nan")
        rmse = float(r["rmse_m"]) if pd.notna(r["rmse_m"]) else float("nan")

        if n < min_n:
            rejections.append(
                {
                    "strip": strip,
                    "variant": variant,
                    "n": n,
                    "median_m": med,
                    "rmse_m": rmse,
                    "reason": "low_n",
                    "threshold": min_n,
                    "value": n,
                }
            )
        if np.isfinite(med) and abs(med) > max_median:
            rejections.append(
                {
                    "strip": strip,
                    "variant": variant,
                    "n": n,
                    "median_m": med,
                    "rmse_m": rmse,
                    "reason": "high_median",
                    "threshold": max_median,
                    "value": abs(med),
                }
            )
        # NaN rmse counts as a rejection (no useful residual = pc_align
        # didn't actually align against control).
        if not np.isfinite(rmse):
            rejections.append(
                {
                    "strip": strip,
                    "variant": variant,
                    "n": n,
                    "median_m": med,
                    "rmse_m": rmse,
                    "reason": "no_rmse",
                    "threshold": float("nan"),
                    "value": float("nan"),
                }
            )
        elif rmse > rmse_thresh:
            rejections.append(
                {
                    "strip": strip,
                    "variant": variant,
                    "n": n,
                    "median_m": med,
                    "rmse_m": rmse,
                    "reason": "high_rmse",
                    "threshold": rmse_thresh,
                    "value": rmse,
                }
            )

    return pd.DataFrame(
        rejections,
        columns=[
            "strip",
            "variant",
            "n",
            "median_m",
            "rmse_m",
            "reason",
            "threshold",
            "value",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", default="is2", choices=list(VARIANTS),
        help="ASP variant to QC (default: is2).",
    )
    parser.add_argument(
        "--geodiff-csv", default=str(DEFAULT_GEODIFF_CSV),
        help=f"Per-strip geodiff CSV (default: {DEFAULT_GEODIFF_CSV}).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path. Default: results/strips_rejected_<variant>.csv.",
    )
    parser.add_argument(
        "--min-n", type=int, default=50,
        help="Minimum control-point count (default 50).",
    )
    parser.add_argument(
        "--max-median", type=float, default=0.5,
        help="Maximum |post-align median| in m (default 0.5).",
    )
    parser.add_argument(
        "--max-rmse", type=float, default=2.0,
        help="Maximum post-align RMSE in m (default 2.0).",
    )
    parser.add_argument(
        "--mad-mult", type=float, default=3.0,
        help="Cohort-MAD multiplier; tighter of (max-rmse, cohort+m·MAD) "
             "applies. 0 disables the cohort cut. Default 3.0.",
    )
    args = parser.parse_args()

    csv_path = Path(args.geodiff_csv)
    if not csv_path.is_file():
        raise SystemExit(f"geodiff CSV not found: {csv_path}")

    out_path = (
        Path(args.output)
        if args.output is not None
        else DEFAULT_OUT_DIR / f"strips_rejected_{args.variant.replace('+', '')}.csv"
    )

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Variant: {args.variant}, stage='final'")
    sub = df[(df["variant"] == args.variant) & (df["stage"] == "final")]
    print(f"  cohort: {len(sub)} strips")
    if not sub.empty:
        print(
            f"  rmse cohort: median={sub['rmse_m'].median():.3f}  "
            f"p90={sub['rmse_m'].quantile(0.9):.3f}  "
            f"max={sub['rmse_m'].max():.3f}"
        )
    print(
        f"Thresholds: min_n={args.min_n}, max_median={args.max_median} m, "
        f"max_rmse={args.max_rmse} m, mad_mult={args.mad_mult}"
    )

    rej = reject_strips(
        df,
        variant=args.variant,
        min_n=args.min_n,
        max_median=args.max_median,
        max_rmse=args.max_rmse,
        mad_mult=args.mad_mult,
    )

    n_unique = rej["strip"].nunique() if not rej.empty else 0
    print(f"\n{n_unique} unique strip(s) rejected out of {len(sub)}:")
    if rej.empty:
        print("  (none)")
    else:
        for strip, grp in rej.groupby("strip"):
            reasons = ", ".join(
                f"{row['reason']}={row['value']:g}>{row['threshold']:g}"
                if np.isfinite(row["threshold"])
                else f"{row['reason']}"
                for _, row in grp.iterrows()
            )
            short = strip[: 60] + ("…" if len(strip) > 60 else "")
            print(f"  {short:<63s}  {reasons}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rej.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
