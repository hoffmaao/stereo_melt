"""Stage 2 of the CS2 GCP integration: CS2 vs IS2 cross-validation over Beardmore.

Reads the per-strip IS2 ASP CSVs and CS2 PointCollection HDF5 caches that
``cache_icesat2`` and ``cache_cryosat2`` have already populated, collocates
them at 750 m / ±30 days, stratifies the residuals by surface class, slope,
and year, and decides whether CS2 is safe to use as a second GCP source.

Outputs:
    beardmore/results/cs2_validation_residuals.csv
    beardmore/results/cs2_validation_summary.csv
    beardmore/figures/cs2_vs_is2_residuals.png

Run:
    python -m beardmore.validate_cs2
    python -m beardmore.validate_cs2 --radius-m 1000 --time-window-days 45
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Point pyproj at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from stereo_melt.validation.cs2_vs_is2 import (
    PassCriteria,
    StratificationContext,
    plot_residuals,
    run_validation,
)

from beardmore import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-m", type=float, default=750.0,
                        help="spatial collocation radius (m)")
    parser.add_argument("--time-window-days", type=float, default=30.0,
                        help="temporal half-window (days)")
    parser.add_argument("--bias-max-m", type=float, default=3.0,
                        help="max |median| residual on the gate class (m)")
    parser.add_argument("--mad-max-m-grounded", type=float, default=8.0,
                        help="max MAD on the gate class (m)")
    parser.add_argument("--yoy-drift-max-m-per-yr", type=float, default=1.5,
                        help="max |Δmedian| between consecutive years on the gate class (m/yr)")
    parser.add_argument("--gate-class", type=str, default="fast_grounded",
                        choices=("fast_grounded", "slow_grounded"),
                        help="surface class the pass gates evaluate (default fast_grounded)")
    parser.add_argument("--peakiness-min", type=float, default=1.5,
                        help="drop CS2 points with peakiness below this (None to disable)")
    parser.add_argument("--retracker-quality-min", type=int, default=1,
                        help="drop CS2 points with retracker_quality below this; 0 means retrack failed")
    args = parser.parse_args()

    asp_root = Path(config.STRIPS_DIR) / "ASP"
    is2_dir = asp_root / "icesat2_data"
    cs2_dir = asp_root / "cs2_data"

    # MEaSUREs phase map is the velocity layer the rest of the pipeline
    # uses (ITS_LIVE has coverage gaps over the Beardmore grounding zone).
    ctx = StratificationContext(
        bedmachine_path=Path(config.BEDMACHINE_NC),
        velocity_path=Path(config.MEASURES_PHASE_NC),
        velocity_var="VX",
        velocity_var_y="VY",
    )
    pc = PassCriteria(
        bias_max_m=args.bias_max_m,
        mad_max_m_grounded=args.mad_max_m_grounded,
        yoy_drift_max_m_per_yr=args.yoy_drift_max_m_per_yr,
        gate_class=args.gate_class,
    )

    print(f"== CS2 vs IS2 validation (Beardmore, {config.START_TIME}..{config.END_TIME}) ==")
    print(f"   IS2 cache: {is2_dir}")
    print(f"   CS2 cache: {cs2_dir}")
    print(f"   collocation: {args.radius_m:.0f} m / ±{args.time_window_days:.0f} d")
    print(f"   CS2 quality: peakiness > {args.peakiness_min}, retr_q >= {args.retracker_quality_min}")
    print(f"   gate: |bias|<{pc.bias_max_m} m, MAD<{pc.mad_max_m_grounded} m, "
          f"YoY drift<{pc.yoy_drift_max_m_per_yr} m/yr  (on {pc.gate_class})")
    result = run_validation(
        is2_dir, cs2_dir, ctx,
        radius_m=args.radius_m,
        time_window_days=args.time_window_days,
        pass_criteria=pc,
        peakiness_min=args.peakiness_min,
        retracker_quality_min=args.retracker_quality_min,
    )

    results_dir = config.MAIN_DIR / "beardmore" / "results"
    figures_dir = config.MAIN_DIR / "beardmore" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    res_csv = results_dir / "cs2_validation_residuals.csv"
    sum_csv = results_dir / "cs2_validation_summary.csv"
    fig_png = figures_dir / "cs2_vs_is2_residuals.png"

    result.residuals.to_csv(res_csv, index=False)
    result.by_class.to_csv(sum_csv, index=False)
    plot_residuals(
        result.residuals, result.by_class, result.by_class_slope,
        result.by_year, fig_png,
    )

    print()
    print("== Per-class summary ==")
    print(result.by_class.to_string(index=False))
    print()
    print("== Per-year summary ==")
    print(result.by_year.to_string(index=False))
    print()
    print("== Decision ==")
    print(json.dumps({
        "passes_all": result.decision["passes_all"],
        "fails": [{"check": f["check"]} for f in result.decision["fails"]],
        "yoy_max_diff_m": result.decision.get("yoy_max_diff_m"),
    }, indent=2, default=str))
    print()
    print(f"  → residuals: {res_csv}")
    print(f"  → summary:   {sum_csv}")
    print(f"  → figure:    {fig_png}")

    if not result.decision["passes_all"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
