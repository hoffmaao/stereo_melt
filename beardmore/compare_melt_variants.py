"""Compare melt-rate inversions across IS2 / CS2 / IS2+CS2 ASP variants.

Reads the per-variant melt NetCDFs produced by `run_variant.py` (and
the canonical IS2-only run at `results/beardmore_melt_<window>.nc`) and
emits:

  - `results/melt_variant_summary.csv` — per-variant, per-solver
    summary stats (median, IQR, p5/p95) over the floating mask
  - `figures/melt_variant_comparison.png` — 3 (variant) x 3 (solver)
    map panels of melt rate, plus pairwise difference panels
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from beardmore import config

WINDOW_TAG = f"{config.START_TIME}_{config.END_TIME}"

SOURCES = {
    "is2":     config.BASIN_DIR / "results" / f"beardmore_melt_{WINDOW_TAG}.nc",
    "cs2":     config.BASIN_DIR / "results" / "cs2"     / f"beardmore_melt_{WINDOW_TAG}.nc",
    "is2+cs2": config.BASIN_DIR / "results" / "is2cs2"  / f"beardmore_melt_{WINDOW_TAG}.nc",
}

SOLVERS = ["melt_rate_eulerian", "melt_rate_lagrangian", "melt_rate_linear_inverse"]


def _stat_row(variant: str, solver: str, da: xr.DataArray) -> dict:
    arr = da.values
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"variant": variant, "solver": solver, "n_finite": 0}
    return {
        "variant": variant,
        "solver": solver,
        "n_finite": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def main() -> None:
    import pandas as pd

    rows = []
    panels = {}
    for variant, path in SOURCES.items():
        if not path.exists():
            print(f"  ⚠️ missing: {path}")
            continue
        ds = xr.open_dataset(path)
        for solver in SOLVERS:
            if solver in ds:
                da = ds[solver]
                rows.append(_stat_row(variant, solver, da))
                panels[(variant, solver)] = da
            else:
                print(f"  ⚠️ {variant}: no {solver} in {path.name}")

    df = pd.DataFrame(rows)
    out_csv = config.RESULTS_DIR.parent / "melt_variant_summary.csv"
    if (config.BASIN_DIR / "results" / "is2cs2").exists():
        out_csv = config.BASIN_DIR / "results" / "melt_variant_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v: .3f}"))

    if not panels:
        return

    variants = [v for v in SOURCES if any(k[0] == v for k in panels)]
    n_v = len(variants)
    n_s = len(SOLVERS)
    fig, axes = plt.subplots(n_v, n_s, figsize=(5 * n_s, 4.5 * n_v), constrained_layout=True)
    if n_v == 1:
        axes = np.atleast_2d(axes)

    vlim = 30.0
    for i, variant in enumerate(variants):
        for j, solver in enumerate(SOLVERS):
            ax = axes[i, j]
            da = panels.get((variant, solver))
            if da is None:
                ax.set_axis_off()
                continue
            im = ax.imshow(
                da.values,
                cmap="RdBu_r",
                origin="upper" if da["y"].values[0] > da["y"].values[-1] else "lower",
                vmin=-vlim,
                vmax=vlim,
                extent=[
                    float(da["x"].min()), float(da["x"].max()),
                    float(da["y"].min()), float(da["y"].max()),
                ],
                aspect="equal",
            )
            ax.set_title(f"{variant} — {solver.replace('melt_rate_', '')}")
            ax.set_xticks([])
            ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="m ice/yr")
    fig.suptitle(f"Beardmore melt rate — variant × solver  ({WINDOW_TAG})", fontsize=13)
    out_fig = config.BASIN_DIR / "figures" / "melt_variant_comparison.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
