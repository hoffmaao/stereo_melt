"""Compare per-strip tilt-fit scalars across IS2 / CS2 / IS2+CS2 ASP variants.

The Shean LSQ tilt-fit produces, per strip:

  - tilt_dz  : scalar vertical offset (m) — the residual mean offset
                ASP pc_align didn't remove, captured by the planar fit.
  - tilt_dx  : scalar x-slope (m / m) — planar tilt along x.
  - tilt_dy  : scalar y-slope (m / m) — planar tilt along y.
  - fit_xy   : RMS misfit metric of the planar fit.

If CS2-only ASP coregistration carries a vertical bias relative to
IS2-only ASP, it shows up directly as a systematic shift in `tilt_dz`
between the two variants on shared strip dates.

Outputs:
  figures/tilt_dz_per_epoch_variants.png  — per-epoch dz timeseries
  figures/tilt_dz_scatter_variants.png    — pairwise scatters (shared dates)
  figures/tilt_dxdy_scatter_variants.png  — same for the x/y slopes
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from beardmore import config


_REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
VARIANTS = {
    "IS2":     f"{_REPO}/examples/beardmore/processed/beardmore_tilt_params_2019-01-01_2023-03-01.nc",
    "CS2":     f"{_REPO}/examples/beardmore/processed/cs2/beardmore_tilt_params_2019-01-01_2023-03-01.nc",
    "IS2+CS2": f"{_REPO}/examples/beardmore/processed/is2cs2/beardmore_tilt_params_2019-01-01_2023-03-01.nc",
}

COLORS = {"IS2": "tab:blue", "CS2": "tab:red", "IS2+CS2": "tab:green"}
MARKERS = {"IS2": "o", "CS2": "s", "IS2+CS2": "D"}


def _to_df(ds: xr.Dataset) -> pd.DataFrame:
    times = pd.to_datetime(ds["time"].values)
    return pd.DataFrame({
        "tilt_dz": ds.tilt_dz.values,
        "tilt_dx": ds.tilt_dx.values,
        "tilt_dy": ds.tilt_dy.values,
        "fit_xy":  ds.fit_xy.values,
        "weight_mean":      ds.weight_mean.values,
        "weight_frac_kept": ds.weight_frac_kept.values,
    }, index=times)


def _scatter(ax, a_df, b_df, a_label, b_label, col, units):
    common = sorted(set(a_df.index.normalize()) & set(b_df.index.normalize()))
    if not common:
        ax.set_title(f"{a_label} vs {b_label}: NO COMMON DATES")
        return
    a = [float(a_df.loc[a_df.index.normalize() == d, col].values[0]) for d in common]
    b = [float(b_df.loc[b_df.index.normalize() == d, col].values[0]) for d in common]
    a, b = np.array(a), np.array(b)
    ax.scatter(a, b, c="tab:purple", s=40, alpha=0.85)
    lim = max(np.max(np.abs(np.concatenate([a, b]))), 1e-6) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7, alpha=0.6)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.axvline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"{a_label}  {col} ({units})")
    ax.set_ylabel(f"{b_label}  {col} ({units})")
    diff = b - a
    ax.set_title(
        f"{a_label} → {b_label}  (n={len(common)})\n"
        f"mean diff={np.mean(diff):.3g}  std={np.std(diff):.3g} {units}"
    )
    ax.grid(alpha=0.3)


def main() -> None:
    config.ensure_output_dirs()

    summaries = {}
    for label, path in VARIANTS.items():
        if not Path(path).exists():
            raise SystemExit(f"missing {path}")
        ds = xr.open_dataset(path)
        summaries[label] = _to_df(ds)
        print(f"\n{label}: n_epochs={len(summaries[label])}")
        print(summaries[label].round(4))

    # ---------- Figure 1: per-epoch tilt_dz / fit_xy / weight_frac_kept ----------
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True, sharex=True)
    for label, df in summaries.items():
        axes[0].plot(df.index, df["tilt_dz"], "-" + MARKERS[label],
                     color=COLORS[label], label=label, markersize=6)
        axes[1].plot(df.index, df["fit_xy"], "-" + MARKERS[label],
                     color=COLORS[label], label=label, markersize=6)
        axes[2].plot(df.index, df["weight_frac_kept"], "-" + MARKERS[label],
                     color=COLORS[label], label=label, markersize=6)
    axes[0].axhline(0, color="k", lw=0.5, alpha=0.5)
    axes[0].set_ylabel("tilt_dz (m)\n(LSQ-fit residual\nvertical offset)")
    axes[0].set_title("Per-strip residual vertical offset by ASP variant")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("fit_xy RMS (m)")
    axes[1].set_title("Per-strip planar-fit RMS misfit (smaller = cleaner)")
    axes[1].grid(alpha=0.3)
    axes[2].set_ylabel("weight_frac_kept")
    axes[2].set_title("Per-strip IRLS weight fraction kept (1 = no outliers; smaller = more clipped)")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.3)
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].set_xlabel("strip date")
    fig.suptitle("Beardmore: tilt-fit scalars across ASP coregistration variants", fontsize=12)
    out1 = config.BASIN_DIR / "figures" / "tilt_dz_per_epoch_variants.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out1}")

    # ---------- Figure 2: pairwise scatters of tilt_dz ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    pairs = [
        ("IS2", "CS2"),
        ("IS2", "IS2+CS2"),
        ("CS2", "IS2+CS2"),
    ]
    for ax, (a, b) in zip(axes, pairs):
        _scatter(ax, summaries[a], summaries[b], a, b, "tilt_dz", "m")
    fig.suptitle(
        "Per-strip vertical-offset comparison: paired ASP-variant scatter",
        fontsize=12,
    )
    out2 = config.BASIN_DIR / "figures" / "tilt_dz_scatter_variants.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out2}")

    # ---------- Figure 3: pairwise scatters of tilt_dx, tilt_dy ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for col_idx, col_name in enumerate(["tilt_dx", "tilt_dy"]):
        for ax, (a, b) in zip(axes[col_idx], pairs):
            _scatter(ax, summaries[a], summaries[b], a, b, col_name, "m/m")
    fig.suptitle(
        "Per-strip planar-tilt slope comparison (x and y) across ASP variants",
        fontsize=12,
    )
    out3 = config.BASIN_DIR / "figures" / "tilt_dxdy_scatter_variants.png"
    fig.savefig(out3, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out3}")

    # ---------- Console: aggregate stats ----------
    print("\n=== aggregate stats ===")
    for label, df in summaries.items():
        for col in ["tilt_dz", "tilt_dx", "tilt_dy", "fit_xy"]:
            v = df[col].dropna().values
            if len(v) == 0:
                continue
            print(f"  {label:8s}  {col:8s}: median={np.median(v):.4g}  "
                  f"|median|={np.median(np.abs(v)):.4g}  "
                  f"std={np.std(v):.4g}  n={len(v)}")


if __name__ == "__main__":
    main()
