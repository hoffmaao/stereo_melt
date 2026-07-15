"""Per-pair DhDt diagnostic for the Beardmore Lagrangian solver.

Operates on pure freeboard (no hydrostatic inversion) so noise doesn't
get rescaled by ~9x. For each consecutive epoch pair (i, j) of the
2019 stack:

    - dt_ij          — pair baseline in years
    - advected DhDt  — seed at epoch-i pixels, forward-Euler with the
                       time-mean (vx, vy), sample h[j] at the endpoint,
                       (h_end - h_seed) / dt_ij
    - fixed-pixel    — (h[j] - h[i]) / dt_ij without advection

Reports median, IQR, and NaN fraction for each. Goal: confirm short-dt
pairs dominate the Lagrangian +285 m/yr bias (noise / dt scaling).

Run:
    python -m beardmore.diagnose_pair_dhdt
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from beardmore.run_melt import (
    load_stack,
    load_velocity_on_grid,
    load_floating_mask,
)

SECONDS_PER_YEAR = 86400.0 * 365.25


def _pct(a: np.ndarray) -> tuple[float, float, float, float]:
    """Median, IQR low, IQR high, NaN fraction."""
    total = a.size
    finite = np.isfinite(a)
    n_finite = int(finite.sum())
    nan_frac = 1.0 - n_finite / max(total, 1)
    if n_finite == 0:
        return (float("nan"), float("nan"), float("nan"), nan_frac)
    v = a[finite]
    med = float(np.median(v))
    lo, hi = np.percentile(v, [25.0, 75.0])
    return (med, float(lo), float(hi), nan_frac)


def advected_dhdt(
    h_i: np.ndarray,
    h_j: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dt_yr: float,
    res_x: float,
    res_y: float,
    sub_dt_yr: float = 0.05,
) -> np.ndarray:
    """Forward-Euler-advected DhDt per seed pixel, pure freeboard."""
    ny, nx = h_i.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    y_idx = yy.astype(np.float64)
    x_idx = xx.astype(np.float64)

    n_steps = max(1, int(np.ceil(dt_yr / sub_dt_yr)))
    actual_dt = dt_yr / n_steps

    for _ in range(n_steps):
        vx_t = map_coordinates(vx, [y_idx, x_idx], order=1, mode="nearest")
        vy_t = map_coordinates(vy, [y_idx, x_idx], order=1, mode="nearest")
        # EPSG:3031: x ascends, y descends (res_y > 0 by construction)
        x_idx = x_idx + vx_t * actual_dt / res_x
        y_idx = y_idx - vy_t * actual_dt / res_y

    h_end = map_coordinates(
        h_j, [y_idx, x_idx], order=1, mode="constant", cval=np.nan
    )
    return (h_end - h_i) / dt_yr


def main() -> None:
    print("Loading stack...")
    stack = load_stack()
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    print("Loading floating mask...")
    floating = load_floating_mask(stack)
    mask_2d = floating.values.astype(bool)
    print(f"  floating frac: {mask_2d.mean():.3f}")

    print("Loading velocity (MEaSUREs)...")
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  source: {vel_source}")
    print(
        f"  |v| median={float(np.hypot(vx, vy).median()):.1f}  "
        f"max={float(np.hypot(vx, vy).max()):.1f} m/yr"
    )

    # Mask-restricted arrays
    h_stack = stack.values.astype(np.float64)  # (time, y, x)
    h_stack = np.where(mask_2d[None, :, :], h_stack, np.nan)
    vx_arr = vx.values.astype(np.float64)
    vy_arr = vy.values.astype(np.float64)

    x_coords = stack["x"].values
    y_coords = stack["y"].values
    res_x = float(x_coords[1] - x_coords[0])
    res_y = float(y_coords[0] - y_coords[1])  # y descends -> positive

    times = pd.to_datetime(stack["time"].values)
    t_years = np.array(
        [(t - times[0]).total_seconds() / SECONDS_PER_YEAR for t in times],
        dtype=np.float64,
    )

    rows = []
    for i in range(len(times) - 1):
        j = i + 1
        dt = t_years[j] - t_years[i]
        if dt <= 0:
            continue

        h_i = h_stack[i]
        h_j = h_stack[j]

        fixed = (h_j - h_i) / dt
        adv = advected_dhdt(h_i, h_j, vx_arr, vy_arr, dt, res_x, res_y)
        # Restrict both to floating mask
        fixed = np.where(mask_2d, fixed, np.nan)
        adv = np.where(mask_2d, adv, np.nan)

        f_med, f_lo, f_hi, f_nan = _pct(fixed)
        a_med, a_lo, a_hi, a_nan = _pct(adv)

        rows.append(
            {
                "pair": f"{i}->{j}",
                "t_i": times[i].strftime("%Y-%m-%d"),
                "t_j": times[j].strftime("%Y-%m-%d"),
                "dt_yr": dt,
                "fixed_med": f_med,
                "fixed_iqr_lo": f_lo,
                "fixed_iqr_hi": f_hi,
                "fixed_nan_frac": f_nan,
                "adv_med": a_med,
                "adv_iqr_lo": a_lo,
                "adv_iqr_hi": a_hi,
                "adv_nan_frac": a_nan,
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:8.2f}")

    print("\n=== Per-pair freeboard DhDt (m/yr) — floating-only ===")
    print(df.to_string(index=False))

    # Summary by dt bucket
    print("\n=== Advected DhDt median (|m/yr|) grouped by dt bucket ===")
    bins = [0.0, 0.1, 0.25, 0.6, 1.1]
    labels = ["<0.1 yr", "0.1-0.25", "0.25-0.6", "0.6-1.1"]
    df["dt_bin"] = pd.cut(df["dt_yr"], bins=bins, labels=labels, include_lowest=True)
    g = df.groupby("dt_bin", observed=True).agg(
        n=("pair", "count"),
        dt_mean=("dt_yr", "mean"),
        fixed_abs_med=("fixed_med", lambda s: float(np.nanmedian(np.abs(s)))),
        adv_abs_med=("adv_med", lambda s: float(np.nanmedian(np.abs(s)))),
        adv_iqr_width=(
            "adv_iqr_hi",
            lambda s: float(
                np.nanmedian(
                    df.loc[s.index, "adv_iqr_hi"] - df.loc[s.index, "adv_iqr_lo"]
                )
            ),
        ),
    )
    print(g.to_string())

    print(
        "\nInterpretation: if adv_iqr_width (or |adv_med|) for the <0.1 yr bucket\n"
        "is much larger than the 0.6-1.1 yr bucket, short-dt pairs are amplifying\n"
        "coregistration residuals into bogus DhDt — matching the suspected cause\n"
        "of the +285 m/yr Lagrangian bias."
    )


if __name__ == "__main__":
    main()
