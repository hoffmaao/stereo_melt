"""Diagnose where the artifacts in the 6-panel inversion comparison come from.

Three independent suspects to interrogate side-by-side:
  - velocity field blotches  -> stationary-spectral & Stubblefield streaks
  - tilt-correction residual -> Eulerian & Lagrangian banding
  - PS solver loaded the un-tilt-corrected stack -> washed-out PS panels

Run:
    python -m beardmore.diagnose_artifacts
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.kinematics import divergence

from beardmore import config
from beardmore.run_melt import load_velocity_on_grid


def _imshow(ax, da, *, cmap, vmin=None, vmax=None, title=""):
    im = ax.imshow(
        da.values,
        extent=[float(da["x"].min()), float(da["x"].max()),
                float(da["y"].min()), float(da["y"].max())],
        origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
    )
    ax.set_title(title, fontsize=10)
    return im


def main() -> None:
    config.ensure_output_dirs()

    stack_tc_path = config.PROCESSED_DIR / f"beardmore_stack_tilt_corrected_{config.START_TIME}_{config.END_TIME}.nc"
    tilt_path = config.PROCESSED_DIR / f"beardmore_tilt_params_{config.START_TIME}_{config.END_TIME}.nc"
    melt_path = config.RESULTS_DIR / f"beardmore_melt_{config.START_TIME}_{config.END_TIME}.nc"

    print(f"Loading {stack_tc_path.name}...")
    ds_tc = xr.open_dataset(stack_tc_path)
    stack_tc = ds_tc[[v for v in ds_tc.data_vars if v != "spatial_ref"][0]]

    print(f"Loading {tilt_path.name}...")
    tilt = xr.open_dataset(tilt_path)

    print(f"Loading {melt_path.name}...")
    melt = xr.open_dataset(melt_path)

    print("Loading velocity...")
    vx, vy, vsrc = load_velocity_on_grid(stack_tc)
    print(f"  velocity source: {vsrc}")

    div_u = divergence(vx, vy)  # 1/yr
    speed = np.hypot(vx, vy)  # m/yr

    # Per-pixel time-std of the tilt correction that was *applied* to the raw
    # stack to produce the tilt-corrected stack.  delta(t, y, x) =
    # tilt_dx[t]*(x - xref[t]) + tilt_dy[t]*(y - yref[t]) + tilt_dz[t]
    print("Building per-pixel tilt-correction history...")
    x = tilt["x"].values
    y = tilt["y"].values
    X, Y = np.meshgrid(x, y)
    T = tilt.sizes["time"]
    delta = np.empty((T, Y.shape[0], X.shape[1]), dtype=np.float64)
    for k in range(T):
        delta[k] = (
            float(tilt["tilt_dx"].values[k]) * (X - float(tilt["xref"].values[k]))
            + float(tilt["tilt_dy"].values[k]) * (Y - float(tilt["yref"].values[k]))
            + float(tilt["tilt_dz"].values[k])
        )
    delta_da = xr.DataArray(
        delta, dims=("time", "y", "x"),
        coords={"time": tilt["time"].values, "y": y, "x": x},
        name="tilt_correction",
    )
    tilt_std = delta_da.std("time")  # spatial pattern of tilt variability
    tilt_mean = delta_da.mean("time")

    eul = melt["melt_rate_eulerian"]
    lag = melt["melt_rate_lagrangian"]

    # ------------------------------------------------------------------
    # Layout: 3 rows x 4 cols
    # row 0: vx, vy, |v|, divergence(u)
    # row 1: tilt time-mean, tilt time-std, dhdt fit, flux_div from melt
    # row 2: Eulerian, Lagrangian, Eulerian-Lagrangian, per-epoch tilt magnitudes
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), constrained_layout=True)

    # row 0: velocity
    im = _imshow(axes[0, 0], vx, cmap="RdBu_r", vmin=-300, vmax=300, title="vx (m/yr)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.045)
    im = _imshow(axes[0, 1], vy, cmap="RdBu_r", vmin=-300, vmax=300, title="vy (m/yr)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.045)
    im = _imshow(axes[0, 2], speed, cmap="viridis", vmin=0, vmax=400, title="|v| (m/yr)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.045)
    im = _imshow(axes[0, 3], div_u, cmap="RdBu", vmin=-0.05, vmax=0.05,
                 title="∇·u (1/yr)  ← divergence blotches")
    fig.colorbar(im, ax=axes[0, 3], fraction=0.045)

    # row 1: tilt residual + dh/dt
    im = _imshow(axes[1, 0], tilt_mean, cmap="RdBu", vmin=-50, vmax=50,
                 title="tilt correction time-MEAN (m)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.045)
    im = _imshow(axes[1, 1], tilt_std, cmap="magma", vmin=0, vmax=40,
                 title="tilt correction time-STD (m)  ← banding source?")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.045)
    im = _imshow(axes[1, 2], tilt["dhdt"], cmap="RdBu", vmin=-5, vmax=5,
                 title="dh/dt fit (m/yr) from tilt_fit")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.045)
    im = _imshow(axes[1, 3], melt["flux_div"], cmap="RdBu", vmin=-10, vmax=10,
                 title="∇·(Hu) (m ice/yr)")
    fig.colorbar(im, ax=axes[1, 3], fraction=0.045)

    # row 2: outputs
    im = _imshow(axes[2, 0], eul, cmap="RdBu_r", vmin=-5, vmax=5, title="Eulerian melt (m ice/yr)")
    fig.colorbar(im, ax=axes[2, 0], fraction=0.045)
    im = _imshow(axes[2, 1], lag, cmap="RdBu_r", vmin=-5, vmax=5, title="Lagrangian melt (m ice/yr)")
    fig.colorbar(im, ax=axes[2, 1], fraction=0.045)
    im = _imshow(axes[2, 2], eul - lag, cmap="PuOr", vmin=-3, vmax=3,
                 title="Eulerian - Lagrangian (diagnostic)")
    fig.colorbar(im, ax=axes[2, 2], fraction=0.045)

    # bottom right: per-epoch tilt magnitudes vs time
    ax = axes[2, 3]
    t = np.arange(T)
    half_diag = 0.5 * float(np.hypot(x.max() - x.min(), y.max() - y.min()))
    tx_m = np.abs(tilt["tilt_dx"].values) * half_diag
    ty_m = np.abs(tilt["tilt_dy"].values) * half_diag
    tz_m = np.abs(tilt["tilt_dz"].values)
    ax.plot(t, tx_m, "-o", label="|αx|·R/2 (m)")
    ax.plot(t, ty_m, "-o", label="|αy|·R/2 (m)")
    ax.plot(t, tz_m, "-o", label="|αz| (m)")
    ax.set_xlabel("epoch index")
    ax.set_ylabel("max tilt residual at domain corner (m)")
    ax.set_title("per-epoch tilt magnitude")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Beardmore artifact diagnostic: velocity + tilt residual vs Eul/Lag outputs",
        fontsize=14,
    )

    out = config.FIGURES_DIR / "diagnose_artifacts.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def _q(name, da, lo=0.01, hi=0.99):
        a = da.where(np.isfinite(da))
        n = int(a.notnull().sum())
        if n == 0:
            print(f"  {name}: no finite cells"); return
        ql, qh = float(a.quantile(lo)), float(a.quantile(hi))
        med = float(a.median())
        print(f"  {name:30s}  median={med:+.3g}  [q{int(lo*100):02d}={ql:+.3g}, q{int(hi*100):02d}={qh:+.3g}]")

    print("\n--- input fields ---")
    _q("vx (m/yr)", vx)
    _q("vy (m/yr)", vy)
    _q("|v|", speed)
    _q("div(u) (1/yr)", div_u)
    _q("tilt time-mean (m)", tilt_mean)
    _q("tilt time-std (m)", tilt_std)
    _q("tilt dhdt fit (m/yr)", tilt["dhdt"])
    _q("flux_div (m ice/yr)", melt["flux_div"])
    print("\n--- outputs ---")
    _q("Eulerian melt", eul)
    _q("Lagrangian melt", lag)
    _q("Eulerian - Lagrangian", eul - lag)

    print("\n--- per-epoch tilt summary ---")
    for k in range(T):
        print(
            f"  t[{k:2d}] {str(tilt['time'].values[k])[:10]}: "
            f"αx={float(tilt['tilt_dx'].values[k]):+.3e}  "
            f"αy={float(tilt['tilt_dy'].values[k]):+.3e}  "
            f"αz={float(tilt['tilt_dz'].values[k]):+7.2f} m  "
            f"max-residual≈{tx_m[k]+ty_m[k]+tz_m[k]:6.1f} m"
        )


if __name__ == "__main__":
    main()
