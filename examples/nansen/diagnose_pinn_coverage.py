"""Diagnose where the streamline PINN loses pixels on Nansen.

Loads the PINN output NetCDF + the Lagrangian melt output + the
MEaSUREs velocity to localize where in the domain the backward
trajectories fail and which mechanism is responsible.

Run: python -m nansen.diagnose_pinn_coverage
"""

from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from nansen import config

PINN_NC = config.RESULTS_DIR / "nansen_streamline_pinn_250m_2019-01-01_2023-03-01.nc"
LAGR_NC = config.RESULTS_DIR / "nansen_melt_250m_2019-01-01_2023-03-01.nc"


def main():
    print(f"Loading PINN output: {PINN_NC.name}")
    pinn = xr.open_dataset(PINN_NC)
    print(f"Loading Lagrangian output: {LAGR_NC.name}")
    lagr = xr.open_dataset(LAGR_NC)

    floating = pinn["floating_mask"]
    n_floating = int(floating.sum())

    # PINN coverage on the floating shelf
    tau = pinn["tau"]
    pinn_m = pinn["melt_rate"]
    tau_finite = np.isfinite(tau.values) & floating.values
    pinn_finite = np.isfinite(pinn_m.values) & floating.values

    # Lagrangian coverage on the floating shelf
    lagr_m = lagr["melt_rate_lagrangian"]
    lagr_finite = np.isfinite(lagr_m.values) & floating.values

    print(f"\nFloating-shelf coverage:")
    print(f"  total floating pixels:           {n_floating}")
    print(f"  PINN (tau finite):               {int(tau_finite.sum())} "
          f"({100*tau_finite.sum()/n_floating:.1f}%)")
    print(f"  PINN (melt_rate finite):         {int(pinn_finite.sum())} "
          f"({100*pinn_finite.sum()/n_floating:.1f}%)")
    print(f"  Lagrangian (melt_rate finite):   {int(lagr_finite.sum())} "
          f"({100*lagr_finite.sum()/n_floating:.1f}%)")

    # Difference: where Lagrangian has data but PINN doesn't
    lagr_only = lagr_finite & ~pinn_finite
    print(f"  pixels Lagr has but PINN lost:   {int(lagr_only.sum())} "
          f"({100*lagr_only.sum()/n_floating:.1f}%)")

    # Load MEaSUREs to check velocity coverage
    print("\nLoading raw MEaSUREs velocity (before fillna)...")
    ds = xr.open_dataset(config.MEASURES_PHASE_NC)
    x_min, x_max = float(pinn["x"].min()), float(pinn["x"].max())
    y_min, y_max = float(pinn["y"].min()), float(pinn["y"].max())
    buf = 2000.0
    by = ds["y"].values
    if by[0] > by[-1]:
        sub = ds.sel(x=slice(x_min - buf, x_max + buf), y=slice(y_max + buf, y_min - buf))
    else:
        sub = ds.sel(x=slice(x_min - buf, x_max + buf), y=slice(y_min - buf, y_max + buf))
    vx = sub["VX"].interp(x=pinn["x"], y=pinn["y"], method="linear")
    vy = sub["VY"].interp(x=pinn["x"], y=pinn["y"], method="linear")
    v_finite = np.isfinite(vx.values) & np.isfinite(vy.values)
    speed = np.sqrt(vx.values**2 + vy.values**2)
    v_finite_floating = v_finite & floating.values
    print(f"  pixels with finite raw MEaSUREs on floating shelf: "
          f"{int(v_finite_floating.sum())} "
          f"({100*v_finite_floating.sum()/n_floating:.1f}%)")

    # PINN lost pixels that have finite velocity (real failure to reach GL)
    pinn_lost_with_v = (~pinn_finite) & v_finite & floating.values
    pinn_lost_no_v = (~pinn_finite) & (~v_finite) & floating.values
    print(f"  PINN-lost pixels WITH velocity data:    "
          f"{int(pinn_lost_with_v.sum())} ({100*pinn_lost_with_v.sum()/n_floating:.1f}%)  "
          f"← slow-flow / max_tau")
    print(f"  PINN-lost pixels WITHOUT velocity data: "
          f"{int(pinn_lost_no_v.sum())} ({100*pinn_lost_no_v.sum()/n_floating:.1f}%)  "
          f"← MEaSUREs gap (velocity=NaN→0)")

    # Plot: 6 panels
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    extent = [x_min, x_max, y_min, y_max]

    def _imshow(ax, data, title, cmap="viridis", vmin=None, vmax=None):
        im = ax.imshow(data, extent=extent, origin="upper", aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.045)
        return im

    # Row 1: data sources
    _imshow(axes[0, 0], floating.values.astype(float), "floating mask (BedMachine)",
            cmap="gray_r", vmin=0, vmax=1)
    _imshow(axes[0, 1], np.where(v_finite, speed, np.nan), "MEaSUREs |v| (raw, m/yr)",
            cmap="plasma", vmin=0, vmax=300)
    _imshow(axes[0, 2], v_finite.astype(float), "MEaSUREs finite mask",
            cmap="gray_r", vmin=0, vmax=1)

    # Row 2: failure modes
    _imshow(axes[1, 0], np.where(floating.values, tau.values, np.nan),
            "PINN recovered tau (yr)", cmap="cividis", vmin=0, vmax=200)

    # Encode pixel state: 0=PINN OK, 1=lost (no velocity), 2=lost (with velocity, slow flow),
    # 3=not floating
    state = np.full(floating.shape, np.nan)
    state[floating.values & pinn_finite] = 0
    state[pinn_lost_no_v] = 1
    state[pinn_lost_with_v] = 2
    _imshow(axes[1, 1], state, "PINN failure mode (floating only)\n"
            "0=ok  1=no MEaSUREs  2=slow/max_tau",
            cmap="tab10", vmin=0, vmax=9)

    # Lagr vs PINN overlap
    overlap = np.full(floating.shape, np.nan)
    overlap[floating.values & pinn_finite & lagr_finite] = 0  # both
    overlap[floating.values & pinn_finite & ~lagr_finite] = 1  # PINN only
    overlap[floating.values & ~pinn_finite & lagr_finite] = 2  # Lagr only
    overlap[floating.values & ~pinn_finite & ~lagr_finite] = 3  # neither
    _imshow(axes[1, 2], overlap, "coverage overlap\n0=both 1=PINN-only 2=Lagr-only 3=neither",
            cmap="tab10", vmin=0, vmax=9)

    fig.suptitle("Nansen streamline-PINN coverage diagnostic", fontsize=13)
    out_path = config.FIGURES_DIR / "pinn_coverage_diagnostic_250m.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path.name}")


if __name__ == "__main__":
    main()
