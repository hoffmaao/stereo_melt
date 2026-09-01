"""Run the streamline-frame PINN melt-rate solver on the Nansen stack.

A Lagrangian PINN that takes (x_gl, y_gl, tau) — grounding-line crossing
point and age since crossing — and predicts freeboard via an autodiff
residual form of Shean Eq. 10. Validates against the existing
`melt_rate_lagrangian` panel from `nansen.run_melt`.

Run:

    python -m nansen.run_streamline_pinn [--res 250]

Assumes nansen.build_stack + nansen.tilt_fit have produced the
stack, and nansen.run_melt has produced the comparison Lagrangian
result.
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

from stereo_melt.dynamics.streamline_pinn import (
    fit_streamline_pinn,
    predict_melt_grid,
)
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.kinematics import (
    backward_advect_pixels,
    build_streamline_dataset,
    divergence,
)

from nansen import config
from nansen.run_melt import (
    load_floating_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
)

# Per-epoch noise floor — Nansen is fully IS2-era, so the scalar default
# from feedback_ez_per_gcp_source applies.
EZ_NANSEN_IS2 = 0.3  # meters of freeboard, 1σ


def load_grounded_mask(stack: xr.DataArray) -> xr.DataArray:
    """Return a boolean grounded-ice mask on the stack grid (BedMachine flag 2)."""
    if not config.BEDMACHINE_NC.exists():
        raise SystemExit(
            f"BedMachine not found at {config.BEDMACHINE_NC}. "
            "Cannot build grounded-ice mask."
        )
    ds = xr.open_dataset(config.BEDMACHINE_NC)
    buf = 2000.0
    x_min = float(stack["x"].min()) - buf
    x_max = float(stack["x"].max()) + buf
    y_min = float(stack["y"].min()) - buf
    y_max = float(stack["y"].max()) + buf
    by = ds["y"].values
    if by[0] > by[-1]:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
    else:
        sub = ds["mask"].sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    sub = sub.load()
    on_grid = sub.interp(x=stack["x"], y=stack["y"], method="nearest")
    grounded = (on_grid == 2)
    grounded.attrs = {
        "source": "BedMachine Antarctica v3 mask == 2 (grounded_ice)",
        "flag_values": "0=ocean 1=ice_free_land 2=grounded_ice 3=floating_ice 4=lake_vostok",
    }
    grounded.name = "grounded_mask"
    return grounded


def _imshow_xr(ax, da, *, cmap, vmin=None, vmax=None):
    return ax.imshow(
        da.values,
        extent=[
            float(da["x"].min()),
            float(da["x"].max()),
            float(da["y"].min()),
            float(da["y"].max()),
        ],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )


def _panel_stats(da):
    v = da.values[np.isfinite(da.values)]
    if v.size == 0:
        return "no data"
    return (
        f"med={float(np.median(v)):+.2f}  "
        f"IQR=[{float(np.percentile(v, 25)):+.2f}, {float(np.percentile(v, 75)):+.2f}]"
    )


def plot_pinn_vs_lagrangian(
    pinn_melt: xr.DataArray,
    lagr_melt: xr.DataArray,
    out_path: Path,
    clim: tuple[float, float] = (-5.0, 5.0),
) -> None:
    diff = pinn_melt - lagr_melt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), constrained_layout=True)
    for ax, (title, da, cmap, vlim) in zip(
        axes,
        [
            ("Streamline PINN\n(Shean Eq. 10, autodiff)", pinn_melt, "RdBu_r", clim),
            ("Lagrangian path-integral\n(existing solver)", lagr_melt, "RdBu_r", clim),
            ("PINN − Lagrangian", diff, "PuOr_r", (-2.0, 2.0)),
        ],
    ):
        im = _imshow_xr(ax, da, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_title(f"{title}\n{_panel_stats(da)}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.045)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"Nansen — streamline PINN vs path-integral Lagrangian — {config.START_TIME} to {config.END_TIME}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(res_override: float | None = None) -> None:
    config.ensure_output_dirs()

    if res_override is not None:
        stack_prefix = f"nansen_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "nansen_stack"
        out_suffix = ""

    print("Loading stack...")
    stack = load_stack(stack_prefix=stack_prefix)
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    print("Floating + grounded masks...")
    floating = load_floating_mask(stack)
    grounded = load_grounded_mask(stack)
    print(
        f"  floating frac: {float(floating.mean()):.3f}  "
        f"grounded frac: {float(grounded.mean()):.3f}"
    )
    stack_masked = stack.where(floating)

    print("Loading velocity...")
    vx, vy, vel_source = load_velocity_on_grid(stack_masked)
    print(f"  source: {vel_source}")

    print("Loading SMB (RACMO2.4p1)...")
    a_dot = load_smb_on_grid(stack_masked)

    print("Loading firn (BedMachine, static)...")
    firn = load_firn_on_grid(stack_masked, config.BEDMACHINE_NC)

    print("Computing ∇·u from velocity...")
    vdiv = divergence(vx, vy)

    print("Backward-advecting every floating pixel to its GL crossing...")
    trajectories = backward_advect_pixels(
        x_coords=stack["x"].values,
        y_coords=stack["y"].values,
        vx=vx.values,
        vy=vy.values,
        grounded_mask=grounded.values,
        floating_mask=floating.values,
        dt_yr=0.05,
        max_tau_yr=200.0,  # Nansen is small; this is generous
    )
    reason = trajectories["reason"]
    n_total = int(reason.size)
    n_success = int((reason == 0).sum())
    print(
        f"  trajectories: {n_success}/{n_total} success ({100*n_success/n_total:.1f}%)  "
        f"oob={int((reason == 1).sum())}  maxtau={int((reason == 2).sum())}  "
        f"non-floating={int((reason == 3).sum())}"
    )
    tau_finite = trajectories["tau"][np.isfinite(trajectories["tau"])]
    print(
        f"  tau: median={float(np.median(tau_finite)):.1f}  "
        f"max={float(tau_finite.max()):.1f} yr"
    )

    print("Assembling streamline training dataset (sigma=Ez=0.3 m for IS2 era)...")
    sigma_per_epoch = np.full(stack_masked.sizes["time"], EZ_NANSEN_IS2, dtype=np.float64)
    dataset = build_streamline_dataset(
        h_stack=stack_masked,
        trajectories=trajectories,
        sigma_per_epoch=sigma_per_epoch,
        a_dot=a_dot,
        vdiv=vdiv,
        d_fac=firn,
    )
    print(f"  rows: {len(dataset)} observations")

    print("Training streamline PINN...")
    result = fit_streamline_pinn(
        dataset,
        hidden_dim=256,
        n_hidden_layers=6,
        n_fourier_features=16,
        n_epochs=30,
        batch_size=4096,
        lr=1e-3,
        test_frac=0.15,
        verbose=True,
    )
    print(
        f"  final train_loss={result.train_losses[-1]:.4e}  "
        f"test_loss={result.test_losses[-1]:.4e}"
    )

    print("Predicting melt rate on Eulerian grid...")
    pred = predict_melt_grid(
        model=result.model,
        trajectories=trajectories,
        a_dot=a_dot,
        vdiv=vdiv,
        d_fac=firn,
    )
    pinn_melt = pred.melt_rate.where(floating)
    print(
        f"  PINN melt_rate: median={float(pinn_melt.median()):+.2f}  "
        f"IQR=[{float(pinn_melt.quantile(0.25)):+.2f}, "
        f"{float(pinn_melt.quantile(0.75)):+.2f}] m ice/yr"
    )

    print("Loading existing Lagrangian melt for comparison...")
    melt_nc = (
        config.RESULTS_DIR
        / f"nansen_melt{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    if not melt_nc.exists():
        raise SystemExit(
            f"Comparison file not found: {melt_nc}\n"
            f"Run `python -m nansen.run_melt {('--res ' + str(int(res_override))) if res_override else ''}` first."
        )
    melt_ds = xr.open_dataset(melt_nc)
    lagr_melt = melt_ds["melt_rate_lagrangian"]
    print(
        f"  Lagrangian melt_rate: median={float(lagr_melt.median()):+.2f}  "
        f"IQR=[{float(lagr_melt.quantile(0.25)):+.2f}, "
        f"{float(lagr_melt.quantile(0.75)):+.2f}] m ice/yr"
    )

    # Write the PINN result
    out_nc = (
        config.RESULTS_DIR
        / f"nansen_streamline_pinn{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    pred_out = pred.assign(floating_mask=floating)
    pred_out.attrs = {
        "shelf": config.SHELF,
        "window_start": config.START_TIME,
        "window_end": config.END_TIME,
        "grid_res_m": float(config.RES),
        "velocity_source": vel_source,
        "sigma_per_obs_m": EZ_NANSEN_IS2,
        "n_train_rows": result.n_train,
        "n_test_rows": result.n_test,
        "final_train_loss": result.train_losses[-1],
        "final_test_loss": result.test_losses[-1],
        "convention": "Shean (positive=accretion, negative=melt)",
    }
    pred_out.to_netcdf(out_nc)
    print(f"  wrote {out_nc.name}")

    out_png = config.FIGURES_DIR / f"streamline_pinn_vs_lagrangian{out_suffix}.png"
    plot_pinn_vs_lagrangian(pinn_melt, lagr_melt, out_png)
    print(f"  wrote {out_png.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", type=float, default=None)
    args = parser.parse_args()
    main(res_override=args.res)
