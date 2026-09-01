"""5-method melt-rate comparison on Beardmore mixed CS2 + IS2 stack.

Runs the standardized inversion set on the same tilt-corrected stack,
plots them side by side, and writes a single NetCDF.

Methods (left to right in the figure):
  1. Eulerian mass-conservation                    (Shean 2019 Eq. 10)
  2. Lagrangian path integration                   (Shean 2019 Eq. 7)
  3. dh/dt reformulation, FFT  (per-pixel OLS slope → 1-step inverse, λ=10)
  4. dh/dt reformulation, DCT  (per-pixel OLS slope → 1-step inverse, λ=10)
  5. D-masked CG DCT  (coverage-mask CG-LSQ in DCT basis,
                        λ=1e-1, length_scale = H_ref/2)
  6. Davison 2023 gridded melt rate (truth panel, RACMO-FAC corrected)

Closed-form per-(t, k) Tikhonov variants (FFT/DCT) are intentionally
excluded: their full-coverage assumption is dramatically violated on
Beardmore's gappy mixed CS2+IS2 stack and they over-smooth most of the
real signal — see ``stereo_melt/tests/diagnose_linear_inverse_coverage.py``.
Per-pixel-OLS (methods 3, 4) and the coverage-mask CG (method 5) handle
the gappy input correctly.

Run::

    python -m beardmore.compare_five_methods
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

if "STEREO_MELT_BACKEND" not in os.environ:
    try:
        import cupy as _cp  # noqa: F401
        os.environ["STEREO_MELT_BACKEND"] = "cupy"
    except ImportError:
        pass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.backend import backend as _BACKEND
from stereo_melt.dynamics import lagrangian_frame_stack
from stereo_melt.dynamics.pseudospectral_lagrangian_stationary import (
    stationary_pseudospectral_lagrangian_inverse,
)
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.io.davison import load_davison_gridded_in_shean
from stereo_melt.melt import (
    eulerian_melt_rate,
    lagrangian_melt_rate,
    linear_inverse_dhdt_lagrangian_melt_rate,
)

from beardmore import config
from beardmore.run_melt import (
    _imshow_xr,
    load_floating_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
)


def _summary(name: str, da: xr.DataArray) -> None:
    print(
        f"[{name:<22s}] median={float(da.median()):+.2f}  "
        f"IQR=[{float(da.quantile(0.25)):+.2f}, {float(da.quantile(0.75)):+.2f}]  "
        f"p05/p95=[{float(da.quantile(0.05)):+.2f}, {float(da.quantile(0.95)):+.2f}]  "
        f"abs_max={float(np.abs(da).max()):.1f}  m ice/yr"
    )


def main(res_override: float | None = None) -> None:
    config.ensure_output_dirs()
    print(f"stereo_melt backend: {_BACKEND}")

    if res_override is not None:
        stack_prefix = f"beardmore_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "beardmore_stack"
        out_suffix = ""

    print("\nLoading Beardmore IS2 tilt-corrected stack...")
    stack = load_stack(stack_prefix=stack_prefix)
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")

    floating = load_floating_mask(stack)
    print(f"  floating-ice fraction: {float(floating.mean()):.3f}")
    stack_f = stack.where(floating)

    print("Loading velocity + SMB...")
    vx, vy, vel_source = load_velocity_on_grid(stack_f)
    a_dot = load_smb_on_grid(stack_f)
    print(f"  velocity: {vel_source}")

    print("Loading firn air content (BedMachine, static)...")
    firn = load_firn_on_grid(stack_f, config.BEDMACHINE_NC)
    firn_finite = firn.values[np.isfinite(firn.values)]
    print(
        f"  firn (m): median={float(np.median(firn_finite)):.2f}  "
        f"IQR=[{float(np.percentile(firn_finite, 25)):.2f}, "
        f"{float(np.percentile(firn_finite, 75)):.2f}]  "
        f"finite frac={float(np.isfinite(firn.values).mean()):.3f}"
    )

    common_d = {"d": firn}

    if "time" in vx.dims:
        vx_m = vx.mean("time", skipna=True)
    else:
        vx_m = vx
    if "time" in vy.dims:
        vy_m = vy.mean("time", skipna=True)
    else:
        vy_m = vy

    # Build h_lag once and share across methods 3-5
    print("\nBuilding Lagrangian-frame stack (one-time)...")
    h_lag = lagrangian_frame_stack(stack_f, vx_m, vy_m, dt_yr=0.05)

    common_lin = dict(floating_mask=floating, dt_yr=0.05, h_lag_precomputed=h_lag, d=firn)

    print("\n=== 1/5 Eulerian mass-conservation ===")
    eul_ds = eulerian_melt_rate(stack_f, vx, vy, a_dot=a_dot, d=firn, robust_dh_dt=True)
    eul = eul_ds.melt_rate.where(floating)
    _summary("eulerian", eul)

    # Keep physical seed spacing ~50 m regardless of grid resolution; at
    # 250 m the 25 m-tuned seed_stride=2 produces a 75%-NaN checkerboard
    # because particles can't escape their seed cell over a single pair.
    grid_res_m = float(abs(stack_f["x"][1] - stack_f["x"][0]))
    seed_stride = max(1, round(50.0 / grid_res_m))
    print(f"\n=== 2/5 Lagrangian path integration (seed_stride={seed_stride}, {grid_res_m:.0f} m grid) ===")
    lagr_ds = lagrangian_melt_rate(
        stack_f, vx, vy, a_dot=a_dot, d=firn,
        dt_yr=0.05, seed_stride=seed_stride, pairs="all",
        min_dt_yr=2.0 / 12.0,
    )
    lagr = lagr_ds.melt_rate.where(floating)
    _summary("lagrangian", lagr)

    print("\n=== 3/5 dh/dt reformulation (per-pixel OLS, FFT) ===")
    dhdt_fft_ds = linear_inverse_dhdt_lagrangian_melt_rate(
        stack_f, vx, vy, transform="fft", reg=10.0, min_n_obs=3,
        robust_dh_dt=True,
        **common_lin,
    )
    dhdt_fft = dhdt_fft_ds.melt_rate.where(floating)
    H_ref_m = float(dhdt_fft_ds.attrs.get("H_ref_m", float("nan")))
    _summary("dh/dt-FFT", dhdt_fft)

    print("\n=== 4/5 dh/dt reformulation (per-pixel OLS, DCT) ===")
    dhdt_dct_ds = linear_inverse_dhdt_lagrangian_melt_rate(
        stack_f, vx, vy, transform="dct", reg=10.0, min_n_obs=3,
        robust_dh_dt=True,
        **common_lin,
    )
    dhdt_dct = dhdt_dct_ds.melt_rate.where(floating)
    _summary("dh/dt-DCT", dhdt_dct)

    L_cg = H_ref_m / 2.0
    print(
        f"\n=== 5/5 D-masked CG DCT  (λ=1e-1, L=H_ref/2={L_cg:.0f} m, max_iter=200) ==="
    )
    cg_dct_ds = stationary_pseudospectral_lagrangian_inverse(
        stack_f, vx, vy,
        floating_mask=floating, H_ref=H_ref_m, d=firn,
        tikhonov=1e-1, length_scale_m=L_cg,
        max_iter=200, cg_tol=1e-6,
        dt_yr=0.05, h_lag_precomputed=h_lag,
        transform="dct", verbose=True,
    )
    cg_dct = cg_dct_ds.melt_rate.where(floating)
    print(
        f"  CG: n_iter={cg_dct_ds.attrs.get('cg_iter', '?')}  "
        f"converged={cg_dct_ds.attrs.get('cg_converged', '?')}  "
        f"residual={cg_dct_ds.attrs.get('cg_residual_norm', float('nan')):.3e}"
    )
    _summary("masked-CG-DCT", cg_dct)

    print("\n=== Davison 2023 (gridded, regridded to our 25 m grid; Shean convention) ===")
    davison = load_davison_gridded_in_shean(eul).where(floating)
    _summary("davison-2023", davison)

    # ----------------------------------------------------------------------
    # Save NetCDF — only the standardized panel set; closed-form per-(t,k)
    # Tikhonov variants are intentionally not saved (they over-smooth real
    # signal; see project_linear_inverse_coverage_diagnosis).
    # ----------------------------------------------------------------------
    out_nc = (
        config.RESULTS_DIR
        / f"beardmore_five_methods{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving -> {out_nc}")
    out_ds = xr.Dataset(
        {
            "melt_rate_eulerian":     eul.rename("melt_rate_eulerian"),
            "melt_rate_lagrangian":   lagr.rename("melt_rate_lagrangian"),
            "melt_rate_dhdt_fft":     dhdt_fft.rename("melt_rate_dhdt_fft"),
            "melt_rate_dhdt_dct":     dhdt_dct.rename("melt_rate_dhdt_dct"),
            "melt_rate_masked_cg_dct": cg_dct.rename("melt_rate_masked_cg_dct"),
            "melt_rate_davison":      davison.rename("melt_rate_davison"),
            "dh_dt_input":            dhdt_fft_ds.dh_dt.rename("dh_dt_input"),
            "n_obs_per_pixel":        dhdt_fft_ds.n_obs.rename("n_obs_per_pixel"),
            "floating_mask":          floating,
        },
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "velocity_source": vel_source,
            "H_ref_m": H_ref_m,
            "dhdt_reg": 10.0,
            "cg_tikhonov": 1e-1,
            "cg_length_scale_m": L_cg,
            "cg_max_iter": 200,
            "cg_iter": int(cg_dct_ds.attrs.get("cg_iter", -1)),
            "cg_converged": int(cg_dct_ds.attrs.get("cg_converged", 0)),
            "cg_residual_norm": float(cg_dct_ds.attrs.get("cg_residual_norm", float("nan"))),
        },
    )
    out_ds.to_netcdf(out_nc)

    # ----------------------------------------------------------------------
    # 6-panel figure (5 methods + Davison)
    # ----------------------------------------------------------------------
    fig_path = config.FIGURES_DIR / f"five_methods_comparison{out_suffix}.png"
    print(f"Plotting -> {fig_path}")

    panels = [
        ("1. Eulerian mass-cons.\n(Shean Eq. 10)",                  eul, (-20, 20)),
        ("2. Lagrangian path-int.\n(Shean Eq. 7)",                  lagr, (-20, 20)),
        ("3. dh/dt FFT\n(per-pixel OLS, λ=10)",                     dhdt_fft, (-20, 20)),
        ("4. dh/dt DCT\n(per-pixel OLS, λ=10)",                     dhdt_dct, (-20, 20)),
        (f"5. D-masked CG DCT\n(λ=0.1, L=H_ref/2={L_cg:.0f}m)",     cg_dct, (-20, 20)),
        ("Davison 2023\n(gridded, RACMO-FAC corr.)",                davison, (-20, 20)),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 4.4, 6.5), constrained_layout=True)
    for col, (title, da, clim) in enumerate(panels):
        im = _imshow_xr(axes[col], da, cmap=melt_cmap(),
                        norm=melt_norm(vmax=max(abs(clim[0]), abs(clim[1]))))
        axes[col].set_title(
            f"{title}\nmedian={float(da.median()):+.2f}  "
            f"IQR=[{float(da.quantile(0.25)):+.2f}, {float(da.quantile(0.75)):+.2f}]  "
            f"abs_max={float(np.abs(da).max()):.0f}",
            fontsize=10,
        )
        add_melt_colorbar(fig, im, ax=axes[col], fraction=0.045)
        axes[col].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(
        f"Beardmore mixed CS2+IS2 melt-rate comparison — 5 methods + Davison 2023 "
        f"({config.START_TIME} → {config.END_TIME}, H_ref={H_ref_m:.0f} m)",
        fontsize=12,
    )
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help=(
            "Resolution variant to load (meters). When set, looks up "
            "beardmore_stack_<N>m*_<start>_<end>.nc and writes outputs at *<N>m* "
            "paths so 25 m production isn't clobbered."
        ),
    )
    args = parser.parse_args()
    main(res_override=args.res)
