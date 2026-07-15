"""Closed-form smoothing-pipeline diagnostic on the Nansen IS2 stack.

DEPRECATED 2026-05-05. The closed-form per-(t,k) Tikhonov variants
(``melt_rate_closed_fft`` / ``melt_rate_closed_dct``) are no longer
saved by ``compare_five_methods.py`` — they over-smooth the real signal
on REMA's gappy stacks. The principled replacement (``D-masked CG DCT``,
λ=0.1, L=H_ref/2) is in the standardized 6-panel set. This script still
runs against an old saved NC that has ``melt_rate_closed_*`` fields, and
will error on a fresh save. Kept as archive of the closed-form sweep
(project_linear_inverse_coverage_diagnosis.md).

Reuses the already-saved 5-method results and adds six diagnostic variants
exploring the closed-form preprocessing chain in both transforms:

  A-FFT/DCT. ``prefilter_sigma_m=0`` -- kills the H_ref/2 prefilter
       Gaussian (keeps the H_ref-σ NaN-aware infill on FFT; DCT has
       infill off by default).
  B-FFT/DCT. ``infill_sigma_m=H_ref/4`` with ``boundary_fix="infill"``
       -- tightens the infill Gaussian (or, for DCT, turns it on at a
       tight scale).
  C-FFT/DCT. ``stationary_pseudospectral_lagrangian_inverse`` (masked
       CG-LSQ with NaN-honest weighting) -- the principled fix flagged
       in ``project_linear_inverse_coverage_diagnosis``.

Plots a 3x4 panel figure: reference methods (top row), closed-FFT
family (middle), closed-DCT family (bottom), all on the same ±5 m/yr
RdBu scale.

Run::

    python -m nansen.compare_closed_form_variants
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

from stereo_melt.backend import backend as _BACKEND
from stereo_melt.dynamics import lagrangian_frame_stack
from stereo_melt.dynamics.pseudospectral_lagrangian_stationary import (
    stationary_pseudospectral_lagrangian_inverse,
)
from stereo_melt.melt import linear_inverse_lagrangian_melt_rate

from nansen import config
from nansen.run_melt import (
    _imshow_xr,
    load_floating_mask,
    load_smb_on_grid,
    load_stack,
    load_velocity_on_grid,
)


def _stats(da: xr.DataArray) -> str:
    v = da.values
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "no data"
    return (
        f"med={np.median(v):+.2f}  "
        f"std={v.std():.2f}  "
        f"IQR=[{np.percentile(v, 25):+.2f}, {np.percentile(v, 75):+.2f}]"
    )


def main() -> None:
    config.ensure_output_dirs()
    print(f"stereo_melt backend: {_BACKEND}")

    saved_path = (
        config.RESULTS_DIR
        / f"nansen_five_methods_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nLoading saved 5-method results: {saved_path}")
    saved = xr.open_dataset(saved_path)
    floating = saved.floating_mask.astype(bool)
    H_ref = float(saved.attrs.get("H_ref_m", float("nan")))
    print(f"  H_ref = {H_ref:.1f} m   (saved attrs)")

    print("\nLoading Nansen IS2 tilt-corrected stack...")
    stack = load_stack()
    print(f"  dims: time={stack.sizes['time']}, y={stack.sizes['y']}, x={stack.sizes['x']}")
    stack_f = stack.where(floating)

    print("Loading velocity + SMB...")
    vx, vy, vel_source = load_velocity_on_grid(stack_f)
    a_dot = load_smb_on_grid(stack_f)
    print(f"  velocity: {vel_source}")

    if "time" in vx.dims:
        vx_m = vx.mean("time", skipna=True)
    else:
        vx_m = vx
    if "time" in vy.dims:
        vy_m = vy.mean("time", skipna=True)
    else:
        vy_m = vy

    print("\nBuilding Lagrangian-frame stack (one-time)...")
    h_lag = lagrangian_frame_stack(stack_f, vx_m, vy_m, dt_yr=0.05)

    common_lin = dict(
        floating_mask=floating,
        dt_yr=0.05,
        h_lag_precomputed=h_lag,
        H_ref=H_ref,
    )

    def _run_closed_form(transform, prefilter, infill_sigma, label):
        boundary_fix = "infill+pad" if transform == "fft" else "infill"
        ds = linear_inverse_lagrangian_melt_rate(
            stack_f, vx, vy,
            boundary_fix=boundary_fix, transform=transform, reg=1e-1,
            prefilter_sigma_m=prefilter,
            infill_sigma_m=infill_sigma,
            **common_lin,
        )
        m = ds.melt_rate.where(floating)
        print(f"  [{label}]  {_stats(m)}")
        return m, ds

    def _run_masked_cg(transform, label, tikhonov=1e-2, length_scale_m=100.0, max_iter=100):
        ds = stationary_pseudospectral_lagrangian_inverse(
            stack_f, vx, vy,
            floating_mask=floating,
            H_ref=H_ref,
            tikhonov=tikhonov,
            length_scale_m=length_scale_m,
            max_iter=max_iter,
            cg_tol=1e-6,
            dt_yr=0.05,
            h_lag_precomputed=h_lag,
            transform=transform,
            verbose=True,
        )
        m = ds.melt_rate.where(floating)
        print(
            f"  [{label}]  {_stats(m)}   "
            f"CG: n_iter={ds.attrs.get('cg_iter', '?')} "
            f"converged={ds.attrs.get('cg_converged', '?')} "
            f"residual={ds.attrs.get('cg_residual_norm', float('nan')):.3e}"
        )
        return m, ds

    print("\n=== A-FFT: closed-FFT, prefilter_sigma_m=0 ===")
    var_a_fft, _ = _run_closed_form("fft", 0.0, None, "A-FFT")

    print("\n=== B-FFT: closed-FFT, infill_sigma_m=H_ref/4 ===")
    var_b_fft, _ = _run_closed_form("fft", None, H_ref / 4.0, "B-FFT")

    print("\n=== C-FFT: masked CG-LSQ (FFT) ===")
    var_c_fft, var_c_fft_ds = _run_masked_cg("fft", "C-FFT")

    print("\n=== A-DCT: closed-DCT, prefilter_sigma_m=0 ===")
    var_a_dct, _ = _run_closed_form("dct", 0.0, None, "A-DCT")

    print("\n=== B-DCT: closed-DCT, boundary_fix=infill, infill_sigma_m=H_ref/4 ===")
    var_b_dct, _ = _run_closed_form("dct", None, H_ref / 4.0, "B-DCT")

    print("\n=== C-DCT: masked CG-LSQ (DCT) ===")
    var_c_dct, var_c_dct_ds = _run_masked_cg("dct", "C-DCT")

    L_d = H_ref / 2.0
    print(f"\n=== D-FFT: masked CG-LSQ (FFT, tikhonov=1e-1, L={L_d:.0f} m) ===")
    var_d_fft, var_d_fft_ds = _run_masked_cg(
        "fft", "D-FFT",
        tikhonov=1e-1, length_scale_m=L_d, max_iter=200,
    )

    print(f"\n=== D-DCT: masked CG-LSQ (DCT, tikhonov=1e-1, L={L_d:.0f} m) ===")
    var_d_dct, var_d_dct_ds = _run_masked_cg(
        "dct", "D-DCT",
        tikhonov=1e-1, length_scale_m=L_d, max_iter=200,
    )

    L_e = 100.0
    print(f"\n=== E-FFT: masked CG-LSQ (FFT, tikhonov=1e-1, L={L_e:.0f} m) ===")
    var_e_fft, var_e_fft_ds = _run_masked_cg(
        "fft", "E-FFT",
        tikhonov=1e-1, length_scale_m=L_e, max_iter=200,
    )

    print(f"\n=== E-DCT: masked CG-LSQ (DCT, tikhonov=1e-1, L={L_e:.0f} m) ===")
    var_e_dct, var_e_dct_ds = _run_masked_cg(
        "dct", "E-DCT",
        tikhonov=1e-1, length_scale_m=L_e, max_iter=200,
    )

    out_nc = (
        config.RESULTS_DIR
        / f"nansen_closed_form_variants_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"\nSaving -> {out_nc}")
    out_ds = xr.Dataset(
        {
            "melt_rate_closed_fft_no_prefilter": var_a_fft.rename(
                "melt_rate_closed_fft_no_prefilter"
            ),
            "melt_rate_closed_fft_tight_infill": var_b_fft.rename(
                "melt_rate_closed_fft_tight_infill"
            ),
            "melt_rate_masked_cg_fft":           var_c_fft.rename("melt_rate_masked_cg_fft"),
            "melt_rate_closed_dct_no_prefilter": var_a_dct.rename(
                "melt_rate_closed_dct_no_prefilter"
            ),
            "melt_rate_closed_dct_tight_infill": var_b_dct.rename(
                "melt_rate_closed_dct_tight_infill"
            ),
            "melt_rate_masked_cg_dct":           var_c_dct.rename("melt_rate_masked_cg_dct"),
            "melt_rate_masked_cg_fft_d":         var_d_fft.rename("melt_rate_masked_cg_fft_d"),
            "melt_rate_masked_cg_dct_d":         var_d_dct.rename("melt_rate_masked_cg_dct_d"),
            "melt_rate_masked_cg_fft_e":         var_e_fft.rename("melt_rate_masked_cg_fft_e"),
            "melt_rate_masked_cg_dct_e":         var_e_dct.rename("melt_rate_masked_cg_dct_e"),
            "floating_mask":                     floating,
        },
        attrs={
            "shelf": config.SHELF,
            "window_start": config.START_TIME,
            "window_end": config.END_TIME,
            "velocity_source": vel_source,
            "H_ref_m": H_ref,
            "infill_sigma_default_m": H_ref,
            "infill_sigma_tight_m": H_ref / 4.0,
            "prefilter_sigma_default_m": H_ref / 2.0,
            "cg_tikhonov": 1e-2,
            "cg_length_scale_m": 100.0,
            "cg_iter_fft": int(var_c_fft_ds.attrs.get("cg_iter", -1)),
            "cg_converged_fft": int(var_c_fft_ds.attrs.get("cg_converged", 0)),
            "cg_residual_fft": float(var_c_fft_ds.attrs.get("cg_residual_norm", float("nan"))),
            "cg_iter_dct": int(var_c_dct_ds.attrs.get("cg_iter", -1)),
            "cg_converged_dct": int(var_c_dct_ds.attrs.get("cg_converged", 0)),
            "cg_residual_dct": float(var_c_dct_ds.attrs.get("cg_residual_norm", float("nan"))),
            "cg_d_tikhonov": 1e-1,
            "cg_d_length_scale_m": L_d,
            "cg_d_iter_fft": int(var_d_fft_ds.attrs.get("cg_iter", -1)),
            "cg_d_converged_fft": int(var_d_fft_ds.attrs.get("cg_converged", 0)),
            "cg_d_residual_fft": float(var_d_fft_ds.attrs.get("cg_residual_norm", float("nan"))),
            "cg_d_iter_dct": int(var_d_dct_ds.attrs.get("cg_iter", -1)),
            "cg_d_converged_dct": int(var_d_dct_ds.attrs.get("cg_converged", 0)),
            "cg_d_residual_dct": float(var_d_dct_ds.attrs.get("cg_residual_norm", float("nan"))),
            "cg_e_tikhonov": 1e-1,
            "cg_e_length_scale_m": L_e,
            "cg_e_iter_fft": int(var_e_fft_ds.attrs.get("cg_iter", -1)),
            "cg_e_converged_fft": int(var_e_fft_ds.attrs.get("cg_converged", 0)),
            "cg_e_residual_fft": float(var_e_fft_ds.attrs.get("cg_residual_norm", float("nan"))),
            "cg_e_iter_dct": int(var_e_dct_ds.attrs.get("cg_iter", -1)),
            "cg_e_converged_dct": int(var_e_dct_ds.attrs.get("cg_converged", 0)),
            "cg_e_residual_dct": float(var_e_dct_ds.attrs.get("cg_residual_norm", float("nan"))),
        },
    )
    out_ds.to_netcdf(out_nc)

    fig_path = config.FIGURES_DIR / "five_methods_with_closed_form_variants.png"
    print(f"Plotting -> {fig_path}")

    eul = saved.melt_rate_eulerian.where(floating)
    lagr = saved.melt_rate_lagrangian.where(floating)
    cfft = saved.melt_rate_closed_fft.where(floating)
    cdct = saved.melt_rate_closed_dct.where(floating)
    dfft = saved.melt_rate_dhdt_fft.where(floating)
    ddct = saved.melt_rate_dhdt_dct.where(floating)
    davison = saved.melt_rate_davison.where(floating)

    rows = [
        ("Reference", [
            ("Eulerian\n(Shean Eq. 10)",                        eul),
            ("Lagrangian path-int\n(Shean Eq. 7)",              lagr),
            ("dh/dt FFT\n(per-pixel OLS)",                      dfft),
            ("dh/dt DCT\n(per-pixel OLS)",                      ddct),
            ("Davison 2023\n(gridded, RACMO-FAC)",              davison),
        ]),
        ("closed-FFT family", [
            (f"baseline\n(infill σ=H_ref={H_ref:.0f}m, prefilt σ=H_ref/2)", cfft),
            ("A: prefilter σ=0\n(infill kept at H_ref)",                    var_a_fft),
            (f"B: infill σ=H_ref/4={H_ref/4:.0f}m\n(prefilt kept)",          var_b_fft),
            (f"D: masked CG\n(λ=1e-1, L={L_d:.0f}m=H_ref/2)",                var_d_fft),
            (f"E: masked CG\n(λ=1e-1, L={L_e:.0f}m)",                        var_e_fft),
        ]),
        ("closed-DCT family", [
            ("baseline\n(reflective, infill off, prefilt σ=H_ref/2)",        cdct),
            ("A: prefilter σ=0\n(reflective, infill off)",                   var_a_dct),
            (f"B: infill on, σ=H_ref/4\n(reflective, prefilt kept)",         var_b_dct),
            (f"D: masked CG DCT\n(λ=1e-1, L={L_d:.0f}m=H_ref/2)",            var_d_dct),
            (f"E: masked CG DCT\n(λ=1e-1, L={L_e:.0f}m)",                    var_e_dct),
        ]),
    ]

    clim = (-5.0, 5.0)
    n_rows = len(rows)
    n_cols = max(len(r[1]) for r in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.4, n_rows * 5.6),
        constrained_layout=True,
    )
    for ri, (row_label, panels) in enumerate(rows):
        for ci, (title, da) in enumerate(panels):
            ax = axes[ri, ci]
            im = _imshow_xr(ax, da, cmap="RdBu_r", vmin=clim[0], vmax=clim[1])
            ax.set_title(f"{title}\n{_stats(da)}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.045)
            ax.set_xlabel("x (m)")
        axes[ri, 0].set_ylabel(f"{row_label}\n\ny (m)")
    fig.suptitle(
        f"Nansen IS2 — closed-form smoothing variants (FFT / DCT)  "
        f"({config.START_TIME} → {config.END_TIME}, H_ref={H_ref:.0f} m)",
        fontsize=12,
    )
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("done.")


if __name__ == "__main__":
    main()
