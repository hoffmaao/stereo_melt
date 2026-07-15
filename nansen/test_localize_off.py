"""Single-knob experiment: re-run the Stubblefield linear inverse with
``localize=False`` and report the floating-only median against the
on-disk values from the production run.

The hypothesis is that ``localize=True`` (default in
``linear_inverse_lagrangian_melt_rate``) subtracts a per-epoch
outer-radial-ring mean from ``h_anom``, and that on Nansen's long thin
tongue the outer ring contains genuine basal-melt signal — so the
subtraction is suppressing the recovered melt rate by roughly the
regional mean.

Reuses ``load_stack`` / ``load_floating_mask`` / ``load_velocity_on_grid``
from ``nansen.run_melt`` so this stays apples-to-apples with the
production call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import xarray as xr

from stereo_melt.dynamics import (
    lagrangian_frame_stack,
    linear_inverse_lagrangian_melt_rate,
)

from nansen import config
from nansen.run_melt import (
    load_floating_mask,
    load_stack,
    load_velocity_on_grid,
)


def _summary(name: str, da: xr.DataArray) -> None:
    print(
        f"  [{name:<28}] median={float(da.median()):+.3f}  "
        f"IQR=[{float(da.quantile(0.25)):+.3f}, {float(da.quantile(0.75)):+.3f}]  "
        f"p05/p95=[{float(da.quantile(0.05)):+.3f}, {float(da.quantile(0.95)):+.3f}]  "
        f"abs_max={float(abs(da).max()):.1f}"
    )


def main() -> None:
    config.ensure_output_dirs()

    print("Loading stack + floating mask + velocity...")
    stack = load_stack()
    floating = load_floating_mask(stack)
    stack_floating = stack.where(floating)
    vx, vy, vel_source = load_velocity_on_grid(stack)
    print(f"  velocity source: {vel_source}")
    print(f"  stack: time={stack.sizes['time']}  y={stack.sizes['y']}  x={stack.sizes['x']}")

    print("Computing Lagrangian-frame stack (one-shot trajectory advection)...")
    h_lag = lagrangian_frame_stack(
        stack_floating,
        vx.where(floating),
        vy.where(floating),
        dt_yr=0.05,
    )

    print("Loading on-disk reference (production run, localize=True)...")
    ref_nc = (
        config.RESULTS_DIR
        / f"nansen_melt_{config.START_TIME}_{config.END_TIME}.nc"
    )
    ref_ds = xr.open_dataset(ref_nc)
    on_disk_lagr = ref_ds["melt_rate_lagrangian"].where(floating)
    on_disk_fft = ref_ds["melt_rate_linear_fft"].where(floating)
    on_disk_dct = ref_ds["melt_rate_linear_dct"].where(floating)
    print("\nProduction run (localize=True):")
    _summary("Lagrangian path-int", on_disk_lagr)
    _summary("linear-FFT (loc=True)", on_disk_fft)
    _summary("linear-DCT (loc=True)", on_disk_dct)

    common = dict(
        floating_mask=floating,
        reg=1e-1,
        dt_yr=0.05,
        h_lag_precomputed=h_lag,
        localize=False,  # the experimental knob
    )

    print("\nRunning linear-inverse FFT with localize=False...")
    linv_fft_off = linear_inverse_lagrangian_melt_rate(
        stack_floating, vx, vy,
        transform="fft", boundary_fix="infill+pad",
        **common,
    )
    print(f"  H_ref={linv_fft_off.attrs['H_ref_m']:.1f}  "
          f"gamma={linv_fft_off.attrs['gamma_dimless']:.3e}  "
          f"t_r={linv_fft_off.attrs['tr_yr']:.2f}")
    _summary("linear-FFT (loc=False)", linv_fft_off.melt_rate.where(floating))

    print("\nRunning linear-inverse DCT with localize=False...")
    linv_dct_off = linear_inverse_lagrangian_melt_rate(
        stack_floating, vx, vy,
        transform="dct", boundary_fix="off",
        **common,
    )
    _summary("linear-DCT (loc=False)", linv_dct_off.melt_rate.where(floating))

    out_nc = (
        config.RESULTS_DIR
        / f"nansen_localize_off_{config.START_TIME}_{config.END_TIME}.nc"
    )
    xr.Dataset(
        {
            "melt_rate_linear_fft_localize_off": linv_fft_off.melt_rate,
            "melt_rate_linear_dct_localize_off": linv_dct_off.melt_rate,
            "floating_mask": floating,
        },
        attrs={
            "experiment": "localize=False (override of run_melt default)",
            "H_ref_m": linv_fft_off.attrs["H_ref_m"],
            "gamma_dimless": linv_fft_off.attrs["gamma_dimless"],
            "tr_yr": linv_fft_off.attrs["tr_yr"],
            "ref_nc": str(ref_nc.name),
        },
    ).to_netcdf(out_nc)
    print(f"\nWrote {out_nc.name}")


if __name__ == "__main__":
    main()
