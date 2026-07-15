"""Sweep ``reg`` for the dh/dt single-step Fourier inverse.

Reads the pre-computed dh/dt slope from the five-methods NetCDF and re-runs
:func:`inverse_dhdt` for several Tikhonov values. Prints summary stats and
saves a small grid of maps so we can pick a value that produces a melt-rate
field on the same ±5 m/yr scale as the closed-form/Lagrangian panels.

Run::

    python -m beardmore.sweep_dhdt_reg
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

from stereo_melt.dynamics.linear_perturbation import inverse_dhdt

from beardmore import config
from beardmore.run_melt import load_stack


def _stats(da: xr.DataArray, mask) -> str:
    v = da.values[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "no data"
    return (
        f"med={np.median(v):+.2f}  IQR=[{np.percentile(v, 25):+.2f}, "
        f"{np.percentile(v, 75):+.2f}]  p05/p95=[{np.percentile(v, 5):+.2f}, "
        f"{np.percentile(v, 95):+.2f}]  |max|={np.max(np.abs(v)):.1f}"
    )


def main() -> None:
    nc_path = (
        config.RESULTS_DIR
        / f"beardmore_five_methods_{config.START_TIME}_{config.END_TIME}.nc"
    )
    print(f"Loading -> {nc_path}")
    ds = xr.open_dataset(nc_path)
    floating = ds.floating_mask.astype(bool).values

    H_ref = float(ds.attrs.get("H_ref_m", float("nan")))
    print(f"H_ref={H_ref:.1f} m")

    stack = load_stack()
    t_vals = stack["time"].values
    if np.issubdtype(np.asarray(t_vals).dtype, np.datetime64):
        t_secs = (np.asarray(t_vals) - np.asarray(t_vals)[0]).astype(
            "timedelta64[s]"
        ).astype(np.float64)
    else:
        t_secs = np.asarray(t_vals, dtype=np.float64)
        t_secs = t_secs - t_secs[0]
    print(f"t_secs span: {t_secs.max() / (86400 * 365.25):.2f} yr, n={len(t_secs)}")

    dh_dt = ds.dh_dt_input.copy()
    dh_dt.attrs.setdefault("units", "m s^-1")

    regs = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0]
    transforms = ["fft", "dct"]

    fig, axes = plt.subplots(
        len(transforms), len(regs), figsize=(len(regs) * 3.6, len(transforms) * 4.4),
        constrained_layout=True,
    )
    if len(transforms) == 1:
        axes = np.atleast_2d(axes)

    for col, reg in enumerate(regs):
        for row, tr in enumerate(transforms):
            m = inverse_dhdt(
                dh_dt, H=H_ref, t_secs=t_secs,
                alpha=0.0, alpha_y=0.0, gamma=ds.attrs.get("gamma", 0.0),
                theta=1e-14, reg=float(reg), transform=tr,
            ).where(xr.DataArray(floating, dims=("y", "x"), coords=dh_dt.coords))
            print(f"  reg={reg:<6g} {tr}: {_stats(m, floating)}")
            ax = axes[row, col]
            ax.imshow(
                m.values,
                extent=(
                    float(dh_dt.x.min()), float(dh_dt.x.max()),
                    float(dh_dt.y.min()), float(dh_dt.y.max()),
                ),
                origin="upper" if dh_dt.y.values[0] > dh_dt.y.values[-1] else "lower",
                cmap="RdBu_r", vmin=-5, vmax=5, aspect="equal", interpolation="nearest",
            )
            v = m.values[floating]
            v = v[np.isfinite(v)]
            stats_lite = (
                f"med={np.median(v):+.2f}  |max|={np.max(np.abs(v)):.0f}"
                if v.size else "no data"
            )
            ax.set_title(f"{tr.upper()}  reg={reg:g}\n{stats_lite}", fontsize=10)

    fig.suptitle(
        "dh/dt single-step Fourier inverse — reg sweep "
        f"({config.START_TIME} → {config.END_TIME}, ±5 m/yr scale)",
        fontsize=12,
    )
    fig_path = config.FIGURES_DIR / "dhdt_reg_sweep.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
