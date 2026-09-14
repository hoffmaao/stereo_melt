"""Package the PIG fast-trunk window for the B-PINN (run in the stereo_melt env).

Reuses run_melt's loaders so the B-PINN sees exactly the production inputs:
the tilt-corrected stack for ``--tag`` (default the qcey canon), the min-extent
floating mask, the fused time-varying velocity, RACMO SMB and BedMachine firn.
The window is the largest connected patch of strong Eulerian melt in the
1 km-smoothed canon product (the same box as ``plot_melt_benchmark --zoom
auto``), padded by ``--pad`` pixels. Thickness is the firn-corrected
hydrostatic thickness the Eulerian solver uses. Writes
``results/bpinn/trunk_<tag>.npz`` with the production Eulerian, Lagrangian and
monolithic v2 fields on the same window as benchmarks.

    cd examples && PIG_VELOCITY=fused $PY -m pig.prep_bpinn_trunk [--tag is2ctempo_sheltilt_qcey]
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
os.environ.setdefault("PIG_VELOCITY", "fused")

from pig import config  # noqa: E402
import pig.run_melt as rm  # noqa: E402
from stereo_melt.constants import rhoi, rhow  # noqa: E402
from stereo_melt.freeboard import freeboard_to_thickness  # noqa: E402
from stereo_melt.io.bedmachine import load_firn_on_grid  # noqa: E402


def _check_grid(name, ds, stack, tag, W) -> None:
    """Raise unless ``ds`` sits on exactly ``stack``'s x/y grid.

    The trunk window is sliced by positional index, so a product on any other grid
    would come out mis-registered by whole pixels with no other symptom.
    """
    if not (np.array_equal(np.asarray(ds.x.values), np.asarray(stack.x.values))
            and np.array_equal(np.asarray(ds.y.values), np.asarray(stack.y.values))):
        raise SystemExit(
            f"grid mismatch: {name} is on a different x/y grid than the stack "
            f"({ds.sizes.get('y')}x{ds.sizes.get('x')} vs {stack.sizes['y']}x{stack.sizes['x']}); "
            f"the trunk window is sliced by position, so its fields would be mis-registered. "
            f"Rebuild it for tag {tag} window {W}.")


def _decimal_year(times) -> np.ndarray:
    t = pd.to_datetime(np.asarray(times))
    return np.asarray(t.year + (t.dayofyear - 1) / 365.25, float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="is2ctempo_sheltilt_qcey")
    ap.add_argument("--pad", type=int, default=8)
    args = ap.parse_args()
    W = f"{config.START_TIME}_{config.END_TIME}"
    tag = args.tag
    stack = rm.load_stack(stack_prefix=f"pig_stack_250m_{tag}")
    floating = rm.load_floating_mask(stack)
    floating = rm.apply_min_extent(floating, f"_250m_{tag}", config.START_TIME, config.END_TIME)
    stack = stack.where(floating)
    vx, vy, vel_source = rm.load_velocity_on_grid(stack)
    vx, vy, _ = rm._maybe_smooth_velocity(vx, vy)
    a_dot = rm.load_smb_on_grid(stack)
    sig = inspect.signature(load_firn_on_grid)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC) if len(sig.parameters) >= 2 else load_firn_on_grid(stack)
    firn = firn if isinstance(firn, xr.DataArray) else xr.DataArray(np.asarray(firn), coords={"y": stack.y, "x": stack.x}, dims=("y", "x"))

    # window: largest connected strong-melt patch of the canon Eulerian product
    prod = xr.open_dataset(config.RESULTS_DIR / f"pig_melt_250m_{tag}_{W}.nc")
    _check_grid(f"run_melt product pig_melt_250m_{tag}_{W}.nc", prod, stack, tag, W)
    e = prod.melt_rate_eulerian.values
    v = np.where(np.isfinite(e), e, 0.0)
    w = np.isfinite(e).astype(float)
    sm = ndimage.gaussian_filter(v, 4) / np.maximum(ndimage.gaussian_filter(w, 4), 1e-6)
    lab, n = ndimage.label((sm < -30.0) & (w > 0))
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    zy, zx = np.where(lab == 1 + int(np.argmax(sizes)))
    r0, r1 = max(zy.min() - args.pad, 0), min(zy.max() + args.pad, e.shape[0])
    c0, c1 = max(zx.min() - args.pad, 0), min(zx.max() + args.pad, e.shape[1])
    print(f"window rows {r0}:{r1} cols {c0}:{c1} = {r1 - r0} x {c1 - c0} px "
          f"(x {float(stack.x[c0])/1e3:.0f}..{float(stack.x[c1-1])/1e3:.0f} km, y {float(stack.y[r1-1])/1e3:.0f}..{float(stack.y[r0])/1e3:.0f} km)")

    h = stack.isel(y=slice(r0, r1), x=slice(c0, c1))
    d = firn.isel(y=slice(r0, r1), x=slice(c0, c1))
    H = freeboard_to_thickness(h, d=d, rho_w=rhow, rho_i=rhoi)
    Hv = np.asarray(H.values, np.float32)
    dom = floating.isel(y=slice(r0, r1), x=slice(c0, c1)).values.astype(bool)
    t_yr = _decimal_year(h.time.values)
    vxw, vyw = vx.isel(y=slice(r0, r1), x=slice(c0, c1)), vy.isel(y=slice(r0, r1), x=slice(c0, c1))
    vt_yr = _decimal_year(vxw.time.values) if "time" in vxw.dims else None
    med = np.nanmedian(Hv, axis=0)
    resid = Hv - med
    off = np.nanmedian(resid.reshape(resid.shape[0], -1), axis=1)
    r = (resid - off[:, None, None])[np.isfinite(Hv)]
    nmad = 1.4826 * np.median(np.abs(r - np.median(r)))
    print(f"stack {Hv.shape}, finite {np.isfinite(Hv).mean():.2f}, H median {np.nanmedian(Hv):.0f} m; "
          f"per-epoch residual NMAD (offset-removed, thickness units) {nmad:.1f} m; velocity {vel_source[:60]}; "
          f"{'time-varying ' + str(vxw.sizes['time']) + ' epochs' if vt_yr is not None else 'static'}; "
          f"|v| mean {float(np.hypot(vxw, vyw).mean()):.0f} m/yr; a_dot mean {float(a_dot.isel(y=slice(r0, r1), x=slice(c0, c1)).mean()):.2f} m/yr")
    bench = {}
    brid = xr.open_dataset(config.PROCESSED_DIR / f"pig_melt_bridging_250m_{tag}_{W}.nc")
    _check_grid(f"bridging product pig_melt_bridging_250m_{tag}_{W}.nc", brid, stack, tag, W)
    for k in ("eulerian", "monolithic_v2", "restored_local_helm"):
        bench[f"bench_{k}"] = brid[k].values[r0:r1, c0:c1]
    bench["bench_lagrangian"] = prod.melt_rate_lagrangian.values[r0:r1, c0:c1]
    out = Path(config.RESULTS_DIR) / "bpinn" / f"trunk_{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, H_obs=Hv, x=h.x.values, y=h.y.values, t_yr=t_yr,
                        vx=np.asarray(vxw.values, np.float32), vy=np.asarray(vyw.values, np.float32),
                        vt_yr=(vt_yr if vt_yr is not None else np.array([])),
                        a_dot=np.asarray(a_dot.isel(y=slice(r0, r1), x=slice(c0, c1)).values, np.float32),
                        domain=dom, rho_i=float(rhoi), rho_w=float(rhow), sigma_H_nmad=float(nmad),
                        window=np.array([r0, r1, c0, c1]), **bench)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
