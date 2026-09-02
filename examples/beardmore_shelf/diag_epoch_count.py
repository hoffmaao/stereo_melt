"""Per-pixel epoch-count diagnostic over the floating shelf.

The FIRST diagnostic for the combined 2012-2024 stack lever
(project_session_resume_2026_07_07, lever 1): the tilt LSQ is degenerate
where a floating pixel is observed by only 1-2 strips, so the coverage-
conditioning question is literally "did the thin 2-image cells become
4-5 image?" once the pre-IS2 CryoTEMPO/ATM epochs are folded in.

This counts, at every grid cell, how many epochs carry a finite value in
the *curated* tilt-corrected stack (BAD_STRIPS already dropped, i.e. the
stack the solvers actually see), restricts to the BedMachine floating
mask, and reports the depth histogram + the fraction of floating cells
below each conditioning threshold. It also reports the same stats inside
the RED_accr stripe band (the target region from diag_stripe_strips) so
we can see whether the stripe cells specifically are coverage-starved.

The count map is saved to
``processed/beardmore_shelf_epoch_count_<res>m[_<tag>]_<start>_<end>.nc``
so the pre-align (IS2-only) baseline and the post-align (combined) run can
be differenced cell-by-cell.

Usage::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    # baseline (IS2-era production stack):
    $PY -m beardmore_shelf.diag_epoch_count --res 125
    # after the combined build (whatever --tag it was written with):
    $PY -m beardmore_shelf.diag_epoch_count --res 125 --tag combined
"""
from __future__ import annotations

import argparse
import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import numpy as np
import pandas as pd
import xarray as xr

from beardmore_shelf import config
from beardmore_shelf.diag_stripe_strips import BANDS
from beardmore_shelf.run_melt import load_floating_mask
from beardmore_shelf.run_melt_path import load_stack_res

# Conditioning thresholds: a floating cell seen by < THIN epochs gives the
# per-epoch tilt LSQ almost nothing to separate melt/dh from a tilt-plane
# offset; >= WELL epochs is comfortably over-determined.
THIN = 3   # cells with < THIN observations are "degenerate"
WELL = 5   # cells with >= WELL observations are "well-conditioned"


def _band_mask(ny: int, nx: int, box, flo: np.ndarray) -> np.ndarray:
    c0, c1, r0, r1 = box
    m = np.zeros((ny, nx), bool)
    m[r0:r1, c0:c1] = True
    return m & flo


def _report(count: np.ndarray, sel: np.ndarray, label: str) -> dict:
    """Print + return the depth stats over the boolean selection ``sel``."""
    c = count[sel]
    n = int(c.size)
    if n == 0:
        print(f"  [{label}] no cells")
        return {}
    stats = {
        "n": n,
        "median": float(np.median(c)),
        "mean": float(np.mean(c)),
        f"frac_lt_{THIN}": float(np.mean(c < THIN)),
        f"frac_lt_{WELL}": float(np.mean(c < WELL)),
        f"frac_ge_{WELL}": float(np.mean(c >= WELL)),
        "frac_eq1": float(np.mean(c == 1)),
        "frac_eq2": float(np.mean(c == 2)),
    }
    # Depth histogram, capped bin at 10+.
    edges = [1, 2, 3, 4, 5, 6, 8, 10]
    print(f"  [{label}]  cells={n:,}  median={stats['median']:.0f}  "
          f"mean={stats['mean']:.1f}")
    hist_parts = []
    for lo, hi in zip([0] + edges, edges + [10**9]):
        if hi == 10**9:
            k = int(np.sum(c >= lo))
            hist_parts.append(f">={lo}:{k}")
        else:
            k = int(np.sum((c >= lo) & (c < hi)))
            hist_parts.append(f"{lo}-{hi-1}:{k}")
    print("        depth: " + "  ".join(hist_parts))
    print(f"        thin(<{THIN}): {stats[f'frac_lt_{THIN}']*100:.1f}%   "
          f"under-det(<{WELL}): {stats[f'frac_lt_{WELL}']*100:.1f}%   "
          f"well(>={WELL}): {stats[f'frac_ge_{WELL}']*100:.1f}%   "
          f"(==1: {stats['frac_eq1']*100:.1f}%  ==2: {stats['frac_eq2']*100:.1f}%)")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res", type=int, default=125)
    p.add_argument("--tag", type=str, default=None,
                   help="Stack variant tag (e.g. 'combined'); None = production.")
    args = p.parse_args()

    tag_txt = args.tag or "production(is2cs2)"
    print(f"=== epoch-count diagnostic  res={args.res}m  tag={tag_txt} ===")
    stack = load_stack_res(args.res, tag=args.tag)
    times = pd.to_datetime(stack["time"].values)
    ny, nx = stack.sizes["y"], stack.sizes["x"]
    print(f"  stack dims: time={stack.sizes['time']}, y={ny}, x={nx}  "
          f"({times.min().date()} .. {times.max().date()})")

    variants = None
    if "source_variant" in stack.coords:
        vc = pd.Series(stack["source_variant"].values).value_counts()
        variants = vc.to_dict()
        print("  source_variant epochs: "
              + ", ".join(f"{k}={v}" for k, v in vc.items()))

    floating = load_floating_mask(stack)
    flo = np.asarray(floating.values, bool)

    # Per-pixel finite-epoch count.
    count = np.isfinite(stack.values).sum(axis=0).astype(np.int32)  # (y, x)

    print(f"\n  floating cells: {int(flo.sum()):,}")
    print("  --- depth over the whole floating shelf ---")
    ref_stats = _report(count, flo, "REF_floating")

    print("  --- depth inside the stripe bands ---")
    band_stats = {}
    for name, box in BANDS.items():
        m = _band_mask(ny, nx, box, flo)
        band_stats[name] = _report(count, m, name)

    # Persist the count map for before/after differencing.
    out = xr.Dataset(
        {"epoch_count": (("y", "x"), np.where(flo, count, 0).astype(np.int32))},
        coords={"x": stack["x"], "y": stack["y"]},
    )
    out["epoch_count"].attrs.update(
        long_name="finite epochs per cell in curated tilt-corrected stack",
        res_m=args.res, tag=tag_txt, n_time=int(stack.sizes["time"]),
    )
    if variants:
        out.attrs["source_variant_counts"] = str(variants)
    suffix = f"_{args.res}m" + (f"_{args.tag}" if args.tag else "")
    out_path = (
        config.PROCESSED_DIR
        / f"beardmore_shelf_epoch_count{suffix}"
        f"_{config.START_TIME}_{config.END_TIME}.nc"
    )
    out.to_netcdf(out_path)
    print(f"\n  saved count map -> {out_path.name}")

    # Headline the shelf-wide degeneracy — this is the number lever (1) must move.
    if ref_stats:
        print(f"\n  HEADLINE: {ref_stats[f'frac_lt_{THIN}']*100:.1f}% of floating "
              f"cells are thin (<{THIN} epochs); "
              f"{ref_stats[f'frac_ge_{WELL}']*100:.1f}% well-conditioned "
              f"(>={WELL}).")


if __name__ == "__main__":
    main()
