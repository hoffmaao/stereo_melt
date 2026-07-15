"""Common-footprint comparison: DEM-interior tilt-fit melt vs static-only baseline.

Stage 1 gate for the PIG ice-shelf-interior tilt-LSQ experiment
(``coregister/tilt.py::build_ice_domain_mask`` -> ``observation_mask``). Compares
the freshly-written DEM-interior melt result against the preserved
``*_staticonly_2026_06_07`` baseline on the pixels both resolve (the common
footprint), for dHdt and the Eulerian / Lagrangian melt fields.

Per field, on the common (both-finite, floating) footprint, it reports:
  median / IQR / robust-sigma   distribution width, new vs base  (did the shelf tighten?)
  tail fraction |.|>thr         the noise tail (Stage 1 target: shelf dHdt noise drops)
  spearman rho                  spatial-pattern agreement, new vs base
  sign-agreement (strong)       fraction agreeing on melt vs accretion where both commit
  median |new-base|             magnitude of the change

Shean convention: melt_rate negative = melt, positive = accretion.

Run::

    python -m pig.compare_deminterior_vs_staticonly                     # full window
    python -m pig.compare_deminterior_vs_staticonly --start 2018-10-01  # is2-era
    python -m pig.compare_deminterior_vs_staticonly --self-test         # baseline vs itself (rho=1)
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
import xarray as xr
from scipy.stats import spearmanr

from pig import config

BASELINE_SUFFIX = "_staticonly_2026_06_07"
RES_TAG = "250m"
# (field, sign-deadband m/yr, noise-tail threshold m/yr)
FIELDS = [
    ("dHdt", 0.5, 50.0),
    ("melt_rate_eulerian", 1.0, 100.0),
    ("melt_rate_lagrangian", 1.0, 100.0),
    ("melt_rate_linear_inverse", 1.0, 100.0),  # known-collapsed on PIG; reported for completeness
]


def _melt_path(start: str, end: str, suffix: str = ""):
    return config.RESULTS_DIR / f"pig_melt_{RES_TAG}_{start}_{end}{suffix}.nc"


def _iqr(a: np.ndarray) -> float:
    q25, q75 = np.percentile(a, [25, 75])
    return float(q75 - q25)


def _mad_sigma(a: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def compare(new_path, base_path) -> dict:
    print(f"  new : {new_path}")
    print(f"  base: {base_path}")
    dn = xr.open_dataset(new_path)
    db = xr.open_dataset(base_path)

    if dn["dHdt"].shape != db["dHdt"].shape:
        raise SystemExit(f"grid mismatch: new {dn['dHdt'].shape} vs base {db['dHdt'].shape}")

    def _float_mask(ds):
        if "floating_mask" in ds:
            return np.nan_to_num(ds["floating_mask"].values).astype(bool)
        return np.ones(ds["dHdt"].shape, bool)

    floating = _float_mask(dn) & _float_mask(db)
    print(f"  common floating pixels: {int(floating.sum()):,}")

    rows = {}
    for field, deadband, tail_thr in FIELDS:
        if field not in dn or field not in db:
            continue
        an = dn[field].values
        ab = db[field].values
        m = floating & np.isfinite(an) & np.isfinite(ab)
        n = int(m.sum())
        print(f"\n  {field}")
        if n < 100:
            print(f"    common n = {n}  (too few -- skipped)")
            continue
        xn, xb = an[m], ab[m]

        iqr_n, iqr_b = _iqr(xn), _iqr(xb)
        rho = float(spearmanr(xn, xb).correlation)
        # sign-agreement only where BOTH commit to a sign (|.| >= deadband)
        strong = (np.abs(xn) >= deadband) & (np.abs(xb) >= deadband)
        ns = int(strong.sum())
        sign = float(np.mean(np.sign(xn[strong]) == np.sign(xb[strong]))) if ns else float("nan")
        tail_n = float(np.mean(np.abs(xn) > tail_thr))
        tail_b = float(np.mean(np.abs(xb) > tail_thr))
        med_abs = float(np.median(np.abs(xn - xb)))
        tighten = (iqr_b / iqr_n) if iqr_n > 0 else float("inf")

        print(f"    common n           : {n:,}")
        print(f"    median   new/base  : {np.median(xn):8.3f} / {np.median(xb):8.3f}  m/yr")
        print(f"    IQR      new/base  : {iqr_n:8.3f} / {iqr_b:8.3f}  m/yr   (x{tighten:.2f} {'tighter' if tighten>=1 else 'WIDER'})")
        print(f"    robust σ new/base  : {_mad_sigma(xn):8.3f} / {_mad_sigma(xb):8.3f}  m/yr  (1.4826*MAD)")
        print(f"    tail |·|>{tail_thr:g} n/b  : {100*tail_n:7.2f}% / {100*tail_b:6.2f}%   (noise tail)")
        print(f"    spearman rho       : {rho:8.3f}")
        print(f"    sign-agree(strong) : {100*sign:7.1f}%  (over {100*ns/n:.1f}% of footprint, both |·|>={deadband:g})")
        print(f"    median |new-base|  : {med_abs:8.3f}  m/yr")
        rows[field] = dict(n=n, rho=rho, sign=sign, iqr_new=iqr_n, iqr_base=iqr_b,
                           tighten=tighten, tail_new=tail_n, tail_base=tail_b,
                           med_new=float(np.median(xn)), med_base=float(np.median(xb)),
                           med_abs=med_abs)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=config.START_TIME)
    p.add_argument("--end", default=config.END_TIME)
    p.add_argument("--new", default=None, help="override new melt nc path")
    p.add_argument("--base", default=None, help="override baseline melt nc path")
    p.add_argument("--self-test", action="store_true",
                   help="compare the baseline against itself (expect rho=1, sign=100%%, med|Δ|=0)")
    args = p.parse_args()

    if args.self_test:
        new = base = _melt_path(args.start, args.end, BASELINE_SUFFIX)
    else:
        new = args.new or _melt_path(args.start, args.end)
        base = args.base or _melt_path(args.start, args.end, BASELINE_SUFFIX)

    for label, path in (("new", new), ("base", base)):
        if not os.path.isfile(path):
            raise SystemExit(f"{label} melt result not found: {path}")

    print(f"PIG DEM-interior vs static-only  |  window {args.start} .. {args.end}\n")
    compare(new, base)
    print("\n  Shean convention: negative = melt, positive = accretion.")
    print("  Stage 1 gate: shelf dHdt IQR/tail drops AND melt map gains coherent structure"
          "\n                WITHOUT inflating the static-control bias (see tilt-QC weight_mean/frac_kept).")


if __name__ == "__main__":
    main()
