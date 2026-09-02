"""Per-strip error structure of a DEM stack: does the twin's model match PIG?

The synthetic observation model (``elmer_synth/scripts/make_dem_stack.py``)
injects, per strip: a plane tilt (0.5-3 dm/km), an offset (sigma 0.3 m), and,
by default, only spatially WHITE noise (0.3-0.9 m); with ``--corr-rms-m`` it
also injects a banded, spatially correlated component (the ``multixy_pigreal``
tier). Every twin-based solver conclusion is calibrated against whichever of
those models generated the twin. This script measures the same quantities on
a real (or twin) stack so the two can be compared like-for-like:

for each epoch, the residual about the per-pixel linear trend is decomposed
into an offset (spatial mean), a plane (least-squares tilt), and a detrended
remainder, and the remainder's spatial correlation is summarized by the
"block excess" at 2 and 4 km,

    E_L = std(L-block means) / ( rms(remainder) / sqrt(px per block) ),

which is 1 for white noise and grows with spatially correlated error (the
jitter/coregistration ripple real strips carry and the default white twin
does not).

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY -m pig.diagnose_stack_error_structure                 # real PIG stack
    $PY -m pig.diagnose_stack_error_structure --nc <twin.nc>  # a twin stack
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import NamedTuple

_REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, f"{_REPO}/examples")
sys.path.insert(0, f"{_REPO}/src")

from stereo_melt import envsetup  # noqa: F401,E402

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from pig import config  # noqa: E402


class PixelTrend(NamedTuple):
    """Per-pixel OLS linear trend of a stack plus the plane-fit coordinates."""

    t: np.ndarray        # epoch times (s) relative to the first
    fin: np.ndarray      # finite mask, (time, y, x)
    slope: np.ndarray    # trend slope per pixel (m/s)
    icept: np.ndarray    # trend intercept per pixel (m)
    ok_px: np.ndarray    # pixels with n >= 3 and a finite trend
    XX: np.ndarray       # x coordinate about the grid centre (km)
    YY: np.ndarray       # y coordinate about the grid centre (km)


def pixel_trend(stack: xr.DataArray) -> PixelTrend:
    """Per-pixel OLS trend by accumulation (no (t, y, x) temporaries)."""
    z = stack.values
    t = (stack.time.values - stack.time.values.min()) / np.timedelta64(1, "s")
    t = np.asarray(t, float)
    fin = np.isfinite(z)
    n = fin.sum(0)
    St = np.tensordot(t, fin, axes=(0, 0))
    Sz = np.nansum(z, 0)
    Stz = np.tensordot(t, np.where(fin, z, 0.0), axes=(0, 0))
    Stt = np.tensordot(t * t, fin, axes=(0, 0))
    den = n * Stt - St * St
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(den > 0, (n * Stz - St * Sz) / den, np.nan)
        icept = np.where(n > 0, (Sz - slope * St) / n, np.nan)
    ok_px = (n >= 3) & np.isfinite(slope)
    xk = (stack.x.values - stack.x.values.mean()) / 1e3     # km
    yk = (stack.y.values - stack.y.values.mean()) / 1e3
    XX, YY = np.meshgrid(xk, yk)
    return PixelTrend(t, fin, slope, icept, ok_px, XX, YY)


def epoch_residual(z: np.ndarray, trend: PixelTrend, i: int):
    """Epoch ``i``'s residual about the trend, offset and plane removed.

    Returns ``(offset_m, tilt_m_per_km, r2)`` where ``r2`` is the detrended
    remainder, NaN outside the epoch's valid (finite and trend-fitted) pixels.
    """
    m = trend.fin[i] & trend.ok_px
    r = z[i] - (trend.icept + trend.slope * trend.t[i])
    off = float(np.nanmean(np.where(m, r, np.nan)))
    # plane fit on the offset-removed residual
    A = np.stack([trend.XX[m], trend.YY[m], np.ones(m.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, (r - off)[m], rcond=None)
    tilt_m_km = float(np.hypot(coef[0], coef[1]))        # m per km
    r2 = np.where(m, r - off - coef[0] * trend.XX - coef[1] * trend.YY - coef[2],
                  np.nan)
    return off, tilt_m_km, r2


def per_epoch_stats(stack: xr.DataArray, block_kms=(2.0, 4.0), min_px=500):
    """Offset, tilt, detrended rms and block excess for every epoch."""
    z = stack.values
    trend = pixel_trend(stack)
    dx = float(abs(stack.x.values[1] - stack.x.values[0]))

    rows = []
    for i in range(z.shape[0]):
        m = trend.fin[i] & trend.ok_px
        if m.sum() < min_px:
            continue
        off, tilt_m_km, r2 = epoch_residual(z, trend, i)
        rms = float(np.sqrt(np.nanmean(r2 ** 2)))
        row = dict(epoch=i, offset_m=off, tilt_dm_km=10.0 * tilt_m_km, rms_m=rms,
                   px=int(m.sum()))
        for L in block_kms:
            bpx = max(int(round(L * 1e3 / dx)), 2)
            ny, nx = r2.shape
            ry, rx = ny // bpx * bpx, nx // bpx * bpx
            blk = r2[:ry, :rx].reshape(ry // bpx, bpx, rx // bpx, bpx)
            cnt = np.isfinite(blk).sum((1, 3))
            mu = np.nanmean(blk, (1, 3))
            good = cnt >= 0.5 * bpx * bpx
            if good.sum() < 8:
                row[f"excess_{L:g}km"] = np.nan
                continue
            npx_eff = float(np.mean(cnt[good]))
            row[f"excess_{L:g}km"] = float(np.nanstd(mu[good])
                                           / (rms / np.sqrt(npx_eff)))
        rows.append(row)
    return rows


REAL_STACK = "pig_stack_250m_is2ctempo_sheltilt"
REAL_LABEL = "REAL PIG 250 m is2ctempo_sheltilt (tilt-corrected)"


def load_real_pig_stack() -> xr.DataArray:
    """The production tilt-corrected PIG stack, floating-masked to the min extent.

    Shared by the diagnostic and the comparison figure so the fingerprint
    recorded by one is measured on exactly the stack drawn by the other.
    """
    from pig.run_melt import apply_min_extent, load_floating_mask, load_stack
    st = load_stack(REAL_STACK)
    fl = apply_min_extent(load_floating_mask(st), "_250m_is2ctempo_sheltilt",
                          str(config.START_TIME), str(config.END_TIME))
    return st.where(fl)


def stack_fingerprint(stack: xr.DataArray, nc=None) -> dict:
    """Identity of the stack the rows were measured on.

    Shape and time span alone do not separate twin tiers built from one
    template and seed (identical footprints), so the fingerprint also carries
    two value statistics: the stack mean and the mean squared x-difference,
    which the injected error model changes by orders of magnitude. ``nc`` is
    the source file (``None`` for the production PIG stack).
    """
    z = stack.values
    with np.errstate(invalid="ignore"):
        d2 = float(np.nanmean(np.diff(z, axis=-1) ** 2))
    return dict(nc=None if nc is None else os.path.realpath(nc),
                n_epochs=int(z.shape[0]), ny=int(z.shape[1]), nx=int(z.shape[2]),
                time_first=str(stack.time.values.min())[:19],
                time_last=str(stack.time.values.max())[:19],
                finite_px=int(np.isfinite(z).sum()),
                mean_m=float(np.nanmean(z)), dx2_m2=d2)


def check_fingerprint(d: dict, stack: xr.DataArray, nc, what: str) -> None:
    """Refuse a diagnostic JSON that was not measured on ``stack``.

    ``d`` is the JSON written by :func:`main`; ``nc`` is the file ``stack``
    was opened from (``None`` for the production PIG stack). A JSON written
    before fingerprints were recorded cannot be verified and is only warned
    about. Any recorded field that disagrees is an error: the JSON's rows,
    epoch indices and captions would otherwise describe a different stack
    from the one drawn.
    """
    want = d.get("fingerprint")
    if not want:
        print(f"  WARNING {what}: JSON carries no stack fingerprint; cannot verify "
              "it was measured on the stack being drawn (re-run "
              "diagnose_stack_error_structure to record one)", flush=True)
        return
    have = stack_fingerprint(stack, nc)
    problems = []
    for k in ("nc", "n_epochs", "ny", "nx", "time_first", "time_last", "finite_px"):
        if want.get(k) != have[k]:
            problems.append(f"{k}: JSON {want.get(k)!r} vs stack {have[k]!r}")
    for k in ("mean_m", "dx2_m2"):
        if not np.isclose(want.get(k, np.nan), have[k], rtol=1e-4, atol=1e-6):
            problems.append(f"{k}: JSON {want.get(k)!r} vs stack {have[k]!r}")
    if problems:
        raise SystemExit(
            f"{what}: the diagnostic JSON was not measured on the stack being "
            "drawn — captions and example epoch would describe a different "
            "stack:\n  " + "\n  ".join(problems))


def summarize(rows, label):
    def pct(key, q):
        v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
        return np.percentile(v, q) if len(v) else np.nan
    print(f"\n  {label}  ({len(rows)} epochs)")
    print(f"    {'metric':16s} {'p16':>8s} {'p50':>8s} {'p84':>8s}")
    for key in ("offset_m", "tilt_dm_km", "rms_m", "excess_2km", "excess_4km"):
        print(f"    {key:16s} {pct(key,16):8.2f} {pct(key,50):8.2f} {pct(key,84):8.2f}")
    return {k: [float(pct(k, q)) for q in (16, 50, 84)]
            for k in ("offset_m", "tilt_dm_km", "rms_m", "excess_2km", "excess_4km")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nc", default=None,
                    help="stack NetCDF (default: the production PIG stack)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.nc is None:
        st = load_real_pig_stack()
        label = args.label or REAL_LABEL
    else:
        ds = xr.open_dataset(args.nc)
        var = [v for v in ds.data_vars if ds[v].dims[-2:] == ("y", "x")
               and "time" in ds[v].dims][0]
        st = ds[var]
        label = args.label or args.nc
    rows = per_epoch_stats(st)
    summary = summarize(rows, label)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"label": label, "summary": summary,
                       "fingerprint": stack_fingerprint(st, args.nc),
                       "rows": rows}, f)
        print(f"  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
