"""Survey-realistic tier: package E2a truth as a DEM-method-consistent stack.

Implements the plan's observation-synthesis tier 2
(`literature/plan_elmer_synthetic_validation.md` section 4): sample the Elmer
truth surface at a REAL basin's acquisition epochs, imprint each epoch's REAL
strip NaN footprint (from the beardmore_shelf full-record stack, transplanted
through a fixed 24x12 km crop), inject the per-strip error model (plane tilts
0.5-3 dm/km at random azimuth, per-strip bias, 0.3-0.9 m white noise, a few
deliberately corrupted strips), and write a ``save_stack``-compatible NetCDF
so ``fit_tilt_stack`` -> screening -> the production melt solvers run
UNMODIFIED. A sidecar JSON records every injected error term so the DEM
chain's tilt recovery can be scored against truth.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY elmer_synth/scripts/make_dem_stack.py --pert trans_gauss_a5 --t0 95 --tag steady
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

sys.path.insert(0, "/wd2/projects/stereo_melt/elmer_synth")
sys.path.insert(0, "/wd2/projects/stereo_melt/elmer_synth/scripts")
sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")
from score_e2a import load_run  # noqa: E402
from stereo_melt.stack import save_stack  # noqa: E402

TEMPLATE = ("/wd2/projects/stereo_melt/beardmore_shelf/processed/"
            "beardmore_shelf_stack_fullrec_2009-01-01_2024-01-10.nc")
COUNTS = ("/wd2/projects/stereo_melt/beardmore_shelf/processed/"
          "beardmore_shelf_epoch_count_125m_2012-11-18_2024-01-10.nc")
OUT_DIR = Path("/wd2/projects/stereo_melt/elmer_synth/data/processed")
RES_DIR = Path("/wd2/projects/stereo_melt/elmer_synth/results")
CROP_NX, CROP_NY = 192, 96  # 24 x 12 km at the template's 125 m posting

# per-epoch vertical prior by control class (mirrors build_per_epoch_ez;
# keys = the fullrec template's actual source_variant values). ctempoatm is
# a CryoTEMPO/ATM mix -- without the per-dem sidecars, take the conservative
# CryoTEMPO-grade prior.
EZ_BY_VARIANT = {"is2cs2": 0.1, "ctempoatm": 1.0, "glas": 0.3, "nocorr": 1.0}


def pick_crop(count_path: str) -> tuple[int, int]:
    """Corner (iy, ix) of the CROP_NY x CROP_NX box maximizing mean epoch count."""
    ds = xr.open_dataset(count_path)
    var = [v for v in ds.data_vars if v != "spatial_ref"][0]
    c = np.nan_to_num(ds[var].values.astype(float))
    sm = uniform_filter(c, size=(CROP_NY, CROP_NX), mode="constant")
    # valid centers only (full box inside the grid)
    sm[: CROP_NY // 2, :] = -1
    sm[-CROP_NY // 2:, :] = -1
    sm[:, : CROP_NX // 2] = -1
    sm[:, -CROP_NX // 2:] = -1
    iyc, ixc = np.unravel_index(np.argmax(sm), sm.shape)
    ds.close()
    return iyc - CROP_NY // 2, ixc - CROP_NX // 2


def plane_fit(field: np.ndarray, mask: np.ndarray, x2d: np.ndarray, y2d: np.ndarray):
    """Least-squares plane of ``field`` over ``mask`` as ``(slope_m_per_m, azimuth_rad)``.

    Same parameterisation as the injected tilt: the plane is
    ``slope * (cos(az) * x + sin(az) * y)`` plus an offset.
    """
    if mask.sum() < 3:
        return 0.0, 0.0
    A = np.stack([x2d[mask], y2d[mask], np.ones(int(mask.sum()))], 1)
    coef, *_ = np.linalg.lstsq(A, field[mask], rcond=None)
    return float(np.hypot(coef[0], coef[1])), float(np.arctan2(coef[1], coef[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pert", required=True, help="E2a truth run (e.g. trans_gauss_a5)")
    ap.add_argument("--t0", type=float, default=95.0,
                    help="truth year mapped to the template record start")
    ap.add_argument("--tag", default="steady")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-coverage", type=float, default=0.05,
                    help="drop strips covering less of the crop than this")
    ap.add_argument("--tilt-dm-km", type=float, nargs=2, default=(0.5, 3.0),
                    help="uniform range of plane-tilt slopes (dm/km)")
    ap.add_argument("--noise-m", type=float, nargs=2, default=(0.3, 0.9))
    ap.add_argument("--bias-m", type=float, default=0.3, help="per-strip offset sigma")
    ap.add_argument("--corrupt-frac", type=float, default=0.03)
    ap.add_argument("--corr-rms-m", type=float, nargs=2, default=(0.0, 0.0),
                    help="per-strip CORRELATED error rms, drawn log-uniform in "
                         "this range (m); 0 0 disables. Calibrated against the "
                         "real PIG 250 m stack 2026-08-29: (1.2, 5.7) reproduces "
                         "the measured detrended-rms spread")
    ap.add_argument("--corr-beta", type=float, default=2.0,
                    help="spectral slope of the correlated error (PSD ~ k^-beta)")
    ap.add_argument("--corr-aniso", type=float, default=3.0,
                    help="anisotropy of the correlated error (banded, random "
                         "azimuth per strip, like jitter ripple)")
    ap.add_argument("--corr-lmax-km", type=float, default=9.0,
                    help="high-pass: wavelengths above this are left to the "
                         "offset/tilt terms. The cut is applied to the stretched "
                         "wavenumber, so along the band direction wavelengths up "
                         "to corr_aniso * corr_lmax_km still pass; the plane the "
                         "field then carries is recorded per strip as "
                         "corr_tilt_slope_m_per_m / corr_tilt_azimuth_rad. The "
                         "field is zero-mean over the full crop, so its mean over "
                         "the strip's kept footprint is an extra per-strip offset; "
                         "it is recorded as corr_offset_m (not removed) so the "
                         "sidecar's offset truth is complete — the tilt fit "
                         "absorbs it like the drawn bias")
    ap.add_argument("--corrupt-bias-m", type=float, default=3.0,
                    help="offset sigma of the deliberately corrupted strips "
                         "(their tilt is 8x the drawn slope); scale it with "
                         "--bias-m on an error ladder or it sets the raw floor")
    ap.add_argument("--max-strips", type=int, default=None)
    ap.add_argument("--clean-too", action="store_true",
                    help="also write the error-free sampled stack")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    truth, meta = load_run(args.pert, None)
    tx, ty = truth.x.values, truth.y.values
    t_frames = truth["time_yr"].values

    iy0, ix0 = pick_crop(COUNTS)
    print(f"[crop] template box corner (iy,ix)=({iy0},{ix0}) "
          f"({CROP_NY}x{CROP_NX} = 12x24 km)", flush=True)

    tpl = xr.open_dataset(TEMPLATE)
    var = [v for v in tpl.data_vars if v != "spatial_ref"][0]
    dates = pd.to_datetime(tpl.time.values)
    date0 = dates[0]
    dem_ids = np.asarray(tpl.dem_id.values, dtype=str)
    variants = (np.asarray(tpl.source_variant.values, dtype=str)
                if "source_variant" in tpl.coords else np.full(len(dates), "nocorr"))

    # nearest-index resampling: template crop (96x192) -> truth grid (61x121)
    jj = np.clip(np.round(np.linspace(0, CROP_NY - 1, ty.size)).astype(int), 0, CROP_NY - 1)
    ii = np.clip(np.round(np.linspace(0, CROP_NX - 1, tx.size)).astype(int), 0, CROP_NX - 1)

    x2d, y2d = np.meshgrid(tx - tx.mean(), ty - ty.mean())
    dx_m = float(abs(tx[1] - tx[0]))
    kept, layers, records = [], [], []
    for k in range(len(dates)):
        t_truth = args.t0 + (dates[k] - date0).days / 365.25
        if not (t_frames[0] <= t_truth <= t_frames[-1]):
            continue
        crop = tpl[var].isel(time=k, y=slice(iy0, iy0 + CROP_NY),
                             x=slice(ix0, ix0 + CROP_NX)).values
        mask = np.isfinite(crop)[np.ix_(jj, ii)]
        cov = float(mask.mean())
        if cov < args.min_coverage:
            continue

        # truth surface at the acquisition time (linear interp between frames)
        i1 = int(np.searchsorted(t_frames, t_truth).clip(1, len(t_frames) - 1))
        w = (t_truth - t_frames[i1 - 1]) / (t_frames[i1] - t_frames[i1 - 1])
        zs = ((1 - w) * truth.zs.isel(time=i1 - 1).values
              + w * truth.zs.isel(time=i1).values)

        # error model (plan section 4, tier 2)
        corrupted = bool(rng.random() < args.corrupt_frac)
        slope = rng.uniform(*args.tilt_dm_km) * 0.1 / 1000.0  # dm/km -> m/m
        azim = rng.uniform(0, 2 * np.pi)
        bias = float(rng.normal(0, args.bias_m))
        sigma = float(rng.uniform(*args.noise_m))
        if corrupted:
            slope *= 8.0
            bias = float(rng.normal(0, args.corrupt_bias_m))
        plane = slope * (np.cos(azim) * x2d + np.sin(azim) * y2d)
        z = zs + plane + bias + rng.normal(0, sigma, zs.shape)
        corr_rms, corr_tilt, corr_azim, corr_off = 0.0, 0.0, 0.0, 0.0
        if args.corr_rms_m[1] > 0:
            # banded correlated error (jitter/coreg ripple): anisotropic
            # power-law field at a random azimuth, high-passed so scales
            # above corr_lmax_km stay with the offset/tilt terms. Amplitude
            # log-uniform across strips, matching the real per-strip
            # detrended-rms spread.
            corr_rms = float(np.exp(rng.uniform(np.log(args.corr_rms_m[0]),
                                                np.log(args.corr_rms_m[1]))))
            ny_, nx_ = zs.shape
            ky = np.fft.fftfreq(ny_, d=dx_m)[:, None]
            kx = np.fft.fftfreq(nx_, d=dx_m)[None, :]
            az_c = rng.uniform(0, np.pi)
            k_al = np.cos(az_c) * kx + np.sin(az_c) * ky
            k_ac = -np.sin(az_c) * kx + np.cos(az_c) * ky
            keff = np.sqrt((k_al * args.corr_aniso) ** 2 + k_ac ** 2)
            keff[0, 0] = np.inf
            filt = keff ** (-args.corr_beta / 2.0)
            kmin = 1.0 / (args.corr_lmax_km * 1e3)
            filt *= 1.0 / (1.0 + (kmin / np.maximum(keff, 1e-12)) ** 4)
            f = np.real(np.fft.ifft2(np.fft.fft2(
                rng.normal(size=(ny_, nx_))) * filt))
            f -= f.mean()
            g = corr_rms * (f / max(f.std(), 1e-12))
            corr_tilt, corr_azim = plane_fit(g, mask, x2d, y2d)
            corr_off = float(g[mask].mean()) if mask.any() else 0.0
            z = z + g
        z[~mask] = np.nan

        kept.append(k)
        layers.append(z.astype(np.float32))
        records.append(dict(
            dem_id=f"SYNTH_{dates[k]:%Y%m%d}_{k:04d}",
            template_dem_id=dem_ids[k], date=str(dates[k].date()),
            t_truth_yr=t_truth, coverage=cov, variant=variants[k],
            Ez=EZ_BY_VARIANT.get(variants[k], 1.0),
            tilt_slope_m_per_m=slope, tilt_azimuth_rad=azim,
            bias_m=bias, noise_sigma_m=sigma, corr_rms_m=corr_rms,
            corr_tilt_slope_m_per_m=corr_tilt, corr_tilt_azimuth_rad=corr_azim,
            corr_offset_m=corr_off, corrupted=corrupted,
        ))
        if args.max_strips and len(kept) >= args.max_strips:
            break
    tpl.close()

    print(f"[keep] {len(kept)} strips of {len(dates)} template epochs "
          f"(coverage>={args.min_coverage}, t in [{t_frames[0]:.0f},{t_frames[-1]:.0f}])",
          flush=True)
    if not kept:
        raise SystemExit("no strips kept — check --t0 mapping vs truth time range")

    times = dates[kept]
    stack = xr.DataArray(
        np.stack(layers), dims=("time", "y", "x"),
        coords=dict(time=times, y=ty, x=tx,
                    dem_id=("time", [r["dem_id"] for r in records]),
                    source_variant=("time", [r["variant"] for r in records])),
        name="z",
        attrs=dict(description=f"survey-realistic synthetic stack from {args.pert}",
                   truth_run=args.pert, t0_map_yr=args.t0, seed=args.seed),
    )
    try:
        import rioxarray  # noqa: F401
        stack = stack.rio.write_crs("EPSG:3031")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] no CRS written ({exc})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_nc = OUT_DIR / f"e2a_stack_200m_{args.tag}.nc"
    save_stack(stack, out_nc)
    print(f"wrote {out_nc}  ({len(kept)} epochs, grid {ty.size}x{tx.size})")

    if args.clean_too:
        clean = stack.copy()
        for j, (k, rec) in enumerate(zip(kept, records)):
            t_truth = rec["t_truth_yr"]
            i1 = int(np.searchsorted(t_frames, t_truth).clip(1, len(t_frames) - 1))
            w = (t_truth - t_frames[i1 - 1]) / (t_frames[i1] - t_frames[i1 - 1])
            zs = ((1 - w) * truth.zs.isel(time=i1 - 1).values
                  + w * truth.zs.isel(time=i1).values)
            layer = np.where(np.isfinite(stack.values[j]), zs, np.nan)
            clean.values[j] = layer.astype(np.float32)
        out_clean = OUT_DIR / f"e2a_stack_200m_{args.tag}_clean.nc"
        save_stack(clean, out_clean)
        print(f"wrote {out_clean}")

    RES_DIR.mkdir(exist_ok=True)
    sidecar = RES_DIR / f"e2a_demstack_{args.tag}_truth.json"
    sidecar.write_text(json.dumps(dict(
        pert=args.pert, t0_map_yr=args.t0, seed=args.seed,
        template=TEMPLATE, crop=dict(iy0=iy0, ix0=ix0, ny=CROP_NY, nx=CROP_NX),
        error_model=dict(tilt_dm_km=list(args.tilt_dm_km), noise_m=list(args.noise_m),
                         bias_sigma_m=args.bias_m, corrupt_frac=args.corrupt_frac,
                         corrupt_bias_m=args.corrupt_bias_m,
                         corr_rms_m=list(args.corr_rms_m), corr_beta=args.corr_beta,
                         corr_aniso=args.corr_aniso, corr_lmax_km=args.corr_lmax_km),
        strips=records), indent=2, default=float))
    print(f"wrote {sidecar}")
    truth.close()


if __name__ == "__main__":
    main()
