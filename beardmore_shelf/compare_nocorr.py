"""A/B metrics: P3 baseline (164 ep) vs Shean-style nocorr stack (~265 ep).

Compares the production path-integrated Lagrangian products (run_melt_path
--tag nocorr vs the P3 baseline file) AND an in-process Eulerian solve on
both tilt-corrected stacks, so the melt medians are reported in BOTH
reference frames (user directive 2026-07-10). Readouts follow the
established gates (GATE-2 / solver-suite): floating median/IQR, GL-2km
median, flux CLIP/raw/robust, RED_accr / LEFT_melt / REF_floating band
median+MAD+cell counts, per-cell epoch/pair-count improvement in bands.

Run (after the nocorr chain's tilt + path solves):

    cd /wd2/projects/stereo_melt
    BEARDMORE_SHELF_SOURCES=nocorr \
    /home/hoffmaao/miniconda3/envs/stereo_melt/bin/python -u \
        -m beardmore_shelf.compare_nocorr \
        > beardmore_shelf/logs/compare_nocorr.log 2>&1
"""
from __future__ import annotations

import stereo_melt.envsetup  # noqa: F401

import os

import numpy as np
import xarray as xr

from stereo_melt.flux import grounding_buffer, integrate_basal_flux
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.melt import eulerian_melt_rate

from beardmore_shelf import config
from beardmore_shelf.diag_stripe_strips import BANDS
from beardmore_shelf.run_melt import (
    load_floating_mask,
    load_smb_on_grid,
    load_velocity_on_grid,
)
from beardmore_shelf.run_melt_path import load_grounded_mask, load_stack_res

WINDOW = f"{config.START_TIME}_{config.END_TIME}"


def band_stats(arr2d: np.ndarray, flo: np.ndarray) -> dict[str, tuple]:
    out = {}
    for name, (c0, c1, r0, r1) in BANDS.items():
        m = np.zeros_like(flo)
        m[r0:r1, c0:c1] = True
        m &= flo
        v = arr2d[m]
        v = v[np.isfinite(v)]
        med = float(np.median(v)) if v.size else np.nan
        mad = float(1.4826 * np.median(np.abs(v - med))) if v.size else np.nan
        out[name] = (med, mad, int(v.size))
    v = arr2d[flo]
    v = v[np.isfinite(v)]
    med = float(np.median(v)) if v.size else np.nan
    out["REF_floating"] = (
        med, float(1.4826 * np.median(np.abs(v - med))) if v.size else np.nan,
        int(v.size),
    )
    return out


def melt_metrics(mr: np.ndarray, count: np.ndarray, flo: np.ndarray,
                 dom: np.ndarray, res_m: float) -> dict:
    v = mr[flo]
    v = v[np.isfinite(v)]
    q = np.nan_to_num(count, nan=0) >= 10
    b = integrate_basal_flux(np.where(flo, mr, np.nan), dom, res_m, quality=q)
    return dict(
        median=float(np.median(v)), q25=float(np.percentile(v, 25)),
        q75=float(np.percentile(v, 75)), cells=int(v.size),
        gl2_med=b["median_myr"], flux_clip=b["gt_clip"],
        flux_raw=b["gt_raw"], flux_robust=b["gt_robust"],
        area_km2=b["area_km2"],
    )


def report(label: str, m: dict, bands: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"  floating median {m['median']:+.2f}  IQR [{m['q25']:+.2f}, {m['q75']:+.2f}]"
          f"  cells {m['cells']:,}")
    print(f"  GL-2km med {m['gl2_med']:+.2f}  flux Gt/yr CLIP {m['flux_clip']:+.2f}"
          f"  raw {m['flux_raw']:+.2f}  robust {m['flux_robust']:+.2f}"
          f"  area {m['area_km2']:.0f} km2")
    for name, (med, mad, n) in bands.items():
        print(f"  {name:13s} med {med:+.2f}  MAD {mad:.2f}  n {n:,}")


def main() -> None:
    # Tag overrides so the same two-frame A/B harness serves other
    # tilt-variant comparisons (e.g. the smoothab trans-only regression, or
    # the fullrec IS2-era-slice gate: TAG_A=smoothab TAG_B=fullrec under the
    # default window — load_basin_stack windows wider stacks since 07-12).
    tag_a = os.environ.get("BEARDMORE_SHELF_COMPARE_TAG_A", "").strip() or None
    label_a = tag_a or "base"
    tag_b = os.environ.get("BEARDMORE_SHELF_COMPARE_TAG_B", "nocorr")
    print(f"=== A/B: '{label_a}' vs '{tag_b}' stack ===")
    results = {}
    for tag, label in ((tag_a, label_a), (tag_b, tag_b)):
        stack = load_stack_res(config.RES, tag=tag)
        floating = load_floating_mask(stack)
        grounded = load_grounded_mask(stack)
        flo = np.asarray(floating.values, bool)
        res_m = abs(float(stack["x"].values[1] - stack["x"].values[0]))
        dom = grounding_buffer(flo, np.asarray(grounded.values, bool), 2000.0, res_m)
        n_ep = stack.sizes["time"]
        print(f"\n=== stack[{label}]: {n_ep} epochs ===")

        # per-cell epoch count over floating (conditioning readout)
        cnt = np.isfinite(np.asarray(stack.values)).sum(axis=0).astype(float)
        cnt_f = cnt[flo]
        print(f"  per-cell epoch count over floating: med {np.median(cnt_f):.0f}  "
              f"p10 {np.percentile(cnt_f, 10):.0f}  p90 {np.percentile(cnt_f, 90):.0f}")
        for name, (med, mad, n) in band_stats(cnt, flo).items():
            print(f"  epochs {name:13s} med {med:.0f}")

        # Lagrangian path product (run_melt_path output)
        suffix = f"{config.RES:.0f}m" if isinstance(config.RES, float) else f"{config.RES}m"
        ptag = suffix + (f"_{tag}" if tag else "")
        path_nc = config.PROCESSED_DIR / f"beardmore_shelf_melt_path_{ptag}_{WINDOW}.nc"
        pm = pb = None
        if path_nc.exists():
            ds = xr.open_dataset(path_nc)
            mr = np.asarray(ds["melt_rate_lagrangian"].values, float)
            pcount = np.asarray(ds["lagrangian_count"].values, float)
            pm = melt_metrics(mr, pcount, flo, dom, res_m)
            pb = band_stats(mr, flo)
            report(f"PATH (Lagrangian) [{label}]", pm, pb)
        else:
            print(f"  !! path product missing: {path_nc}")

        # Eulerian solve in-process (both frames directive)
        vx, vy, vel_source = load_velocity_on_grid(stack)
        a_dot = load_smb_on_grid(stack)
        firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)
        print(f"  running eulerian_melt_rate (velocity: {vel_source})...")
        eul = eulerian_melt_rate(stack, vx, vy, a_dot=a_dot, d=firn)
        emr = np.asarray(eul["melt_rate"].where(floating).values, float)
        ecount = np.asarray(eul["count"].values, float)
        em = melt_metrics(emr, ecount, flo, dom, res_m)
        eb = band_stats(emr, flo)
        report(f"EULERIAN [{label}]", em, eb)

        results[label] = dict(n_ep=n_ep, path=pm, path_bands=pb,
                              eul=em, eul_bands=eb)

    # Delta summary
    if label_a in results and tag_b in results and label_a != tag_b:
        b, n = results[label_a], results[tag_b]
        print(f"\n=== DELTA ({tag_b} - {label_a}) ===")
        print(f"  epochs: {b['n_ep']} -> {n['n_ep']}  (+{n['n_ep'] - b['n_ep']})")
        for frame in ("path", "eul"):
            if b[frame] is None or n[frame] is None:
                continue
            print(f"  [{frame}] median {b[frame]['median']:+.2f} -> {n[frame]['median']:+.2f}"
                  f"   flux CLIP {b[frame]['flux_clip']:+.2f} -> {n[frame]['flux_clip']:+.2f}"
                  f"   cells {b[frame]['cells']:,} -> {n[frame]['cells']:,}")
            for band in ("RED_accr", "LEFT_melt", "REF_floating"):
                bm, bmad, _ = b[f"{frame}_bands"][band]
                nm, nmad, _ = n[f"{frame}_bands"][band]
                print(f"    {band:13s} med {bm:+.2f} -> {nm:+.2f}   MAD {bmad:.2f} -> {nmad:.2f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
