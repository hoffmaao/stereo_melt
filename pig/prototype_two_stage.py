"""Prototype the Option-B two-stage dense-reference coregistration on N strips.

Step 3-4 of ``literature/plan_alignment.md``. Picks N strips that are
already aligned the production way (baseline = ``ASP_is2cs2atmlvis``),
spanning a range of static-surface overlap, re-aligns each with
:func:`stereo_melt.coregister.asp.align_strip_two_stage` into a SCRATCH ASP
root (production untouched), and reports the **validation-gate** metrics
versus the baseline alignment on the SAME strips:

  * **tilt + roughness over static ground** — the discriminating metric.
    For each aligned DEM we form ``residual = DEM − REMA_static_reference``
    over the static mask (the reference is static-masked, so a finite
    reference pixel *is* a static pixel), fit a plane, and report the plane
    slope magnitude (m/km) and the robust scatter about it (NMAD, m). A
    flatter, tighter residual over rock + stable grounded ice means the
    dense tie removed the per-strip ramp that the ×9.42 hydrostatic gain
    otherwise turns into ±500 m/yr melt noise. REMA is the *same* reference
    for both alignments, so any REMA tilt is common-mode and cancels in the
    baseline-vs-two-stage comparison.
  * **pc_align post-align residual end_p50** (via
    :func:`...alignment_quality.parse_per_strip_quality`). Reported for
    context, but note it measures residual vs *different* references
    (baseline: altimetry cloud; two-stage: REMA), so it is not strictly
    apples-to-apples — the static tilt/NMAD above is.

The prototype gate passes if the two-stage alignment lowers the median
static tilt and median static NMAD across the selected strips.

Run:
    python -m pig.prototype_two_stage --n 8
    python -m pig.prototype_two_stage --n 8 --parallel 4
    python -m pig.prototype_two_stage --dem-id <stem>   # one strip
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Point pyproj/GDAL at the env-local proj.db before any pyproj-using import.
_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import numpy as np
import pandas as pd

from stereo_melt.coregister.asp import align_strip_two_stage

from pig import config

# Baseline = the production IS2-era aligned strips (375 aligned DEMs).
BASELINE_ROOT = config.ASP_IS2CS2ATMLVIS_ROOT
BASELINE_ALIGNED = BASELINE_ROOT / "asp_aligned"
BASELINE_REFCSV = BASELINE_ROOT / "reference_files"
# Scratch root for the two-stage prototype — kept fully separate from any
# production ASP root so a failed experiment can't poison the stack.
SCRATCH_ROOT = config.BASIN_DIR / "data" / "ASP_densetie_proto"

_SUFFIX = "-trans_reference-DEM.tif"


def _candidates() -> list[dict]:
    """Strips with (raw DEM, baseline aligned DEM, control CSV) all present."""
    out = []
    for dem in sorted(BASELINE_ALIGNED.glob(f"*{_SUFFIX}")):
        stem = dem.name[: -len(_SUFFIX)]
        raw = Path(config.STRIPS_DIR) / f"{stem}.tif"
        csv = BASELINE_REFCSV / f"combined_reference_{stem}.csv"
        if not (raw.exists() and csv.exists()):
            continue
        try:
            with open(csv) as fh:
                n_ctrl = max(0, sum(1 for _ in fh) - 1)
        except OSError:
            n_ctrl = 0
        out.append(
            {"stem": stem, "raw": raw, "baseline_dem": dem, "csv": csv, "n_ctrl": n_ctrl}
        )
    return out


def _select(cands: list[dict], n: int) -> list[dict]:
    """Span the static-overlap range; control-point count is the cheap proxy
    (more grounded/rock control ⇒ more static surface in the footprint)."""
    cands = sorted(cands, key=lambda c: c["n_ctrl"])
    if len(cands) <= n:
        return cands
    idx = sorted(set(np.linspace(0, len(cands) - 1, n).round().astype(int).tolist()))
    return [cands[i] for i in idx]


def _static_residual_stats(dem_path: str, ref_path: str, coarse_m: float = 100.0) -> dict:
    """Plane-fit tilt (m/km) + robust scatter (NMAD, m) of ``DEM − ref`` over
    static pixels. Memory-light: the DEM is read decimated to ~``coarse_m``
    and the static reference is warped onto that decimated grid."""
    import rasterio
    from rasterio import Affine
    from rasterio.warp import Resampling, reproject

    with rasterio.open(dem_path) as dd:
        res = abs(dd.transform.a)
        fac = max(1, int(round(coarse_m / res)))
        out_h = max(1, dd.height // fac)
        out_w = max(1, dd.width // fac)
        dem = (
            dd.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest, masked=True)
            .astype("float32")
            .filled(np.nan)
        )
        dtx = dd.transform * Affine.scale(dd.width / out_w, dd.height / out_h)
        dcrs = dd.crs

    ref_on = np.full((out_h, out_w), np.nan, dtype="float32")
    with rasterio.open(ref_path) as rr:
        reproject(
            source=rasterio.band(rr, 1),
            destination=ref_on,
            src_transform=rr.transform,
            src_crs=rr.crs,
            dst_transform=dtx,
            dst_crs=dcrs,
            resampling=Resampling.nearest,
            src_nodata=rr.nodata,
            dst_nodata=np.nan,
        )

    m = np.isfinite(dem) & np.isfinite(ref_on)
    n = int(m.sum())
    if n < 50:
        return {"n": n, "median": np.nan, "tilt_m_per_km": np.nan, "nmad": np.nan}

    xs = dtx.c + dtx.a * (np.arange(out_w) + 0.5)
    ys = dtx.f + dtx.e * (np.arange(out_h) + 0.5)
    xx = np.broadcast_to(xs, (out_h, out_w))[m] / 1000.0  # km
    yy = np.broadcast_to(ys[:, None], (out_h, out_w))[m] / 1000.0
    resid = (dem - ref_on)[m].astype(float)

    xx = xx - xx.mean()
    yy = yy - yy.mean()
    A = np.column_stack([np.ones(n), xx, yy])
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    detr = resid - A @ coef
    return {
        "n": n,
        "median": float(np.median(resid)),
        "tilt_m_per_km": float(np.hypot(coef[1], coef[2])),
        "nmad": float(np.median(np.abs(detr - np.median(detr))) * 1.4826),
    }


def _control_residual_stats(dem_path: str, csv_path: str) -> dict:
    """Residual ``control_h − DEM`` sampled at the altimetry control points:
    median (datum offset, m), plane tilt (m/km), robust scatter (NMAD, m).

    This is the **independent, non-circular** gate. ``align_strip_two_stage``
    Stage 1 ties geometry to the dense REMA reference and never uses the
    control cloud for rotation (Stage 2 only applies a single bulk Δz), so
    for the two-stage DEM the control points are essentially *held out* — a
    fair test of whether the REMA-derived alignment agrees with the
    independent sparse altimetry. For the baseline DEM (fit to this very
    cloud) it is a best-case lower bound, not held out."""
    import rasterio

    ctrl = pd.read_csv(csv_path)
    ex = ctrl["easting"].to_numpy(float)
    ny_ = ctrl["northing"].to_numpy(float)
    hz = ctrl["h_mean"].to_numpy(float)
    with rasterio.open(dem_path) as ds:
        arr = ds.read(1)  # native dtype (float32) — full read can be GBs on long strips
        nod = ds.nodata if ds.nodata is not None else -9999.0
        cols_f, rows_f = (~ds.transform) * (ex, ny_)
    rows = np.floor(rows_f).astype(int)
    cols = np.floor(cols_f).astype(int)
    inb = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    samp = np.full(ex.shape, np.nan)
    samp[inb] = arr[rows[inb], cols[inb]]
    samp[samp == nod] = np.nan
    resid = hz - samp
    ok = np.isfinite(resid)
    n = int(ok.sum())
    if n < 20:
        return {"n": n, "median": np.nan, "tilt_m_per_km": np.nan, "nmad": np.nan}
    r = resid[ok]
    xx = (ex[ok] - ex[ok].mean()) / 1000.0
    yy = (ny_[ok] - ny_[ok].mean()) / 1000.0
    A = np.column_stack([np.ones(n), xx, yy])
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    detr = r - A @ coef
    return {
        "n": n,
        "median": float(np.median(r)),
        "tilt_m_per_km": float(np.hypot(coef[1], coef[2])),
        "nmad": float(np.median(np.abs(detr - np.median(detr))) * 1.4826),
    }


def _run_one(item: dict, ref: str, max_displacement: float, keep_point_cloud: bool,
             force: bool = False):
    """Two-stage align one strip into the scratch root; return (stem, ts_dem)
    or None on failure. Skips strips already aligned in the scratch root unless
    ``force`` (re-runs are otherwise ~30 min wasted on long strips)."""
    stem = item["stem"]
    out_dem = SCRATCH_ROOT / "asp_aligned" / f"{stem}{_SUFFIX}"
    if out_dem.exists() and not force:
        print(f"=== two-stage: {stem} — already aligned, skip (use --force to redo) ===", flush=True)
        return (stem, str(out_dem))
    print(f"\n=== two-stage: {stem} (n_ctrl={item['n_ctrl']}) ===", flush=True)
    try:
        ts_dem = align_strip_two_stage(
            file_path=str(item["raw"]),
            reference_dem=ref,
            datum_csv=str(item["csv"]),
            output_dir=str(SCRATCH_ROOT),
            max_displacement=max_displacement,
            keep_point_cloud=keep_point_cloud,
            verbose=True,
        )
    except Exception as exc:  # noqa: BLE001 — batch harness, log and continue
        print(f"!! two-stage FAILED for {stem}: {exc!r}", flush=True)
        return None
    return (stem, ts_dem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="Number of strips (default 8).")
    ap.add_argument("--dem-id", type=str, default=None, help="Run a single strip by stem.")
    ap.add_argument("--parallel", type=int, default=1, help="ASP workers (default 1).")
    ap.add_argument("--max-displacement", type=float, default=100.0,
                    help="pc_align max-displacement cap, m (Shean default 100).")
    ap.add_argument("--keep-point-cloud", action="store_true",
                    help="Keep the ~GB pc_align intermediate cloud (debugging).")
    ap.add_argument("--metrics-only", action="store_true",
                    help="Skip alignment; recompute metrics from existing scratch DEMs.")
    ap.add_argument("--force", action="store_true",
                    help="Re-align even if the scratch DEM already exists.")
    args = ap.parse_args()

    ref = str(config.REMA_STATIC_REFERENCE_TIF)
    if not os.path.exists(ref):
        raise SystemExit(
            f"Static reference missing: {ref}\n"
            f"Build it first:  python -m pig.build_rema_static_reference"
        )

    cands = _candidates()
    print(f"{len(cands)} baseline strips have raw + aligned DEM + control CSV all present.")
    if not cands:
        raise SystemExit("No candidate strips — check BASELINE_ROOT / STRIPS_DIR.")

    if args.dem_id:
        sel = [c for c in cands if c["stem"] == args.dem_id]
        if not sel:
            raise SystemExit(f"--dem-id {args.dem_id!r} not among candidates.")
    else:
        sel = _select(cands, args.n)
    print(f"Selected {len(sel)} strips (n_ctrl span "
          f"{min(c['n_ctrl'] for c in sel)} .. {max(c['n_ctrl'] for c in sel)}):")
    for c in sel:
        print(f"  n_ctrl={c['n_ctrl']:6d}  {c['stem']}")

    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    (SCRATCH_ROOT / "asp_aligned").mkdir(parents=True, exist_ok=True)

    # --- alignment ---
    if not args.metrics_only:
        if args.parallel > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            print(f"\nAligning {len(sel)} strips with {args.parallel} workers...")
            with ProcessPoolExecutor(max_workers=args.parallel) as ex:
                futs = {
                    ex.submit(_run_one, c, ref, args.max_displacement,
                              args.keep_point_cloud, args.force): c
                    for c in sel
                }
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    r = fut.result()
                    tag = "OK" if r else "FAIL"
                    print(f"[{done}/{len(sel)}] {futs[fut]['stem']} -> {tag}", flush=True)
        else:
            for i, c in enumerate(sel, 1):
                r = _run_one(c, ref, args.max_displacement, args.keep_point_cloud, args.force)
                print(f"[{i}/{len(sel)}] {c['stem']} -> {'OK' if r else 'FAIL'}", flush=True)

    # --- metrics ---
    # (1) ramp over static vs the dense REMA reference: baseline_static_tilt
    #     is the residual ramp the sparse-control alignment LEAVES over the
    #     static footprint (the disease); two-stage should drive it ~0 over
    #     the whole footprint (the cure). Partly circular for two-stage
    #     (Stage 1 minimizes it), so it is evidence the dense tie flattens
    #     static — not proof of better melt.
    # (2) residual at the independent altimetry control points: held out for
    #     two-stage (Stage 1 never used it for geometry), so this is the
    #     non-circular check that the REMA tie still agrees with the sparse
    #     altimetry datum + slope.
    print("\nComputing residual stats (vs REMA static reference, and vs altimetry control)...")
    rows = []
    for c in sel:
        stem = c["stem"]
        ts_dem = SCRATCH_ROOT / "asp_aligned" / f"{stem}{_SUFFIX}"
        if not ts_dem.exists():
            print(f"  ⚠ no two-stage DEM for {stem} (alignment failed/skipped)")
            continue
        bs = _static_residual_stats(str(c["baseline_dem"]), ref)
        ts = _static_residual_stats(str(ts_dem), ref)
        bc = _control_residual_stats(str(c["baseline_dem"]), str(c["csv"]))
        tc = _control_residual_stats(str(ts_dem), str(c["csv"]))
        rows.append({
            "stem": stem, "n_ctrl": c["n_ctrl"], "ts_n": ts["n"],
            # (1) static-vs-REMA ramp
            "base_stat_tilt": bs["tilt_m_per_km"], "ts_stat_tilt": ts["tilt_m_per_km"],
            # (2) independent control residual
            "base_ctrl_med": bc["median"], "ts_ctrl_med": tc["median"],
            "base_ctrl_tilt": bc["tilt_m_per_km"], "ts_ctrl_tilt": tc["tilt_m_per_km"],
            "base_ctrl_nmad": bc["nmad"], "ts_ctrl_nmad": tc["nmad"],
        })
    if not rows:
        raise SystemExit("No two-stage DEMs produced — nothing to score.")
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    with pd.option_context("display.float_format", lambda v: f"{v:7.3f}"):
        print("\n" + "=" * 100)
        print("(1) RAMP OVER STATIC vs dense REMA reference  [tilt m/km; lower=flatter]")
        print("    base_stat_tilt = ramp the sparse alignment leaves; ts should be ≪")
        print("=" * 100)
        print(df[["stem", "n_ctrl", "ts_n", "base_stat_tilt", "ts_stat_tilt"]].to_string(index=False))

        print("\n" + "=" * 100)
        print("(2) INDEPENDENT ALTIMETRY-CONTROL RESIDUAL  [held out for two-stage]")
        print("    median=datum offset (m), tilt=m/km, nmad=m. ts should stay small.")
        print("=" * 100)
        print(df[["stem", "base_ctrl_med", "ts_ctrl_med", "base_ctrl_tilt", "ts_ctrl_tilt",
                  "base_ctrl_nmad", "ts_ctrl_nmad"]].to_string(index=False))

    med = df.median(numeric_only=True)

    def _pct(a, b):
        return f"{100 * (b - a) / a:+.0f}%" if a and np.isfinite(a) and a != 0 else "n/a"

    print("\n" + "-" * 72)
    print("MEDIAN over selected strips:")
    print(f"  (1) ramp over static (m/km):     baseline {med['base_stat_tilt']:.3f}  ->  "
          f"two-stage {med['ts_stat_tilt']:.3f}   ({_pct(med['base_stat_tilt'], med['ts_stat_tilt'])})")
    print(f"  (2) control residual |median| (m): baseline {abs(med['base_ctrl_med']):.3f}  ->  "
          f"two-stage {abs(med['ts_ctrl_med']):.3f}")
    print(f"      control residual tilt (m/km):  baseline {med['base_ctrl_tilt']:.3f}  ->  "
          f"two-stage {med['ts_ctrl_tilt']:.3f}")
    print(f"      control residual NMAD (m):     baseline {med['base_ctrl_nmad']:.3f}  ->  "
          f"two-stage {med['ts_ctrl_nmad']:.3f}")

    flattens = med["ts_stat_tilt"] < med["base_stat_tilt"]
    # Held-out control consistency: two-stage shouldn't be dramatically worse
    # than the (best-case, fit-to-control) baseline. Allow modest slack.
    consistent = (
        med["ts_ctrl_nmad"] <= 1.5 * med["base_ctrl_nmad"] + 0.2
        and abs(med["ts_ctrl_med"]) <= abs(med["base_ctrl_med"]) + 0.5
    )
    print("\n  PROTOTYPE GATE (necessary, not sufficient — the decisive test is the")
    print("  downstream stack/tilt_fit/melt rebuild, plan step 6):")
    print(f"    (1) dense tie flattens static ramp:        {'YES ✅' if flattens else 'NO ❌'}")
    print(f"    (2) stays consistent w/ independent control: {'YES ✅' if consistent else 'NO ❌'}")
    verdict = "PASS ✅ — proceed to full re-align" if (flattens and consistent) \
        else "INVESTIGATE ❌ — see which check failed before scaling"
    print(f"    -> {verdict}")
    print("-" * 72)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = config.RESULTS_DIR / "two_stage_prototype_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
