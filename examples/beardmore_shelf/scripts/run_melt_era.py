"""Era-resolved production melt on a wide (full-record) stack slice.

P3 of the full-record program (project_fullrecord_program_2026_07_11).
Since 2026-07-12 ``load_basin_stack`` windows a fallback-loaded wider stack
to the requested ``[start, end)``, so env-windowed ``run_melt_path`` runs
are also era-safe; this driver is the explicit era harness on top of that:
it loads the wide tilt-corrected stack ONCE by its exact window (no glob
fallback), slices epochs to ``[--start, --end)`` in memory, and runs BOTH
production frames on the slice with compare_nocorr-style readouts:

- Lagrangian path solver, exact ``run_melt_path`` production config
  (seed_stride=1, pair_median, pairs=all, 1.5-2.5 yr, dt=0.05);
- Eulerian dh/dt-budget solve (the ``compare_nocorr`` in-process form).

SMB is re-integrated over the ERA window (``run_melt.load_smb_on_grid``
windows on config.START/END, which would put the IS2-era mean rate under
every era). Readouts mirror ``compare_nocorr``: floating median/IQR, GL-2km
flux CLIP/raw/robust, RED/LEFT/REF band med+MAD.

Examples (after FULLREC_CHAIN_DONE):

    cd /wd2/projects/stereo_melt/examples
    $PY -u -m beardmore_shelf.scripts.run_melt_era \
        --start 2019-01-01 --end 2024-01-10 --label is2era
    $PY -u -m beardmore_shelf.scripts.run_melt_era \
        --start 2009-01-01 --end 2013-01-01 --label era0913
"""
from __future__ import annotations

import stereo_melt.envsetup  # noqa: F401  (PROJ_DATA before any pyproj import)

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from stereo_melt.colormaps import add_melt_colorbar, melt_cmap, melt_norm
from stereo_melt.flux import grounding_buffer
from stereo_melt.io.bedmachine import load_firn_on_grid
from stereo_melt.io.smb import smb_over_window
from stereo_melt.melt import eulerian_melt_rate, lagrangian_melt_rate
from stereo_melt.stack import load_basin_stack

from beardmore_shelf import config
from beardmore_shelf.compare_nocorr import band_stats, melt_metrics, report
from beardmore_shelf.run_melt import (
    SECONDS_PER_YEAR,
    load_floating_mask,
    load_velocity_on_grid,
)
from beardmore_shelf.run_melt_path import load_grounded_mask


def era_smb_on_grid(stack: xr.DataArray, start: str, end: str) -> xr.DataArray:
    """RACMO SMB mean rate over the ERA window on the stack grid."""
    m_ice_cumulative = smb_over_window(
        str(config.RACMO_SMB_NC),
        stack["x"].values,
        stack["y"].values,
        start=start,
        end=end,
        method="linear",
    )
    dt_years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / SECONDS_PER_YEAR
    return xr.DataArray(
        m_ice_cumulative / dt_years,
        dims=("y", "x"),
        coords={"y": stack["y"].values, "x": stack["x"].values},
        name="a_dot",
        attrs={
            "units": "m ice yr^-1",
            "integration_window": f"{start} to {end}",
            "source": "RACMO2.4p1 smbgl (Zenodo 19255213)",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res", type=int, default=config.RES)
    p.add_argument("--tag", type=str, default="fullrec",
                   help="stack variant tag (default fullrec)")
    p.add_argument("--stack-start", type=str, default="2009-01-01",
                   help="window in the STACK FILENAME to load")
    p.add_argument("--stack-end", type=str, default="2024-01-10")
    p.add_argument("--start", type=str, required=True,
                   help="era slice start (inclusive)")
    p.add_argument("--end", type=str, required=True,
                   help="era slice end (exclusive)")
    p.add_argument("--label", type=str, required=True,
                   help="era label carried into output names (e.g. is2era)")
    p.add_argument("--dry-run", action="store_true",
                   help="load + slice + report epoch counts, no solve")
    p.add_argument("--skip-path", action="store_true")
    p.add_argument("--skip-eulerian", action="store_true")
    args = p.parse_args()
    t0 = time.time()

    prefix = (
        "beardmore_shelf_stack"
        if args.res == config.RES
        else f"beardmore_shelf_stack_{args.res}m"
    )
    if args.tag:
        prefix += f"_{args.tag}"
    stack, src_path = load_basin_stack(
        config.PROCESSED_DIR,
        prefix,
        args.stack_start,
        args.stack_end,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    if "tilt_corrected" not in src_path.name:
        print("  !! era slice on a RAW (un-tilt-corrected) stack — QC use only")

    t = pd.to_datetime(stack["time"].values)
    keep = (t >= pd.Timestamp(args.start)) & (t < pd.Timestamp(args.end))
    n_era = int(keep.sum())
    print(f"era [{args.start}, {args.end}) '{args.label}': "
          f"{n_era}/{stack.sizes['time']} epochs")
    if n_era == 0:
        raise SystemExit("no epochs in the era window")
    stack = stack.isel(time=np.where(keep)[0])
    if "source_variant" in stack.coords:
        vals, cnts = np.unique(
            np.asarray(stack["source_variant"].values, dtype=str), return_counts=True
        )
        print(f"  source_variant: {dict(zip(vals.tolist(), cnts.tolist()))}")
    span_yr = (t[keep].max() - t[keep].min()).total_seconds() / SECONDS_PER_YEAR
    print(f"  epoch span: {t[keep].min().date()} -> {t[keep].max().date()} "
          f"({span_yr:.2f} yr)")
    if args.dry_run:
        print("dry run: stopping before masks/solve")
        return

    floating = load_floating_mask(stack)
    grounded = load_grounded_mask(stack)
    flo = np.asarray(floating.values, bool)
    res_m = abs(float(stack["x"].values[1] - stack["x"].values[0]))
    dom = grounding_buffer(flo, np.asarray(grounded.values, bool), 2000.0, res_m)

    cnt = np.isfinite(np.asarray(stack.values)).sum(axis=0).astype(float)
    cnt_f = cnt[flo]
    print(f"  per-cell epoch count over floating: med {np.median(cnt_f):.0f}  "
          f"p10 {np.percentile(cnt_f, 10):.0f}  p90 {np.percentile(cnt_f, 90):.0f}")

    vx, vy, vel_source = load_velocity_on_grid(stack)
    a_dot = era_smb_on_grid(stack, args.start, args.end)
    firn = load_firn_on_grid(stack, config.BEDMACHINE_NC)

    data_vars: dict[str, xr.DataArray] = {"floating_mask": floating}
    attrs = {
        "era_label": args.label,
        "era_window": f"{args.start} to {args.end} (end-exclusive)",
        "parent_stack": src_path.name,
        "n_epochs": n_era,
        "velocity_source": vel_source + " (static)",
        "units": "m ice yr^-1; Shean convention: negative = melt",
    }

    if not args.skip_path:
        print("PATH frame: production Lagrangian path solver (narrates)...")
        lagr = lagrangian_melt_rate(
            stack.where(floating), vx, vy, a_dot=a_dot, d=firn, dt_yr=0.05,
            seed_stride=1, output="path", aggregator="pair_median",
            pairs="all", min_dt_yr=1.5, max_dt_yr=2.5,
        )
        mr_p = lagr.melt_rate.where(floating)
        arr_p = np.asarray(mr_p.values, float)
        pm = melt_metrics(arr_p, np.asarray(lagr["count"].values, float),
                          flo, dom, res_m)
        report(f"PATH (Lagrangian) [{args.label}]", pm, band_stats(arr_p, flo))
        data_vars["melt_rate_lagrangian"] = mr_p
        data_vars["lagrangian_count"] = lagr["count"]
        data_vars["lagrangian_rmse"] = lagr["rmse"]
        attrs["path_solver"] = (
            "lagrangian_melt_rate output=path seed_stride=1 pair_median "
            "pairs=all 1.5-2.5yr dt=0.05 (production config)"
        )
        attrs["path_flux_clip_gt_yr"] = float(pm["flux_clip"])

    if not args.skip_eulerian:
        print("EULERIAN frame: eulerian_melt_rate (compare_nocorr form)...")
        eul = eulerian_melt_rate(stack, vx, vy, a_dot=a_dot, d=firn)
        mr_e = eul["melt_rate"].where(floating)
        arr_e = np.asarray(mr_e.values, float)
        em = melt_metrics(arr_e, np.asarray(eul["count"].values, float),
                          flo, dom, res_m)
        report(f"EULERIAN [{args.label}]", em, band_stats(arr_e, flo))
        data_vars["melt_rate_eulerian"] = mr_e
        data_vars["eulerian_count"] = eul["count"]
        attrs["eulerian_flux_clip_gt_yr"] = float(em["flux_clip"])

    tag_out = f"{args.res}m_{args.tag}_{args.label}" if args.tag else f"{args.res}m_{args.label}"
    out = xr.Dataset(data_vars, attrs=attrs)
    out_nc = (
        config.PROCESSED_DIR
        / f"beardmore_shelf_melt_era_{tag_out}_{args.start}_{args.end}.nc"
    )
    comp = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_nc, encoding=comp)
    print(f"Saved -> {out_nc}")

    panels = [k for k in ("melt_rate_lagrangian", "melt_rate_eulerian")
              if k in data_vars]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.8),
                                 constrained_layout=True, squeeze=False)
        for ax, key in zip(axes[0], panels):
            im = ax.imshow(data_vars[key].values, cmap=melt_cmap(),
                           norm=melt_norm(vmax=10.0),
                           interpolation="nearest")
            ax.set_title(f"{key} [{args.label}] (m ice/yr)", fontsize=10)
            add_melt_colorbar(fig, im, ax=ax, shrink=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
        out_png = config.FIGURES_DIR / f"melt_era_{tag_out}.png"
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"  wrote {out_png}")
    print(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
