"""Score Pine Island epochs by post-tilt-fit static-control residual + IRLS weight.

Thin wrapper over :mod:`stereo_melt.coregister.tilt_qc`. Run::

    python -m pig.scripts.find_bad_epochs            # 25 m production stack
    python -m pig.scripts.find_bad_epochs --res 250  # resolution-pilot stack

The ``--res`` switch mirrors :mod:`pig.tilt_fit` / :mod:`pig.run_melt`: it
selects the ``pig_stack_<N>m`` tilt-corrected stack and the matching
``pig_tilt_params_<N>m_*`` parameters, so the screen runs on whichever
stack the downstream solvers actually consume. Without it the script
hardcoded the no-suffix ``pig_stack`` prefix and so could not see the
fused ``--res 250`` stack at all.
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import xarray as xr

from stereo_melt.coregister.alignment_quality import aggregate_basin_quality
from stereo_melt.coregister.tilt_qc import (
    print_bad_epoch_report,
    score_tilt_residuals,
    suggest_bad_epochs,
)
from stereo_melt.stack import load_basin_stack

import numpy as np

from stereo_melt.coregister.tilt import build_static_area_polygon_mask

from pig import config


def main(res_override: float | None = None, tag: str | None = None,
         end_p50_threshold_m: float | None = 10.0) -> None:
    if res_override is not None:
        stack_prefix = f"pig_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "pig_stack"
        out_suffix = ""
    if tag:
        stack_prefix += f"_{tag}"
        out_suffix += f"_{tag}"

    tc, src_path = load_basin_stack(
        config.PROCESSED_DIR, stack_prefix,
        config.START_TIME, config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    print(f"  scoring {src_path.name}  ({tc.sizes['time']} epochs after BAD_EPOCHS drop)")
    static = build_static_area_polygon_mask(tc, config.BEDMACHINE_NC).values.astype(bool)

    params = xr.open_dataset(
        config.PROCESSED_DIR
        / f"pig_tilt_params{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )

    # P4: join per-strip pc_align quality (end_p50) so the screen can drop
    # geometry-failed strips the tilt-residual gate alone misses.
    aq = aggregate_basin_quality(config.STRIP_SOURCES)
    if not aq.empty:
        ep = aq["end_p50"].to_numpy(dtype=float)
        ep = ep[np.isfinite(ep)]
        if ep.size:
            pcts = np.percentile(ep, [50, 75, 90, 95, 99])
            print(f"  end_p50 over {ep.size} aligned strips (m): "
                  f"p50={pcts[0]:.2f} p75={pcts[1]:.2f} p90={pcts[2]:.2f} "
                  f"p95={pcts[3]:.2f} p99={pcts[4]:.2f} max={ep.max():.2f}")
    else:
        print("  (no pc_align end-error CSVs found; end_p50 gate inert)")

    df = score_tilt_residuals(tc, params, static, alignment_quality=aq if not aq.empty else None)
    suggested = suggest_bad_epochs(df, end_p50_threshold_m=end_p50_threshold_m)
    print_bad_epoch_report(df, suggested, end_p50_threshold_m=end_p50_threshold_m)

    # Emit the drop list across ALL gates (end_p50 + catastrophic + IRLS-fail),
    # not just the end_p50 subset. With a dem_id coord (strip-addressable
    # stack) emit per-strip BAD_STRIPS -- the Shean-faithful unit that spares
    # clean same-day siblings; on legacy date-only stacks fall back to
    # date-keyed BAD_EPOCHS and warn about the same-day over-drop.
    if "dem_id" in suggested.columns:
        strip_ids = sorted(suggested["dem_id"].astype(str).unique())
        print(f"\n  {len(strip_ids)} strip(s) flagged; BAD_STRIPS (copy into config):")
        print("BAD_STRIPS: tuple[str, ...] = (")
        for s in strip_ids:
            print(f'    "{s}",')
        print(")")
        print("  ↳ per-DEM rejection (Shean): paste into config.BAD_STRIPS "
              "and EMPTY config.BAD_EPOCHS -- on a strip-addressable stack PIG "
              "rejects strips by dem_id, never by date (date-keying over-drops "
              "clean same-day siblings).")
    else:
        dates = sorted({str(e)[:10] for e in suggested["epoch"]})
        print(f"\n  {len(dates)} date(s) flagged; BAD_EPOCHS additions (copy into config):")
        for d in dates:
            print(f'    "{d}",')
        print("  ⚠️ stack carries no dem_id coord -- this date-keyed list will "
              "also drop clean same-day siblings. Rebuild with `pig.build_stack` "
              "to enable per-strip BAD_STRIPS.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help=(
            "Resolution variant to score (meters). When set, loads "
            "pig_stack_<N>m_tilt_corrected_<start>_<end>.nc + "
            "pig_tilt_params_<N>m_<start>_<end>.nc."
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Experiment tag matching the `pig.build_stack --tag` build: "
            "loads pig_stack_<N>m_<tag>_tilt_corrected_*.nc + "
            "pig_tilt_params_<N>m_<tag>_*.nc."
        ),
    )
    parser.add_argument(
        "--end-p50",
        type=float,
        default=10.0,
        help=(
            "pc_align-quality (Shean coregistration) gate, ON by default: flag "
            "epochs whose post-align residual median end_p50 exceeds this (m). "
            "10 m sits well above PIG's clean population (median ~0.3 m, CS2-era "
            "~0.6 m) and above the 3 m offset-only demote band, so only genuine "
            "pc_align divergence is dropped. Pass a large value to effectively "
            "disable; combine with the |med_resid|>50 m catastrophic backstop."
        ),
    )
    args = parser.parse_args()
    main(res_override=args.res, tag=args.tag, end_p50_threshold_m=args.end_p50)
