"""Score Venable epochs by post-tilt-fit static-control residual + IRLS weight.

Thin wrapper over :mod:`stereo_melt.coregister.tilt_qc`. Run::

    python -m venable.scripts.find_bad_epochs
"""
from __future__ import annotations

import os
import sys

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ["PROJ_DATA"] = _env_proj
    os.environ["PROJ_LIB"] = _env_proj

import xarray as xr

from stereo_melt.coregister.tilt_qc import (
    print_bad_epoch_report,
    score_tilt_residuals,
    suggest_bad_epochs,
)
from stereo_melt.stack import load_basin_stack

from stereo_melt.coregister.tilt import build_static_area_polygon_mask

from venable import config


def main(res_override: float | None = None) -> None:
    if res_override is not None:
        stack_prefix = f"venable_stack_{int(round(res_override))}m"
        out_suffix = f"_{int(round(res_override))}m"
    else:
        stack_prefix = "venable_stack"
        out_suffix = ""

    tc, _ = load_basin_stack(
        config.PROCESSED_DIR, stack_prefix,
        config.START_TIME, config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    static = build_static_area_polygon_mask(tc, config.BEDMACHINE_NC).values.astype(bool)

    params = xr.open_dataset(
        config.PROCESSED_DIR
        / f"venable_tilt_params{out_suffix}_{config.START_TIME}_{config.END_TIME}.nc"
    )

    df = score_tilt_residuals(tc, params, static)
    # Legacy coverage-coupled gate pinned (resid_threshold_m=0.5) so this
    # basin's screen is unchanged. Migrate to the Shean coregistration
    # end_p50 gate (join aggregate_basin_quality like pig/beardmore) when
    # re-screening — see project_tilt_qc_shared.
    suggested = suggest_bad_epochs(df, resid_threshold_m=0.5)
    print_bad_epoch_report(df, suggested, resid_threshold_m=0.5)

    # Emit the drop list. On a strip-addressable stack (dem_id coord present)
    # emit per-strip BAD_STRIPS -- the Shean-faithful unit that spares clean
    # same-day siblings; on legacy date-only stacks fall back to date-keyed
    # BAD_EPOCHS and warn about the same-day over-drop.
    if "dem_id" in suggested.columns:
        strip_ids = sorted(suggested["dem_id"].astype(str).unique())
        print(f"\n  {len(strip_ids)} strip(s) flagged; BAD_STRIPS (copy into config):")
        print("BAD_STRIPS: tuple[str, ...] = (")
        for s in strip_ids:
            print(f'    "{s}",')
        print(")")
        print("  per-DEM rejection (Shean): paste into config.BAD_STRIPS and "
              "EMPTY config.BAD_EPOCHS -- on a strip-addressable stack reject "
              "strips by dem_id, never by date (date-keying over-drops clean "
              "same-day siblings).")
    else:
        dates = sorted({str(e)[:10] for e in suggested["epoch"]})
        print(f"\n  {len(dates)} date(s) flagged; BAD_EPOCHS additions (copy into config):")
        for d in dates:
            print(f'    "{d}",')
        print("  WARNING: stack carries no dem_id coord -- this date-keyed list "
              "will also drop clean same-day siblings. Rebuild with build_stack "
              "to enable per-strip BAD_STRIPS.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res",
        type=float,
        default=None,
        help="Resolution variant (meters) matching a `--res` build/tilt run.",
    )
    args = parser.parse_args()
    main(res_override=args.res)
