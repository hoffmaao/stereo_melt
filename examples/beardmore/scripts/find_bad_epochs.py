"""Score Beardmore epochs by post-tilt-fit static-control residual + IRLS weight.

Thin wrapper over :mod:`stereo_melt.coregister.tilt_qc`. Run::

    python -m beardmore.scripts.find_bad_epochs
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
from stereo_melt.coregister.tilt import build_static_area_polygon_mask
from stereo_melt.coregister.tilt_qc import (
    print_bad_epoch_report,
    score_tilt_residuals,
    suggest_bad_epochs,
)
from stereo_melt.stack import load_basin_stack

from beardmore import config


# Gate strips whose pc_align Euclidean residual (p50 of end_errors.csv)
# exceeds this magnitude. Strips with a p50 > END_P50_THRESHOLD_M are
# flagged regardless of the per-epoch tilt-LSQ residual gates, so
# pc_align-degenerate or distortion-rich strips don't survive into the
# stack. Tuned conservatively: Beardmore's IS2+CS2 era is at ~1 m, the
# CS2-only era at ~4 m, so 6.0 m catches gross failures without
# rejecting clean CS2-only strips.
END_P50_THRESHOLD_M: float = 6.0


def main() -> None:
    tc, _ = load_basin_stack(
        config.PROCESSED_DIR, "beardmore_stack",
        config.START_TIME, config.END_TIME,
        prefer_tilt_corrected=True,
        bad_epochs=tuple(getattr(config, "BAD_EPOCHS", ())),
        bad_strips=tuple(getattr(config, "BAD_STRIPS", ())),
    )
    static = build_static_area_polygon_mask(tc, config.BEDMACHINE_NC).values.astype(bool)

    params = xr.open_dataset(
        config.PROCESSED_DIR
        / f"beardmore_tilt_params_{config.START_TIME}_{config.END_TIME}.nc"
    )

    # Pull per-strip pc_align quality (NED + beg/end percentiles) from
    # the basin's STRIP_SOURCES dirs and cache to CSV alongside results.
    quality = aggregate_basin_quality(config.STRIP_SOURCES)
    if not quality.empty:
        quality_csv = config.RESULTS_DIR / "alignment_quality.csv"
        quality.to_csv(quality_csv, index=False)
        print(f"alignment_quality: {len(quality)} strips, "
              f"end_p50 median={quality['end_p50'].median():.2f} m, "
              f"max={quality['end_p50'].max():.2f} m  ({quality_csv})")
    else:
        print("alignment_quality: (no aligned strips found)")

    df = score_tilt_residuals(tc, params, static, alignment_quality=quality)
    # end_p50=6 m coregistration gate + legacy coverage gate pinned
    # (resid_threshold_m=0.5) so Beardmore's screen is unchanged. At ~12%
    # static control the frac_static clause is still meaningful here, unlike
    # PIG; revisit when migrating Beardmore fully to the coregistration gate.
    suggested = suggest_bad_epochs(
        df, end_p50_threshold_m=END_P50_THRESHOLD_M, resid_threshold_m=0.5,
    )
    print_bad_epoch_report(
        df, suggested, end_p50_threshold_m=END_P50_THRESHOLD_M, resid_threshold_m=0.5,
    )

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
    main()
