"""Retro-scan existing PIG alignments and quarantine catastrophic ones.

For every aligned strip in ``<asp_root>/asp_aligned/`` whose pc_align log
reports a translation magnitude exceeding ``--max-displacement`` (Shean
2019's pc_align cap, default 100 m), move all of that strip's outputs
into ``<asp_root>/bad_align/<dem_id>/`` and drop a ``.bad_align`` sentinel
so subsequent runs skip the strip.

This is a one-shot cleanup for alignments that ran before the in-library
sanity gate was added (see
:func:`stereo_melt.coregister.asp.align_strip_with_asp`). New runs are
guarded automatically.

Run::

    python -m pig.quarantine_bad_aligns                  # dry-run report
    python -m pig.quarantine_bad_aligns --apply          # actually quarantine
    python -m pig.quarantine_bad_aligns --asp-root <p>   # override root
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from pathlib import Path

from pig import config

_TRANSLATION_MAGNITUDE_RE = re.compile(
    r"Translation vector magnitude \(meters\):\s*([0-9eE+\-.]+)"
)


def _read_translation_magnitude(alignment_dir: str) -> float | None:
    logs = sorted(glob.glob(alignment_dir + "*-log-pc_align-*.txt"))
    if not logs:
        return None
    with open(logs[-1]) as fh:
        for line in fh:
            m = _TRANSLATION_MAGNITUDE_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def _dem_ids_in(asp_aligned: Path) -> list[str]:
    ids: set[str] = set()
    for p in asp_aligned.glob("SETSM_*-trans_reference-DEM.tif"):
        ids.add(p.name[: -len("-trans_reference-DEM.tif")])
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asp-root", type=Path,
        default=Path(config.STRIPS_DIR) / "ASP",
        help="ASP root containing asp_aligned/. Defaults to the current "
             "shared location; pass `<basin>/data/ASP` after migration.",
    )
    parser.add_argument(
        "--max-displacement", type=float, default=100.0,
        help="Shean 2019 pc_align cap (default 100 m). Strips with "
             "|Δ| > this are quarantined.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually move files. Without it, just print what would happen.",
    )
    args = parser.parse_args()

    asp_root: Path = args.asp_root
    aligned_dir = asp_root / "asp_aligned"
    if not aligned_dir.is_dir():
        raise SystemExit(f"❌ {aligned_dir} not found")

    bad: list[tuple[str, float]] = []
    missing: list[str] = []
    for dem_id in _dem_ids_in(aligned_dir):
        alignment_dir = str(aligned_dir / dem_id)
        delta = _read_translation_magnitude(alignment_dir)
        if delta is None:
            missing.append(dem_id)
            continue
        if delta > args.max_displacement:
            bad.append((dem_id, delta))

    print(f"ASP root:        {asp_root}")
    print(f"Threshold:       |Δ| > {args.max_displacement} m (Shean 2019)")
    print(f"Strips scanned:  {len(_dem_ids_in(aligned_dir))}")
    print(f"To quarantine:   {len(bad)}")
    if missing:
        print(f"Log missing:     {len(missing)} (cannot evaluate; left in place)")

    if not bad:
        print("\nNothing to quarantine.")
        return

    print("\nWill quarantine:")
    for dem_id, delta in bad:
        print(f"  |Δ|={delta:>14.1f} m  {dem_id}")

    if not args.apply:
        print("\n(dry-run — pass --apply to actually move files)")
        return

    bad_align_root = asp_root / "bad_align"
    bad_align_root.mkdir(parents=True, exist_ok=True)
    for dem_id, delta in bad:
        target = bad_align_root / dem_id
        target.mkdir(parents=True, exist_ok=True)
        moved = 0
        for src in glob.glob(str(aligned_dir / dem_id) + "*"):
            shutil.move(src, str(target / os.path.basename(src)))
            moved += 1
        reason = (
            f"|Δ|={delta:.1f} m > max_displacement={args.max_displacement} m "
            f"(Shean threshold). Quarantined retroactively by "
            f"pig.quarantine_bad_aligns."
        )
        (target / "_REASON.txt").write_text(reason + "\n")
        (aligned_dir / f"{dem_id}.bad_align").write_text(reason + "\n")
        print(f"  → moved {moved} files to {target}")

    print(f"\n✅ Quarantined {len(bad)} strips to {bad_align_root}/")


if __name__ == "__main__":
    main()
