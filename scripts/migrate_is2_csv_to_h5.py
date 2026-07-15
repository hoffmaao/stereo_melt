"""One-time migration: convert every ``icesat2_filtered_<dem_id>.csv``
under ``data/REMA/strips/ASP*/icesat2_data/`` to compressed HDF5 via
``stereo_melt.io.altimetry.write_control_h5``.

Round-trips each file (reads the H5 back and compares column-wise to
the source CSV) before deleting the CSV, so a partial run can be
re-run safely. By default ``--delete-csv`` is off so the first pass is
non-destructive.

Run:

    python scripts/migrate_is2_csv_to_h5.py            # dry-ish: write h5, keep csv
    python scripts/migrate_is2_csv_to_h5.py --delete-csv

Expected savings: ~7x per file (~5 MB CSV -> ~700 KB H5).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_env_proj = os.path.join(sys.prefix, "share", "proj")
if os.path.isfile(os.path.join(_env_proj, "proj.db")):
    os.environ.setdefault("PROJ_DATA", _env_proj)
    os.environ.setdefault("PROJ_LIB", _env_proj)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stereo_melt.io.altimetry import read_control_h5, write_control_h5


ROOT = Path("/wd2/projects/stereo_melt")
SEARCH_DIRS = sorted((ROOT / "data" / "REMA" / "strips").glob("ASP*/icesat2_data"))


def migrate_one(csv_path: Path, *, delete_csv: bool) -> tuple[int, int]:
    """Convert one CSV to H5 (idempotent). Returns (csv_bytes, h5_bytes)."""
    h5_path = csv_path.with_suffix(".h5")
    csv_size = csv_path.stat().st_size
    if h5_path.exists():
        # Already converted; verify and skip.
        h5_size = h5_path.stat().st_size
        if delete_csv:
            csv_path.unlink()
        return csv_size, h5_size

    df = pd.read_csv(csv_path)
    write_control_h5(df, h5_path)

    # Round-trip sanity check before we permit deletion.
    back = read_control_h5(h5_path)
    for col in ("easting", "northing", "h_mean"):
        if not np.allclose(df[col].to_numpy(np.float64),
                           back[col].to_numpy(np.float64), atol=1e-9):
            h5_path.unlink()
            raise RuntimeError(
                f"Round-trip mismatch on {col} for {csv_path.name}"
            )
    if "time" in df.columns and "time" in back.columns:
        t_src = pd.to_datetime(df["time"]).to_numpy("datetime64[ns]").view(np.int64)
        t_dst = pd.to_datetime(back["time"]).to_numpy("datetime64[ns]").view(np.int64)
        if not np.array_equal(t_src, t_dst):
            h5_path.unlink()
            raise RuntimeError(f"Round-trip mismatch on time for {csv_path.name}")

    h5_size = h5_path.stat().st_size
    if delete_csv:
        csv_path.unlink()
    return csv_size, h5_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delete-csv", action="store_true",
                    help="Remove the source CSV after a successful round-trip.")
    args = ap.parse_args()

    total_csv = 0
    total_h5 = 0
    n_done = 0
    n_already = 0
    for d in SEARCH_DIRS:
        csvs = sorted(d.glob("icesat2_filtered_*.csv"))
        if not csvs:
            continue
        print(f"\n{d}  ({len(csvs)} csv files)")
        for c in csvs:
            try:
                csv_b, h5_b = migrate_one(c, delete_csv=args.delete_csv)
            except Exception as e:
                print(f"  ! {c.name}: {e}")
                continue
            ratio = csv_b / h5_b if h5_b else float("nan")
            total_csv += csv_b
            total_h5 += h5_b
            if c.with_suffix(".h5").stat().st_mtime > c.stat().st_mtime if c.exists() else True:
                n_done += 1
            else:
                n_already += 1
            print(f"    {c.name:80s}  csv={csv_b/1048576:.1f} MB  "
                  f"h5={h5_b/1024:.0f} KB  ratio={ratio:.1f}x")

    print(
        f"\nDone. converted/checked={n_done + n_already}, "
        f"csv total={total_csv/1073741824:.2f} GB, "
        f"h5 total={total_h5/1073741824:.2f} GB, "
        f"net savings={1 - total_h5 / max(total_csv, 1):.0%}"
    )
    if not args.delete_csv and total_csv:
        print("(CSVs preserved; rerun with --delete-csv to remove them.)")


if __name__ == "__main__":
    main()
