"""Per-strip A/B of the pre-IS2 align: CryoTEMPO control (ASP_ctempoatmlvis)
vs the raw-CS2 baseline (ASP_cs2atmlvis).

Each row reports median/MAD of the post-align geodiff against that run's OWN
control set, so this measures convergence quality, not absolute accuracy.
Rock-dominated strips (rock GCPs swamp the altimetry) converge to identical
solutions in both runs and show up as identical-to-the-cent rows; the
discriminating rows are the altimetry-dominated ones, flagged DIVERGED below.

Run: $PY -m pig.scripts.compare_ctempo_align_stats   (or as a plain script)
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)
CTEMPO = os.path.join(DATA, "ASP_ctempoatmlvis")
BASELINE = os.path.join(DATA, "ASP_cs2atmlvis")


def strip_stats(diff_csv):
    df = pd.read_csv(diff_csv, comment="#", names=["easting", "northing", "diff"])
    d = pd.to_numeric(df["diff"], errors="coerce").dropna().values
    if len(d) == 0:
        return None
    return np.median(d), stats.median_abs_deviation(d), len(d)


def main():
    dems = sorted(
        glob.glob(os.path.join(CTEMPO, "asp_aligned", "*-trans_reference-DEM.tif"))
    )
    print(f"{len(dems)} ctempo-aligned strips")
    print(
        f"{'strip':<58} {'ctempo med/MAD (n)':>24} "
        f"{'cs2 med/MAD (n)':>24}  flag"
    )
    diverged, identical, new_strips = [], [], []
    for dem in dems:
        stem = os.path.basename(dem).replace("-trans_reference-DEM.tif", "")
        c_csv = os.path.join(CTEMPO, "final", f"{stem}-final-diff.csv")
        b_csv = os.path.join(BASELINE, "final", f"{stem}-final-diff.csv")
        c = strip_stats(c_csv) if os.path.exists(c_csv) else None
        b = strip_stats(b_csv) if os.path.exists(b_csv) else None
        if c is None:
            continue
        cs = f"{c[0]:+.2f}/{c[1]:.2f} ({c[2]})"
        if b is None:
            new_strips.append(stem)
            print(f"{stem.replace('SETSM_s2s041_', ''):<58} {cs:>24} "
                  f"{'(not in baseline)':>24}  NEW")
            continue
        bs = f"{b[0]:+.2f}/{b[1]:.2f} ({b[2]})"
        same = abs(c[0] - b[0]) < 0.05 and (b[1] == 0 or c[1] / b[1] > 1 / 1.2 and c[1] / b[1] < 1.2)
        flag = "" if same else "DIVERGED"
        (identical if same else diverged).append((stem, c, b))
        print(f"{stem.replace('SETSM_s2s041_', ''):<58} {cs:>24} {bs:>24}  {flag}")

    print(f"\n{len(identical)} rock-pinned/equivalent, {len(diverged)} diverged, "
          f"{len(new_strips)} only in ctempo")
    if diverged:
        a = np.array([(c[0], c[1], b[0], b[1]) for _, c, b in diverged])
        print(
            "diverged-strip aggregate: |median| "
            f"ctempo {np.median(np.abs(a[:, 0])):.2f} vs cs2 {np.median(np.abs(a[:, 2])):.2f} m; "
            f"MAD ctempo {np.median(a[:, 1]):.2f} vs cs2 {np.median(a[:, 3]):.2f} m"
        )


if __name__ == "__main__":
    main()
