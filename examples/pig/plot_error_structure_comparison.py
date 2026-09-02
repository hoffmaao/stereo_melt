"""Real-PIG vs twin per-strip error structure, side by side.

Consumes the JSON summaries from :mod:`pig.diagnose_stack_error_structure`
(run on the real 250 m PIG stack and on a twin corrected stack — by default
the white-noise ``multixy_pig`` tier; each JSON must have been measured on
the stack it is paired with here, which is checked against the fingerprint
the diagnostic records) and shows (a) the per-strip metric
distributions and (b) one example detrended residual field from each, on a
common colour scale. The headline (rms ratio, block excess at 2/4 km) and the
annotation are derived from the loaded rows, so the figure reports whatever
the chosen twin tier actually measures — white and undersized for the default
tier, matched for the calibrated ``multixy_pigreal`` tier.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY -m pig.plot_error_structure_comparison
    $PY -m pig.plot_error_structure_comparison \\
        --twin-json <pigreal.json> --twin-nc <pigreal.nc> \\
        --twin-label "twin pigreal" --out <png>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, f"{_REPO}/examples")
sys.path.insert(0, f"{_REPO}/src")

from stereo_melt import envsetup  # noqa: F401,E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from pig import config  # noqa: E402
from pig.diagnose_stack_error_structure import (  # noqa: E402
    check_fingerprint, epoch_residual, load_real_pig_stack, pixel_trend)

J_REAL = config.PROCESSED_DIR / "error_structure_real.json"
J_TWIN = config.PROCESSED_DIR / "error_structure_twin.json"
TWIN_NC = (f"{_REPO}/examples/elmer_synth/data/processed/"
           "e2a_stack_200m_multixy_pig_tilt_corrected.nc")
OUT_PNG = config.FIGURES_DIR / "error_structure_real_vs_twin.png"
METRICS = [("offset_m", "strip offset (m)", "symlog"),
           ("tilt_dm_km", "residual tilt (dm/km)", "log"),
           ("rms_m", "detrended rms (m)", "log"),
           ("excess_2km", "block excess @2 km\n(1 = white)", "log"),
           ("excess_4km", "block excess @4 km\n(1 = white)", "log")]


def metric_values(rows, key):
    v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
    return np.abs(v) if key == "offset_m" else v


def median(rows, key):
    v = metric_values(rows, key)
    return float(np.median(v)) if len(v) else np.nan


def pick_epoch(z, trend, rows):
    """Epoch nearest the median detrended rms among the largest footprints."""
    big = sorted(rows, key=lambda r: -r["px"])[:max(len(rows) // 4, 8)]
    med = np.median([r["rms_m"] for r in big])
    tgt = min(big, key=lambda r: abs(r["rms_m"] - med))
    if "epoch" in tgt:
        return int(tgt["epoch"])
    # pre-'epoch' JSON: match on the same (finite & trend-fitted) pixel count,
    # breaking ties on the row's detrended rms
    px = (trend.fin & trend.ok_px[None]).sum((1, 2))
    cand = np.flatnonzero(px == tgt["px"])
    if len(cand) == 0:
        cand = np.array([np.argmin(np.abs(px - tgt["px"]))])
    if len(cand) == 1:
        return int(cand[0])

    def rms_of(i):
        r2 = epoch_residual(z, trend, i)[2]
        return float(np.sqrt(np.nanmean(r2 ** 2)))
    return int(min(cand, key=lambda i: abs(rms_of(i) - tgt["rms_m"])))


def short_label(d, override):
    if override:
        return override
    lab = d.get("label") or "twin"
    return os.path.basename(lab) if os.sep in lab else lab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-json", default=str(J_REAL),
                    help="diagnose_stack_error_structure JSON for the real stack")
    ap.add_argument("--twin-json", default=str(J_TWIN),
                    help="diagnose_stack_error_structure JSON for the twin stack")
    ap.add_argument("--twin-nc", default=TWIN_NC,
                    help="twin corrected stack the example residual map is drawn from")
    ap.add_argument("--twin-label", default=None,
                    help="caption label for the twin (default: the JSON's label)")
    ap.add_argument("--out", default=str(OUT_PNG))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    real = json.load(open(args.real_json))
    twin = json.load(open(args.twin_json))
    twin_lab = short_label(twin, args.twin_label)

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.35], hspace=0.34, wspace=0.45)
    for j, (key, label, scale) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j])
        for k, (d, colour) in enumerate(((real, "#1f77b4"), (twin, "#ff7f0e"))):
            v = metric_values(d["rows"], key)
            p16, p50, p84 = np.percentile(v, [16, 50, 84])
            ax.errorbar([k], [p50], yerr=[[p50 - p16], [p84 - p50]], fmt="o",
                        color=colour, capsize=5, ms=8, lw=2)
        if key.startswith("excess"):
            ax.axhline(1.0, color="0.5", lw=1.0, ls="--")
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["real\nPIG", "twin"], fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(("|" + label + "|") if key == "offset_m" else label, fontsize=9.5)
        ax.grid(True, axis="y", alpha=0.3)

    # example residual maps
    st_r = load_real_pig_stack()
    ds_t = xr.open_dataset(args.twin_nc)
    var = [v for v in ds_t.data_vars if ds_t[v].dims[-2:] == ("y", "x")
           and "time" in ds_t[v].dims][0]
    st_t = ds_t[var]
    check_fingerprint(real, st_r, None, f"--real-json {args.real_json}")
    check_fingerprint(twin, st_t, args.twin_nc, f"--twin-json {args.twin_json}")
    for col, (st, rows, lab) in enumerate(
            ((st_r, real["rows"], "REAL PIG strip residual"),
             (st_t, twin["rows"], "TWIN strip residual"))):
        trend = pixel_trend(st)
        i = pick_epoch(st.values, trend, rows)
        _, _, r = epoch_residual(st.values, trend, i)
        dx = float(abs(st.x.values[1] - st.x.values[0]))
        ax = fig.add_subplot(gs[1, col * 2:(col * 2 + 2)])
        im = ax.pcolormesh(st.x.values / 1e3, st.y.values / 1e3, r,
                           cmap="RdBu_r", vmin=-4, vmax=4, shading="nearest",
                           rasterized=True)
        ys, xs = np.where(np.isfinite(r))
        ax.set_xlim(st.x.values[xs.min()] / 1e3 - 2, st.x.values[xs.max()] / 1e3 + 2)
        ax.set_ylim(min(st.y.values[ys.min()], st.y.values[ys.max()]) / 1e3 - 2,
                    max(st.y.values[ys.min()], st.y.values[ys.max()]) / 1e3 + 2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        rms = float(np.sqrt(np.nanmean(r ** 2)))
        ax.set_title(f"{lab} ({dx:.0f} m) — epoch {i}, detrended rms {rms:.2f} m",
                     fontsize=10)
        if col == 1:
            fig.colorbar(im, ax=ax, shrink=0.8, label="residual about trend (m), ±4")

    rms_r, rms_t = median(real["rows"], "rms_m"), median(twin["rows"], "rms_m")
    ex_r = [median(real["rows"], k) for k in ("excess_2km", "excess_4km")]
    ex_t = [median(twin["rows"], k) for k in ("excess_2km", "excess_4km")]
    cax = fig.add_subplot(gs[1, 4])
    cax.set_axis_off()
    cax.text(0, 0.5,
             f"real strips (median):\nrms {rms_r:.2f} m\n"
             f"block excess {ex_r[0]:.1f} @2 km,\n{ex_r[1]:.1f} @4 km\n\n"
             f"twin strips, {twin_lab}\n(median):\nrms {rms_t:.2f} m\n"
             f"block excess {ex_t[0]:.1f} @2 km,\n{ex_t[1]:.1f} @4 km\n\n"
             "(excess 1 = white)",
             fontsize=10, va="center")
    fig.suptitle("Per-strip error structure: real PIG stack vs the synthetic twin "
                 f"({twin_lab}; both tilt-corrected tiers)\n"
                 f"twin/real median rms = {rms_t / rms_r:.2f}; "
                 f"block excess @2/4 km real {ex_r[0]:.1f}/{ex_r[1]:.1f}, "
                 f"twin {ex_t[0]:.1f}/{ex_t[1]:.1f} (1 = white)",
                 fontsize=12.5)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
