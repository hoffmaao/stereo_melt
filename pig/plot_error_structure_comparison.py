"""Real-PIG vs twin per-strip error structure, side by side.

Consumes the JSON summaries from :mod:`pig.diagnose_stack_error_structure`
(run on the real 250 m PIG stack and on the twin ``multixy_pig`` corrected
stack) and shows (a) the per-strip metric distributions and (b) one example
detrended residual field from each, on a common colour scale — the twin's is
salt-and-pepper white noise, the real one carries multi-km correlated
structure in exactly the band the melt solvers work in.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY -m pig.plot_error_structure_comparison
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/wd2/projects/stereo_melt")
sys.path.insert(0, "/wd2/projects/stereo_melt/stereo_melt/src")

from stereo_melt import envsetup  # noqa: F401,E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from pig import config  # noqa: E402

J_REAL = config.PROCESSED_DIR / "error_structure_real.json"
J_TWIN = config.PROCESSED_DIR / "error_structure_twin.json"
TWIN_NC = ("/wd2/projects/stereo_melt/elmer_synth/data/processed/"
           "e2a_stack_200m_multixy_pig_tilt_corrected.nc")
METRICS = [("offset_m", "strip offset (m)", "symlog"),
           ("tilt_dm_km", "residual tilt (dm/km)", "log"),
           ("rms_m", "detrended rms (m)", "log"),
           ("excess_2km", "block excess @2 km\n(1 = white)", "log"),
           ("excess_4km", "block excess @4 km\n(1 = white)", "log")]


def residual_field(stack: xr.DataArray, i: int):
    z = stack.values
    t = (stack.time.values - stack.time.values.min()) / np.timedelta64(1, "s")
    t = np.asarray(t, float)
    fin = np.isfinite(z)
    n = fin.sum(0)
    St = np.tensordot(t, fin, axes=(0, 0))
    Sz = np.nansum(z, 0)
    Stz = np.tensordot(t, np.where(fin, z, 0.0), axes=(0, 0))
    Stt = np.tensordot(t * t, fin, axes=(0, 0))
    den = n * Stt - St * St
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(den > 0, (n * Stz - St * Sz) / den, np.nan)
        icept = np.where(n > 0, (Sz - slope * St) / n, np.nan)
    r = z[i] - (icept + slope * t[i])
    return r - np.nanmean(r)


def pick_epoch(stack, rows):
    """Epoch nearest the median detrended rms among the largest footprints."""
    big = sorted(rows, key=lambda r: -r["px"])[:max(len(rows) // 4, 8)]
    med = np.median([r["rms_m"] for r in big])
    tgt = min(big, key=lambda r: abs(r["rms_m"] - med))
    # rows carry no index; find epoch whose footprint size matches
    fin = np.isfinite(stack.values).sum((1, 2))
    return int(np.argmin(np.abs(fin - tgt["px"])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    real = json.load(open(J_REAL))
    twin = json.load(open(J_TWIN))

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.35], hspace=0.34, wspace=0.45)
    for j, (key, label, scale) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j])
        for k, (src, colour) in enumerate((("real", "#1f77b4"), ("twin", "#ff7f0e"))):
            d = real if src == "real" else twin
            v = np.array([r[key] for r in d["rows"]
                          if np.isfinite(r.get(key, np.nan))])
            if key == "offset_m":
                v = np.abs(v)
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
    from pig.run_melt import apply_min_extent, load_floating_mask, load_stack
    st_r = load_stack("pig_stack_250m_is2ctempo_sheltilt")
    fl = apply_min_extent(load_floating_mask(st_r), "_250m_is2ctempo_sheltilt",
                          str(config.START_TIME), str(config.END_TIME))
    st_r = st_r.where(fl)
    ds_t = xr.open_dataset(TWIN_NC)
    var = [v for v in ds_t.data_vars if "time" in ds_t[v].dims][0]
    st_t = ds_t[var]
    for col, (st, rows, lab) in enumerate(
            ((st_r, real["rows"], "REAL PIG strip residual (250 m)"),
             (st_t, twin["rows"], "TWIN strip residual (200 m)"))):
        i = pick_epoch(st, rows)
        r = residual_field(st, i)
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
        ax.set_title(f"{lab} — epoch {i}, detrended rms {rms:.2f} m", fontsize=10)
        if col == 1:
            fig.colorbar(im, ax=ax, shrink=0.8, label="residual about trend (m), ±4")
    cax = fig.add_subplot(gs[1, 4])
    cax.set_axis_off()
    cax.text(0, 0.5,
             "real strips: correlated\nundulations at km scales\n"
             "(jitter, coreg ripple,\ntide/IBE/firn residuals)\n\n"
             "twin strips: white noise\n+ plane, by construction",
             fontsize=10, va="center")
    fig.suptitle("Per-strip error structure: real PIG stack vs the synthetic twin "
                 "(both tilt-corrected tiers)\nthe twin's noise is white and ~5x "
                 "too small; real residuals are correlated at exactly the 1-4 km "
                 "melt band", fontsize=12.5)
    out = config.FIGURES_DIR / "error_structure_real_vs_twin.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
