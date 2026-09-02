"""PIG: how the melt error falls with DEM count - measured, not assumed.

The noise-floor extrapolation ("~1.4x the strips lifts the trunk bridging
band to SNR 1") assumes noise variance ~ 1/n. Two measurements of the actual
scaling:

1. SUBSET LADDER (clean). The stack is dealt into interleaved halves and
   quarters (disjoint strips, full time span; ``run_noise_floor --quarters``
   writes both rungs from one run, and a halves file from another run is
   only accepted if its recorded solver settings match), the Eulerian melt
   is solved
   from each, and pairwise differences give the noise at n/2 and n/4 over
   the SAME pixels: Var[A-B] = 2 sigma^2 at that count, so sigma_{n/2} =
   rms(A-B)/sqrt(2) and sigma_{n/4} = rms(Qi-Qj)/sqrt(2). Everything else
   (region, velocity, SMB, window) is
   held fixed, so the slope between the two rungs is the count scaling
   alone. White per-epoch errors: sigma ~ n^{-1/2}.

2. SPATIAL BINNING (deliberately NOT computed or drawn). Binning the
   per-pixel half-difference by that pixel's DEM count LOOKS like the same
   measurement but is confounded: count correlates with location (the
   well-tasked trunk has both more strips and intrinsically larger errors),
   so the whole-shelf "slope" comes out positive. The figure omits it; only
   the subset ladder over identical pixels is a clean count scaling.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY -m pig.plot_error_vs_count
"""
from __future__ import annotations

import argparse
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
from pig.plot_noise_floor import check_provenance  # noqa: E402
from pig.run_melt import load_stack  # noqa: E402

NC_HALF = config.PROCESSED_DIR / "pig_noise_floor_250m_is2ctempo_sheltilt.nc"
NC_Q = config.PROCESSED_DIR / "pig_noise_floor_250m_is2ctempo_sheltilt_q.nc"
HALVES = ("eulerian_A", "eulerian_B")
QUARTERS = ("eulerian_Q0", "eulerian_Q1", "eulerian_Q2", "eulerian_Q3")


def open_ladder(quarters_nc, half_nc=None):
    """``(halves, quarters)`` datasets for the ladder.

    The quarters file carries the same run's halves, so by default both rungs
    are read from it; an explicit ``half_nc`` (or a quarters file without
    halves, from an older run) falls back to the separate halves file and
    is checked for matching provenance.
    """
    q = xr.open_dataset(quarters_nc)
    if half_nc is None and all(k in q for k in HALVES):
        print(f"  halves and quarters from one run: {quarters_nc}", flush=True)
        return q, q
    half = xr.open_dataset(half_nc or NC_HALF)
    ce, vel = check_provenance(
        q, half, [(f"ladder {qk}", qk, *HALVES, None) for qk in QUARTERS],
        full_name="quarters", purpose="the count scaling",
        hint="re-run run_noise_floor --quarters with the halves' settings (its "
             "output carries both rungs) or pass a matching --half-nc")[0]
    print(f"  halves {half_nc or NC_HALF} + quarters {quarters_nc}: "
          f"provenance agrees (common_epoch={ce}, velocity={vel!r})", flush=True)
    return half, q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quarters-nc", default=str(NC_Q),
                    help="run_noise_floor --quarters output (halves + quarters)")
    ap.add_argument("--half-nc", default=None,
                    help="separate halves file (default: the halves solved in the "
                         "same run as the quarters); refused if its common_epoch/"
                         "velocity provenance differs from the quarters")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    half, q = open_ladder(args.quarters_nc, args.half_nc)
    st = load_stack("pig_stack_250m_is2ctempo_sheltilt")
    n_full = np.isfinite(st.values).sum(0).astype(float)
    z = np.load(config.PROCESSED_DIR / "pig_eta_field_250m_dual_20260730_t0era5.npz")
    u = xr.DataArray(np.hypot(z["u_model_x"], z["u_model_y"]), dims=("y", "x"),
                     coords={"y": z["y"], "x": z["x"]})
    u = u.reindex_like(half.eulerian_A, method="nearest").values

    # common pixels: finite in halves and all four quarters
    fin = np.isfinite(half.eulerian_A.values) & np.isfinite(half.eulerian_B.values)
    for k in QUARTERS:
        fin &= np.isfinite(q[k].values)
    d_half = (half.eulerian_A - half.eulerian_B).values
    d_q01 = (q.eulerian_Q0 - q.eulerian_Q1).values
    d_q23 = (q.eulerian_Q2 - q.eulerian_Q3).values

    regions = [("whole shelf", fin, "0.15"),
               ("fast trunk (u≥1 km/yr)", fin & np.isfinite(u) & (u >= 1000), "#d62728"),
               ("slow shelf (u<1 km/yr)", fin & np.isfinite(u) & (u < 1000), "#1f77b4")]

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    print(f"  {'region':24s} {'n/4':>7s} {'n/2':>7s} {'sig(n/4)':>9s} {'sig(n/2)':>9s} "
          f"{'slope':>7s} {'strips for trunk SNR=1':>12s}")
    results = {}
    for label, m, colour in regions:
        nm = float(np.median(n_full[m]))
        s4 = float(np.sqrt(0.5 * np.nanmean(np.concatenate(
            [d_q01[m] ** 2, d_q23[m] ** 2]))))
        s2 = float(np.sqrt(0.5 * np.nanmean(d_half[m] ** 2)))
        slope = np.log(s2 / s4) / np.log((nm / 2) / (nm / 4))
        results[label] = slope
        ax.loglog([nm / 4, nm / 2], [s4, s2], "o-", color=colour, lw=2.0, ms=7,
                  label=f"{label}: slope {slope:+.2f}")
        # extrapolate to full count with the measured slope
        sf = s2 * (nm / (nm / 2)) ** slope
        ax.loglog([nm / 2, nm], [s2, sf], ":", color=colour, lw=1.4)
        ax.loglog([nm], [sf], "s", color=colour, ms=6, mfc="none")
        need = (1.0 / 0.72) ** (1.0 / max(-2.0 * slope, 1e-9))
        need_s = f"{need:.1f}x" if need < 99 else ">99x"
        print(f"  {label:24s} {nm/4:7.1f} {nm/2:7.1f} {s4:9.1f} {s2:9.1f} "
              f"{slope:+7.2f} {need_s:>12s}")

    # white-noise reference
    nm = float(np.median(n_full[fin]))
    s2w = float(np.sqrt(0.5 * np.nanmean(d_half[fin] ** 2)))
    nn = np.array([nm / 4, nm])
    ax.loglog(nn, s2w * (nn / (nm / 2)) ** -0.5, "--", color="0.55", lw=1.5,
              label="σ ∝ n$^{-1/2}$ (white per-epoch errors)")

    ax.set_xlabel("DEM count n (median epochs per pixel in the subset)")
    ax.set_ylabel("melt noise σ (m a$^{-1}$)  from disjoint-subset differences")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    sl = results.get("fast trunk (u≥1 km/yr)", -0.5)
    need = (1.0 / 0.72) ** (1.0 / max(-2.0 * sl, 1e-9))
    ax.set_title("PIG melt error vs DEM count — subset ladder (quarters → halves), "
                 "same pixels, same everything\nempirical trunk slope "
                 f"{sl:+.2f} (white −0.50) ⇒ trunk bridging band needs "
                 f"~{need:.1f}× the strips (1/n assumption said 1.4×)", fontsize=11)
    out = args.out or (config.FIGURES_DIR / "melt_error_vs_dem_count.png")
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
