"""Gate: the per-strip residual plane against independent control, and the
coloured-noise prior built from it (2026-09-02).

Why this exists. The melt inverse's strip-noise model needs the variance of
the residual per-strip plane error, and nothing built on the DEM stack alone
can give it: the per-strip tilt is not identifiable from the stack (loosening
the tilt prior grows the estimates without bound while their correlation
with truth FALLS). Independent altimetric control has no such degeneracy.
This gate checks the machinery on a synthetic where the answer is known:

C1  the plane fit recovers each strip's residual (offset, ax, ay) against
    control, with honest standard errors;
C2  the population tau^2 is the true across-strip variance -- INCLUDING the
    method-of-moments correction: with noisy, badly-spread control the naive
    var(est) overshoots, and subtracting mean(se^2) brings it back;
C3  the prior maps onto strip_mode_design's (strip_index, component) modes,
    per-strip and population; offset modes refuse to run without offset_var.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_control_strip_prior.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

X0C = Y0C = 0.0  # grid centre of the last make_strips call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from stereo_melt.coregister.alignment_quality import (  # noqa: E402
    fit_residual_plane,
    residual_planes_for_strips,
)
from stereo_melt.dynamics.budget_bridging import strip_prior_from_residual_planes  # noqa: E402

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def make_strips(tmp, n_strips, *, tau_x, tau_y, tau_c, dem_noise, ctl_noise,
                ctl_spread_y, n_ctl, seed):
    """Synthetic aligned DEMs = truth (0) + plane + white noise; control =
    0 + white noise, restricted to a band of half-height ctl_spread_y (so the
    y-slope is poorly constrained, like control along one strip edge)."""
    rng = np.random.default_rng(seed)
    res, nx, ny = 50.0, 300, 200
    x0, y0 = -1_600_000.0, -290_000.0
    tr = from_origin(x0, y0, res, res)
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 - (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    xc, yc = X.mean(), Y.mean()
    global X0C, Y0C
    X0C, Y0C = float(xc), float(yc)
    items, truth = [], []
    for k in range(n_strips):
        ax, ay, c = rng.normal(0, tau_x), rng.normal(0, tau_y), rng.normal(0, tau_c)
        dem = c + ax * (X - xc) + ay * (Y - yc) + rng.normal(0, dem_noise, X.shape)
        p = tmp / f"S{k:03d}-trans_reference-DEM.tif"
        with rasterio.open(p, "w", driver="GTiff", height=ny, width=nx, count=1,
                           dtype="float32", transform=tr, nodata=-9999.0) as dst:
            dst.write(dem.astype(np.float32), 1)
        ce = rng.uniform(xs.min() + res, xs.max() - res, n_ctl)
        cn = yc + rng.uniform(-ctl_spread_y, ctl_spread_y, n_ctl)
        ch = rng.normal(0, ctl_noise, n_ctl)               # truth 0 + control noise
        items.append((f"S{k:03d}", p, np.column_stack([ce, cn, ch])))
        truth.append((c, ax, ay))
    return items, np.array(truth)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gate_ctl_prior_"))
    print("C1  per-strip plane recovery against control (well-spread, quiet control)")
    items, truth = make_strips(tmp, 40, tau_x=1.5e-4, tau_y=1.5e-4, tau_c=0.3,
                               dem_noise=0.4, ctl_noise=0.1, ctl_spread_y=5000.0,
                               n_ctl=800, seed=1)
    df = residual_planes_for_strips(items)
    # The fit reports the offset at the CONTROL centroid (xc, yc); the strip's
    # offset is defined at the grid centre, so refer the truth there --
    # otherwise the plane times the ~150 m centroid shift reads as error.
    off_true = truth[:, 0] + truth[:, 1] * (df.xc.to_numpy() - X0C) + truth[:, 2] * (df.yc.to_numpy() - Y0C)
    for j, col in enumerate(("offset", "ax", "ay")):
        est, tru = df[col].to_numpy(), (off_true if col == "offset" else truth[:, j])
        corr = np.corrcoef(est, tru)[0, 1]
        slope = np.polyfit(tru, est, 1)[0]
        check(f"{col}: corr > 0.98 and slope within 5 %", corr > 0.98 and abs(slope - 1) < 0.05,
              f"corr {corr:.4f} slope {slope:.3f}")
    # standard errors are honest: z-scores of (est - true)/se have unit spread
    z = np.concatenate([(df[c].to_numpy() - (off_true if c == "offset" else truth[:, j]))
                        / df[f"se_{c}"].to_numpy() for j, c in enumerate(("offset", "ax", "ay"))])
    check("standard errors honest (rms z in [0.7, 1.4])", 0.7 < np.sqrt(np.mean(z ** 2)) < 1.4,
          f"rms z {np.sqrt(np.mean(z ** 2)):.2f}")

    print("C2  population tau^2 with noisy, one-edge control (the correction matters)")
    # Control confined to a 2 km band with 0.6 m noise: the y-slope's
    # estimator variance is ~0.3-0.4x the signal, so naive var(est) overshoots
    # by ~15-45 % and the moment correction is both material and stable
    # (4 seeds: corrected 0.86-1.08). Pushing the estimator variance to 2-3x
    # the signal makes var(est) - mean(se^2) a difference of two large noisy
    # numbers that no estimator resolves with ~100 strips -- not a regime to
    # gate on.
    # 200 strips: the population rule is MAD-based (robust to the alignment
    # failures real ASP roots carry -- C4), and MAD^2 has ~1.4x the sampling
    # spread of a variance, so tolerances below are set at ~2 sigma for N2.
    N2 = 200
    items2, truth2 = make_strips(tmp, N2, tau_x=1.5e-4, tau_y=1.5e-4, tau_c=0.3,
                                 dem_noise=0.6, ctl_noise=0.6, ctl_spread_y=1000.0,
                                 n_ctl=300, seed=2)
    df2 = residual_planes_for_strips(items2)
    sidx = np.repeat(np.arange(N2), 2)
    comp = np.tile(np.array(["tilt_x", "tilt_y"]), N2)
    tau2, summ = strip_prior_from_residual_planes(df2, df2.dem_id.values, sidx, comp,
                                                  return_summary=True)
    true_var = {"tilt_x": float(np.var(truth2[:, 1])), "tilt_y": float(np.var(truth2[:, 2]))}
    for c in ("tilt_x", "tilt_y"):
        naive = summ[c]["var_raw"] / true_var[c]
        corrected = summ[c]["tau2"] / true_var[c]
        print(f"      {c}: naive var(est)/true {naive:.2f}  corrected tau2/true {corrected:.2f}"
              f"   (mean se^2 / true {summ[c]['mean_se2'] / true_var[c]:.2f})")
        check(f"{c}: corrected tau2 within 25 % of truth", abs(corrected - 1) < 0.25,
              f"{corrected:.2f}")
    nv, cv = summ["tilt_y"]["var_raw"] / true_var["tilt_y"], summ["tilt_y"]["tau2"] / true_var["tilt_y"]
    check("tilt_y: naive overshoots by > 20 % and the correction lands closer",
          nv > 1.2 and abs(cv - 1) < abs(nv - 1), f"naive {nv:.2f} -> corrected {cv:.2f}")
    check("tau2 broadcast per mode (population = same value per component)",
          np.allclose(tau2[comp == "tilt_x"], summ["tilt_x"]["tau2"])
          and np.allclose(tau2[comp == "tilt_y"], summ["tilt_y"]["tau2"]))

    print("C3  mode mapping, per-strip variant, offset guard")
    # stack epoch order differs from the table order: mapping must go via dem_id
    perm = np.random.default_rng(3).permutation(N2)
    dem_ids = df2.dem_id.values[perm]
    sidx3 = np.array([5, 5, 17, 17, 40])
    comp3 = np.array(["tilt_x", "tilt_y", "tilt_x", "tilt_y", "tilt_x"])
    t_pop = strip_prior_from_residual_planes(df2, dem_ids, sidx3, comp3)
    t_ps = strip_prior_from_residual_planes(df2, dem_ids, sidx3, comp3, per_strip=True)
    row = df2.set_index("dem_id").loc[dem_ids[5]]
    exp = max(row.ax ** 2 - row.se_ax ** 2, 0.05 * summ["tilt_x"]["tau2"])
    check("per-strip tau2 resolves the strip through dem_id, not table order",
          np.isclose(t_ps[0], exp), f"{t_ps[0]:.3e} vs {exp:.3e}")
    check("population variant ignores the strip", np.allclose(t_pop[[0, 2, 4]], t_pop[0]))
    try:
        strip_prior_from_residual_planes(df2, dem_ids, np.array([0]), np.array(["offset"]))
        check("offset mode without offset_var raises", False)
    except ValueError as exc:
        check("offset mode without offset_var raises", "offset_var" in str(exc))
    ov = np.linspace(0.1, 0.5, N2) ** 2
    t_off = strip_prior_from_residual_planes(df2, dem_ids, np.array([7, 7]),
                                             np.array(["offset", "tilt_x"]), offset_var=ov)
    check("offset mode takes the per-epoch offset_var", np.isclose(t_off[0], ov[7]))

    print("C4  robustness to alignment failures (real ASP roots have them)")
    # PIG canon: ~15 % of strips are pc_align failures with residual offsets
    # of 60-207 m and scatter of 68-175 m; they carried 100 % of a
    # non-robust variance. Inject the same and require the population tau2
    # to be unmoved while the naive one blows up, and the QC to name them.
    bad = df2.copy()
    nb = 20   # 10 % of the population
    bad.loc[: nb - 1, ["offset", "ax", "ay", "sd"]] = [[150.0, 0.02, -0.03, 90.0]] * nb
    t_rob, s_rob = strip_prior_from_residual_planes(bad, bad.dem_id.values, sidx, comp, return_summary=True)
    t_nai, s_nai = strip_prior_from_residual_planes(bad, bad.dem_id.values, sidx, comp, robust=False,
                                                    max_sd=None, min_n=0, return_summary=True)
    for c in ("tilt_x", "tilt_y"):
        r_rob, r_nai = s_rob[c]["tau2"] / true_var[c], s_nai[c]["tau2"] / true_var[c]
        print(f"      {c}: robust+QC tau2/true {r_rob:.2f}   naive tau2/true {r_nai:.1f}")
        check(f"{c}: robust+QC tau2 within 30 % of truth with failed strips injected",
              abs(r_rob - 1) < 0.30, f"{r_rob:.2f}")
        check(f"{c}: naive tau2 blown up by the failures (> 5x)", r_nai > 5, f"{r_nai:.1f}")
    check("QC names exactly the failed strips",
          s_rob["n_qc_dropped"] == nb and set(s_rob["qc_dropped"]) == set(bad.dem_id.values[:nb]))
    # a robust fit on CLEAN strips must agree with the non-robust one
    s_c = strip_prior_from_residual_planes(df2, df2.dem_id.values, sidx, comp, robust=False, return_summary=True)[1]
    check("robust == non-robust on Gaussian strips (no systematic gap; within 25 %)",
          all(abs(summ[c]["tau2"] / s_c[c]["tau2"] - 1) < 0.25 for c in ("tilt_x", "tilt_y")),
          "  ".join(f"{c} {summ[c]['tau2'] / s_c[c]['tau2']:.2f}" for c in ("tilt_x", "tilt_y")))

    # direct fit_residual_plane sanity: too few points -> NaNs, n reported
    r = fit_residual_plane(np.arange(10.0), np.arange(10.0), np.zeros(10))
    check("fit refuses < min_points with NaNs and n", np.isnan(r["ax"]) and r["n"] == 10)

    print("\nGATE " + ("PASSED" if not FAILS else f"FAILED: {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
