"""Gate: empirical variogram + correlated-error propagation (Hugonnet 2022).

Why this exists. Every regularisation selector we have tried assumes white
observation noise and fails on real stacks, because DEM strip error is
correlated at kilometre scales. This module measures that correlation without
a model and turns it into the error of a SPATIAL AVERAGE -- which is what a
melt flux is. Using the pixel count as the sample count when the noise is
correlated understates a flux uncertainty by an order of magnitude, so the
n_eff path is the one that has to be right.

G1  recovery: simulate a Gaussian field with a KNOWN correlation range and
    recover range and sill from the empirical variogram.
G2  limits: white noise -> pure nugget, n_eff ~ N; a field correlated across
    the whole domain -> n_eff ~ 1. These bracket every real case.
G3  robustness: Dowd resists blunders that wreck Matheron (real DEM residuals
    are heavy-tailed).
G4  the propagation identity: n_eff computed by double sum reproduces the
    variance of many independent realisations of the field's own mean.
G5  our per-strip Sigma model (offset + plane) implies a specific variogram;
    check the measured one matches, which is what makes the variogram a test
    of the control-derived Sigma rather than a separate number.
H1-H4  heteroscedasticity, the step that must come BEFORE the variogram:
    recover a known sigma(count, rmse), show standardisation removes the
    heteroscedasticity, and show that only the standardised variogram has a
    meaningful (dimensionless, ~1) sill and an unbiased range.

Run::

    PY=/home/hoffmaao/miniconda3/envs/stereo_melt/bin/python
    $PY tests/gate_variogram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stereo_melt.spatialstats import (  # noqa: E402
    covariance_from_variogram,
    empirical_variogram,
    fit_variogram,
    infer_heteroscedasticity_from_stable,
    interp_nd_binning,
    nmad,
    number_effective_samples,
    variogram_model,
)

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def gaussian_field(n, res, corr_len, sigma, rng):
    """Stationary field with a Gaussian covariance, by spectral synthesis."""
    k = np.fft.fftfreq(n, d=res)
    KX, KY = np.meshgrid(k, k)
    k2 = KX ** 2 + KY ** 2
    # Gaussian covariance exp(-(h/a)^2) <-> Gaussian spectrum
    P = np.exp(-(np.pi ** 2) * k2 * corr_len ** 2)
    w = rng.normal(size=(n, n))
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * np.sqrt(P)))
    return sigma * f / f.std()


def main() -> int:
    rng = np.random.default_rng(0)
    N, RES = 256, 100.0                      # 25.6 km domain
    xs = np.arange(N) * RES
    X, Y = np.meshgrid(xs, xs)
    e, n = X.ravel(), Y.ravel()

    print("G1  recover a known correlation range")
    CORR, SIG = 2000.0, 1.5
    z = gaussian_field(N, RES, CORR, SIG, rng).ravel()
    ev = empirical_variogram(e, n, z, max_lag=10000.0, n_bins=18, seed=1)
    fit = fit_variogram(ev["lags"], ev["gamma"], counts=ev["counts"],
                        models=("nugget", "gaussian"))
    rec_range = fit["ranges"][0]
    print(f"      sill {fit['total_sill']:.3f} (sample var {ev['variance']:.3f})   range {rec_range:.0f} m "
          f"(true {CORR:.0f})   r2 {fit['r2']:.4f}   bins {fit['n_bins']}")
    check("range recovered within 30 %", abs(rec_range / CORR - 1) < 0.30,
          f"{rec_range:.0f} vs {CORR:.0f} m")
    check("sill recovered within 25 % of the sample variance",
          abs(fit["total_sill"] / ev["variance"] - 1) < 0.25,
          f"{fit['total_sill']:.3f} vs {ev['variance']:.3f}")
    check("fit quality r2 > 0.95", fit["r2"] > 0.95, f"{fit['r2']:.4f}")

    print("G2  limits: white noise, and a fully-correlated field")
    zw = rng.normal(0, 1.0, e.size)
    evw = empirical_variogram(e, n, zw, max_lag=10000.0, n_bins=12, seed=2)
    flat = float(np.std(evw["gamma"]) / np.mean(evw["gamma"]))
    print(f"      white: gamma flat to {100*flat:.1f} % across lags")
    check("white noise gives a flat (nugget) variogram", flat < 0.10, f"{100*flat:.1f} %")
    sub = rng.choice(e.size, 4000, replace=False)
    nw = number_effective_samples(e[sub], n[sub], [("nugget", 1.0, 0.0)],
                                  n_subsample=800, seed=3)
    print(f"      white: n_eff {nw['n_eff']:.0f} of {nw['n_points']} points "
          f"(n_subsample 800 -- the answer must not depend on it)")
    check("white noise: n_eff ~ N (within 15 %)", abs(nw["n_eff"] / nw["n_points"] - 1) < 0.15,
          f"{nw['n_eff']:.0f}/{nw['n_points']}")
    nw2 = number_effective_samples(e[sub], n[sub], [("nugget", 1.0, 0.0)],
                                   n_subsample=2500, seed=3)
    check("n_eff independent of the subsample size (within 5 %)",
          abs(nw2["n_eff"] / nw["n_eff"] - 1) < 0.05,
          f"{nw['n_eff']:.0f} (800) vs {nw2['n_eff']:.0f} (2500)")
    huge = [("gaussian", 1.0, 1e6)]          # correlated far beyond the domain
    nc = number_effective_samples(e[sub], n[sub], huge, seed=3)
    print(f"      fully correlated: n_eff {nc['n_eff']:.2f}")
    check("fully correlated: n_eff ~ 1", nc["n_eff"] < 1.5, f"{nc['n_eff']:.2f}")

    print("G3  robust vs classical estimator under blunders")
    zb = z.copy()
    bad = rng.choice(zb.size, size=int(0.02 * zb.size), replace=False)
    zb[bad] += rng.normal(0, 30.0, bad.size)          # 2 % blunders, 20x the signal
    ma = empirical_variogram(e, n, zb, max_lag=10000.0, n_bins=18, estimator="matheron", seed=4)
    do = empirical_variogram(e, n, zb, max_lag=10000.0, n_bins=18, estimator="dowd", seed=4)
    clean = empirical_variogram(e, n, z, max_lag=10000.0, n_bins=18, estimator="dowd", seed=4)
    infl_m = float(np.median(ma["gamma"] / np.interp(ma["lags"], clean["lags"], clean["gamma"])))
    infl_d = float(np.median(do["gamma"] / np.interp(do["lags"], clean["lags"], clean["gamma"])))
    print(f"      sill inflation with 2 % blunders: matheron x{infl_m:.1f}   dowd x{infl_d:.2f}")
    check("Dowd inflated < 1.5x", infl_d < 1.5, f"x{infl_d:.2f}")
    check("Matheron inflated much more than Dowd", infl_m > 3 * infl_d, f"x{infl_m:.1f} vs x{infl_d:.2f}")

    print("G4  the propagation identity, against a field drawn from a KNOWN covariance")
    # Draw z ~ N(0, C) with C built from the model at the actual point
    # locations (Cholesky). This tests the double-sum identity itself, with no
    # dependence on how a synthetic field was synthesised -- the spectral
    # construction above is only approximately the covariance it targets, and
    # that approximation, not the propagation, is what an end-to-end check
    # would be measuring.
    TRUE_P = [("nugget", 0.30, 0.0), ("spherical", 1.70, 3000.0)]
    m2 = 400
    sub2 = rng.choice(e.size, m2, replace=False)
    ee2, nn2 = e[sub2], n[sub2]
    d2 = np.hypot(ee2[:, None] - ee2[None, :], nn2[:, None] - nn2[None, :])
    C = covariance_from_variogram(d2, TRUE_P)
    C[np.diag_indices(m2)] = sum(p[1] for p in TRUE_P)      # C(0) = total sill
    L = np.linalg.cholesky(C + 1e-10 * np.eye(m2))
    r4 = np.random.default_rng(4242)
    means = [float((L @ r4.normal(size=m2)).mean()) for _ in range(4000)]
    emp_var = float(np.var(means))
    pred = number_effective_samples(ee2, nn2, TRUE_P, n_subsample=m2, n_draws=1, seed=5)
    print(f"      Var(mean) empirical {emp_var:.5f}   predicted {pred['var_mean']:.5f}   "
          f"ratio {pred['var_mean']/emp_var:.3f}   n_eff {pred['n_eff']:.1f} of {m2}")
    check("predicted Var(mean) within 10 % of 4000 realisations",
          abs(pred["var_mean"] / emp_var - 1) < 0.10, f"ratio {pred['var_mean']/emp_var:.3f}")
    check("n_eff far below the point count (correlation is doing work)",
          pred["n_eff"] < 0.5 * m2, f"{pred['n_eff']:.1f} vs {m2}")
    # What the white-noise assumption costs, and the invariant that makes it a
    # test of the method rather than of this fixture: assuming independence
    # divides the variance by N instead of n_eff, so the error bar is
    # understated by exactly sqrt(N / n_eff).
    naive = sum(p[1] for p in TRUE_P) / m2
    understated = float(np.sqrt(emp_var / naive))
    expected = float(np.sqrt(m2 / pred["n_eff"]))
    print(f"      (a white-noise error bar would be {np.sqrt(naive):.4f}; correlated "
          f"truth {np.sqrt(emp_var):.4f} -- understated {understated:.2f}x; "
          f"sqrt(N/n_eff) = {expected:.2f}x)")
    check("understatement factor == sqrt(N / n_eff) within 5 %",
          abs(understated / expected - 1) < 0.05, f"{understated:.2f}x vs {expected:.2f}x")
    check("the white-noise assumption does understate the error", understated > 1.5,
          f"{understated:.2f}x")

    print("G5  the per-strip Sigma model implies the variogram we measure")
    # One strip's residual = offset + plane, the model behind
    # strip_prior_from_residual_planes. Its variogram is analytic: an offset is
    # pure covariance (no variogram), a plane with slope variance tau^2 in each
    # axis gives gamma(h) = tau^2 h^2 / 2 (isotropic, averaged over direction).
    TAU = 3.0e-6
    sub3 = rng.choice(e.size, 1200, replace=False)
    ee, nn = e[sub3], n[sub3]
    edges = np.linspace(0.0, 8000.0, 15)
    gammas, lags_ref = [], None
    for i in range(200):
        r = np.random.default_rng(500 + i)
        ax, ay = r.normal(0, TAU), r.normal(0, TAU)
        off = r.normal(0, 0.5)          # cancels within a realisation
        zi = off + ax * (ee - ee.mean()) + ay * (nn - nn.mean())
        evi = empirical_variogram(ee, nn, zi, bin_edges=edges, estimator="matheron",
                                  n_subsample=1200, n_draws=1, seed=600 + i)
        if lags_ref is None:
            lags_ref = evi["lags"]
        gammas.append(np.interp(lags_ref, evi["lags"], evi["gamma"]))
    evp = dict(lags=lags_ref, gamma=np.mean(gammas, axis=0))
    pred_g = TAU ** 2 * evp["lags"] ** 2 / 2.0
    ratio = float(np.median(evp["gamma"] / pred_g))
    print(f"      measured/predicted gamma across lags: median {ratio:.2f}")
    check("plane-model variogram matches tau^2 h^2 / 2 within 25 %",
          abs(ratio - 1) < 0.25, f"{ratio:.2f}")
    q = variogram_model(np.array([0.0, 1e9]), "spherical", 2.0, 1000.0)
    check("spherical model: gamma(0)=0 and saturates at the sill",
          q[0] == 0.0 and abs(q[1] - 2.0) < 1e-12)
    c = covariance_from_variogram(np.array([0.0]), [("nugget", 0.5, 0.0), ("spherical", 1.5, 900.0)])
    check("C(0) == total sill", abs(float(c[0]) - 2.0) < 1e-12, f"{float(c[0]):.3f}")

    print("H1  heteroscedasticity: recover a known sigma(v1, v2)")
    # dh = sigma(count, rmse) * correlated_field. sigma falls with count and
    # rises with rmse -- the shape a DEM-stack error actually has.
    rh = np.random.default_rng(11)
    m6 = 20000
    sub6 = rh.choice(e.size, m6, replace=False)
    ee6, nn6 = e[sub6], n[sub6]
    cnt = rh.integers(4, 60, m6).astype(float)
    rms = rh.uniform(0.2, 3.0, m6)

    def true_sigma(c, r):
        return 2.0 * (c / 10.0) ** -0.6 * (1.0 + 0.8 * r)

    base = gaussian_field(N, RES, 1500.0, 1.0, rh).ravel()[sub6]
    base = base / nmad(base)                       # unit-NMAD correlated field
    dh = true_sigma(cnt, rms) * base
    z, sig_fun, df, scale = infer_heteroscedasticity_from_stable(
        dh, [cnt, rms], ["count", "rmse"], list_var_bins=[np.linspace(4, 60, 9),
                                                          np.linspace(0.2, 3.0, 9)])
    pred = sig_fun(cnt, rms)
    tru = true_sigma(cnt, rms)
    rat = float(np.median(pred / tru))
    sprd = float(nmad(pred / tru))
    print(f"      sigma recovered: median ratio {rat:.3f}, spread {sprd:.3f}; "
          f"scale {scale:.3f}; sigma spans {tru.min():.2f}-{tru.max():.2f}")
    check("fitted sigma tracks the truth (median ratio within 10 %)",
          abs(rat - 1) < 0.10, f"{rat:.3f}")
    check("and does so across the whole range (spread < 15 %)", sprd < 0.15, f"{sprd:.3f}")
    check("standardised values have unit NMAD", abs(nmad(z) - 1.0) < 0.05, f"{nmad(z):.3f}")

    print("H2  standardisation makes the field homoscedastic")
    lo_c, hi_c = cnt < 15, cnt > 45
    raw_ratio = nmad(dh[lo_c]) / nmad(dh[hi_c])
    std_ratio = nmad(z[lo_c]) / nmad(z[hi_c])
    print(f"      NMAD(low count)/NMAD(high count): raw {raw_ratio:.2f} -> standardised {std_ratio:.2f}")
    check("raw data are strongly heteroscedastic (>2x across count)", raw_ratio > 2.0,
          f"{raw_ratio:.2f}")
    check("standardised data are homoscedastic (within 15 %)", abs(std_ratio - 1) < 0.15,
          f"{std_ratio:.2f}")

    print("H3  the payoff: only the standardised variogram has a meaningful sill")
    ev_raw = empirical_variogram(ee6, nn6, dh, max_lag=8000.0, n_bins=14, seed=12)
    ev_std = empirical_variogram(ee6, nn6, z, max_lag=8000.0, n_bins=14, seed=12)
    f_std = fit_variogram(ev_std["lags"], ev_std["gamma"], counts=ev_std["counts"],
                          models=("nugget", "gaussian"))
    print(f"      standardised: sill {f_std['total_sill']:.2f} (expect ~1), "
          f"range {f_std['ranges'][0]:.0f} m (true 1500), r2 {f_std['r2']:.3f}")
    check("standardised sill ~ 1 (dimensionless correlation)",
          abs(f_std["total_sill"] - 1.0) < 0.35, f"{f_std['total_sill']:.2f}")
    check("standardised variogram recovers the true range within 35 %",
          abs(f_std["ranges"][0] / 1500.0 - 1) < 0.35, f"{f_std['ranges'][0]:.0f} m")
    check("the raw variogram sill is inflated by the sigma variation",
          ev_raw["gamma"][-1] > 3 * ev_std["gamma"][-1] * np.median(tru) ** 2 / 3,
          f"raw {ev_raw['gamma'][-1]:.2f} vs standardised {ev_std['gamma'][-1]:.2f}")

    print("H4  binning table and interpolator behaviour")
    marg = df[(df["nd"] == 1) & df["count_mid"].notna()].sort_values("count_mid")
    trend = float(np.corrcoef(marg["count_mid"], marg["nmad"])[0, 1])
    print(f"      1-D marginal in count: nmad {marg['nmad'].iloc[0]:.2f} -> "
          f"{marg['nmad'].iloc[-1]:.2f}, corr with count {trend:+.2f}")
    check("marginal recovers the decreasing sigma(count)", trend < -0.8, f"{trend:+.2f}")
    holed = df.copy()
    holed.loc[(holed["nd"] == 2) & (holed["count_mid"] < 12), "nmad"] = np.nan
    f2 = interp_nd_binning(holed, ["count", "rmse"], statistic="nmad")
    q = f2(np.array([5.0, 30.0]), np.array([1.0, 1.0]))
    check("empty cells filled, no NaN leaks", np.all(np.isfinite(q)), f"{q}")
    far = sig_fun(np.array([1e6]), np.array([1e6]))
    check("queries outside the binned range are clamped, not extrapolated",
          np.isfinite(far[0]) and far[0] <= 1.2 * float(np.nanmax(sig_fun.grid)) * scale,
          f"{far[0]:.2f}")

    print("\nGATE " + ("PASSED" if not FAILS else f"FAILED: {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
