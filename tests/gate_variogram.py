"""Gate: empirical variogram + correlated-error propagation (Hugonnet 2022).

Why this exists. Every regularisation selector we have tried assumes white
observation noise and fails on real stacks, because DEM strip error is
correlated at kilometre scales. This module measures that correlation without
a model and turns it into the error of a SPATIAL AVERAGE -- which is what a
melt flux is. Using the pixel count as the sample count when the noise is
correlated understates a flux uncertainty by an order of magnitude, so the
n_eff path is the one that has to be right.

G1  recovery: simulate a Gaussian field with a KNOWN correlation range and
    recover range and sill from the empirical variogram. Ranges are the
    standard geostatistical EFFECTIVE range, so a Gaussian covariance of e-folding
    scale L is the gaussian model with range 2L -- the expectations below are
    stated that way, not re-tuned to whatever the fit prints.
G2  limits: white noise -> pure nugget, n_eff ~ N; a field correlated across
    the whole domain -> n_eff ~ 1. These bracket every real case. Also the
    SMALL-N limits, which a block estimator hits whenever the shelf mask clips
    a block: N = 1 must give n_eff = 1 and Var = the sill (not inf, which reads
    as zero error for the least informative domain there is), N = 2 must give
    the ordinary one-pair formula, and N = 0 must raise rather than divide.
    A region with no pair (N < 2) and a subsample too small to sample one
    (n_subsample < 2) are different situations: the first is exact, the second
    is a usage error and must raise rather than report the white-noise answer
    for a correlated field.
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
H5  binning keeps samples sitting exactly at a covariate's CEILING. Quantile
    edges put the top edge at the data maximum, and a per-pixel epoch count
    piles up there, so dropping them would estimate the top sigma bin without
    the very pixels it describes.
G6  `counts` is a DISTINCT-pair count at every n_subsample, not a replication
    count. Overlapping draws (total overlap once n_subsample reaches the cloud
    size, ~n_draws-fold just below it) must not inflate `counts` nor let a
    pair-starved bin clear `min_pairs`.
H6  a covariate may share a name with a statistic -- `count` is this module's
    headline covariate -- so the bin interval and the sample count must both
    survive in the binning table.

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
    standard_error_of_mean,
    empirical_variogram,
    fit_variogram,
    infer_heteroscedasticity_from_stable,
    interp_nd_binning,
    nd_binning,
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
    # gaussian_field synthesises C(h) = exp(-(h/CORR)^2), i.e. the gaussian
    # model with e-folding scale a = CORR. The effective range reported by
    # fit_variogram is r = 2a (gamma(r) = 98 % of the sill), which is the
    # number a published fit of the same field would report.
    EFF_RANGE = 2.0 * CORR
    z = gaussian_field(N, RES, CORR, SIG, rng).ravel()
    ev = empirical_variogram(e, n, z, max_lag=10000.0, n_bins=18, seed=1)
    fit = fit_variogram(ev["lags"], ev["gamma"], counts=ev["counts"],
                        models=("nugget", "gaussian"))
    rec_range = fit["ranges"][0]
    print(f"      sill {fit['total_sill']:.3f} (sample var {ev['variance']:.3f})   range {rec_range:.0f} m "
          f"(true effective {EFF_RANGE:.0f}, e-folding {CORR:.0f})   r2 {fit['r2']:.4f}   bins {fit['n_bins']}")
    check("effective range recovered within 30 %", abs(rec_range / EFF_RANGE - 1) < 0.30,
          f"{rec_range:.0f} vs {EFF_RANGE:.0f} m")
    check("gaussian model saturates at its effective range (98 % of sill)",
          abs(variogram_model(np.array([EFF_RANGE]), "gaussian", 1.0, EFF_RANGE)[0]
              - 0.9817) < 1e-3)
    check("exponential model reaches 95 % of sill at its effective range",
          abs(variogram_model(np.array([1000.0]), "exponential", 1.0, 1000.0)[0]
              - 0.9502) < 1e-3)
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

    # Small-N limits. A block clipped by the shelf mask can hold one pixel, and
    # the double sum over a single point is exactly C(0) -- so the answer is
    # n_eff = 1 and Var(z_bar) = the sill, never inf.
    P_SMALL = [("nugget", 1.0, 0.0), ("spherical", 2.0, 4000.0)]
    sill = sum(p[1] for p in P_SMALL)
    one = number_effective_samples(np.array([0.0]), np.array([0.0]), P_SMALL)
    sem_one = standard_error_of_mean(np.array([0.0]), np.array([0.0]), P_SMALL)
    print(f"      N=1: n_eff {one['n_eff']:.3f}  var_mean {one['var_mean']:.4f} "
          f"(sill {sill:.4f})  mean_offdiag {one['mean_offdiag_cov']}  "
          f"sem {sem_one:.4f}")
    check("N=1: n_eff is 1, not inf", one["n_eff"] == 1.0, f"{one['n_eff']}")
    check("N=1: Var(z_bar) is the sill", abs(one["var_mean"] - sill) < 1e-12,
          f"{one['var_mean']:.6f} vs {sill:.6f}")
    check("N=1: the standard error is finite and sqrt(sill)",
          np.isfinite(sem_one) and abs(sem_one - np.sqrt(sill)) < 1e-12,
          f"{sem_one:.6f} vs {np.sqrt(sill):.6f}")
    check("N=1: no NaN leaks into the reported diagnostics",
          np.isfinite(one["mean_offdiag_cov"]) and one["offdiag_draws"] == [],
          f"mean_offdiag {one['mean_offdiag_cov']}  draws {one['offdiag_draws']}")
    # N=2 must be the ordinary formula over its single pair, checked against the
    # covariance evaluated independently at that separation.
    h12 = 3000.0
    two = number_effective_samples(np.array([0.0, h12]), np.array([0.0, 0.0]), P_SMALL)
    c12 = float(covariance_from_variogram(np.array([h12]), P_SMALL)[0])
    var2 = sill / 2.0 + 0.5 * c12
    print(f"      N=2: n_eff {two['n_eff']:.4f}  var_mean {two['var_mean']:.4f} "
          f"(closed form {var2:.4f}, C({h12:.0f} m) = {c12:.4f})")
    check("N=2: Var(z_bar) matches the one-pair double sum",
          abs(two["var_mean"] - var2) < 1e-12, f"{two['var_mean']:.6f} vs {var2:.6f}")
    check("N=2: mean_offdiag_cov is that single pair's covariance",
          abs(two["mean_offdiag_cov"] - c12) < 1e-12, f"{two['mean_offdiag_cov']:.6f}")
    check("N=2: n_eff lies between 1 and 2", 1.0 <= two["n_eff"] <= 2.0,
          f"{two['n_eff']:.4f}")
    # No finite point at all is an error, not a division by zero.
    for label, (ee_, nn_) in {"empty": (np.array([]), np.array([])),
                              "all-NaN": (np.full(3, np.nan), np.full(3, np.nan))}.items():
        try:
            number_effective_samples(ee_, nn_, P_SMALL)
        except ValueError as exc:
            check(f"{label} input raises ValueError, not ZeroDivisionError",
                  "finite" in str(exc), str(exc)[:70])
        except Exception as exc:            # noqa: BLE001
            check(f"{label} input raises ValueError, not ZeroDivisionError",
                  False, f"{type(exc).__name__}: {exc}")
        else:
            check(f"{label} input raises ValueError, not ZeroDivisionError",
                  False, "no error raised")
    # A subsample too small to form a pair is a USAGE error, not a pairless
    # region: 100 points correlated over 1000 km across a 1 km domain have
    # n_eff ~ 1, and silently averaging zero pairs would report n_eff = 100 --
    # the white-noise answer, understating the error 10x.
    e_corr, n_corr = np.arange(100) * 10.0, np.zeros(100)
    P_CORR = [("spherical", 1.0, 1e6)]
    truth = number_effective_samples(e_corr, n_corr, P_CORR)
    print(f"      correlated fixture: n_eff {truth['n_eff']:.4f} of 100 points "
          f"(var_mean {truth['var_mean']:.4f})")
    check("the fixture really is domain-wide correlated (n_eff ~ 1)",
          truth["n_eff"] < 1.5, f"{truth['n_eff']:.4f}")
    for bad_ns in (0, 1):
        try:
            got = number_effective_samples(e_corr, n_corr, P_CORR, n_subsample=bad_ns)
        except ValueError as exc:
            check(f"n_subsample={bad_ns} raises and names the parameter",
                  "n_subsample" in str(exc), str(exc)[:70])
        except Exception as exc:            # noqa: BLE001
            check(f"n_subsample={bad_ns} raises and names the parameter",
                  False, f"{type(exc).__name__}: {exc}")
        else:
            check(f"n_subsample={bad_ns} raises and names the parameter",
                  False, f"returned n_eff={got['n_eff']:.1f} instead of raising")
    # ...while a subsample AT the pair threshold is legitimate and must work.
    two_s = number_effective_samples(e_corr, n_corr, P_CORR, n_subsample=2, n_draws=64)
    check("n_subsample=2 is accepted and still reports a correlated field",
          np.isfinite(two_s["n_eff"]) and two_s["n_eff"] < 1.5,
          f"n_eff {two_s['n_eff']:.4f}")

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
    H3_EFF_RANGE = 2.0 * 1500.0            # e-folding 1500 m -> effective range
    print(f"      standardised: sill {f_std['total_sill']:.2f} (expect ~1), "
          f"range {f_std['ranges'][0]:.0f} m (true effective {H3_EFF_RANGE:.0f}), "
          f"r2 {f_std['r2']:.3f}")
    check("standardised sill ~ 1 (dimensionless correlation)",
          abs(f_std["total_sill"] - 1.0) < 0.35, f"{f_std['total_sill']:.2f}")
    check("standardised variogram recovers the true effective range within 35 %",
          abs(f_std["ranges"][0] / H3_EFF_RANGE - 1) < 0.35, f"{f_std['ranges'][0]:.0f} m")
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
          np.isfinite(far[0]) and far[0] <= 1.2 * float(np.nanmax(sig_fun.grid)),
          f"{far[0]:.2f}")
    # .grid must be in the units the function itself returns, or plotting it as
    # "the error model" is wrong by exactly `scale` with nothing to signal it.
    on_grid = sig_fun(np.array([sig_fun.centres[0][2]]), np.array([sig_fun.centres[1][3]]))
    check("sig_fun.grid is the CALIBRATED model (same units as sig_fun(...))",
          abs(float(on_grid[0]) / float(sig_fun.grid[2, 3]) - 1) < 1e-9,
          f"{float(on_grid[0]):.3f} vs grid {float(sig_fun.grid[2, 3]):.3f}")
    check("the raw binned dispersion is still reachable as .unscaled.grid",
          abs(float(sig_fun.grid[2, 3]) / (scale * float(sig_fun.unscaled.grid[2, 3])) - 1) < 1e-9,
          f"scale {scale:.3f}")

    print("H5  samples at a covariate ceiling are binned, not dropped")
    # A per-pixel epoch COUNT field saturates: a large mass sits exactly at the
    # maximum. With quantile edges the top edge IS that maximum.
    rh5 = np.random.default_rng(31)
    n_ceil, n_rest = 400, 600
    cnt5 = np.concatenate([rh5.integers(4, 60, n_rest).astype(float),
                           np.full(n_ceil, 60.0)])
    val5 = np.concatenate([rh5.normal(0, 3.0, n_rest), rh5.normal(0, 0.5, n_ceil)])
    df5 = nd_binning(val5, [cnt5], ["count"])
    marg5 = df5[df5["nd"] == 1]
    binned = int(marg5["count"].sum())
    print(f"      {n_ceil} of {cnt5.size} samples sit at the ceiling "
          f"{cnt5.max():.0f}; binned total {binned}")
    check("no sample is dropped at the covariate maximum",
          binned == cnt5.size, f"{binned} of {cnt5.size}")
    top = marg5.sort_values("count_mid").iloc[-1]
    check("the ceiling mass lands in the TOP bin",
          int(top["count"]) >= n_ceil, f"top bin holds {int(top['count'])}")
    # ...and the error model that bin feeds is the quiet one those pixels have,
    # not the loud one the rest of the population has.
    check("the top bin's nmad describes the pixels it is supposed to",
          abs(float(top["nmad"]) / 0.5 - 1.0) < 0.25,
          f"nmad {float(top['nmad']):.3f} vs true 0.5")

    print("G6  counts are pairs, not repeated draws")
    rg6 = np.random.default_rng(17)
    m6b = 300
    e6 = rg6.uniform(0, 25000, m6b)
    n6 = rg6.uniform(0, 25000, m6b)
    v6 = rg6.normal(0, 1.0, m6b)
    edges6 = np.concatenate([[0.0], np.geomspace(20.0, 10000.0, 18)])
    one = empirical_variogram(e6, n6, v6, bin_edges=edges6, n_draws=1, seed=1)
    ten = empirical_variogram(e6, n6, v6, bin_edges=edges6, n_draws=10, seed=1)
    print(f"      n_subsample(2000) >= N({m6b}): n_draws 1 -> {one['n_draws_used']} pass, "
          f"n_draws 10 -> {ten['n_draws_used']} pass; bins {len(one['lags'])} vs {len(ten['lags'])}")
    check("extra draws over an exhaustive subsample change nothing",
          np.array_equal(one["counts"], ten["counts"])
          and np.allclose(one["gamma"], ten["gamma"], rtol=0, atol=0)
          and len(one["lags"]) == len(ten["lags"]),
          f"counts equal {np.array_equal(one['counts'], ten['counts'])}")
    # An exhaustive subsample re-forms the identical pair set every pass, so
    # ONE pass is run and the pooled total is scaled by n_draws arithmetically:
    # the repeats provably add no distinct pair, so `counts` is unmoved while
    # `n_pairs_pooled` still reports what n_draws passes would have pooled.
    check("repeat draws over an exhaustive subsample add no distinct pair",
          ten["n_draws_used"] == 10
          and np.array_equal(ten["n_pairs_pooled"], 10 * one["n_pairs_pooled"])
          and np.array_equal(ten["counts"], one["counts"]),
          f"pooled {ten['n_pairs_pooled'][:3]} vs counts {ten['counts'][:3]}")
    # every admitted bin must really hold min_pairs DISTINCT pairs
    iu6, ju6 = np.triu_indices(m6b, k=1)
    d6 = np.hypot(e6[iu6] - e6[ju6], n6[iu6] - n6[ju6])
    true_counts = np.array([int(((d6 >= edges6[b]) & (d6 < edges6[b + 1])).sum())
                            for b in range(edges6.size - 1)])
    kept = np.searchsorted(0.5 * (edges6[:-1] + edges6[1:]), ten["lags"])
    print(f"      admitted bins hold {true_counts[kept].min()}-{true_counts[kept].max()} "
          f"distinct pairs (min_pairs 30)")
    check("counts match the true distinct-pair count",
          np.array_equal(ten["counts"], true_counts[kept]),
          f"{ten['counts'][:4]} vs {true_counts[kept][:4]}")
    check("no bin is admitted on fewer than min_pairs distinct pairs",
          int(true_counts[kept].min()) >= 30, f"{int(true_counts[kept].min())}")
    # and the subsampled path still pools its independent draws
    sub = empirical_variogram(e6, n6, v6, bin_edges=edges6, n_subsample=120,
                              n_draws=4, seed=1)
    check("a genuine subsample still pools every draw", sub["n_draws_used"] == 4,
          f"{sub['n_draws_used']}")

    # Just BELOW the subsample size is the regime the exhaustive special case
    # missed: 301 points at n_subsample=300 re-forms a given pair in ~10 of 10
    # draws, so counts must still be distinct pairs, not ~10x them.
    e7 = np.append(e6, 12500.0)
    n7 = np.append(n6, 12500.0)
    v7 = np.append(v6, 0.25)
    near = empirical_variogram(e7, n7, v7, bin_edges=edges6, n_subsample=m6b,
                               n_draws=10, seed=1)
    iu7, ju7 = np.triu_indices(e7.size, k=1)
    d7 = np.hypot(e7[iu7] - e7[ju7], n7[iu7] - n7[ju7])
    all_counts7 = np.array([int(((d7 >= edges6[b]) & (d7 < edges6[b + 1])).sum())
                            for b in range(edges6.size - 1)])
    kept7 = np.searchsorted(0.5 * (edges6[:-1] + edges6[1:]), near["lags"])
    infl = near["n_pairs_pooled"] / np.maximum(near["counts"], 1)
    print(f"      N={e7.size}, n_subsample={m6b}, n_draws=10: pooled/distinct "
          f"{infl.min():.1f}-{infl.max():.1f}x, counts <= all-pairs: "
          f"{bool(np.all(near['counts'] <= all_counts7[kept7]))}")
    check("draws really do overlap in this regime (pooled >> distinct)",
          float(infl.max()) > 5.0, f"{float(infl.max()):.1f}x")
    check("counts stay a DISTINCT-pair count just below n_subsample",
          bool(np.all(near["counts"] <= all_counts7[kept7])),
          f"{near['counts'][:4]} vs all-pairs {all_counts7[kept7][:4]}")
    check("no bin admitted on fewer than min_pairs distinct pairs (subsampled)",
          int(near["counts"].min()) >= 30, f"{int(near['counts'].min())}")

    print("H6  a covariate may be named after a statistic")
    rh6 = np.random.default_rng(41)
    cnt6 = rh6.integers(4, 60, 800).astype(float)
    df6 = nd_binning(rh6.normal(0, 1.0, 800), [cnt6], ["count"])
    m6 = df6[df6["nd"] == 1]
    import pandas as _pd
    check("the bin interval survives under <name>_bin",
          "count_bin" in df6.columns
          and all(isinstance(x, _pd.Interval) for x in m6["count_bin"]),
          f"{list(df6.columns)}")
    check("the 'count' statistic is still the sample count",
          int(m6["count"].sum()) == cnt6.size, f"{int(m6['count'].sum())} of {cnt6.size}")
    check("each interval brackets its own midpoint",
          all(iv.left <= mid <= iv.right for iv, mid in zip(m6["count_bin"], m6["count_mid"])))
    # dropping 'count' from `statistics` must not leave an Interval where
    # interp_nd_binning expects a number
    df6b = nd_binning(rh6.normal(0, 1.0, 800), [cnt6], ["count"],
                      statistics=("nmad",))
    f6 = interp_nd_binning(df6b, ["count"], statistic="nmad", min_count=0)
    check("binning without the 'count' statistic still interpolates",
          np.isfinite(f6(np.array([30.0]))[0]), f"{f6(np.array([30.0]))}")

    print("\nGATE " + ("PASSED" if not FAILS else f"FAILED: {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
